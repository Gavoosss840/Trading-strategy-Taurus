"""
Taurus – Main strategy orchestrator.

Execution pipeline (mirrors the diagram exactly)
──────────────────────────────────────────────────
S&P 500 universe
    │
    ├── CAPM + FF5 per stock ──► SML alpha  (|t| > 2 filter)
    │
    └── MM screen ─────────────► Capital structure gap (under/overleveraged)
                │
                ▼
        Combined signal (dual confirmation: α + leverage)
                │
                ▼
        Momentum filter (12M, skip last month)
                │
                ▼
        CML position sizing (max-Sharpe on efficient frontier)
                │
                ▼
        Beta neutralisation (net β ≈ 0)
               / \\
          LONG    SHORT
    (α>0, under, mom+) (α<0, over, mom-)

Public API
──────────
    strategy = TaurusStrategy(cfg)
    portfolio = strategy.run(as_of="2024-01-31")
    # or full back-test:
    book = strategy.backtest(start="2015-01-01", end="2024-12-31")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .capital_structure import mm_capital_structure_screen
from .config import TaurusConfig, DEFAULT_CONFIG
from .data import (
    compute_monthly_returns,
    get_ff5_factors,
    get_fundamentals,
    get_monthly_prices,
    get_sp500_tickers,
)
from .factors import compute_ff5_alpha
from .momentum import momentum_signal
from .portfolio import (
    beta_neutralise,
    build_leg,
    estimate_betas,
    futures_beta_hedge,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data-classes for outputs                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class PortfolioSnapshot:
    """All state for one rebalance date."""
    date:             pd.Timestamp
    long_weights:     pd.Series          # sums to 1
    short_weights:    pd.Series          # sums to 1 (positive values)
    alpha_df:         pd.DataFrame       # FF5 results
    mm_df:            pd.DataFrame       # MM screen results
    momentum_df:      pd.DataFrame       # Momentum signal
    combined_longs:   pd.Index           # tickers that passed all 3 filters
    combined_shorts:  pd.Index
    futures_weight:   float = 0.0        # ES/SPY overlay (-ve = short futures)

    @property
    def net_weights(self) -> pd.Series:
        """Long minus Short (short positions are negative)."""
        net = self.long_weights.copy()
        for t, w in self.short_weights.items():
            net[t] = net.get(t, 0.0) - w
        return net

    @property
    def n_longs(self) -> int:
        return len(self.long_weights)

    @property
    def n_shorts(self) -> int:
        return len(self.short_weights)


# --------------------------------------------------------------------------- #
#  Strategy class                                                              #
# --------------------------------------------------------------------------- #

class TaurusStrategy:
    """
    Long/short equity strategy implementing the Taurus alpha-generation diagram.

    Parameters
    ----------
    cfg : TaurusConfig – strategy parameters (uses DEFAULT_CONFIG if omitted)
    """

    def __init__(self, cfg: TaurusConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self._prices:       Optional[pd.DataFrame] = None
        self._returns:      Optional[pd.DataFrame] = None
        self._factors:      Optional[pd.DataFrame] = None
        self._fundamentals: Optional[pd.DataFrame] = None
        self._tickers:      Optional[List[str]]    = None

    # ── Data loading ──────────────────────────────────────────────────────── #

    def load_data(
        self,
        start: str,
        end: str,
        tickers: Optional[List[str]] = None,
        factors_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Pre-fetch and cache all data required for the strategy.

        Parameters
        ----------
        start, end  : ISO date strings  (e.g. "2015-01-01")
        tickers     : optional list; defaults to full S&P 500
        factors_df  : optional pre-loaded FF5 factors (for multi-universe use);
                      if None, falls back to get_ff5_factors() for US data
        """
        cfg = self.cfg

        if tickers is None:
            tickers = get_sp500_tickers(cfg)
        self._tickers = tickers

        logger.info("Loading price data (%d tickers)...", len(tickers))
        self._prices  = get_monthly_prices(tickers, start, end, cfg)
        self._returns = compute_monthly_returns(self._prices)

        logger.info("Loading Fama-French 5-factor data...")
        self._factors = factors_df if factors_df is not None else get_ff5_factors(start, end, cfg)

        logger.info("Loading fundamental data...")
        # Only load fundamentals for tickers that survived the price filter
        live_tickers = self._prices.columns.tolist()
        self._fundamentals = get_fundamentals(live_tickers, cfg)

        logger.info(
            "Data loaded: %d tickers, %d months, FF5 %d months.",
            self._prices.shape[1],
            self._prices.shape[0],
            len(self._factors),
        )

    # ── Single rebalance ─────────────────────────────────────────────────── #

    def run(
        self,
        as_of: Optional[str | pd.Timestamp] = None,
    ) -> PortfolioSnapshot:
        """
        Run one rebalance as of `as_of` (defaults to last available date).

        load_data() must have been called first.
        """
        if self._prices is None:
            raise RuntimeError("Call load_data() before run().")

        as_of = (
            pd.Timestamp(as_of) if as_of is not None
            else self._prices.index[-1]
        )

        cfg     = self.cfg
        prices  = self._prices
        returns = self._returns
        factors = self._factors
        fund    = self._fundamentals

        # ── 1. Rolling window of returns for FF5 regression ───────────── #
        idx       = prices.index
        end_loc   = idx.get_loc(as_of) if as_of in idx else len(idx) - 1
        start_loc = max(0, end_loc - cfg.lookback_months + 1)
        window_returns = returns.iloc[start_loc : end_loc + 1]

        valid_cols = window_returns.columns[
            window_returns.notna().sum() >= cfg.min_obs
        ]
        window_returns = window_returns[valid_cols].dropna()

        # ── 2. FF5 regression → SML alpha ─────────────────────────────── #
        logger.info("[%s] Running FF5 regression...", as_of.date())
        alpha_df = compute_ff5_alpha(window_returns, factors, cfg)
        if alpha_df.empty:
            logger.warning("FF5 regression returned no results.")
            return self._empty_snapshot(as_of)

        # ── 3. MM valuation screen (full universe) ────────────────────────── #
        logger.info("[%s] Running MM valuation screen...", as_of.date())
        live_fund = fund.loc[fund.index.intersection(alpha_df.index)]
        if live_fund.empty:
            logger.warning("No fundamental data for alpha candidates.")
            return self._empty_snapshot(as_of)

        # Pass returns for Merton volatility estimation
        mm_df = mm_capital_structure_screen(live_fund, cfg, returns=window_returns)

        # ── 4. Combined signal — dual confirmation ─────────────────────── #
        #
        # LONG  : VL > market_cap (MM undervalued)  AND  top-half alpha  AND  mom+
        # SHORT : VL < market_cap (MM overvalued)   AND  bottom-quintile alpha  AND  mom-
        #
        # Using cross-sectional alpha ranking for the short leg avoids the
        # "no negative alpha in bull markets" problem: we always short the
        # *relative* underperformers confirmed by MM overvaluation.
        #
        under_tickers = mm_df[mm_df["underleveraged"]].index   # MM undervalued → LONG
        over_tickers  = mm_df[mm_df["overleveraged"]].index    # MM overvalued  → SHORT

        # LONG alpha — 3-stage fallback to handle tech-heavy universes (e.g. NASDAQ 100)
        # Stage 1: absolute filter — positive alpha AND |t| > threshold ∩ MM-undervalued
        alpha_long_abs      = alpha_df[alpha_df["signal"] & (alpha_df["alpha_sign"] > 0)]
        long_candidates_abs = alpha_long_abs.index.intersection(under_tickers)

        # Stage 2: if too few from Stage 1, use cross-sectional top-50% alpha t-stat
        #           within the MM-undervalued pool (relative ranking, like the short leg)
        min_long_candidates = max(3, cfg.n_longs // 4)
        if len(long_candidates_abs) >= min_long_candidates:
            long_candidates = long_candidates_abs
        else:
            under_alpha = alpha_df.loc[alpha_df.index.intersection(under_tickers)]
            if len(under_alpha) >= 2:
                q50_under    = under_alpha["alpha_tstat"].quantile(0.50)
                alpha_long_cs = under_alpha[under_alpha["alpha_tstat"] >= q50_under]
                long_candidates = alpha_long_cs.index
                logger.info(
                    "[%s] LONG Stage-2 (cross-sect. top-50%% of %d undervalued) → %d candidates.",
                    as_of.date(), len(under_alpha), len(long_candidates),
                )
            else:
                # Stage 3: MM pool is empty/tiny — take top alpha stocks from full
                #          universe (alpha signal only, drop MM requirement)
                q75_all     = alpha_df["alpha_tstat"].quantile(0.75)
                long_candidates = alpha_df[alpha_df["alpha_tstat"] >= q75_all].index
                logger.info(
                    "[%s] LONG Stage-3 (MM pool empty, top-25%% alpha full universe) → %d candidates.",
                    as_of.date(), len(long_candidates),
                )

        # SHORT alpha: bottom 20% of t-stat cross-sectionally (worst relative alpha)
        q20 = alpha_df["alpha_tstat"].quantile(0.20)
        alpha_short = alpha_df[alpha_df["alpha_tstat"] <= q20]

        short_candidates = alpha_short.index.intersection(over_tickers)

        logger.info(
            "[%s] Dual confirmation: %d long (MM underval + α+), %d short (MM overval + α-).",
            as_of.date(), len(long_candidates), len(short_candidates),
        )

        # ── 5. Momentum filter ─────────────────────────────────────────── #
        logger.info("[%s] Applying momentum filter...", as_of.date())
        mom_df = momentum_signal(prices, as_of, cfg)
        if mom_df.empty:
            logger.warning("Momentum signal is empty.")
            return self._empty_snapshot(as_of)

        mom_long  = mom_df[mom_df["mom_pos"]].index   # above-median momentum → LONG
        mom_short = mom_df[mom_df["mom_neg"]].index   # below-median momentum → SHORT

        long_final  = long_candidates.intersection(mom_long)
        short_final = short_candidates.intersection(mom_short)

        # Momentum fallback: if momentum filter kills all candidates, relax to top/bottom
        # 60% so concentration from a small alpha pool doesn't wipe out all positions
        min_final = max(2, cfg.n_longs // 5)
        if len(long_final) < min_final and not mom_df.empty:
            q40_mom = mom_df["mom_raw"].quantile(0.40)
            mom_long_relaxed = mom_df[mom_df["mom_raw"] >= q40_mom].index
            long_final_relaxed = long_candidates.intersection(mom_long_relaxed)
            if len(long_final_relaxed) >= len(long_final):
                long_final = long_final_relaxed
                logger.info("[%s] LONG momentum relaxed (top-60%%) → %d candidates.", as_of.date(), len(long_final))

        if len(short_final) < min_final and not mom_df.empty:
            q60_mom = mom_df["mom_raw"].quantile(0.60)
            mom_short_relaxed = mom_df[mom_df["mom_raw"] <= q60_mom].index
            short_final_relaxed = short_candidates.intersection(mom_short_relaxed)
            if len(short_final_relaxed) >= len(short_final):
                short_final = short_final_relaxed
                logger.info("[%s] SHORT momentum relaxed (bottom-60%%) → %d candidates.", as_of.date(), len(short_final))

        logger.info(
            "[%s] Post-momentum: %d long, %d short.",
            as_of.date(), len(long_final), len(short_final),
        )

        if len(long_final) == 0 or len(short_final) == 0:
            logger.warning("Empty leg(s) after filtering – returning empty snapshot.")
            return self._empty_snapshot(as_of)

        # ── 6. CML position sizing ────────────────────────────────────── #
        logger.info("[%s] Building CML-optimal legs...", as_of.date())
        sector_map = fund["sector"].to_dict() if "sector" in fund.columns else {}
        alpha_scores = alpha_df["alpha_annual"]

        long_weights  = build_leg(
            long_final, alpha_scores, returns, sector_map, cfg, cfg.n_longs
        )
        short_weights = build_leg(
            short_final, -alpha_scores, returns, sector_map, cfg, cfg.n_shorts
        )

        # ── 7. Beta neutralisation ────────────────────────────────────── #
        logger.info("[%s] Beta neutralising...", as_of.date())
        mkt_returns = factors["Mkt-RF"] + factors["RF"]
        betas = estimate_betas(window_returns, mkt_returns)

        futures_weight = 0.0

        if cfg.use_futures_hedge:
            # Futures overlay: keep alpha weights intact, hedge β via ES/SPY
            futures_weight = futures_beta_hedge(long_weights, short_weights, betas, cfg)
        else:
            # Classic weight-rescaling beta neutralisation
            long_weights, short_weights = beta_neutralise(
                long_weights, short_weights, betas, cfg
            )

        logger.info(
            "[%s] Final portfolio: %d longs, %d shorts%s.",
            as_of.date(), len(long_weights), len(short_weights),
            f", futures={futures_weight:+.4f}×NAV" if cfg.use_futures_hedge else "",
        )

        return PortfolioSnapshot(
            date=as_of,
            long_weights=long_weights,
            short_weights=short_weights,
            alpha_df=alpha_df,
            mm_df=mm_df,
            momentum_df=mom_df,
            combined_longs=long_final,
            combined_shorts=short_final,
            futures_weight=futures_weight,
        )

    def _empty_snapshot(self, as_of: pd.Timestamp) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            date=as_of,
            long_weights=pd.Series(dtype=float),
            short_weights=pd.Series(dtype=float),
            alpha_df=pd.DataFrame(),
            mm_df=pd.DataFrame(),
            momentum_df=pd.DataFrame(),
            combined_longs=pd.Index([]),
            combined_shorts=pd.Index([]),
        )

    # ── Full back-test ────────────────────────────────────────────────────── #

    def backtest(
        self,
        start: str,
        end: str,
        tickers: Optional[List[str]] = None,
        factors_df: Optional[pd.DataFrame] = None,
    ) -> "BacktestResult":
        """
        Run a full vectorised back-test over [start, end].

        Returns a BacktestResult with positions, returns, and analytics.
        """
        # Add extra history for warm-up
        warm_up_months = cfg_warm_months(self.cfg)
        warm_start = (
            pd.Timestamp(start) - pd.DateOffset(months=warm_up_months)
        ).strftime("%Y-%m-%d")

        self.load_data(warm_start, end, tickers, factors_df=factors_df)

        prices  = self._prices
        returns = self._returns
        idx     = prices.index

        rebalance_dates = pd.date_range(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            freq=self.cfg.rebalance_freq,
        ).intersection(idx)

        snapshots: List[PortfolioSnapshot] = []
        for reb_date in rebalance_dates:
            snap = self.run(as_of=reb_date)
            snapshots.append(snap)

        return BacktestResult(
            snapshots=snapshots,
            prices=prices,
            returns=returns,
            cfg=self.cfg,
        )


def cfg_warm_months(cfg: TaurusConfig) -> int:
    return cfg.lookback_months + cfg.momentum_months + cfg.momentum_skip + 3


# --------------------------------------------------------------------------- #
#  BacktestResult                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class BacktestResult:
    snapshots: List[PortfolioSnapshot]
    prices:    pd.DataFrame
    returns:   pd.DataFrame
    cfg:       TaurusConfig

    def portfolio_returns(self) -> pd.Series:
        """
        Compute monthly strategy P&L from the position snapshots.

        Each snapshot's weights are held until the next rebalance.
        Transaction costs are deducted at each rebalance.
        """
        pnl_records = []
        cost_bps = self.cfg.total_cost_bps / 10_000  # to decimal

        prev_lw = pd.Series(dtype=float)
        prev_sw = pd.Series(dtype=float)

        for i, snap in enumerate(self.snapshots):
            hold_start = snap.date
            hold_end   = (
                self.snapshots[i + 1].date if i + 1 < len(self.snapshots)
                else self.returns.index[-1]
            )

            # Monthly returns during holding period
            month_range = self.returns.loc[
                (self.returns.index > hold_start) &
                (self.returns.index <= hold_end)
            ]

            # Leverage scaling: gross_leverage=1.5 → each leg runs at 75% of NAV
            lev  = self.cfg.gross_leverage
            half = lev / 2.0  # weight applied to each leg

            # Monthly financing cost (margin + borrow), deducted pro-rata
            margin_cost_m = (self.cfg.margin_cost_annual + self.cfg.borrow_cost_annual) / 12
            leverage_cost_m = (lev - 1.0) * margin_cost_m  # only on borrowed portion

            # Futures roll cost: deducted once per quarter (~3 months)
            # Approximated as monthly fraction when futures hedge is active
            futures_roll_m = (
                self.cfg.futures_roll_cost_quarterly / 3.0
                if self.cfg.use_futures_hedge and snap.futures_weight != 0.0
                else 0.0
            )

            for month_date, month_ret in month_range.iterrows():
                long_ret  = (snap.long_weights  * month_ret.reindex(snap.long_weights.index,  fill_value=0)).sum()
                short_ret = (snap.short_weights * month_ret.reindex(snap.short_weights.index, fill_value=0)).sum()
                gross_ret = half * long_ret - half * short_ret  # leveraged dollar-neutral
                gross_ret -= leverage_cost_m                     # deduct financing cost

                # Futures P&L: futures_weight × market return (β_futures = 1)
                if snap.futures_weight != 0.0:
                    mkt_ret = month_ret.mean()   # proxy for index return
                    gross_ret += snap.futures_weight * mkt_ret
                    gross_ret -= futures_roll_m  # roll cost

                pnl_records.append({"date": month_date, "return": gross_ret})

            # Turnover-based cost at rebalance
            turnover = _compute_turnover(snap.long_weights, prev_lw) + \
                       _compute_turnover(snap.short_weights, prev_sw)
            cost = turnover * cost_bps
            if pnl_records:
                pnl_records[-1]["return"] -= cost

            prev_lw = snap.long_weights
            prev_sw = snap.short_weights

        if not pnl_records:
            return pd.Series(dtype=float)

        pnl = pd.DataFrame(pnl_records).set_index("date")["return"]
        return pnl

    def analytics(self) -> Dict[str, float]:
        """Compute standard performance metrics."""
        ret = self.portfolio_returns()
        if ret.empty:
            return {}

        cum   = (1 + ret).cumprod()
        total = cum.iloc[-1] - 1
        ann   = (1 + total) ** (12 / len(ret)) - 1
        vol   = ret.std() * np.sqrt(12)
        sharpe = (ann - self.cfg.risk_free_rate_annual) / vol if vol > 0 else np.nan
        drawdown = (cum / cum.cummax() - 1).min()
        calmar   = ann / abs(drawdown) if drawdown < 0 else np.nan

        return {
            "total_return":         float(total),
            "total_return_pct":     float(total * 100),
            "annual_return":        float(ann),
            "annual_return_pct":    float(ann * 100),
            "annual_vol":           float(vol),
            "annual_vol_pct":       float(vol * 100),
            "sharpe_ratio":         float(sharpe),
            "max_drawdown":         float(drawdown),
            "max_drawdown_pct":     float(drawdown * 100),
            "calmar_ratio":         float(calmar),
            "n_months":             len(ret),
            "n_rebalances":         len(self.snapshots),
        }

    def positions_df(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame of all positions across all rebalance dates.

        Columns: date, ticker, leg (long/short), weight
        """
        rows = []
        for snap in self.snapshots:
            for t, w in snap.long_weights.items():
                rows.append({"date": snap.date, "ticker": t, "leg": "long",  "weight": w})
            for t, w in snap.short_weights.items():
                rows.append({"date": snap.date, "ticker": t, "leg": "short", "weight": w})
        return pd.DataFrame(rows)


def _compute_turnover(new: pd.Series, old: pd.Series) -> float:
    """One-way turnover between two weight vectors."""
    all_tickers = new.index.union(old.index)
    n = new.reindex(all_tickers, fill_value=0)
    o = old.reindex(all_tickers, fill_value=0)
    return (n - o).abs().sum() / 2.0
