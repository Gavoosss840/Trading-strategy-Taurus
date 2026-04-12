"""
Taurus – Main strategy orchestrator.

Execution pipeline
──────────────────
Universe (S&P 500 / NASDAQ / CAC / FTSE / Nikkei / …)
    │
    ├── FF5/FF6 per stock ──► SML alpha t-stat
    │
    ├── MM valuation screen ─► divergence_pct (under/overleveraged)
    │
    └── Momentum ───────────► vol-adjusted Sharpe momentum score
                │
                ▼
        Composite z-score  (default)  OR  binary AND-filter  (legacy)
        composite = w_α × z(alpha_tstat) + w_mm × z(divergence_pct) + w_mom × z(mom_score)
                │
                ▼
        CML / Min-variance position sizing  (max-Sharpe legacy available)
                │
                ▼
        Beta neutralisation  (Blume-adjusted betas)
               / \\
          LONG    SHORT

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
#  Robust z-score helper                                                        #
# --------------------------------------------------------------------------- #

def _zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """
    Robust cross-sectional z-score (median / MAD normalisation).
    Clipped at ±clip to prevent extreme values dominating the composite.
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad > 0:
        z = (s - med) / (mad * 1.4826)   # MAD → σ scaling
    else:
        z = s - med
    return z.clip(-clip, clip)


# --------------------------------------------------------------------------- #
#  Data-classes for outputs                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class PortfolioSnapshot:
    """All state for one rebalance date."""
    date:             pd.Timestamp
    long_weights:     pd.Series          # sums to 1
    short_weights:    pd.Series          # sums to 1 (positive values)
    alpha_df:         pd.DataFrame       # FF5/FF6 results
    mm_df:            pd.DataFrame       # MM screen results
    momentum_df:      pd.DataFrame       # Momentum signal
    combined_longs:   pd.Index           # tickers that passed all filters
    combined_shorts:  pd.Index
    composite_scores: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    futures_weight:   float = 0.0        # ES/SPY overlay (-ve = short futures)

    @property
    def net_weights(self) -> pd.Series:
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
#  Strategy class                                                               #
# --------------------------------------------------------------------------- #

class TaurusStrategy:
    """
    Long/short equity strategy implementing the Taurus alpha-generation pipeline.

    Parameters
    ----------
    cfg : TaurusConfig – strategy parameters (uses DEFAULT_CONFIG if omitted)
    """

    def __init__(self, cfg: TaurusConfig = DEFAULT_CONFIG):
        self.cfg              = cfg
        self._prices:         Optional[pd.DataFrame] = None
        self._returns:        Optional[pd.DataFrame] = None
        self._factors:        Optional[pd.DataFrame] = None
        self._fundamentals:   Optional[pd.DataFrame] = None
        self._tickers:        Optional[List[str]]    = None
        # Previous weights for turnover penalty
        self._prev_long_w:    Optional[pd.Series]    = None
        self._prev_short_w:   Optional[pd.Series]    = None

    # ── Data loading ──────────────────────────────────────────────────────── #

    def load_data(
        self,
        start: str,
        end: str,
        tickers: Optional[List[str]] = None,
        factors_df: Optional[pd.DataFrame] = None,
        fundamentals_fn=None,
    ) -> None:
        cfg = self.cfg

        if tickers is None:
            tickers = get_sp500_tickers(cfg)
        self._tickers = tickers

        logger.info("Loading price data (%d tickers)...", len(tickers))
        self._prices  = get_monthly_prices(tickers, start, end, cfg)
        self._returns = compute_monthly_returns(self._prices)

        logger.info("Loading factor data...")
        self._factors = factors_df if factors_df is not None else get_ff5_factors(start, end, cfg)

        logger.info("Loading fundamental data...")
        live_tickers = self._prices.columns.tolist()
        _fund_fn     = fundamentals_fn if fundamentals_fn is not None else get_fundamentals
        self._fundamentals = _fund_fn(live_tickers, cfg)

        logger.info(
            "Data loaded: %d tickers, %d months, factors %d months.",
            self._prices.shape[1], self._prices.shape[0], len(self._factors),
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

        # ── 1. Rolling window of returns for FF5/FF6 regression ──────────── #
        idx       = prices.index
        end_loc   = idx.get_loc(as_of) if as_of in idx else len(idx) - 1
        start_loc = max(0, end_loc - cfg.lookback_months + 1)
        window_returns = returns.iloc[start_loc : end_loc + 1]

        valid_cols = window_returns.columns[
            window_returns.notna().sum() >= cfg.min_obs
        ]
        window_returns = window_returns[valid_cols].dropna()

        # ── 2. FF5/FF6 regression → SML alpha ────────────────────────────── #
        logger.info("[%s] Running FF5/FF6 regression...", as_of.date())
        alpha_df = compute_ff5_alpha(window_returns, factors, cfg)
        if alpha_df.empty:
            logger.warning("FF5/FF6 regression returned no results.")
            return self._empty_snapshot(as_of)

        # ── 3. MM valuation screen ────────────────────────────────────────── #
        logger.info("[%s] Running MM valuation screen...", as_of.date())
        live_fund = fund.loc[fund.index.intersection(alpha_df.index)]
        mm_df     = mm_capital_structure_screen(live_fund, cfg, returns=window_returns) \
                    if not live_fund.empty else pd.DataFrame()

        # ── 4. Momentum signal ────────────────────────────────────────────── #
        logger.info("[%s] Computing momentum signal...", as_of.date())
        mkt_returns = (factors["Mkt-RF"] + factors["RF"]).reindex(returns.index)
        mom_df = momentum_signal(prices, as_of, cfg, market_returns=mkt_returns)
        if mom_df.empty:
            logger.warning("Momentum signal is empty.")
            return self._empty_snapshot(as_of)

        # ── 5. Signal combination ─────────────────────────────────────────── #
        signal_method = getattr(cfg, "signal_method", "composite")

        if signal_method == "composite":
            long_final, short_final, composite = self._composite_signal(
                alpha_df, mm_df, mom_df, factors, as_of, cfg
            )
        else:
            long_final, short_final, composite = self._binary_signal(
                alpha_df, mm_df, mom_df, as_of, cfg
            )

        logger.info(
            "[%s] Signal (%s): %d long candidates, %d short candidates.",
            as_of.date(), signal_method, len(long_final), len(short_final),
        )

        if len(long_final) == 0 or len(short_final) == 0:
            logger.warning("Empty leg(s) after signal combination – returning empty snapshot.")
            return self._empty_snapshot(as_of)

        # ── 6. CML / Min-variance position sizing ─────────────────────────── #
        logger.info("[%s] Building optimal legs...", as_of.date())
        sector_map = fund["sector"].to_dict() if "sector" in fund.columns else {}

        # Score for each leg: composite for composite-mode, alpha for binary
        if signal_method == "composite" and not composite.empty:
            long_scores  = composite.reindex(long_final,  fill_value=0.0)
            short_scores = (-composite).reindex(short_final, fill_value=0.0)
        else:
            long_scores  = alpha_df["alpha_annual"].reindex(long_final,  fill_value=0.0)
            short_scores = (-alpha_df["alpha_annual"]).reindex(short_final, fill_value=0.0)

        long_weights = build_leg(
            long_final, long_scores, returns, sector_map, cfg, cfg.n_longs,
            prev_weights=self._prev_long_w,
        )
        short_weights = build_leg(
            short_final, short_scores, returns, sector_map, cfg, cfg.n_shorts,
            prev_weights=self._prev_short_w,
        )

        # ── 7. Beta neutralisation ────────────────────────────────────────── #
        logger.info("[%s] Beta neutralising...", as_of.date())
        betas = estimate_betas(window_returns, mkt_returns, cfg)

        futures_weight = 0.0
        if cfg.use_futures_hedge:
            futures_weight = futures_beta_hedge(long_weights, short_weights, betas, cfg)
        else:
            long_weights, short_weights = beta_neutralise(
                long_weights, short_weights, betas, cfg
            )

        logger.info(
            "[%s] Final portfolio: %d longs, %d shorts%s.",
            as_of.date(), len(long_weights), len(short_weights),
            f", futures={futures_weight:+.4f}×NAV" if cfg.use_futures_hedge else "",
        )

        # Cache weights for next rebalance turnover penalty
        self._prev_long_w  = long_weights
        self._prev_short_w = short_weights

        return PortfolioSnapshot(
            date=as_of,
            long_weights=long_weights,
            short_weights=short_weights,
            alpha_df=alpha_df,
            mm_df=mm_df,
            momentum_df=mom_df,
            combined_longs=long_final,
            combined_shorts=short_final,
            composite_scores=composite,
            futures_weight=futures_weight,
        )

    # ── Composite z-score signal (default, high-alpha mode) ──────────────── #

    def _composite_signal(
        self,
        alpha_df: pd.DataFrame,
        mm_df: pd.DataFrame,
        mom_df: pd.DataFrame,
        factors: pd.DataFrame,
        as_of: pd.Timestamp,
        cfg: TaurusConfig,
    ) -> Tuple[pd.Index, pd.Index, pd.Series]:
        """
        Build a continuous composite score for each stock and return the
        top-N longs and bottom-N shorts.

        composite = w_α × z(alpha_tstat) + w_mm × z(divergence_pct) + w_mom × z(mom_score)

        Benefits over binary filtering:
        • Every stock gets a score — no information discarded at filter boundaries
        • Signals combine additively — a stock slightly below any single threshold
          can still rank highly overall
        • Momentum crash dampening reduces momentum weight in turbulent regimes
        """
        common_idx = alpha_df.index

        # ── Alpha z-score ─────────────────────────────────────────────────── #
        z_alpha = _zscore(alpha_df["alpha_tstat"]).reindex(common_idx, fill_value=0.0)

        # ── MM divergence z-score ─────────────────────────────────────────── #
        if not mm_df.empty and "divergence_pct" in mm_df.columns:
            z_mm = _zscore(mm_df["divergence_pct"]).reindex(common_idx, fill_value=0.0)
        else:
            z_mm = pd.Series(0.0, index=common_idx)

        # ── Momentum z-score  (vol-adjusted score if available) ───────────── #
        mom_col = "mom_score"   # already vol-adjusted in momentum.py when enabled
        if not mom_df.empty and mom_col in mom_df.columns:
            z_mom = _zscore(mom_df[mom_col]).reindex(common_idx, fill_value=0.0)
        else:
            z_mom = pd.Series(0.0, index=common_idx)

        # ── Momentum crash dampening ──────────────────────────────────────── #
        w_alpha = float(cfg.w_alpha)
        w_mm    = float(cfg.w_mm)
        w_mom   = float(cfg.w_momentum)

        crash_regime = (
            not mom_df.empty
            and "crash_regime" in mom_df.columns
            and bool(mom_df["crash_regime"].any())
        )
        if crash_regime:
            # Shift half of momentum weight to alpha in crash regime
            dampen   = w_mom * 0.50
            w_mom   -= dampen
            w_alpha += dampen
            logger.info(
                "[%s] Momentum crash-regime: w_mom %.2f→%.2f, w_alpha %.2f→%.2f",
                as_of.date(), cfg.w_momentum, w_mom, cfg.w_alpha, w_alpha,
            )

        # ── Composite score ───────────────────────────────────────────────── #
        composite = w_alpha * z_alpha + w_mm * z_mm + w_mom * z_mom

        # Take top-2N pool for each leg so the optimizer can select the best N
        pool_size   = cfg.n_longs * 2
        long_pool   = composite.nlargest(pool_size).index
        short_pool  = composite.nsmallest(pool_size).index

        logger.info(
            "[%s] Composite: top-pool %d long, bottom-pool %d short (crash=%s).",
            as_of.date(), len(long_pool), len(short_pool), crash_regime,
        )
        return long_pool, short_pool, composite

    # ── Legacy binary AND-filter signal ──────────────────────────────────── #

    def _binary_signal(
        self,
        alpha_df: pd.DataFrame,
        mm_df: pd.DataFrame,
        mom_df: pd.DataFrame,
        as_of: pd.Timestamp,
        cfg: TaurusConfig,
    ) -> Tuple[pd.Index, pd.Index, pd.Series]:
        """
        Original binary filter: alpha signal AND MM signal AND momentum.
        Kept for comparison / fallback.  Lower Sharpe than composite.
        """
        under_tickers = mm_df[mm_df["underleveraged"]].index if not mm_df.empty else pd.Index([])
        over_tickers  = mm_df[mm_df["overleveraged"]].index  if not mm_df.empty else pd.Index([])
        MM_MIN_RATIO  = 0.10

        mm_underval_ratio = len(under_tickers) / max(len(alpha_df), 1)
        if mm_underval_ratio >= MM_MIN_RATIO:
            alpha_long = alpha_df[alpha_df["signal"] & (alpha_df["alpha_sign"] > 0)]
            long_candidates = alpha_long.index.intersection(under_tickers)
            if len(long_candidates) < max(3, cfg.n_longs // 4):
                under_alpha = alpha_df.loc[alpha_df.index.intersection(under_tickers)]
                if not under_alpha.empty:
                    q50 = under_alpha["alpha_tstat"].quantile(0.50)
                    long_candidates = under_alpha[under_alpha["alpha_tstat"] >= q50].index
        else:
            q70 = alpha_df["alpha_tstat"].quantile(0.70)
            long_candidates = alpha_df[alpha_df["alpha_tstat"] >= q70].index

        mm_overval_ratio = len(over_tickers) / max(len(alpha_df), 1)
        if mm_overval_ratio >= MM_MIN_RATIO:
            q20 = alpha_df["alpha_tstat"].quantile(0.20)
            short_candidates = alpha_df[alpha_df["alpha_tstat"] <= q20].index.intersection(over_tickers)
        else:
            q30 = alpha_df["alpha_tstat"].quantile(0.30)
            short_candidates = alpha_df[alpha_df["alpha_tstat"] <= q30].index

        mom_long  = mom_df[mom_df["mom_pos"]].index
        mom_short = mom_df[mom_df["mom_neg"]].index
        long_final  = long_candidates.intersection(mom_long)
        short_final = short_candidates.intersection(mom_short)

        # Fallbacks
        min_final = max(2, cfg.n_longs // 5)
        if len(long_final) < min_final:
            q40 = mom_df["mom_raw"].quantile(0.40)
            long_final = long_candidates.intersection(mom_df[mom_df["mom_raw"] >= q40].index)

        if len(short_final) < min_final:
            q60 = mom_df["mom_raw"].quantile(0.60)
            short_final = short_candidates.intersection(mom_df[mom_df["mom_raw"] <= q60].index)

        if len(long_final) == 0:
            q75 = alpha_df["alpha_tstat"].quantile(0.75)
            long_final = alpha_df[alpha_df["alpha_tstat"] >= q75].index

        if len(short_final) == 0:
            q25 = alpha_df["alpha_tstat"].quantile(0.25)
            short_final = alpha_df[alpha_df["alpha_tstat"] <= q25].index

        composite = pd.Series(dtype=float)   # empty for binary mode
        return long_final, short_final, composite

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

    # ── Full back-test ─────────────────────────────────────────────────────── #

    def backtest(
        self,
        start: str,
        end: str,
        tickers: Optional[List[str]] = None,
        factors_df: Optional[pd.DataFrame] = None,
        fundamentals_fn=None,
    ) -> "BacktestResult":
        """Run a full vectorised back-test over [start, end]."""
        warm_up_months = cfg_warm_months(self.cfg)
        warm_start = (
            pd.Timestamp(start) - pd.DateOffset(months=warm_up_months)
        ).strftime("%Y-%m-%d")

        self.load_data(warm_start, end, tickers, factors_df=factors_df,
                       fundamentals_fn=fundamentals_fn)

        prices  = self._prices
        returns = self._returns
        idx     = prices.index

        rebalance_dates = pd.date_range(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            freq=self.cfg.rebalance_freq,
        ).intersection(idx)

        # Reset previous weights for clean backtest
        self._prev_long_w  = None
        self._prev_short_w = None

        snapshots: List[PortfolioSnapshot] = []
        for reb_date in rebalance_dates:
            snap = self.run(as_of=reb_date)
            snapshots.append(snap)

        return BacktestResult(
            snapshots=snapshots,
            prices=prices,
            returns=returns,
            factors=self._factors,
            cfg=self.cfg,
        )


def cfg_warm_months(cfg: TaurusConfig) -> int:
    return cfg.lookback_months + cfg.momentum_months + cfg.momentum_skip + 3


# --------------------------------------------------------------------------- #
#  BacktestResult                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class BacktestResult:
    snapshots: List[PortfolioSnapshot]
    prices:    pd.DataFrame
    returns:   pd.DataFrame
    factors:   pd.DataFrame       # FF5/FF6 factors (needed for correct Mkt return)
    cfg:       TaurusConfig

    def portfolio_returns(self) -> pd.Series:
        """
        Compute monthly strategy P&L from position snapshots.

        Fixes:
        • Market return for futures P&L uses factors["Mkt-RF"] + factors["RF"]
          (previously: stock average, which ≠ index return).
        • Turnover cost deducted at rebalance date.
        • Financing cost deducted monthly.
        """
        pnl_records = []
        cost_bps    = self.cfg.total_cost_bps / 10_000

        prev_lw = pd.Series(dtype=float)
        prev_sw = pd.Series(dtype=float)

        # Pre-compute true market monthly returns for futures P&L
        mkt_ret_series = (
            self.factors["Mkt-RF"] + self.factors["RF"]
        ).reindex(self.returns.index)

        lev  = self.cfg.gross_leverage
        half = lev / 2.0
        margin_cost_m   = (self.cfg.margin_cost_annual + self.cfg.borrow_cost_annual) / 12
        leverage_cost_m = (lev - 1.0) * margin_cost_m

        for i, snap in enumerate(self.snapshots):
            hold_start = snap.date
            hold_end   = (
                self.snapshots[i + 1].date if i + 1 < len(self.snapshots)
                else self.returns.index[-1]
            )

            month_range = self.returns.loc[
                (self.returns.index > hold_start) &
                (self.returns.index <= hold_end)
            ]

            futures_roll_m = (
                self.cfg.futures_roll_cost_quarterly / 3.0
                if self.cfg.use_futures_hedge and snap.futures_weight != 0.0
                else 0.0
            )

            for month_date, month_ret in month_range.iterrows():
                long_ret  = (snap.long_weights * month_ret.reindex(
                    snap.long_weights.index,  fill_value=0)).sum()
                short_ret = (snap.short_weights * month_ret.reindex(
                    snap.short_weights.index, fill_value=0)).sum()
                gross_ret = half * long_ret - half * short_ret
                gross_ret -= leverage_cost_m

                # Futures P&L: use true index return (not stock average)
                if snap.futures_weight != 0.0:
                    true_mkt_ret = float(mkt_ret_series.get(month_date, 0.0))
                    gross_ret   += snap.futures_weight * true_mkt_ret
                    gross_ret   -= futures_roll_m

                pnl_records.append({"date": month_date, "return": gross_ret})

            # Turnover cost at rebalance
            turnover = _compute_turnover(snap.long_weights, prev_lw) + \
                       _compute_turnover(snap.short_weights, prev_sw)
            cost = turnover * cost_bps
            if pnl_records:
                pnl_records[-1]["return"] -= cost

            prev_lw = snap.long_weights
            prev_sw = snap.short_weights

        if not pnl_records:
            return pd.Series(dtype=float)

        return pd.DataFrame(pnl_records).set_index("date")["return"]

    def analytics(self) -> Dict[str, float]:
        """Compute standard performance metrics."""
        ret = self.portfolio_returns()
        if ret.empty:
            return {}

        cum      = (1 + ret).cumprod()
        total    = cum.iloc[-1] - 1
        ann      = (1 + total) ** (12 / len(ret)) - 1
        vol      = ret.std() * np.sqrt(12)
        sharpe   = (ann - self.cfg.risk_free_rate_annual) / vol if vol > 0 else np.nan
        drawdown = (cum / cum.cummax() - 1).min()
        calmar   = ann / abs(drawdown) if drawdown < 0 else np.nan

        # Sortino ratio (downside deviation)
        neg_ret  = ret[ret < 0]
        down_vol = neg_ret.std() * np.sqrt(12) if len(neg_ret) > 1 else np.nan
        sortino  = (ann - self.cfg.risk_free_rate_annual) / down_vol \
                   if (down_vol is not None and down_vol > 0) else np.nan

        return {
            "total_return":         float(total),
            "total_return_pct":     float(total * 100),
            "annual_return":        float(ann),
            "annual_return_pct":    float(ann * 100),
            "annual_vol":           float(vol),
            "annual_vol_pct":       float(vol * 100),
            "sharpe_ratio":         float(sharpe),
            "sortino_ratio":        float(sortino) if not np.isnan(sortino) else 0.0,
            "max_drawdown":         float(drawdown),
            "max_drawdown_pct":     float(drawdown * 100),
            "calmar_ratio":         float(calmar),
            "n_months":             len(ret),
            "n_rebalances":         len(self.snapshots),
        }

    def positions_df(self) -> pd.DataFrame:
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
