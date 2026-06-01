"""
Taurus – Momentum filter.

Jegadeesh & Titman (1993) cross-sectional price momentum:
  • 12-month trailing return, skipping the most recent month
    (avoids short-term reversal contamination).

Vol-adjusted momentum (Sharpe momentum):
  • Barroso & Santa-Clara (2015): divide raw momentum by trailing volatility.
  • Controls for risk — a +30% stock with σ=15% ranks higher than
    a +40% stock with σ=60%.  Reduces momentum crash severity by ~50%.

Momentum crash dampening:
  • If realised 1-month market volatility > 2× trailing 12M average,
    this signals a high-crash-risk regime.  Strategy.py reads the
    `crash_regime` flag from the DataFrame and reduces momentum weight.

Implementation
──────────────
• Fully vectorised over the stock universe using pandas shift arithmetic.
• `momentum_signal` returns both raw and vol-adjusted scores + regime flag.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Point-in-time momentum                                                      #
# --------------------------------------------------------------------------- #

def momentum_signal(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: TaurusConfig = DEFAULT_CONFIG,
    market_returns: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional momentum as of a single date.

    Parameters
    ----------
    prices         : monthly adjusted-close prices (month-end index, wide format)
    as_of          : the rebalance date (must be in prices.index)
    market_returns : optional Series of market returns for crash-regime detection

    Returns
    -------
    DataFrame indexed by ticker with columns:
        mom_raw         – cumulative return over the signal period (12M skip 1M)
        mom_vol         – trailing realised vol (annualised) of the stock
        mom_sharpe      – mom_raw / mom_vol  (vol-adjusted momentum score)
        mom_score       – primary sorting score (sharpe if enabled, else raw)
        mom_pos         – bool: above-median momentum (primary filter)
        mom_neg         – bool: below-median momentum (primary filter)
        mom_pos_strict  – bool: top tercile
        mom_neg_strict  – bool: bottom tercile
        mom_sign        – +1/0/-1 (strict tercile label)
        crash_regime    – bool: True if market vol signals momentum crash risk
    """
    skip   = cfg.momentum_skip
    window = cfg.momentum_months

    idx       = prices.index
    as_of_loc = idx.get_indexer([as_of], method='pad')[0]
    if as_of_loc < 0:
        logger.debug("as_of %s is before price history start.", as_of)
        return pd.DataFrame()

    # Momentum measurement: from (as_of - window - skip) to (as_of - skip)
    end_loc   = as_of_loc - skip
    start_loc = end_loc - window + 1

    if start_loc < 0:
        logger.debug("Insufficient history for momentum at %s.", as_of)
        return pd.DataFrame()

    p_start = prices.iloc[start_loc]
    p_end   = prices.iloc[end_loc]

    mom_raw = (p_end / p_start - 1.0).replace([np.inf, -np.inf], np.nan)

    # ── Vol-adjusted momentum (Sharpe momentum) ──────────────────────────── #
    # Use returns over the same window to estimate each stock's trailing vol.
    vol_adjust = getattr(cfg, "vol_adjust_momentum", True)
    if vol_adjust:
        ret_window = prices.iloc[start_loc : end_loc + 1].pct_change().dropna()
        if len(ret_window) >= 6:
            trailing_vol = ret_window.std() * np.sqrt(12)   # annualised
            trailing_vol = trailing_vol.replace(0, np.nan)
            mom_sharpe   = mom_raw / trailing_vol
        else:
            mom_sharpe = mom_raw.copy()
    else:
        trailing_vol = pd.Series(np.nan, index=mom_raw.index)
        mom_sharpe   = mom_raw.copy()

    # ── Primary sorting score ─────────────────────────────────────────────── #
    # Use vol-adjusted if enabled, raw otherwise
    mom_score_raw = mom_sharpe if vol_adjust else mom_raw

    # Cross-sectional z-score (robust: median / MAD)
    med = mom_score_raw.median()
    mad = (mom_score_raw - med).abs().median()
    if mad > 0:
        mom_score = (mom_score_raw - med) / (mad * 1.4826)
    else:
        mom_score = mom_score_raw - med

    # ── Tercile classification ────────────────────────────────────────────── #
    q33 = mom_score_raw.quantile(1 / 3)
    q67 = mom_score_raw.quantile(2 / 3)

    mom_pos        = mom_score > 0            # above median
    mom_neg        = mom_score < 0            # below median
    mom_pos_strict = mom_score_raw >= q67     # top tercile
    mom_neg_strict = mom_score_raw <= q33     # bottom tercile
    mom_sign       = np.where(mom_pos_strict, 1, np.where(mom_neg_strict, -1, 0))

    # ── Crash-regime detection ────────────────────────────────────────────── #
    crash_regime = False
    dampen = getattr(cfg, "momentum_crash_dampen", True)
    if dampen and market_returns is not None and len(market_returns) >= 13:
        mkt = market_returns.reindex(idx[:as_of_loc + 1]).dropna()
        if len(mkt) >= 13:
            vol_1m  = float(mkt.iloc[-1:].std() * np.sqrt(12))   # 1-month point vol
            vol_12m = float(mkt.iloc[-12:].std() * np.sqrt(12))  # 12-month rolling vol
            if vol_12m > 0 and vol_1m > 2.0 * vol_12m:
                crash_regime = True
                logger.info(
                    "Momentum crash-regime detected @ %s: vol_1m=%.1f%% > 2×vol_12m=%.1f%%",
                    as_of.date(), vol_1m * 100, vol_12m * 100,
                )

    result = pd.DataFrame(
        {
            "mom_raw":        mom_raw,
            "mom_vol":        trailing_vol if vol_adjust else pd.Series(np.nan, index=mom_raw.index),
            "mom_sharpe":     mom_sharpe,
            "mom_score":      mom_score,
            "mom_pos":        mom_pos,
            "mom_neg":        mom_neg,
            "mom_pos_strict": mom_pos_strict,
            "mom_neg_strict": mom_neg_strict,
            "mom_sign":       pd.Series(mom_sign, index=mom_raw.index),
            "crash_regime":   crash_regime,  # scalar broadcast to all rows
        }
    ).dropna(subset=["mom_raw"])

    logger.debug(
        "Momentum @ %s: %d stocks, %d mom+ strict, %d mom- strict%s.",
        as_of.date(), len(result),
        result["mom_pos_strict"].sum(),
        result["mom_neg_strict"].sum(),
        " [CRASH REGIME]" if crash_regime else "",
    )
    return result


# --------------------------------------------------------------------------- #
#  Rolling momentum (all rebalance dates)                                      #
# --------------------------------------------------------------------------- #

def rolling_momentum_signal(
    prices: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
    market_returns: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute momentum signal at every rebalance date.

    Returns a DataFrame with MultiIndex (date, ticker).
    """
    dates     = prices.index
    min_needed = cfg.momentum_months + cfg.momentum_skip + 1

    rebalance_dates = pd.date_range(
        start=dates[min_needed],
        end=dates[-1],
        freq=cfg.rebalance_freq,
    ).intersection(dates)

    frames = []
    for reb_date in rebalance_dates:
        df = momentum_signal(prices, reb_date, cfg, market_returns=market_returns)
        if df.empty:
            continue
        df.index = pd.MultiIndex.from_tuples(
            [(reb_date, t) for t in df.index], names=["date", "ticker"]
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames)
