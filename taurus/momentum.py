"""
Taurus – Momentum filter.

Jegadeesh & Titman (1993) cross-sectional price momentum:
  • 12-month trailing return, skipping the most recent month
    (avoids short-term reversal contamination).
  • Cross-sectionally z-scored so rankings are comparable.
  • Stocks are labelled mom+ (top tercile) or mom- (bottom tercile).

Implementation
──────────────
• Fully vectorised over the stock universe using pandas shift arithmetic.
• `momentum_signal` returns both the raw score and a ternary label.
• `rolling_momentum_signal` produces the signal at every rebalance date.
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
) -> pd.DataFrame:
    """
    Compute cross-sectional momentum as of a single date.

    Parameters
    ----------
    prices  : monthly adjusted-close prices (month-end index, wide format)
    as_of   : the rebalance date (must be in prices.index)

    Returns
    -------
    DataFrame indexed by ticker with columns:
        mom_raw   – cumulative return over the signal period
        mom_score – cross-sectional z-score of mom_raw
        mom_pos   – bool: top tercile (momentum+)
        mom_neg   – bool: bottom tercile (momentum-)
        mom_sign  – +1 for mom+, -1 for mom-, 0 otherwise
    """
    skip   = cfg.momentum_skip
    window = cfg.momentum_months

    idx = prices.index
    as_of_loc = idx.get_loc(as_of)

    # End of measurement period = skip N months back
    end_loc   = as_of_loc - skip
    start_loc = end_loc - window + 1

    if start_loc < 0:
        logger.debug("Insufficient history for momentum at %s.", as_of)
        return pd.DataFrame()

    p_start = prices.iloc[start_loc]
    p_end   = prices.iloc[end_loc]

    mom_raw = (p_end / p_start - 1.0).replace([np.inf, -np.inf], np.nan)

    # Cross-sectional z-score (robust: use median / MAD)
    med  = mom_raw.median()
    mad  = (mom_raw - med).abs().median()
    if mad > 0:
        mom_score = (mom_raw - med) / (mad * 1.4826)   # normalise to σ
    else:
        mom_score = mom_raw - med

    # Tercile classification
    q33 = mom_raw.quantile(1 / 3)
    q67 = mom_raw.quantile(2 / 3)

    mom_pos = mom_score > 0                  # above median
    mom_neg = mom_score < 0                  # below median

    # Stricter: require top/bottom tercile for stronger signal
    mom_pos_strict = mom_raw >= q67
    mom_neg_strict = mom_raw <= q33

    mom_sign = np.where(mom_pos_strict, 1, np.where(mom_neg_strict, -1, 0))

    result = pd.DataFrame(
        {
            "mom_raw":   mom_raw,
            "mom_score": mom_score,
            "mom_pos":   mom_pos,
            "mom_neg":   mom_neg,
            "mom_pos_strict": mom_pos_strict,
            "mom_neg_strict": mom_neg_strict,
            "mom_sign":  pd.Series(mom_sign, index=mom_raw.index),
        }
    ).dropna(subset=["mom_raw"])

    logger.debug(
        "Momentum @ %s: %d stocks, %d mom+ (strict), %d mom- (strict).",
        as_of.date(),
        len(result),
        result["mom_pos_strict"].sum(),
        result["mom_neg_strict"].sum(),
    )
    return result


# --------------------------------------------------------------------------- #
#  Rolling momentum (all rebalance dates)                                      #
# --------------------------------------------------------------------------- #

def rolling_momentum_signal(
    prices: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute momentum signal at every rebalance date.

    Returns a DataFrame with MultiIndex (date, ticker).
    """
    dates = prices.index
    min_needed = cfg.momentum_months + cfg.momentum_skip + 1

    rebalance_dates = pd.date_range(
        start=dates[min_needed],
        end=dates[-1],
        freq=cfg.rebalance_freq,
    ).intersection(dates)

    frames = []
    for reb_date in rebalance_dates:
        df = momentum_signal(prices, reb_date, cfg)
        if df.empty:
            continue
        df.index = pd.MultiIndex.from_tuples(
            [(reb_date, t) for t in df.index], names=["date", "ticker"]
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames)
