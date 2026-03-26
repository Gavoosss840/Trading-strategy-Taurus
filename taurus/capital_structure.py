"""
Taurus – Modigliani-Miller capital structure screen.

Theory
──────
Under MM with corporate taxes, the value of a levered firm exceeds an
unlevered firm by the present value of the tax shield (τ × D).  The
trade-off theory adds financial distress costs, giving an interior optimal
leverage ratio.

Empirical target-leverage model (Flannery & Rangan 2006)
──────────────────────────────────────────────────────────
Target D/A* = β₀ + β₁·Profitability + β₂·Tangibility + β₃·log(Assets)
              + β₄·M/B + β₅·Tax_rate

Where:
  Profitability = EBIT / Assets
  Tangibility   = approximated via sector (asset-heavy sectors carry more debt)
  M/B           = market_cap / book_equity
  Tax_rate      = effective tax rate (higher → more benefit from debt)

Because individual stock financials arrive with noise and lag, we:
  1. Compute each stock's target leverage cross-sectionally using sector medians.
  2. Flag the capital structure gap: (actual − target) / target.
  3. Apply an interest-coverage floor for distress detection.

Outputs
────────
DataFrame indexed by ticker with columns:
  actual_de    – Debt/Equity (book)
  actual_da    – Debt/Assets
  target_da    – estimated optimal Debt/Assets
  cs_gap       – (actual_da − target_da) / target_da
  underleveraged  – bool
  overleveraged   – bool
  ic_ratio        – EBIT / interest expense
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Sector-level tangibility proxies                                            #
# --------------------------------------------------------------------------- #

# Typical asset tangibility ratios by GICS sector (empirical estimates).
# Higher tangibility → sector can support more debt.
_SECTOR_TANGIBILITY = {
    "Energy":                   0.70,
    "Materials":                0.55,
    "Industrials":              0.50,
    "Consumer Discretionary":   0.35,
    "Consumer Staples":         0.40,
    "Health Care":              0.30,
    "Financials":               0.10,   # leverage has different meaning
    "Information Technology":   0.25,
    "Communication Services":   0.30,
    "Utilities":                0.75,
    "Real Estate":              0.80,
    "Unknown":                  0.40,
}


# --------------------------------------------------------------------------- #
#  Target leverage estimation                                                  #
# --------------------------------------------------------------------------- #

def estimate_target_leverage(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Estimate target Debt/Assets for each ticker using cross-sectional
    characteristics.

    Parameters
    ----------
    fundamentals : DataFrame indexed by ticker with columns produced by
                   data.get_fundamentals()

    Returns
    -------
    target_da : Series indexed by ticker
    """
    df = fundamentals.copy()

    # ── Derived ratios ───────────────────────────────────────────────────── #
    df["profitability"] = np.where(
        df["total_assets"] > 0,
        df["ebit"] / df["total_assets"],
        np.nan,
    )
    df["log_assets"] = np.log(np.where(df["total_assets"] > 0, df["total_assets"], np.nan))
    df["mb_ratio"] = np.where(
        df["total_equity"] > 0,
        df["market_cap"] / df["total_equity"],
        np.nan,
    )
    df["tangibility"] = df["sector"].map(_SECTOR_TANGIBILITY).fillna(0.40)
    df["tax_rate"] = df["tax_rate"].clip(0.0, 0.40).fillna(0.21)

    # ── Target leverage coefficients (calibrated on empirical literature) ── #
    #   D/A* ≈ 0.10
    #         + 0.30 × tangibility          (asset-heavy → more debt)
    #         − 0.20 × profitability        (profitable → less need for debt)
    #         + 0.02 × log_assets           (larger firms → more access to debt)
    #         − 0.02 × log(M/B)             (growth firms → less debt)
    #         + 0.15 × tax_rate             (higher tax → more benefit from shield)

    log_mb = np.log(np.where(df["mb_ratio"].fillna(1) > 0, df["mb_ratio"].fillna(1), 1))

    target_da = (
        0.10
        + 0.30 * df["tangibility"]
        - 0.20 * df["profitability"].fillna(0.05)
        + 0.02 * df["log_assets"].fillna(df["log_assets"].median())
        - 0.02 * log_mb
        + 0.15 * df["tax_rate"]
    )

    # Clamp to plausible range [0.05, 0.85]
    target_da = target_da.clip(0.05, 0.85)

    # ── Financial firms: capital structure is regulatory, not trade-off ─── #
    financials_mask = df["sector"].isin(["Financials", "Real Estate"])
    # For financials, use sector median as best estimate
    sector_medians = _sector_median_leverage(df)
    target_da = target_da.where(~financials_mask, sector_medians)

    return target_da.rename("target_da")


def _sector_median_leverage(df: pd.DataFrame) -> pd.Series:
    """Return a Series aligned to df.index with each stock's sector median D/A."""
    if "actual_da" not in df.columns:
        # Compute on the fly
        total_debt   = df.get("total_debt",   pd.Series(np.nan, index=df.index))
        total_assets = df.get("total_assets", pd.Series(np.nan, index=df.index))
        actual_da = (total_debt / np.where(total_assets > 0, total_assets, np.nan)).clip(0, 1)
        df = df.copy()
        df["actual_da"] = actual_da

    sector_med = df.groupby("sector")["actual_da"].transform("median")
    return sector_med.fillna(0.30)


# --------------------------------------------------------------------------- #
#  Main screen                                                                 #
# --------------------------------------------------------------------------- #

def mm_capital_structure_screen(
    fundamentals: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Apply the MM capital structure screen to the universe.

    Returns a DataFrame indexed by ticker with columns:
        actual_de, actual_da, target_da, cs_gap,
        underleveraged, overleveraged, ic_ratio
    """
    df = fundamentals.copy()

    # ── Actual leverage ──────────────────────────────────────────────────── #
    debt   = df["total_debt"].fillna(0.0)
    equity = df["total_equity"]
    assets = df["total_assets"]

    actual_da = (debt / assets.where(assets > 0)).clip(0.0, 0.99)
    actual_de = (debt / equity.where(equity > 0)).clip(0.0, 10.0)

    # ── Target leverage ──────────────────────────────────────────────────── #
    df["actual_da"] = actual_da
    target_da = estimate_target_leverage(df)

    # ── Capital structure gap ─────────────────────────────────────────────── #
    cs_gap = (actual_da - target_da) / target_da.where(target_da > 0.01)

    # ── Interest coverage ─────────────────────────────────────────────────── #
    ic_ratio = (
        df["ebit"] / df["interest_expense"].where(df["interest_expense"] > 0)
    ).fillna(np.inf).replace([np.inf, -np.inf], np.nan)

    # ── Flags ─────────────────────────────────────────────────────────────── #
    threshold = cfg.leverage_gap_threshold
    ic_floor  = cfg.min_interest_coverage

    underleveraged = (cs_gap < -threshold)                   # too little debt
    overleveraged  = (cs_gap >  threshold) | (
        ic_ratio.fillna(np.inf) < ic_floor                   # or distressed
    )

    result = pd.DataFrame(
        {
            "actual_de":      actual_de,
            "actual_da":      actual_da,
            "target_da":      target_da,
            "cs_gap":         cs_gap,
            "ic_ratio":       ic_ratio,
            "underleveraged": underleveraged.astype(bool),
            "overleveraged":  overleveraged.astype(bool),
        },
        index=df.index,
    )

    n_under = result["underleveraged"].sum()
    n_over  = result["overleveraged"].sum()
    logger.info(
        "MM screen: %d underleveraged, %d overleveraged (of %d).",
        n_under, n_over, len(result),
    )
    return result
