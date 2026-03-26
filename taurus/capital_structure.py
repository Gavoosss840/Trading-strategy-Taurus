"""
Taurus – Modigliani-Miller capital structure screen.

Upgraded MM Valuation Engine (adapted from user's MMOptionsBot):
────────────────────────────────────────────────────────────────
VL (levered firm value) = VU + PV(Tax Shield) - PV(Distress) - Agency Costs

Where:
  VU             = Unlevered value  = (Market Cap + Net Debt) - Tax Shield
  Tax Shield     = τ × Interest / (rf + spread)
  Distress Costs = Merton-model P(default) × 20% × EV
  Agency Costs   = f(leverage, FCF yield)

Signal:
  divergence = (VL_theoretical - Market Cap) / Market Cap
  > +threshold  → undervalued  → underleveraged flag  → LONG candidate
  < -threshold  → overvalued   → overleveraged flag   → SHORT candidate

This replaces the simpler sector-median leverage approach and naturally
produces both LONG and SHORT candidates in any market regime.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Single-stock MM valuation                                                   #
# --------------------------------------------------------------------------- #

def _mm_valuation(row: pd.Series, rf: float = 0.045) -> Dict:
    """
    Compute MM theoretical value for one stock from a fundamentals row.

    Parameters
    ----------
    row : Series with keys: market_cap, total_debt, total_equity, ebit,
          interest_expense, total_assets, tax_rate, fcf (optional),
          beta (optional), price_hist_vol (optional)
    rf  : annual risk-free rate

    Returns
    -------
    dict with: VU, tax_shield, distress_costs, agency_costs,
               VL_theoretical, divergence_pct, signal
    """
    market_cap        = float(row.get("market_cap",        0) or 0)
    total_debt        = float(row.get("total_debt",        0) or 0)
    cash              = float(row.get("cash",              0) or 0)
    total_equity      = float(row.get("total_equity",      0) or 1)
    ebit              = float(row.get("ebit",              0) or 0)
    interest_expense  = float(row.get("interest_expense",  0) or 0)
    total_assets      = float(row.get("total_assets",      0) or 1)
    tax_rate          = float(row.get("tax_rate",         0.21) or 0.21)
    fcf               = float(row.get("fcf",               0) or 0)
    sigma_equity      = float(row.get("price_vol_annual",  0.30) or 0.30)

    if market_cap <= 0:
        return _empty_mm(market_cap)

    # ── Net debt ──────────────────────────────────────────────────────────── #
    net_debt = max(total_debt - cash, 0)
    ev = market_cap + net_debt

    # ── 1. PV of Tax Shield ───────────────────────────────────────────────── #
    if interest_expense == 0 and total_debt > 0:
        interest_expense = total_debt * 0.05   # impute 5% coupon

    annual_tax_shield = tax_rate * interest_expense
    shield_discount   = rf + 0.02               # slight credit spread
    pv_tax_shield     = annual_tax_shield / shield_discount if shield_discount > 0 else 0

    # ── 2. Unlevered value ────────────────────────────────────────────────── #
    VU = market_cap + net_debt - pv_tax_shield

    # ── 3. Financial distress costs (Merton model) ──────────────────────── #
    E          = max(market_cap, 1)
    D          = max(total_debt, 1)
    V_firm     = max(ev, 1)

    sigma_assets = sigma_equity * (E / (E + D))   # de-lever equity vol
    mu = rf
    T  = 1.0

    try:
        d2 = (np.log(V_firm / D) + (mu - 0.5 * sigma_assets**2) * T) \
             / (sigma_assets * np.sqrt(T))
        prob_default = float(norm.cdf(-d2))
    except Exception:
        prob_default = 0.0

    distress_cost_rate = 0.20
    pv_distress = prob_default * distress_cost_rate * V_firm

    # ── 4. Agency costs ───────────────────────────────────────────────────── #
    leverage = total_debt / total_equity if total_equity > 0 else 0
    agency_score = 0.0
    if leverage > 2.0:
        agency_score += (leverage - 2.0) * 0.05
    fcf_yield = abs(fcf) / market_cap if market_cap > 0 else 0
    if fcf_yield > 0.10:
        agency_score += (fcf_yield - 0.10) * 0.5
    pv_agency = agency_score * market_cap

    # ── 5. Levered theoretical value ──────────────────────────────────────── #
    VL = VU + pv_tax_shield - pv_distress - pv_agency

    divergence = (VL - market_cap) / market_cap if market_cap > 0 else 0.0
    divergence_pct = divergence * 100.0

    return {
        "VU":             VU,
        "pv_tax_shield":  pv_tax_shield,
        "pv_distress":    pv_distress,
        "pv_agency":      pv_agency,
        "VL_theoretical": VL,
        "divergence_pct": divergence_pct,
        "prob_default":   prob_default,
    }


def _empty_mm(market_cap: float) -> Dict:
    return {
        "VU": 0.0, "pv_tax_shield": 0.0, "pv_distress": 0.0,
        "pv_agency": 0.0, "VL_theoretical": 0.0,
        "divergence_pct": 0.0, "prob_default": 0.0,
    }


# --------------------------------------------------------------------------- #
#  Historical volatility helper (called from data layer if available)         #
# --------------------------------------------------------------------------- #

def add_price_vol(
    fundamentals: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Attach annualised price volatility to a fundamentals DataFrame.
    Call this before mm_capital_structure_screen when returns are available.
    """
    vols = returns.tail(lookback).std() * np.sqrt(12)
    vols.name = "price_vol_annual"
    return fundamentals.join(vols, how="left")


# --------------------------------------------------------------------------- #
#  Main screen                                                                 #
# --------------------------------------------------------------------------- #

def mm_capital_structure_screen(
    fundamentals: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
    returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply the full MM valuation screen.

    Parameters
    ----------
    fundamentals : DataFrame indexed by ticker (from data.get_fundamentals)
    cfg          : TaurusConfig
    returns      : optional monthly returns DataFrame to compute price vol

    Returns
    -------
    DataFrame indexed by ticker with columns:
        VL_theoretical, divergence_pct, prob_default,
        underleveraged (bool), overleveraged (bool),
        pv_tax_shield, pv_distress, pv_agency
    """
    rf = cfg.risk_free_rate_annual
    threshold = cfg.leverage_gap_threshold * 100   # convert to %

    # Attach historical vol if returns provided
    df = fundamentals.copy()
    if returns is not None:
        df = add_price_vol(df, returns)

    # Run MM valuation row-by-row (fast enough for 500 stocks)
    rows = []
    for ticker, row in df.iterrows():
        result = _mm_valuation(row, rf=rf)
        result["ticker"] = ticker
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).set_index("ticker")

    # ── Flags ─────────────────────────────────────────────────────────────── #
    result_df["underleveraged"] = result_df["divergence_pct"] >  threshold   # undervalued → LONG
    result_df["overleveraged"]  = result_df["divergence_pct"] < -threshold   # overvalued  → SHORT

    # Interest coverage guard (still useful)
    ic = (df["ebit"] / df["interest_expense"].where(df["interest_expense"] > 0)
          ).reindex(result_df.index).fillna(np.inf)
    result_df["ic_ratio"] = ic
    result_df["overleveraged"] |= (ic < cfg.min_interest_coverage)

    n_under = result_df["underleveraged"].sum()
    n_over  = result_df["overleveraged"].sum()
    logger.info(
        "MM screen: %d undervalued (LONG), %d overvalued (SHORT) of %d stocks.",
        n_under, n_over, len(result_df),
    )
    return result_df
