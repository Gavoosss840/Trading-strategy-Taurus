"""
Taurus – Modigliani-Miller capital structure screen.

Upgraded MM Valuation Engine
─────────────────────────────
VL (levered firm value) = VU + PV(Tax Shield) - PV(Distress) - Agency Costs

Where:
  VU             = Unlevered value  = (Market Cap + Net Debt) - Tax Shield
  Tax Shield     = τ × Interest / (rf + credit_spread)
  Distress Costs = Merton-model P(default) × distress_rate × EV
  Agency Costs   = f(leverage, FCF yield)

Improvements over baseline:
  • Industry-specific distress cost rates (intangible-heavy firms lose more in default)
  • Leverage-based credit spread (instead of flat +200 bps for all firms)
  • Student-t Merton default probability (fat-tail correction)

Signal:
  divergence = (VL_theoretical - Market Cap) / Market Cap
  > +threshold  → undervalued  → underleveraged flag  → LONG candidate
  < -threshold  → overvalued   → overleveraged flag   → SHORT candidate
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, t as _t_dist

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Industry-specific distress cost rates                                       #
# --------------------------------------------------------------------------- #
# Distress costs = fraction of firm value destroyed in bankruptcy.
# Intangible-heavy firms (tech, pharma) lose customer relationships, IP;
# asset-heavy firms (utilities, real estate) recover more via liquidation.
#
# Sources: Altman (1984), Andrade & Kaplan (1998), Bris et al. (2006).

_SECTOR_DISTRESS_RATE: Dict[str, float] = {
    "Information Technology":  0.40,   # high intangibles (IP, talent, SaaS contracts)
    "Communication Services":  0.35,   # brand value, content rights at risk
    "Health Care":             0.35,   # regulatory risk, IP / pipeline destruction
    "Consumer Discretionary":  0.25,   # brand + inventory write-downs
    "Consumer Staples":        0.20,   # moderate — physical assets + brands
    "Industrials":             0.18,   # machinery, equipment retain value
    "Materials":               0.15,   # physical commodities / land
    "Energy":                  0.15,   # reserves, equipment recoverable
    "Financials":              0.10,   # regulated; liquid assets, recoverable
    "Real Estate":             0.08,   # tangible land/buildings: high recovery
    "Utilities":               0.10,   # regulated; essential assets stay intact
    "Unknown":                 0.20,   # default
}


def _distress_rate(sector: str, cfg: TaurusConfig) -> float:
    """Return distress cost rate for a given sector."""
    if not getattr(cfg, "industry_distress_costs", True):
        return 0.20   # flat legacy value
    return _SECTOR_DISTRESS_RATE.get(sector, _SECTOR_DISTRESS_RATE["Unknown"])


def _credit_spread(leverage_ratio: float, cfg: TaurusConfig) -> float:
    """
    Estimate the credit spread on debt as a function of leverage (D/E).

    Calibrated roughly to US investment-grade / high-yield spreads:
      D/E < 0.5  → spread ≈ 0.01 (100 bps, AAA/AA)
      D/E = 1.0  → spread ≈ 0.02 (200 bps, A/BBB)
      D/E = 2.0  → spread ≈ 0.04 (400 bps, BB)
      D/E > 5.0  → capped at 0.10 (1000 bps, deep distress)
    """
    if not getattr(cfg, "variable_credit_spread", True):
        return 0.02   # flat legacy value
    spread = 0.005 + 0.03 * min(leverage_ratio, 5.0)
    return float(np.clip(spread, 0.005, 0.10))


# --------------------------------------------------------------------------- #
#  Single-stock MM valuation                                                   #
# --------------------------------------------------------------------------- #

def _mm_valuation(
    row: pd.Series,
    rf: float = 0.045,
    return_df: Optional[float] = None,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> Dict:
    """
    Compute MM theoretical value for one stock from a fundamentals row.

    Parameters
    ----------
    row       : Series with keys: market_cap, total_debt, total_equity, ebit,
                interest_expense, total_assets, tax_rate, fcf, sector,
                price_vol_annual (optional)
    rf        : annual risk-free rate
    return_df : Student-t degrees of freedom for Merton (None → Normal)
    cfg       : TaurusConfig for sector distress + spread settings

    Returns
    -------
    dict with: VU, pv_tax_shield, pv_distress, pv_agency,
               VL_theoretical, divergence_pct, prob_default
    """
    market_cap       = float(row.get("market_cap",        0) or 0)
    total_debt       = float(row.get("total_debt",        0) or 0)
    cash             = float(row.get("cash",              0) or 0)
    total_equity     = float(row.get("total_equity",      0) or 1)
    ebit             = float(row.get("ebit",              0) or 0)
    interest_expense = float(row.get("interest_expense",  0) or 0)
    total_assets     = float(row.get("total_assets",      0) or 1)
    tax_rate         = float(row.get("tax_rate",        0.21) or 0.21)
    fcf              = float(row.get("fcf",               0) or 0)
    sigma_equity     = float(row.get("price_vol_annual",  0.30) or 0.30)
    sector           = str(row.get("sector", "Unknown") or "Unknown")

    if market_cap <= 0:
        return _empty_mm(market_cap)

    # ── Net debt ──────────────────────────────────────────────────────────── #
    net_debt = max(total_debt - cash, 0)
    ev = market_cap + net_debt

    # ── Leverage ratio (robust) ───────────────────────────────────────────── #
    # Buyback-heavy firms often report negative book equity; dividing by
    # max(equity, 1) then yields leverage_ratio ≈ total_debt in absolute
    # currency units (e.g. 5e10) → astronomical agency costs and a guaranteed
    # spurious SHORT flag.  Fall back to market cap as the equity base and cap
    # the ratio.
    equity_base    = total_equity if total_equity > 0 else market_cap
    leverage_ratio = min(total_debt / max(equity_base, 1), 10.0)

    # ── 1. PV of Tax Shield ───────────────────────────────────────────────── #
    spread          = _credit_spread(leverage_ratio, cfg)
    if interest_expense <= 0 and total_debt > 0:
        # Impute a coupon at the firm's estimated cost of debt (consistent
        # with the shield discount rate → PV ≈ τ·D, the MM perpetuity value)
        interest_expense = total_debt * (rf + spread)

    annual_tax_shield = tax_rate * max(interest_expense, 0.0)
    shield_discount   = rf + spread
    pv_tax_shield     = annual_tax_shield / shield_discount if shield_discount > 0 else 0

    # ── 2. Unlevered value ────────────────────────────────────────────────── #
    VU = market_cap + net_debt - pv_tax_shield

    # ── 3. Financial distress costs (Merton model) ──────────────────────── #
    # Consistent GROSS-debt convention throughout: firm value = E + D_gross,
    # default barrier = D_gross, asset vol de-levered with E/(E + D_gross).
    # (Previously V used net debt while the barrier and de-levering used gross
    # debt — internally inconsistent.)
    E       = max(market_cap, 1)
    D       = max(total_debt, 1)
    V_firm  = E + D

    sigma_assets = sigma_equity * (E / (E + D))   # de-lever equity vol
    mu = rf
    T  = 1.0

    try:
        d2 = (np.log(V_firm / D) + (mu - 0.5 * sigma_assets ** 2) * T) \
             / (sigma_assets * np.sqrt(T))
        # Student-t for fat-tail default probability
        if return_df is not None and return_df > 2:
            prob_default = float(_t_dist.cdf(-d2, df=return_df))
        else:
            prob_default = float(norm.cdf(-d2))
    except Exception as _exc:
        # Fails toward "no distress" — log it so silent zeroing is visible
        logger.debug("Merton d2 failed (%s); prob_default=0.", _exc)
        prob_default = 0.0

    # Industry-specific distress cost rate
    distress_rate = _distress_rate(sector, cfg)
    pv_distress   = prob_default * distress_rate * V_firm

    # ── 4. Agency costs ───────────────────────────────────────────────────── #
    agency_score = 0.0
    if leverage_ratio > 2.0:
        agency_score += (leverage_ratio - 2.0) * 0.05
    # Jensen's free-cash-flow agency cost applies to POSITIVE FCF only —
    # abs() would penalise cash-burning firms for the wrong reason.
    fcf_yield = max(fcf, 0.0) / market_cap if market_cap > 0 else 0
    if fcf_yield > 0.10:
        agency_score += (fcf_yield - 0.10) * 0.5
    pv_agency = agency_score * market_cap

    # ── 5. Levered theoretical value ──────────────────────────────────────── #
    VL = VU + pv_tax_shield - pv_distress - pv_agency

    # ── Divergence: EQUITY level vs EQUITY level ──────────────────────────── #
    # VL is a firm/enterprise value (contains net debt); market_cap is
    # equity-only.  Comparing them directly makes divergence ≈ net_debt/MC:
    # the signal then measures LEVERAGE, not mispricing, and flags the most
    # indebted firms as the strongest LONGs (inverted economics).  Subtract
    # net debt to obtain the implied equity value before comparing.
    VL_equity      = VL - net_debt
    divergence     = (VL_equity - market_cap) / market_cap if market_cap > 0 else 0.0
    divergence_pct = divergence * 100.0

    return {
        "VU":             VU,
        "pv_tax_shield":  pv_tax_shield,
        "pv_distress":    pv_distress,
        "pv_agency":      pv_agency,
        "VL_theoretical": VL,
        "divergence_pct": divergence_pct,
        "prob_default":   prob_default,
        "credit_spread":  spread,
        "distress_rate":  distress_rate,
    }


def _empty_mm(market_cap: float) -> Dict:
    return {
        "VU": 0.0, "pv_tax_shield": 0.0, "pv_distress": 0.0,
        "pv_agency": 0.0, "VL_theoretical": 0.0,
        "divergence_pct": 0.0, "prob_default": 0.0,
        "credit_spread": 0.02, "distress_rate": 0.20,
    }


# --------------------------------------------------------------------------- #
#  Historical volatility helper                                                 #
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
        pv_tax_shield, pv_distress, pv_agency,
        credit_spread, distress_rate
    """
    rf        = cfg.risk_free_rate_annual
    threshold = cfg.leverage_gap_threshold * 100
    return_df = getattr(cfg, "return_df", None)

    df = fundamentals.copy()
    if returns is not None:
        df = add_price_vol(df, returns)

    rows = []
    for ticker, row in df.iterrows():
        result = _mm_valuation(row, rf=rf, return_df=return_df, cfg=cfg)
        result["ticker"] = ticker
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).set_index("ticker")

    # ── Flags ─────────────────────────────────────────────────────────────── #
    result_df["underleveraged"] = result_df["divergence_pct"] >  threshold
    result_df["overleveraged"]  = result_df["divergence_pct"] < -threshold

    # Interest coverage guard — impute the same coupon the valuation uses so a
    # debt-carrying firm with unreported interest can't dodge the check, and
    # flag negative-EBIT firms that carry debt regardless of interest data.
    _imputed_interest = df["interest_expense"].where(
        df["interest_expense"] > 0,
        df["total_debt"].clip(lower=0) * 0.05,
    )
    ic = (df["ebit"] / _imputed_interest.where(_imputed_interest > 0)
          ).reindex(result_df.index).fillna(np.inf)
    result_df["ic_ratio"]    = ic
    result_df["overleveraged"] |= (ic < cfg.min_interest_coverage)
    _neg_ebit_with_debt = (
        (df["ebit"] < 0) & (df["total_debt"] > 0)
    ).reindex(result_df.index).fillna(False)
    result_df["overleveraged"] |= _neg_ebit_with_debt

    n_under = result_df["underleveraged"].sum()
    n_over  = result_df["overleveraged"].sum()
    logger.info(
        "MM screen: %d undervalued (LONG), %d overvalued (SHORT) of %d stocks "
        "[industry distress=%s, variable_spread=%s].",
        n_under, n_over, len(result_df),
        getattr(cfg, "industry_distress_costs", True),
        getattr(cfg, "variable_credit_spread", True),
    )
    return result_df
