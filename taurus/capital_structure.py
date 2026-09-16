"""
Taurus – Modigliani-Miller capital structure screen (APV formulation).

MM Valuation Engine
────────────────────
VL (levered firm value) = VU + PV(Tax Shield) - PV(Distress) - Agency Costs

Where VU, the UNLEVERED firm value, is discounted from fundamentals:

  NOPAT = EBIT × (1 - τ)                        after-tax operating profit
  β_U   = β_L / (1 + (1 - τ) · D/E)             Hamada un-levering
  r_U   = rf + β_U × equity risk premium        CAPM without financial risk
  VU    = NOPAT × (1 + g) / (r_U - g)           growing perpetuity

  Tax Shield     = τ × Interest / (rf + credit_spread)
  Distress Costs = Merton-model P(default) × distress_rate × EV
  Agency Costs   = f(leverage, FCF yield)

Why VU is no longer backed out of market cap
─────────────────────────────────────────────
This module previously computed

    VU = (Market Cap + Net Debt) - Tax Shield
    VL = VU + Tax Shield - Distress - Agency

The tax shield cancels between the two lines, leaving

    VL_equity  = Market Cap - Distress - Agency
    divergence = -(Distress + Agency) / Market Cap        ≤ 0 ALWAYS

The "theoretical" value was therefore *defined* from the very price it was
meant to judge, and the signal could never flag a stock as undervalued.
Verified numerically on four profiles: divergence ran from -0.001% (low-debt
tech) to -53.6% (distressed firm), never positive.

Downstream consequence in strategy.py::_binary_signal — `under_tickers` was
always empty, so `mm_underval_ratio` was always 0, the `>= MM_MIN_RATIO`
branch never fired, and the long leg fell through to the alpha-quantile
fallback on every single rebalance.  The MM screen only ever contributed on
the short side, as a bankruptcy-risk filter rather than a valuation measure.

The APV formulation above restores a genuine, price-independent fair value.
The three frictions (tax shield, Merton distress, agency) are unchanged.

A residual coupling to price remains and is intended: Hamada un-levering uses
the MARKET value of equity in D/E, as is standard (book equity is distorted by
buybacks, often negative).  A higher price lowers D/E, raises β_U and r_U, and
lowers VU — an order of magnitude weaker than before (a tripled price moves VU
by <10%, versus 100% previously) and in the stabilising direction.

The Merton default probability uses Student-t (ν = 5) rather than the Normal:
fat-tailed equity returns make the Normal badly understate tail default risk.

Signal:
  divergence = (VL_equity - Market Cap) / Market Cap
  > +threshold  → undervalued  → underleveraged flag  → LONG candidate
  < -threshold  → overvalued   → overleveraged flag   → SHORT candidate
  NaN           → not valuable (non-positive EBIT, or no market cap):
                  neither flag, and the composite treats it as no opinion.
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


# Secteurs exemptés du garde-fou de couverture des intérêts : ceux dont les
# intérêts versés sont un coût d'exploitation et non une charge de financement.
_NO_COVERAGE_GUARD = frozenset({"Financials"})

# Secteurs que l'écran MM ne prétend pas valoriser : ceux dont le levier est
# l'activité elle-même, et non un choix de structure du capital.
_NO_MM_VALUATION = frozenset({"Financials"})


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
#  Unlevered firm value (APV)                                                  #
# --------------------------------------------------------------------------- #

def debt_beta(spread: float, cfg: TaurusConfig = DEFAULT_CONFIG) -> float:
    """Systematic risk borne by the debt itself, inferred from its spread.

    Hamada's textbook form assumes RISKLESS debt (β_D = 0).  For a heavily
    indebted firm that assumption drives β_U far too low, hence r_U too low and
    the perpetuity too high: the model would reward leverage, the very
    inversion the equity-level divergence already guards against.  A firm at
    D/E = 1.5 with β_L = 0.95 un-levers to β_U = 0.44 under β_D = 0 — an asset
    beta below a utility's, for a levered cyclical.

    Only part of a credit spread compensates systematic risk; the rest is
    expected default loss and illiquidity.  Taking half is the usual
    approximation (Cooper & Davydenko, 2007), capped at 0.4 — beyond that the
    claim behaves like equity and the capital-structure split loses meaning.
    """
    if cfg.equity_risk_premium <= 0:
        return 0.0
    return float(np.clip(0.5 * spread / cfg.equity_risk_premium, 0.0, 0.4))


def unlever_beta(
    levered_beta: float,
    total_debt: float,
    equity_value: float,
    tax_rate: float,
    beta_debt: float = 0.0,
) -> float:
    """Un-lever an equity beta into an asset beta.

        β_U = (E·β_L + D(1-τ)·β_D) / (E + D(1-τ))

    which collapses to Hamada's β_L / (1 + (1-τ)·D/E) when β_D = 0.

    The beta estimated by the FF5 regression is an EQUITY beta: it embeds the
    financial risk created by debt.  Discounting an operating flow at that rate
    would count leverage twice — once in the discount rate, once in the net
    debt subtracted at the end.

    D/E uses the MARKET value of equity, as is standard: book equity is
    distorted by buybacks and frequently negative.
    """
    if not np.isfinite(levered_beta) or equity_value <= 0:
        return float("nan")
    debt_value = max(total_debt, 0.0) * (1.0 - tax_rate)
    return float(
        (equity_value * levered_beta + debt_value * beta_debt)
        / (equity_value + debt_value)
    )


def unlevered_cost_of_capital(
    beta_unlevered: float,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> float:
    """CAPM without financial risk:  r_U = rf + β_U × equity risk premium."""
    beta = beta_unlevered if np.isfinite(beta_unlevered) else cfg.default_unlevered_beta
    # A zero or negative beta would discount below the risk-free rate and blow
    # up the perpetuity.
    beta = max(beta, 0.2)
    return float(cfg.risk_free_rate_annual + beta * cfg.equity_risk_premium)


def initial_growth(row: pd.Series, cfg: TaurusConfig = DEFAULT_CONFIG) -> float:
    """The company's own starting growth rate, bounded.

    Estimated from past revenue growth: margins move less than earnings, so an
    exceptional margin is not mistaken for a growth trajectory.

    Capped by an economic maximum — nothing grows at 20% for a decade — but
    NOT by the discount rate. The g < r constraint binds only the terminal
    perpetuity, which would otherwise diverge; over a finite stage, growth
    above the discount rate is the normal case for an expanding company.
    """
    observed = row.get("revenue_cagr", float("nan"))
    try:
        observed = float(observed)
    except (TypeError, ValueError):
        observed = float("nan")
    if not np.isfinite(observed):
        observed = cfg.default_initial_growth
    return float(np.clip(observed, cfg.min_initial_growth, cfg.max_initial_growth))


def _two_stage_value(
    nopat: float,
    discount: float,
    growth_start: float,
    growth_terminal: float,
    years: int,
) -> float:
    """Present value of a flow whose growth fades to its terminal rate.

    Stage 1: `years` periods whose growth declines linearly from
    `growth_start` to `growth_terminal`.  Stage 2: perpetuity at the terminal
    rate.

    A single-rate perpetuity undervalues every company growing faster than
    that rate, and the screen then shorts precisely the fastest growers — the
    bias this replaces.  The linear fade also avoids the step change of
    switching regimes at once, and matches the observed erosion of competitive
    advantage.
    """
    if not np.isfinite(nopat) or discount <= growth_terminal or years < 1:
        return float("nan")

    # Only the TERMINAL rate must stay below the discount rate: stage-1 flows
    # are finite in number, so their growth may legitimately exceed it.
    flow = nopat
    present_value = 0.0
    for year in range(1, years + 1):
        weight = (year - 1) / max(years - 1, 1)
        growth = growth_start + (growth_terminal - growth_start) * weight
        flow *= 1.0 + growth
        present_value += flow / (1.0 + discount) ** year

    terminal = flow * (1.0 + growth_terminal) / (discount - growth_terminal)
    present_value += terminal / (1.0 + discount) ** years
    return float(present_value)


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
                price_vol_annual (optional), levered_beta (optional)
    rf        : annual risk-free rate
    return_df : Student-t degrees of freedom for Merton (None → Normal)
    cfg       : TaurusConfig for sector distress + spread settings

    Returns
    -------
    dict with: VU, nopat, unlevered_beta, discount_rate, growth_rate,
               pv_tax_shield, pv_distress, pv_agency,
               VL_theoretical, divergence_pct, prob_default

    divergence_pct is NaN when the firm cannot be valued (non-positive EBIT,
    or no market cap) — NOT 0.0, which would read as "fairly valued".
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
    levered_beta     = float(row.get("levered_beta", np.nan))

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

    # ── 2. Unlevered firm value — from FUNDAMENTALS, not from market cap ── #
    # This is the fix.  The previous line was
    #     VU = market_cap + net_debt - pv_tax_shield
    # which made VL_equity collapse to market_cap - distress - agency, so the
    # divergence could never be positive (see the module docstring).
    # NaN must be caught explicitly: `NaN <= 0` is False, so a missing EBIT
    # would otherwise slip through and only be stopped at the perpetuity.
    if not np.isfinite(ebit) or ebit <= 0:
        logger.debug(
            "EBIT missing or non-positive (%s) — no perpetuity, no MM valuation.",
            ebit,
        )
        return _empty_mm(market_cap)

    nopat = ebit * (1.0 - tax_rate)

    # `spread` was computed in step 1 from this firm's leverage.
    beta_unlevered = unlever_beta(
        levered_beta, total_debt, market_cap, tax_rate,
        beta_debt=debt_beta(spread, cfg),
    )
    discount_rate  = unlevered_cost_of_capital(beta_unlevered, cfg)

    # The terminal rate must stay clear of r_U or the perpetuity diverges.
    growth_rate = min(cfg.terminal_growth, discount_rate - cfg.min_discount_spread)
    growth_start = initial_growth(row, cfg)

    VU = _two_stage_value(
        nopat, discount_rate, growth_start, growth_rate, cfg.explicit_growth_years,
    )
    if not np.isfinite(VU) or VU <= 0:
        logger.debug(
            "Valuation not computable (r_U=%.4f, g1=%.4f, g=%.4f).",
            discount_rate, growth_start, growth_rate,
        )
        return _empty_mm(market_cap)

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
        "nopat":          nopat,
        "unlevered_beta": beta_unlevered if np.isfinite(beta_unlevered)
                          else cfg.default_unlevered_beta,
        "discount_rate":  discount_rate,
        "growth_rate":    growth_rate,
        "growth_start":   growth_start,
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
    """Row for a firm that cannot be valued.

    divergence_pct is NaN, never 0.0: a stock we could not value must not read
    as "fairly valued".  NaN keeps it out of both flags (NaN comparisons are
    False) and lets the composite treat the MM pillar as having no opinion.
    """
    return {
        "VU": np.nan, "nopat": np.nan, "unlevered_beta": np.nan,
        "discount_rate": np.nan, "growth_rate": np.nan, "growth_start": np.nan,
        "pv_tax_shield": np.nan, "pv_distress": np.nan,
        "pv_agency": np.nan, "VL_theoretical": np.nan,
        "divergence_pct": np.nan, "prob_default": np.nan,
        "credit_spread": np.nan, "distress_rate": np.nan,
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
    betas: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Apply the full MM valuation screen.

    Parameters
    ----------
    fundamentals : DataFrame indexed by ticker (from data.get_fundamentals)
    cfg          : TaurusConfig
    returns      : optional monthly returns DataFrame to compute price vol
    betas        : optional Series of market betas indexed by ticker, used to
                   un-lever the discount rate.  Pass alpha_df["beta_mkt"] from
                   the FF5 regression: same stock, same window, so the discount
                   rate is consistent with the alpha computed alongside it.
                   Missing betas fall back to cfg.default_unlevered_beta.

    Returns
    -------
    DataFrame indexed by ticker with columns:
        VL_theoretical, divergence_pct, prob_default,
        underleveraged (bool), overleveraged (bool),
        VU, nopat, unlevered_beta, discount_rate, growth_rate, growth_start,
        pv_tax_shield, pv_distress, pv_agency,
        credit_spread, distress_rate, ic_ratio, thin_coverage

    `underleveraged` and `overleveraged` are mutually exclusive: the solvency
    guards override a cheap valuation.
    """
    rf        = cfg.risk_free_rate_annual
    threshold = cfg.leverage_gap_threshold * 100
    return_df = getattr(cfg, "return_df", None)

    df = fundamentals.copy()
    if returns is not None:
        df = add_price_vol(df, returns)
    if betas is not None:
        # Assign rather than join: a join would raise on an overlap if the
        # fundamentals frame already carries the column.  Tickers absent from
        # `betas` get NaN and fall back to cfg.default_unlevered_beta.
        df["levered_beta"] = betas.reindex(df.index)

    rows = []
    for ticker, row in df.iterrows():
        result = _mm_valuation(row, rf=rf, return_df=return_df, cfg=cfg)
        result["ticker"] = ticker
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).set_index("ticker")

    # ── Flags ─────────────────────────────────────────────────────────────── #
    # NaN divergence (unvaluable firm) compares False both ways, so such a
    # stock is neither a long nor a short candidate on valuation grounds.  The
    # two solvency guards below still apply to it: a loss-making, indebted firm
    # is a short candidate regardless of whether a perpetuity can be computed.
    # A bank has no unlevered firm to value.  APV separates operating assets
    # from a financing choice, but for a financial institution leverage IS the
    # business: deposits are raw material, not a capital-structure decision,
    # and "EBIT" is not an operating flow that can be capitalised.  The screen
    # therefore abstains rather than producing an uninterpretable number —
    # exactly as it does for a company with non-positive EBIT.  Financials
    # remain tradeable through the alpha and momentum pillars.
    if getattr(cfg, "mm_skip_financials", True) and "sector" in df.columns:
        _no_valuation = (
            df["sector"].reindex(result_df.index).fillna("Unknown").isin(_NO_MM_VALUATION)
        )
        if _no_valuation.any():
            result_df.loc[_no_valuation, "divergence_pct"] = np.nan
            logger.info(
                "MM screen: %d financials left unvalued (no unlevered firm to value).",
                int(_no_valuation.sum()),
            )

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
    result_df["ic_ratio"] = ic

    # …except for financials, where interest is not a financing charge but the
    # cost of the raw material.  A bank funds itself with deposits and lends
    # the proceeds: paying more interest than its operating income is the
    # normal shape of the business, not a solvency warning.  JPMorgan covers
    # its interest 0.9× and was flagged SHORT every month on that basis, while
    # the valuation put it 47% undervalued — it ended up in BOTH legs at once.
    # Real Estate keeps the guard: a REIT's debt is genuine leverage.
    _financial = (
        df["sector"].reindex(result_df.index).fillna("Unknown").isin(_NO_COVERAGE_GUARD)
        if "sector" in df.columns
        else pd.Series(False, index=result_df.index)
    )
    result_df["thin_coverage"] = (ic < cfg.min_interest_coverage) & ~_financial
    result_df["overleveraged"] |= result_df["thin_coverage"]
    _neg_ebit_with_debt = (
        (df["ebit"] < 0) & (df["total_debt"] > 0)
    ).reindex(result_df.index).fillna(False)
    result_df["overleveraged"] |= _neg_ebit_with_debt

    # A stock must never be a candidate for both legs.  The solvency guards
    # win: being cheap is worth nothing if the company cannot service its debt
    # through the holding period.
    result_df["underleveraged"] &= ~result_df["overleveraged"]

    n_under  = int(result_df["underleveraged"].sum())
    n_over   = int(result_df["overleveraged"].sum())
    n_novalue = int(result_df["divergence_pct"].isna().sum())
    logger.info(
        "MM screen: %d undervalued (LONG), %d overvalued (SHORT), %d not "
        "valuable, of %d stocks [industry distress=%s, variable_spread=%s].",
        n_under, n_over, n_novalue, len(result_df),
        getattr(cfg, "industry_distress_costs", True),
        getattr(cfg, "variable_credit_spread", True),
    )
    if n_novalue:
        logger.debug(
            "%d stocks carry no MM valuation (non-positive EBIT or missing "
            "market cap); their divergence is NaN, not 0.", n_novalue,
        )
    return result_df
