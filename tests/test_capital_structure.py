"""
Tests for the MM capital-structure screen.

The centrepiece is `test_undervaluation_is_now_reachable`: before the APV fix,
the divergence reduced algebraically to -(distress + agency)/market_cap and was
therefore always ≤ 0, so the LONG flag could never fire and strategy.py's
binary signal silently fell through to its alpha-quantile fallback on every
rebalance.
"""

import numpy as np
import pandas as pd
import pytest

from taurus.capital_structure import (
    _SECTOR_DISTRESS_RATE,
    _credit_spread,
    _distress_rate,
    _mm_valuation,
    debt_beta,
    mm_capital_structure_screen,
    unlever_beta,
    unlevered_cost_of_capital,
)
from taurus.config import TaurusConfig

CFG = TaurusConfig()
RF = CFG.risk_free_rate_annual


def firm(**overrides) -> pd.Series:
    """A profitable, moderately geared company; overrides shape each variant."""
    base = dict(
        market_cap=1.0e11,
        total_debt=2.0e10,
        cash=1.0e10,
        total_equity=5.0e10,
        total_assets=1.2e11,
        ebit=1.2e10,
        interest_expense=8.0e8,
        tax_rate=0.21,
        fcf=9.0e9,
        sector="Industrials",
        price_vol_annual=0.25,
        levered_beta=1.0,
    )
    base.update(overrides)
    return pd.Series(base)


def value(row: pd.Series, cfg: TaurusConfig = CFG) -> dict:
    return _mm_valuation(row, rf=RF, return_df=cfg.return_df, cfg=cfg)


# --------------------------------------------------------------------------- #
#  The regression this fix addresses                                           #
# --------------------------------------------------------------------------- #

def test_undervaluation_is_now_reachable():
    """The LONG flag must be attainable at all.

    Under the old formulation no set of fundamentals could produce a positive
    divergence, so `underleveraged` was dead code.
    """
    result = value(firm(market_cap=4.0e10))
    assert result["divergence_pct"] > CFG.leverage_gap_threshold * 100


def test_overvaluation_is_still_reachable():
    result = value(firm(market_cap=1.0e12))
    assert result["divergence_pct"] < -CFG.leverage_gap_threshold * 100


def test_divergence_is_not_minus_frictions_over_market_cap():
    """Guard against a regression to the circular formulation.

    The old model always satisfied divergence == -(distress + agency)/mcap.
    """
    row = firm()
    result = value(row)
    old_formula = -(
        (result["pv_distress"] + result["pv_agency"]) / row["market_cap"] * 100
    )
    assert abs(result["divergence_pct"] - old_formula) > 1.0


def test_unlevered_value_does_not_track_market_cap():
    """VU must come from fundamentals, not from the price it is judging."""
    cheap = value(firm(market_cap=1.0e11))
    rich = value(firm(market_cap=3.0e11))

    # A residual coupling remains by design: Hamada un-levering uses market D/E.
    drift = abs(rich["VU"] / cheap["VU"] - 1.0)
    assert drift < 0.10, f"residual coupling too strong: {drift:.1%}"

    # The comparison to price, however, must swing across zero.
    assert cheap["divergence_pct"] > 0 > rich["divergence_pct"]


# --------------------------------------------------------------------------- #
#  Un-levering                                                                 #
# --------------------------------------------------------------------------- #

def test_unlever_beta_reduces_beta_when_debt_present():
    assert unlever_beta(1.2, 5.0e10, 1.0e11, 0.21) < 1.2


def test_unlever_beta_matches_hamada_when_debt_is_riskless():
    got = unlever_beta(1.2, 5.0e10, 1.0e11, 0.21, beta_debt=0.0)
    assert got == pytest.approx(1.2 / (1 + 0.79 * 0.5), rel=1e-9)


def test_debt_beta_raises_the_asset_beta_of_a_levered_firm():
    """Assuming riskless debt drives β_U far too low for a geared company."""
    hamada = unlever_beta(0.95, 3.0e10, 2.0e10, 0.21, beta_debt=0.0)
    with_debt_beta = unlever_beta(0.95, 3.0e10, 2.0e10, 0.21, beta_debt=0.4)
    assert with_debt_beta > hamada


def test_debt_beta_grows_with_the_spread_and_is_capped():
    assert debt_beta(0.01, CFG) < debt_beta(0.05, CFG)
    assert debt_beta(1.0, CFG) <= 0.4


def test_unlever_beta_is_identity_without_debt():
    assert unlever_beta(1.1, 0.0, 1.0e11, 0.21) == pytest.approx(1.1)


def test_unlever_beta_undefined_without_equity():
    assert np.isnan(unlever_beta(1.1, 1.0e10, 0.0, 0.21))


def test_cost_of_capital_falls_back_to_default_beta():
    expected = RF + CFG.default_unlevered_beta * CFG.equity_risk_premium
    assert unlevered_cost_of_capital(float("nan"), CFG) == pytest.approx(expected)


def test_cost_of_capital_stays_above_the_risk_free_rate():
    assert unlevered_cost_of_capital(-0.5, CFG) > RF


def test_missing_beta_uses_the_configured_default():
    row = firm()
    del row["levered_beta"]
    result = value(row)
    assert result["unlevered_beta"] == pytest.approx(CFG.default_unlevered_beta)


# --------------------------------------------------------------------------- #
#  Firms that cannot be valued                                                 #
# --------------------------------------------------------------------------- #

def test_negative_ebit_yields_nan_not_zero():
    """NaN, never 0.0 — a stock we could not value is not "fairly valued"."""
    result = value(firm(ebit=-1.0e9))
    assert np.isnan(result["divergence_pct"])


def test_missing_ebit_yields_nan():
    """`NaN <= 0` is False, so a missing EBIT needs its own guard."""
    result = value(firm(ebit=float("nan")))
    assert np.isnan(result["divergence_pct"])


def test_zero_market_cap_yields_nan():
    assert np.isnan(value(firm(market_cap=0.0))["divergence_pct"])


def test_unvaluable_firm_carries_neither_valuation_flag():
    frame = pd.DataFrame([firm(ebit=-1.0e9)], index=["LOSS"])
    screen = mm_capital_structure_screen(frame, CFG)
    assert not screen.loc["LOSS", "underleveraged"]


def test_loss_making_indebted_firm_is_still_flagged_short():
    """The solvency guards must outlive the valuation."""
    frame = pd.DataFrame([firm(ebit=-1.0e9, total_debt=3.0e10)], index=["LOSS"])
    screen = mm_capital_structure_screen(frame, CFG)
    assert screen.loc["LOSS", "overleveraged"]


def test_thin_interest_coverage_is_flagged_short():
    frame = pd.DataFrame(
        [firm(ebit=1.0e9, interest_expense=1.0e9)], index=["THIN"],
    )
    screen = mm_capital_structure_screen(frame, CFG)
    assert screen.loc["THIN", "ic_ratio"] < CFG.min_interest_coverage
    assert screen.loc["THIN", "overleveraged"]


# --------------------------------------------------------------------------- #
#  Frictions, unchanged from the original screen                               #
# --------------------------------------------------------------------------- #

def test_credit_spread_increases_with_leverage_and_is_capped():
    cfg = TaurusConfig(variable_credit_spread=True)
    assert _credit_spread(0.2, cfg) < _credit_spread(1.0, cfg) < _credit_spread(3.0, cfg)
    assert _credit_spread(50.0, cfg) <= 0.10


def test_distress_rate_is_sector_specific():
    cfg = TaurusConfig(industry_distress_costs=True)
    assert _distress_rate("Information Technology", cfg) > _distress_rate("Utilities", cfg)
    assert _distress_rate("nonexistent", cfg) == _SECTOR_DISTRESS_RATE["Unknown"]


def test_distress_costs_grow_with_volatility():
    calm = value(firm(price_vol_annual=0.15))
    wild = value(firm(price_vol_annual=0.90))
    assert wild["prob_default"] > calm["prob_default"]
    assert wild["pv_distress"] > calm["pv_distress"]


def test_interest_is_imputed_when_unreported():
    result = value(firm(interest_expense=0.0))
    assert result["pv_tax_shield"] > 0


def test_negative_book_equity_does_not_explode_leverage():
    """Buyback-heavy firms report negative book equity."""
    frame = pd.DataFrame([firm(total_equity=-5.0e9)], index=["BUYBACK"])
    screen = mm_capital_structure_screen(frame, CFG)
    assert np.isfinite(screen.loc["BUYBACK", "divergence_pct"])


# --------------------------------------------------------------------------- #
#  Cross-sectional screen                                                      #
# --------------------------------------------------------------------------- #

def build_universe() -> pd.DataFrame:
    """A small universe spanning cheap, fair and expensive names."""
    return pd.DataFrame(
        [
            firm(market_cap=4.0e10),                       # cheap
            firm(market_cap=1.6e11),                       # around fair
            firm(market_cap=8.0e11),                       # expensive
            firm(market_cap=1.0e11, ebit=-2.0e9),          # unvaluable
        ],
        index=["CHEAP", "FAIR", "RICH", "LOSS"],
    )


def test_screen_flags_both_directions():
    screen = mm_capital_structure_screen(build_universe(), CFG)
    assert screen.loc["CHEAP", "underleveraged"]
    assert screen.loc["RICH", "overleveraged"]


def test_screen_produces_a_workable_long_ratio():
    """strategy.py needs mm_underval_ratio ≥ 0.10 to use the MM branch.

    It was structurally 0 before the fix.
    """
    screen = mm_capital_structure_screen(build_universe(), CFG)
    ratio = screen["underleveraged"].sum() / len(screen)
    assert ratio >= 0.10


def test_screen_accepts_betas_and_uses_them():
    universe = build_universe()
    low = mm_capital_structure_screen(
        universe, CFG, betas=pd.Series(0.5, index=universe.index),
    )
    high = mm_capital_structure_screen(
        universe, CFG, betas=pd.Series(1.8, index=universe.index),
    )
    # A riskier stock is discounted harder, so it is worth less.
    assert high.loc["FAIR", "discount_rate"] > low.loc["FAIR", "discount_rate"]
    assert high.loc["FAIR", "divergence_pct"] < low.loc["FAIR", "divergence_pct"]


def test_screen_tolerates_missing_betas():
    universe = build_universe()
    partial = pd.Series({"CHEAP": 1.2})       # only one ticker covered
    screen = mm_capital_structure_screen(universe, CFG, betas=partial)
    assert np.isfinite(screen.loc["FAIR", "divergence_pct"])


def test_screen_is_empty_for_an_empty_universe():
    assert mm_capital_structure_screen(pd.DataFrame(), CFG).empty
