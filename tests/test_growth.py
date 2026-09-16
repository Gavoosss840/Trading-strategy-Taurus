"""
Tests for the two-stage growth valuation and its data feed.

The bias this guards against: with a single perpetual growth rate applied to
every company, the MM screen shorted 100% of the names growing faster than 8%
a year and went long the declining telecoms — growth and divergence correlated
at −0.49 on 20 US large caps.  The screen was betting against growth, which
nobody chose.

`initial_growth` only fixes that if `revenue_cagr` actually reaches it: without
the extraction tested here, every company falls back on
`default_initial_growth` and the bias survives the rewrite untouched.
"""

import numpy as np
import pandas as pd

from taurus.capital_structure import (
    _mm_valuation,
    _two_stage_value,
    initial_growth,
)
from taurus.config import TaurusConfig
from taurus.data import _revenue_cagr_yf
from taurus.sec_edgar import _annual_series, revenue_cagr

CFG = TaurusConfig()


# --------------------------------------------------------------------------- #
#  EDGAR: annual revenue series                                                #
# --------------------------------------------------------------------------- #

def book(values: dict[int, float], *, unit: str = "USD", concept: str = "Revenues") -> dict:
    """An XBRL companyfacts book carrying one full-year fact per year."""
    return {
        concept: {
            "units": {
                unit: [
                    {
                        "start": f"{year}-01-01",
                        "end":   f"{year}-12-31",
                        "val":   value,
                        "filed": f"{year + 1}-02-15",
                        "form":  "10-K",
                    }
                    for year, value in values.items()
                ]
            }
        }
    }


def test_annual_series_keeps_only_full_year_periods():
    facts = book({2020: 100.0, 2021: 110.0})
    facts["Revenues"]["units"]["USD"].append({
        "start": "2022-01-01", "end": "2022-03-31", "val": 30.0,
        "filed": "2022-04-20", "form": "10-Q",
    })
    assert set(_annual_series(facts, ("Revenues",))) == {"2020", "2021"}


def test_annual_series_prefers_the_latest_filing_for_a_year():
    facts = book({2020: 100.0})
    facts["Revenues"]["units"]["USD"].append({
        "start": "2020-01-01", "end": "2020-12-31", "val": 96.0,
        "filed": "2022-02-15", "form": "10-K",   # restated, filed later
    })
    assert _annual_series(facts, ("Revenues",))["2020"] == 96.0


def test_annual_series_does_not_mix_currencies():
    """A reporting-currency switch would read as a 20% jump in revenue."""
    facts = book({y: 100.0 + y for y in range(2016, 2024)})
    facts["Revenues"]["units"]["EUR"] = [{
        "start": "2023-01-01", "end": "2023-12-31", "val": 5.0,
        "filed": "2024-02-15", "form": "10-K",
    }]
    series = _annual_series(facts, ("Revenues",))
    assert 5.0 not in series.values()


# --------------------------------------------------------------------------- #
#  EDGAR: revenue CAGR                                                         #
# --------------------------------------------------------------------------- #

def test_revenue_cagr_recovers_a_known_rate():
    values = {2016 + i: 100.0 * 1.12 ** i for i in range(8)}
    assert np.isclose(revenue_cagr(book(values)), 0.12, atol=1e-6)


def test_revenue_cagr_smooths_an_exceptional_final_year():
    """One blowout year must not set the rate for the next decade."""
    steady = {2016 + i: 100.0 * 1.05 ** i for i in range(8)}
    spiked = dict(steady)
    spiked[2023] = steady[2023] * 1.60
    assert revenue_cagr(book(spiked)) < 0.05 + (0.60 / 7)


def test_revenue_cagr_is_nan_without_enough_history():
    assert np.isnan(revenue_cagr(book({2022: 100.0, 2023: 110.0})))


def test_revenue_cagr_is_nan_on_a_negative_endpoint():
    values = {2016 + i: 100.0 for i in range(7)}
    values[2023] = -5.0
    assert np.isnan(revenue_cagr(book(values)))


def test_revenue_cagr_is_negative_for_a_shrinking_business():
    values = {2016 + i: 100.0 * 0.95 ** i for i in range(8)}
    assert revenue_cagr(book(values)) < 0.0


# --------------------------------------------------------------------------- #
#  yfinance counterpart (CAC, Nikkei, Hang Seng)                               #
# --------------------------------------------------------------------------- #

class _FakeTicker:
    def __init__(self, frame):
        self.income_stmt = frame


def _income_stmt(values: dict[int, float], label: str = "Total Revenue") -> pd.DataFrame:
    # yfinance returns the most recent year first.
    years = sorted(values, reverse=True)
    return pd.DataFrame(
        [[values[y] for y in years]],
        index=[label],
        columns=[pd.Timestamp(f"{y}-12-31") for y in years],
    )


def test_yf_revenue_cagr_recovers_a_known_rate():
    frame = _income_stmt({2016 + i: 100.0 * 1.08 ** i for i in range(8)})
    assert np.isclose(_revenue_cagr_yf(_FakeTicker(frame)), 0.08, atol=1e-6)


def test_yf_revenue_cagr_is_nan_without_enough_history():
    frame = _income_stmt({2022: 100.0, 2023: 110.0})
    assert np.isnan(_revenue_cagr_yf(_FakeTicker(frame)))


def test_yf_revenue_cagr_is_nan_without_a_revenue_row():
    frame = _income_stmt({2016 + i: 100.0 for i in range(8)}, label="Gross Profit")
    assert np.isnan(_revenue_cagr_yf(_FakeTicker(frame)))


def test_yf_revenue_cagr_survives_a_broken_provider():
    class _Broken:
        @property
        def income_stmt(self):
            raise RuntimeError("yfinance down")

    assert np.isnan(_revenue_cagr_yf(_Broken()))


def test_both_sources_agree_on_the_same_trajectory():
    values = {2016 + i: 100.0 * 1.10 ** i for i in range(8)}
    edgar = revenue_cagr(book(values))
    yahoo = _revenue_cagr_yf(_FakeTicker(_income_stmt(values)))
    assert np.isclose(edgar, yahoo, atol=1e-9)


# --------------------------------------------------------------------------- #
#  initial_growth                                                              #
# --------------------------------------------------------------------------- #

def test_initial_growth_uses_the_observed_rate():
    assert np.isclose(initial_growth(pd.Series({"revenue_cagr": 0.09}), CFG), 0.09)


def test_initial_growth_falls_back_when_unobserved():
    assert initial_growth(pd.Series({}), CFG) == CFG.default_initial_growth
    assert initial_growth(pd.Series({"revenue_cagr": np.nan}), CFG) == CFG.default_initial_growth


def test_initial_growth_is_bounded_both_ways():
    assert initial_growth(pd.Series({"revenue_cagr": 0.58}), CFG) == CFG.max_initial_growth
    assert initial_growth(pd.Series({"revenue_cagr": -0.40}), CFG) == CFG.min_initial_growth


def test_initial_growth_is_not_capped_by_the_discount_rate():
    """The g < r constraint binds the terminal perpetuity, not stage 1."""
    fast = initial_growth(pd.Series({"revenue_cagr": 0.15}), CFG)
    assert fast > CFG.risk_free_rate_annual + CFG.equity_risk_premium * 0.8


# --------------------------------------------------------------------------- #
#  Two-stage value                                                             #
# --------------------------------------------------------------------------- #

def test_two_stage_matches_a_perpetuity_when_growth_never_fades():
    g, r = 0.025, 0.09
    staged = _two_stage_value(100.0, r, g, g, CFG.explicit_growth_years)
    assert np.isclose(staged, 100.0 * (1.0 + g) / (r - g), rtol=1e-9)


def test_two_stage_value_increases_with_initial_growth():
    slow = _two_stage_value(100.0, 0.09, 0.02, 0.025, 10)
    fast = _two_stage_value(100.0, 0.09, 0.14, 0.025, 10)
    assert fast > slow * 1.5


def test_two_stage_value_is_nan_when_the_terminal_rate_exceeds_the_discount():
    assert np.isnan(_two_stage_value(100.0, 0.03, 0.05, 0.04, 10))


def test_stage_one_growth_above_the_discount_rate_stays_finite():
    value = _two_stage_value(100.0, 0.08, 0.15, 0.025, 10)
    assert np.isfinite(value) and value > 0.0


# --------------------------------------------------------------------------- #
#  End to end: the growth bias                                                 #
# --------------------------------------------------------------------------- #

def firm(**overrides) -> pd.Series:
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
        sector="Technology",
        price_vol_annual=0.28,
        levered_beta=1.1,
    )
    base.update(overrides)
    return pd.Series(base)


def test_a_fast_grower_is_valued_above_an_identical_slow_one():
    slow = _mm_valuation(firm(revenue_cagr=0.01), cfg=CFG)
    fast = _mm_valuation(firm(revenue_cagr=0.15), cfg=CFG)
    assert fast["VL_theoretical"] > slow["VL_theoretical"]
    assert fast["divergence_pct"] > slow["divergence_pct"]


def test_growth_and_divergence_now_move_together():
    """The sign of the old −0.49 correlation, reversed."""
    rates = [-0.05, 0.0, 0.05, 0.10, 0.15]
    divergences = [
        _mm_valuation(firm(revenue_cagr=g), cfg=CFG)["divergence_pct"] for g in rates
    ]
    assert np.corrcoef(rates, divergences)[0, 1] > 0.95


def test_a_missing_cagr_leaves_the_valuation_computable():
    result = _mm_valuation(firm(), cfg=CFG)
    assert np.isfinite(result["VL_theoretical"])
    assert result["growth_start"] == CFG.default_initial_growth
