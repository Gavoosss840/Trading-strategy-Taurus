"""
Tests for sector neutralisation and the financials exemption.

Two defects these cover:

1. `sector` came back "Unknown" for every company, which silently disabled four
   mechanisms at once — the MM sector distress rate, the 30% sector cap in
   `portfolio.py` (one sector observed ⇒ cap relaxed to 100%), sector-neutral
   scoring, and the financials exemption below.

2. The interest-coverage guard flagged every bank SHORT.  Interest is a bank's
   raw material, not a financing charge: JPMorgan covers its interest 0.9× and
   was flagged over-levered every month while the valuation put it 47%
   undervalued — so it entered BOTH legs at once.
"""

import numpy as np
import pandas as pd

from taurus.capital_structure import mm_capital_structure_screen
from taurus.config import TaurusConfig
from taurus.sectors import sector_from_sic
from taurus.strategy import _zscore, _zscore_by_sector

CFG = TaurusConfig()


# --------------------------------------------------------------------------- #
#  SIC → sector                                                                #
# --------------------------------------------------------------------------- #

def test_sic_maps_the_sectors_the_engine_relies_on():
    assert sector_from_sic(6022) == "Financials"          # banque commerciale
    assert sector_from_sic(2834) == "Health Care"         # pharmacie
    assert sector_from_sic(3674) == "Information Technology"
    assert sector_from_sic(4813) == "Communication Services"
    assert sector_from_sic(6798) == "Real Estate"         # REIT, dans la plage 6600-6999
    assert sector_from_sic(4911) == "Utilities"


def test_sic_is_unknown_rather_than_wrong_when_absent():
    assert sector_from_sic(None) == "Unknown"
    assert sector_from_sic("") == "Unknown"
    assert sector_from_sic("abc") == "Unknown"


# --------------------------------------------------------------------------- #
#  Sector-neutral z-score                                                      #
# --------------------------------------------------------------------------- #

def _universe(tech_values, telecom_values):
    values, sectors = {}, {}
    for i, v in enumerate(tech_values):
        values[f"TECH{i}"] = v
        sectors[f"TECH{i}"] = "Information Technology"
    for i, v in enumerate(telecom_values):
        values[f"TEL{i}"] = v
        sectors[f"TEL{i}"] = "Communication Services"
    return pd.Series(values, dtype=float), sectors


def test_a_whole_sector_no_longer_sits_at_one_end_of_the_ranking():
    """The bias being removed: every tech name cheap-looking, every telecom rich."""
    values, sectors = _universe([-70, -60, -50, -40], [10, 20, 30, 40])

    absolute = _zscore(values)
    assert absolute[["TECH0", "TECH1", "TECH2", "TECH3"]].max() < 0
    assert absolute[["TEL0", "TEL1", "TEL2", "TEL3"]].min() > 0

    neutral = _zscore_by_sector(values, sectors)
    assert neutral[["TECH0", "TECH1", "TECH2", "TECH3"]].max() > 0
    assert neutral[["TEL0", "TEL1", "TEL2", "TEL3"]].min() < 0


def test_the_ranking_inside_a_sector_is_preserved():
    values, sectors = _universe([-70, -60, -50, -40], [10, 20, 30, 40])
    neutral = _zscore_by_sector(values, sectors)
    assert neutral["TECH3"] > neutral["TECH2"] > neutral["TECH1"] > neutral["TECH0"]


def test_each_sector_is_centred_on_itself():
    values, sectors = _universe([-70, -60, -50, -40], [10, 20, 30, 40])
    neutral = _zscore_by_sector(values, sectors)
    tech = neutral[["TECH0", "TECH1", "TECH2", "TECH3"]]
    tel = neutral[["TEL0", "TEL1", "TEL2", "TEL3"]]
    assert abs(tech.median()) < 1e-9 and abs(tel.median()) < 1e-9


def test_a_thin_sector_is_scored_against_the_universe():
    """A median over one or two stocks is noise, and would hand them z = 0."""
    values, sectors = _universe([-70, -60, -50, -40], [10, 20, 30, 40])
    values["LONE"] = 200.0
    sectors["LONE"] = "Utilities"
    neutral = _zscore_by_sector(values, sectors, min_members=4)
    assert neutral["LONE"] != 0.0


def test_unknown_sectors_are_pooled_not_treated_as_a_sector():
    values = pd.Series({f"S{i}": float(i) for i in range(8)})
    sectors = {t: "Unknown" for t in values.index}
    neutral = _zscore_by_sector(values, sectors)
    assert np.allclose(neutral.values, _zscore(values).values)


def test_empty_input_is_returned_unchanged():
    assert _zscore_by_sector(pd.Series(dtype=float), {}).empty


# --------------------------------------------------------------------------- #
#  Financials exemption and leg exclusivity                                    #
# --------------------------------------------------------------------------- #

def bank(**overrides) -> dict:
    """A bank: interest paid exceeds operating income, which is normal."""
    base = dict(
        market_cap=6.0e11,
        total_debt=4.0e11,
        cash=1.5e12,
        total_equity=3.5e11,
        total_assets=4.0e12,
        ebit=7.0e10,
        interest_expense=7.9e10,      # IC ≈ 0.9×
        tax_rate=0.21,
        fcf=6.0e10,
        sector="Financials",
        price_vol_annual=0.25,
        levered_beta=1.0,
    )
    base.update(overrides)
    return base


def screen(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    return mm_capital_structure_screen(df, cfg=CFG)


def test_a_bank_is_not_flagged_over_levered_on_interest_coverage():
    out = screen({"JPM": bank()})
    assert out.loc["JPM", "ic_ratio"] < CFG.min_interest_coverage
    assert not bool(out.loc["JPM", "thin_coverage"])


def test_the_same_balance_sheet_outside_financials_is_still_flagged():
    out = screen({"ACME": bank(sector="Industrials")})
    assert bool(out.loc["ACME", "thin_coverage"])
    assert bool(out.loc["ACME", "overleveraged"])


def test_a_reit_keeps_the_guard_its_debt_is_real_leverage():
    out = screen({"REIT": bank(sector="Real Estate")})
    assert bool(out.loc["REIT", "thin_coverage"])


def test_the_screen_abstains_from_valuing_a_bank():
    """APV has no unlevered firm to value when leverage IS the business."""
    out = screen({"JPM": bank()})
    assert np.isnan(out.loc["JPM", "divergence_pct"])
    assert not bool(out.loc["JPM", "underleveraged"])
    assert not bool(out.loc["JPM", "overleveraged"])


def test_abstention_can_be_turned_off():
    cfg = TaurusConfig()
    cfg.mm_skip_financials = False
    out = mm_capital_structure_screen(pd.DataFrame({"JPM": bank()}).T, cfg=cfg)
    assert np.isfinite(out.loc["JPM", "divergence_pct"])


def test_a_non_financial_is_still_valued():
    out = screen({"ACME": bank(sector="Industrials")})
    assert np.isfinite(out.loc["ACME", "divergence_pct"])


def test_a_loss_making_bank_is_still_flagged():
    """The exemption covers thin coverage, not insolvency."""
    out = screen({"JPM": bank(ebit=-1.0e10)})
    assert bool(out.loc["JPM", "overleveraged"])


def test_a_stock_is_never_a_candidate_for_both_legs():
    out = screen({
        "JPM":  bank(),
        "ACME": bank(sector="Industrials"),
        "GOOD": bank(sector="Industrials", interest_expense=1.0e9, market_cap=2.0e11),
    })
    assert not (out["underleveraged"] & out["overleveraged"]).any()


def test_solvency_wins_over_a_cheap_valuation():
    """Cheap is worth nothing if the company cannot service its debt."""
    out = screen({"ACME": bank(sector="Industrials", market_cap=5.0e10)})
    assert bool(out.loc["ACME", "overleveraged"])
    assert not bool(out.loc["ACME", "underleveraged"])


def test_a_missing_sector_column_does_not_break_the_screen():
    row = bank(sector="Industrials")
    row.pop("sector")
    df = pd.DataFrame({"ACME": row}).T
    out = mm_capital_structure_screen(df, cfg=CFG)
    assert bool(out.loc["ACME", "thin_coverage"])
