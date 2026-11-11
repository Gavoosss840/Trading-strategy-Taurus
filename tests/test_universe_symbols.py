"""
Tests for symbol normalisation and sector normalisation outside the US.

Two defects these cover:

1. Yahoo accepts Hong Kong codes only on four digits: "700.HK" returns 404,
   "0700.HK" returns Tencent's series.  Fifteen of the Hang Seng's thirty-five
   names were written without padding — Tencent, HSBC, CK Hutchison, HKEX and
   China Mobile among them — and vanished from the universe without an error,
   since a failed price download is not fatal.  43% of the index, including its
   largest caps.  Measured after the fix: 34 of 35 names return data, against
   20 before.

2. yfinance publishes its own sector taxonomy, not GICS: "Financial Services",
   "Consumer Cyclical", "Basic Materials".  Non-SEC filers go through yfinance
   only, so none of their labels matched `_SECTOR_DISTRESS_RATE` (flat 20%),
   `_NO_COVERAGE_GUARD` or `_NO_MM_VALUATION` — the financials exemption
   applied to no European or Asian bank at all.
"""

import pandas as pd
import pytest

from taurus.capital_structure import mm_capital_structure_screen
from taurus.data import _match_statement_row
from taurus.config import TaurusConfig, UniverseConfig
from taurus.sectors import normalise_sector, sector_from_sic
from taurus.universe import _FALLBACKS, _UNIVERSE_CONFIGS, UniverseDef, _normalise_symbol

CFG = TaurusConfig()


# --------------------------------------------------------------------------- #
#  Hong Kong: four-digit codes                                                 #
# --------------------------------------------------------------------------- #

HK = _UNIVERSE_CONFIGS["hangseng"]


def test_short_hong_kong_codes_are_padded():
    assert _normalise_symbol("700.HK", HK) == "0700.HK"
    assert _normalise_symbol("5.HK", HK) == "0005.HK"
    assert _normalise_symbol("11.HK", HK) == "0011.HK"
    assert _normalise_symbol("388.HK", HK) == "0388.HK"


def test_codes_already_four_digits_are_untouched():
    assert _normalise_symbol("2318.HK", HK) == "2318.HK"
    assert _normalise_symbol("1398.HK", HK) == "1398.HK"


def test_longer_codes_are_not_truncated():
    assert _normalise_symbol("12345.HK", HK) == "12345.HK"


def test_alphabetic_symbols_are_left_alone():
    """A letter code is not a numeric listing code and must not gain zeros."""
    assert _normalise_symbol("HSBA.HK", HK) == "HSBA.HK"


def test_universes_without_padding_are_untouched():
    for name in ("sp500", "cac40", "ftse100", "nikkei225", "tadawul"):
        config = _UNIVERSE_CONFIGS[name]
        for symbol in _FALLBACKS[name][:5]:
            assert _normalise_symbol(symbol, config) == symbol


def test_padding_is_a_no_op_without_the_suffix():
    bare = UniverseConfig(
        name="x", display_name="X", region="Asia", currency="HKD",
        ff5_dataset="Global_5_Factors", futures_symbol="", futures_exchange="",
        futures_currency="HKD", futures_multiplier=1.0, ibkr_exchange="SEHK",
        wikipedia_url="", wikipedia_table_id="", ticker_pad=4,
    )
    assert _normalise_symbol("700", bare) == "0700"


def test_every_hang_seng_fallback_code_ends_up_on_four_digits():
    """The list itself keeps the usual spelling; loading normalises it."""
    loaded = [_normalise_symbol(t, HK) for t in _FALLBACKS["hangseng"]]
    assert all(len(t.split(".")[0]) == 4 for t in loaded)
    assert "0700.HK" in loaded and "0005.HK" in loaded


def test_the_cache_key_changed_so_a_stale_list_is_not_served():
    """A pre-fix cache holds the same NUMBER of tickers, so the key must differ.

    The key embedded the fallback list's length, which padding leaves
    unchanged: without a version bump, a cache written before the fix would go
    on serving the 404 tickers.
    """
    import inspect
    assert "_v2_" in inspect.getsource(UniverseDef.get_tickers)


# --------------------------------------------------------------------------- #
#  Sector labels                                                               #
# --------------------------------------------------------------------------- #

def test_yfinance_labels_map_to_gics():
    assert normalise_sector("Financial Services") == "Financials"
    assert normalise_sector("Consumer Cyclical") == "Consumer Discretionary"
    assert normalise_sector("Consumer Defensive") == "Consumer Staples"
    assert normalise_sector("Basic Materials") == "Materials"
    assert normalise_sector("Healthcare") == "Health Care"
    assert normalise_sector("Technology") == "Information Technology"


def test_labels_already_in_gics_pass_through():
    """The same function serves both sources; a correct label must not degrade."""
    for sector in ("Financials", "Information Technology", "Health Care",
                   "Consumer Discretionary", "Real Estate", "Utilities"):
        assert normalise_sector(sector) == sector


def test_matching_ignores_case_and_spacing():
    assert normalise_sector("  financial services  ") == "Financials"


def test_an_unknown_label_is_unknown_not_guessed():
    assert normalise_sector("Conglomerates") == "Unknown"
    assert normalise_sector(None) == "Unknown"
    assert normalise_sector("") == "Unknown"


def test_both_sources_agree_on_a_bank():
    """SIC 6022 and yfinance's label must land on the same sector."""
    assert sector_from_sic(6022) == normalise_sector("Financial Services") == "Financials"


# --------------------------------------------------------------------------- #
#  End to end: a non-US bank now reaches the financials exemption              #
# --------------------------------------------------------------------------- #

def bank_row(sector: str) -> dict:
    return dict(
        market_cap=1.0e11, total_debt=5.0e11, cash=2.0e11, total_equity=8.0e10,
        total_assets=2.5e12, ebit=1.2e10, interest_expense=1.4e10,   # IC ≈ 0.86×
        tax_rate=0.21, fcf=1.0e10, sector=sector,
        price_vol_annual=0.25, levered_beta=1.0,
    )


@pytest.mark.parametrize("label", ["Financial Services", "Financials"])
def test_a_european_bank_is_exempt_once_its_label_is_normalised(label):
    row = bank_row(normalise_sector(label))
    out = mm_capital_structure_screen(pd.DataFrame({"BNP.PA": row}).T, cfg=CFG)
    assert not bool(out.loc["BNP.PA", "thin_coverage"])
    assert not bool(out.loc["BNP.PA", "overleveraged"])


def test_the_raw_yfinance_label_would_not_have_been_exempt():
    """Why the fix was needed: the un-normalised label matches nothing."""
    out = mm_capital_structure_screen(
        pd.DataFrame({"BNP.PA": bank_row("Financial Services")}).T, cfg=CFG
    )
    assert bool(out.loc["BNP.PA", "thin_coverage"])


# --------------------------------------------------------------------------- #
#  Statement-row matching                                                      #
# --------------------------------------------------------------------------- #

def statement(rows: dict) -> pd.DataFrame:
    """A yfinance-shaped statement: rows are line items, newest column first."""
    return pd.DataFrame(
        {pd.Timestamp("2026-09-30"): rows, pd.Timestamp("2026-06-30"): rows}
    )


def test_a_negating_label_does_not_answer_for_the_real_one():
    """BNP's case: no operating-income row, so the substring caught a residual."""
    frame = statement({
        "Other Non Operating Income Expenses": -4.1e7,
        "Net Income": 3.456e9,
    })
    assert pd.isna(_match_statement_row(frame, ["EBIT", "Operating Income"]))


def test_the_real_operating_income_row_is_found():
    frame = statement({
        "Operating Income": 1.2e10,
        "Other Non Operating Income Expenses": -4.1e7,
    })
    assert _match_statement_row(frame, ["EBIT", "Operating Income"]) == 1.2e10


def test_an_exact_label_beats_a_partial_match():
    frame = statement({"Total Operating Income As Reported": 9.0e9, "Operating Income": 1.2e10})
    assert _match_statement_row(frame, ["Operating Income"]) == 1.2e10


def test_ebitda_does_not_answer_for_ebit():
    frame = statement({"EBITDA": 2.0e10})
    assert pd.isna(_match_statement_row(frame, ["EBIT"]))


def test_keywords_are_tried_in_order():
    frame = statement({"Total Debt": 5.0e10, "Long Term Debt": 3.0e10})
    assert _match_statement_row(frame, ["Total Debt", "Long Term Debt"]) == 5.0e10


def test_the_most_recent_column_is_used():
    frame = pd.DataFrame({
        pd.Timestamp("2026-09-30"): {"Operating Income": 1.2e10},
        pd.Timestamp("2026-06-30"): {"Operating Income": 1.0e10},
    })
    assert _match_statement_row(frame, ["Operating Income"]) == 1.2e10


def test_an_empty_statement_is_nan_not_an_error():
    assert pd.isna(_match_statement_row(pd.DataFrame(), ["Operating Income"]))
    assert pd.isna(_match_statement_row(None, ["Operating Income"]))


def test_a_missing_line_is_nan_rather_than_an_approximation():
    """A wrong EBIT produces a sell signal; an absent one reads as absent."""
    frame = statement({"Gross Profit": 5.0e9})
    assert pd.isna(_match_statement_row(frame, ["EBIT", "Operating Income"]))
