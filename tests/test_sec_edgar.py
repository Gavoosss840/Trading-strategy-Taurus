"""
Tests for the EDGAR reading layer.

The bug these guard against: the previous reader returned the most recent XBRL
fact for a concept whatever period it covered.  For Coca-Cola that was ONE
quarter of revenue (12.5 bn$) where the financial year is 47.9 bn$, and the MM
screen then capitalised a flow divided by four into a perpetuity — Coca-Cola
came out "82% overvalued" and was flagged SHORT on data-reading grounds alone.
Which companies were hit depended on which XBRL concept happened to be the
freshest, so the distortion was arbitrary from one name to the next.

Measured on 20 US large caps: the fix changes 8 of the 20 verdicts, and takes
the long leg from 0 names to 4 — below the screen's 10% minimum ratio, the
strategy's binary path silently ignored the MM pillar altogether.
"""

import numpy as np

from taurus.sec_edgar import (
    _latest_instant,
    _shares_outstanding,
    _ttm,
    _ttm_from_facts,
)


def fact(start, end, val, filed="2026-02-15", form="10-Q"):
    f = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        f["start"] = start
    return f


def concept(name, facts, unit="USD"):
    return {name: {"units": {unit: facts}}}


# --------------------------------------------------------------------------- #
#  Flows: twelve trailing months                                               #
# --------------------------------------------------------------------------- #

def test_four_quarters_are_summed_not_taken_alone():
    quarters = [
        fact("2025-01-01", "2025-03-31", 10.0),
        fact("2025-04-01", "2025-06-30", 11.0),
        fact("2025-07-01", "2025-09-30", 12.0),
        fact("2025-10-01", "2025-12-31", 13.0),
    ]
    assert _ttm(concept("Revenues", quarters), "Revenues") == 46.0


def test_overlapping_year_to_date_periods_are_not_double_counted():
    """A 10-Q publishes both the quarter and the cumulative period."""
    facts = [
        fact("2025-01-01", "2025-03-31", 10.0),
        fact("2025-04-01", "2025-06-30", 11.0),
        fact("2025-01-01", "2025-06-30", 21.0),   # cumulative — overlaps
        fact("2025-07-01", "2025-09-30", 12.0),
        fact("2025-01-01", "2025-09-30", 33.0),   # cumulative — overlaps
        fact("2025-10-01", "2025-12-31", 13.0),
    ]
    assert _ttm(concept("Revenues", facts), "Revenues") == 46.0


def test_an_annual_fact_is_used_when_quarters_are_missing():
    facts = [fact("2025-01-01", "2025-12-31", 47.9, form="10-K")]
    assert _ttm(concept("Revenues", facts), "Revenues") == 47.9


def test_a_partial_year_is_annualised_as_a_last_resort():
    facts = [fact("2025-01-01", "2025-06-30", 20.0)]
    value = _ttm(concept("Revenues", facts), "Revenues")
    assert 38.0 < value < 42.0


def test_a_lone_quarter_is_not_passed_off_as_a_year():
    """The Coca-Cola case: one quarter must not be capitalised as a year."""
    facts = [fact("2026-01-01", "2026-04-03", 12.5)]
    assert np.isnan(_ttm(concept("Revenues", facts), "Revenues"))


def test_a_restated_period_is_taken_from_the_latest_filing():
    facts = [
        fact("2025-01-01", "2025-12-31", 100.0, filed="2026-02-15", form="10-K"),
        fact("2025-01-01", "2025-12-31",  96.0, filed="2026-08-01", form="10-K"),
    ]
    assert _ttm(concept("Revenues", facts), "Revenues") == 96.0


def test_ttm_from_facts_returns_none_without_usable_periods():
    assert _ttm_from_facts([fact("2025-01-01", "2025-01-20", 1.0)]) is None


# --------------------------------------------------------------------------- #
#  Concept selection: freshness over priority                                  #
# --------------------------------------------------------------------------- #

def test_a_stale_concept_does_not_win_on_priority():
    """Microsoft stopped filling `Revenues` in 2010; J&J in 2014."""
    book = {}
    book.update(concept("RevenueFromContractWithCustomerExcludingAssessedTax",
                        [fact("2025-01-01", "2025-12-31", 331.0, form="10-K")]))
    book.update(concept("Revenues",
                        [fact("2010-01-01", "2010-12-31", 62.5, form="10-K")]))
    value = _ttm(book, "Revenues",
                 "RevenueFromContractWithCustomerExcludingAssessedTax")
    assert value == 331.0


def test_priority_decides_between_equally_fresh_concepts():
    book = {}
    book.update(concept("OperatingIncomeLoss",
                        [fact("2025-01-01", "2025-12-31", 30.0, form="10-K")]))
    book.update(concept("IncomeLossFromContinuingOperations",
                        [fact("2025-01-01", "2025-12-31", 28.0, form="10-K")]))
    value = _ttm(book, "OperatingIncomeLoss", "IncomeLossFromContinuingOperations")
    assert value == 30.0


# --------------------------------------------------------------------------- #
#  Balance sheet: instants                                                     #
# --------------------------------------------------------------------------- #

def test_latest_instant_ignores_duration_facts():
    facts = [
        fact(None, "2025-12-31", 82.3, form="10-K"),
        fact("2025-01-01", "2025-12-31", 999.0, form="10-K"),
    ]
    assert _latest_instant(concept("Assets", facts), "Assets") == 82.3


def test_latest_instant_takes_the_most_recent_close():
    facts = [
        fact(None, "2024-12-31", 70.0, form="10-K"),
        fact(None, "2025-12-31", 82.3, form="10-K"),
    ]
    assert _latest_instant(concept("Assets", facts), "Assets") == 82.3


def test_latest_instant_is_nan_for_an_unknown_concept():
    assert np.isnan(_latest_instant({}, "Assets"))


# --------------------------------------------------------------------------- #
#  Share count                                                                 #
# --------------------------------------------------------------------------- #

def test_share_count_is_read_in_shares_not_dollars():
    """Read in USD, the count came back NaN for every company."""
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {
        "units": {"shares": [fact(None, "2026-01-30", 1.459e10, form="10-K")]}
    }}}}
    assert _shares_outstanding(facts) == 1.459e10


def test_share_count_falls_back_to_the_diluted_average():
    facts = {"facts": {"us-gaap": {"WeightedAverageNumberOfDilutedSharesOutstanding": {
        "units": {"shares": [fact("2025-01-01", "2025-12-31", 7.4e9, form="10-K")]}
    }}}}
    assert _shares_outstanding(facts) == 7.4e9


def test_share_count_is_nan_when_unpublished():
    assert np.isnan(_shares_outstanding({"facts": {}}))
