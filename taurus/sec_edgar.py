"""
Taurus – SEC EDGAR fundamentals layer.

Source primaire officielle : les entreprises déposent leurs 10-Q/10-K
directement sur EDGAR. Données disponibles en <15 min après publication.

API gratuite, aucune clé requise.
Docs : https://www.sec.gov/developer
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

EDGAR_BASE  = "https://data.sec.gov"
SEC_BASE    = "https://www.sec.gov"
HEADERS     = {"User-Agent": "TaurusStrategy taurus-algo@protonmail.com"}


# --------------------------------------------------------------------------- #
#  CIK lookup (ticker → identifiant SEC)                                      #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_cik_map() -> dict[str, str]:
    """Charge le mapping ticker → CIK depuis SEC (mis en cache en mémoire)."""
    try:
        r = requests.get(
            f"{SEC_BASE}/files/company_tickers.json",
            headers=HEADERS, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
    except Exception as e:
        logger.error("Impossible de charger le mapping CIK: %s", e)
        return {}


def ticker_to_cik(ticker: str) -> Optional[str]:
    cik_map = _load_cik_map()
    return cik_map.get(ticker.upper())


# --------------------------------------------------------------------------- #
#  Concepts XBRL → valeur la plus récente                                     #
# --------------------------------------------------------------------------- #

def _latest_value(us_gaap: dict, *concepts: str) -> float:
    """
    Essaie plusieurs noms de concepts GAAP (les entreprises n'utilisent
    pas toutes le même nom) et retourne la valeur trimestrielle la plus récente.
    """
    for concept in concepts:
        data = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
        quarterly = [
            d for d in data
            if d.get("form") in ("10-Q", "10-K") and d.get("val") is not None
        ]
        if quarterly:
            quarterly.sort(key=lambda x: x["end"], reverse=True)
            return float(quarterly[0]["val"])
    return float("nan")


# --------------------------------------------------------------------------- #
#  Fondamentaux par ticker                                                     #
# --------------------------------------------------------------------------- #

def get_fundamentals(ticker: str) -> dict:
    """
    Retourne les derniers fondamentaux depuis SEC EDGAR XBRL.
    Mis à jour en <15 min après publication du 10-Q/10-K.

    Champs alignés sur capital_structure.py :
        market_cap, total_debt, net_debt, total_equity, total_assets,
        ebit, interest_expense, cash, revenue, net_income,
        shares_outstanding, sector
    """
    result = {
        "market_cap":         float("nan"),
        "total_debt":         float("nan"),
        "net_debt":           float("nan"),
        "total_equity":       float("nan"),
        "total_assets":       float("nan"),
        "ebit":               float("nan"),
        "interest_expense":   float("nan"),
        "cash":               float("nan"),
        "revenue":            float("nan"),
        "net_income":         float("nan"),
        "shares_outstanding": float("nan"),
        "sector":             "Unknown",
    }

    cik = ticker_to_cik(ticker)
    if not cik:
        logger.warning("CIK introuvable pour %s", ticker)
        return result

    try:
        r = requests.get(
            f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
            headers=HEADERS, timeout=20,
        )
        r.raise_for_status()
        facts   = r.json()
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
    except Exception as e:
        logger.warning("EDGAR XBRL failed pour %s: %s", ticker, e)
        return result

    result["total_debt"] = _latest_value(
        us_gaap,
        "LongTermDebtAndCapitalLeaseObligation",
        "LongTermDebt",
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
    )
    result["total_equity"] = _latest_value(
        us_gaap,
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )
    result["total_assets"] = _latest_value(
        us_gaap, "Assets",
    )
    result["cash"] = _latest_value(
        us_gaap,
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    )
    result["ebit"] = _latest_value(
        us_gaap,
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    )
    result["interest_expense"] = abs(_latest_value(
        us_gaap,
        "InterestExpense",
        "InterestAndDebtExpense",
    ))
    result["revenue"] = _latest_value(
        us_gaap,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    )
    result["net_income"] = _latest_value(
        us_gaap,
        "NetIncomeLoss",
        "ProfitLoss",
    )
    result["shares_outstanding"] = _latest_value(
        us_gaap,
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    )

    # Net debt = total_debt - cash
    if result["total_debt"] == result["total_debt"] and result["cash"] == result["cash"]:
        result["net_debt"] = result["total_debt"] - result["cash"]

    return result


# --------------------------------------------------------------------------- #
#  Batch fondamentaux                                                          #
# --------------------------------------------------------------------------- #

def get_fundamentals_batch(
    tickers: list[str],
    max_workers: int = 5,
) -> dict[str, dict]:
    """
    Télécharge les fondamentaux pour une liste de tickers en parallèle.
    SEC EDGAR est gratuit et sans limite de calls.
    """
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_fundamentals, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                logger.warning("EDGAR failed pour %s: %s", ticker, e)
                results[ticker] = {}
            if (i + 1) % 50 == 0:
                logger.info("EDGAR: %d/%d tickers done", i + 1, len(tickers))
            # Petite pause pour ne pas surcharger SEC EDGAR
            time.sleep(0.05)

    return results
