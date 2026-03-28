"""
Taurus – Financial Modeling Prep (FMP) data layer.

Remplace yfinance pour les fondamentaux (dette, equity, EBIT, etc.)
Mis à jour dès la publication des résultats trimestriels (10-Q/10-K).

Clé API stockée dans .env → FMP_API_KEY (jamais dans le code)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _get_api_key() -> str:
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FMP_API_KEY manquante. Ajoute-la dans ton fichier .env : FMP_API_KEY=ta_clé"
        )
    return key


def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> list | dict:
    """GET avec retry et gestion d'erreurs."""
    key = _get_api_key()
    params = params or {}
    params["apikey"] = key
    url = f"{FMP_BASE}/{endpoint}"

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning("FMP error for %s: %s", endpoint, data["Error Message"])
                return []
            return data
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("FMP request failed (%s), retry in %ds: %s", attempt + 1, wait, e)
            time.sleep(wait)
    return []


# --------------------------------------------------------------------------- #
#  Fondamentaux par ticker                                                     #
# --------------------------------------------------------------------------- #

def get_fundamentals(ticker: str) -> dict:
    """
    Retourne les derniers fondamentaux trimestriels depuis FMP.
    Mis à jour dans l'heure suivant la publication du 10-Q.

    Champs retournés (alignés sur capital_structure.py) :
        market_cap, total_debt, net_debt, total_equity, total_assets,
        ebit, interest_expense, cash, revenue, net_income,
        shares_outstanding, beta, sector
    """
    # Bilan le plus récent (trimestriel)
    bs = _get(f"balance-sheet-statement/{ticker}", {"period": "quarter", "limit": 1})
    # Compte de résultat
    is_ = _get(f"income-statement/{ticker}", {"period": "quarter", "limit": 1})
    # Profil (market cap, beta, secteur)
    profile = _get(f"profile/{ticker}")

    result: dict = {
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
        "beta":               float("nan"),
        "sector":             "Unknown",
    }

    if bs and isinstance(bs, list):
        b = bs[0]
        result["total_debt"]         = float(b.get("totalDebt") or 0)
        result["net_debt"]           = float(b.get("netDebt") or 0)
        result["total_equity"]       = float(b.get("totalStockholdersEquity") or 0)
        result["total_assets"]       = float(b.get("totalAssets") or 0)
        result["cash"]               = float(b.get("cashAndCashEquivalents") or 0)
        result["shares_outstanding"] = float(b.get("commonStock") or 0)

    if is_ and isinstance(is_, list):
        i = is_[0]
        result["ebit"]             = float(i.get("operatingIncome") or 0)
        result["interest_expense"] = abs(float(i.get("interestExpense") or 0))
        result["revenue"]          = float(i.get("revenue") or 0)
        result["net_income"]       = float(i.get("netIncome") or 0)

    if profile and isinstance(profile, list):
        p = profile[0]
        result["market_cap"] = float(p.get("mktCap") or 0)
        result["beta"]       = float(p.get("beta") or 1.0)
        result["sector"]     = p.get("sector") or "Unknown"

    return result


# --------------------------------------------------------------------------- #
#  Earnings calendar (savoir quand relancer le MM screen)                     #
# --------------------------------------------------------------------------- #

def get_earnings_calendar(tickers: list[str]) -> list[dict]:
    """
    Retourne les prochaines dates de publication de résultats.
    Utile pour déclencher le MM screen automatiquement dès la publication.
    """
    data = _get("earning_calendar")
    if not data:
        return []
    ticker_set = set(t.upper() for t in tickers)
    return [
        {"ticker": d["symbol"], "date": d["date"], "eps": d.get("eps"), "eps_est": d.get("epsEstimated")}
        for d in data
        if d.get("symbol", "").upper() in ticker_set
    ]


# --------------------------------------------------------------------------- #
#  Batch fondamentaux (liste de tickers)                                      #
# --------------------------------------------------------------------------- #

def get_fundamentals_batch(tickers: list[str], max_workers: int = 5) -> dict[str, dict]:
    """
    Télécharge les fondamentaux pour une liste de tickers en parallèle.
    Respecte la limite FMP (250 calls/jour en gratuit).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_fundamentals, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                logger.warning("FMP fundamentals failed for %s: %s", ticker, e)
                results[ticker] = {}
            # Log progression toutes les 50 tickers
            if (i + 1) % 50 == 0:
                logger.info("FMP fundamentals: %d/%d tickers done", i + 1, len(tickers))

    return results
