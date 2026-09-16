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
#  Concepts XBRL → valeur courante                                             #
# --------------------------------------------------------------------------- #
# Deux natures de faits cohabitent dans XBRL, et les confondre fausse la
# valorisation d'un facteur quatre :
#
#   • les faits INSTANTANÉS (bilan) n'ont pas de date de début — on prend le
#     plus récent ;
#   • les faits de DURÉE (compte de résultat, flux de trésorerie) couvrent un
#     trimestre, un cumul ou un exercice — il faut les ramener à douze mois
#     glissants avant de les capitaliser à l'infini.
#
# L'ancienne lecture prenait le fait le plus récent quelle que soit sa durée :
# pour Coca-Cola elle renvoyait le chiffre d'affaires d'UN trimestre (12,5 Md$)
# là où l'exercice fait 47,9 Md$.  Le screen MM capitalisait donc un flux
# divisé par quatre et classait le titre « surévalué de 82 % » — un signal de
# vente fabriqué de toutes pièces par la lecture des données.

# Tolérance de fraîcheur entre concepts : deux concepts dont les arrêtés sont
# séparés de moins d'un trimestre décrivent la même situation comptable.
_FRESHNESS_WINDOW_DAYS = 100


def _money_facts(us_gaap: dict, concept: str) -> list[dict]:
    """Faits monétaires d'un concept, dédupliqués sur la période déclarée.

    Une même période est souvent republiée (10-K reprenant un trimestre,
    amendement 10-K/A…).  On conserve le dépôt le plus récent, qui porte les
    chiffres retraités.
    """
    units = (us_gaap.get(concept) or {}).get("units") or {}
    best: dict[tuple, dict] = {}
    for fact in units.get("USD") or []:
        if fact.get("val") is None or not fact.get("end"):
            continue
        key = (fact.get("start"), fact["end"])
        previous = best.get(key)
        if previous is None or str(fact.get("filed", "")) >= str(previous.get("filed", "")):
            best[key] = fact
    return sorted(best.values(), key=lambda f: f["end"], reverse=True)


def _pick_freshest(candidates: list[tuple[int, dict]]) -> Optional[dict]:
    """Choisit un fait parmi des candidats (rang de priorité, fait).

    La fraîcheur prime sur la priorité : les entreprises changent de concept
    XBRL au fil des ans et continuent parfois d'exposer l'ancien, figé sur sa
    dernière valeur.  Johnson & Johnson n'alimente plus `Revenues` depuis 2014,
    Microsoft depuis 2010 ; suivre l'ordre de priorité d'abord renverrait un
    chiffre vieux de dix ans.

    À fraîcheur comparable (moins d'un trimestre d'écart), c'est la priorité
    qui tranche — elle encode la définition comptable la plus pertinente.
    """
    if not candidates:
        return None
    newest = max(str(fact["end"]) for _, fact in candidates)
    fresh = [
        (rank, fact) for rank, fact in candidates
        if (gap := _days_between(str(fact["end"]), newest)) is not None
        and gap <= _FRESHNESS_WINDOW_DAYS
    ]
    if not fresh:
        fresh = candidates
    return min(fresh, key=lambda item: item[0])[1]


def _latest_instant(us_gaap: dict, *concepts: str) -> float:
    """Dernière valeur de bilan (fait instantané : pas de date de début)."""
    candidates: list[tuple[int, dict]] = []
    for rank, concept in enumerate(concepts):
        instants = [f for f in _money_facts(us_gaap, concept) if not f.get("start")]
        if instants:
            candidates.append((rank, instants[0]))

    chosen = _pick_freshest(candidates)
    return float(chosen["val"]) if chosen is not None else float("nan")


def _duration_days(fact: dict) -> Optional[int]:
    return _days_between(str(fact.get("start") or ""), str(fact.get("end") or ""))


def _ttm_from_facts(facts: list[dict]) -> Optional[tuple[float, str]]:
    """Reconstruit un flux sur 12 mois glissants à partir des faits d'un concept.

    Stratégie, du plus fiable au moins fiable :
      1. somme des 4 derniers trimestres consécutifs et disjoints ;
      2. à défaut, dernier exercice annuel complet ;
      3. à défaut, dernière période cumulée (YTD) annualisée.
    """
    quarters = [f for f in facts if (d := _duration_days(f)) and 80 <= d <= 100]
    annuals  = [f for f in facts if (d := _duration_days(f)) and 350 <= d <= 380]

    # 1. Quatre trimestres consécutifs, sans chevauchement.
    if len(quarters) >= 4:
        selected, cursor = [], None
        for fact in quarters:               # déjà triés du plus récent au plus ancien
            if cursor is not None and str(fact["end"]) >= cursor:
                continue                    # chevauche la période déjà retenue
            selected.append(fact)
            cursor = str(fact["start"])
            if len(selected) == 4:
                break
        if len(selected) == 4:
            span = _days_between(str(selected[-1]["start"]), str(selected[0]["end"]))
            if span and 330 <= span <= 400:  # les 4 trimestres couvrent bien un an
                return sum(float(f["val"]) for f in selected), str(selected[0]["end"])

    # 2. Dernier exercice annuel.
    if annuals:
        return float(annuals[0]["val"]), str(annuals[0]["end"])

    # 3. Dernière période cumulée, ramenée à 12 mois.
    ytd = [f for f in facts if (d := _duration_days(f)) and d >= 150]
    if ytd:
        days = _duration_days(ytd[0]) or 365
        return float(ytd[0]["val"]) * 365.0 / days, str(ytd[0]["end"])

    return None


def _ttm(us_gaap: dict, *concepts: str) -> float:
    """Flux sur 12 mois glissants, en privilégiant le concept le plus à jour."""
    candidates: list[tuple[int, dict]] = []
    for rank, concept in enumerate(concepts):
        facts = [f for f in _money_facts(us_gaap, concept) if f.get("start")]
        if not facts:
            continue
        result = _ttm_from_facts(facts)
        if result is not None:
            candidates.append((rank, {"end": result[1], "val": result[0]}))

    chosen = _pick_freshest(candidates)
    return float(chosen["val"]) if chosen is not None else float("nan")


def _shares_outstanding(facts: dict) -> float:
    """Actions en circulation : espace de noms `dei` d'abord, puis us-gaap.

    Le nombre d'actions est publié en unité « shares », pas en dollars : il ne
    passe donc pas par `_money_facts`.  L'ancienne lecture le cherchait en USD
    et renvoyait NaN pour tout le monde.
    """
    for namespace, concept in (
        ("dei",     "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei",     "EntityCommonStockSharesOutstandingBasic"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ):
        units = ((facts.get("facts", {}).get(namespace, {}).get(concept) or {})
                 .get("units") or {})
        entries = [e for e in (units.get("shares") or []) if e.get("val")]
        if entries:
            entries.sort(key=lambda e: str(e.get("end", "")), reverse=True)
            return float(entries[0]["val"])
    return float("nan")


# --------------------------------------------------------------------------- #
#  Croissance historique du chiffre d'affaires                                 #
# --------------------------------------------------------------------------- #
# Le chiffre d'affaires est préféré au résultat : ses marges fluctuent moins
# d'un exercice à l'autre, et une marge exceptionnelle ne se confond pas avec
# une trajectoire de croissance.
_REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

GROWTH_WINDOW_YEARS = 8
MIN_GROWTH_YEARS    = 4


def _days_between(earlier: str, later: str) -> Optional[int]:
    from datetime import date

    try:
        y1, m1, d1 = (int(part) for part in str(earlier).split("-"))
        y2, m2, d2 = (int(part) for part in str(later).split("-"))
        return (date(y2, m2, d2) - date(y1, m1, d1)).days
    except (ValueError, TypeError):
        return None


def _annual_series(book: dict, concepts: tuple[str, ...]) -> dict[str, float]:
    """Valeurs annuelles d'un agrégat, par année de clôture.

    Seules les périodes d'environ un exercice sont retenues ; pour une même
    année le dépôt le plus récent l'emporte, afin de prendre les chiffres
    retraités plutôt que la première publication.

    Toutes les années sont lues dans la même devise — celle qui couvre le plus
    d'exercices — pour qu'un changement d'unité ne se lise pas comme de la
    croissance.
    """
    units: dict[str, list] = {}
    for concept in concepts:
        for unit, entries in ((book.get(concept) or {}).get("units") or {}).items():
            units.setdefault(unit, []).extend(entries or [])
    if not units:
        return {}
    currency = max(units, key=lambda u: len(units[u]))

    best: dict[str, tuple[float, str]] = {}
    for fact in units[currency]:
        if fact.get("val") is None or not fact.get("start") or not fact.get("end"):
            continue
        span = _days_between(str(fact["start"]), str(fact["end"]))
        if span is None or not (350 <= span <= 380):
            continue
        year  = str(fact["end"])[:4]
        filed = str(fact.get("filed", ""))
        if year not in best or filed >= best[year][1]:
            best[year] = (float(fact["val"]), filed)
    return {year: value for year, (value, _) in best.items()}


def revenue_cagr(us_gaap: dict) -> float:
    """Croissance annuelle composée du chiffre d'affaires sur 8 exercices.

    Les extrémités sont lissées sur deux exercices : un exercice de départ
    déprimé ou un exercice d'arrivée exceptionnel décalerait le taux de
    plusieurs points, et ce taux pilote la première étape de la valorisation.

    Renvoie NaN si l'historique est trop court — l'appelant retombe alors sur
    la croissance par défaut, jamais sur une estimation de deux points.
    """
    annual = _annual_series(us_gaap, _REVENUE_CONCEPTS)
    if len(annual) < MIN_GROWTH_YEARS:
        return float("nan")

    years  = sorted(annual)[-GROWTH_WINDOW_YEARS:]
    values = [annual[y] for y in years]
    if any(v <= 0 for v in values):
        return float("nan")

    start = sum(values[:2]) / 2.0
    end   = sum(values[-2:]) / 2.0
    span  = (int(years[-1]) + int(years[-2])) / 2.0 - (int(years[0]) + int(years[1])) / 2.0
    if span <= 0 or start <= 0:
        return float("nan")

    return float((end / start) ** (1.0 / span) - 1.0)


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
        shares_outstanding, revenue_cagr, sector
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
        "revenue_cagr":       float("nan"),
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

    # ── Bilan : faits instantanés ─────────────────────────────────────────── #
    result["total_debt"] = _latest_instant(
        us_gaap,
        "LongTermDebtAndCapitalLeaseObligation",
        "LongTermDebt",
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
    )
    result["total_equity"] = _latest_instant(
        us_gaap,
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )
    result["total_assets"] = _latest_instant(
        us_gaap, "Assets",
    )
    result["cash"] = _latest_instant(
        us_gaap,
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    )

    # ── Flux : ramenés à douze mois glissants ─────────────────────────────── #
    result["ebit"] = _ttm(
        us_gaap,
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    )
    result["interest_expense"] = abs(_ttm(
        us_gaap,
        "InterestExpense",
        "InterestAndDebtExpense",
    ))
    result["revenue"] = _ttm(
        us_gaap,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    )
    result["net_income"] = _ttm(
        us_gaap,
        "NetIncomeLoss",
        "ProfitLoss",
    )
    # Free Cash Flow = Operating Cash Flow − CapEx (proper FCF, not net income proxy)
    op_cf = _ttm(
        us_gaap,
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )
    capex = abs(_ttm(
        us_gaap,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresContinuingOperations",
        "PaymentsForCapitalImprovements",
    ))
    import math
    if not math.isnan(op_cf) and not math.isnan(capex):
        result["free_cash_flow"] = op_cf - capex
    elif not math.isnan(op_cf):
        result["free_cash_flow"] = op_cf          # best proxy when capex unavailable
    else:
        result["free_cash_flow"] = result["net_income"]  # last resort

    result["shares_outstanding"] = _shares_outstanding(facts)
    result["revenue_cagr"] = revenue_cagr(us_gaap)

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
