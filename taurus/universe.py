"""
Taurus – Multi-universe registry.

Defines investable universes:
  US      : S&P 500, NASDAQ 100
  Europe  : CAC 40, FTSE 100
  Asia    : Nikkei 225 (Japan), Hang Seng (China/HK)
  Middle East: TADAWUL (Saudi Arabia)

Each universe knows its own:
  - tickers (Wikipedia scrape + fallback)
  - Fama-French factor dataset (US / Europe / Japan / Asia-Pac / Global)
  - IBKR futures contract for beta hedging
  - Exchange and currency for order routing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import TaurusConfig, UniverseConfig, DEFAULT_CONFIG
from .data import (
    _cache_load, _cache_save,
    get_fundamentals,
    _download_ff5_directly,
    get_umd_factor,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Market metadata per Yahoo ticker suffix                                     #
# --------------------------------------------------------------------------- #
#
# Every IBKR contract in this codebase used to be built as
#
#     Stock(symbol, universe_cfg.ibkr_exchange, universe_cfg.currency)
#
# i.e. with the suffix stripped off the Yahoo ticker and ONE currency for the
# whole universe.  Two things break on that:
#
#   1. The symbol alone is ambiguous.  Stock("SAN", "SMART", "EUR") matches both
#      Sanofi (Paris) and Banco Santander (Madrid) — both EUR.  stoxx_eur holds
#      SAN.PA and SAN.MC, and UNI.MI and UNI.MC, so four names resolve to two
#      contracts and two of them would be traded as the wrong company.
#   2. A universe spanning several quote currencies cannot be expressed at all.
#
# primaryExchange is what disambiguates a listing, so it is derived here from
# the suffix, which is static and reliable — unlike Yahoo's `.info` currency,
# which needs a network call and distinguishes pence from pounds by case alone.
#
# The minor-unit factor is how many quote units make one major currency unit:
# the LSE quotes in GBp, so 100 pence per pound.
#
# CAUTION: the IBKR exchange codes below come from IBKR's documented list and
# have NOT been verified against a live TWS session from here.  Qualify them
# against the account before the first real order on a new market.
SUFFIX_MARKET: Dict[str, Tuple[str, float, str]] = {
    ".L":  ("GBP", 100.0, "LSE"),        # London — quoted in GBp (pence)
    ".PA": ("EUR",   1.0, "SBF"),        # Euronext Paris
    ".AS": ("EUR",   1.0, "AEB"),        # Euronext Amsterdam
    ".BR": ("EUR",   1.0, "ENEXT.BE"),   # Euronext Brussels
    ".LS": ("EUR",   1.0, "BVL"),        # Euronext Lisbon
    ".IR": ("EUR",   1.0, "ISE"),        # Euronext Dublin
    ".DE": ("EUR",   1.0, "IBIS"),       # Xetra
    ".MI": ("EUR",   1.0, "BVME"),       # Borsa Italiana
    ".MC": ("EUR",   1.0, "BM"),         # BME Madrid
    ".HE": ("EUR",   1.0, "HEX"),        # Nasdaq Helsinki
    ".VI": ("EUR",   1.0, "VSE"),        # Wiener Börse
    ".AT": ("EUR",   1.0, "ATH"),        # Athens
    ".T":  ("JPY",   1.0, "TSEJ"),       # Tokyo
}


def ticker_market(
    ticker: str,
    universe_cfg=None,
) -> Tuple[str, str, float, str]:
    """
    Resolve a Yahoo ticker to its IBKR contract identity.

    Returns (ibkr_symbol, quote_currency, minor_units_per_major, primary_exchange).

    A ticker with no suffix, or an unknown one, falls back to the universe's own
    currency and an empty primaryExchange — i.e. exactly the previous behaviour,
    so US universes are unaffected.
    """
    if "." in ticker:
        sym, _, suf = ticker.partition(".")
        suffix = "." + suf
    else:
        sym, suffix = ticker, ""

    spec = SUFFIX_MARKET.get(suffix)
    if spec is None:
        ccy = getattr(universe_cfg, "currency", "USD") if universe_cfg else "USD"
        minor = float(getattr(universe_cfg, "nav_scale", 1.0) or 1.0) if universe_cfg else 1.0
        if suffix:
            logger.debug("No market metadata for suffix %r (%s) — using %s.",
                         suffix, ticker, ccy)
        return sym, ccy, minor, ""

    ccy, minor, primary = spec
    return sym, ccy, minor, primary


def _download_ff5_region(dataset_name: str, start: str, end: str):
    """
    Generic downloader for any Kenneth French 5-factor monthly dataset.

    Kenneth French files are whitespace-delimited (not CSV despite the extension).
    We use sep=r'\\s+' and scan for the first data block automatically.

    dataset_name examples:
      "Europe_5_Factors"
      "Japan_5_Factors"
      "Asia_Pacific_ex_Japan_5_Factors"
      "Global_5_Factors"
    """
    import io, zipfile, requests

    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        f"{dataset_name}_CSV.zip"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        zf       = zipfile.ZipFile(io.BytesIO(resp.content))
        fname    = [n for n in zf.namelist() if n.upper().endswith(".CSV")][0]
        raw_text = zf.read(fname).decode("latin-1")

        # Kenneth French regional files are comma-separated.
        # Format: "199007    ,4.46    ,0.20   ,-1.52    ,0.28    ,1.10    ,0.68"
        # Find the header line containing "Mkt-RF" and read from there.
        lines = raw_text.splitlines()
        header_idx = next(
            (i for i, l in enumerate(lines) if "Mkt-RF" in l),
            None,
        )
        if header_idx is None:
            logger.error("%s: could not find header line with 'Mkt-RF'", dataset_name)
            return None

        # Skip everything before the header; use comma separator
        raw = pd.read_csv(
            io.StringIO(raw_text),
            skiprows=header_idx,
            sep=",",
            engine="python",
        )

        # Column 0 is the date field (may have trailing spaces: "199007    ")
        # Rename it and strip whitespace
        raw = raw.rename(columns={raw.columns[0]: "Date"})
        raw["Date"] = raw["Date"].astype(str).str.strip()

        # Keep only rows where Date is a 6-digit YYYYMM integer
        mask = raw["Date"].str.match(r"^\d{6}$")
        raw  = raw[mask].copy()

        # Normalise column names (strip any extra whitespace from header)
        raw.columns = [c.strip() for c in raw.columns]
        # Keep exactly the 7 expected columns
        raw = raw[["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()

        raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str.strip(), format="%Y%m")
        raw = raw.set_index("Date")
        raw.index = raw.index.to_period("M").to_timestamp("M")
        raw = raw.apply(pd.to_numeric, errors="coerce") / 100.0
        raw = raw.dropna(how="all")
        raw = raw.loc[start:end]

        logger.info(
            "%s FF5 factors downloaded from Kenneth French (%d months).",
            dataset_name, len(raw),
        )
        return raw
    except Exception as e:
        logger.error("FF5 download failed for %s: %s", dataset_name, e)
        return None


# Backward-compat alias
def _download_ff5_europe(start: str, end: str):
    return _download_ff5_region("Europe_5_Factors", start, end)


# --------------------------------------------------------------------------- #
#  Universe definitions                                                        #
# --------------------------------------------------------------------------- #

_UNIVERSE_CONFIGS: Dict[str, UniverseConfig] = {

    "sp500": UniverseConfig(
        name="sp500",
        display_name="S&P 500",
        region="US",
        currency="USD",
        ff5_dataset="F-F_Research_Data_5_Factors_2x3",
        futures_symbol="ES",
        futures_exchange="CME",
        futures_currency="USD",
        futures_multiplier=50.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        wikipedia_table_id="constituents",
        min_market_cap_usd=2e9,
        fractional_shares=False,            # IBKR API rejects fractional quantities (Error 10243)
        n_longs=35,
        n_shorts=35,
    ),

    "nasdaq100": UniverseConfig(
        name="nasdaq100",
        display_name="NASDAQ 100",
        region="US",
        currency="USD",
        ff5_dataset="F-F_Research_Data_5_Factors_2x3",
        futures_symbol="NQ",
        futures_exchange="CME",
        futures_currency="USD",
        futures_multiplier=20.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        wikipedia_table_id="constituents",
        min_market_cap_usd=5e9,
        fractional_shares=False,            # IBKR API rejects fractional quantities (Error 10243)
        n_longs=20,
        n_shorts=20,
    ),

    "cac40": UniverseConfig(
        name="cac40",
        display_name="CAC 40",
        region="Europe",
        currency="EUR",
        ff5_dataset="Europe_5_Factors",
        futures_symbol="CAC40",
        futures_exchange="MONEP",
        futures_currency="EUR",
        futures_multiplier=10.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/CAC_40",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        supports_trail_stop=False,          # Euronext rejects TRAIL on MKT parent (Error 328)
        n_longs=10,
        n_shorts=10,
    ),

    "ftse100": UniverseConfig(
        name="ftse100",
        display_name="FTSE 100",
        region="Europe",
        currency="GBP",
        ff5_dataset="Europe_5_Factors",
        futures_symbol="Z",
        futures_exchange="LIFFE",
        futures_currency="GBP",
        futures_multiplier=10.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/FTSE_100",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        nav_scale=100.0,                    # LSE prices in GBp (pence): NAV(GBP)*100 = GBp units
        supports_trail_stop=False,          # LSE: MKT transmits alone, STP sent as standalone order
        n_longs=15,
        n_shorts=15,
    ),

    # ── STOXX Europe 600, split by quote currency ─────────────────────────── #
    # Replaces cac40 + ftse100, which earn less than cash over 10 years:
    # CAC 40 +3.25%/yr (Sharpe -0.44) and FTSE 100 +3.46%/yr (Sharpe -0.23)
    # against a 4.5% risk-free rate, measured on 29 and 35 names.  Breadth is
    # the fix: 289 and 127 names give the cross-section something to rank.
    #
    # Deliberately TWO universes rather than one pan-European book, because
    # UniverseConfig carries a single `currency` and a single `nav_scale`, both
    # used to fetch the NAV, convert it and size the orders.  A mixed EUR+GBP
    # universe would make both wrong for part of its names — the same class of
    # bug that sized CPG.L and IHG.L 74x too large.  Split, each universe stays
    # mono-currency and the Sharpe allocator decides the split between them.

    "stoxx_eur": UniverseConfig(
        name="stoxx_eur",
        display_name="STOXX Europe 600 (Eurozone)",
        region="Europe",
        currency="EUR",
        ff5_dataset="Europe_5_Factors",
        futures_symbol="ESTX50",             # EURO STOXX 50 futures on EUREX
        futures_exchange="EUREX",
        futures_currency="EUR",
        futures_multiplier=10.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/STOXX_Europe_600",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        supports_trail_stop=False,           # Euronext rejects TRAIL on MKT parent
        n_longs=35,
        n_shorts=35,
    ),

    "stoxx_uk": UniverseConfig(
        name="stoxx_uk",
        display_name="STOXX Europe 600 (UK)",
        region="Europe",
        currency="GBP",
        ff5_dataset="Europe_5_Factors",
        futures_symbol="Z",                  # FTSE 100 futures
        futures_exchange="LIFFE",
        futures_currency="GBP",
        futures_multiplier=10.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/STOXX_Europe_600",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        nav_scale=100.0,                     # LSE quotes in GBp (pence)
        supports_trail_stop=False,           # LSE: STP sent as a standalone order
        n_longs=30,
        n_shorts=30,
    ),

    # One pan-European book instead of the two above, EUR and London together.
    #
    # On the full combined book, Sharpe-weighted across every universe, it is
    # worth +0.159 of Sharpe and +2.16pp of annual return over running
    # stoxx_eur and stoxx_uk side by side (1.96 against 1.80), with a slightly
    # worse drawdown (-6.78% against -6.20%).  Far less than the per-universe
    # figures suggest (2.19 against 1.98 and 1.09), because the two split legs
    # already draw 47.9% of NAV between them while a single universe is capped
    # at max_universe_weight = 35%.
    #
    # multi_currency means `currency` is the BASE currency: the NAV is read in
    # EUR, every price is converted to EUR per line (get_prices for the signal,
    # build_price_scale for the orders), and nav_scale stays 1.0 because the
    # NAV is no longer scaled to a quote unit.
    #
    # NOT a drop-in replacement for the two split universes: it exercises the
    # multi-currency order path, which could not be tested against a live TWS
    # from the build environment.  Dry-run it and check the share counts before
    # trading it, and do not run it at the same time as stoxx_eur / stoxx_uk —
    # it holds the same names and the books would fight over them.
    "stoxx_europe": UniverseConfig(
        name="stoxx_europe",
        display_name="STOXX Europe 600 (EUR + UK)",
        region="Europe",
        currency="EUR",                      # base currency for the NAV
        multi_currency=True,
        nav_scale=1.0,                       # required: prices are converted, not the NAV
        ff5_dataset="Europe_5_Factors",
        futures_symbol="ESTX50",
        futures_exchange="EUREX",
        futures_currency="EUR",
        futures_multiplier=10.0,
        ibkr_exchange="SMART",
        wikipedia_url="https://en.wikipedia.org/wiki/STOXX_Europe_600",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        supports_trail_stop=False,
        n_longs=35,
        n_shorts=35,
    ),

    # ── Asia ──────────────────────────────────────────────────────────────── #

    "nikkei225": UniverseConfig(
        name="nikkei225",
        display_name="Nikkei 225",
        region="Japan",
        currency="JPY",
        ff5_dataset="Japan_5_Factors",
        futures_symbol="NK225",           # Nikkei 225 futures on OSE
        futures_exchange="OSE.JPN",
        futures_currency="JPY",
        futures_multiplier=1000.0,
        ibkr_exchange="SMART",            # SMART routes to TSEJ; avoids Error 10311 direct-routing block
        wikipedia_url="https://en.wikipedia.org/wiki/Nikkei_225",
        wikipedia_table_id="constituents",
        ticker_suffix=".T",               # Yahoo Finance: 7203.T
        ticker_col_index=1,               # Code column in Wikipedia table
        use_wikipedia_scrape=False,       # Wikipedia page has no reliable constituents table; use fallback
        min_market_cap_usd=1e9,
        min_lot_size=100,                 # TSE requires orders in multiples of 100 shares
        n_longs=30,
        n_shorts=30,
    ),


}


# --------------------------------------------------------------------------- #
#  Fallback ticker lists                                                       #
# --------------------------------------------------------------------------- #

_FALLBACKS: Dict[str, List[str]] = {

    "sp500": [
        # Top ~100 S&P 500 constituents by market cap (Yahoo Finance tickers)
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
        "BRK-B", "AVGO", "JPM", "LLY", "V", "UNH", "XOM", "MA", "JNJ",
        "COST", "PG", "HD", "ABBV", "WMT", "NFLX", "MRK", "CVX", "BAC",
        "CRM", "ORCL", "AMD", "KO", "ACN", "PEP", "TMO", "MCD", "CSCO",
        "LIN", "ABT", "GE", "IBM", "TXN", "DHR", "INTU", "PM", "CMCSA",
        "WFC", "ISRG", "SPGI", "VZ", "NOW", "GS", "UBER", "AXP", "RTX",
        "MS", "AMGN", "CAT", "HON", "T", "QCOM", "NEE", "BLK", "SYK",
        "LOW", "UNP", "BSX", "BKNG", "MDT", "AMAT", "PLD", "GILD", "C",
        "ADI", "CB", "SBUX", "DE", "TJX", "LRCX", "SO", "MMC", "MU",
        "SCHW", "ADP", "BMY", "PANW", "CI", "ZTS", "REGN", "ETN", "BA",
        "ELV", "DUK", "AON", "CME", "PH", "NOC", "ITW", "WM", "FI",
        "APH", "MSI", "ECL", "USB", "GD", "PYPL", "ORLY", "CSX", "NSC",
        "PSA", "SHW", "KLAC", "PNC", "MET", "HCA", "COF", "ROP", "TGT",
    ],

    "nasdaq100": [
        # ~100-stock fallback covering current + recent NASDAQ-100 constituents
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
        "AVGO", "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTU", "CSCO",
        "TXN", "AMAT", "ISRG", "BKNG", "MU", "LRCX", "KLAC", "PANW",
        "REGN", "MELI", "ASML", "SNPS", "CDNS", "MAR", "ORLY", "CTAS",
        "FTNT", "ABNB", "DXCM", "CEG", "ROP", "PCAR", "MNST", "FAST",
        "AMGN", "ADI", "ADP", "HON", "CMCSA", "INTC", "PEP", "PYPL",
        "GILD", "SBUX", "MDLZ", "VRTX", "TMUS", "CSX", "MRVL", "MCHP",
        "NXPI", "KDP", "KHC", "EA", "WMT", "ROST", "IDXX", "WBD",
        "PAYX", "XEL", "AEP", "EXC", "ODFL", "CPRT", "FANG", "BKR",
        "ADSK", "LIN", "PDD", "TTWO", "DDOG", "CRWD", "WDAY", "GEHC",
        "MPWR", "TER", "PLTR", "DASH", "AXON", "ARM", "APP", "MSTR",
        "WDC", "STX", "SHOP", "LITE", "CCEP", "FER", "HONA", "TRI",
    ],

    "cac40": [
        "MC.PA", "TTE.PA", "SAN.PA", "AI.PA", "RI.PA", "BNP.PA", "SU.PA",
        "AIR.PA", "OR.PA", "DG.PA", "KER.PA", "SGO.PA", "CAP.PA", "ACA.PA",
        "STM.PA", "VIE.PA", "ORA.PA", "SAF.PA", "SW.PA", "HO.PA",
        "EL.PA", "CS.PA", "GLE.PA", "ML.PA", "URW.PA", "ENGI.PA",
        "PUB.PA", "AM.PA", "RMS.PA", "DSY.PA", "SHL.PA",
    ],


    "ftse100": [
        "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "GSK.L",
        "DGE.L", "BATS.L", "VOD.L", "LSEG.L", "NG.L", "BHP.L", "AAL.L",
        "LLOY.L", "NWG.L", "PRU.L", "BARC.L", "IMB.L", "REL.L",
        "CPG.L", "EXPN.L", "RKT.L", "SSE.L", "AHT.L", "WPP.L",
        "JD.L", "TSCO.L", "AUTO.L", "CRH.L", "PSN.L", "TW.L",
        "IHG.L", "BNZL.L", "FRES.L", "HIK.L", "MNDI.L", "RSA.L",
    ],

    # ── STOXX Europe 600, scinde par devise ───────────────────────────────── #
    # Un univers par devise, et non un univers paneuropeen unique :
    # UniverseConfig ne porte QU'UNE devise et QU'UN nav_scale, tous deux
    # utilises pour recuperer le NAV, le convertir et dimensionner les ordres.
    # Un univers bi-devise les rendrait tous les deux faux pour une partie des
    # lignes — exactement la classe de bug qui a produit le facteur 74x sur
    # CPG.L et IHG.L.  Scinde, chaque univers reste mono-devise et
    # l'allocation Sharpe decide elle-meme du partage entre les deux.

    # Zone euro : 289 constituants du STOXX 600 cotes en EUR
    # (Euronext Paris/Amsterdam/Bruxelles/Lisbonne, Xetra, Borsa Italiana,
    #  BME, Helsinki, Vienne, Athenes, Dublin)
    "stoxx_eur": [
        "A2A.MI", "A5G.IR", "AALB.AS", "ABI.BR", "ABN.AS", "ABVX.PA",
        "AC.PA", "ACA.PA", "ACKB.BR", "ACS.MC", "ACX.MC", "AD.AS",
        "ADP.PA", "ADS.DE", "ADYEN.AS", "AED.BR", "AENA.MC", "AG1.DE",
        "AGN.AS", "AGS.BR", "AI.PA", "AIR.PA", "AIXA.DE", "AKE.PA",
        "AKZA.AS", "ALLFG.AS", "ALO.PA", "ALV.DE", "AM.PA", "AMS.MC",
        "AMUN.PA", "ANA.MC", "ANDR.VI", "ARCAD.AS", "ARGX.BR", "ASM.AS",
        "ASML.AS", "ASRNL.AS", "ATS.VI", "AYV.PA", "AZM.MI", "BAMI.MI",
        "BAMNB.AS", "BAS.DE", "BAYN.DE", "BBVA.MC", "BC.MI", "BC8.DE",
        "BCP.LS", "BEI.DE", "BELA.AT", "BESI.AS", "BG.VI", "BGN.MI",
        "BIM.PA", "BIRG.IR", "BKT.MC", "BMED.MI", "BMPS.MI", "BMW.DE",
        "BN.PA", "BNP.PA", "BNR.DE", "BOL.PA", "BPE.MI", "BVI.PA",
        "BZU.MI", "CA.PA", "CABK.MC", "CAP.PA", "CBK.DE", "CLNX.MC",
        "CON.DE", "COV.PA", "CPR.MI", "CS.PA", "DB1.DE", "DBK.DE",
        "DG.PA", "DHER.DE", "DHL.DE", "DIE.BR", "DIM.PA", "DLG.MI",
        "DSFIR.AS", "DSY.PA", "DTE.DE", "DTG.DE", "EBS.VI", "EDEN.PA",
        "EDP.LS", "EDPR.LS", "EL.PA", "ELE.MC", "ELI.BR", "ELIS.PA",
        "ELISA.HE", "EN.PA", "ENEL.MI", "ENG.MC", "ENGI.PA", "ENI.MI",
        "ENR.DE", "ENX.PA", "EOAN.DE", "ERF.PA", "ETE.AT", "EUROB.AT",
        "EVD.DE", "EVK.DE", "EXO.AS", "FBK.MI", "FDJU.PA", "FER.MC",
        "FGR.PA", "FME.DE", "FNTN.DE", "FORTUM.HE", "FPE3.DE", "FR.PA",
        "FRE.DE", "FTK.DE", "G.MI", "G1A.DE", "G24.DE", "GALP.LS",
        "GBF.DE", "GBLB.BR", "GEKTERNA.AT", "GET.PA", "GFC.PA", "GL9.IR",
        "GLE.PA", "GRF.MC", "GTT.PA", "HAG.DE", "HEI.DE", "HEIA.AS",
        "HEIO.AS", "HEN3.DE", "HER.MI", "HIAB.HE", "HNR1.DE", "HO.PA",
        "HOT.DE", "HTO.AT", "HUH1V.HE", "IBE.MC", "IDR.MC", "IFX.DE",
        "IG.MI", "IMCD.AS", "INGA.AS", "IP.MI", "IPN.PA", "ISP.MI",
        "ITX.MC", "IVG.MI", "JMT.LS", "KBC.BR", "KBCA.BR", "KBX.DE",
        "KCR.HE", "KER.PA", "KESKOB.HE", "KGX.DE", "KNEBV.HE", "KPN.AS",
        "KRX.IR", "KRZ.IR", "LDO.MI", "LEG.DE", "LHA.DE", "LI.PA",
        "LOTB.BR", "LR.PA", "LTMC.MI", "MANTA.HE", "MAP.MC", "MBG.DE",
        "MC.PA", "METSO.HE", "ML.PA", "MOH.AT", "MONC.MI", "MRK.DE",
        "MRL.MC", "MTX.DE", "MUV2.DE", "NDA-FI.HE", "NDA.DE", "NDX1.DE",
        "NEM.DE", "NESTE.HE", "NEX.PA", "NEXI.MI", "NN.AS", "NOKIA.HE",
        "NTGY.MC", "OMV.VI", "OR.PA", "ORA.PA", "ORNBV.HE", "P911.DE",
        "PAH3.DE", "PHIA.AS", "PIRC.MI", "PPC.AT", "PRX.AS", "PRY.MI",
        "PST.MI", "PUB.PA", "PUM.DE", "QIA.DE", "RAA.DE", "RACE.MI",
        "RAND.AS", "RBI.VI", "REC.MI", "RED.MC", "REP.MC", "REY.MI",
        "RHM.DE", "RI.PA", "RMS.PA", "RNO.PA", "RUI.PA", "RWE.DE",
        "RXL.PA", "RYA.IR", "SAB.MC", "SAF.PA", "SAMPO.HE", "SAN.MC",
        "SAN.PA", "SAP.DE", "SBMO.AS", "SCR.PA", "SCYR.MC", "SDF.DE",
        "SGO.PA", "SHL.DE", "SIE.DE", "SOF.BR", "SOI.PA", "SOP.PA",
        "SPIE.PA", "SPM.MI", "SRG.MI", "SRT3.DE", "STERV.HE", "STLAM.MI",
        "STMMI.MI", "SU.PA", "SW.PA", "SY1.DE", "TE.PA", "TEF.MC",
        "TEG.DE", "TEN.MI", "TEP.PA", "TKA.DE", "TLX.DE", "TPEIR.AT",
        "TPRO.MI", "TRN.MI", "TTE.PA", "TUB.BR", "TUI1.DE", "UCB.BR",
        "UCG.MI", "UMG.AS", "UMI.BR", "UNI.MC", "UNI.MI", "UPM.HE",
        "URW.PA", "VALMT.HE", "VER.VI", "VIE.PA", "VK.PA", "VNA.DE",
        "VOE.VI", "VOW3.DE", "VPK.AS", "WDP.BR", "WKL.AS", "WRT1V.HE",
        "ZAL.DE",
    ],

    # Londres : 127 constituants du STOXX 600 cotes au LSE (GBp)
    "stoxx_uk": [
        "AAF.L", "AAL.L", "ABDN.L", "ABF.L", "ADM.L", "ANTO.L", "AUTO.L",
        "AV.L", "AZN.L", "BA.L", "BAB.L", "BARC.L", "BATS.L", "BBOX.L",
        "BBY.L", "BGEO.L", "BKG.L", "BLND.L", "BME.L", "BNZL.L", "BP.L",
        "BPT.L", "BRBY.L", "BT-A.L", "BTRW.L", "BWY.L", "CCC.L", "CCH.L",
        "CNA.L", "CPG.L", "CRDA.L", "CTEC.L", "CWK.L", "DCC.L", "DGE.L",
        "DLN.L", "DPLM.L", "DRX.L", "EDV.L", "EMG.L", "ENT.L", "EXPN.L",
        "EZJ.L", "FRES.L", "GAW.L", "GLEN.L", "GSK.L", "HIK.L", "HILS.L",
        "HLMA.L", "HLN.L", "HOC.L", "HSBA.L", "HSX.L", "HWDN.L", "IAG.L",
        "ICG.L", "IGG.L", "IHG.L", "III.L", "IMB.L", "IMI.L", "INCH.L",
        "INF.L", "INVP.L", "ITRK.L", "ITV.L", "JMAT.L", "KGF.L", "LAND.L",
        "LGEN.L", "LLOY.L", "LMP.L", "LSEG.L", "MKS.L", "MNDI.L", "MNG.L",
        "MRO.L", "MTO.L", "NG.L", "NWG.L", "NXT.L", "PAF.L", "PHP.L",
        "PLUS.L", "PNN.L", "PRU.L", "PSN.L", "PSON.L", "QLT.L", "QQ.L",
        "REL.L", "RIO.L", "RKT.L", "RMV.L", "ROR.L", "RR.L", "RS1.L",
        "RTO.L", "SBRY.L", "SCT.L", "SDLF.L", "SGE.L", "SGRO.L", "SHC.L",
        "SHEL.L", "SMIN.L", "SN.L", "SPX.L", "SRP.L", "SSE.L", "STAN.L",
        "STJ.L", "SVT.L", "TBCG.L", "TCAP.L", "TSCO.L", "TW.L", "ULVR.L",
        "UTG.L", "UU.L", "VOD.L", "WEIR.L", "WISE.L", "WPP.L", "WTB.L",
        "ZEG.L",
    ],


    # Nikkei 225 — Yahoo Finance tickers (.T suffix)
    # ~150-stock fallback used when Wikipedia scraping fails
    "nikkei225": [
        # ── Technology / electronics ─────────────────────────────────────── #
        "7203.T",  # Toyota Motor
        "6758.T",  # Sony Group
        "9984.T",  # SoftBank Group
        "7974.T",  # Nintendo
        "6861.T",  # Keyence
        "8035.T",  # Tokyo Electron
        "6857.T",  # Advantest
        "6954.T",  # Fanuc
        "4063.T",  # Shin-Etsu Chemical
        "6367.T",  # Daikin Industries
        "7741.T",  # HOYA
        "6501.T",  # Hitachi
        "6702.T",  # Fujitsu
        "6503.T",  # Mitsubishi Electric
        "6902.T",  # DENSO
        "6594.T",  # Nidec
        "6752.T",  # Panasonic Holdings
        "7751.T",  # Canon
        "6645.T",  # OMRON
        "6762.T",  # TDK Corp
        "6723.T",  # Renesas Electronics
        "6724.T",  # Seiko Epson
        "6988.T",  # Nitto Denko
        "6976.T",  # Murata Manufacturing
        "6971.T",  # Kyocera
        "6506.T",  # Yaskawa Electric
        "4704.T",  # Trend Micro
        # ── Autos / machinery ────────────────────────────────────────────── #
        "7267.T",  # Honda Motor
        "7269.T",  # Suzuki Motor
        "7270.T",  # Subaru
        "7201.T",  # Nissan Motor
        "7261.T",  # Mazda Motor
        "6301.T",  # Komatsu
        "6326.T",  # Kubota
        "5108.T",  # Bridgestone
        "7013.T",  # IHI Corp
        "7012.T",  # Kawasaki Heavy Industries
        "7731.T",  # Nikon
        "6103.T",  # Okuma Corp
        "6113.T",  # Amada Holdings
        "6471.T",  # NSK Ltd
        "6481.T",  # THK Co
        # ── Financials ───────────────────────────────────────────────────── #
        "8316.T",  # Sumitomo Mitsui Financial
        "8411.T",  # Mizuho Financial
        "8306.T",  # Mitsubishi UFJ Financial (MUFG)
        "8766.T",  # Tokio Marine Holdings
        "8750.T",  # Dai-ichi Life Insurance
        "8604.T",  # Nomura Holdings
        "8801.T",  # Mitsui Fudosan
        "8053.T",  # Sumitomo Corp
        "8001.T",  # ITOCHU Corp
        "8015.T",  # Toyota Tsusho
        "8309.T",  # Sumitomo Mitsui Trust Holdings
        "8830.T",  # Sumitomo Realty & Development
        "8697.T",  # Japan Exchange Group
        # ── Consumer / retail ────────────────────────────────────────────── #
        "9983.T",  # Fast Retailing (Uniqlo)
        "3382.T",  # Seven & i Holdings
        "8267.T",  # AEON
        "4661.T",  # Oriental Land (Tokyo Disney)
        "7832.T",  # Bandai Namco
        "4911.T",  # Shiseido
        "4452.T",  # Kao
        "2802.T",  # Ajinomoto
        "2914.T",  # Japan Tobacco
        "3086.T",  # J Front Retailing
        # ── Healthcare / pharma ──────────────────────────────────────────── #
        "4519.T",  # Chugai Pharmaceutical
        "4543.T",  # Terumo
        "4502.T",  # Takeda Pharmaceutical
        "4568.T",  # Daiichi Sankyo
        "4523.T",  # Eisai
        "4578.T",  # Otsuka Holdings
        "7733.T",  # Olympus
        "4503.T",  # Astellas Pharma
        "4507.T",  # Shionogi
        "4151.T",  # Kyowa Kirin
        "4091.T",  # Nippon Sanso Holdings
        # ── Telecom / media / IT services ────────────────────────────────── #
        "9433.T",  # KDDI
        "9432.T",  # NTT
        "6098.T",  # Recruit Holdings
        "4307.T",  # Nomura Research Institute
        "9613.T",  # NTT Data
        "6701.T",  # NEC Corp
        "2413.T",  # M3 Inc
        # ── Industrials / conglomerates ──────────────────────────────────── #
        "7011.T",  # Mitsubishi Heavy Industries
        "5401.T",  # Nippon Steel
        "5411.T",  # JFE Holdings
        "8058.T",  # Mitsubishi Corp
        "8031.T",  # Mitsui & Co.
        "8002.T",  # Marubeni
        "9022.T",  # Central Japan Railway
        "9020.T",  # East Japan Railway
        "9021.T",  # West Japan Railway
        "1925.T",  # Daiwa House Industry
        "1928.T",  # Sekisui House
        "1801.T",  # Taisei Corp
        "1802.T",  # Obayashi Corp
        "1803.T",  # Shimizu Corp
        "1812.T",  # Kajima Corp
        "3407.T",  # Asahi Kasei
        "3402.T",  # Toray Industries
        "4901.T",  # Fujifilm Holdings
        "3401.T",  # Teijin
        "5333.T",  # NGK Insulators
        "5631.T",  # Japan Steel Works
        # ── Energy / chemicals ───────────────────────────────────────────── #
        "5020.T",  # ENEOS Holdings
        "1605.T",  # Inpex Corp
        "9531.T",  # Tokyo Gas
        "9532.T",  # Osaka Gas
        "4021.T",  # Nissan Chemical
        "3436.T",  # SUMCO (silicon wafers)
        "4005.T",  # Sumitomo Chemical
        "4042.T",  # Tosoh Corp
        "4061.T",  # Denka Co
        "4004.T",  # Resonac Holdings
        "4188.T",  # Mitsubishi Chemical Group
        # ── Metals / mining ──────────────────────────────────────────────── #
        "5713.T",  # Sumitomo Metal Mining
        "5714.T",  # DOWA Holdings
        "5711.T",  # Mitsubishi Materials
        "5801.T",  # Furukawa Electric
        "5802.T",  # Sumitomo Electric Industries
        # ── Shipping / transport ─────────────────────────────────────────── #
        "9101.T",  # Nippon Yusen (NYK Line)
        "9104.T",  # Mitsui O.S.K. Lines
        "9107.T",  # Kawasaki Kisen
        "9202.T",  # ANA Holdings
        # ── Food & beverage ──────────────────────────────────────────────── #
        "2502.T",  # Asahi Group Holdings
        "2503.T",  # Kirin Holdings
        "2501.T",  # Sapporo Holdings
        # ── Misc / diversified ───────────────────────────────────────────── #
        "3861.T",  # Oji Holdings
        "7762.T",  # Citizen Watch
        "3289.T",  # Tokyu Fudosan Holdings
    ],


}


# The pan-European universe is the union of the two split lists, derived rather
# than duplicated so the three can never drift apart.
_FALLBACKS["stoxx_europe"] = sorted(
    set(_FALLBACKS["stoxx_eur"]) | set(_FALLBACKS["stoxx_uk"])
)


# --------------------------------------------------------------------------- #
#  Universe definition class                                                   #
# --------------------------------------------------------------------------- #

class UniverseDef:
    """Runtime wrapper around UniverseConfig with data-fetching methods."""

    def __init__(self, config: UniverseConfig):
        self.config = config

    # ── Tickers ─────────────────────────────────────────────────────────── #

    def get_tickers(self, cfg: TaurusConfig = DEFAULT_CONFIG) -> List[str]:
        """
        Return constituent tickers for this universe.

        US universes: scrape Wikipedia (tickers are in col 0, plain symbol).
        Non-US universes: use the hardcoded fallback list directly — Wikipedia
        pages for European/Asian indices have company names in the ticker
        column and require manual mapping.
        """
        _fb_len = len(_FALLBACKS.get(self.config.name, []))
        cache_key = f"tickers_{self.config.name}_v{_fb_len}"
        cached = _cache_load(cache_key, cfg)
        if cached is not None:
            return cached

        if self.config.region == "US" or self.config.use_wikipedia_scrape:
            tickers = self._scrape_wikipedia()
            if not tickers:
                logger.warning(
                    "[%s] Wikipedia scrape failed, using fallback tickers.",
                    self.config.name,
                )
                tickers = _FALLBACKS.get(self.config.name, [])
        else:
            # Non-US without Wikipedia scraping: use curated fallback lists
            # that already have correct Yahoo Finance suffixes (.PA, .L, .T, .HK, .SR)
            tickers = _FALLBACKS.get(self.config.name, [])
            if not tickers:
                logger.warning("[%s] No fallback tickers defined.", self.config.name)

        _cache_save(cache_key, tickers, cfg)
        logger.info("[%s] %d tickers loaded.", self.config.name, len(tickers))
        return tickers

    def _scrape_wikipedia(self) -> List[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        cfg = self.config
        try:
            resp = requests.get(cfg.wikipedia_url, headers=headers, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"id": cfg.wikipedia_table_id})
            if table is None:
                # Find the largest wikitable (most rows) — the constituents list is usually the biggest
                all_wt = soup.find_all("table", {"class": "wikitable"})
                table = max(all_wt, key=lambda t: len(t.find_all("tr"))) if all_wt else None
            if table is None:
                return []
            rows  = table.find_all("tr")[1:]
            col   = cfg.ticker_col_index        # which column holds the ticker
            tickers = []
            for row in rows:
                cells = row.find_all("td")
                if len(cells) > col:
                    raw = cells[col].get_text(strip=True)
                    # For non-US exchanges the symbol is numeric or has no dot
                    # Strip any trailing notes / parentheses
                    sym = raw.split()[0].split("(")[0].strip()
                    if not sym:
                        continue
                    # US markets: replace "." with "-" for Yahoo Finance (BRK.B → BRK-B)
                    if cfg.ticker_suffix == "":
                        sym = sym.replace(".", "-")
                    # Append exchange suffix (.T / .HK / .SR / .L etc.)
                    if cfg.ticker_suffix and not sym.endswith(cfg.ticker_suffix):
                        sym = sym + cfg.ticker_suffix
                    tickers.append(sym)
            return tickers if len(tickers) >= 10 else []
        except Exception as e:
            logger.warning("[%s] Wikipedia scrape error: %s", self.config.name, e)
            return []

    # ── Prices ──────────────────────────────────────────────────────────── #

    def get_prices(
        self,
        start: str,
        end: str,
        tickers: Optional[List[str]] = None,
        cfg: TaurusConfig = DEFAULT_CONFIG,
    ) -> pd.DataFrame:
        """
        Monthly prices for this universe, expressed in ONE currency.

        A mono-currency universe needs nothing: its prices are already in a
        single numeraire (EUR on Euronext, GBp on the LSE) and the download is
        returned untouched.

        A multi-currency universe must be converted, and before any return is
        computed.  Left in native units, a monthly return would mean "return in
        EUR" for a Paris line and "return in GBp" for a London one, so the
        cross-sectional ranking, the covariance matrix and the beta
        neutralisation would all be mixing two numeraires. Everything is
        therefore brought to `currency` (the base).

        A month where the rate is unknown propagates NaN rather than being
        filled: an invented rate is an invented return.
        """
        from .data import get_monthly_prices

        tickers = tickers if tickers is not None else self.get_tickers(cfg)
        prices = get_monthly_prices(tickers, start, end, cfg)

        if not getattr(self.config, "multi_currency", False) or prices.empty:
            return prices

        base = (self.config.currency or "EUR").upper()
        groupes: Dict[str, List[str]] = {}
        for t in prices.columns:
            _sym, ccy, minor, _px = ticker_market(t, self.config)
            if (ccy or base).upper() == base and float(minor) == 1.0:
                continue                              # already in the base unit
            groupes.setdefault(f"{ccy}:{minor}", []).append(t)

        if not groupes:
            return prices

        import yfinance as _yf
        out = prices.copy()
        for cle, cols in groupes.items():
            ccy, minor = cle.split(":")
            minor = float(minor)
            if ccy.upper() == base:
                out[cols] = out[cols].div(minor)      # minor unit only, no FX
                logger.info("[%s] %d line(s) rescaled from the minor unit of %s.",
                            self.config.name, len(cols), ccy)
                continue
            rate = None
            try:
                raw = _yf.download(f"{ccy}{base}=X", start=start, end=end,
                                   interval="1mo", auto_adjust=True,
                                   progress=False, threads=False)
                if raw is not None and not raw.empty:
                    s = raw["Close"] if "Close" in raw else raw.iloc[:, [0]]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    s = s.dropna()
                    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
                    rate = s[~s.index.duplicated(keep="last")].reindex(prices.index).ffill()
            except Exception as e:
                logger.error("[%s] FX series %s->%s failed: %s",
                             self.config.name, ccy, base, e)
            if rate is None or rate.notna().sum() == 0:
                logger.error(
                    "[%s] No %s->%s monthly rate — dropping %d line(s) rather "
                    "than mixing numeraires: %s",
                    self.config.name, ccy, base, len(cols), ", ".join(sorted(cols)[:8]),
                )
                out = out.drop(columns=cols)
                continue
            n_nan = int(rate.isna().sum())
            if n_nan:
                logger.warning("[%s] %s->%s rate missing for %d month(s) — those "
                               "months stay NaN.", self.config.name, ccy, base, n_nan)
            out[cols] = out[cols].div(minor).mul(rate, axis=0)
            logger.info("[%s] %d line(s) converted %s -> %s (median rate %.4f).",
                        self.config.name, len(cols), ccy, base,
                        float(rate.median()))

        return out.dropna(axis=1, how="all")

    # ── FF5 Factors ─────────────────────────────────────────────────────── #

    def get_ff5_factors(
        self,
        start: str,
        end: str,
        cfg: TaurusConfig = DEFAULT_CONFIG,
    ) -> pd.DataFrame:
        """
        Download Fama-French 5 (or 6) factors for this universe's region.

        Routing:
          US               → F-F_Research_Data_5_Factors_2x3  (pandas_datareader or direct)
          Europe           → Europe_5_Factors                  (direct download)
          Japan            → Japan_5_Factors                   (direct download)
          Asia / MiddleEast→ Asia_Pacific_ex_Japan_5_Factors   (best proxy)
          Global           → Global_5_Factors                  (fallback)

        When cfg.use_umd_factor=True the UMD momentum factor is appended,
        yielding a FF6 dataset for alpha orthogonalisation.
        """
        from .data import _cache_load, _cache_save

        include_umd = getattr(cfg, "use_umd_factor", False)
        dataset   = self.config.ff5_dataset
        region    = self.config.region
        cache_key = f"ff{'6' if include_umd else '5'}_{self.config.name}_{start}_{end}"
        cached    = _cache_load(cache_key, cfg)
        if cached is not None:
            return cached

        ff5 = None

        if region == "US":
            # Try pandas_datareader first (works on Python < 3.12)
            try:
                import pandas_datareader.data as web
                ff5 = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
                ff5.index = ff5.index.to_timestamp("M")
                ff5 = ff5 / 100.0
                ff5.index.name = "Date"
                logger.info("[%s] FF5 loaded via pandas_datareader.", self.config.name)
            except Exception as e:
                logger.warning("[%s] pandas_datareader failed (%s). Using direct download.", self.config.name, e)

            if ff5 is None:
                ff5 = _download_ff5_directly(start, end)
        else:
            # Europe, Japan, Asia, MiddleEast → direct download by dataset name
            ff5 = _download_ff5_region(dataset, start, end)

        if ff5 is None or ff5.empty:
            raise RuntimeError(
                f"Cannot load FF5 factors for universe {self.config.name} "
                f"(dataset={dataset}, start={start}, end={end})"
            )

        # Optionally append UMD (FF5 → FF6) — same factor regardless of region
        if include_umd:
            umd = get_umd_factor(start, end, cfg)
            if umd is not None:
                ff5 = ff5.join(umd.rename("UMD"), how="left")
                ff5["UMD"] = ff5["UMD"].fillna(0.0)
                logger.info("[%s] UMD factor merged → FF6 dataset (%d rows).", self.config.name, len(ff5))
            else:
                logger.warning("[%s] UMD unavailable — running FF5 without momentum factor.", self.config.name)

        _cache_save(cache_key, ff5, cfg)
        return ff5

    # ── Fundamentals ─────────────────────────────────────────────────────── #

    def get_fundamentals(
        self,
        tickers: List[str],
        cfg: TaurusConfig = DEFAULT_CONFIG,
    ) -> pd.DataFrame:
        """
        US   → SEC EDGAR (primary) + yfinance fallback.
        All other regions → yfinance only (not on EDGAR).
        """
        # Every region now takes the same path.  The US one used to go through
        # SEC EDGAR, which is fresher after a filing but, for this strategy,
        # worse on every count that matters: it hard-coded tax_rate to 0.21 for
        # all 495 S&P 500 names (one distinct value, against a measured 17%-41%
        # spread), it capped the market_cap backfill at 50 tickers (10% coverage
        # in the cache), and it carries NONE of the fields a valuation leg needs
        # — common equity, net income to common shareholders, share count.
        #
        # Routing US through EDGAR and everything else through yfinance also
        # made the two incomparable: different tax rates, different statement
        # horizons, different field definitions, on universes whose returns are
        # then Sharpe-weighted against each other.  One path, one definition.
        #
        # get_fundamentals and the EDGAR module are left in place; nothing calls
        # them by default any more.
        if not tickers:
            return pd.DataFrame(columns=[
                "total_debt", "total_equity", "equity_common", "total_assets",
                "ebit", "ebitda", "interest_expense", "net_income_common",
                "normalized_income", "roe", "roe_normalized",
                "effective_tax_rate", "market_cap", "sector", "tax_rate",
                "cash", "fcf", "shares_outstanding", "book_per_share",
            ])

        # "_v2": the record gained the valuation fields and the field
        # definitions changed (exact row matching, twelve-month flows,
        # effective tax rate), so v1 entries must not be reused.
        # "_v3": sector / industry are now resolved by a dedicated pass rather
        # than taken from whatever `t.info` happened to return during the
        # statement download.  v2 entries carry sector="Unknown" for 68% to 91%
        # of the non-US names — a quota refusal cached as a measurement — and
        # must not be reused.
        _key = f"fundamentals_{self.config.name}_v3"
        cached = _cache_load(_key, cfg)
        if cached is not None:
            return cached

        from .data import _fetch_single_fundamental, fetch_profiles
        from concurrent.futures import (ThreadPoolExecutor, as_completed,
                                        TimeoutError as FuturesTimeout)

        # Sector / industry FIRST, before the statements.
        #
        # This pass was originally placed after the statement download, on the
        # reasoning that `info` answers again once the statements are done.  On
        # the author's machine it did; on the user's it did not — the quota was
        # still spent, pass 1 resolved 14 names of 495 and the two retries
        # resolved none.  The run was saved only by the fallback to whatever
        # sector the statement pass had happened to collect, which is the very
        # thing this pass exists to stop depending on.
        #
        # Asking for `info` alone costs ONE call per ticker and was measured at
        # 288 successes out of 289 on a cold quota.  The statements cost six to
        # seven calls each and are what exhausts it.  So the cheap, reliable
        # request goes first and the expensive one absorbs the rationing.
        fetch_profiles(tickers, cfg)

        records = []
        logger.info("[%s] Downloading fundamentals for %d tickers (parallel)…",
                    self.config.name, len(tickers))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_single_fundamental, t): t for t in tickers}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    records.append(fut.result(timeout=60))
                except (FuturesTimeout, Exception) as e:
                    logger.debug("Fundamentals timeout/error for %s: %s", ticker, e)
                    records.append({"ticker": ticker})   # NaN-filled downstream

        df = pd.DataFrame(records).set_index("ticker")

        # Fill the frame from the profile store populated above.  Anything the
        # pre-pass could not resolve is retried here, in case the quota has
        # recovered; a name that still fails keeps whatever the statement pass
        # saw, and failing that stays an explicit "Unknown".
        from .data import apply_profiles
        df = apply_profiles(df, cfg)

        _cache_save(_key, df, cfg)
        return df

    # ── IBKR Futures Contract ────────────────────────────────────────────── #

    def get_futures_local_symbol(self) -> str:
        """
        Compute front-month local symbol for this universe's futures contract.
        E.g. ESM6, NQM6, FDAXM6, etc.
        """
        from datetime import date, timedelta

        sym = self.config.futures_symbol
        # ES and NQ: quarterly H/M/U/Z, expires 3rd Friday
        # CAC: quarterly, similar convention
        month_codes = {3: "H", 6: "M", 9: "U", 12: "Z"}
        today = date.today()

        for year in [today.year, today.year + 1]:
            for m in [3, 6, 9, 12]:
                d = date(year, m, 1)
                d += timedelta(days=(4 - d.weekday()) % 7)   # 1st Friday
                d += timedelta(weeks=2)                        # 3rd Friday
                if d > today:
                    code = month_codes[m]
                    yr   = str(year)[-1]
                    return f"{sym}{code}{yr}"

        return f"{sym}M6"  # hard fallback


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

class UniverseRegistry:
    """Dict-like registry of all available universes."""

    def __init__(self):
        self._universes: Dict[str, UniverseDef] = {
            name: UniverseDef(ucfg)
            for name, ucfg in _UNIVERSE_CONFIGS.items()
        }

    def get(self, name: str) -> UniverseDef:
        if name not in self._universes:
            raise KeyError(
                f"Unknown universe '{name}'. "
                f"Available: {list(self._universes.keys())}"
            )
        return self._universes[name]

    def all_names(self) -> List[str]:
        return list(self._universes.keys())

    def all(self) -> List[UniverseDef]:
        return list(self._universes.values())


# Module-level singleton
REGISTRY = UniverseRegistry()
