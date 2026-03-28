"""
Taurus – Multi-universe registry.

Defines investable universes: S&P 500, NASDAQ 100, CAC 40, DAX, FTSE 100.
Each universe knows its own:
  - tickers (Wikipedia scrape + fallback)
  - Fama-French factor dataset (US vs European)
  - IBKR futures contract for beta hedging
  - Exchange and currency for order routing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import TaurusConfig, UniverseConfig, DEFAULT_CONFIG
from .data import (
    _cache_load, _cache_save,
    get_fundamentals,
    _download_ff5_directly,
)

logger = logging.getLogger(__name__)


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
        n_longs=25,
        n_shorts=25,
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
        wikipedia_url="https://en.wikipedia.org/wiki/Nasdaq-100",
        wikipedia_table_id="constituents",
        min_market_cap_usd=5e9,
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
        ibkr_exchange="SBF",
        wikipedia_url="https://en.wikipedia.org/wiki/CAC_40",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        n_longs=10,
        n_shorts=10,
    ),

    "dax": UniverseConfig(
        name="dax",
        display_name="DAX 40",
        region="Europe",
        currency="EUR",
        ff5_dataset="Europe_5_Factors",
        futures_symbol="DAX",
        futures_exchange="EUREX",
        futures_currency="EUR",
        futures_multiplier=25.0,
        ibkr_exchange="IBIS",
        wikipedia_url="https://en.wikipedia.org/wiki/DAX",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
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
        ibkr_exchange="LSE",
        wikipedia_url="https://en.wikipedia.org/wiki/FTSE_100",
        wikipedia_table_id="constituents",
        min_market_cap_usd=1e9,
        n_longs=15,
        n_shorts=15,
    ),
}


# --------------------------------------------------------------------------- #
#  Fallback ticker lists                                                       #
# --------------------------------------------------------------------------- #

_FALLBACKS: Dict[str, List[str]] = {

    "nasdaq100": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
        "AVGO", "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTU", "CSCO",
        "TXN", "AMAT", "ISRG", "BKNG", "MU", "LRCX", "KLAC", "PANW",
        "REGN", "MELI", "ASML", "SNPS", "CDNS", "MAR", "ORLY", "CTAS",
        "FTNT", "ABNB", "DXCM", "CEG", "ROP", "PCAR", "MNST", "FAST",
    ],

    "cac40": [
        "MC.PA", "TTE.PA", "SAN.PA", "AI.PA", "RI.PA", "BNP.PA", "SU.PA",
        "AIR.PA", "OR.PA", "DG.PA", "KER.PA", "SGO.PA", "CAP.PA", "ACA.PA",
        "STM.PA", "VIE.PA", "ORA.PA", "SAF.PA", "SW.PA", "HO.PA",
        "EL.PA", "CS.PA", "GLE.PA", "ML.PA", "URW.PA", "ENGI.PA",
        "PUB.PA", "AM.PA", "RMS.PA", "DSY.PA", "SHL.PA", "STLAM.MI",
    ],

    "dax": [
        "SAP.DE", "SIE.DE", "ALV.DE", "MRK.DE", "DTE.DE", "BAYN.DE",
        "BMW.DE", "MUV2.DE", "DBK.DE", "BAS.DE", "ADS.DE", "VOW3.DE",
        "HEN3.DE", "RWE.DE", "MTX.DE", "EOAN.DE", "BEI.DE", "FRE.DE",
        "HEI.DE", "ZAL.DE", "VNA.DE", "CON.DE", "HAB.DE", "PAH3.DE",
        "DHER.DE", "DB1.DE", "QIA.DE", "SHL.DE", "ENR.DE", "AIR.DE",
    ],

    "ftse100": [
        "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "GSK.L",
        "DGE.L", "BATS.L", "VOD.L", "LSEG.L", "NG.L", "BHP.L", "AAL.L",
        "LLOY.L", "NWG.L", "PRU.L", "BARC.L", "IMB.L", "REL.L",
        "CPG.L", "EXPN.L", "RKT.L", "SSE.L", "AHT.L", "WPP.L",
        "JD.L", "TSCO.L", "AUTO.L", "CRH.L", "PSN.L", "TW.L",
        "IHG.L", "BNZL.L", "FRES.L", "HIK.L", "MNDI.L", "RSA.L",
    ],
}


# --------------------------------------------------------------------------- #
#  Universe definition class                                                   #
# --------------------------------------------------------------------------- #

class UniverseDef:
    """Runtime wrapper around UniverseConfig with data-fetching methods."""

    def __init__(self, config: UniverseConfig):
        self.config = config

    # ── Tickers ─────────────────────────────────────────────────────────── #

    def get_tickers(self, cfg: TaurusConfig = DEFAULT_CONFIG) -> List[str]:
        """Scrape Wikipedia for universe constituents, fallback to hardcoded list."""
        cache_key = f"tickers_{self.config.name}"
        cached = _cache_load(cache_key, cfg)
        if cached is not None:
            return cached

        tickers = self._scrape_wikipedia()
        if not tickers:
            logger.warning(
                "[%s] Wikipedia scrape failed, using fallback tickers.",
                self.config.name,
            )
            tickers = _FALLBACKS.get(
                self.config.name,
                _FALLBACKS.get("nasdaq100", []),  # last resort
            )

        _cache_save(cache_key, tickers, cfg)
        logger.info("[%s] %d tickers loaded.", self.config.name, len(tickers))
        return tickers

    def _scrape_wikipedia(self) -> List[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            resp = requests.get(self.config.wikipedia_url, headers=headers, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"id": self.config.wikipedia_table_id})
            if table is None:
                # Try first wikitable as fallback
                table = soup.find("table", {"class": "wikitable"})
            if table is None:
                return []
            rows = table.find_all("tr")[1:]
            tickers = []
            for row in rows:
                cells = row.find_all("td")
                if cells:
                    sym = cells[0].get_text(strip=True).replace(".", "-")
                    if sym:
                        tickers.append(sym)
            return tickers if len(tickers) >= 10 else []
        except Exception as e:
            logger.warning("[%s] Wikipedia scrape error: %s", self.config.name, e)
            return []

    # ── FF5 Factors ─────────────────────────────────────────────────────── #

    def get_ff5_factors(
        self,
        start: str,
        end: str,
        cfg: TaurusConfig = DEFAULT_CONFIG,
    ) -> pd.DataFrame:
        """Download Fama-French 5 factors for this universe's region."""
        from .data import _cache_load, _cache_save
        import pandas_datareader.data as web

        dataset = self.config.ff5_dataset
        cache_key = f"ff5_{self.config.name}_{start}_{end}"
        cached = _cache_load(cache_key, cfg)
        if cached is not None:
            return cached

        ff5 = None
        try:
            ff5 = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
            ff5.index = ff5.index.to_timestamp("M")
            ff5 = ff5 / 100.0
            ff5.index.name = "Date"
            logger.info("[%s] FF5 factors loaded via pandas_datareader.", self.config.name)
        except Exception as e:
            logger.warning("[%s] pandas_datareader failed: %s. Trying direct download.", self.config.name, e)
            ff5 = _download_ff5_directly(start, end)   # US fallback

        if ff5 is None:
            raise RuntimeError(f"Cannot load FF5 factors for universe {self.config.name}")

        _cache_save(cache_key, ff5, cfg)
        return ff5

    # ── Fundamentals ─────────────────────────────────────────────────────── #

    def get_fundamentals(
        self,
        tickers: List[str],
        cfg: TaurusConfig = DEFAULT_CONFIG,
    ) -> pd.DataFrame:
        """
        US universes: SEC EDGAR (primary) + yfinance fallback.
        European universes: yfinance only (IFRS, not on EDGAR).
        """
        if self.config.region == "Europe":
            # Force yfinance path for European stocks
            from .data import _fetch_single_fundamental
            records = [_fetch_single_fundamental(t) for t in tickers]
            return pd.DataFrame(records).set_index("ticker")
        else:
            return get_fundamentals(tickers, cfg)

    # ── IBKR Futures Contract ────────────────────────────────────────────── #

    def get_futures_local_symbol(self) -> str:
        """
        Compute front-month local symbol for this universe's futures contract.
        E.g. ESM6, NQM6, FDAXM6, etc.
        """
        from datetime import date, timedelta

        sym = self.config.futures_symbol
        # ES and NQ: quarterly H/M/U/Z, expires 3rd Friday
        # DAX/CAC: quarterly, similar convention
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
