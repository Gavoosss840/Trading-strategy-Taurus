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
        supports_trail_stop=False,          # LSE rejects TRAIL on MKT parent (Error 328)
        n_longs=15,
        n_shorts=15,
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
        min_market_cap_usd=1e9,
        min_lot_size=100,                 # TSE requires orders in multiples of 100 shares
        n_longs=20,
        n_shorts=20,
    ),

    "hangseng": UniverseConfig(
        name="hangseng",
        display_name="Hang Seng (China/HK)",
        region="Asia",
        currency="HKD",
        ff5_dataset="Asia_Pacific_ex_Japan_5_Factors",
        futures_symbol="HHI",             # Hang Seng China Enterprises futures
        futures_exchange="HKEX",
        futures_currency="HKD",
        futures_multiplier=50.0,
        ibkr_exchange="SEHK",             # Stock Exchange of Hong Kong
        wikipedia_url="https://en.wikipedia.org/wiki/Hang_Seng_Index",
        wikipedia_table_id="constituents",
        ticker_suffix=".HK",              # Yahoo Finance: 700.HK
        ticker_col_index=0,
        min_market_cap_usd=1e9,
        n_longs=15,
        n_shorts=15,
    ),

    # ── Middle East ───────────────────────────────────────────────────────── #

    "tadawul": UniverseConfig(
        name="tadawul",
        display_name="TADAWUL (Saudi Arabia)",
        region="MiddleEast",
        currency="SAR",
        ff5_dataset="Global_5_Factors",   # Best available proxy (no Saudi-specific)
        futures_symbol="",                # No liquid futures on TASI
        futures_exchange="",
        futures_currency="SAR",
        futures_multiplier=1.0,
        ibkr_exchange="TADAWUL",          # IBKR uses "TADAWUL" for Saudi exchange
        wikipedia_url="https://en.wikipedia.org/wiki/Tadawul_All_Share_Index",
        wikipedia_table_id="constituents",
        ticker_suffix=".SR",              # Yahoo Finance: 2222.SR
        ticker_col_index=1,
        min_market_cap_usd=500e6,         # Lower cap for Saudi market
        n_longs=12,
        n_shorts=12,
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


    "ftse100": [
        "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "GSK.L",
        "DGE.L", "BATS.L", "VOD.L", "LSEG.L", "NG.L", "BHP.L", "AAL.L",
        "LLOY.L", "NWG.L", "PRU.L", "BARC.L", "IMB.L", "REL.L",
        "CPG.L", "EXPN.L", "RKT.L", "SSE.L", "AHT.L", "WPP.L",
        "JD.L", "TSCO.L", "AUTO.L", "CRH.L", "PSN.L", "TW.L",
        "IHG.L", "BNZL.L", "FRES.L", "HIK.L", "MNDI.L", "RSA.L",
    ],

    # Nikkei 225 — Yahoo Finance tickers (.T suffix)
    "nikkei225": [
        "7203.T",  # Toyota Motor
        "6758.T",  # Sony Group
        "9984.T",  # SoftBank Group
        "7974.T",  # Nintendo
        "6861.T",  # Keyence
        "4063.T",  # Shin-Etsu Chemical
        "8316.T",  # Sumitomo Mitsui Financial
        "7267.T",  # Honda Motor
        "6098.T",  # Recruit Holdings
        "9983.T",  # Fast Retailing (Uniqlo)
        "6954.T",  # Fanuc
        "7751.T",  # Canon
        "4519.T",  # Chugai Pharmaceutical
        "9433.T",  # KDDI
        "8035.T",  # Tokyo Electron
        "4543.T",  # Terumo
        "6367.T",  # Daikin Industries
        "4502.T",  # Takeda Pharmaceutical
        "7741.T",  # HOYA
        "8411.T",  # Mizuho Financial
        "2914.T",  # Japan Tobacco
        "9432.T",  # NTT (Nippon Telegraph)
        "6594.T",  # Nidec
        "4661.T",  # Oriental Land (Tokyo Disney)
        "7269.T",  # Suzuki Motor
        "3382.T",  # Seven & i Holdings
        "6857.T",  # Advantest
        "4911.T",  # Shiseido
        "8766.T",  # Tokio Marine Holdings
        "8801.T",  # Mitsui Fudosan
        "6501.T",  # Hitachi
        "7832.T",  # Bandai Namco
        "4568.T",  # Daiichi Sankyo
        "1925.T",  # Daiwa House Industry
        "5108.T",  # Bridgestone
        "8031.T",  # Mitsui & Co.
        "6503.T",  # Mitsubishi Electric
        "9022.T",  # Central Japan Railway
        "3407.T",  # Asahi Kasei
        "6702.T",  # Fujitsu
    ],

    # Hang Seng Index — Yahoo Finance tickers (.HK suffix)
    "hangseng": [
        "700.HK",   # Tencent Holdings
        "941.HK",   # China Mobile
        "388.HK",   # Hong Kong Exchanges (HKEX)
        "2318.HK",  # Ping An Insurance
        "1299.HK",  # AIA Group
        "3988.HK",  # Bank of China
        "1398.HK",  # ICBC
        "939.HK",   # China Construction Bank
        "2628.HK",  # China Life Insurance
        "883.HK",   # CNOOC
        "857.HK",   # PetroChina
        "27.HK",    # Galaxy Entertainment
        "5.HK",     # HSBC Holdings
        "11.HK",    # Hang Seng Bank
        "1.HK",     # CKH Holdings
        "2388.HK",  # BOC Hong Kong
        "12.HK",    # Henderson Land
        "16.HK",    # Sun Hung Kai Properties
        "823.HK",   # Link REIT
        "1044.HK",  # Hengan International
        "669.HK",   # Techtronic Industries
        "2382.HK",  # Sunny Optical
        "9999.HK",  # NetEase
        "1024.HK",  # Kuaishou Technology
        "9618.HK",  # JD.com
        "9988.HK",  # Alibaba Group
        "1810.HK",  # Xiaomi
        "6690.HK",  # Haier Smart Home
        "2269.HK",  # WuXi Biologics
        "1211.HK",  # BYD Company
        "2020.HK",  # ANTA Sports
        "6862.HK",  # Haidilao
        "3690.HK",  # Meituan
        "175.HK",   # Geely Automobile
        "2313.HK",  # Shenzhou International
    ],

    # TADAWUL (Saudi Exchange) — Yahoo Finance tickers (.SR suffix)
    "tadawul": [
        "2222.SR",  # Saudi Aramco
        "1180.SR",  # Al Rajhi Bank
        "2010.SR",  # SABIC
        "2380.SR",  # Petro Rabigh
        "4130.SR",  # Saudi Telecom (STC)
        "1120.SR",  # Al Jazira Bank
        "1020.SR",  # Bank Al Jazira (BJAZ)
        "2280.SR",  # Almarai
        "2350.SR",  # Saudi Arabian Mining (Maaden)
        "3030.SR",  # Saudi Kayan Petrochemical
        "3010.SR",  # SIPCHEM
        "1010.SR",  # Riyad Bank
        "4081.SR",  # Elm Company
        "2370.SR",  # Saudi Ceramics
        "7010.SR",  # Saudi Electricity Company (SEC)
        "2030.SR",  # Saudi Basic Industries (SABIC listed)
        "2050.SR",  # Savola Group
        "2060.SR",  # National Petrochemical
        "3020.SR",  # Saudi Advanced Industries
        "4020.SR",  # Mouwasat Medical
        "1050.SR",  # Banque Saudi Fransi
        "3040.SR",  # Abdullah Al Othaim Markets
        "4001.SR",  # Saudi Research & Media Group
        "8010.SR",  # Saudi Re
        "4290.SR",  # Tawuniya Insurance
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
        """
        Return constituent tickers for this universe.

        US universes: scrape Wikipedia (tickers are in col 0, plain symbol).
        Non-US universes: use the hardcoded fallback list directly — Wikipedia
        pages for European/Asian indices have company names in the ticker
        column and require manual mapping.
        """
        cache_key = f"tickers_{self.config.name}"
        cached = _cache_load(cache_key, cfg)
        if cached is not None:
            return cached

        if self.config.region == "US":
            tickers = self._scrape_wikipedia()
            if not tickers:
                logger.warning(
                    "[%s] Wikipedia scrape failed, using fallback tickers.",
                    self.config.name,
                )
                tickers = _FALLBACKS.get(self.config.name, [])
        else:
            # Non-US: skip Wikipedia scraping, use curated fallback lists
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
                table = soup.find("table", {"class": "wikitable"})
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

    # ── FF5 Factors ─────────────────────────────────────────────────────── #

    def get_ff5_factors(
        self,
        start: str,
        end: str,
        cfg: TaurusConfig = DEFAULT_CONFIG,
    ) -> pd.DataFrame:
        """
        Download Fama-French 5 factors for this universe's region.

        Routing:
          US               → F-F_Research_Data_5_Factors_2x3  (pandas_datareader or direct)
          Europe           → Europe_5_Factors                  (direct download)
          Japan            → Japan_5_Factors                   (direct download)
          Asia / MiddleEast→ Asia_Pacific_ex_Japan_5_Factors   (best proxy)
          Global           → Global_5_Factors                  (fallback)
        """
        from .data import _cache_load, _cache_save

        dataset   = self.config.ff5_dataset
        region    = self.config.region
        cache_key = f"ff5_{self.config.name}_{start}_{end}"
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
        if self.config.region == "US":
            return get_fundamentals(tickers, cfg)
        else:
            # Europe / Japan / Asia / MiddleEast — IFRS/local GAAP, not on EDGAR
            if not tickers:
                _empty_cols = [
                    "total_debt", "total_equity", "ebit", "interest_expense",
                    "total_assets", "market_cap", "sector", "tax_rate", "cash", "fcf",
                ]
                return pd.DataFrame(columns=_empty_cols)
            from .data import _fetch_single_fundamental
            records = [_fetch_single_fundamental(t) for t in tickers]
            return pd.DataFrame(records).set_index("ticker")

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
