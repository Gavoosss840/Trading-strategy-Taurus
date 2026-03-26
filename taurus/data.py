"""
Taurus – Data acquisition layer.

Handles:
  • S&P 500 universe (Wikipedia scrape with fallback)
  • Monthly price/return data (yfinance)
  • Fama-French 5-factor data (pandas_datareader / French library)
  • Per-stock fundamental data (balance sheet, income stmt via yfinance)
  • Disk cache with TTL to avoid redundant downloads

All public functions return clean, aligned DataFrames ready for consumption
by the analytical modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pandas_datareader.data as web
    _HAS_PDR = True
except ImportError:
    _HAS_PDR = False

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Cache helpers                                                               #
# --------------------------------------------------------------------------- #

def _cache_path(key: str, cfg: TaurusConfig) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(cfg.cache_dir) / f"{h}.pkl"


def _cache_load(key: str, cfg: TaurusConfig):
    p = _cache_path(key, cfg)
    if not p.exists():
        return None
    age_hours = (time.time() - p.stat().st_mtime) / 3600
    if age_hours > cfg.cache_ttl_hours:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(key: str, data, cfg: TaurusConfig) -> None:
    p = _cache_path(key, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(data, f)


# --------------------------------------------------------------------------- #
#  Universe                                                                    #
# --------------------------------------------------------------------------- #

def get_sp500_tickers(cfg: TaurusConfig = DEFAULT_CONFIG) -> List[str]:
    """
    Return current S&P 500 constituent tickers.

    Tries Wikipedia first; falls back to a hard-coded representative set
    so the strategy can still run offline / in tests.
    """
    cache_key = "sp500_tickers"
    cached = _cache_load(cache_key, cfg)
    if cached is not None:
        return cached

    tickers: List[str] = []
    try:
        import requests as _req
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _req.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=20,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            raise ValueError("Table #constituents not found in Wikipedia page")
        rows = table.find_all("tr")[1:]   # skip header
        tickers = []
        for row in rows:
            cells = row.find_all("td")
            if cells:
                sym = cells[0].get_text(strip=True).replace(".", "-")
                if sym:
                    tickers.append(sym)
        if len(tickers) < 100:
            raise ValueError(f"Only {len(tickers)} tickers parsed — page structure may have changed")
        logger.info("Loaded %d S&P 500 tickers from Wikipedia.", len(tickers))
    except ImportError:
        logger.warning("beautifulsoup4 not installed – using fallback tickers. Run: pip install beautifulsoup4")
        tickers = _SP500_FALLBACK
    except Exception as exc:
        logger.warning("Wikipedia fetch failed (%s). Using fallback tickers.", exc)
        tickers = _SP500_FALLBACK

    _cache_save(cache_key, tickers, cfg)
    return tickers


# 120-stock diversified fallback covering all 11 GICS sectors so the MM
# screen always has enough overleveraged AND underleveraged candidates.
_SP500_FALLBACK = [
    # Information Technology (low debt → underleveraged)
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "CSCO", "TXN",
    "INTU", "ORCL", "QCOM", "ACN", "IBM", "NOW", "AMAT", "MU", "LRCX", "KLAC",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "VZ", "T", "CMCSA", "CHTR",
    # Consumer Discretionary
    "AMZN", "TSLA", "MCD", "HD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "MAR",
    # Consumer Staples (moderate debt)
    "PG", "COST", "KO", "PEP", "WMT", "PM", "MO", "CL", "KHC", "GIS",
    # Health Care
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "CVS", "CI", "HUM", "ISRG", "REGN",
    # Financials (high leverage by design)
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "BRK-B",
    "CB", "PGR", "MET", "PRU", "AFL",
    # Industrials (moderate-high debt)
    "RTX", "CAT", "GE", "HON", "UPS", "BA", "LMT", "NOC", "DE", "MMM",
    "EMR", "ETN", "PH", "ROK", "FDX",
    # Energy (high leverage, cyclical)
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "HAL", "DVN",
    # Materials (moderate-high debt)
    "LIN", "APD", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM",
    # Utilities (highest leverage → overleveraged candidates)
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "ETR", "FE",
    # Real Estate (high leverage)
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "AVB",
    # Misc large-cap
    "V", "MA", "SPGI", "MCO", "BX", "KKR",
]


# --------------------------------------------------------------------------- #
#  Price data                                                                  #
# --------------------------------------------------------------------------- #

def get_monthly_prices(
    tickers: List[str],
    start: str,
    end: str,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Return a DataFrame of monthly adjusted close prices.

    Index: DatetimeIndex (month-end)
    Columns: tickers
    """
    key = f"prices_{'_'.join(sorted(tickers[:5]))}_{start}_{end}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    logger.info("Downloading prices for %d tickers (%s → %s)...", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Align to month-end
    prices.index = prices.index.to_period("M").to_timestamp("M")
    prices = prices.sort_index()

    # Drop tickers with insufficient history
    min_rows = cfg.min_obs
    prices = prices.loc[:, prices.notna().sum() >= min_rows]

    _cache_save(key, prices, cfg)
    logger.info("Price data shape: %s", prices.shape)
    return prices


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly return (pct change) from price DataFrame."""
    return prices.pct_change().iloc[1:]


# --------------------------------------------------------------------------- #
#  Fama-French 5 Factors                                                       #
# --------------------------------------------------------------------------- #

def get_ff5_factors(
    start: str,
    end: str,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Return monthly Fama-French 5-factor data (decimal returns).

    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    Index:   month-end DatetimeIndex
    """
    key = f"ff5_{start}_{end}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    ff5: Optional[pd.DataFrame] = None

    if _HAS_PDR:
        try:
            ff5 = web.DataReader(
                "F-F_Research_Data_5_Factors_2x3",
                "famafrench",
                start=start,
                end=end,
            )[0]
            ff5.index = ff5.index.to_timestamp("M")
            ff5 = ff5 / 100.0          # percent → decimal
            ff5.index.name = "Date"
            logger.info("FF5 factors loaded via pandas_datareader.")
        except Exception as exc:
            logger.warning("pandas_datareader FF5 fetch failed: %s", exc)

    if ff5 is None:
        ff5 = _download_ff5_directly(start, end)

    if ff5 is None:
        raise RuntimeError(
            "Could not retrieve Fama-French 5-factor data.\n"
            "Try: pip install pandas-datareader\n"
            "Or check your internet connection."
        )

    _cache_save(key, ff5, cfg)
    return ff5


def _download_ff5_directly(start: str, end: str) -> Optional[pd.DataFrame]:
    """Fallback: download FF5 CSV directly from Kenneth French's website."""
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    try:
        import io, zipfile, requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        name = [n for n in zf.namelist() if n.upper().endswith(".CSV")][0]
        raw = pd.read_csv(
            io.StringIO(zf.read(name).decode()),
            skiprows=3,
        )
        # Drop annual block at the bottom
        raw = raw[raw.iloc[:, 0].astype(str).str.match(r"^\d{6}$")]
        raw.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        raw["Date"] = pd.to_datetime(raw["Date"], format="%Y%m")
        raw = raw.set_index("Date")
        raw.index = raw.index.to_period("M").to_timestamp("M")
        raw = raw.astype(float) / 100.0
        raw = raw.loc[start:end]
        logger.info("FF5 factors downloaded directly from Kenneth French's website.")
        return raw
    except Exception as exc:
        logger.error("Direct FF5 download also failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
#  Fundamental data (for MM screen)                                            #
# --------------------------------------------------------------------------- #

def get_fundamentals(
    tickers: List[str],
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Retrieve key fundamental metrics for each ticker via yfinance.

    Returns a DataFrame indexed by ticker with columns:
        total_debt, total_equity, ebit, interest_expense,
        total_assets, market_cap, sector, tax_rate
    """
    key = f"fundamentals_{'_'.join(sorted(tickers[:5]))}_{len(tickers)}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    records = []
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        for ticker in batch:
            rec = _fetch_single_fundamental(ticker)
            records.append(rec)

    df = pd.DataFrame(records).set_index("ticker")
    _cache_save(key, df, cfg)
    logger.info("Fundamentals fetched for %d tickers.", len(df))
    return df


def _fetch_single_fundamental(ticker: str) -> dict:
    rec: dict = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        rec["market_cap"] = info.get("marketCap", np.nan)
        rec["sector"] = info.get("sector", "Unknown")
        rec["industry"] = info.get("industry", "Unknown")

        # Balance sheet
        bs = t.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            def _row(keywords):
                for col in bs.index:
                    for kw in keywords:
                        if kw.lower() in col.lower():
                            vals = bs.loc[col].dropna()
                            return float(vals.iloc[0]) if len(vals) else np.nan
                return np.nan

            rec["total_debt"] = _row(["Total Debt", "Long Term Debt"])
            rec["total_equity"] = _row(["Stockholders Equity", "Total Equity"])
            rec["total_assets"] = _row(["Total Assets"])
        else:
            rec["total_debt"] = rec["total_equity"] = rec["total_assets"] = np.nan

        # Income statement
        inc = t.quarterly_income_stmt
        if inc is not None and not inc.empty:
            def _inc_row(keywords):
                for col in inc.index:
                    for kw in keywords:
                        if kw.lower() in col.lower():
                            vals = inc.loc[col].dropna()
                            return float(vals.iloc[0]) if len(vals) else np.nan
                return np.nan

            ebit_raw = _inc_row(["EBIT", "Operating Income"])
            interest_raw = _inc_row(["Interest Expense"])
            tax_exp = _inc_row(["Tax Provision", "Income Tax"])
            pretax = _inc_row(["Pretax Income"])

            rec["ebit"] = ebit_raw if not np.isnan(ebit_raw) else np.nan
            rec["interest_expense"] = abs(interest_raw) if not np.isnan(interest_raw) else np.nan
            if not np.isnan(tax_exp) and not np.isnan(pretax) and pretax > 0:
                rec["tax_rate"] = min(tax_exp / pretax, 0.40)
            else:
                rec["tax_rate"] = 0.21  # US statutory rate
        else:
            rec["ebit"] = rec["interest_expense"] = np.nan
            rec["tax_rate"] = 0.21

    except Exception as exc:
        logger.debug("Fundamental fetch error for %s: %s", ticker, exc)
        for field in ("market_cap", "sector", "industry", "total_debt",
                      "total_equity", "total_assets", "ebit",
                      "interest_expense", "tax_rate"):
            rec.setdefault(field, np.nan if field != "sector" else "Unknown")

    return rec


# --------------------------------------------------------------------------- #
#  Sector map                                                                  #
# --------------------------------------------------------------------------- #

def get_sector_map(
    tickers: List[str],
    fundamentals: Optional[pd.DataFrame] = None,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> Dict[str, str]:
    """Return {ticker: sector} mapping."""
    if fundamentals is not None and "sector" in fundamentals.columns:
        return fundamentals["sector"].to_dict()
    fund = get_fundamentals(tickers, cfg)
    return fund["sector"].to_dict()
