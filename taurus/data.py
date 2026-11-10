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

_HAS_PDR = False  # lazy-loaded inside get_ff5_factors to avoid import-time errors

from .config import TaurusConfig, DEFAULT_CONFIG

try:
    from .sec_edgar import get_fundamentals_batch as _edgar_batch
    _HAS_EDGAR = True
except ImportError:
    _HAS_EDGAR = False

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
    # Cache key carries "_d2" since the switch from Yahoo monthly bars to
    # daily-resampled prices: older entries hold the corrupted monthly series.
    key = f"prices_d2_{'_'.join(sorted(tickers[:5]))}_{start}_{end}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    # ── Daily bars, resampled to month-end ────────────────────────────────── #
    # Yahoo's MONTHLY bars (interval="1mo", auto_adjust=True) misapply the
    # dividend adjustment on some non-US names, and the corruption is silent.
    # Measured against the daily series over 2009-2026:
    #   S&P 500 / NASDAQ 100 / Nikkei 225 : 0.00 % of points wrong
    #   CAC 40                            : 0.55 % wrong (up to 2.2 %)
    #   FTSE 100                          : 2.75 % wrong, CPG.L off by +70 %
    #   STOXX 600 (London)                : III.L off by -99 % on 3 months
    # A -10 % or +70 % error passes every outlier screen and flows straight
    # into the FF5 alpha, the betas and the covariance.  The daily series is
    # the reference and does not depend on Yahoo's monthly adjustment logic,
    # so we resample it ourselves.  Chunked to keep payloads reasonable.
    logger.info("Downloading prices for %d tickers (%s → %s)...", len(tickers), start, end)

    frames: List[pd.DataFrame] = []
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            logger.warning("No price data returned for chunk %d-%d.", i, i + len(chunk))
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            part = raw["Close"]
        else:
            part = raw[["Close"]].copy()
            part.columns = chunk[:1] if len(chunk) == 1 else chunk
        frames.append(part)

    if not frames:
        logger.error("Price download returned nothing for %d tickers.", len(tickers))
        return pd.DataFrame()

    daily = pd.concat(frames, axis=1)
    daily = daily.loc[:, ~daily.columns.duplicated(keep="first")]
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Last traded price of each month; .last() skips NaN, so a stock that did
    # not trade on the final session keeps its most recent price that month.
    prices = daily.resample("ME").last()
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
#  Quote-currency normalisation                                               #
# --------------------------------------------------------------------------- #

_QUOTE_CCY_CACHE: Dict[str, Optional[str]] = {}
_FX_RATE_CACHE: Dict[Tuple[str, str], Optional[float]] = {}


def _split_quote_currency(code: str) -> Tuple[str, float]:
    """
    Split a Yahoo currency code into (major ISO code, minor units per major).

    Yahoo distinguishes pence from pounds by CASE ALONE: "GBp" means pence,
    "GBP" means pounds.  Never upper-case this value before testing it.
    """
    c = (code or "").strip()
    if c in ("GBp", "GBX"):
        return "GBP", 100.0
    if c in ("ZAc", "ZAC"):
        return "ZAR", 100.0
    if c in ("ILa", "ILA"):
        return "ILS", 100.0
    return c.upper(), 1.0


def _quote_currency(ticker: str) -> Optional[str]:
    """Yahoo's reported quote currency for one ticker; cached per process."""
    if ticker in _QUOTE_CCY_CACHE:
        return _QUOTE_CCY_CACHE[ticker]
    ccy = None
    try:
        t = yf.Ticker(ticker)
        fast = getattr(t, "fast_info", None)
        if fast is not None:
            try:
                ccy = fast["currency"]
            except Exception:
                ccy = None
        if not ccy:
            ccy = (t.info or {}).get("currency")
    except Exception as e:
        logger.debug("Quote currency lookup failed for %s: %s", ticker, e)
    _QUOTE_CCY_CACHE[ticker] = ccy
    return ccy


def _fx_rate(from_ccy: str, to_ccy: str) -> Optional[float]:
    """Spot FX rate: 1 unit of from_ccy expressed in to_ccy."""
    if from_ccy == to_ccy:
        return 1.0
    k = (from_ccy, to_ccy)
    if k in _FX_RATE_CACHE:
        return _FX_RATE_CACHE[k]
    rate = None
    try:
        raw = yf.download(f"{from_ccy}{to_ccy}=X", period="5d", interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        if raw is not None and not raw.empty:
            s = raw["Close"] if "Close" in raw else raw.iloc[:, [0]]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            if len(s):
                rate = float(s.iloc[-1])
    except Exception as e:
        logger.debug("FX rate %s->%s failed: %s", from_ccy, to_ccy, e)
    _FX_RATE_CACHE[k] = rate
    return rate


def normalise_quote_prices(
    prices: Dict[str, float],
    universe_cfg,
    fx_fn=None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Bring yfinance close prices into the universe's expected quote unit.

    Yahoo does not quote every line of an exchange in that exchange's own
    currency.  On the LSE most names come back in GBp (pence), but CPG.L and
    IHG.L come back in USD.  compute_target_shares divides a NAV already
    scaled by universe_cfg.nav_scale (100 on the LSE: GBP → GBp) by this
    price, so a USD quote near 180 where ~8500 pence is expected sizes the
    position roughly 47x too large.  Nothing downstream checks the unit.

    This is a strict no-op wherever the reported currency already matches the
    universe (GBp on the LSE, EUR on Euronext, USD in the US, JPY in Tokyo):
    the scale factor comes out at exactly 1.0 and the price is passed through.
    It only acts on a genuine mismatch, and a ticker whose unit cannot be
    established is DROPPED rather than sized on an unverified price — an
    undeployed position costs the expected return of one name, a position
    sized 47x too large costs far more.

    Returns (usable prices, {ticker: reason dropped}).
    """
    expected_major = (getattr(universe_cfg, "currency", "") or "").upper()
    expected_minor = float(getattr(universe_cfg, "nav_scale", 1.0) or 1.0)
    rate_fn = fx_fn or _fx_rate

    out: Dict[str, float] = {}
    dropped: Dict[str, str] = {}

    for ticker, px in prices.items():
        try:
            px = float(px)
        except (TypeError, ValueError):
            dropped[ticker] = "price not numeric"
            continue
        if not (px > 0) or not np.isfinite(px):
            dropped[ticker] = "no usable price"
            continue

        code = _quote_currency(ticker)
        if not code:
            dropped[ticker] = "quote currency unknown"
            continue

        major, minor = _split_quote_currency(code)
        if major == expected_major:
            factor = expected_minor / minor
        else:
            rate = rate_fn(major, expected_major)
            if rate is None or not (rate > 0):
                dropped[ticker] = (f"quoted in {code}, no "
                                   f"{major}->{expected_major} rate available")
                continue
            factor = rate * expected_minor / minor

        if not np.isfinite(factor) or factor <= 0:
            dropped[ticker] = f"quoted in {code}, bad scale factor"
            continue

        out[ticker] = px * factor
        if abs(factor - 1.0) > 1e-9:
            logger.warning(
                "[%s] %s is quoted in %s, not the expected unit — price "
                "rescaled %.6g -> %.6g (x%.6g) before sizing.",
                getattr(universe_cfg, "name", "?"), ticker, code,
                px, out[ticker], factor,
            )

    return out, dropped


# --------------------------------------------------------------------------- #
#  Fama-French 5 Factors                                                       #
# --------------------------------------------------------------------------- #

def get_ff5_factors(
    start: str,
    end: str,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Return monthly Fama-French factor data (decimal returns).

    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF  [+ UMD when use_umd_factor=True]
    Index:   month-end DatetimeIndex

    When cfg.use_umd_factor=True the UMD (momentum) factor is appended so that
    factors.py can orthogonalise alpha against it (FF6 instead of FF5).
    """
    include_umd = getattr(cfg, "use_umd_factor", False)
    key = f"ff{'6' if include_umd else '5'}_{start}_{end}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    ff5: Optional[pd.DataFrame] = None

    try:
        import pandas_datareader.data as web
        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3",
            "famafrench",
            start=start,
            end=end,
        )[0]
        ff5.index = ff5.index.to_timestamp("M")
        ff5 = ff5 / 100.0
        ff5.index.name = "Date"
        logger.info("FF5 factors loaded via pandas_datareader.")
    except Exception as exc:
        logger.warning("pandas_datareader FF5 fetch failed: %s", exc)

    if ff5 is None:
        ff5 = _download_ff5_directly(start, end)

    if ff5 is None:
        raise RuntimeError(
            "Could not retrieve Fama-French factor data.\n"
            "Try: pip install pandas-datareader\n"
            "Or check your internet connection."
        )

    # Optionally append UMD (FF5 → FF6)
    if include_umd:
        umd = get_umd_factor(start, end, cfg)
        if umd is not None:
            ff5 = ff5.join(umd.rename("UMD"), how="left")
            ff5["UMD"] = ff5["UMD"].fillna(0.0)   # fill gaps (rare) with zero
            logger.info("UMD factor merged → FF6 dataset (%d rows).", len(ff5))
        else:
            logger.warning("UMD unavailable — running FF5 without momentum factor.")

    _cache_save(key, ff5, cfg)
    return ff5


def get_umd_factor(
    start: str,
    end: str,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> Optional[pd.Series]:
    """
    Download the Fama-French momentum factor (UMD = Up Minus Down).

    Returns a monthly Series indexed by month-end dates, or None if unavailable.
    """
    key = f"umd_{start}_{end}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    umd: Optional[pd.Series] = None

    # Try pandas_datareader first
    try:
        import pandas_datareader.data as web
        raw = web.DataReader("F-F_Momentum_Factor", "famafrench", start=start, end=end)[0]
        raw.index = raw.index.to_timestamp("M")
        raw = raw / 100.0
        umd = raw.iloc[:, 0].rename("UMD")   # single column (Mom or WML)
        logger.info("UMD factor loaded via pandas_datareader.")
    except Exception as exc:
        logger.warning("pandas_datareader UMD failed: %s. Trying direct download.", exc)

    # Direct download fallback
    if umd is None:
        try:
            import io, zipfile, requests
            url = (
                "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "F-F_Momentum_Factor_CSV.zip"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            zf   = zipfile.ZipFile(io.BytesIO(resp.content))
            name = [n for n in zf.namelist() if n.upper().endswith(".CSV")][0]
            raw  = pd.read_csv(io.StringIO(zf.read(name).decode()), skiprows=13)
            raw  = raw[raw.iloc[:, 0].astype(str).str.match(r"^\d{6}$")]
            raw.columns = [c.strip() for c in raw.columns]
            raw.index = pd.to_datetime(raw.iloc[:, 0].astype(str), format="%Y%m")
            raw.index = raw.index.to_period("M").to_timestamp("M")
            raw = raw.astype(float, errors="ignore") / 100.0
            mom_col = [c for c in raw.columns if c not in ("", "Date")][0]
            umd = raw[mom_col].rename("UMD").loc[start:end]
            logger.info("UMD factor downloaded directly from Kenneth French's website.")
        except Exception as exc2:
            logger.warning("Direct UMD download failed: %s. Running without UMD.", exc2)
            return None

    _cache_save(key, umd, cfg)
    return umd


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
    Retrieve key fundamental metrics for each ticker.

    Source priority:
      1. SEC EDGAR (gratuit, <15min après publication 10-Q/10-K)
      2. yfinance (fallback si EDGAR indisponible)

    Returns a DataFrame indexed by ticker with columns:
        total_debt, total_equity, ebit, interest_expense,
        total_assets, market_cap, sector, tax_rate, cash, fcf
    """
    if not tickers:
        _empty_cols = [
            "total_debt", "total_equity", "ebit", "interest_expense",
            "total_assets", "market_cap", "sector", "tax_rate", "cash", "fcf",
        ]
        return pd.DataFrame(columns=_empty_cols)

    key = f"fundamentals_{'_'.join(sorted(tickers[:5]))}_{len(tickers)}"
    cached = _cache_load(key, cfg)
    if cached is not None:
        return cached

    if _HAS_EDGAR:
        logger.info("Fetching fundamentals from SEC EDGAR for %d tickers...", len(tickers))
        edgar_data = _edgar_batch(tickers, max_workers=5)
        records = []
        for ticker in tickers:
            row = edgar_data.get(ticker, {})
            rec = {"ticker": ticker}
            rec["total_debt"]       = row.get("total_debt",       np.nan)
            rec["total_equity"]     = row.get("total_equity",     np.nan)
            rec["total_assets"]     = row.get("total_assets",     np.nan)
            rec["ebit"]             = row.get("ebit",             np.nan)
            rec["interest_expense"] = row.get("interest_expense", np.nan)
            rec["cash"]             = row.get("cash",             np.nan)
            rec["market_cap"]       = row.get("market_cap",       np.nan)
            rec["sector"]           = row.get("sector",           "Unknown")
            rec["tax_rate"]         = 0.21   # taux standard US
            rec["fcf"]              = row.get("free_cash_flow", row.get("net_income", np.nan))  # OpCF-CapEx
            records.append(rec)
        df = pd.DataFrame(records).set_index("ticker")

        # Compléter market_cap et sector via yfinance (données légères)
        missing_mktcap = df[df["market_cap"].isna()].index.tolist()
        if missing_mktcap:
            logger.info("Completing market_cap/sector for %d tickers via yfinance...", len(missing_mktcap))
            for ticker in missing_mktcap[:50]:   # limite pour éviter les timeouts
                try:
                    info = yf.Ticker(ticker).info or {}
                    df.loc[ticker, "market_cap"] = info.get("marketCap", np.nan)
                    df.loc[ticker, "sector"]     = info.get("sector", "Unknown")
                except Exception:
                    pass
    else:
        logger.info("SEC EDGAR non disponible, utilisation de yfinance...")
        records = []
        for ticker in tickers:
            rec = _fetch_single_fundamental(ticker)
            records.append(rec)
        df = pd.DataFrame(records).set_index("ticker")

    _cache_save(key, df, cfg)
    logger.info("Fundamentals ready for %d tickers.", len(df))
    return df


def _fetch_single_fundamental(ticker: str) -> dict:
    rec: dict = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        rec["market_cap"] = info.get("marketCap", np.nan)
        rec["sector"] = info.get("sector", "Unknown")
        rec["industry"] = info.get("industry", "Unknown")

        # ── Statement lookup with lazy ANNUAL fallback ──────────────────── #
        # Many European issuers report semi-annually, so quarterly frames are
        # sparse for them: on a STOXX 600 sample the quarterly income statement
        # carries EBIT for 32% of names against 80% for the annual one, and
        # interest_expense / fcf drop to ~45%.  The MM screen reads a third of
        # the signal from these fields, so it was running partly blind on
        # CAC 40 and FTSE 100.
        #
        # Quarterly is still preferred when present (fresher, and complete for
        # US names); the annual statement is fetched only when a field is
        # actually missing, so this costs no extra request in the common case.
        _annual: dict = {}

        def _annual_frame(attr: str):
            if attr not in _annual:
                try:
                    _annual[attr] = getattr(t, attr)
                except Exception:
                    _annual[attr] = None
            return _annual[attr]

        def _scan(frame, keywords):
            if frame is None or getattr(frame, "empty", True):
                return np.nan
            for row in frame.index:
                for kw in keywords:
                    if kw.lower() in row.lower():
                        vals = frame.loc[row].dropna()
                        if len(vals):
                            return float(vals.iloc[0])
            return np.nan

        def _pick(quarterly, annual_attr, keywords):
            val = _scan(quarterly, keywords)
            if val == val:                       # not NaN
                return val
            return _scan(_annual_frame(annual_attr), keywords)

        bs  = t.quarterly_balance_sheet
        inc = t.quarterly_income_stmt
        cf  = t.quarterly_cashflow

        # Balance sheet
        rec["total_debt"]    = _pick(bs, "balance_sheet", ["Total Debt", "Long Term Debt"])
        rec["total_equity"]  = _pick(bs, "balance_sheet", ["Stockholders Equity", "Total Equity"])
        rec["total_assets"]  = _pick(bs, "balance_sheet", ["Total Assets"])
        rec["cash"]          = _pick(bs, "balance_sheet", ["Cash And Cash Equivalents", "Cash"])

        # Income statement
        ebit_raw     = _pick(inc, "income_stmt", ["EBIT", "Operating Income"])
        interest_raw = _pick(inc, "income_stmt", ["Interest Expense"])
        tax_exp      = _pick(inc, "income_stmt", ["Tax Provision", "Income Tax"])
        pretax       = _pick(inc, "income_stmt", ["Pretax Income"])

        rec["ebit"] = ebit_raw
        rec["interest_expense"] = abs(interest_raw) if interest_raw == interest_raw else np.nan
        if tax_exp == tax_exp and pretax == pretax and pretax > 0:
            rec["tax_rate"] = min(tax_exp / pretax, 0.40)
        else:
            rec["tax_rate"] = 0.21

        # Free cash flow = operating cash flow − capex
        op_cf = _pick(cf, "cashflow", ["Operating Cash Flow", "Cash Flow From Operations"])
        capex = _pick(cf, "cashflow", ["Capital Expenditure"])
        if op_cf == op_cf and capex == capex:
            rec["fcf"] = op_cf - abs(capex)
        elif op_cf == op_cf:
            rec["fcf"] = op_cf
        else:
            rec["fcf"] = _pick(cf, "cashflow", ["Free Cash Flow"])

    except Exception as exc:
        logger.debug("Fundamental fetch error for %s: %s", ticker, exc)
        for field in ("market_cap", "sector", "industry", "total_debt",
                      "total_equity", "total_assets", "cash", "ebit",
                      "interest_expense", "tax_rate", "fcf"):
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
