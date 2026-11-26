"""
Taurus – IBKR order execution engine.

Converts PortfolioSnapshot weights → share counts → IBKR orders.

Safety principles:
  - dry_run=True by default: logs orders without submitting
  - Exits before entries (margin safety)
  - All orders go through qualifyContracts() first
  - ExecutionReport saved to disk after every rebalance
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import TaurusConfig, UniverseConfig, DEFAULT_CONFIG
from .data import normalise_quote_prices


def _stock_contract(ticker: str, universe_cfg):
    """
    IBKR Stock contract for a Yahoo ticker.

    Every call site used to build Stock(symbol, universe_exchange,
    universe_currency) with the suffix stripped, which cannot tell two listings
    apart: Stock("SAN", "SMART", "EUR") matches Sanofi (Paris) AND Banco
    Santander (Madrid).  stoxx_eur holds SAN.PA and SAN.MC, plus UNI.MI and
    UNI.MC, so four names collapsed onto two contracts and two of them would
    have been traded as the wrong company.

    primaryExchange disambiguates the listing; the currency comes from the
    ticker's own market rather than the universe, which is what lets a universe
    span several quote currencies.  A ticker with no suffix (US) resolves to the
    universe currency and no primaryExchange, exactly as before.
    """
    from ib_insync import Stock
    from .universe import ticker_market

    sym, ccy, _minor, primary = ticker_market(ticker, universe_cfg)
    if primary:
        return Stock(sym, universe_cfg.ibkr_exchange, ccy,
                     primaryExchange=primary)
    return Stock(sym, universe_cfg.ibkr_exchange, ccy)

logger = logging.getLogger(__name__)


def session_state(trading_hours: str, time_zone_id: str):
    """
    Is this exchange in session right now? -> (open?, human text)

    open? is None when IBKR's own strings cannot be read, in which case the
    text is the raw first segment and no claim is made.

    IBKR gives tradingHours as "20261126:0930-20261126:1600;2026...", or
    "20261126:CLOSED", in the exchange's own zone, named by timeZoneId —
    "US/Eastern", "Japan", "MET", "GB-Eire". Those are tz database names, so
    the comparison can be done properly instead of eyeballing the first
    segment: Tokyo's string read 20261125:1700-20261126:0600, which looks
    like a live session and is in fact one that ended hours ago.
    """
    from datetime import datetime

    brut = (trading_hours or "").split(";")[0]
    try:
        from zoneinfo import ZoneInfo
        maintenant = datetime.now(ZoneInfo(time_zone_id)).replace(tzinfo=None)
    except Exception:
        return None, brut or "?"

    prochain = None
    for seg in (trading_hours or "").split(";"):
        if not seg or "CLOSED" in seg.upper():
            continue
        deb, _, fin = seg.partition("-")
        try:
            a = datetime.strptime(deb.strip(), "%Y%m%d:%H%M")
            b = datetime.strptime(fin.strip(), "%Y%m%d:%H%M")
        except ValueError:
            continue
        if a <= maintenant <= b:
            return True, f"open, closes {b:%H:%M}"
        if maintenant < a and (prochain is None or a < prochain):
            prochain = a
    if prochain is not None:
        return False, f"CLOSED, opens {prochain:%a %d %b %H:%M}"
    return False, "CLOSED"


def resolve_contract(conn, ticker: str, universe_cfg):
    """
    Resolve a Yahoo ticker to a contract IBKR will actually accept.

    Returns (contract, route) on success, (None, why) on failure.

    Three routes, tried in order, because one spec does not cover every
    venue:

      1. SMART with the primary listing named. Works for the US, and for the
         European venues SMART routes.
      2. The exchange itself. SMART does not route everything — Athens and
         Dublin among them — and naming it as primaryExchange is not the same
         as sending the order there. A book that holds ETE.AT and cannot
         qualify it is proof the instrument exists and the spec is wrong.
      3. IBKR's own symbol search. The Yahoo stem is not always IBKR's
         symbol: London files Rolls-Royce as "RR.", Smith & Nephew as "SN."
         and United Utilities as "UU.", which Yahoo writes RR.L, SN.L, UU.L —
         the trailing dot is part of the ticker, and stripping ".L" loses it.
         The search is asked what it knows under the stem and the answer is
         matched on security type, currency and listing.
    """
    from ib_insync import Contract, Stock

    from .universe import ticker_market

    sym, ccy, _minor, primary = ticker_market(ticker, universe_cfg)
    smart = getattr(universe_cfg, "ibkr_exchange", "SMART")
    essais = []

    def _q(c, etiquette):
        # qualifyContracts RETURNS the contracts it resolved, and that return
        # is the only trustworthy signal. Checking c.conId instead passes
        # unconditionally for a contract built from a conId we already had —
        # which is how five Athens lines cleared the pre-flight with an empty
        # symbol and were then refused at order time with error 200, "no
        # security definition found". A full contract also carries a symbol
        # once IBKR has confirmed it, so an empty one means no confirmation.
        try:
            ok = conn.ib.qualifyContracts(c)
        except Exception as e:
            essais.append(f"{etiquette}: {type(e).__name__}")
            return None
        if ok and getattr(c, "conId", 0) and getattr(c, "symbol", ""):
            return c
        essais.append(f"{etiquette}: no match")
        return None

    # The route is tagged, not just described, so a caller can tell the
    # ordinary path from a fallback without parsing prose: every US line
    # resolves on route 1 and listing them all as "a second route" buries the
    # handful that matter.
    # Route 1 — the Yahoo stem, routed through SMART.
    if primary:
        c = _q(Stock(sym, smart, ccy, primaryExchange=primary), f"{smart}+{primary}")
        if c:
            return c, f"smart:{smart}/{primary}"
    else:
        c = _q(Stock(sym, smart, ccy), smart)
        if c:
            return c, f"smart:{smart}"

    # Route 2 — ask IBKR what it files under this stem, then route THAT
    # through SMART too.
    #
    # Qualifying a contract and being allowed to send an order to it are two
    # different things. Naming the exchange directly qualifies fine and is
    # then refused at order time with error 10311, "this order will be
    # directly routed to LSE" — direct routing is off on the account. So the
    # symbol the search gives back is used to build an ordinary SMART
    # contract, which is what fixes RR./SN./UU.: London files Rolls-Royce as
    # "RR.", and the trailing dot is part of the ticker that stripping ".L"
    # from the Yahoo name throws away.
    try:
        trouves = conn.ib.reqMatchingSymbols(sym) or []
    except Exception as e:
        trouves = []
        essais.append(f"symbol search: {type(e).__name__}")

    for d in trouves:
        dc = getattr(d, "contract", None)
        if dc is None or dc.secType != "STK" or dc.currency != ccy:
            continue
        if primary and dc.primaryExchange and dc.primaryExchange != primary:
            continue
        if dc.symbol == sym and not primary:
            continue                      # route 1 already tried that spec
        pri = primary or dc.primaryExchange
        c = _q(Stock(dc.symbol, smart, ccy, primaryExchange=pri) if pri
               else Stock(dc.symbol, smart, ccy), f"search {dc.symbol}+{smart}")
        if c:
            return c, f"search:{dc.symbol}/{smart}"

    # Route 3 — the exchange itself, last. It qualifies where SMART does not
    # reach, but the order may still be refused for direct routing, which is
    # a setting on the account rather than anything this code can fix. Taking
    # it last means a name that has any SMART route uses it.
    if primary:
        c = _q(Stock(sym, primary, ccy), primary)
        if c:
            return c, f"native:{primary}"

    return None, "; ".join(essais) or "no route"


# --------------------------------------------------------------------------- #
#  IBKR Connection manager                                                    #
# --------------------------------------------------------------------------- #

class IBKRConnection:
    """
    Manages a single ib_insync IB() instance with reconnect logic.
    All execution classes receive this object; they never create their own IB().
    """

    def __init__(self, cfg: TaurusConfig = DEFAULT_CONFIG):
        from ib_insync import IB
        self.cfg        = cfg
        self.ib         = IB()
        self._attempt   = 0

    def connect(self, timeout: int = 15) -> bool:
        try:
            if self.ib.isConnected():
                return True
            self.ib.connect(
                self.cfg.ibkr_host,
                self.cfg.ibkr_port,
                clientId=self.cfg.ibkr_client_id,   # fixed — never increment; same clientId every session
                timeout=timeout,
                readonly=False,
            )
            self._attempt = 0
            logger.info(
                "IBKR connected → %s:%d (clientId=%d, %s)",
                self.cfg.ibkr_host,
                self.cfg.ibkr_port,
                self.cfg.ibkr_client_id,
                "LIVE" if self.is_live else "PAPER",
            )
            return True
        except Exception as e:
            self._attempt += 1
            logger.error("IBKR connect failed (attempt %d): %s", self._attempt, e)
            return False

    def ensure_connected(self) -> bool:
        """Called before every operation. Connects, then retries with backoff.

        The first attempt is made IMMEDIATELY.  It used to be preceded by the
        first 60-second wait, because a freshly built connection reports
        isConnected() False and the loop slept before trying: every connection
        in the program cost a minute before a single packet was sent, and the
        log said "disconnected, reconnecting" about a connection that had never
        been attempted.  With a connection opened per universe plus one for the
        NAV, that was seven minutes of pure waiting per order sheet, and it
        looked exactly like a broker problem.

        The backoff still applies BETWEEN attempts, which is what it is for.
        """
        if self.ib.isConnected():
            return True
        if self.connect():
            return True
        waits = [60, 120, 240, 480, 900, 1800]   # seconds, between retries
        for wait in waits:
            logger.warning("IBKR unreachable — retrying in %ds...", wait)
            time.sleep(wait)
            if self.connect():
                return True
        logger.error("IBKR reconnect failed after all attempts.")
        return False

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKR disconnected.")

    @property
    def is_live(self) -> bool:
        return self.cfg.ibkr_port == 7496


# --------------------------------------------------------------------------- #
#  Position reconciler                                                         #
# --------------------------------------------------------------------------- #

class PositionReconciler:
    """
    Compares target weights (PortfolioSnapshot) with live IBKR positions.
    Produces the delta (trades needed) as signed share counts.
    """

    def get_live_positions(self, conn: IBKRConnection, currency: str = None) -> pd.DataFrame:
        """Returns DataFrame: ticker, quantity (+long / -short), avg_cost.

        currency: if provided, only returns positions in that currency.
        This prevents cross-universe contamination (e.g. SP500 seeing CAC40 EUR
        positions) and fixes duplicate-ticker crashes when the same symbol exists
        on multiple exchanges in different currencies (e.g. SU, MC, CRH, SW).
        """
        positions = conn.ib.positions(account=conn.cfg.ibkr_account or "")
        if not positions:
            return pd.DataFrame(columns=["ticker", "quantity", "avg_cost"])

        records = []
        for p in positions:
            if currency and p.contract.currency != currency:
                continue
            records.append({
                "ticker":   p.contract.symbol,
                "quantity": p.position,
                "avg_cost": p.avgCost,
            })
        if not records:
            return pd.DataFrame(columns=["ticker", "quantity", "avg_cost"])
        return pd.DataFrame(records).set_index("ticker")

    def get_account_nav(self, conn: IBKRConnection, currency: str = None) -> float:
        """
        Returns NAV in the requested currency.

        Strategy:
        1. If IBKR account summary already contains NetLiquidation in the target
           currency, return it directly (rare for non-base currencies).
        2. Otherwise get base-currency NAV and convert via a live FX quote from IBKR.
        3. Final fallback: return base-currency NAV unchanged (logs a warning).

        This correctly handles EUR-base accounts trading JPY (Nikkei), USD (SP500),
        GBP (FTSE 100), etc.
        """
        summary = conn.ib.accountSummary(account=conn.cfg.ibkr_account or "")

        # 1. Try direct match in requested currency
        if currency:
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == currency:
                    return float(item.value)

        # 2. Get base-currency NAV (first non-BASE entry)
        base_nav = None
        base_currency = None
        for item in summary:
            if item.tag == "NetLiquidation" and item.currency not in ("BASE", ""):
                base_nav = float(item.value)
                base_currency = item.currency
                break

        if base_nav is None:
            return conn.cfg.nav_usd

        if currency is None or currency == base_currency:
            return base_nav

        # 3. FX conversion: base_currency → requested currency
        rate = self._fetch_fx_rate(conn, base_currency, currency)
        if rate and rate > 0:
            converted = base_nav * rate
            logger.info(
                "NAV: %.2f %s × %.4f (%s%s) = %.2f %s",
                base_nav, base_currency, rate, base_currency, currency, converted, currency,
            )
            return converted

        logger.warning(
            "Could not fetch FX %s→%s — using %s NAV; position sizes for %s will be wrong",
            base_currency, currency, base_currency, currency,
        )
        return base_nav

    @staticmethod
    def _fetch_fx_rate(conn: IBKRConnection, from_ccy: str, to_ccy: str) -> Optional[float]:
        """
        Fetch live spot rate from_ccy → to_ccy from IBKR.
        Tries the direct pair first (e.g. EURJPY), then the inverse (JPYEUR → 1/rate).
        Returns None if both fail.
        """
        if from_ccy == to_ccy:
            return 1.0
        from ib_insync import Forex
        import logging as _log
        _ib_log = _log.getLogger("ib_insync.wrapper")
        _prev = _ib_log.level
        _ib_log.setLevel(_log.ERROR)   # suppress Error 200 spam during Forex qualify
        try:
            for symbol, inverse in [(f"{from_ccy}{to_ccy}", False),
                                     (f"{to_ccy}{from_ccy}", True)]:
                try:
                    contract = Forex(symbol)
                    conn.ib.qualifyContracts(contract)
                    if not contract.conId:
                        continue
                    td = conn.ib.reqMktData(contract, "", True, False)
                    conn.ib.sleep(2)
                    bid  = td.bid  if td.bid  and td.bid  > 0 else None
                    ask  = td.ask  if td.ask  and td.ask  > 0 else None
                    last = td.last if td.last and td.last > 0 else None
                    mid  = (bid + ask) / 2 if bid and ask else last
                    if mid and mid > 0:
                        return (1.0 / mid) if inverse else mid
                except Exception as e:
                    logger.debug("FX pair %s failed: %s", symbol, e)
        finally:
            _ib_log.setLevel(_prev)
        return None

    def build_price_scale(
        self,
        conn:         IBKRConnection,
        tickers:      List[str],
        universe_cfg,
    ) -> Dict[str, float]:
        """
        {ticker: factor} such that price_quoted × factor is the price expressed
        in the universe's base currency — for compute_target_shares.

        factor = (1 / minor_units_per_major) × fx(quote_currency → base)

        The minor unit and the quote currency come from the ticker's suffix
        (universe.ticker_market), which is static and needs no network call.
        The rate is asked of IBKR first and of the market-data layer second; a
        currency whose rate cannot be established yields NO entry, so its lines
        are left unsized rather than sized on an unconverted price.
        """
        from .universe import ticker_market
        from .data import _fx_rate as _offline_fx

        base = (getattr(universe_cfg, "currency", "") or "").upper()
        rates: Dict[str, Optional[float]] = {base: 1.0}
        out: Dict[str, float] = {}
        failed: Dict[str, List[str]] = {}

        for t in tickers:
            _sym, ccy, minor, _primary = ticker_market(t, universe_cfg)
            ccy = (ccy or base).upper()
            if ccy not in rates:
                r = None
                try:
                    r = self._fetch_fx_rate(conn, ccy, base)
                except Exception as e:
                    logger.debug("IBKR FX %s->%s failed: %s", ccy, base, e)
                if r is None or not (r > 0):
                    r = _offline_fx(ccy, base)
                    if r is not None and r > 0:
                        logger.info("FX %s->%s from market data (IBKR had none): %.6g",
                                    ccy, base, r)
                rates[ccy] = r if (r is not None and r > 0) else None
            rate = rates[ccy]
            if rate is None:
                failed.setdefault(ccy, []).append(t)
                continue
            if not (minor > 0):
                failed.setdefault(ccy, []).append(t)
                continue
            out[t] = rate / float(minor)

        if failed:
            for ccy, ts in failed.items():
                logger.error(
                    "[%s] No %s->%s rate — %d line(s) will NOT be sized: %s",
                    getattr(universe_cfg, "name", "?"), ccy, base, len(ts),
                    ", ".join(sorted(ts)[:10]),
                )
        return out

    def compute_target_shares(
        self,
        long_weights:    pd.Series,
        short_weights:   pd.Series,
        nav_usd:         float,
        cfg:             TaurusConfig,
        prices:          Dict[str, float],
        universe_cfg=None,
        price_scale:     Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Convert fractional weights → integer share counts.
        Long: positive shares. Short: negative shares.

        price_scale (multi-currency universes only): {ticker: factor} where
        price_quoted × factor is the price expressed in the NAV's own currency.
        For a London line in a EUR-based book that is (1/100) × GBP→EUR: the
        minor unit and the exchange rate in one number.  A ticker absent from
        the map is NOT sized — a line sized on an unconverted price is how
        CPG.L and IHG.L came out 74x too large.

        Left at None (every mono-currency universe) the old single nav_scale
        applies to the whole book and the arithmetic is unchanged.
        """
        half = cfg.gross_leverage / 2.0
        target: Dict[str, float] = {}

        fractional = getattr(universe_cfg, "fractional_shares", False) if universe_cfg else False

        if price_scale is None:
            # nav_scale converts NAV to match price units (LSE: GBP × 100 → GBp)
            nav_scale     = getattr(universe_cfg, "nav_scale", 1.0) if universe_cfg else 1.0
            effective_nav = nav_usd * nav_scale
            eff_price     = dict(prices)
            unscaled: List[str] = []
        else:
            # Prices are brought to the NAV's currency instead, so the NAV is
            # used as-is and nav_scale must stay 1.0 for such a universe.
            effective_nav = nav_usd
            eff_price = {}
            unscaled = []
            for t, p in prices.items():
                sc = price_scale.get(t)
                if sc is None or not (sc > 0):
                    unscaled.append(t)
                    continue
                eff_price[t] = p * sc

        # lot-size rounding for non-fractional markets
        lot  = getattr(cfg, "min_lot_size", 1) or 1
        ulot = getattr(universe_cfg, "min_lot_size", 1) if universe_cfg else 1
        lot  = max(lot, ulot)

        def _round_lot(shares: float) -> float:
            """Round DOWN to nearest lot (integer). Returns 0.0 if below 1 lot."""
            n = int(shares) // lot * lot
            return float(n)

        skipped = []
        for ticker, w in long_weights.items():
            price = eff_price.get(ticker, 0.0)
            if price > 0:
                dollars = w * effective_nav * half
                if fractional:
                    shares = round(dollars / price, 2)
                    if shares >= 0.01:
                        target[ticker] = shares
                    else:
                        skipped.append(ticker)
                else:
                    rounded = _round_lot(dollars / price)
                    if rounded > 0:
                        target[ticker] = rounded
                    else:
                        skipped.append(ticker)

        for ticker, w in short_weights.items():
            price = eff_price.get(ticker, 0.0)
            if price > 0:
                dollars = w * effective_nav * half
                if fractional:
                    shares = round(dollars / price, 2)
                    if shares >= 0.01:
                        target[ticker] = -shares
                    else:
                        skipped.append(ticker)
                else:
                    rounded = _round_lot(dollars / price)
                    if rounded > 0:
                        target[ticker] = -rounded
                    else:
                        skipped.append(ticker)

        if unscaled:
            logger.error(
                "%d ticker(s) had no price-scale entry and were NOT sized — a "
                "line sized on an unconverted price is how CPG.L and IHG.L came "
                "out 74x too large: %s",
                len(unscaled), ", ".join(sorted(unscaled)),
            )

        if skipped:
            if fractional:
                logger.warning(
                    "Skipped %d positions (allocation < 0.01 shares): %s",
                    len(skipped), skipped,
                )
            else:
                logger.warning(
                    "Skipped %d positions below min lot size (%d): %s",
                    len(skipped), lot, skipped,
                )

        return pd.Series(target, dtype=float)

    def get_pending_shares(self, conn: IBKRConnection) -> pd.Series:
        """
        Returns signed share counts for open entry orders (MKT/LMT) that are
        not yet filled.  TRAIL stop orders are excluded — they are protective,
        not entries.

        Used to avoid duplicate orders when restarting the algo on the same day
        before fills have settled.
        """
        try:
            trades = conn.ib.openTrades()
        except Exception:
            return pd.Series(dtype=float)

        ACTIVE = {"PreSubmitted", "Submitted", "PendingSubmit"}
        ENTRY_TYPES = {"MKT"}  # Only MKT = entry orders; LMT/STP are always protective orders in this system

        pending: dict = {}
        for trade in trades:
            if trade.orderStatus.status not in ACTIVE:
                continue
            if trade.order.orderType not in ENTRY_TYPES:
                continue   # skip TRAIL, STOP, etc.
            symbol = trade.contract.symbol
            qty    = trade.order.totalQuantity
            signed = qty if trade.order.action == "BUY" else -qty
            pending[symbol] = pending.get(symbol, 0) + signed

        if pending:
            logger.info(
                "Pending entry orders detected (%d tickers) — included in delta calc to avoid duplicates",
                len(pending),
            )
        return pd.Series(pending, dtype=float)

    def compute_order_delta(
        self,
        target_shares:  pd.Series,
        live_positions: pd.DataFrame,
        pending_shares: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns: ticker, current, target, delta.
        Filters out zero deltas.

        pending_shares: signed share counts for open but unfilled entry orders.
        Passing this prevents duplicate orders when relaunching on the same day.
        """
        if pending_shares is None:
            pending_shares = pd.Series(dtype=float)

        all_tickers = target_shares.index.union(
            live_positions.index if not live_positions.empty else pd.Index([])
        )
        records = []
        for ticker in all_tickers:
            filled   = float(live_positions.loc[ticker, "quantity"]) \
                       if ticker in live_positions.index else 0.0
            inflight = float(pending_shares.get(ticker, 0.0))
            current  = filled + inflight
            target   = float(target_shares.get(ticker, 0.0))
            delta    = target - current
            if abs(delta) >= 0.01:   # ignore dust (< 0.01 share)
                records.append({
                    "ticker":  ticker,
                    "current": current,
                    "target":  target,
                    "delta":   delta,
                })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values("delta", key=abs, ascending=False)


# --------------------------------------------------------------------------- #
#  Order manager                                                               #
# --------------------------------------------------------------------------- #

def _tse_tick(price: float) -> float:
    """Return the TSE Prime Market minimum tick size for a given price level."""
    if price <  1_000:  return 0.1
    if price <  3_000:  return 0.5
    if price <  5_000:  return 1.0
    if price < 10_000:  return 5.0
    if price < 30_000:  return 10.0
    if price < 50_000:  return 50.0
    if price < 100_000: return 100.0
    return 1_000.0


def _round_price(price: float, currency: str) -> float:
    """
    Round a STP/LMT price to the minimum tick size for the given currency.

    JPY (TSE): tick depends on price range — avoids Warning 110
      "Le prix n'est pas conforme à la variation minimum autorisée".
    All others: two decimal places (cents / pence).
    """
    if not price:
        return price
    if currency == "JPY":
        tick = _tse_tick(price)
        # round to nearest tick, then strip floating-point noise
        return round(round(price / tick) * tick, 10)
    return round(price, 2)


# --------------------------------------------------------------------------- #

class OrderManager:
    """Places and monitors orders via ib_insync."""

    def get_shortable_shares(
        self,
        conn:         IBKRConnection,
        tickers:      List[str],
        universe_cfg: UniverseConfig,
    ) -> Dict[str, float]:
        """
        Borrow availability per name, via IBKR generic tick 236.

        Returns {ticker: shortable shares}.  A name absent from the result has
        no usable borrow quote; IBKR rejects or queues a short with no locate,
        so shorting it silently fails and leaves the book long-biased.

        NaN is treated as unknown (not zero) — the caller decides whether to
        trust an unknown, so a missing data subscription cannot wipe the whole
        short leg.
        """
        from ib_insync import Stock

        out: Dict[str, float] = {}
        if not tickers:
            return out

        contracts, tmap = [], {}
        for t in tickers:
            c = _stock_contract(t, universe_cfg)
            contracts.append(c)
            tmap[c.symbol] = t

        _wrap = logging.getLogger("ib_insync.wrapper")
        _lvl = _wrap.level
        _wrap.setLevel(logging.ERROR)
        qualified = []
        for c in contracts:
            try:
                conn.ib.qualifyContracts(c)
                if c.conId:
                    qualified.append(c)
            except Exception:
                pass
        _wrap.setLevel(_lvl)

        # Fall back to DELAYED data before asking.
        #
        # A paper account without the API market-data subscription answers
        # every tick-236 request with error 10089 — "requires an additional
        # subscription for API connections" — and then floods the log with
        # error 300 as the never-established requests are cancelled. The borrow
        # column then reads UNKNOWN for every short, which is the one answer it
        # must not give by default. IBKR's own message says delayed data is
        # available, and shortable-share counts do not need to be real time: a
        # locate that existed fifteen minutes ago is the information wanted.
        #
        # reqMarketDataType(3) = delayed, (4) = delayed frozen. Live is tried
        # first and the fallback only matters where live is refused.
        try:
            conn.ib.reqMarketDataType(3)
        except Exception as e:
            logger.debug("Could not switch to delayed market data: %s", e)

        tds = []
        for c in qualified:
            try:
                tds.append((c, conn.ib.reqMktData(c, "236", False, False)))
            except Exception as e:
                logger.debug("shortable request failed for %s: %s", c.symbol, e)
        conn.ib.sleep(4.0)          # 236 arrives asynchronously; delayed is slower

        for c, td in tds:
            orig = tmap.get(c.symbol, c.symbol)
            val = getattr(td, "shortableShares", float("nan"))
            try:
                conn.ib.cancelMktData(c)
            except Exception:
                pass
            if val is None or val != val:        # NaN -> unknown
                continue
            out[orig] = float(val)
        return out

    def get_live_prices(
        self,
        conn:         IBKRConnection,
        tickers:      List[str],
        universe_cfg: UniverseConfig,
    ) -> Dict[str, float]:
        """Request snapshot mid prices for position sizing."""
        from ib_insync import Stock
        from datetime import datetime, time as dtime
        from zoneinfo import ZoneInfo

        # Use live data during market hours, delayed otherwise
        now_et = datetime.now(ZoneInfo("America/New_York"))
        market_open = (
            now_et.weekday() < 5 and
            dtime(9, 30) <= now_et.time() <= dtime(16, 0)
        )
        conn.ib.reqMarketDataType(1 if market_open else 2)

        prices: Dict[str, float] = {}
        contracts = []
        # Map ibkr_ticker → original ticker for price dict
        ticker_map: Dict[str, str] = {}
        for ticker in tickers:
            c = _stock_contract(ticker, universe_cfg)
            contracts.append(c)
            ticker_map[c.symbol] = ticker

        # Qualify individually so one failed contract (Error 200) doesn't block others.
        # Suppress ib_insync's internal Error 200 warning during qualify — we handle failures ourselves.
        _ib_wrapper_log = logging.getLogger("ib_insync.wrapper")
        _prev_level = _ib_wrapper_log.level
        _ib_wrapper_log.setLevel(logging.ERROR)
        qualified_contracts = []
        for c in contracts:
            try:
                conn.ib.qualifyContracts(c)
                if c.conId:
                    qualified_contracts.append(c)
                else:
                    logger.debug("Could not qualify %s (%s/%s) — skipping price fetch",
                                 c.symbol, c.exchange, c.currency)
            except Exception as e:
                logger.debug("qualifyContracts skipped %s: %s", c.symbol, e)
        _ib_wrapper_log.setLevel(_prev_level)

        tickers_data = []
        for c in qualified_contracts:
            td = conn.ib.reqMktData(c, "", True, False)  # snapshot
            tickers_data.append((c.symbol, td))

        conn.ib.sleep(3)

        for symbol, td in tickers_data:
            bid = td.bid if td.bid and td.bid > 0 else float("nan")
            ask = td.ask if td.ask and td.ask > 0 else float("nan")
            last = td.last if td.last and td.last > 0 else float("nan")
            # Mid price or last; store under original ticker key
            orig = ticker_map.get(symbol, symbol)
            if bid == bid and ask == ask:
                prices[orig] = (bid + ask) / 2.0
            elif last == last:
                prices[orig] = last

        return prices

    def cancel_protective_orders(
        self,
        conn:     IBKRConnection,
        ticker:   str,
        dry_run:  bool = False,
        symbol:   str  = None,
    ) -> list:
        """
        Cancel all open STP/LMT protective orders for a ticker before rebalancing.
        Called before placing an adjustment order so stale stop/limit orders
        (sized for the old position) don't persist alongside the new ones.
        """
        PROTECTIVE = {"STP", "LMT"}
        ACTIVE     = {"PreSubmitted", "Submitted", "PendingSubmit"}
        # The Yahoo stem is not always IBKR's symbol: London files Rolls-Royce
        # as "RR.", so stripping ".L" leaves "RR" and a stale stop on it would
        # never be matched. Callers that already hold a resolved contract pass
        # its symbol.
        ibkr_ticker = symbol or (ticker.split(".")[0] if "." in ticker else ticker)
        cancelled: list = []
        try:
            for trade in conn.ib.openTrades():
                if trade.contract.symbol != ibkr_ticker:
                    continue
                if trade.order.orderType not in PROTECTIVE:
                    continue
                if trade.orderStatus.status not in ACTIVE:
                    continue
                if not dry_run:
                    conn.ib.cancelOrder(trade.order)
                cancelled.append(trade.order.orderId)
        except Exception as e:
            logger.warning("cancel_protective_orders failed for %s: %s", ticker, e)
        if cancelled:
            logger.info(
                "Cancelled %d stale protective orders for %s before rebalance: %s",
                len(cancelled), ticker, cancelled,
            )
        return cancelled

    def cancel_all_orders_for_tickers(
        self,
        conn:     IBKRConnection,
        tickers:  set,
        dry_run:  bool = False,
    ) -> int:
        """
        Cancel ALL open orders (entry MKT + protective STP/LMT) for a set of tickers.

        Called at the start of every rebalance pass to guarantee a clean slate.
        DAY entry orders from a previous run expire overnight but successive intra-day
        runs can also accumulate duplicates when pending_shares misses cancelled/filled
        orders.  Wiping every open order for managed tickers and recomputing delta
        purely from filled positions is the only reliable way to avoid duplicates.
        """
        ACTIVE = {"PreSubmitted", "Submitted", "PendingSubmit"}
        ibkr_tickers = {t.split(".")[0] if "." in t else t for t in tickers}
        n_cancelled = 0
        try:
            for trade in conn.ib.openTrades():
                if trade.contract.symbol not in ibkr_tickers:
                    continue
                if trade.orderStatus.status not in ACTIVE:
                    continue
                if not dry_run:
                    conn.ib.cancelOrder(trade.order)
                n_cancelled += 1
        except Exception as e:
            logger.warning("cancel_all_orders_for_tickers: %s", e)
        if n_cancelled:
            logger.info(
                "Cancelled %d stale orders for %d managed tickers (clean-slate before rebalance)",
                n_cancelled, len(ibkr_tickers),
            )
        return n_cancelled

    def place_protective_only(
        self,
        conn:            IBKRConnection,
        ticker:          str,
        target_qty:      float,            # signed: positive=long, negative=short
        entry_price:     float,
        universe_cfg:    UniverseConfig,
        stop_pct:        float = 0.10,
        take_profit_pct: float = 0.20,
        dry_run:         bool  = False,
    ) -> None:
        """
        Disabled. The book holds no stops and no take-profits.

        It used to place a standalone STP + LMT pair on an existing position.
        Keeping the function, inert, rather than deleting it leaves its two
        callers working and means nothing in this module can place a
        protective order: every path through it now returns here.
        """
        logger.debug(
            "place_protective_only(%s): protective orders are not placed on "
            "this book — nothing done.", ticker)
        return

    def place_order(
        self,
        conn:             IBKRConnection,
        ticker:           str,
        delta:            float,            # positive=buy, negative=sell/short (fractional ok)
        universe_cfg:     UniverseConfig,
        dry_run:          bool  = True,
        trail_pct:        float = 0.10,     # ignored, kept for callers
        entry_price:      float = 0.0,      # ignored, kept for callers
        take_profit_pct:  float = 0.20,     # ignored, kept for callers
        target_qty:       float = None,     # ignored, kept for callers
        contract                = None,     # pre-resolved contract, if any
    ) -> Optional[dict]:
        """
        Place a plain market order. No stop, no take-profit, on any market.

        The account is rebalanced monthly and holds no protective orders by
        decision, so none are placed here and there is no option to place
        them.  What they used to cost is worth recording: on US names the
        three went out as an IBKR bracket, market order first with
        transmit=False, and only the take-profit — placed third — transmitted
        all of them.  A stop or target the exchange refused therefore left the
        MARKET order in PreSubmitted, unsent, after this function had already
        reported "submitted".  A book reported as rebalanced had not moved.

        Stale protective orders are still cancelled before the order goes
        out: a stop left from an earlier run is sized for a position that no
        longer exists, and clearing it is the point.

        trail_pct, entry_price, take_profit_pct and target_qty are kept so
        that existing callers still work. They no longer do anything.
        """
        from ib_insync import MarketOrder

        action = "BUY" if delta > 0 else "SELL"
        qty    = abs(delta)

        order_info = {
            "ticker":   ticker,
            "action":   action,
            "quantity": qty,
            "type":     "MKT",
            "dry_run":  dry_run,
            "status":   "pending",
        }

        if dry_run:
            logger.info("[DRY RUN] %s %d %s @ MKT", action, qty, ticker)
            order_info["status"] = "dry_run"
            return order_info

        if conn.is_live and not conn.cfg.live_trading:
            logger.error("Live port but live_trading=False. Aborting order.")
            order_info["status"] = "aborted"
            return order_info

        try:
            # A contract the caller has already resolved is used as it stands.
            # The point is that what was checked is what gets sent: resolving
            # again here could pick a different route than the one verified,
            # and the old fallback "retry with SMART" was no fallback at all —
            # _stock_contract already builds the contract with exchange=SMART,
            # so it retried the identical spec and reported the same failure.
            if contract is None:
                contract, route = resolve_contract(conn, ticker, universe_cfg)
                if contract is None:
                    logger.error("Cannot resolve %s at IBKR (%s) — not sent.",
                                 ticker, route)
                    order_info["status"] = "unresolved"
                    order_info["error"]  = route
                    return order_info

            # Stale protective orders are cleared now that the contract is
            # resolved, so the cancel matches on IBKR's own symbol rather than
            # the Yahoo stem. A stop left from an earlier run is sized for a
            # position that no longer exists.
            self.cancel_protective_orders(conn, ticker, dry_run=False,
                                          symbol=contract.symbol)

            # Plain market order, transmitted on the spot.
            #
            # There used to be a bracket here: the market order was sent with
            # transmit=False and only the take-profit child, placed third,
            # transmitted the three together.  A stop or a target the exchange
            # refused therefore left the MARKET order sitting in PreSubmitted
            # — unsent, while this function had already returned "submitted".
            # A book reported as rebalanced had not moved.
            #
            # The account is rebalanced monthly and holds no stops by
            # decision, so there is nothing to weigh up: the orders are gone,
            # and with them the only path on which a market order could fail
            # to transmit.  Stale protective orders are still cancelled above
            # — removing a stop left over from an earlier run is the point.
            ordre          = MarketOrder(action, qty)
            ordre.tif      = "DAY"
            ordre.transmit = True

            trade = conn.ib.placeOrder(contract, ordre)
            order_info["ibkr_order_id"] = trade.order.orderId
            order_info["status"]        = "submitted"
            logger.info("MKT submitted: %s %d %s | id=%d",
                        action, qty, ticker, trade.order.orderId)

            return order_info
        except Exception as e:
            logger.error("Order failed for %s: %s", ticker, e)
            order_info["status"] = "error"
            order_info["error"]  = str(e)
            return order_info

    def place_futures_hedge(
        self,
        conn:         IBKRConnection,
        futures_weight: float,
        nav_usd:      float,
        universe_cfg: UniverseConfig,
        dry_run:      bool = True,
    ) -> Optional[dict]:
        """Sell/buy index futures to neutralize net portfolio beta."""
        from ib_insync import Future, MarketOrder

        if abs(futures_weight) < 0.01:
            return None   # too small to hedge

        # Approximate number of contracts
        from .universe import REGISTRY
        udef          = REGISTRY.get(universe_cfg.name)
        local_symbol  = udef.get_futures_local_symbol()
        notional      = futures_weight * nav_usd
        price_approx  = 5500.0    # rough ES level — will be corrected by live price
        n_contracts   = abs(int(notional / (universe_cfg.futures_multiplier * price_approx)))
        n_contracts   = max(1, n_contracts)
        action        = "SELL" if futures_weight < 0 else "BUY"

        order_info = {
            "ticker":     local_symbol,
            "action":     action,
            "quantity":   n_contracts,
            "type":       "MKT",
            "futures":    True,
            "dry_run":    dry_run,
            "status":     "pending",
        }

        if dry_run:
            logger.info("[DRY RUN FUTURES] %s %d %s", action, n_contracts, local_symbol)
            order_info["status"] = "dry_run"
            return order_info

        try:
            contract = Future(
                localSymbol=local_symbol,
                exchange=universe_cfg.futures_exchange,
                currency=universe_cfg.futures_currency,
            )
            conn.ib.qualifyContracts(contract)
            order = MarketOrder(action, n_contracts)
            trade = conn.ib.placeOrder(contract, order)
            order_info["ibkr_order_id"] = trade.order.orderId
            order_info["status"]        = "submitted"
            logger.info("Futures order: %s %d %s (id=%d)", action, n_contracts, local_symbol, trade.order.orderId)
            return order_info
        except Exception as e:
            logger.error("Futures order failed: %s", e)
            order_info["status"] = "error"
            order_info["error"]  = str(e)
            return order_info

    def wait_for_fills(
        self,
        conn:    IBKRConnection,
        timeout: int = 300,
    ) -> None:
        """Wait for all open orders to fill or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            open_orders = conn.ib.openOrders()
            if not open_orders:
                logger.info("All orders filled.")
                return
            conn.ib.sleep(5)
        logger.warning("Fill timeout after %ds. %d orders still open.", timeout, len(conn.ib.openOrders()))


# --------------------------------------------------------------------------- #
#  Execution report                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class ExecutionReport:
    universe:       str
    rebalance_date: pd.Timestamp
    executed_at:    datetime
    orders:         List[dict]         = field(default_factory=list)
    futures_order:  Optional[dict]     = None
    errors:         List[str]          = field(default_factory=list)
    dry_run:        bool               = True

    def save(self, output_dir: str = "output") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{output_dir}/execution_{self.universe}_{self.rebalance_date.date()}.json"
        with open(fname, "w") as f:
            data = {
                "universe":       self.universe,
                "rebalance_date": str(self.rebalance_date.date()),
                "executed_at":    self.executed_at.isoformat(),
                "dry_run":        self.dry_run,
                "orders":         self.orders,
                "futures_order":  self.futures_order,
                "errors":         self.errors,
            }
            json.dump(data, f, indent=2)
        logger.info("ExecutionReport saved → %s", fname)
        return fname


# --------------------------------------------------------------------------- #
#  High-level executor facade                                                  #
# --------------------------------------------------------------------------- #

class IBKRExecutor:
    """
    Orchestrates one full rebalance: reconcile → price → size → order → report.
    Entry point called by the scheduler.
    """

    def __init__(
        self,
        conn:         IBKRConnection,
        cfg:          TaurusConfig,
        universe_cfg: UniverseConfig,
        output_dir:   str = "output",
    ):
        self.conn         = conn
        self.cfg          = cfg
        self.udef_cfg     = universe_cfg
        self.output_dir   = output_dir
        self.reconciler   = PositionReconciler()
        self.order_mgr    = OrderManager()


    def _get_available_cash(self) -> float:
        """
        Return available funds expressed in this universe's currency.

        First tries the exact currency match from IBKR (works when the account
        already holds that currency).  If not found (e.g. no JPY cash before
        the first NIKKEI rebalance), falls back to the base-currency total and
        converts via a live FX quote so the result is always in the correct
        units for the order_value comparison in _handle_sub_lot.
        """
        currency = self.udef_cfg.currency
        try:
            summary = self.conn.ib.accountSummary()
        except Exception:
            return 0.0

        # 1. Exact match
        for item in summary:
            if item.tag == "AvailableFunds" and item.currency == currency:
                return float(item.value)

        # 2. Base-currency total → FX convert to universe currency
        base_val = None
        base_ccy = None
        for item in summary:
            if item.tag == "AvailableFunds" and item.currency not in ("BASE", ""):
                base_val = float(item.value)
                base_ccy = item.currency
                break

        if base_val is None:
            return 0.0
        if base_ccy == currency:
            return base_val

        rate = PositionReconciler._fetch_fx_rate(self.conn, base_ccy, currency)
        if rate and rate > 0:
            converted = base_val * rate
            logger.debug(
                "[%s] AvailableFunds: %.2f %s × %.4f = %.2f %s",
                self.udef_cfg.name, base_val, base_ccy, rate, converted, currency,
            )
            return converted

        # 3. No FX rate → return 0 so sub-lot positions go to manual report
        logger.warning(
            "[%s] Could not convert AvailableFunds %s→%s — sub-lot rescue disabled",
            self.udef_cfg.name, base_ccy, currency,
        )
        return 0.0

    def _handle_sub_lot(
        self,
        target:      "pd.Series",
        snapshot,
        prices:      dict,
        nav_budget:  float = None,
    ) -> "tuple[pd.Series, list]":
        """
        Handle positions where the strategy weight converts to < 1 share.

        Rules:
          1. If (1 lot × price) ≤ remaining universe budget → round up to 1 lot.
          2. Otherwise → add to manual-orders list (report saved separately).

        nav_budget: the universe's NAV allocation in local currency (already
        multiplied by nav_fraction).  Sub-lot round-ups are capped to this budget
        so an expensive stock (e.g. Fast Retailing at ¥50 000 × 100 lot = ¥5 M)
        cannot consume more than its fair share of the account.
        If not provided, falls back to account available cash (old behaviour).

        Returns (updated_target, manual_rows).
        """
        import pandas as pd
        from .risk import RiskConfig as _RC
        risk_cfg = _RC()

        def norm(t: str) -> str:
            return t.split(".")[0] if "." in t else t

        # Build {normalised_ticker: (weight, is_long)} for all weighted tickers
        all_weighted: dict = {}
        for t, w in snapshot.long_weights.items():
            all_weighted.setdefault(norm(t), (w, True))
        for t, w in snapshot.short_weights.items():
            all_weighted.setdefault(norm(t), (w, False))

        # Sub-lot = in weights but not in target (rounded to 0 by _round_lot)
        sub_lot = {
            ticker: info
            for ticker, info in all_weighted.items()
            if ticker not in target.index and prices.get(ticker, 0) > 0
        }

        if not sub_lot:
            return target, []

        lot        = getattr(self.udef_cfg, "min_lot_size", 1) or 1
        avail_cash = self._get_available_cash()

        # Cap rescue budget: min(account available cash, universe NAV budget).
        # This prevents expensive stocks (e.g. Nikkei 100-lot min) from consuming
        # far more than their intended portfolio weight just because 1 lot is large.
        # The universe NAV budget (gross_leverage × nav_budget) is the theoretical
        # maximum deployment for this universe; sub-lot round-ups stay within it.
        if nav_budget and nav_budget > 0:
            universe_cap = nav_budget * self.cfg.gross_leverage
            remaining    = min(avail_cash, universe_cap)
        else:
            remaining = avail_cash
        roundup: dict = {}
        manual: list  = []

        for ticker, (w, is_long) in sub_lot.items():
            price         = prices[ticker]
            order_value   = price * lot

            if order_value <= remaining:
                # Affordable — round up to 1 lot
                roundup[ticker] = float(lot) if is_long else float(-lot)
                remaining -= order_value
                logger.info(
                    "[%s] Sub-lot %s: rounded up to %d share(s) (%.2f %s; cash left %.2f)",
                    self.udef_cfg.name, ticker, lot, order_value,
                    self.udef_cfg.currency, remaining,
                )
            else:
                # Not affordable — flag for manual report
                if is_long:
                    stp = _round_price(price * (1 - risk_cfg.stop_loss_pct),  self.udef_cfg.currency)
                    lmt = _round_price(price * (1 + risk_cfg.take_profit_pct), self.udef_cfg.currency)
                    action = "BUY"
                else:
                    stp = _round_price(price * (1 + risk_cfg.stop_loss_pct),  self.udef_cfg.currency)
                    lmt = _round_price(price * (1 - risk_cfg.take_profit_pct), self.udef_cfg.currency)
                    action = "SELL"
                manual.append({
                    "ticker":      ticker,
                    "action":      action,
                    "qty":         lot,
                    "price":       round(price, 4),
                    "stop_loss":   stp,
                    "take_profit": lmt,
                    "currency":    self.udef_cfg.currency,
                    "universe":    self.udef_cfg.name,
                    "reason":      (
                        f"Fractional: need {order_value:.2f} {self.udef_cfg.currency}, "
                        f"only {remaining:.2f} available"
                    ),
                })
                logger.warning(
                    "[%s] Sub-lot %s: MANUAL ORDER NEEDED — %.2f %s required, %.2f available",
                    self.udef_cfg.name, ticker, order_value, self.udef_cfg.currency, remaining,
                )

        if roundup:
            target = pd.concat([target, pd.Series(roundup, dtype=float)])

        return target, manual

    def _save_fractional_report(self, rows: list, as_of) -> None:
        """Write a human-readable + JSON report of orders that need manual execution."""
        import json, os
        date_str = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)[:10]
        base = os.path.join(
            self.output_dir,
            f"manual_orders_{self.udef_cfg.name}_{date_str}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # ── Human-readable text ──────────────────────────────────────────────
        lines = [
            f"MANUAL FRACTIONAL ORDERS — {self.udef_cfg.name.upper()} @ {date_str}",
            "=" * 60,
            "These positions could not be placed automatically:",
            "  • allocation < 1 share at current prices AND",
            "  • insufficient cash to round up to 1 share",
            "",
            f"  {'ACTION':<5}  {'QTY':>4}  {'TICKER':<10}  {'PRIX':>10}  {'STOP':>10}  {'LIMIT':>10}  {'CCY'}",
            "  " + "-" * 58,
        ]
        for r in rows:
            lines.append(
                f"  {r['action']:<5}  {r['qty']:>4}  {r['ticker']:<10}"
                f"  {r['price']:>10.4f}  {r['stop_loss']:>10.4f}  {r['take_profit']:>10.4f}"
                f"  {r['currency']}"
            )
            lines.append(f"         ↳ {r['reason']}")
            lines.append("")

        txt_path = base + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # ── JSON (machine-readable) ──────────────────────────────────────────
        json_path = base + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)

        logger.info(
            "[%s] %d manual order(s) saved → %s (.txt / .json)",
            self.udef_cfg.name, len(rows), base,
        )

    def _restore_protective_orders(self, prices: dict, mm_tp_pcts: dict = None) -> None:
        """
        Disabled. The book holds no stops and no take-profits.

        This was the post-rebalance pass that gave every live position a
        fresh STP + LMT. It is kept inert so its two call sites in
        execute_rebalance still work, and so that no path in this module
        places a protective order.

        Clearing stale ones is NOT done here: place_order cancels a
        position's leftovers before it trades it, and --mode cancel-orders
        wipes the account outright.
        """
        logger.info("Protective orders are not placed on this book — "
                    "post-rebalance restore skipped.")
        return

    def _log_reconciliation(
        self,
        target:    pd.Series,
        delta_df:  pd.DataFrame,
        report:    "ExecutionReport",
    ) -> None:
        """
        After orders are placed, log a full reconciliation summary:
        - Submitted orders: ticker, action, qty, IBKR status
        - Orders that were cancelled/errored
        - Positions already at target (no rebalancing needed)
        Stored in report.orders as 'reconciliation' entries.
        """
        try:
            # Current open trades and fills from IBKR
            open_trades = self.conn.ib.openTrades()
            open_by_id  = {t.order.orderId: t for t in open_trades}

            # Re-read live positions after fills
            live_now = self.reconciler.get_live_positions(self.conn, currency=self._position_currency())

            lines = [
                f"\n{'='*60}",
                f"RECONCILIATION — {self.udef_cfg.name.upper()} @ {datetime.now().strftime('%H:%M:%S')}",
                f"{'='*60}",
            ]

            submitted_ok  = 0
            cancelled_err = 0

            for o in report.orders:
                ticker  = o.get("ticker", "?")
                action  = o.get("action", "?")
                qty     = o.get("quantity", 0)
                status  = o.get("status", "?")
                oid     = o.get("ibkr_order_id")

                # Check live IBKR status if we have an order ID
                live_status = status
                if oid and oid in open_by_id:
                    live_status = open_by_id[oid].orderStatus.status
                elif oid:
                    # Not in open trades → likely filled or cancelled
                    live_status = "Filled/Cancelled"

                # "Filled/Cancelled" = internal label for orders no longer in openTrades (likely filled)
                # Only treat as error if truly Cancelled (not Filled/Cancelled) or status is error/aborted
                is_error = (live_status == "Cancelled") or (status in ("error", "aborted"))
                if is_error:
                    cancelled_err += 1
                    icon = "✗"
                else:
                    submitted_ok += 1
                    icon = "✓"

                stp = o.get("stop_price")
                lmt = o.get("tp_price")
                line = (
                    f"  {icon} {action:4s} {int(qty):5d} {ticker:<12s}"
                    f"  STP={stp:.2f}  LMT={lmt:.2f}"
                    f"  [{live_status}]"
                ) if stp and lmt else (
                    f"  {icon} {action:4s} {int(qty):5d} {ticker:<12s}  [{live_status}]"
                )
                lines.append(line)

            # Positions already at target (no delta needed)
            tickers_with_orders = set(delta_df["ticker"].tolist()) if not delta_df.empty else set()
            positions_at_target = []
            for ticker in target.index:
                if ticker not in tickers_with_orders:
                    qty_live = float(live_now.loc[ticker, "quantity"]) if ticker in live_now.index else 0
                    positions_at_target.append(f"    {ticker:<12s}  qty={qty_live:.0f}  (no rebalancing needed)")

            if positions_at_target:
                lines.append(f"\n  Already at target ({len(positions_at_target)} positions):")
                lines.extend(positions_at_target)

            lines.append(f"\n  SUMMARY: {submitted_ok} submitted OK | {cancelled_err} cancelled/error")
            lines.append(f"  Live positions after rebalance: {len(live_now)} total")
            lines.append(f"{'='*60}")

            summary = "\n".join(lines)
            logger.info(summary)

            # Store reconciliation in report
            report.orders.append({
                "type":           "reconciliation_summary",
                "submitted_ok":   submitted_ok,
                "cancelled_err":  cancelled_err,
                "at_target_count": len(positions_at_target),
                "live_positions": len(live_now),
            })

        except Exception as e:
            logger.warning("Reconciliation failed: %s", e)

    def _position_currency(self) -> Optional[str]:
        """
        Currency to filter live positions by, or None.

        A single currency isolates two same-currency universes from each other
        (sp500 and nasdaq100 are both USD).  A multi-currency universe has no
        single currency to filter on — filtering would hide its London lines
        from a EUR-based book — so it filters on nothing here and relies on the
        ticker-list and RiskState isolation applied further down.
        """
        if getattr(self.udef_cfg, "multi_currency", False):
            return None
        return self.udef_cfg.currency

    def execute_rebalance(
        self,
        snapshot,
        nav_fraction: float = 1.0,
        yf_prices: Optional[Dict[str, float]] = None,
    ) -> ExecutionReport:
        """
        Full rebalance pipeline:
        1. Get live NAV and positions
        2. Get live prices for all target stocks (IBKR snapshot, fallback to yf_prices)
        3. Compute target share counts
        4. Compute delta vs live positions
        5. Place exits first, then entries
        6. Place futures hedge
        7. Save and return ExecutionReport

        yf_prices: optional dict {ticker: last_close_price} from yfinance (already
                   downloaded by strategy.load_data()).  Used as fallback when IBKR
                   market data subscription is missing (e.g. LSE, TSEJ on paper
                   account).  Tickers missing from IBKR prices but present in
                   yf_prices will use the yfinance close.
        """
        report = ExecutionReport(
            universe=self.udef_cfg.name,
            rebalance_date=snapshot.date,
            executed_at=datetime.now(),
            dry_run=self.cfg.dry_run,
        )

        if not self.conn.ensure_connected():
            report.errors.append("IBKR not connected")
            return report

        try:
            # 1. NAV in universe's local currency (scaled by Sharpe-weighted fraction)
            nav = self.reconciler.get_account_nav(
                self.conn, currency=self.udef_cfg.currency
            ) * nav_fraction
            logger.info(
                "[%s] NAV = %.2f %s (fraction=%.1f%%)",
                self.udef_cfg.name, nav, self.udef_cfg.currency, nav_fraction * 100,
            )

            # 2. Live prices (IBKR snapshot), with yfinance fallback for missing tickers
            all_tickers = list(snapshot.long_weights.index) + list(snapshot.short_weights.index)
            prices = self.order_mgr.get_live_prices(self.conn, all_tickers, self.udef_cfg)

            # Fill any tickers that had no IBKR market data (no subscription / market closed)
            #
            # The fallback price must be checked for its quote UNIT before it
            # reaches compute_target_shares, which divides a NAV already scaled
            # by nav_scale (100 on the LSE) by it.  Yahoo quotes CPG.L and
            # IHG.L in USD while the rest of the LSE comes back in GBp, and a
            # ~180 USD quote where ~8500 pence is expected sizes the position
            # about 47x too large.  normalise_quote_prices rescales a mismatch
            # and drops any name whose unit it cannot establish.
            #
            # Each price is converted exactly ONCE, by whoever knows its unit:
            # an IBKR price by build_price_scale (the suffix gives the listing's
            # currency, which is what IBKR quotes in), a yfinance price here by
            # normalise_quote_prices (Yahoo's own currency field is the
            # authority for a Yahoo price, and the two can disagree — the suffix
            # says CPG.L is GBp, Yahoo says USD).  Prices filled here are
            # therefore already in the base currency and are pinned to a scale
            # of 1.0 below, so price_scale cannot convert them a second time.
            yf_filled: set = set()
            if yf_prices:
                missing = [t for t in all_tickers if t not in prices or prices[t] <= 0]
                candidates = {t: yf_prices[t] for t in missing if yf_prices.get(t, 0) > 0}
                usable, unusable = normalise_quote_prices(candidates, self.udef_cfg)
                for t, px in usable.items():
                    prices[t] = px
                yf_filled = set(usable)
                if usable:
                    logger.info(
                        "[%s] Used yfinance close prices for %d tickers (no IBKR data): %s",
                        self.udef_cfg.name, len(usable), sorted(usable),
                    )
                if unusable:
                    logger.error(
                        "[%s] %d ticker(s) left UNPRICED — not sized rather than "
                        "sized on an unverified quote unit: %s",
                        self.udef_cfg.name, len(unusable),
                        ", ".join(f"{t} ({why})" for t, why in sorted(unusable.items())),
                    )

            # ── Borrow availability screen (short leg) ──────────────────── #
            # IBKR rejects or queues a short with no locate, so an unborrowable
            # name silently fails to fill and leaves the book long-biased.
            # Drop those names and REDISTRIBUTE their weight across the shorts
            # that can actually be borrowed, so the leg still deploys in full.
            long_w, short_w = snapshot.long_weights, snapshot.short_weights
            if getattr(self.cfg, "check_borrow_availability", False) and len(short_w):
                try:
                    avail = self.order_mgr.get_shortable_shares(
                        self.conn, list(short_w.index), self.udef_cfg
                    )
                    floor = float(getattr(self.cfg, "min_shortable_shares", 0.0))
                    # Unknown (ticker absent) is kept: a missing data subscription
                    # must not wipe the short leg.
                    blocked = [t for t in short_w.index
                               if t in avail and avail[t] <= floor]
                    if blocked:
                        kept = short_w.drop(index=blocked)
                        if kept.sum() > 0:
                            k_s = float(short_w.sum())
                            short_w = kept * (k_s / kept.sum())   # leg size preserved
                            logger.warning(
                                "[%s] %d short(s) not borrowable — weight redistributed "
                                "over %d names: %s",
                                self.udef_cfg.name, len(blocked), len(short_w),
                                ", ".join(f"{t}({avail[t]:.0f})" for t in blocked[:10]),
                            )
                        else:
                            logger.error(
                                "[%s] NO borrowable short available — short leg skipped.",
                                self.udef_cfg.name,
                            )
                            short_w = kept
                    n_unknown = sum(1 for t in short_w.index if t not in avail)
                    if n_unknown:
                        logger.info(
                            "[%s] Borrow availability unknown for %d short(s) "
                            "(no tick 236) — kept.", self.udef_cfg.name, n_unknown,
                        )
                except Exception as e:
                    logger.warning("[%s] Borrow check failed (%s) — proceeding without it.",
                                   self.udef_cfg.name, e)

            # 3. Target shares
            price_scale = None
            if getattr(self.udef_cfg, "multi_currency", False):
                if float(getattr(self.udef_cfg, "nav_scale", 1.0) or 1.0) != 1.0:
                    logger.error(
                        "[%s] multi_currency universe must have nav_scale = 1.0 "
                        "(prices are converted per line, the NAV is not scaled) "
                        "— refusing to size this run.",
                        self.udef_cfg.name,
                    )
                    report.save(self.output_dir)
                    return report
                price_scale = self.reconciler.build_price_scale(
                    self.conn, all_tickers, self.udef_cfg
                )
                # Already in the base currency — do not convert twice.
                for t in yf_filled:
                    price_scale[t] = 1.0

            target = self.reconciler.compute_target_shares(
                long_w, short_w,
                nav, self.cfg, prices, universe_cfg=self.udef_cfg,
                price_scale=price_scale,
            )

            # Normalise target tickers: strip Yahoo Finance exchange suffixes
            # (.L, .PA, .T, .HK, .SR, etc.) so they match IBKR position symbols,
            # which never carry exchange suffixes.  Without this, TSCO.L (target)
            # and TSCO (live) are treated as different stocks → close + reopen
            # instead of a simple adjustment.
            target.index = pd.Index([t.split(".")[0] if "." in t else t for t in target.index])
            prices        = {(t.split(".")[0] if "." in t else t): v for t, v in prices.items()}

            # 3b. Handle sub-lot positions (strategy weight → < 1 share)
            #     Round up to 1 lot if cash permits; otherwise write manual-orders report.
            #     Pass nav_budget so the rescue is capped at this universe's allocation.
            target, manual_rows = self._handle_sub_lot(target, snapshot, prices, nav_budget=nav)
            if manual_rows and not self.cfg.dry_run:
                self._save_fractional_report(manual_rows, snapshot.date)

            # 4. Delta vs live (filled positions + pending entry orders)
            live    = self.reconciler.get_live_positions(self.conn, currency=self._position_currency())
            pending = self.reconciler.get_pending_shares(self.conn)

            # Universe isolation: restrict live positions to tickers this universe
            # manages.  Without this, two same-currency universes (e.g. sp500 and
            # nasdaq100, both USD) contaminate each other — nasdaq100 sees sp500
            # positions not in its target and generates close orders for them.
            # We manage: (a) tickers currently in the target, plus (b) any ticker
            # previously opened by this universe (recorded in RiskState), so that
            # positions removed from the target are still closed gracefully.
            try:
                from .risk import RiskManager, RiskConfig as _RC
                _rm = RiskManager(_RC())
                universe_owned = {
                    t for t, p in _rm.state.positions.items()
                    if p.universe == self.udef_cfg.name
                }
                # Tickers positively attributed to a DIFFERENT universe — exclude these
                # so we don't accidentally close another universe's positions.
                other_universe_tickers = {
                    t for t, p in _rm.state.positions.items()
                    if p.universe != self.udef_cfg.name
                }
            except Exception:
                universe_owned = set()
                other_universe_tickers = set()

            # Include:
            #  (a) tickers in the new target
            #  (b) tickers previously opened by this universe (from RiskState)
            #  (c) live tickers in the right currency that are NOT owned by another
            #      universe — catches positions opened before RiskState tracking or
            #      after a rebalance that reset the state (e.g. the 100%-NAV run).
            untracked_live = set() if live.empty else (set(live.index) - other_universe_tickers)
            managed_tickers = set(target.index) | universe_owned | untracked_live
            if not live.empty:
                live = live[live.index.isin(managed_tickers)]

            # Cancel ALL open orders for managed tickers before computing delta.
            # This guarantees a clean slate on every run: DAY entry orders from a
            # previous session have expired so pending_shares won't see them, leading
            # to duplicate orders on a re-run.  By wiping stale orders first and
            # computing delta purely from FILLED positions we can never double-order.
            if not self.cfg.dry_run:
                self.order_mgr.cancel_all_orders_for_tickers(
                    self.conn, managed_tickers, dry_run=False
                )
                # Brief pause so cancellations propagate before we place new orders
                self.conn.ib.sleep(1.5)
            # After cancelling, pending_shares is empty for managed tickers —
            # pass an empty Series so delta = target - filled_positions only.
            pending_clean = pending.drop(
                index=[t for t in pending.index if t in managed_tickers],
                errors="ignore",
            )

            delta_df = self.reconciler.compute_order_delta(target, live, pending_clean)

            if delta_df.empty:
                logger.info("[%s] No trades needed — portfolio already at target.", self.udef_cfg.name)
                if not self.cfg.dry_run:
                    self._log_reconciliation(target, delta_df, report)
                    self._restore_protective_orders(prices, mm_tp_pcts)
                report.save(self.output_dir)
                return report

            # 5. Exits first (delta reduces position), then entries
            exits   = delta_df[delta_df.apply(
                lambda r: (r["current"] > 0 and r["delta"] < 0) or
                          (r["current"] < 0 and r["delta"] > 0), axis=1
            )]
            entries = delta_df[~delta_df.index.isin(exits.index)]

            from .risk import RiskConfig
            risk_cfg  = RiskConfig()
            trail_pct = risk_cfg.stop_loss_pct
            tp_pct    = risk_cfg.take_profit_pct

            # Take-profit: the FLAT level, for every name.
            #
            # This used to be derived from the MM divergence, clamped into
            # [5%, 80%]. Two things were wrong with that. The divergence does
            # not measure mispricing — it is identically
            # -(distress + agency)/market_cap, so it cannot say where fair
            # value is. And it is tiny: -0.00% to -1.32% on current data, so
            # the clamp pinned essentially every name to the 5% FLOOR. The
            # comment above it claimed a fallback to the flat tp_pct when the
            # divergence was near zero; the code clamped instead, and the
            # fallback never happened.
            #
            # A +5% take-profit against a -10% stop is 1:2 reward/risk,
            # inverted for a momentum strategy, and it fired constantly.
            # Measured by replaying ten years of DAILY paths through the real
            # exit rules:
            #
            #   S&P 500     no exits  +23.67%/yr  Sharpe 2.09  maxDD -6.70%
            #               -10/+5     +5.92%/yr        0.29        -8.01%
            #               -10/+20   +20.80%/yr        2.27        -5.33%
            #   NASDAQ 100  no exits  +10.91%/yr        0.66       -13.10%
            #               -10/+5     -0.17%/yr       -0.88       -17.81%
            #               -10/+20   +11.67%/yr        0.95        -6.39%
            #
            # Half the positions exited on the +5% take-profit (50% on the S&P
            # 500, 46% on the NASDAQ) against 17% on the stop. The flat 20%
            # level is not merely better than 5% — it beats holding to the next
            # rebalance on both Sharpe and drawdown.
            mm_tp_pcts: dict = {}

            for _, row in pd.concat([exits, entries]).reset_index(drop=True).iterrows():
                ticker        = str(row["ticker"])
                ticker_tp_pct = mm_tp_pcts.get(ticker, tp_pct)   # always the flat level now
                order_info = self.order_mgr.place_order(
                    self.conn, row["ticker"], float(row["delta"]),
                    self.udef_cfg, dry_run=self.cfg.dry_run,
                    trail_pct=trail_pct,
                    entry_price=prices.get(row["ticker"], 0.0),
                    take_profit_pct=ticker_tp_pct,
                    target_qty=float(row["target"]),  # full position size for STP/LMT sizing
                )
                if order_info:
                    report.orders.append(order_info)

            # 6. Futures beta hedge
            if self.cfg.use_futures_hedge and snapshot.futures_weight != 0.0:
                fut_order = self.order_mgr.place_futures_hedge(
                    self.conn, snapshot.futures_weight, nav,
                    self.udef_cfg, dry_run=self.cfg.dry_run,
                )
                report.futures_order = fut_order

            # 7. Wait for fills (skipped in dry run)
            if not self.cfg.dry_run:
                self.order_mgr.wait_for_fills(self.conn)

            # 7b. Restore / refresh protective orders for all live positions
            if not self.cfg.dry_run:
                self._restore_protective_orders(prices, mm_tp_pcts)

            # 7c. Reconciliation report — compare submitted orders vs live state
            if not self.cfg.dry_run:
                self._log_reconciliation(target, delta_df, report)

            # 8. Enregistrer les entry prices dans le RiskManager
            try:
                from .risk import RiskManager, RiskConfig
                risk_mgr = RiskManager(RiskConfig())
                risk_mgr.state.register_rebalance(
                    snapshot=snapshot,
                    prices=prices,
                    nav=nav,
                    universe=self.udef_cfg.name,
                )
            except Exception as e:
                logger.warning("RiskState update failed: %s", e)

        except Exception as e:
            logger.error("[%s] Execution error: %s", self.udef_cfg.name, e)
            report.errors.append(str(e))

        report.save(self.output_dir)
        return report
