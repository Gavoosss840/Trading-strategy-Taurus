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

logger = logging.getLogger(__name__)


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
        """Called before every operation. Reconnects with exponential backoff."""
        if self.ib.isConnected():
            return True
        waits = [60, 120, 240, 480, 900, 1800]   # seconds
        for wait in waits:
            logger.warning("IBKR disconnected. Reconnecting in %ds...", wait)
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

    def compute_target_shares(
        self,
        long_weights:    pd.Series,
        short_weights:   pd.Series,
        nav_usd:         float,
        cfg:             TaurusConfig,
        prices:          Dict[str, float],
        universe_cfg=None,
    ) -> pd.Series:
        """
        Convert fractional weights → integer share counts.
        Long: positive shares. Short: negative shares.
        """
        half = cfg.gross_leverage / 2.0
        target: Dict[str, float] = {}

        fractional = getattr(universe_cfg, "fractional_shares", False) if universe_cfg else False

        # nav_scale converts NAV to match price units (e.g. LSE: GBP × 100 → GBp pence)
        nav_scale     = getattr(universe_cfg, "nav_scale", 1.0) if universe_cfg else 1.0
        effective_nav = nav_usd * nav_scale

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
            price = prices.get(ticker, 0.0)
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
            price = prices.get(ticker, 0.0)
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
            ibkr_ticker = ticker.split(".")[0] if "." in ticker else ticker
            c = Stock(ibkr_ticker, universe_cfg.ibkr_exchange, universe_cfg.currency)
            contracts.append(c)
            ticker_map[ibkr_ticker] = ticker

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
    ) -> list:
        """
        Cancel all open STP/LMT protective orders for a ticker before rebalancing.
        Called before placing an adjustment order so stale stop/limit orders
        (sized for the old position) don't persist alongside the new ones.
        """
        PROTECTIVE = {"STP", "LMT"}
        ACTIVE     = {"PreSubmitted", "Submitted", "PendingSubmit"}
        ibkr_ticker = ticker.split(".")[0] if "." in ticker else ticker
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
        Place standalone STP + LMT protective orders for an EXISTING position.
        No parent MKT — these are independent GTC orders used when restoring
        protective orders after rebalancing adjustments or initial placement.
        Works on all exchanges (no bracket/OCA group → no Euronext Error 328).
        """
        from ib_insync import Stock, Order, LimitOrder

        ibkr_ticker = ticker.split(".")[0] if "." in ticker else ticker
        pos_qty     = abs(target_qty)
        stop_action = "SELL" if target_qty > 0 else "BUY"

        stop_price = (
            entry_price * (1 - stop_pct) if stop_action == "SELL"
            else entry_price * (1 + stop_pct)
        )
        tp_price = (
            entry_price * (1 + take_profit_pct) if stop_action == "SELL"
            else entry_price * (1 - take_profit_pct)
        )

        if dry_run:
            logger.info(
                "[DRY RUN] Restore STP/LMT %s %s qty=%.0f  STP=%.4f  LMT=%.4f",
                stop_action, ibkr_ticker, pos_qty,
                round(stop_price, 4), round(tp_price, 4),
            )
            return

        contract = Stock(ibkr_ticker, universe_cfg.ibkr_exchange, universe_cfg.currency)
        try:
            conn.ib.qualifyContracts(contract)
        except Exception as e:
            logger.warning("qualifyContracts failed for %s: %s — using unqualified contract", ibkr_ticker, e)

        stp_price_rounded = _round_price(stop_price, universe_cfg.currency)
        lmt_price_rounded = _round_price(tp_price,   universe_cfg.currency)

        # OCA group: when STP fills IBKR auto-cancels LMT and vice-versa.
        # Prevents zombie orders after the position is closed by one of them.
        oca_group = f"TRS_{ibkr_ticker}_{id(conn)}"

        # Standalone STP (GTC, OCA)
        stp               = Order()
        stp.action        = stop_action
        stp.orderType     = "STP"
        stp.tif           = "GTC"
        stp.totalQuantity = pos_qty
        stp.auxPrice      = stp_price_rounded
        stp.ocaGroup      = oca_group
        stp.ocaType       = 1   # cancel with block
        stp.transmit      = True
        try:
            stp_trade = conn.ib.placeOrder(contract, stp)
            conn.ib.sleep(0.5)   # let IBKR process before placing the second order
            if stp_trade.orderStatus.status == "Inactive":
                logger.error(
                    "STP order for %s was REJECTED by IBKR (Inactive) — price=%g",
                    ibkr_ticker, stp_price_rounded,
                )
            else:
                logger.info(
                    "STP placed for %s: qty=%.0f  price=%g (-%.0f%%)  status=%s",
                    ibkr_ticker, pos_qty, stp_price_rounded,
                    stop_pct * 100, stp_trade.orderStatus.status,
                )
        except Exception as e:
            logger.error("Failed to place STP for %s: %s", ibkr_ticker, e)

        # Standalone LMT (GTC, same OCA group)
        lmt           = LimitOrder(stop_action, pos_qty, lmt_price_rounded)
        lmt.tif       = "GTC"
        lmt.ocaGroup  = oca_group
        lmt.ocaType   = 1
        lmt.transmit  = True
        try:
            lmt_trade = conn.ib.placeOrder(contract, lmt)
            conn.ib.sleep(0.3)
            if lmt_trade.orderStatus.status == "Inactive":
                logger.error(
                    "LMT order for %s was REJECTED by IBKR (Inactive) — price=%g",
                    ibkr_ticker, lmt_price_rounded,
                )
            else:
                logger.info(
                    "LMT placed for %s: qty=%.0f  price=%g (+%.0f%%)  status=%s",
                    ibkr_ticker, pos_qty, lmt_price_rounded,
                    take_profit_pct * 100, lmt_trade.orderStatus.status,
                )
        except Exception as e:
            logger.error("Failed to place LMT for %s: %s", ibkr_ticker, e)

    def place_order(
        self,
        conn:             IBKRConnection,
        ticker:           str,
        delta:            float,            # positive=buy, negative=sell/short (fractional ok)
        universe_cfg:     UniverseConfig,
        dry_run:          bool  = True,
        trail_pct:        float = 0.10,     # stop loss % below/above entry (10%)
        entry_price:      float = 0.0,      # last known price for STP/LMT calculation
        take_profit_pct:  float = 0.20,     # take profit % above/below entry (20%)
        target_qty:       float = None,     # full target position size (unsigned); if None, uses abs(delta)
    ) -> Optional[dict]:
        """
        Place MKT + STP (stop loss) + LMT (take profit) bracket order.

        US markets (bracket_mode=True): all three transmit together as IBKR bracket.
          STP and LMT share the same parentId — IBKR auto-cancels one when the other fills.
        EU markets (bracket_mode=False): MKT transmits immediately; STP and LMT sent
          as standalone GTC orders (no parentId) so a rejection never freezes the MKT.

        trail_pct = 0.10 → stop follows the price, cuts if -10% from peak.
        """
        from ib_insync import Stock, MarketOrder, Order, LimitOrder

        action       = "BUY"  if delta > 0 else "SELL"
        qty          = abs(delta)

        # STP/LMT are sized to the FULL target position, not just the delta.
        # On an adjustment (+10 on a 100-share long), we cancel the old STP/LMT
        # (sized for 100) and place new ones sized for 110.
        protective_qty = abs(target_qty) if target_qty is not None else qty

        # Direction of protective orders depends on the final position direction
        # (target sign), not the delta direction.
        if target_qty is not None:
            stop_action = "SELL" if target_qty > 0 else "BUY"
        else:
            stop_action = "SELL" if delta > 0 else "BUY"

        # Cancel stale protective orders before placing new ones
        if not dry_run:
            self.cancel_protective_orders(conn, ticker, dry_run=False)

        # Fixed stop loss and take profit prices from entry
        stop_pct   = trail_pct
        stop_price = (entry_price * (1 - stop_pct)) if stop_action == "SELL" else (entry_price * (1 + stop_pct))
        tp_price   = (entry_price * (1 + take_profit_pct)) if stop_action == "SELL" else (entry_price * (1 - take_profit_pct))

        # For EU markets: MKT transmits independently; STP/LMT sent as standalone orders
        bracket_mode = getattr(universe_cfg, "supports_trail_stop", True)

        order_info = {
            "ticker":     ticker,
            "action":     action,
            "quantity":   qty,
            "type":       "MKT+STP+LMT",
            "stop_price": round(stop_price, 4) if stop_price else None,
            "tp_price":   round(tp_price, 4)   if tp_price   else None,
            "stop_pct":   stop_pct,
            "tp_pct":     take_profit_pct,
            "tp_source":  "flat" if abs(take_profit_pct - 0.20) < 0.001 else "MM",
            "dry_run":    dry_run,
            "status":     "pending",
        }

        if dry_run:
            logger.info(
                "[DRY RUN] %s %d %s @ MKT | STP %.4f (-%.0f%%) | LMT %.4f (+%.0f%%)",
                action, qty, ticker,
                stop_price, stop_pct * 100,
                tp_price, take_profit_pct * 100,
            )
            order_info["status"] = "dry_run"
            return order_info

        if conn.is_live and not conn.cfg.live_trading:
            logger.error("Live port but live_trading=False. Aborting order.")
            order_info["status"] = "aborted"
            return order_info

        try:
            # Strip Yahoo Finance exchange suffixes (.L, .PA, .T, .HK, .SR, etc.)
            ibkr_ticker = ticker.split(".")[0] if "." in ticker else ticker
            contract = Stock(ibkr_ticker, universe_cfg.ibkr_exchange, universe_cfg.currency)

            # Try to qualify with primary exchange; fall back to SMART if market is closed
            # (e.g. ENXTPA/LSE/TSEJ during overnight US session).
            # ib_insync logs Error 200 as a WARNING without raising — check conId instead.
            try:
                conn.ib.qualifyContracts(contract)
            except Exception:
                contract.conId = 0   # force fallback below

            if not contract.conId:
                logger.warning(
                    "qualifyContracts failed for %s on %s (conId=0) — retrying with SMART",
                    ibkr_ticker, universe_cfg.ibkr_exchange,
                )
                contract = Stock(ibkr_ticker, "SMART", universe_cfg.currency)
                conn.ib.qualifyContracts(contract)

            if bracket_mode:
                # ── Bracket: MKT (hold) + STP child (transmits both together) ──
                # Works on US exchanges (NYSE/NASDAQ). Stop is attached to the parent.
                parent          = MarketOrder(action, qty)
                parent.tif      = "DAY"

                if entry_price > 0:
                    parent.transmit = False   # STP+LMT children will trigger transmission
                else:
                    # No price available — transmit MKT immediately, skip STP/LMT
                    # (avoids MKT stuck in PreSubmitted when LMT price=0 gets rejected)
                    parent.transmit = True
                    logger.warning(
                        "No entry_price for %s — MKT transmitted immediately, STP/LMT skipped",
                        ticker,
                    )

                parent_trade = conn.ib.placeOrder(contract, parent)
                order_info["ibkr_order_id"] = parent_trade.order.orderId

                if entry_price > 0:
                    # STP (stop loss) — second child, hold transmission
                    stp               = Order()
                    stp.action        = stop_action
                    stp.orderType     = "STP"
                    stp.tif           = "GTC"
                    stp.totalQuantity = protective_qty   # full target position size
                    stp.auxPrice      = _round_price(stop_price, universe_cfg.currency)
                    stp.parentId      = parent_trade.order.orderId
                    stp.transmit      = False   # LMT child will trigger final transmission

                    stp_trade = conn.ib.placeOrder(contract, stp)

                    # LMT (take profit) — third child, triggers transmission of all 3
                    lmt               = LimitOrder(stop_action, protective_qty, _round_price(tp_price, universe_cfg.currency))
                    lmt.tif           = "GTC"
                    lmt.parentId      = parent_trade.order.orderId
                    lmt.transmit      = True    # transmits MKT + STP + LMT together

                    lmt_trade = conn.ib.placeOrder(contract, lmt)

                    order_info["ibkr_stp_order_id"] = stp_trade.order.orderId
                    order_info["ibkr_lmt_order_id"] = lmt_trade.order.orderId
                order_info["status"] = "submitted"
                logger.info(
                    "Bracket submitted: %s %d %s | mkt=%d stp=%.2f lmt=%.2f",
                    action, qty, ticker,
                    parent_trade.order.orderId, stop_price, tp_price,
                )
            else:
                # ── EU markets: MKT transmits immediately; STP + LMT as standalone GTC ──
                # If exchange rejects stop/limit, MKT is already through — no PreSubmitted freeze.
                parent          = MarketOrder(action, qty)
                parent.tif      = "DAY"
                parent.transmit = True

                parent_trade = conn.ib.placeOrder(contract, parent)
                order_info["ibkr_order_id"] = parent_trade.order.orderId
                order_info["status"] = "submitted"

                def _try_order(o: Order, label: str) -> Optional[int]:
                    try:
                        t = conn.ib.placeOrder(contract, o)
                        return t.order.orderId
                    except Exception as err:
                        logger.warning("Standalone %s failed for %s: %s", label, ticker, err)
                        return None

                if entry_price > 0:
                    # OCA group: when STP fills, IBKR auto-cancels LMT and vice-versa.
                    # Prevents zombie orders after position is closed by one of them.
                    oca_group = f"TRS_{ibkr_ticker}_{parent_trade.order.orderId}"

                    stp               = Order()
                    stp.action        = stop_action
                    stp.orderType     = "STP"
                    stp.tif           = "GTC"
                    stp.totalQuantity = protective_qty   # full target position size
                    stp.auxPrice      = _round_price(stop_price, universe_cfg.currency)
                    stp.ocaGroup      = oca_group
                    stp.ocaType       = 1   # cancel with block (most protective)
                    stp.transmit      = True
                    stp_id = _try_order(stp, "STP")
                    if stp_id:
                        order_info["ibkr_stp_order_id"] = stp_id

                    lmt = LimitOrder(stop_action, protective_qty, _round_price(tp_price, universe_cfg.currency))
                    lmt.tif      = "GTC"
                    lmt.ocaGroup = oca_group
                    lmt.ocaType  = 1
                    lmt.transmit = True
                    lmt_id = _try_order(lmt, "LMT")
                    if lmt_id:
                        order_info["ibkr_lmt_order_id"] = lmt_id

                    logger.info(
                        "MKT+STP+LMT submitted: %s %d %s | mkt=%d stp=%s(%.2f) lmt=%s(%.2f)",
                        action, qty, ticker,
                        parent_trade.order.orderId,
                        stp_id, stop_price, lmt_id, tp_price,
                    )
                else:
                    logger.warning("No entry_price for %s — STP/LMT not placed", ticker)

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
        """Return available funds in this universe's currency from IBKR account summary."""
        currency = self.udef_cfg.currency
        try:
            summary = self.conn.ib.accountSummary()
        except Exception:
            return 0.0
        for item in summary:
            if item.tag == "AvailableFunds" and item.currency == currency:
                return float(item.value)
        # Fallback: base-currency available funds
        for item in summary:
            if item.tag == "AvailableFunds" and item.currency not in ("BASE", ""):
                return float(item.value)
        return 0.0

    def _handle_sub_lot(
        self,
        target:   "pd.Series",
        snapshot,
        prices:   dict,
    ) -> "tuple[pd.Series, list]":
        """
        Handle positions where the strategy weight converts to < 1 share.

        Rules:
          1. If (1 lot × price) ≤ available cash → round up to 1 lot and execute.
          2. Otherwise → add to manual-orders list (report saved separately).

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

        lot           = getattr(self.udef_cfg, "min_lot_size", 1) or 1
        avail_cash    = self._get_available_cash()
        remaining     = avail_cash
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
        Post-rebalancing pass: ensure every live position has correct STP + LMT.

        For each live position in this universe's currency:
          - If STP and LMT both exist with the exact right quantity → skip
          - Otherwise: cancel stale protective orders + place fresh ones using
            the current market price as reference for stop/limit levels

        mm_tp_pcts: per-ticker take-profit fractions from MM fair-value divergence.
                    Falls back to flat RiskConfig.take_profit_pct when absent.

        Called after wait_for_fills so new entries are already reflected in live
        positions and their bracket STP/LMT have been placed by place_order().
        """
        from .risk import RiskConfig
        risk_cfg = RiskConfig()
        if mm_tp_pcts is None:
            mm_tp_pcts = {}

        live = self.reconciler.get_live_positions(self.conn, currency=self.udef_cfg.currency)
        if live.empty:
            return

        # Build map: ibkr_symbol → {orderType: (qty, trade)}
        prot: dict = {}
        try:
            for trade in self.conn.ib.openTrades():
                if trade.order.orderType not in {"STP", "LMT"}:
                    continue
                sym = trade.contract.symbol
                qty = abs(float(trade.order.totalQuantity))
                prot.setdefault(sym, {})[trade.order.orderType] = (qty, trade)
        except Exception as e:
            logger.warning("_restore_protective_orders: could not read open trades: %s", e)
            return

        for sym, row in live.iterrows():
            pos_qty = abs(float(row["quantity"]))

            if pos_qty == 0:
                # Closed position — cancel any leftover protective orders
                for otype, (_, trade) in prot.get(sym, {}).items():
                    if not self.cfg.dry_run:
                        self.conn.ib.cancelOrder(trade.order)
                    logger.info("[%s] Cancelled orphaned %s for closed %s",
                                self.udef_cfg.name, otype, sym)
                continue

            sym_orders = prot.get(sym, {})
            stp_entry  = sym_orders.get("STP")
            lmt_entry  = sym_orders.get("LMT")
            stp_ok     = stp_entry is not None and stp_entry[0] == pos_qty
            lmt_ok     = lmt_entry is not None and lmt_entry[0] == pos_qty

            if stp_ok and lmt_ok:
                continue  # both present and correctly sized → nothing to do

            # Reference price: use current market price (stripped ticker)
            entry_price = prices.get(sym, 0.0)
            if entry_price <= 0:
                logger.warning("[%s] No price for %s — cannot restore STP/LMT",
                               self.udef_cfg.name, sym)
                continue

            # Cancel any stale protective orders for this ticker
            self.order_mgr.cancel_protective_orders(self.conn, sym, dry_run=self.cfg.dry_run)

            # Place fresh standalone STP + LMT
            target_qty    = pos_qty if float(row["quantity"]) > 0 else -pos_qty
            ticker_tp_pct = mm_tp_pcts.get(sym, risk_cfg.take_profit_pct)
            self.order_mgr.place_protective_only(
                self.conn, sym, target_qty, entry_price,
                self.udef_cfg,
                stop_pct        = risk_cfg.stop_loss_pct,
                take_profit_pct = ticker_tp_pct,
                dry_run         = self.cfg.dry_run,
            )

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
            live_now = self.reconciler.get_live_positions(self.conn, currency=self.udef_cfg.currency)

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
            if yf_prices:
                missing = [t for t in all_tickers if t not in prices or prices[t] <= 0]
                filled  = [t for t in missing if yf_prices.get(t, 0) > 0]
                for t in filled:
                    prices[t] = yf_prices[t]
                if filled:
                    logger.info(
                        "[%s] Used yfinance close prices for %d tickers (no IBKR data): %s",
                        self.udef_cfg.name, len(filled), filled,
                    )

            # 3. Target shares
            target = self.reconciler.compute_target_shares(
                snapshot.long_weights, snapshot.short_weights,
                nav, self.cfg, prices, universe_cfg=self.udef_cfg,
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
            target, manual_rows = self._handle_sub_lot(target, snapshot, prices)
            if manual_rows and not self.cfg.dry_run:
                self._save_fractional_report(manual_rows, snapshot.date)

            # 4. Delta vs live (filled positions + pending entry orders)
            live    = self.reconciler.get_live_positions(self.conn, currency=self.udef_cfg.currency)
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

            # Build per-ticker take-profit from MM fair-value divergence.
            # divergence_pct = (VL_theoretical - MarketCap) / MarketCap × 100
            # Long  (+div): stock is undervalued → fair price is div% above current → LMT there
            # Short (-div): stock is overvalued  → fair price is |div|% below → LMT there
            # Fallback to flat tp_pct when MM data is missing or divergence ≈ 0.
            MIN_TP = 0.05   # floor: always at least 5% upside required
            MAX_TP = 0.80   # cap: discard outlier MM values > 80%
            mm_tp_pcts: dict = {}
            try:
                mm_df = getattr(snapshot, "mm_df", None)
                if mm_df is not None and not mm_df.empty and "divergence_pct" in mm_df.columns:
                    for raw_t, mm_row in mm_df.iterrows():
                        t   = raw_t.split(".")[0] if "." in str(raw_t) else str(raw_t)
                        div = mm_row.get("divergence_pct", None)
                        if div is None or pd.isna(div):
                            continue
                        tp = abs(float(div)) / 100.0        # always positive; direction handled by place_order
                        tp = max(MIN_TP, min(tp, MAX_TP))   # clamp
                        mm_tp_pcts[t] = tp
                if mm_tp_pcts:
                    logger.info(
                        "[%s] MM fair-value take-profits: %d tickers (range %.0f%%–%.0f%%)",
                        self.udef_cfg.name, len(mm_tp_pcts),
                        min(mm_tp_pcts.values()) * 100,
                        max(mm_tp_pcts.values()) * 100,
                    )
            except Exception as e:
                logger.warning("[%s] Could not build MM take-profits: %s — using flat %.0f%%",
                               self.udef_cfg.name, e, tp_pct * 100)

            for _, row in pd.concat([exits, entries]).reset_index(drop=True).iterrows():
                ticker        = str(row["ticker"])
                ticker_tp_pct = mm_tp_pcts.get(ticker, tp_pct)   # MM-derived or flat fallback
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
