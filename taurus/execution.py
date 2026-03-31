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
                clientId=self.cfg.ibkr_client_id + self._attempt % 90,
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

    def get_live_positions(self, conn: IBKRConnection) -> pd.DataFrame:
        """Returns DataFrame: ticker, quantity (+long / -short), avg_cost."""
        positions = conn.ib.positions(account=conn.cfg.ibkr_account or "")
        if not positions:
            return pd.DataFrame(columns=["ticker", "quantity", "avg_cost"])

        records = []
        for p in positions:
            records.append({
                "ticker":   p.contract.symbol,
                "quantity": p.position,
                "avg_cost": p.averageCost,
            })
        return pd.DataFrame(records).set_index("ticker")

    def get_account_nav(self, conn: IBKRConnection, currency: str = None) -> float:
        """Returns NAV in the requested currency (falls back to account base currency)."""
        summary = conn.ib.accountSummary(account=conn.cfg.ibkr_account or "")
        # IBKR reports NetLiquidation in multiple currencies — prefer the universe's currency
        if currency:
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == currency:
                    return float(item.value)
        # Fallback: account base currency
        for item in summary:
            if item.tag == "NetLiquidation":
                return float(item.value)
        return conn.cfg.nav_usd

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
        target: Dict[str, int] = {}

        lot = getattr(cfg, "min_lot_size", 1) or 1  # fallback to 1 if not set
        # Use universe min_lot_size if available (e.g. TSE = 100)
        ulot = getattr(universe_cfg, "min_lot_size", 1) if universe_cfg else 1
        lot = max(lot, ulot)

        def _round_lot(shares: int) -> int:
            """Round DOWN to nearest lot. Returns 0 if below 1 lot (skip position)."""
            return (shares // lot) * lot

        skipped = []
        for ticker, w in long_weights.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                dollars = w * nav_usd * half
                rounded = _round_lot(int(dollars / price))
                if rounded > 0:
                    target[ticker] = rounded
                else:
                    skipped.append(ticker)

        for ticker, w in short_weights.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                dollars = w * nav_usd * half
                rounded = _round_lot(int(dollars / price))
                if rounded > 0:
                    target[ticker] = -rounded
                else:
                    skipped.append(ticker)

        if skipped:
            logger.warning(
                "Skipped %d positions below min lot size (%d): %s",
                len(skipped), lot, skipped,
            )

        return pd.Series(target, dtype=int)

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
        ENTRY_TYPES = {"MKT", "LMT", "MOC", "MKT ON CLOSE"}

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
            filled  = int(live_positions.loc[ticker, "quantity"]) \
                      if ticker in live_positions.index else 0
            inflight = int(pending_shares.get(ticker, 0))
            current  = filled + inflight
            target   = int(target_shares.get(ticker, 0))
            delta    = target - current
            if delta != 0:
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

        try:
            conn.ib.qualifyContracts(*contracts)
        except Exception as e:
            logger.warning("qualifyContracts failed: %s", e)

        tickers_data = []
        for c in contracts:
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

    def place_order(
        self,
        conn:         IBKRConnection,
        ticker:       str,
        delta:        int,              # positive=buy, negative=sell/short
        universe_cfg: UniverseConfig,
        dry_run:      bool = True,
        trail_pct:    float = 0.10,     # trailing stop % (10% par défaut)
    ) -> Optional[dict]:
        """
        Place a bracket order: MarketOrder + native TrailingStop on IBKR servers.

        The trailing stop lives on IBKR's servers — your PC can be off after
        the order is placed and the stop will still trigger automatically.

        trail_pct = 0.10 → stop follows the price, cuts if -10% from peak.
        """
        from ib_insync import Stock, MarketOrder, Order

        action       = "BUY"  if delta > 0 else "SELL"
        stop_action  = "SELL" if delta > 0 else "BUY"
        qty          = abs(delta)

        _use_trail = getattr(universe_cfg, "supports_trail_stop", True)
        order_info = {
            "ticker":    ticker,
            "action":    action,
            "quantity":  qty,
            "type":      "MKT+TRAIL" if _use_trail else "MKT",
            "trail_pct": trail_pct if _use_trail else None,
            "dry_run":   dry_run,
            "status":    "pending",
        }

        if dry_run:
            if _use_trail:
                logger.info(
                    "[DRY RUN] %s %d %s @ MKT + TrailingStop %.0f%%",
                    action, qty, ticker, trail_pct * 100,
                )
            else:
                logger.info("[DRY RUN] %s %d %s @ MKT (no trail)", action, qty, ticker)
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

            use_trail = getattr(universe_cfg, "supports_trail_stop", True)

            if use_trail:
                # ── Bracket: MKT (hold) + TRAIL child (transmits both) ── #
                parent          = MarketOrder(action, qty)
                parent.tif      = "DAY"
                parent.transmit = False   # child will trigger transmission

                parent_trade = conn.ib.placeOrder(contract, parent)

                trail                 = Order()
                trail.action          = stop_action
                trail.orderType       = "TRAIL"
                trail.tif             = "GTC"
                trail.totalQuantity   = qty
                trail.trailingPercent = trail_pct * 100
                trail.parentId        = parent_trade.order.orderId
                trail.transmit        = True   # transmits both orders

                trail_trade = conn.ib.placeOrder(contract, trail)

                order_info["ibkr_order_id"]       = parent_trade.order.orderId
                order_info["ibkr_trail_order_id"] = trail_trade.order.orderId
                order_info["status"] = "submitted"
                logger.info(
                    "Bracket submitted: %s %d %s | parent=%d trail=%d (%.0f%%)",
                    action, qty, ticker,
                    parent_trade.order.orderId, trail_trade.order.orderId,
                    trail_pct * 100,
                )
            else:
                # ── Simple MKT (Euronext/LSE — TRAIL not supported on MKT) ── #
                parent          = MarketOrder(action, qty)
                parent.tif      = "DAY"
                parent.transmit = True   # transmit immediately, no stop attached

                parent_trade = conn.ib.placeOrder(contract, parent)

                order_info["ibkr_order_id"] = parent_trade.order.orderId
                order_info["status"] = "submitted"
                logger.info(
                    "MKT submitted (no trail): %s %d %s | orderId=%d",
                    action, qty, ticker, parent_trade.order.orderId,
                )

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

            # 4. Delta vs live (filled positions + pending entry orders)
            live    = self.reconciler.get_live_positions(self.conn)
            pending = self.reconciler.get_pending_shares(self.conn)
            delta_df = self.reconciler.compute_order_delta(target, live, pending)

            if delta_df.empty:
                logger.info("[%s] No trades needed — portfolio already at target.", self.udef_cfg.name)
                report.save(self.output_dir)
                return report

            # 5. Exits first (delta reduces position), then entries
            exits   = delta_df[delta_df.apply(
                lambda r: (r["current"] > 0 and r["delta"] < 0) or
                          (r["current"] < 0 and r["delta"] > 0), axis=1
            )]
            entries = delta_df[~delta_df.index.isin(exits.index)]

            from .risk import RiskConfig
            trail_pct = RiskConfig().trailing_stop_pct

            for _, row in pd.concat([exits, entries]).iterrows():
                order_info = self.order_mgr.place_order(
                    self.conn, row["ticker"], int(row["delta"]),
                    self.udef_cfg, dry_run=self.cfg.dry_run,
                    trail_pct=trail_pct,
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
