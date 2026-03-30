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
        """Returns DataFrame: ticker, quantity (+long / -short), market_value."""
        positions = conn.ib.positions(account=conn.cfg.ibkr_account or "")
        if not positions:
            return pd.DataFrame(columns=["ticker", "quantity", "market_value", "avg_cost"])

        records = []
        for p in positions:
            records.append({
                "ticker":       p.contract.symbol,
                "quantity":     p.position,
                "market_value": p.marketValue,
                "avg_cost":     p.averageCost,
            })
        return pd.DataFrame(records).set_index("ticker")

    def get_account_nav(self, conn: IBKRConnection) -> float:
        """Returns total NAV in account currency from IBKR."""
        summary = conn.ib.accountSummary(account=conn.cfg.ibkr_account or "")
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
    ) -> pd.Series:
        """
        Convert fractional weights → integer share counts.
        Long: positive shares. Short: negative shares.
        """
        half = cfg.gross_leverage / 2.0
        target: Dict[str, int] = {}

        for ticker, w in long_weights.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                dollars = w * nav_usd * half
                target[ticker] = max(1, int(dollars / price))

        for ticker, w in short_weights.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                dollars = w * nav_usd * half
                target[ticker] = -max(1, int(dollars / price))

        return pd.Series(target, dtype=int)

    def compute_order_delta(
        self,
        target_shares:  pd.Series,
        live_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns: ticker, current, target, delta.
        Filters out zero deltas.
        """
        all_tickers = target_shares.index.union(
            live_positions.index if not live_positions.empty else pd.Index([])
        )
        records = []
        for ticker in all_tickers:
            current = int(live_positions.loc[ticker, "quantity"]) \
                      if ticker in live_positions.index else 0
            target  = int(target_shares.get(ticker, 0))
            delta   = target - current
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

        from .test_utils import _market_is_open
        market_open = _market_is_open()
        conn.ib.reqMarketDataType(1 if market_open else 2)

        prices: Dict[str, float] = {}
        contracts = []
        for ticker in tickers:
            c = Stock(ticker, universe_cfg.ibkr_exchange, universe_cfg.currency)
            contracts.append(c)

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
            # Mid price or last
            if bid == bid and ask == ask:
                prices[symbol] = (bid + ask) / 2.0
            elif last == last:
                prices[symbol] = last

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

        order_info = {
            "ticker":    ticker,
            "action":    action,
            "quantity":  qty,
            "type":      "MKT+TRAIL",
            "trail_pct": trail_pct,
            "dry_run":   dry_run,
            "status":    "pending",
        }

        if dry_run:
            logger.info(
                "[DRY RUN] %s %d %s @ MKT + TrailingStop %.0f%%",
                action, qty, ticker, trail_pct * 100,
            )
            order_info["status"] = "dry_run"
            return order_info

        if conn.is_live and not conn.cfg.live_trading:
            logger.error("Live port but live_trading=False. Aborting order.")
            order_info["status"] = "aborted"
            return order_info

        try:
            contract = Stock(ticker, universe_cfg.ibkr_exchange, universe_cfg.currency)
            conn.ib.qualifyContracts(contract)

            # ── Parent: market order (don't transmit yet) ──────────────── #
            parent          = MarketOrder(action, qty)
            parent.transmit = False   # hold until child is ready

            # ── Child: native trailing stop on IBKR servers ────────────── #
            trail           = Order()
            trail.action    = stop_action
            trail.orderType = "TRAIL"
            trail.totalQuantity   = qty
            trail.trailingPercent = trail_pct * 100   # IBKR expects integer %
            trail.parentId  = parent.orderId
            trail.transmit  = True   # transmits both parent + child together

            parent_trade = conn.ib.placeOrder(contract, parent)
            trail_trade  = conn.ib.placeOrder(contract, trail)

            order_info["ibkr_order_id"]       = parent_trade.order.orderId
            order_info["ibkr_trail_order_id"] = trail_trade.order.orderId
            order_info["status"] = "submitted"
            logger.info(
                "Bracket submitted: %s %d %s | parent=%d trail=%d (%.0f%%)",
                action, qty, ticker,
                parent_trade.order.orderId, trail_trade.order.orderId,
                trail_pct * 100,
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

    def execute_rebalance(self, snapshot, nav_fraction: float = 1.0) -> ExecutionReport:
        """
        Full rebalance pipeline:
        1. Get live NAV and positions
        2. Get live prices for all target stocks
        3. Compute target share counts
        4. Compute delta vs live positions
        5. Place exits first, then entries
        6. Place futures hedge
        7. Save and return ExecutionReport
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
            # 1. NAV (scaled by Sharpe-weighted fraction)
            nav = self.reconciler.get_account_nav(self.conn) * nav_fraction
            logger.info(
                "[%s] NAV = %.2f %s (fraction=%.1f%%)",
                self.udef_cfg.name, nav, self.udef_cfg.currency, nav_fraction * 100,
            )

            # 2. Live prices
            all_tickers = list(snapshot.long_weights.index) + list(snapshot.short_weights.index)
            prices = self.order_mgr.get_live_prices(self.conn, all_tickers, self.udef_cfg)

            # 3. Target shares
            target = self.reconciler.compute_target_shares(
                snapshot.long_weights, snapshot.short_weights,
                nav, self.cfg, prices,
            )

            # 4. Delta vs live
            live = self.reconciler.get_live_positions(self.conn)
            delta_df = self.reconciler.compute_order_delta(target, live)

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
