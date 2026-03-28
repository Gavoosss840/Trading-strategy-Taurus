"""
Taurus – 24/7 Continuous rebalancing scheduler.

Runs an infinite loop that:
  1. Checks if it's time to rebalance (last business day of the month)
  2. Runs TaurusStrategy.run() for each universe
  3. Executes orders via IBKRExecutor
  4. Persists state to survive restarts
  5. Reconnects to TWS after daily restart (23:45 ET) or network drops

Usage (from main.py --mode live):
    scheduler = RebalanceScheduler(cfg, universes=["sp500", "nasdaq100"])
    scheduler.run_forever()
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .config import TaurusConfig, DEFAULT_CONFIG
from .execution import IBKRConnection, IBKRExecutor, ExecutionReport
from .universe import REGISTRY

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
CET = ZoneInfo("Europe/Paris")


# --------------------------------------------------------------------------- #
#  Market calendar helpers                                                     #
# --------------------------------------------------------------------------- #

def _last_business_day_of_month(dt: date) -> date:
    """Return the last business day (Mon-Fri) of dt's month."""
    # Go to last calendar day, walk back to Friday if weekend
    import calendar
    last = date(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])
    while last.weekday() >= 5:   # 5=Sat, 6=Sun
        last -= timedelta(days=1)
    return last


def _should_rebalance(last_rebalance: Optional[date], check_date: date) -> bool:
    """
    True on the last business day of the month IF we haven't rebalanced
    this month yet.
    """
    month_end = _last_business_day_of_month(check_date)
    if check_date < month_end:
        return False
    if last_rebalance is not None and last_rebalance.month == check_date.month \
            and last_rebalance.year == check_date.year:
        return False
    return True


# --------------------------------------------------------------------------- #
#  Scheduler                                                                   #
# --------------------------------------------------------------------------- #

class RebalanceScheduler:
    """
    24/7 infinite loop that triggers monthly rebalances across all universes.

    State is persisted in .cache/scheduler_state.json so that restarts
    (e.g. after TWS daily restart at 23:45 ET) don't cause double-rebalances.
    """

    POLL_INTERVAL   = 60     # seconds between each "should I rebalance?" check
    STATE_FILE      = ".cache/scheduler_state.json"

    def __init__(
        self,
        cfg:            TaurusConfig = DEFAULT_CONFIG,
        universes:      List[str]    = None,
        output_dir:     str          = "output",
    ):
        self.cfg        = cfg
        self.universes  = universes or ["sp500"]
        self.output_dir = output_dir
        self.conn       = IBKRConnection(cfg)

        Path(".cache").mkdir(exist_ok=True)
        Path(output_dir).mkdir(exist_ok=True)

    # ── Main loop ────────────────────────────────────────────────────────── #

    def run_forever(self) -> None:
        """Entry point. Blocks forever. Call from main.py --mode live."""
        logger.info(
            "Taurus Scheduler starting | universes=%s | dry_run=%s | %s",
            self.universes, self.cfg.dry_run,
            "LIVE" if self.cfg.live_trading else "PAPER",
        )

        while True:
            try:
                if not self.conn.ensure_connected():
                    logger.error("Cannot connect to TWS. Retrying in 5 min.")
                    time.sleep(300)
                    continue

                now_et = datetime.now(ET)
                today  = now_et.date()

                state         = self._load_state()
                last_reb_str  = state.get("last_rebalance_date")
                last_reb      = date.fromisoformat(last_reb_str) if last_reb_str else None

                if _should_rebalance(last_reb, today):
                    logger.info("=== REBALANCE DAY: %s ===", today)
                    self._run_all_universes(pd.Timestamp(today))
                    self._save_state(today)
                    logger.info("=== Rebalance complete ===")
                else:
                    next_reb = _last_business_day_of_month(today)
                    logger.info(
                        "Waiting for rebalance day. Next: %s | Last: %s",
                        next_reb, last_reb or "never",
                    )

                time.sleep(self.POLL_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user.")
                self.conn.disconnect()
                break
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
                time.sleep(300)   # 5 min cooldown before retry

    # ── Rebalance all universes ──────────────────────────────────────────── #

    def _run_all_universes(self, as_of: pd.Timestamp) -> None:
        """
        For each universe:
        1. Build strategy snapshot
        2. Execute orders via IBKRExecutor
        European markets open ~09:00 CET, US markets open 09:30 ET.
        We run European universes first, then US.
        """
        european = [u for u in self.universes if REGISTRY.get(u).config.region == "Europe"]
        american = [u for u in self.universes if REGISTRY.get(u).config.region == "US"]

        for universe_name in european + american:
            try:
                logger.info("--- Running universe: %s ---", universe_name)
                self._run_single_universe(universe_name, as_of)
            except Exception as e:
                logger.error("Universe %s failed: %s", universe_name, e, exc_info=True)

    def _run_single_universe(self, universe_name: str, as_of: pd.Timestamp) -> ExecutionReport:
        from .strategy import TaurusStrategy

        udef = REGISTRY.get(universe_name)
        ucfg = udef.config

        # Override n_longs/n_shorts from universe config
        cfg = TaurusConfig(
            **{k: v for k, v in self.cfg.__dict__.items()
               if not k.startswith("_")},
        )
        cfg.n_longs  = ucfg.n_longs
        cfg.n_shorts = ucfg.n_shorts

        # Load data and run strategy
        strategy = TaurusStrategy(cfg)
        warm_months = cfg.lookback_months + cfg.momentum_months + 6
        start = (as_of - pd.DateOffset(months=warm_months)).strftime("%Y-%m-%d")

        tickers = udef.get_tickers(cfg)
        factors = udef.get_ff5_factors(start, as_of.strftime("%Y-%m-%d"), cfg)

        strategy.load_data(
            start=start,
            end=as_of.strftime("%Y-%m-%d"),
            tickers=tickers,
            factors_df=factors,
        )
        snapshot = strategy.run(as_of=as_of.strftime("%Y-%m-%d"))

        logger.info(
            "[%s] Snapshot: %d longs, %d shorts",
            universe_name, snapshot.n_longs, snapshot.n_shorts,
        )

        # Execute via IBKR
        executor = IBKRExecutor(
            conn=self.conn,
            cfg=cfg,
            universe_cfg=ucfg,
            output_dir=self.output_dir,
        )
        report = executor.execute_rebalance(snapshot)
        logger.info(
            "[%s] Execution: %d orders placed, %d errors",
            universe_name, len(report.orders), len(report.errors),
        )
        return report

    # ── State persistence ────────────────────────────────────────────────── #

    def _load_state(self) -> dict:
        try:
            with open(self.STATE_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self, rebalance_date: date) -> None:
        """Atomic write: write to temp file then rename."""
        tmp = self.STATE_FILE + ".tmp"
        data = {
            "last_rebalance_date": rebalance_date.isoformat(),
            "saved_at": datetime.now().isoformat(),
        }
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp).rename(self.STATE_FILE)
        logger.info("State saved: last_rebalance=%s", rebalance_date)
