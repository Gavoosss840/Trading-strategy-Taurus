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
from .risk import RiskManager, RiskConfig

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

# --------------------------------------------------------------------------- #
#  Live MTD (month-to-date) return computation                                 #
# --------------------------------------------------------------------------- #

def compute_live_mtd_returns(output_dir: str, universes: List[str]) -> dict:
    """
    For each universe with a last_snapshot.csv, compute the portfolio return
    from the snapshot date to today using current market prices.

    Returns {universe_name: (last_price_date, portfolio_return, n_matched, n_total)}

    The result is ephemeral (not saved to CSV) — it represents the current
    incomplete period.  Once the next rebalance runs, that period's return
    is locked into monthly_returns.csv and the snapshot is replaced.
    """
    import yfinance as _yf

    results = {}
    for name in universes:
        snap_path = Path(output_dir) / name / "last_snapshot.csv"
        if not snap_path.exists():
            continue

        try:
            prev = pd.read_csv(snap_path)
            if prev.empty or "ticker" not in prev.columns:
                continue

            start_date = pd.Timestamp(prev["as_of"].iloc[0])
            weights = prev.set_index("ticker")["weight"].astype(float)
            tickers = weights.index.tolist()

            if not tickers:
                continue

            today = pd.Timestamp.today()
            if (today - start_date).days < 1:
                continue

            raw = _yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if raw is None or raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]]
                if len(tickers) == 1:
                    close.columns = tickers

            first_prices = close.bfill().iloc[0]
            last_prices = close.ffill().iloc[-1]
            rets = (last_prices - first_prices) / first_prices.replace(0, float("nan"))

            common = weights.index.intersection(rets.index)
            if common.empty:
                continue

            port_return = float((weights[common] * rets[common]).sum())
            last_price_date = close.index[-1]

            results[name] = (last_price_date, port_return, len(common), len(tickers))

            logger.info(
                "[%s] Live MTD: %s → %s = %+.2f%% (%d/%d tickers)",
                name, start_date.date(), last_price_date.date(),
                port_return * 100, len(common), len(tickers),
            )

        except Exception as e:
            logger.warning("[%s] Could not compute MTD return: %s", name, e)

    return results


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
        risk_config:    RiskConfig   = None,
    ):
        self.cfg         = cfg
        self.universes   = universes or ["sp500"]
        self.output_dir  = output_dir
        self.conn        = IBKRConnection(cfg)
        self.risk_mgr    = RiskManager(risk_config or RiskConfig())

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

        # ── Seed NAV history on first startup ─────────────────────────────── #
        # If no NAV history exists yet, record current NAV as the baseline so
        # that the first rebalancing can compute a monthly return.
        self._seed_initial_nav_if_needed()

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
                    # Save state FIRST — prevents double-run if _run_all_universes
                    # crashes after placing orders but before state was persisted.
                    self._save_state(today)
                    self._run_all_universes(pd.Timestamp(today))
                    logger.info("=== Rebalance complete ===")
                else:
                    next_reb = _last_business_day_of_month(today)
                    logger.info(
                        "Waiting for rebalance day. Next: %s | Last: %s",
                        next_reb, last_reb or "never",
                    )
                    # ── Daily risk check (entre les rebalancements) ───── #
                    self._daily_risk_check()

                time.sleep(self.POLL_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user.")
                self.conn.disconnect()
                break
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
                time.sleep(300)   # 5 min cooldown before retry

    # ── Rebalance all universes ──────────────────────────────────────────── #

    def _sharpe_weights(self, window: int = 12) -> dict:
        """
        Compute Sharpe-weighted NAV allocation fractions from stored monthly
        returns.  Weights are proportional to max(Sharpe, 0) over the last
        `window` months; falls back to equal-weight if all Sharpes are ≤ 0.
        """
        import numpy as np

        returns = {}
        for name in self.universes:
            path = Path(self.output_dir) / name / "monthly_returns.csv"
            if path.exists():
                try:
                    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
                    returns[name] = s.tail(window)
                except Exception:
                    pass

        if len(returns) < 2:
            n = len(self.universes)
            return {u: 1.0 / n for u in self.universes}

        sharpes = {}
        rf = self.cfg.risk_free_rate_annual
        for name, r in returns.items():
            r = r.dropna()
            if len(r) < 3:
                sharpes[name] = 0.0
            else:
                ann = (1 + r.mean()) ** 12 - 1
                vol = r.std() * np.sqrt(12)
                sharpes[name] = (ann - rf) / vol if vol > 0 else 0.0

        # Universes with no stored history default to 0
        for name in self.universes:
            if name not in sharpes:
                sharpes[name] = 0.0

        raw = {k: max(v, 0.0) for k, v in sharpes.items()}
        total = sum(raw.values())
        if total < 1e-9:
            n = len(self.universes)
            weights = {k: 1.0 / n for k in self.universes}
        else:
            weights = {k: v / total for k, v in raw.items()}

        logger.info(
            "Sharpe-weighted NAV allocation: %s",
            "  ".join(f"{k.upper()}={v*100:.1f}%" for k, v in weights.items()),
        )
        return weights

    def _run_all_universes(self, as_of: pd.Timestamp) -> None:
        """
        For each universe:
        1. Build strategy snapshot
        2. Execute orders via IBKRExecutor with Sharpe-weighted NAV fraction
        European markets open ~09:00 CET, US markets open 09:30 ET.
        We run European universes first, then US.
        """
        nav_weights = self._sharpe_weights()

        # Order: Asia/Japan first (markets open earliest), then Europe, then US
        asian    = [u for u in self.universes if REGISTRY.get(u).config.region in ("Japan", "Asia")]
        european = [u for u in self.universes if REGISTRY.get(u).config.region == "Europe"]
        american = [u for u in self.universes if REGISTRY.get(u).config.region == "US"]
        others   = [u for u in self.universes if u not in asian + european + american]

        for universe_name in asian + european + american + others:
            try:
                logger.info("--- Running universe: %s ---", universe_name)
                self._run_single_universe(
                    universe_name, as_of,
                    nav_fraction=nav_weights.get(universe_name, 1.0 / len(self.universes)),
                )
            except Exception as e:
                logger.error("Universe %s failed: %s", universe_name, e, exc_info=True)

        # ── Record live performance snapshot after all universes ──────────── #
        self._record_live_snapshot(as_of)

    def _run_single_universe(self, universe_name: str, as_of: pd.Timestamp, nav_fraction: float = 1.0) -> ExecutionReport:
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
            fundamentals_fn=udef.get_fundamentals,  # non-US → yfinance; US → EDGAR+yfinance
        )
        snapshot = strategy.run(as_of=as_of.strftime("%Y-%m-%d"))

        logger.info(
            "[%s] Snapshot: %d longs, %d shorts",
            universe_name, snapshot.n_longs, snapshot.n_shorts,
        )

        # Append last month's realised return BEFORE overwriting the snapshot file
        self._compute_and_append_monthly_return(universe_name, as_of)

        # Extract CURRENT yfinance prices for STP/LMT calculation.
        # CRITICAL: must use TODAY's price, NOT as_of (March 31) price.
        # If the stock dropped -20% between as_of and today, a stop based on
        # the March 31 price would be placed above the current market price,
        # triggering immediately or never protecting against further downside.
        yf_prices: dict = {}
        try:
            prices_df = strategy._prices
            if prices_df is not None and not prices_df.empty:
                stale_date = prices_df.index[-1].date()
                today_date = pd.Timestamp.today().date()
                days_stale = (today_date - stale_date).days

                if days_stale <= 1:
                    # Data is current (same day or 1 day old) — use directly
                    last_row   = prices_df.iloc[-1].dropna()
                    yf_prices  = last_row.to_dict()
                    logger.info(
                        "[%s] yfinance prices ready for %d tickers (date: %s)",
                        universe_name, len(yf_prices), stale_date,
                    )
                else:
                    # as_of is in the past — download TODAY's prices to avoid stale
                    # stop/limit levels that are already breached.
                    logger.info(
                        "[%s] as_of prices are %d days old (%s) — downloading fresh prices for correct STP/LMT levels",
                        universe_name, days_stale, stale_date,
                    )
                    import yfinance as _yf
                    all_t = list(snapshot.long_weights.index) + list(snapshot.short_weights.index)
                    if all_t:
                        raw = _yf.download(
                            all_t,
                            period="2d",
                            interval="1d",
                            auto_adjust=True,
                            progress=False,
                            threads=True,
                        )
                        if isinstance(raw.columns, pd.MultiIndex):
                            close = raw["Close"]
                        else:
                            close = raw[["Close"]]
                            close.columns = all_t
                        fresh_row = close.iloc[-1].dropna()
                        yf_prices = fresh_row.to_dict()
                        logger.info(
                            "[%s] Fresh prices downloaded for %d tickers (date: %s)",
                            universe_name, len(yf_prices), close.index[-1].date(),
                        )
                    # Fill any gaps with as_of prices (better than nothing)
                    stale_row = prices_df.iloc[-1].dropna().to_dict()
                    for t, p in stale_row.items():
                        if t not in yf_prices or yf_prices[t] <= 0:
                            yf_prices[t] = p
        except Exception as e:
            logger.warning("[%s] Could not extract yfinance prices: %s", universe_name, e)

        # Execute via IBKR
        executor = IBKRExecutor(
            conn=self.conn,
            cfg=cfg,
            universe_cfg=ucfg,
            output_dir=self.output_dir,
        )
        report = executor.execute_rebalance(snapshot, nav_fraction=nav_fraction, yf_prices=yf_prices)
        logger.info(
            "[%s] Execution: %d orders placed, %d errors",
            universe_name, len(report.orders), len(report.errors),
        )

        # Save current weights so next rebalance can compute this month's return
        self._save_universe_snapshot(universe_name, snapshot, as_of)

        return report

    # ── Live monthly return tracking ─────────────────────────────────────── #

    def _save_universe_snapshot(
        self, universe_name: str, snapshot, as_of: pd.Timestamp
    ) -> None:
        """
        Persist long/short portfolio weights to TWO files:

        1. output/{universe}/snapshots/snapshot_{date}.csv  — permanent archive,
           never overwritten.  Full history of every rebalance is preserved.
        2. output/{universe}/last_snapshot.csv  — copy of the latest, used by
           _compute_and_append_monthly_return() and the MTD report.

        Weights are stored signed: +w for longs, -w for shorts.
        """
        uni_dir  = Path(self.output_dir) / universe_name
        hist_dir = uni_dir / "snapshots"
        hist_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for ticker, w in snapshot.long_weights.items():
            rows.append({"ticker": ticker, "weight": float(w), "side": "LONG"})
        for ticker, w in snapshot.short_weights.items():
            rows.append({"ticker": ticker, "weight": -float(w), "side": "SHORT"})

        if not rows:
            return

        df = pd.DataFrame(rows)
        df["as_of"] = as_of.date().isoformat()

        # 1. Permanent dated archive (never overwritten)
        date_str  = as_of.date().isoformat()
        hist_path = hist_dir / f"snapshot_{date_str}.csv"
        df.to_csv(hist_path, index=False)

        # 2. Latest pointer (overwritten each rebalance)
        df.to_csv(uni_dir / "last_snapshot.csv", index=False)

        logger.info(
            "[%s] Snapshot archived: %s (%d positions).",
            universe_name, hist_path.name, len(rows),
        )

    def _compute_and_append_monthly_return(
        self, universe_name: str, as_of: pd.Timestamp
    ) -> None:
        """
        Compute the realised portfolio return for the previous period using the
        weights saved in last_snapshot.csv, then append it to
        output/{universe}/monthly_returns.csv.

        This keeps Sharpe weights current without requiring a full backtest after
        each live rebalance.  The return is computed as:

            r_portfolio = Σ  w_i × (P_i_end / P_i_start - 1)

        where w_i is signed (+long / -short), P_i_start is the closing price on
        the previous rebalance date, and P_i_end is the closing price on as_of.
        """
        snap_path = Path(self.output_dir) / universe_name / "last_snapshot.csv"
        ret_path  = Path(self.output_dir) / universe_name / "monthly_returns.csv"

        if not snap_path.exists():
            logger.info("[%s] No previous snapshot — skipping return computation.", universe_name)
            return

        try:
            prev = pd.read_csv(snap_path)
            if prev.empty or "ticker" not in prev.columns:
                return

            prev_date = pd.Timestamp(prev["as_of"].iloc[0])
            weights   = prev.set_index("ticker")["weight"].astype(float)  # signed
            tickers   = weights.index.tolist()

            if not tickers:
                return

            # Duplicate guard: don't append the same month twice
            month_label = prev_date.strftime("%Y-%m-%d")
            if ret_path.exists():
                existing = pd.read_csv(ret_path, index_col=0, parse_dates=True)
                existing.columns = ["return"]
                if month_label in existing.index.strftime("%Y-%m-%d").tolist():
                    logger.info(
                        "[%s] Return for %s already recorded — skipping.",
                        universe_name, month_label,
                    )
                    return
            else:
                existing = pd.DataFrame(columns=["return"])

            # Download prices for [prev_date, as_of]
            import yfinance as _yf
            raw = _yf.download(
                tickers,
                start=prev_date.strftime("%Y-%m-%d"),
                end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                logger.warning("[%s] No price data for return computation.", universe_name)
                return

            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]]
                if len(tickers) == 1:
                    close.columns = tickers

            first_prices = close.bfill().iloc[0]
            last_prices  = close.ffill().iloc[-1]
            rets = (last_prices - first_prices) / first_prices.replace(0, float("nan"))

            common = weights.index.intersection(rets.index)
            if common.empty:
                logger.warning("[%s] No ticker overlap for return computation.", universe_name)
                return

            port_return = float((weights[common] * rets[common]).sum())

            new_row  = pd.DataFrame({"return": [port_return]}, index=[pd.Timestamp(month_label)])
            if existing.empty:
                combined = new_row
            else:
                combined = pd.concat([existing, new_row]).sort_index()

            combined.to_csv(ret_path, header=["return"])
            logger.info(
                "[%s] Monthly return appended: %s → %+.2f%% (%d/%d tickers matched).",
                universe_name, month_label, port_return * 100, len(common), len(tickers),
            )

        except Exception as e:
            logger.warning("[%s] Could not compute monthly return: %s", universe_name, e)

    # ── Daily risk check ────────────────────────────────────────────────── #

    def _sync_risk_state_from_ibkr(self) -> None:
        """
        Sync risk state with live IBKR positions so every held position is monitored,
        not just those recorded during the last rebalancing run.
        - Adds positions present in IBKR but missing from risk state (uses avg_cost as entry price)
        - Removes positions no longer held in IBKR
        """
        from .execution import PositionReconciler
        from .risk import PositionState

        rec = PositionReconciler()
        try:
            # Fetch all live positions (no currency filter — we want everything)
            live = rec.get_live_positions(self.conn)
            if live.empty:
                return
        except Exception as e:
            logger.warning("_sync_risk_state: could not fetch live positions: %s", e)
            return

        # Build a currency→universe_name map from the configured universes
        currency_to_universe: dict = {}
        for uname in self.universes:
            udef = REGISTRY.get(uname)
            if udef:
                currency_to_universe[udef.config.currency] = uname

        live_tickers = set(live.index)

        # Add positions not yet tracked
        added = 0
        for ticker, row in live.iterrows():
            if ticker in self.risk_mgr.state.positions:
                continue
            qty = float(row["quantity"])
            avg_cost = float(row["avg_cost"])
            if avg_cost <= 0 or qty == 0:
                continue
            # Infer universe from existing state or fall back to first configured universe
            universe = (
                next((u for u in self.universes if REGISTRY.get(u)), self.universes[0])
                if self.universes else "unknown"
            )
            self.risk_mgr.state.positions[ticker] = PositionState(
                ticker=ticker,
                side="LONG" if qty > 0 else "SHORT",
                entry_price=avg_cost,
                entry_date=date.today().isoformat(),
                peak_price=avg_cost,
                shares=abs(int(qty)),
                universe=universe,
                entry_vol=0.0,   # unknown at sync time; vol-adjusted stops use flat fallback
            )
            added += 1

        # Remove positions no longer held
        removed = [t for t in list(self.risk_mgr.state.positions) if t not in live_tickers]
        for t in removed:
            del self.risk_mgr.state.positions[t]

        if added or removed:
            logger.info("Risk state synced: +%d added, -%d removed", added, len(removed))
            self.risk_mgr.state.save()

    def _seed_initial_nav_if_needed(self) -> None:
        """
        If output/live/nav_history.csv does not exist yet, record current IBKR
        NAV as the baseline starting point so the first rebalancing can produce
        a monthly return.  Does nothing if history already exists.
        """
        from pathlib import Path as _P
        from .live_reporting import load_nav_history, record_nav_snapshot

        nav_path = _P(self.output_dir) / "live" / "nav_history.csv"
        if nav_path.exists():
            return   # already seeded

        logger.info("No NAV history found — recording current NAV as baseline.")
        try:
            from .execution import PositionReconciler
            rec = PositionReconciler()
            nav = rec.get_account_nav(self.conn)
            if nav and nav > 0:
                try:
                    summary = self.conn.ib.accountSummary(
                        account=self.conn.cfg.ibkr_account or ""
                    )
                    currency = next(
                        (v.currency for v in summary
                         if v.tag == "NetLiquidation" and v.currency not in ("BASE", "")),
                        "EUR",
                    )
                except Exception:
                    currency = "EUR"
                record_nav_snapshot(nav, currency, self.output_dir, note="initial baseline")
                logger.info("Baseline NAV recorded: %s %.2f", currency, nav)
            else:
                logger.warning("Could not seed initial NAV: NAV=%.2f", nav or 0)
        except Exception as e:
            logger.warning("_seed_initial_nav_if_needed failed: %s", e)

    def _record_live_snapshot(self, as_of: pd.Timestamp) -> None:
        """
        Called once after all universes finish rebalancing.
        Records total portfolio NAV and position snapshot for live reports.
        """
        from .execution import PositionReconciler
        from .live_reporting import record_nav_snapshot, record_positions_snapshot

        rec = PositionReconciler()
        try:
            # Get total NAV in account base currency (EUR for this portfolio)
            nav = rec.get_account_nav(self.conn)
            if nav and nav > 0:
                # Detect account currency from account summary
                try:
                    summary = self.conn.ib.accountSummary(
                        account=self.conn.cfg.ibkr_account or ""
                    )
                    currency = next(
                        (v.currency for v in summary
                         if v.tag == "NetLiquidation" and v.currency not in ("BASE", "")),
                        "EUR",
                    )
                except Exception:
                    currency = "EUR"
                record_nav_snapshot(nav, currency, self.output_dir,
                                    note=f"rebalance {as_of.date()}")
            else:
                logger.warning("Could not record NAV snapshot: NAV=%.2f", nav or 0)
        except Exception as e:
            logger.warning("_record_live_snapshot: NAV fetch failed: %s", e)

        try:
            live = rec.get_live_positions(self.conn)   # all positions, no filter
            if not live.empty:
                record_positions_snapshot(live, str(as_of.date()), self.output_dir)
        except Exception as e:
            logger.warning("_record_live_snapshot: positions fetch failed: %s", e)

    def _daily_risk_check(self) -> None:
        """
        Vérifie chaque position quotidiennement contre :
          - Stop-loss individuel (-10% depuis entrée)
          - Trailing stop (-10% depuis le pic)
          - Circuit breaker global (-15% portfolio)
        Coupe les positions qui franchissent les seuils.
        """
        # First sync risk state from actual IBKR positions so all positions are monitored
        self._sync_risk_state_from_ibkr()

        if not self.risk_mgr.state.positions:
            return   # aucune position à surveiller

        logger.info("Daily risk check: %d positions surveillées", len(self.risk_mgr.state.positions))

        # Récupérer les prix actuels pour toutes les positions
        from .execution import OrderManager
        mgr = OrderManager()

        # Grouper par univers pour récupérer les prix avec le bon exchange
        by_universe: dict = {}
        for ticker, pos in self.risk_mgr.state.positions.items():
            by_universe.setdefault(pos.universe, []).append(ticker)

        current_prices: dict = {}
        for universe_name, tickers in by_universe.items():
            try:
                udef = REGISTRY.get(universe_name)
                prices = mgr.get_live_prices(self.conn, tickers, udef.config)
                current_prices.update(prices)
            except Exception as e:
                logger.warning("Impossible de récupérer les prix pour %s: %s", universe_name, e)

        if not current_prices:
            logger.warning("Aucun prix disponible pour le risk check.")
            return

        # NAV actuelle
        from .execution import PositionReconciler
        rec = PositionReconciler()
        try:
            current_nav = rec.get_account_nav(self.conn)
        except Exception:
            current_nav = self.cfg.nav_usd

        # Évaluer les seuils
        events, circuit_open = self.risk_mgr.check_positions(current_prices, current_nav)

        if not events:
            logger.info("✅ Risk check OK — aucun seuil franchi.")
            return

        logger.warning("⚠️  %d position(s) à couper (circuit_breaker=%s)", len(events), circuit_open)

        # Exécuter les exits par univers
        for universe_name in by_universe:
            udef = REGISTRY.get(universe_name)
            universe_events = [e for e in events
                               if self.risk_mgr.state.positions.get(e.ticker, None) is None
                               or self.risk_mgr.state.positions.get(e.ticker).universe == universe_name]
            if universe_events:
                self.risk_mgr.execute_exits(
                    events=universe_events,
                    conn=self.conn,
                    universe_cfg=udef.config,
                    dry_run=self.cfg.dry_run,
                    output_dir=self.output_dir,
                )

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
        Path(tmp).replace(self.STATE_FILE)  # replace() works on Windows (rename() fails if dest exists)
        logger.info("State saved: last_rebalance=%s", rebalance_date)
