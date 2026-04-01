"""
Taurus – Entry point.

Usage examples
──────────────

# 1. Run a single rebalance (live / paper):
python main.py --mode snapshot --end 2024-12-31

# 2. Run a full back-test and produce reports:
python main.py --mode backtest --start 2015-01-01 --end 2024-12-31

# 3. Override config parameters from the command line:
python main.py --mode backtest --start 2018-01-01 --end 2024-12-31 \\
               --n-longs 30 --n-shorts 30 --alpha-t 2.5

Output artefacts are written to ./output/:
  equity_curve.png
  monthly_heatmap.png
  rolling_sharpe.png
  positions.csv
  analytics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("taurus.main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Taurus Long/Short Equity Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode",    choices=["snapshot", "backtest", "live", "report", "live-report", "force-rebalance"], default="backtest")
    p.add_argument("--start",   default="2015-01-01", help="Back-test start date (YYYY-MM-DD)")
    p.add_argument("--end",     default="2024-12-31", help="Back-test / snapshot end date")
    p.add_argument("--output",  default="output",     help="Output directory")

    # Config overrides
    p.add_argument("--n-longs",     type=int,   default=25)
    p.add_argument("--n-shorts",    type=int,   default=25)
    p.add_argument("--alpha-t",     type=float, default=2.0, help="|t-stat| threshold for alpha")
    p.add_argument("--lookback",    type=int,   default=60,  help="FF5 regression window (months)")
    p.add_argument("--leverage-gap",type=float, default=0.25,help="MM leverage gap threshold")
    p.add_argument("--rf-rate",     type=float, default=0.045, help="Annual risk-free rate")
    p.add_argument("--no-shrinkage",action="store_true", help="Disable Ledoit-Wolf shrinkage")
    p.add_argument("--cache-dir",   default=".cache")
    p.add_argument("--cache-ttl",   type=float, default=12.0, help="Cache TTL in hours")

    # Leverage
    p.add_argument("--leverage",      type=float, default=1.0,
                   help="Gross leverage (1.0=no leverage, 1.5=150%% gross)")
    p.add_argument("--margin-cost",   type=float, default=0.058,
                   help="Annual margin financing cost (default: 5.8%% IBKR)")
    p.add_argument("--borrow-cost",   type=float, default=0.010,
                   help="Annual stock borrow fee for shorts (default: 1.0%%)")

    # Futures beta hedge (Phase 2)
    p.add_argument("--futures-hedge", action="store_true",
                   help="Use ES/SPY futures overlay for beta neutralisation instead of weight rescaling")
    p.add_argument("--futures-roll-cost", type=float, default=0.0015,
                   help="Quarterly futures roll cost as fraction of notional (default: 0.15%%)")

    # Optional ticker list
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Explicit ticker list (defaults to full S&P 500)")

    # Live trading
    p.add_argument("--universes", nargs="+",
                   default=["sp500"],
                   help="Universes to trade: sp500 nasdaq100 cac40 ftse100 nikkei225 hangseng tadawul")
    p.add_argument("--live",      action="store_true",
                   help="Live trading (port 7496). Default: paper (7497)")
    p.add_argument("--no-dry-run", action="store_true",
                   help="Submit real orders. Default: dry-run (log only)")
    p.add_argument("--nav",       type=float, default=100_000.0,
                   help="Total portfolio NAV in USD for position sizing")

    return p.parse_args()


def build_config(args: argparse.Namespace):
    from taurus.config import TaurusConfig

    return TaurusConfig(
        n_longs=args.n_longs,
        n_shorts=args.n_shorts,
        alpha_tstat_threshold=args.alpha_t,
        lookback_months=args.lookback,
        leverage_gap_threshold=args.leverage_gap,
        risk_free_rate_annual=args.rf_rate,
        cov_shrinkage=not args.no_shrinkage,
        cache_dir=args.cache_dir,
        cache_ttl_hours=args.cache_ttl,
        gross_leverage=args.leverage,
        margin_cost_annual=args.margin_cost,
        borrow_cost_annual=args.borrow_cost,
        use_futures_hedge=args.futures_hedge,
        futures_roll_cost_quarterly=args.futures_roll_cost,
        ibkr_port=7496 if args.live else 7497,
        live_trading=args.live,
        dry_run=not args.no_dry_run,
        nav_usd=args.nav,
    )


def run_snapshot(args, cfg) -> None:
    from taurus.strategy import TaurusStrategy
    from taurus.reporting import print_analytics, top_positions_table

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    strategy = TaurusStrategy(cfg)
    # Warm-up: load enough history for the regression + momentum
    from taurus.strategy import cfg_warm_months
    import pandas as pd
    warm_months = cfg_warm_months(cfg)
    start_date = (pd.Timestamp(args.end) - pd.DateOffset(months=warm_months + 5)).strftime("%Y-%m-%d")

    strategy.load_data(start=start_date, end=args.end, tickers=args.tickers)
    snap = strategy.run(as_of=args.end)

    logger.info("Snapshot date: %s", snap.date.date())
    logger.info("Long positions (%d):", snap.n_longs)
    for t, w in snap.long_weights.sort_values(ascending=False).items():
        logger.info("  %-8s  %.2f%%", t, w * 100)

    logger.info("Short positions (%d):", snap.n_shorts)
    for t, w in snap.short_weights.sort_values(ascending=False).items():
        logger.info("  %-8s  %.2f%%", t, w * 100)

    pos_path = output / "positions.csv"
    pos_df = snap.long_weights.rename("weight").to_frame()
    pos_df["leg"] = "LONG"
    sh_df = snap.short_weights.rename("weight").to_frame()
    sh_df["leg"] = "SHORT"
    import pandas as pd
    pd.concat([pos_df, sh_df]).to_csv(pos_path)
    logger.info("Positions saved to %s", pos_path)


def run_backtest(args, cfg) -> None:
    from taurus.strategy import TaurusStrategy
    from taurus.reporting import (
        combined_positions_df,
        generate_combined_report,
        plot_combined_equity_curve,
        plot_combined_heatmap,
        plot_combined_rolling_sharpe,
        plot_equity_curve,
        plot_monthly_heatmap,
        plot_rolling_sharpe,
        print_analytics,
        top_positions_table,
    )
    from taurus.universe import REGISTRY
    import dataclasses
    import json
    import pandas as pd

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    universes = args.universes if hasattr(args, "universes") and args.universes else ["sp500"]

    all_analytics: dict = {}
    all_results:   dict = {}

    for universe_name in universes:
        logger.info("=" * 60)
        logger.info("Back-test: %s  (%s → %s)", universe_name.upper(), args.start, args.end)
        logger.info("=" * 60)

        udef    = REGISTRY.get(universe_name)
        ucfg    = udef.config
        uni_out = output / universe_name
        uni_out.mkdir(parents=True, exist_ok=True)

        u_cfg = dataclasses.replace(cfg, n_longs=ucfg.n_longs, n_shorts=ucfg.n_shorts)

        warm_months = u_cfg.lookback_months + u_cfg.momentum_months + 6
        warm_start  = (pd.Timestamp(args.start) - pd.DateOffset(months=warm_months)).strftime("%Y-%m-%d")

        tickers = args.tickers if args.tickers else udef.get_tickers(u_cfg)
        factors = udef.get_ff5_factors(warm_start, args.end, u_cfg)

        strategy = TaurusStrategy(u_cfg)
        result   = strategy.backtest(
            start=args.start,
            end=args.end,
            tickers=tickers,
            factors_df=factors,
            fundamentals_fn=udef.get_fundamentals,   # yfinance for non-US, EDGAR for US
        )

        analytics = result.analytics()
        analytics["universe"] = universe_name
        all_analytics[universe_name] = analytics
        all_results[universe_name]   = result

        print_analytics(analytics)

        # ── Per-universe artefacts ──────────────────────────────────────── #
        with open(uni_out / "analytics.json", "w") as f:
            json.dump(analytics, f, indent=2)
        result.positions_df().to_csv(uni_out / "positions.csv", index=False)

        try:
            plot_equity_curve(result,    save_path=str(uni_out / "equity_curve.png"))
            plot_monthly_heatmap(result, save_path=str(uni_out / "monthly_heatmap.png"))
            plot_rolling_sharpe(result,  save_path=str(uni_out / "rolling_sharpe.png"))
            logger.info("Per-universe charts saved to %s/", uni_out)
        except Exception as exc:
            logger.warning("Per-universe charts failed for %s: %s", universe_name, exc)

        top = top_positions_table(result, n=10)
        if not top.empty:
            logger.info("[%s] Top positions:\n%s", universe_name, top.to_string(index=False))

    # ── Save monthly returns per universe (used by scheduler for Sharpe weights) ── #
    for name, result in all_results.items():
        ret_path = output / name / "monthly_returns.csv"
        result.portfolio_returns().to_csv(ret_path, header=["return"])

    # ── Combined summary ─────────────────────────────────────────────────── #
    logger.info("\n%s", "=" * 60)
    logger.info("COMBINED SUMMARY  (Sharpe-weighted allocation)")
    logger.info("=" * 60)
    for name, a in all_analytics.items():
        logger.info(
            "  %-12s  Return=%+.1f%%  Annual=%+.1f%%  Sharpe=%.2f  MaxDD=%.1f%%",
            name.upper(),
            a.get("total_return",  0) * 100,
            a.get("annual_return", 0) * 100,
            a.get("sharpe_ratio",  0),
            a.get("max_drawdown",  0) * 100,
        )

    # ── Compute Sharpe-weighted combined row (no lookahead) ──────────────── #
    if len(all_results) > 1:
        import numpy as np

        ret_map = {name: result.portfolio_returns() for name, result in all_results.items()}
        df_ret  = pd.concat(ret_map, axis=1).sort_index()

        combined_monthly = []
        final_weights    = {name: 1.0 / len(all_results) for name in all_results}
        window           = 12

        for i, t in enumerate(df_ret.index):
            hist = df_ret.iloc[max(0, i - window): i]
            sharpes = {}
            for name in df_ret.columns:
                r = hist[name].dropna()
                if len(r) < 3:
                    sharpes[name] = 0.0
                else:
                    ann = (1 + r.mean()) ** 12 - 1
                    vol = r.std() * np.sqrt(12)
                    sharpes[name] = (ann - cfg.risk_free_rate_annual) / vol if vol > 0 else 0.0

            raw   = {k: max(v, 0.0) for k, v in sharpes.items()}
            total = sum(raw.values())
            if total < 1e-9:
                w = {k: 1.0 / len(df_ret.columns) for k in df_ret.columns}
            else:
                w = {k: v / total for k, v in raw.items()}

            row_ret = sum(
                w[name] * df_ret.loc[t, name]
                for name in df_ret.columns
                if not pd.isna(df_ret.loc[t, name])
            )
            combined_monthly.append(row_ret)
            final_weights = w   # keep last for display

        combined_ret = pd.Series(combined_monthly, index=df_ret.index).dropna()

        if not combined_ret.empty:
            cum_c    = (1 + combined_ret).cumprod()
            total_c  = cum_c.iloc[-1] - 1
            ann_c    = (1 + total_c) ** (12 / len(combined_ret)) - 1
            vol_c    = combined_ret.std() * np.sqrt(12)
            sharpe_c = (ann_c - cfg.risk_free_rate_annual) / vol_c if vol_c > 0 else float("nan")
            mdd_c    = (cum_c / cum_c.cummax() - 1).min()

            logger.info("  %s", "-" * 56)
            logger.info(
                "  %-12s  Return=%+.1f%%  Annual=%+.1f%%  Sharpe=%.2f  MaxDD=%.1f%%",
                "COMBINED",
                total_c  * 100,
                ann_c    * 100,
                sharpe_c,
                mdd_c    * 100,
            )
            logger.info("  %s", "-" * 56)
            logger.info("  Current NAV allocation (next rebalance):")
            for name, w in sorted(final_weights.items(), key=lambda x: -x[1]):
                logger.info("    %-12s  %.1f%%", name.upper(), w * 100)

            all_analytics["combined"] = {
                "total_return":   float(total_c),
                "annual_return":  float(ann_c),
                "annual_vol":     float(vol_c),
                "sharpe_ratio":   float(sharpe_c),
                "max_drawdown":   float(mdd_c),
                "n_months":       len(combined_ret),
                "nav_allocation": {k: round(v, 4) for k, v in final_weights.items()},
            }

            # Save Sharpe-weighted combined returns for charts
            combined_ret.to_csv(output / "combined_monthly_returns.csv", header=["return"])

    logger.info("%s", "=" * 60)

    with open(output / "combined_analytics.json", "w") as f:
        json.dump(all_analytics, f, indent=2)

    # ── Combined positions CSV (all universes, universe column added) ──── #
    pos_path = output / "positions.csv"
    if len(all_results) > 1:
        combined_positions_df(all_results).to_csv(pos_path, index=False)
    else:
        next(iter(all_results.values())).positions_df().to_csv(pos_path, index=False)
    logger.info("Positions saved to %s", pos_path)

    # ── Combined charts (multi-universe OR single) ───────────────────────── #
    try:
        if len(all_results) > 1:
            plot_combined_equity_curve(
                all_results, save_path=str(output / "equity_curve.png"))
            plot_combined_heatmap(
                all_results, save_path=str(output / "monthly_heatmap.png"))
            plot_combined_rolling_sharpe(
                all_results, save_path=str(output / "rolling_sharpe.png"))
        else:
            result = next(iter(all_results.values()))
            plot_equity_curve(result,    save_path=str(output / "equity_curve.png"))
            plot_monthly_heatmap(result, save_path=str(output / "monthly_heatmap.png"))
            plot_rolling_sharpe(result,  save_path=str(output / "rolling_sharpe.png"))
        logger.info("Combined charts saved to %s/", output)
    except Exception as exc:
        logger.warning("Combined chart generation failed: %s", exc)

    # ── One-page combined report (blended backtest + live if available) ─────── #
    try:
        from taurus.live_reporting import blend_returns_with_live
        bt_rets = {name: r.portfolio_returns() for name, r in all_results.items()}
        blended = blend_returns_with_live(bt_rets, output_dir=str(output))
        combined_override = blended if not blended.empty else None
        report_end = (
            str(blended.index[-1].date()) if combined_override is not None else args.end
        )
        generate_combined_report(
            results           = all_results,
            all_analytics     = all_analytics,
            save_path         = str(output / "report.png"),
            start             = args.start,
            end               = report_end,
            combined_override = combined_override,
        )
        logger.info("One-page report saved to %s", output / "report.png")
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)

    # ── Top positions (most recent, all universes) ───────────────────────── #
    for name, result in all_results.items():
        top = top_positions_table(result, n=10)
        if not top.empty:
            logger.info("[%s] Latest top positions:\n%s", name.upper(), top.to_string(index=False))


def run_report(args, cfg) -> None:
    """
    Rebuild report.png from stored monthly_returns.csv files (no backtest needed).
    Run this anytime after a live rebalance to get fresh charts.

    Reads:  output/{universe}/monthly_returns.csv   (written by live scheduler)
    Writes: output/report.png
            output/equity_curve.png
            output/rolling_sharpe.png
            output/monthly_heatmap.png
    """
    import json
    import numpy as np
    import pandas as pd
    from taurus.reporting import (
        generate_combined_report,
        plot_combined_equity_curve,
        plot_combined_heatmap,
        plot_combined_rolling_sharpe,
    )

    output    = Path(args.output)
    universes = args.universes if hasattr(args, "universes") and args.universes else ["sp500"]

    # ── Load stored monthly returns ──────────────────────────────────────── #
    ret_map = {}
    for name in universes:
        path = output / name / "monthly_returns.csv"
        if not path.exists():
            logger.warning("[%s] No monthly_returns.csv found — skipping.", name)
            continue
        try:
            s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            s.name = name
            ret_map[name] = s
            logger.info("[%s] Loaded %d months of returns.", name, len(s))
        except Exception as e:
            logger.warning("[%s] Could not load returns: %s", name, e)

    if not ret_map:
        logger.error("No stored returns found in %s. Run a backtest or live rebalance first.", output)
        return

    # ── Load analytics if available ──────────────────────────────────────── #
    analytics_path = output / "combined_analytics.json"
    all_analytics  = {}
    if analytics_path.exists():
        with open(analytics_path) as f:
            all_analytics = json.load(f)
    else:
        # Compute analytics from stored returns
        for name, ret in ret_map.items():
            ret = ret.dropna()
            if ret.empty:
                continue
            cum      = (1 + ret).cumprod()
            total    = cum.iloc[-1] - 1
            ann      = (1 + total) ** (12 / len(ret)) - 1
            vol      = ret.std() * np.sqrt(12)
            sharpe   = (ann - cfg.risk_free_rate_annual) / vol if vol > 0 else float("nan")
            drawdown = (cum / cum.cummax() - 1).min()
            all_analytics[name] = {
                "total_return":  float(total),
                "annual_return": float(ann),
                "annual_vol":    float(vol),
                "sharpe_ratio":  float(sharpe),
                "max_drawdown":  float(drawdown),
                "n_months":      len(ret),
            }

    # ── Print summary ─────────────────────────────────────────────────────── #
    logger.info("=" * 60)
    logger.info("LIVE PORTFOLIO REPORT")
    logger.info("=" * 60)
    for name, a in all_analytics.items():
        if name == "combined":
            continue
        logger.info(
            "  %-12s  Return=%+.1f%%  Annual=%+.1f%%  Sharpe=%.2f  MaxDD=%.1f%%",
            name.upper(),
            a.get("total_return",  0) * 100,
            a.get("annual_return", 0) * 100,
            a.get("sharpe_ratio",  0),
            a.get("max_drawdown",  0) * 100,
        )
    ca = all_analytics.get("combined", {})
    if ca:
        logger.info("  %s", "-" * 56)
        logger.info(
            "  %-12s  Return=%+.1f%%  Annual=%+.1f%%  Sharpe=%.2f  MaxDD=%.1f%%",
            "COMBINED",
            ca.get("total_return",  0) * 100,
            ca.get("annual_return", 0) * 100,
            ca.get("sharpe_ratio",  0),
            ca.get("max_drawdown",  0) * 100,
        )
    logger.info("=" * 60)

    # ── Build fake BacktestResult-like objects for chart functions ────────── #
    # We wrap each Series in a lightweight object that exposes portfolio_returns()
    class _ReturnProxy:
        def __init__(self, ret, cfg):
            self.ret = ret
            self.cfg = cfg
            self.snapshots = []

        def portfolio_returns(self):
            return self.ret

    proxy_results = {
        name: _ReturnProxy(ret.dropna(), cfg)
        for name, ret in ret_map.items()
    }

    # ── Blend backtest returns with live NAV data (if available) ─────────── #
    from taurus.live_reporting import blend_returns_with_live
    blended_combined = blend_returns_with_live(
        {name: r.portfolio_returns() for name, r in proxy_results.items()},
        output_dir=str(output),
    )
    combined_override = blended_combined if not blended_combined.empty else None

    # ── Generate charts ───────────────────────────────────────────────────── #
    try:
        if len(proxy_results) > 1:
            plot_combined_equity_curve(
                proxy_results, save_path=str(output / "equity_curve.png"))
            plot_combined_heatmap(
                proxy_results, save_path=str(output / "monthly_heatmap.png"))
            plot_combined_rolling_sharpe(
                proxy_results, save_path=str(output / "rolling_sharpe.png"))

        all_ret = next(iter(ret_map.values()))
        blended_end = blended_combined.index[-1].date() if not blended_combined.empty else all_ret.index[-1].date()
        generate_combined_report(
            results           = proxy_results,
            all_analytics     = all_analytics,
            save_path         = str(output / "report.png"),
            start             = str(all_ret.index[0].date()),
            end               = str(blended_end),
            combined_override = combined_override,
        )
        logger.info("Report saved → %s/report.png", output)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)


def run_live_report(args, cfg) -> None:
    """
    Generate live trading reports from IBKR NAV snapshots.

    Reads:  output/live/nav_history.csv   (written by scheduler after each rebalance)
    Writes: output/live/analytics.json
            output/live/monthly_returns.csv
            output/live/equity_curve.png
            output/live/monthly_heatmap.png
            output/live/rolling_sharpe.png
            output/live/report.png
    """
    from taurus.live_reporting import generate_live_reports

    output = Path(args.output)
    logger.info("Generating live reports from %s/live/nav_history.csv …", output)
    generate_live_reports(
        output_dir=str(output),
        rf_annual=cfg.risk_free_rate_annual,
    )


def _cleanup_orphaned_protective_orders(conn, dry_run: bool = False) -> list:
    """
    Cancel STP/LMT orders that are stale or inconsistent with live positions.

    An order is cancelled when:
      (a) No live position exists for that ticker  →  orphaned (position closed)
      (b) Live position exists but order qty ≠ position qty  →  wrong size
              (position was adjusted; old protective order is for old size)

    An order is KEPT when:
      live position exists  AND  abs(order.totalQuantity) == abs(position qty)

    Returns list of (ticker, order_type, order_id, reason) cancelled.
    """
    from taurus.execution import PositionReconciler
    rec  = PositionReconciler()
    live = rec.get_live_positions(conn)   # all positions, all currencies

    # Build {ticker: abs(quantity)} from live positions
    live_qty: dict = {}
    if not live.empty:
        for ticker, row in live.iterrows():
            live_qty[ticker] = abs(float(row["quantity"]))

    PROTECTIVE = {"STP", "LMT"}
    ACTIVE     = {"PreSubmitted", "Submitted", "PendingSubmit"}

    kept      = []
    cancelled = []
    try:
        for trade in conn.ib.openTrades():
            if trade.order.orderType not in PROTECTIVE:
                continue
            if trade.orderStatus.status not in ACTIVE:
                continue

            ticker    = trade.contract.symbol
            order_qty = abs(float(trade.order.totalQuantity))

            if ticker not in live_qty:
                reason = "no_position"
            elif order_qty != live_qty[ticker]:
                reason = f"qty_mismatch(order={order_qty:.0f} vs pos={live_qty[ticker]:.0f})"
            else:
                kept.append((ticker, trade.order.orderType, trade.order.orderId))
                continue   # ← correct size, keep it

            if not dry_run:
                conn.ib.cancelOrder(trade.order)
            cancelled.append((ticker, trade.order.orderType, trade.order.orderId, reason))

    except Exception as e:
        logger.warning("cleanup_orphaned_protective_orders failed: %s", e)

    logger.info(
        "Protective orders audit: %d kept (correct size)  |  %d cancelled (%s)",
        len(kept), len(cancelled), "dry-run" if dry_run else "live",
    )
    for ticker, otype, oid, reason in cancelled:
        logger.info("  ✗ cancelled %s %s #%s  [%s]", otype, ticker, oid, reason)
    for ticker, otype, oid in kept:
        logger.info("  ✓ kept      %s %s #%s", otype, ticker, oid)

    return cancelled


def run_force_rebalance(args, cfg) -> None:
    """
    Force-rebalance specific universe(s) NOW, bypassing the end-of-month check.

    Use cases:
      - Deploy a new/fixed universe (e.g. nikkei225 after FX fix) without
        waiting for month-end
      - Re-run a single universe after a bug fix

    Safety guarantees:
      - Only touches the specified --universes; all others are untouched
      - scheduler_state.json is NOT updated → April 30 rebalancing is unaffected
      - Reconciliation ensures adjust (not close+reopen) for existing positions
      - Orphaned STP/LMT (no matching position) are cancelled before the run
      - Run with --no-dry-run to submit real orders (default: dry-run)

    Usage:
      python main.py --mode force-rebalance --universes nikkei225 --no-dry-run --live
    """
    import pandas as pd
    from taurus.execution import IBKRConnection
    from taurus.scheduler import RebalanceScheduler

    universes = args.universes
    dry_run   = cfg.dry_run

    logger.info(
        "=== FORCE-REBALANCE | universes=%s | %s | dry_run=%s ===",
        universes, "LIVE" if args.live else "PAPER", dry_run,
    )

    conn = IBKRConnection(cfg)
    if not conn.ensure_connected():
        logger.error("Cannot connect to IBKR TWS/Gateway. Is TWS/IB Gateway running?")
        return

    try:
        # ── Step 1: Clean up orphaned STP/LMT (no matching live position) ── #
        logger.info("Step 1/2 — Cleaning up orphaned protective orders …")
        _cleanup_orphaned_protective_orders(conn, dry_run=dry_run)

        # ── Step 2: Rebalance specified universe(s) ──────────────────────── #
        logger.info("Step 2/2 — Running rebalance for: %s", universes)

        # Reuse scheduler logic (Sharpe weights, execution pipeline, reports)
        # but skip _save_state() so the monthly cycle is unaffected.
        scheduler = RebalanceScheduler(
            cfg=cfg,
            universes=universes,
            output_dir=args.output,
        )
        scheduler.conn = conn   # reuse already-connected instance

        nav_weights = scheduler._sharpe_weights()

        # as_of must be the last business day of the most recently completed month
        # so the date exists in the monthly price data.
        # (The normal scheduler only runs on month-end so it never hits this issue.)
        from taurus.scheduler import _last_business_day_of_month
        today_d = pd.Timestamp.today().date()
        prev_month = (pd.Timestamp.today() - pd.offsets.MonthBegin(1)).date()
        as_of     = pd.Timestamp(_last_business_day_of_month(prev_month))
        logger.info("Force-rebalance as_of = %s (last completed month-end)", as_of.date())

        for universe_name in universes:
            try:
                logger.info("--- Force-rebalancing: %s ---", universe_name)
                scheduler._run_single_universe(
                    universe_name,
                    as_of,
                    nav_fraction=nav_weights.get(universe_name, 1.0 / len(universes)),
                )
            except Exception as e:
                logger.error("Force-rebalance failed for %s: %s", universe_name, e, exc_info=True)

        logger.info("=== Force-rebalance complete — scheduler_state.json NOT modified ===")

    finally:
        conn.disconnect()


def run_live(args, cfg) -> None:
    from taurus.scheduler import RebalanceScheduler
    logger.info(
        "Starting live scheduler | universes=%s | %s | dry_run=%s",
        args.universes,
        "LIVE" if args.live else "PAPER",
        cfg.dry_run,
    )
    scheduler = RebalanceScheduler(
        cfg=cfg,
        universes=args.universes,
        output_dir=args.output,
    )
    scheduler.run_forever()


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)

    logger.info("Taurus Strategy  |  mode=%s", args.mode)

    if args.mode == "snapshot":
        run_snapshot(args, cfg)
    elif args.mode == "backtest":
        run_backtest(args, cfg)
    elif args.mode == "live":
        run_live(args, cfg)
    elif args.mode == "report":
        run_report(args, cfg)
    elif args.mode == "live-report":
        run_live_report(args, cfg)
    elif args.mode == "force-rebalance":
        run_force_rebalance(args, cfg)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
