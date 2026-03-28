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

    p.add_argument("--mode",    choices=["snapshot", "backtest", "live"], default="backtest")
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
                   help="Universes to trade: sp500 nasdaq100 cac40 dax ftse100")
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
        plot_equity_curve,
        plot_monthly_heatmap,
        plot_rolling_sharpe,
        print_analytics,
        top_positions_table,
    )
    from taurus.universe import REGISTRY
    import json
    import pandas as pd

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    universes = args.universes if hasattr(args, "universes") and args.universes else ["sp500"]

    all_analytics = {}
    all_results   = {}

    for universe_name in universes:
        logger.info("=" * 60)
        logger.info("Back-test: %s  (%s → %s)", universe_name.upper(), args.start, args.end)
        logger.info("=" * 60)

        udef    = REGISTRY.get(universe_name)
        ucfg    = udef.config
        uni_out = output / universe_name
        uni_out.mkdir(parents=True, exist_ok=True)

        # Per-universe config overrides
        import dataclasses
        u_cfg = dataclasses.replace(
            cfg,
            n_longs=ucfg.n_longs,
            n_shorts=ucfg.n_shorts,
        )

        tickers = args.tickers if args.tickers else udef.get_tickers(u_cfg)
        factors = udef.get_ff5_factors(args.start, args.end, u_cfg)

        strategy = TaurusStrategy(u_cfg)
        result   = strategy.backtest(
            start=args.start,
            end=args.end,
            tickers=tickers,
            factors_df=factors,
        )

        analytics = result.analytics()
        analytics["universe"] = universe_name
        all_analytics[universe_name] = analytics
        all_results[universe_name]   = result

        print_analytics(analytics)

        # Per-universe artefacts
        with open(uni_out / "analytics.json", "w") as f:
            json.dump(analytics, f, indent=2)
        result.positions_df().to_csv(uni_out / "positions.csv", index=False)

        try:
            plot_equity_curve(result,    save_path=str(uni_out / "equity_curve.png"))
            plot_monthly_heatmap(result, save_path=str(uni_out / "monthly_heatmap.png"))
            plot_rolling_sharpe(result,  save_path=str(uni_out / "rolling_sharpe.png"))
        except Exception as exc:
            logger.warning("Charts failed for %s: %s", universe_name, exc)

        top = top_positions_table(result, n=10)
        if not top.empty:
            logger.info("[%s] Top positions:\n%s", universe_name, top.to_string(index=False))

    # ── Combined summary (multi-universe) ───────────────────────────────── #
    if len(universes) > 1:
        logger.info("\n%s", "=" * 60)
        logger.info("COMBINED SUMMARY")
        logger.info("=" * 60)
        for name, a in all_analytics.items():
            logger.info(
                "  %-12s  Return=%+.1f%%  Sharpe=%.2f  MaxDD=%.1f%%",
                name.upper(),
                a.get("total_return_pct", 0),
                a.get("sharpe_ratio", 0),
                a.get("max_drawdown_pct", 0),
            )
        with open(output / "combined_analytics.json", "w") as f:
            json.dump(all_analytics, f, indent=2)

    # Keep single-universe analytics for backward compat
    analytics = all_analytics.get(universes[0], {})
    result    = all_results.get(universes[0])

    # ── Save artefacts ──────────────────────────────────────────────────── #
    analytics_path = output / "analytics.json"
    with open(analytics_path, "w") as f:
        json.dump(analytics, f, indent=2)
    logger.info("Analytics saved to %s", analytics_path)

    positions_path = output / "positions.csv"
    result.positions_df().to_csv(positions_path, index=False)
    logger.info("Positions saved to %s", positions_path)

    try:
        plot_equity_curve(result,  save_path=str(output / "equity_curve.png"))
        plot_monthly_heatmap(result, save_path=str(output / "monthly_heatmap.png"))
        plot_rolling_sharpe(result,  save_path=str(output / "rolling_sharpe.png"))
        logger.info("Charts saved to %s/", output)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    # ── Top positions ────────────────────────────────────────────────────── #
    top = top_positions_table(result, n=20)
    if not top.empty:
        logger.info("Top positions (most recent rebalance):\n%s", top.to_string(index=False))


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
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
