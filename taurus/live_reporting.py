"""
Taurus – Live trading reports.

Tracks real portfolio performance using NAV snapshots recorded at each
monthly rebalancing. Generates the same charts as backtesting but from
live IBKR data.

Data flow:
  Scheduler  →  record_nav_snapshot()    →  output/live/nav_history.csv
  Scheduler  →  record_positions_snapshot()  →  output/live/positions/{date}.csv
  main.py --mode live-report  →  generate_live_reports()  →  output/live/*.png

Usage:
  python main.py --mode live-report --universes sp500 nasdaq100 cac40 ftse100 nikkei225
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LIVE_DIR      = "live"
_NAV_FILE      = "nav_history.csv"
_POSITIONS_DIR = "positions"


# --------------------------------------------------------------------------- #
#  Data recording (called by scheduler at each rebalancing)                   #
# --------------------------------------------------------------------------- #

def record_nav_snapshot(
    nav:        float,
    currency:   str,
    output_dir: str = "output",
    note:       str = "",
) -> None:
    """
    Record total portfolio NAV after a rebalancing cycle.
    Appends one row to output/live/nav_history.csv.
    Idempotent: overwrites any row with the same date.
    """
    live_dir = Path(output_dir) / _LIVE_DIR
    live_dir.mkdir(parents=True, exist_ok=True)
    path = live_dir / _NAV_FILE

    today = datetime.now().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{
        "date":        today,
        "nav":         round(nav, 2),
        "currency":    currency,
        "note":        note or "",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }])

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["date"] != today]   # remove duplicate for today
        new_row = pd.concat([existing, new_row], ignore_index=True)

    new_row.to_csv(path, index=False)
    logger.info("NAV snapshot recorded: %s  %s %.2f", today, currency, nav)


def record_positions_snapshot(
    positions_df: pd.DataFrame,
    rebalance_date: str,
    output_dir: str = "output",
) -> None:
    """
    Save a full IBKR positions snapshot to output/live/positions/{date}.csv.
    positions_df: DataFrame with index=ticker, columns=[quantity, avg_cost].
    Idempotent: overwrites any file with the same date.
    """
    pos_dir = Path(output_dir) / _LIVE_DIR / _POSITIONS_DIR
    pos_dir.mkdir(parents=True, exist_ok=True)
    path = pos_dir / f"{rebalance_date}.csv"
    positions_df.to_csv(path)
    logger.info(
        "Positions snapshot saved: %s  (%d positions)",
        path, len(positions_df),
    )


# --------------------------------------------------------------------------- #
#  Data loading & return computation                                           #
# --------------------------------------------------------------------------- #

def load_nav_history(output_dir: str = "output") -> pd.DataFrame:
    """
    Load NAV history from output/live/nav_history.csv.
    Returns DataFrame indexed by date (DatetimeIndex), sorted ascending.
    """
    path = Path(output_dir) / _LIVE_DIR / _NAV_FILE
    if not path.exists():
        return pd.DataFrame(columns=["nav", "currency"])
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.set_index("date")


def compute_live_monthly_returns(output_dir: str = "output") -> pd.Series:
    """
    Derive monthly returns from consecutive NAV snapshots.
    return_i = (NAV_i / NAV_{i-1}) - 1

    Index is aligned to month-end dates (Period M → timestamp) so the
    heatmap year×month grid is consistent with backtesting output.
    """
    df = load_nav_history(output_dir)
    if len(df) < 2:
        return pd.Series(dtype=float, name="live_returns")

    returns = df["nav"].pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    # Align to month-end timestamps for heatmap compatibility
    returns.index = returns.index.to_period("M").to_timestamp("M")
    returns.name = "live_returns"
    return returns


def load_live_positions_history(output_dir: str = "output") -> pd.DataFrame:
    """
    Load all position snapshots from output/live/positions/*.csv.
    Returns DataFrame with columns: date, ticker, quantity, avg_cost.
    """
    pos_dir = Path(output_dir) / _LIVE_DIR / _POSITIONS_DIR
    if not pos_dir.exists():
        return pd.DataFrame()

    frames = []
    for f in sorted(pos_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f, index_col=0)
            df.insert(0, "date", f.stem)
            frames.append(df)
        except Exception as exc:
            logger.debug("Could not load positions file %s: %s", f, exc)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def blend_returns_with_live(
    backtest_returns: dict,
    output_dir: str = "output",
) -> pd.Series:
    """
    Build a blended combined return series for the main backtesting reports:
      - dates <  live_start : Sharpe-weighted (or equal-weighted) backtest combined
      - dates >= live_start : actual IBKR NAV monthly returns

    Parameters
    ----------
    backtest_returns : {universe_name: pd.Series of monthly returns}
    output_dir       : root output directory (live/ subfolder is read automatically)

    Returns a single pd.Series with a continuous DatetimeIndex.
    The calling chart function can use it as a drop-in replacement for the
    normal equal-weighted combined series.
    """
    live = compute_live_monthly_returns(output_dir)

    if not backtest_returns:
        return live if not live.empty else pd.Series(dtype=float, name="combined")

    # Equal-weighted backtest combined (same as _combined_returns in reporting.py)
    df = pd.concat(backtest_returns, axis=1).dropna(how="all")
    bt_combined = df.mean(axis=1)
    bt_combined.name = "combined"

    if live.empty:
        return bt_combined   # no live data yet → pure backtest

    live_start = live.index[0]
    bt_slice   = bt_combined[bt_combined.index < live_start]

    # For months in the live period where live NAV is missing (e.g. May skipped),
    # fill from backtest if available.  Live data always takes precedence.
    bt_in_live = bt_combined[
        (bt_combined.index >= live_start) & (bt_combined.index <= live.index[-1])
    ]
    if not bt_in_live.empty:
        gap_filled = bt_in_live.copy()
        for idx, val in live.items():
            gap_filled[idx] = val          # live overrides backtest for same month
        live_combined = gap_filled
    else:
        live_combined = live

    blended = pd.concat([bt_slice, live_combined]).sort_index()
    blended.name = "combined"
    # Store live_start so chart functions use the correct transition date (not
    # the last individual-universe backtest date which may be later).
    blended.attrs["live_start"] = live_start.isoformat()

    n_bt   = len(bt_slice)
    n_live = len(live_combined)
    logger.info(
        "Blended combined returns: %d backtest months + %d live/gap-filled months (from %s)",
        n_bt, n_live, live_start.strftime("%Y-%m"),
    )
    return blended


def compute_live_analytics(returns: pd.Series, rf_annual: float = 0.045) -> dict:
    """Compute standard performance metrics from a monthly returns Series."""
    if returns.empty or len(returns) < 2:
        return {}

    cum      = (1 + returns).cumprod()
    total    = float(cum.iloc[-1] - 1)
    n        = len(returns)
    ann      = float((1 + total) ** (12 / n) - 1) if n >= 1 else 0.0
    vol      = float(returns.std() * np.sqrt(12))
    sharpe   = float((ann - rf_annual) / vol) if vol > 0 else float("nan")
    drawdown = float((cum / cum.cummax() - 1).min())
    calmar   = float(ann / abs(drawdown)) if drawdown < 0 else float("nan")

    return {
        "total_return":  total,
        "annual_return": ann,
        "annual_vol":    vol,
        "sharpe_ratio":  sharpe,
        "max_drawdown":  drawdown,
        "calmar_ratio":  calmar,
        "n_months":      n,
        "start_date":    str(returns.index[0].date()),
        "end_date":      str(returns.index[-1].date()),
    }


# --------------------------------------------------------------------------- #
#  Report generation                                                           #
# --------------------------------------------------------------------------- #

def generate_live_reports(
    output_dir: str  = "output",
    rf_annual:  float = 0.045,
) -> None:
    """
    Generate all live reports in output/live/:
      - nav_history.csv (source data, already exists)
      - monthly_returns.csv
      - analytics.json
      - equity_curve.png
      - monthly_heatmap.png      (requires ≥ 2 months)
      - rolling_sharpe.png       (requires > 12 months)
      - report.png               (one-page combined)
    """
    from .reporting import _render_heatmap, _render_rolling_sharpe, _render_heatmap_ax

    try:
        import matplotlib
        matplotlib.use("Agg")   # headless – no display needed
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.error("matplotlib is required for live reports. pip install matplotlib")
        return

    live_dir = Path(output_dir) / _LIVE_DIR
    live_dir.mkdir(parents=True, exist_ok=True)

    # ── Load returns ──────────────────────────────────────────────────────── #
    returns = compute_live_monthly_returns(output_dir)
    if returns.empty:
        logger.warning(
            "No NAV history found in %s/%s. "
            "NAV snapshots are recorded automatically after each rebalancing.",
            output_dir, _LIVE_DIR,
        )
        return

    logger.info(
        "Live returns loaded: %d months  (%s → %s)",
        len(returns),
        returns.index[0].strftime("%Y-%m"),
        returns.index[-1].strftime("%Y-%m"),
    )

    # ── Analytics ─────────────────────────────────────────────────────────── #
    analytics = compute_live_analytics(returns, rf_annual)

    with open(live_dir / "analytics.json", "w") as f:
        json.dump(analytics, f, indent=2)

    returns.to_frame("return").to_csv(live_dir / "monthly_returns.csv")

    _print_live_analytics(analytics)

    # ── Equity curve ──────────────────────────────────────────────────────── #
    _plot_live_equity_curve(
        returns, analytics,
        save_path=str(live_dir / "equity_curve.png"),
        plt=plt, gridspec=gridspec,
    )

    # ── Monthly heatmap ────────────────────────────────────────────────────── #
    if len(returns) >= 2:
        _render_heatmap(
            returns,
            title="Taurus Live – Monthly Returns Heatmap",
            plt=plt,
            save_path=str(live_dir / "monthly_heatmap.png"),
        )
        logger.info("Live monthly heatmap  → %s/monthly_heatmap.png", live_dir)

    # ── Rolling Sharpe ─────────────────────────────────────────────────────── #
    window = 12
    if len(returns) > window:
        _render_rolling_sharpe(
            {None: returns},
            rf_m=rf_annual / 12,
            window=window,
            title=f"Taurus Live – Rolling {window}-Month Sharpe Ratio",
            plt=plt,
            save_path=str(live_dir / "rolling_sharpe.png"),
        )
        logger.info("Live rolling Sharpe   → %s/rolling_sharpe.png", live_dir)

    # ── One-page combined report ───────────────────────────────────────────── #
    _generate_live_one_page_report(
        returns=returns,
        analytics=analytics,
        live_dir=live_dir,
        plt=plt,
        gridspec=gridspec,
        window=window,
    )
    logger.info("Live report           → %s/report.png", live_dir)


def _print_live_analytics(analytics: dict) -> None:
    logger.info("=" * 55)
    logger.info("LIVE PORTFOLIO PERFORMANCE")
    logger.info("=" * 55)
    logger.info("  Period          : %s → %s",
                analytics.get("start_date", "?"), analytics.get("end_date", "?"))
    logger.info("  Total return    : %+.2f%%", analytics.get("total_return",  0) * 100)
    logger.info("  Annual return   : %+.2f%%", analytics.get("annual_return", 0) * 100)
    logger.info("  Annual vol      : %.2f%%",  analytics.get("annual_vol",    0) * 100)
    logger.info("  Sharpe ratio    : %.2f",    analytics.get("sharpe_ratio",  float("nan")))
    logger.info("  Max drawdown    : %.2f%%",  analytics.get("max_drawdown",  0) * 100)
    logger.info("  Calmar ratio    : %.2f",    analytics.get("calmar_ratio",  float("nan")))
    logger.info("  Months live     : %d",      analytics.get("n_months", 0))
    logger.info("=" * 55)


def _plot_live_equity_curve(
    returns: pd.Series,
    analytics: dict,
    save_path: str,
    plt,
    gridspec,
) -> None:
    cum = (1 + returns).cumprod()
    dd  = cum / cum.cummax() - 1

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(cum.index, cum.values, linewidth=1.8, color="#1f77b4", label="Taurus Live")
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cumulative Return", fontsize=11)
    ax1.set_title(
        f"Taurus Live – Equity Curve  "
        f"({analytics.get('start_date','?')} → {analytics.get('end_date','?')})",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Live equity curve     → %s", save_path)


def _generate_live_one_page_report(
    returns:   pd.Series,
    analytics: dict,
    live_dir:  Path,
    plt,
    gridspec,
    window:    int = 12,
) -> None:
    """
    One-page report mirroring generate_combined_report() layout:
      ┌─ Performance table (left) + Equity curve + drawdown (right)
      ├─ Rolling 12-month Sharpe
      └─ Monthly returns heatmap
    """
    from .reporting import _render_heatmap_ax, _rolling_sharpe

    n_years = returns.index.year.nunique() if not returns.empty else 2
    hmap_h  = max(3, n_years * 0.65)
    fig_h   = 5 + 4 + hmap_h

    fig = plt.figure(figsize=(18, fig_h), facecolor="#f8f8f8")
    outer = gridspec.GridSpec(
        3, 1,
        height_ratios=[5, 4, hmap_h],
        hspace=0.35,
        left=0.06, right=0.97, top=0.95, bottom=0.04,
    )

    # ── Row 0: table + equity curve ───────────────────────────────────────── #
    row0   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              wspace=0.08, width_ratios=[1, 2])
    ax_tbl = fig.add_subplot(row0[0])
    eq_gs  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=row0[1],
                                              height_ratios=[3, 1], hspace=0.06)
    ax_eq  = fig.add_subplot(eq_gs[0])
    ax_dd  = fig.add_subplot(eq_gs[1], sharex=ax_eq)

    # Stats table
    ax_tbl.axis("off")
    ax_tbl.set_title(
        f"Taurus Live Portfolio\n"
        f"{analytics.get('start_date','?')} → {analytics.get('end_date','?')}",
        fontsize=13, fontweight="bold", pad=8, loc="left",
    )

    def _fmt(v, pct=True, signed=True):
        if v != v:   # nan
            return "—"
        if pct:
            return f"{v*100:+.2f}%" if signed else f"{v*100:.2f}%"
        return f"{v:.2f}"

    rows_data = [
        ["Total Return",   _fmt(analytics.get("total_return",  float("nan")))],
        ["Annual Return",  _fmt(analytics.get("annual_return", float("nan")))],
        ["Annual Vol",     _fmt(analytics.get("annual_vol",    float("nan")), signed=False)],
        ["Sharpe Ratio",   _fmt(analytics.get("sharpe_ratio",  float("nan")), pct=False)],
        ["Max Drawdown",   _fmt(analytics.get("max_drawdown",  float("nan")))],
        ["Calmar Ratio",   _fmt(analytics.get("calmar_ratio",  float("nan")), pct=False)],
        ["Months Live",    str(analytics.get("n_months", 0))],
    ]

    tbl = ax_tbl.table(
        cellText=rows_data, colLabels=["Metric", "Value"],
        cellLoc="center", loc="upper left", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(2):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Equity curve
    cum = (1 + returns).cumprod()
    dd  = cum / cum.cummax() - 1
    ax_eq.plot(cum.index, cum.values, linewidth=2.0, color="#1f77b4", label="Taurus Live")
    ax_eq.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")
    ax_eq.set_title("Live Equity Curve", fontsize=11, fontweight="bold")
    ax_eq.set_ylabel("Cumulative Return")
    ax_eq.legend(fontsize=8, loc="upper left")
    ax_eq.grid(alpha=0.25)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    ax_dd.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.45, label="Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(fontsize=7, loc="lower left")
    ax_dd.grid(alpha=0.25)

    # ── Row 1: Rolling Sharpe ─────────────────────────────────────────────── #
    ax_sh = fig.add_subplot(outer[1])
    if len(returns) > window:
        roll_sh = _rolling_sharpe(returns, 0.0, window)
        ax_sh.plot(roll_sh.index, roll_sh.values, linewidth=2.0,
                   color="#1f77b4", label="Live Sharpe")
        ax_sh.fill_between(roll_sh.index, roll_sh.values, 0,
                           where=roll_sh.values > 0, alpha=0.25, color="#1f77b4")
        ax_sh.fill_between(roll_sh.index, roll_sh.values, 0,
                           where=roll_sh.values < 0, alpha=0.25, color="#d62728")
    else:
        ax_sh.text(
            0.5, 0.5,
            f"Rolling Sharpe requires > {window} months of data\n(currently {len(returns)} months)",
            ha="center", va="center", transform=ax_sh.transAxes,
            fontsize=10, color="grey",
        )
    ax_sh.axhline(0, color="grey", linewidth=0.7)
    ax_sh.axhline(1, color="#1f77b4", linewidth=0.8, linestyle=":", label="Sharpe = 1")
    ax_sh.set_title(f"Rolling {window}-Month Sharpe Ratio", fontsize=11, fontweight="bold")
    ax_sh.set_ylabel("Sharpe (annualised)")
    ax_sh.set_xlabel("Date")
    ax_sh.legend(fontsize=7, loc="upper left")
    ax_sh.grid(alpha=0.25)

    # ── Row 2: Monthly heatmap ─────────────────────────────────────────────── #
    ax_hm = fig.add_subplot(outer[2])
    if len(returns) >= 2:
        _render_heatmap_ax(returns, ax=ax_hm, title="Live Monthly Returns Heatmap")
    else:
        ax_hm.text(
            0.5, 0.5, "Heatmap requires at least 2 months of data",
            ha="center", va="center", transform=ax_hm.transAxes,
            fontsize=10, color="grey",
        )
        ax_hm.axis("off")

    fig.savefig(str(live_dir / "report.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
