"""
Taurus – Reporting and visualisation helpers.

Produces:
  • Console summary table (analytics dict → formatted text)
  • Equity-curve and drawdown chart
  • Monthly-returns heatmap
  • Positions summary table
  • Rolling Sharpe chart

All chart functions return matplotlib Figure objects so callers can save
or display them without coupling to a specific backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .strategy import BacktestResult

logger = logging.getLogger(__name__)

# Lazy matplotlib import to avoid hard dependency in headless environments.
def _plt():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib") from e


# --------------------------------------------------------------------------- #
#  Text summary                                                                #
# --------------------------------------------------------------------------- #

def print_analytics(analytics: dict) -> None:
    """Print a formatted performance summary to stdout."""
    lines = [
        "=" * 50,
        "  Taurus Strategy – Performance Summary",
        "=" * 50,
        f"  Total return       : {analytics.get('total_return',  0):.2%}",
        f"  Annual return      : {analytics.get('annual_return', 0):.2%}",
        f"  Annual volatility  : {analytics.get('annual_vol',    0):.2%}",
        f"  Sharpe ratio       : {analytics.get('sharpe_ratio',  0):.2f}",
        f"  Max drawdown       : {analytics.get('max_drawdown',  0):.2%}",
        f"  Calmar ratio       : {analytics.get('calmar_ratio',  0):.2f}",
        f"  Months             : {analytics.get('n_months',      0):d}",
        f"  Rebalances         : {analytics.get('n_rebalances',  0):d}",
        "=" * 50,
    ]
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
#  Equity curve & drawdown                                                     #
# --------------------------------------------------------------------------- #

def plot_equity_curve(
    result: "BacktestResult",
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Taurus Long/Short Equity",
    save_path: Optional[str] = None,
):
    """
    Plot cumulative returns and drawdown.

    Parameters
    ----------
    result            : BacktestResult
    benchmark_returns : optional monthly benchmark (e.g. S&P 500) for comparison
    save_path         : if given, save figure to this path
    """
    plt, gridspec = _plt()

    ret = result.portfolio_returns()
    if ret.empty:
        logger.warning("No returns to plot.")
        return

    cum = (1 + ret).cumprod()
    dd  = cum / cum.cummax() - 1

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(cum.index, cum.values, linewidth=1.8, color="#1f77b4", label="Taurus")

    if benchmark_returns is not None:
        bench_cum = (1 + benchmark_returns.reindex(ret.index).fillna(0)).cumprod()
        ax1.plot(bench_cum.index, bench_cum.values, linewidth=1.2,
                 color="#aec7e8", linestyle="--", label="Benchmark")

    ax1.set_ylabel("Cumulative Return", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")

    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left")

    plt.setp(ax1.get_xticklabels(), visible=False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Equity curve saved to %s", save_path)

    return fig


# --------------------------------------------------------------------------- #
#  Monthly returns heatmap                                                     #
# --------------------------------------------------------------------------- #

def plot_monthly_heatmap(
    result: "BacktestResult",
    save_path: Optional[str] = None,
):
    """Matplotlib heatmap of monthly returns by year × month."""
    plt, _ = _plt()

    ret = result.portfolio_returns()
    if ret.empty:
        return

    df = ret.to_frame("return")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].max()), 0.01)

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, format=lambda x, _: f"{x:.1%}")

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=9)
    ax.set_title("Monthly Returns Heatmap", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=7, color="black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Rolling Sharpe                                                              #
# --------------------------------------------------------------------------- #

def plot_rolling_sharpe(
    result: "BacktestResult",
    window: int = 12,
    save_path: Optional[str] = None,
):
    """Plot rolling 12-month Sharpe ratio."""
    plt, _ = _plt()

    ret = result.portfolio_returns()
    if len(ret) < window + 1:
        return

    rf_m    = result.cfg.rf_monthly
    roll_m  = ret.rolling(window).mean()
    roll_s  = ret.rolling(window).std()
    roll_sh = (roll_m - rf_m) / roll_s.replace(0, np.nan) * np.sqrt(12)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(roll_sh.index, roll_sh.values, linewidth=1.5, color="#2ca02c")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axhline(1, color="#1f77b4", linewidth=0.8, linestyle="--", label="Sharpe=1")
    ax.fill_between(roll_sh.index, roll_sh.values, 0,
                    where=roll_sh.values > 0, alpha=0.3, color="#2ca02c")
    ax.fill_between(roll_sh.index, roll_sh.values, 0,
                    where=roll_sh.values < 0, alpha=0.3, color="#d62728")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe (annualised)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Top positions table                                                         #
# --------------------------------------------------------------------------- #

def top_positions_table(result: "BacktestResult", n: int = 20) -> pd.DataFrame:
    """
    Return a DataFrame of the most recent rebalance's top N positions by |weight|.
    """
    if not result.snapshots:
        return pd.DataFrame()

    # Find the most recent non-empty snapshot
    snap = None
    for s in reversed(result.snapshots):
        if not s.long_weights.empty or not s.short_weights.empty:
            snap = s
            break

    if snap is None:
        return pd.DataFrame()

    rows = []
    for t, w in snap.long_weights.items():
        rows.append({"ticker": t, "leg": "LONG",  "weight": w})
    for t, w in snap.short_weights.items():
        rows.append({"ticker": t, "leg": "SHORT", "weight": -w})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    return df.head(n).reset_index(drop=True)
