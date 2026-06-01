"""
Taurus – Reporting and visualisation helpers.

Produces:
  • Console summary table (analytics dict → formatted text)
  • Equity-curve and drawdown chart
  • Monthly-returns heatmap
  • Positions summary table
  • Rolling Sharpe chart
  • Combined multi-universe versions of the above

All chart functions return matplotlib Figure objects so callers can save
or display them without coupling to a specific backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

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


# Colour palette for multi-universe charts (cycles if > 6 universes)
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _combined_returns(results: Dict[str, "BacktestResult"]) -> pd.Series:
    """Equal-weighted average of all universe monthly returns."""
    frames = {name: r.portfolio_returns() for name, r in results.items()}
    df = pd.concat(frames, axis=1).dropna(how="all")
    return df.mean(axis=1)


# --------------------------------------------------------------------------- #
#  Text summary                                                                #
# --------------------------------------------------------------------------- #

def print_analytics(analytics: dict) -> None:
    """Print a formatted performance summary to stdout."""
    universe = analytics.get("universe", "")
    header   = f"  Taurus Strategy – {universe.upper() + ' ' if universe else ''}Performance Summary"
    lines = [
        "=" * 50,
        header,
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
#  Single-universe charts                                                      #
# --------------------------------------------------------------------------- #

def plot_equity_curve(
    result: "BacktestResult",
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Taurus Long/Short Equity",
    save_path: Optional[str] = None,
):
    """Plot cumulative returns and drawdown for one universe."""
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


def plot_monthly_heatmap(
    result: "BacktestResult",
    save_path: Optional[str] = None,
):
    """Matplotlib heatmap of monthly returns by year × month."""
    plt, _ = _plt()

    ret = result.portfolio_returns()
    if ret.empty:
        return

    _render_heatmap(ret, title="Monthly Returns Heatmap", plt=plt,
                    save_path=save_path)


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

    rf_m = result.cfg.rf_monthly
    series = {result.cfg.__class__.__name__: ret}  # single series
    _render_rolling_sharpe(
        {None: ret}, rf_m, window=window, title=f"Rolling {window}-Month Sharpe Ratio",
        plt=plt, save_path=save_path,
    )


# --------------------------------------------------------------------------- #
#  Combined multi-universe charts                                              #
# --------------------------------------------------------------------------- #

def plot_combined_equity_curve(
    results: Dict[str, "BacktestResult"],
    save_path: Optional[str] = None,
    combined_override: Optional[pd.Series] = None,
):
    """
    Equity curve showing each universe + combined line.

    Parameters
    ----------
    results            : {universe_name: BacktestResult}
    save_path          : if given, save figure to this path
    combined_override  : pre-computed blended series (backtest + live).
                         When provided, replaces equal-weighted combined.
    """
    plt, gridspec = _plt()

    combined = combined_override.dropna() if combined_override is not None else _combined_returns(results)
    if combined.empty:
        logger.warning("No combined returns to plot.")
        return

    cum_combined = (1 + combined).cumprod()
    dd_combined  = cum_combined / cum_combined.cummax() - 1

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Compute live_start once so individual lines can be split
    live_start = None
    if combined_override is not None and results:
        bt_frames = [r.portfolio_returns() for r in results.values()]
        if bt_frames:
            last_bt   = pd.concat(bt_frames, axis=1).dropna(how="all").index[-1]
            live_dates = combined_override.index[combined_override.index > last_bt]
            if len(live_dates):
                live_start = live_dates[0]

    # Individual universe lines — bright in backtest, faded after live_start
    for i, (name, result) in enumerate(results.items()):
        ret = result.portfolio_returns()
        if ret.empty:
            continue
        cum   = (1 + ret).cumprod()
        color = _PALETTE[i % len(_PALETTE)]
        if live_start is not None:
            cum_bt   = cum[cum.index <= last_bt]
            cum_post = cum[cum.index >  last_bt]
            ax1.plot(cum_bt.index,   cum_bt.values,   linewidth=1.2, color=color,
                     linestyle="--", alpha=0.65, label=name.upper())
            if not cum_post.empty:
                ax1.plot(cum_post.index, cum_post.values, linewidth=0.8, color="grey",
                         linestyle="--", alpha=0.25)
        else:
            ax1.plot(cum.index, cum.values, linewidth=1.2, color=color,
                     linestyle="--", alpha=0.7, label=name.upper())

    label = "Combined (Backtest → Live)" if combined_override is not None else "Combined (equal-weighted)"
    ax1.plot(cum_combined.index, cum_combined.values, linewidth=2.2,
             color="black", label=label)

    # Mark backtest → live transition + shade live region
    if live_start is not None:
        x_end = cum_combined.index[-1]
        ax1.axvspan(live_start, x_end, alpha=0.06, color="#e67e22", zorder=0)
        ax2.axvspan(live_start, x_end, alpha=0.06, color="#e67e22", zorder=0)
        ax1.axvline(live_start, color="#e67e22", linewidth=1.4, linestyle="--",
                    label=f"Live start ({live_start.strftime('%Y-%m')})")
        ax2.axvline(live_start, color="#e67e22", linewidth=1.4, linestyle="--")

    ax1.set_ylabel("Cumulative Return", fontsize=11)
    ax1.set_title("Taurus – Combined Equity Curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")

    ax2.fill_between(dd_combined.index, dd_combined.values, 0,
                     color="#d62728", alpha=0.5, label="Combined Drawdown")
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left")
    plt.setp(ax1.get_xticklabels(), visible=False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Combined equity curve saved to %s", save_path)
    return fig


def plot_combined_heatmap(
    results: Dict[str, "BacktestResult"],
    save_path: Optional[str] = None,
    combined_override: Optional[pd.Series] = None,
):
    """
    Multi-panel heatmap: one row of subplots per universe + combined row.

    Parameters
    ----------
    results            : {universe_name: BacktestResult}
    save_path          : if given, save figure to this path
    combined_override  : pre-computed blended series (backtest + live) for the
                         Combined row. When provided, live months appear in the heatmap.
    """
    plt, _ = _plt()

    combined = combined_override.dropna() if combined_override is not None else _combined_returns(results)
    combined_label = "COMBINED (Backtest→Live)" if combined_override is not None else "COMBINED"
    all_series = {**{k: v.portfolio_returns() for k, v in results.items()},
                  combined_label: combined}

    n = len(all_series)
    fig, axes = plt.subplots(n, 1, figsize=(16, max(3, n * 2.5)))
    if n == 1:
        axes = [axes]

    fig.suptitle("Monthly Returns Heatmap", fontsize=14, fontweight="bold", y=1.01)

    for ax, (name, ret) in zip(axes, all_series.items()):
        if ret.empty:
            ax.set_visible(False)
            continue
        _render_heatmap_ax(ret, ax=ax, title=name.upper())

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Combined heatmap saved to %s", save_path)
    return fig


def plot_combined_rolling_sharpe(
    results: Dict[str, "BacktestResult"],
    window: int = 12,
    save_path: Optional[str] = None,
    combined_override: Optional[pd.Series] = None,
):
    """
    Rolling Sharpe for each universe + combined, all on a single chart.

    Parameters
    ----------
    results            : {universe_name: BacktestResult}
    window             : rolling window in months (default 12)
    combined_override  : pre-computed blended series (backtest + live).
                         When provided, the combined rolling Sharpe extends into live months.
    """
    plt, _ = _plt()

    combined = combined_override.dropna() if combined_override is not None else _combined_returns(results)
    rf_m = next(iter(results.values())).cfg.rf_monthly

    combined_label = "Combined (B→L)" if combined_override is not None else "Combined"

    # Compute live_start once
    live_start = None
    last_bt    = None
    if combined_override is not None and results:
        bt_frames = [r.portfolio_returns() for r in results.values()]
        if bt_frames:
            last_bt    = pd.concat(bt_frames, axis=1).dropna(how="all").index[-1]
            live_dates = combined_override.index[combined_override.index > last_bt]
            if len(live_dates):
                live_start = live_dates[0]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Individual universe lines (pure backtest)
    for i, (name, ret) in enumerate(results.items()):
        if len(ret) < window + 1:
            continue
        roll_sh = _rolling_sharpe(ret, rf_m, window)
        color   = _PALETTE[i % len(_PALETTE)]
        if live_start is not None:
            # Bright up to live_start, then fade to grey
            sh_bt   = roll_sh[roll_sh.index <= last_bt]
            sh_post = roll_sh[roll_sh.index >  last_bt]
            ax.plot(sh_bt.index,   sh_bt.values,   linewidth=1.2, color=color,
                    linestyle="--", alpha=0.65, label=name.upper())
            if not sh_post.empty:
                ax.plot(sh_post.index, sh_post.values, linewidth=0.8, color="grey",
                        linestyle="--", alpha=0.20)
        else:
            ax.plot(roll_sh.index, roll_sh.values, linewidth=1.2, color=color,
                    linestyle="--", alpha=0.65, label=name.upper())

    # Combined line (blended backtest + live)
    if len(combined) >= window + 1:
        roll_c = _rolling_sharpe(combined, rf_m, window)
        ax.plot(roll_c.index, roll_c.values, linewidth=2.2, color="black",
                linestyle="-", alpha=0.9, label=combined_label.upper())

    # Shade live region + vertical line
    if live_start is not None:
        x_end = combined.index[-1]
        ax.axvspan(live_start, x_end, alpha=0.07, color="#e67e22", zorder=0)
        ax.axvline(live_start, color="#e67e22", linewidth=1.4,
                   linestyle="--", label=f"Live start ({live_start.strftime('%Y-%m')})")

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axhline(1, color="#1f77b4", linewidth=0.8, linestyle=":", label="Sharpe=1")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio – All Universes",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe (annualised)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Combined rolling Sharpe saved to %s", save_path)
    return fig


def combined_positions_df(
    results: Dict[str, "BacktestResult"],
) -> pd.DataFrame:
    """
    Merge all universes' position DataFrames into one, adding a 'universe' column.

    Columns: universe, date, ticker, leg, weight
    """
    frames = []
    for name, result in results.items():
        df = result.positions_df()
        if not df.empty:
            df.insert(0, "universe", name)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _rolling_sharpe(ret: pd.Series, rf_m: float, window: int) -> pd.Series:
    roll_m  = ret.rolling(window).mean()
    roll_s  = ret.rolling(window).std()
    return (roll_m - rf_m) / roll_s.replace(0, np.nan) * np.sqrt(12)


def _render_heatmap(ret: pd.Series, title: str, plt, save_path=None):
    """Standalone heatmap figure for a single return series."""
    fig, ax = plt.subplots(figsize=(14, max(3, ret.index.year.nunique() * 0.6)))
    _render_heatmap_ax(ret, ax=ax, title=title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _render_heatmap_ax(ret: pd.Series, ax, title: str = "") -> None:
    """Render a year×month heatmap onto an existing Axes, with annual Total column."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df = ret.to_frame("return")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Annual total: compound monthly returns for each year
    annual = pivot.apply(
        lambda row: (1 + row.dropna()).prod() - 1, axis=1
    )

    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = max(abs(vals).max(), 0.01) if len(vals) else 0.01

    # Plot heatmap on columns 0..11, leave column 12 for Total
    n_rows, n_cols = pivot.shape[0], 12
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto", extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])

    # Month labels + cell text
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(list(pivot.columns), fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=6, color="black")

    try:
        plt.colorbar(im, ax=ax, format=lambda x, _: f"{x:.1%}")
    except Exception:
        pass

    # ── Total column drawn as text boxes outside the imshow area ─────────── #
    ax.set_xlim(-0.5, n_cols + 0.6)   # extend x-axis to fit Total column

    # Header "Total"
    ax.text(n_cols + 0.05, -0.65, "Total", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2c3e50", edgecolor="none"))

    for i, (year, tot) in enumerate(annual.items()):
        color = "#2ecc71" if tot >= 0 else "#e74c3c"   # green / red
        ax.text(n_cols + 0.05, i, f"{tot:+.1%}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none"))


def _render_rolling_sharpe(series_dict, rf_m, window, title, plt, save_path=None):
    """Standalone rolling Sharpe figure (used by single-universe plot_rolling_sharpe)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (name, ret) in enumerate(series_dict.items()):
        if len(ret) < window + 1:
            continue
        roll_sh = _rolling_sharpe(ret, rf_m, window)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(roll_sh.index, roll_sh.values, linewidth=1.5, color=color,
                label=str(name) if name else "Strategy")
        ax.fill_between(roll_sh.index, roll_sh.values, 0,
                        where=roll_sh.values > 0, alpha=0.25, color=color)
        ax.fill_between(roll_sh.index, roll_sh.values, 0,
                        where=roll_sh.values < 0, alpha=0.25, color="#d62728")

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axhline(1, color="#1f77b4", linewidth=0.8, linestyle="--", label="Sharpe=1")
    ax.set_title(title, fontsize=13, fontweight="bold")
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

def generate_combined_report(
    results:           Dict[str, "BacktestResult"],
    all_analytics:     dict,
    save_path:         Optional[str] = None,
    start:             str = "",
    end:               str = "",
    combined_override: Optional[pd.Series] = None,
):
    """
    One-page PDF/PNG report:
      ┌──────────────────────────────────────────────────┐
      │  Header + performance summary table              │
      ├──────────────────────┬───────────────────────────┤
      │  Combined equity     │  Rolling Sharpe           │
      │  curve + drawdown    │  (all universes)          │
      ├──────────────────────┴───────────────────────────┤
      │  Combined monthly returns heatmap                │
      └──────────────────────────────────────────────────┘

    Parameters
    ----------
    combined_override : optional pre-computed combined return series (e.g. blended
                        backtest + live).  When provided, replaces the Sharpe-weighted
                        combined computation for the equity curve, heatmap and rolling
                        Sharpe.  Individual universe lines are unaffected.
    """
    plt, gridspec = _plt()

    # ── Build combined return series ─────────────────────────────────────── #
    ret_map = {name: r.portfolio_returns() for name, r in results.items()}
    df_ret  = pd.concat(ret_map, axis=1).sort_index()
    window  = 12

    if combined_override is not None:
        # Use the caller-supplied series (e.g. backtest + live blend)
        combined_ret = combined_override.dropna()
    else:
        # Sharpe-weighted rolling combination (original logic)
        combined_monthly = []
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
                    sharpes[name] = ann / vol if vol > 0 else 0.0
            raw   = {k: max(v, 0.0) for k, v in sharpes.items()}
            total = sum(raw.values())
            w     = ({k: v / total for k, v in raw.items()} if total > 1e-9
                     else {k: 1.0 / len(df_ret.columns) for k in df_ret.columns})
            row_ret = sum(
                w[name] * df_ret.loc[t, name]
                for name in df_ret.columns if not pd.isna(df_ret.loc[t, name])
            )
            combined_monthly.append(row_ret)
        combined_ret = pd.Series(combined_monthly, index=df_ret.index).dropna()

    cum_c = (1 + combined_ret).cumprod()
    dd_c  = cum_c / cum_c.cummax() - 1

    # ── Layout ───────────────────────────────────────────────────────────── #
    n_years   = df_ret.index.year.nunique() if not df_ret.empty else 5
    hmap_h    = max(3, n_years * 0.65)
    fig_h     = 5 + 5 + hmap_h        # header+equity / rolling_sharpe / heatmap

    fig = plt.figure(figsize=(18, fig_h), facecolor="#f8f8f8")
    outer = gridspec.GridSpec(
        3, 1,
        height_ratios=[5, 4, hmap_h],
        hspace=0.35,
        left=0.06, right=0.97, top=0.95, bottom=0.04,
    )

    # ── Row 0: table (left) + equity curve (right) ───────────────────────── #
    row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.08, width_ratios=[1, 2])
    ax_tbl  = fig.add_subplot(row0[0])
    ax_eq_g = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=row0[1],
                                               height_ratios=[3, 1], hspace=0.06)
    ax_eq   = fig.add_subplot(ax_eq_g[0])
    ax_dd   = fig.add_subplot(ax_eq_g[1], sharex=ax_eq)

    # Stats table
    ax_tbl.axis("off")
    date_label = f"{start} → {end}" if start and end else ""
    ax_tbl.set_title(
        f"Taurus Strategy\n{date_label}",
        fontsize=13, fontweight="bold", pad=8, loc="left",
    )

    col_headers = ["Universe", "Return", "Annual", "Sharpe", "MaxDD"]
    rows_data   = []
    for name, a in all_analytics.items():
        if name == "combined":
            continue
        rows_data.append([
            name.upper(),
            f"{a.get('total_return',  0)*100:+.1f}%",
            f"{a.get('annual_return', 0)*100:+.1f}%",
            f"{a.get('sharpe_ratio',  0):.2f}",
            f"{a.get('max_drawdown',  0)*100:.1f}%",
        ])

    # Combined row from analytics if present
    ca = all_analytics.get("combined", {})
    if ca:
        rows_data.append([
            "COMBINED",
            f"{ca.get('total_return',  0)*100:+.1f}%",
            f"{ca.get('annual_return', 0)*100:+.1f}%",
            f"{ca.get('sharpe_ratio',  0):.2f}",
            f"{ca.get('max_drawdown',  0)*100:.1f}%",
        ])
        # NAV allocation sub-rows
        nav = ca.get("nav_allocation", {})
        if nav:
            rows_data.append(["─" * 10, "", "", "", ""])
            rows_data.append(["NAV Allocation", "", "", "", ""])
            for uname, w in sorted(nav.items(), key=lambda x: -x[1]):
                rows_data.append([f"  {uname.upper()}", f"{w*100:.1f}%", "", "", ""])

    tbl = ax_tbl.table(
        cellText   = rows_data,
        colLabels  = col_headers,
        cellLoc    = "center",
        loc        = "upper left",
        bbox       = [0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style header row
    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style combined row
    combined_row_idx = next(
        (i + 1 for i, r in enumerate(rows_data) if r[0] == "COMBINED"), None
    )
    if combined_row_idx is not None:
        for j in range(len(col_headers)):
            tbl[combined_row_idx, j].set_facecolor("#d5e8d4")
            tbl[combined_row_idx, j].set_text_props(fontweight="bold")

    # Detect live_start for visual split
    _live_start = None
    _last_bt    = None
    if combined_override is not None and not df_ret.empty:
        _last_bt   = df_ret.dropna(how="all").index[-1]
        _live_d    = combined_override.index[combined_override.index > _last_bt]
        if len(_live_d):
            _live_start = _live_d[0]

    # Equity curve — bright in backtest, grey/faded in live period
    for i, (name, result) in enumerate(results.items()):
        ret = result.portfolio_returns()
        if ret.empty:
            continue
        cum   = (1 + ret).cumprod()
        color = _PALETTE[i % len(_PALETTE)]
        if _live_start is not None:
            cum_bt   = cum[cum.index <= _last_bt]
            cum_post = cum[cum.index >  _last_bt]
            ax_eq.plot(cum_bt.index,   cum_bt.values,   linewidth=1.2, color=color,
                       linestyle="--", alpha=0.65, label=name.upper())
            if not cum_post.empty:
                ax_eq.plot(cum_post.index, cum_post.values, linewidth=0.8, color="grey",
                           linestyle="--", alpha=0.20)
        else:
            ax_eq.plot(cum.index, cum.values, linewidth=1.2, color=color,
                       linestyle="--", alpha=0.65, label=name.upper())

    combined_label = "COMBINED (Live)" if combined_override is not None else "COMBINED (Sharpe-wtd)"
    ax_eq.plot(cum_c.index, cum_c.values, linewidth=2.2,
               color="black", label=combined_label)
    ax_eq.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")

    if _live_start is not None:
        x_end = cum_c.index[-1]
        ax_eq.axvspan(_live_start, x_end, alpha=0.06, color="#e67e22", zorder=0)
        ax_dd.axvspan(_live_start, x_end, alpha=0.06, color="#e67e22", zorder=0)
        ax_eq.axvline(_live_start, color="#e67e22", linewidth=1.2,
                      linestyle="--", label=f"Live start ({_live_start.strftime('%Y-%m')})")
        ax_dd.axvline(_live_start, color="#e67e22", linewidth=1.2, linestyle="--")

    title_suffix = " (Backtest → Live)" if combined_override is not None else ""
    ax_eq.set_title(f"Equity Curve – Combined{title_suffix}", fontsize=11, fontweight="bold")
    ax_eq.set_ylabel("Cumulative Return")
    ax_eq.legend(fontsize=7, loc="upper left")
    ax_eq.grid(alpha=0.25)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    ax_dd.fill_between(dd_c.index, dd_c.values, 0, color="#d62728", alpha=0.45, label="Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(fontsize=7, loc="lower left")
    ax_dd.grid(alpha=0.25)

    # ── Row 1: Rolling Sharpe ─────────────────────────────────────────────── #
    ax_sh = fig.add_subplot(outer[1])

    for i, (name, result) in enumerate(results.items()):
        ret = result.portfolio_returns()
        if len(ret) < window + 1:
            continue
        roll_sh = _rolling_sharpe(ret, 0.0, window)
        color   = _PALETTE[i % len(_PALETTE)]
        if _live_start is not None:
            sh_bt   = roll_sh[roll_sh.index <= _last_bt]
            sh_post = roll_sh[roll_sh.index >  _last_bt]
            ax_sh.plot(sh_bt.index,   sh_bt.values,   linewidth=1.2, color=color,
                       linestyle="--", alpha=0.65, label=name.upper())
            if not sh_post.empty:
                ax_sh.plot(sh_post.index, sh_post.values, linewidth=0.8, color="grey",
                           linestyle="--", alpha=0.20)
        else:
            ax_sh.plot(roll_sh.index, roll_sh.values, linewidth=1.2, color=color,
                       linestyle="--", alpha=0.65, label=name.upper())

    if len(combined_ret) > window:
        roll_c = _rolling_sharpe(combined_ret, 0.0, window)
        ax_sh.plot(roll_c.index, roll_c.values, linewidth=2.2,
                   color="black", label="COMBINED")

    if _live_start is not None:
        ax_sh.axvspan(_live_start, combined_ret.index[-1], alpha=0.07, color="#e67e22", zorder=0)
        ax_sh.axvline(_live_start, color="#e67e22", linewidth=1.2,
                      linestyle="--", label=f"Live start ({_live_start.strftime('%Y-%m')})")

    ax_sh.axhline(0, color="grey", linewidth=0.7)
    ax_sh.axhline(1, color="#1f77b4", linewidth=0.8, linestyle=":", label="Sharpe=1")
    ax_sh.set_title(f"Rolling {window}-Month Sharpe Ratio", fontsize=11, fontweight="bold")
    ax_sh.set_ylabel("Sharpe (annualised)")
    ax_sh.set_xlabel("Date")
    ax_sh.legend(fontsize=7, loc="upper left")
    ax_sh.grid(alpha=0.25)

    # ── Row 2: Combined monthly heatmap ──────────────────────────────────── #
    ax_hm = fig.add_subplot(outer[2])
    _render_heatmap_ax(combined_ret, ax=ax_hm,
                       title="Combined Monthly Returns Heatmap (Sharpe-Weighted)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Combined report saved to %s", save_path)

    plt.close(fig)
    return fig


def top_positions_table(result: "BacktestResult", n: int = 20) -> pd.DataFrame:
    """Return most recent rebalance's top N positions by |weight|."""
    if not result.snapshots:
        return pd.DataFrame()

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
