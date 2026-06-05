"""
Taurus – Factor model: CAPM + Fama-French 5/6 Factors → SML alpha.

Key design decisions
─────────────────────
• FF6 support:  When the factors DataFrame includes a "UMD" column
  (i.e. cfg.use_umd_factor=True), the regression automatically switches
  to 7 regressors [1, Mkt-RF, SMB, HML, RMW, CMA, UMD].  This strips
  the momentum premium from alpha → purer residual signal.

• Vectorised OLS:  All N stocks are regressed in a single matrix solve.
  X  (T × K) = [1, Mkt-RF, SMB, HML, RMW, CMA, (UMD)]
  Y  (T × N) = excess stock returns
  β  (K × N) = (X'X)⁻¹ X'Y  – solved once for all stocks simultaneously.

• Heteroskedasticity-consistent (HC1) standard errors for the alpha t-stat.

• Student-t critical value when cfg.return_df is set (conservative, fat tails).

Outputs
────────
• alpha        : float array (N,) – annualised intercept
• alpha_tstat  : float array (N,) – t-statistic of the intercept
• betas        : DataFrame (N × K-1) – factor loadings
• signal       : bool Series indexed by ticker – True iff |t| > threshold
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core vectorised OLS                                                         #
# --------------------------------------------------------------------------- #

def _ols_vectorised(
    X: np.ndarray,   # (T, K)
    Y: np.ndarray,   # (T, N)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordinary Least Squares for multiple dependent variables simultaneously.

    Returns
    -------
    beta      : (K, N)  – coefficient matrix
    residuals : (T, N)
    XtX_inv   : (K, K)  – precomputed for SE calculation
    """
    XtX_inv = np.linalg.pinv(X.T @ X)          # (K, K)
    beta = XtX_inv @ (X.T @ Y)                  # (K, N)
    residuals = Y - X @ beta                     # (T, N)
    return beta, residuals, XtX_inv


def _hc1_se_intercept(
    X: np.ndarray,          # (T, K)
    residuals: np.ndarray,  # (T, N)
    XtX_inv: np.ndarray,   # (K, K)
) -> np.ndarray:
    """
    HC1 heteroskedasticity-consistent SE for the intercept, for each of N stocks.

    Returns se_alpha : (N,)
    """
    T, K = X.shape
    df_correction = T / (T - K)
    q = X @ XtX_inv[:, 0]          # (T,) – hat vector for intercept
    e2 = residuals ** 2             # (T, N)
    V00 = (q[:, None] ** 2 * e2).sum(axis=0) * df_correction   # (N,)
    return np.sqrt(np.maximum(V00, 1e-16))


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def compute_ff5_alpha(
    returns: pd.DataFrame,        # (T, N) monthly stock returns (not excess)
    factors: pd.DataFrame,        # (T, 6 or 7) FF5/FF6 + RF columns
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Run the FF5 or FF6 regression for every stock simultaneously.

    Automatically uses FF6 when the factors DataFrame contains a "UMD" column
    and cfg.use_umd_factor=True.  The UMD factor strips the momentum premium
    from alpha, producing a purer residual signal for stock selection.

    Parameters
    ----------
    returns : monthly stock returns (RF is subtracted inside)
    factors : columns Mkt-RF, SMB, HML, RMW, CMA, RF [, UMD] (decimal)

    Returns
    -------
    DataFrame indexed by ticker with columns:
        alpha_monthly, alpha_annual, alpha_tstat,
        beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma [, beta_umd],
        r_squared, signal (bool), alpha_sign (+1/-1)
    """
    # ── Align ──────────────────────────────────────────────────────────── #
    aligned_idx = returns.index.intersection(factors.index)
    # Hard minimum: at least 24 months for a meaningful regression.
    # Soft minimum (cfg.min_obs, default 36): warn but proceed — 30-35 months
    # is still statistically reliable and avoids a complete blackout caused by
    # a single recently-listed constituent truncating the dropna() window.
    if len(aligned_idx) < 24:
        logger.warning(
            "Only %d overlapping observations (need at least 24) — skipping.",
            len(aligned_idx),
        )
        return pd.DataFrame()
    if len(aligned_idx) < cfg.min_obs:
        logger.warning(
            "Only %d overlapping observations (soft threshold %d) — proceeding with reduced sample.",
            len(aligned_idx), cfg.min_obs,
        )

    ret = returns.loc[aligned_idx]
    fac = factors.loc[aligned_idx]

    rf     = fac["RF"].values
    mkt_rf = fac["Mkt-RF"].values
    smb    = fac["SMB"].values
    hml    = fac["HML"].values
    rmw    = fac["RMW"].values
    cma    = fac["CMA"].values

    # ── FF6: use UMD when available and configured ──────────────────────── #
    use_umd = (
        getattr(cfg, "use_umd_factor", False)
        and "UMD" in fac.columns
        and fac["UMD"].notna().any()
    )

    # ── Excess returns ──────────────────────────────────────────────────── #
    Y = ret.values - rf[:, None]             # (T, N)

    # ── Design matrix ──────────────────────────────────────────────────── #
    T = len(aligned_idx)
    if use_umd:
        umd = fac["UMD"].fillna(0.0).values
        X = np.column_stack([np.ones(T), mkt_rf, smb, hml, rmw, cma, umd])  # (T, 7)
        model_label = "FF6"
    else:
        X = np.column_stack([np.ones(T), mkt_rf, smb, hml, rmw, cma])        # (T, 6)
        model_label = "FF5"

    # ── Vectorised OLS ─────────────────────────────────────────────────── #
    beta, residuals, XtX_inv = _ols_vectorised(X, Y)
    se_alpha = _hc1_se_intercept(X, residuals, XtX_inv)

    alpha_m = beta[0]                         # (N,) monthly
    alpha_a = (1 + alpha_m) ** 12 - 1        # annualised

    t_stats = alpha_m / np.where(se_alpha > 0, se_alpha, np.nan)

    # ── R² ─────────────────────────────────────────────────────────────── #
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)

    # ── Assemble ─────────────────────────────────────────────────────────── #
    tickers = ret.columns.tolist()
    out = {
        "alpha_monthly": alpha_m,
        "alpha_annual":  alpha_a,
        "alpha_tstat":   t_stats,
        "beta_mkt":  beta[1],
        "beta_smb":  beta[2],
        "beta_hml":  beta[3],
        "beta_rmw":  beta[4],
        "beta_cma":  beta[5],
        "r_squared": r2,
    }
    if use_umd:
        out["beta_umd"] = beta[6]

    result = pd.DataFrame(out, index=tickers)

    # ── Signal: |t| ≥ threshold (Student-t critical value) ─────────────── #
    K = X.shape[1]
    df_resid = T - K

    return_df = getattr(cfg, "return_df", None)
    if return_df is not None and return_df > 2:
        from scipy.stats import t as _t
        eff_df     = min(df_resid, return_df)
        t_crit     = float(_t.ppf(0.975, df=eff_df))
        dist_label = f"Student-t(ν={eff_df:.0f})"
    else:
        t_crit     = cfg.alpha_tstat_threshold
        dist_label = "Normal (ν→∞)"

    result["signal"]     = result["alpha_tstat"].abs() >= t_crit
    result["alpha_sign"] = np.sign(result["alpha_monthly"])

    logger.info(
        "%s regression: %d stocks, %d with |t|≥%.3f [%s, p<5%% two-sided] (%.0f%%).",
        model_label, len(tickers), result["signal"].sum(),
        t_crit, dist_label, 100 * result["signal"].mean(),
    )
    return result


# --------------------------------------------------------------------------- #
#  Rolling (time-series) alpha surface                                         #
# --------------------------------------------------------------------------- #

def rolling_alpha_signal(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute the FF5/FF6 alpha signal at every rebalance date.

    Returns a DataFrame with a MultiIndex (date, ticker) and the same
    columns as `compute_ff5_alpha`.
    """
    returns = prices.pct_change().dropna(how="all")
    dates   = returns.index
    window  = cfg.lookback_months

    frames = []
    rebalance_dates = pd.date_range(
        start=dates[window],
        end=dates[-1],
        freq=cfg.rebalance_freq,
    ).intersection(dates)

    for reb_date in rebalance_dates:
        loc       = dates.get_loc(reb_date)
        start_loc = max(0, loc - window + 1)
        window_ret = returns.iloc[start_loc : loc + 1]
        valid_cols = window_ret.columns[window_ret.notna().sum() >= cfg.min_obs]
        window_ret = window_ret[valid_cols]
        _med = window_ret.median(axis=1)
        window_ret = window_ret.apply(lambda col: col.fillna(_med)).dropna()

        alpha_df = compute_ff5_alpha(window_ret, factors, cfg)
        if alpha_df.empty:
            continue
        alpha_df.index = pd.MultiIndex.from_tuples(
            [(reb_date, t) for t in alpha_df.index], names=["date", "ticker"]
        )
        frames.append(alpha_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames)
