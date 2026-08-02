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

    # ── Design matrix ──────────────────────────────────────────────────── #
    T = len(aligned_idx)
    if use_umd:
        umd = fac["UMD"].fillna(0.0).values
        X = np.column_stack([np.ones(T), mkt_rf, smb, hml, rmw, cma, umd])  # (T, 7)
        model_label = "FF6"
    else:
        X = np.column_stack([np.ones(T), mkt_rf, smb, hml, rmw, cma])        # (T, 6)
        model_label = "FF5"
    K = X.shape[1]

    def _fit(Xd: np.ndarray, Yd: np.ndarray):
        """OLS + HC1 SE + R² for a block of columns sharing the same rows."""
        beta_b, resid, XtX_inv = _ols_vectorised(Xd, Yd)
        se_b   = _hc1_se_intercept(Xd, resid, XtX_inv)
        ss_res = (resid ** 2).sum(axis=0)
        ss_tot = ((Yd - Yd.mean(axis=0)) ** 2).sum(axis=0)
        r2_b   = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
        return beta_b, se_b, r2_b

    # ── Fit: vectorised for full-history stocks, per-stock on own valid    #
    #    rows for partial-history stocks.  (NO cross-sectional imputation — #
    #    imputed months are near-collinear with the market factor and       #
    #    artificially inflate alpha t-stats.)                               #
    complete_cols = ret.columns[ret.notna().all()].tolist()
    partial_cols  = [c for c in ret.columns if c not in set(complete_cols)]

    rows: dict = {}

    if complete_cols:
        Yc = ret[complete_cols].values - rf[:, None]
        beta_c, se_c, r2_c = _fit(X, Yc)
        for j, tkr in enumerate(complete_cols):
            rows[tkr] = (beta_c[:, j], se_c[j], r2_c[j], T)

    n_skipped = 0
    for tkr in partial_cols:
        mask  = ret[tkr].notna().values
        n_obs = int(mask.sum())
        if n_obs < 24 or n_obs <= K + 2:
            n_skipped += 1
            continue
        Xs = X[mask]
        ys = (ret[tkr].values[mask] - rf[mask])[:, None]
        beta_s, se_s, r2_s = _fit(Xs, ys)
        rows[tkr] = (beta_s[:, 0], se_s[0], r2_s[0], n_obs)

    if n_skipped:
        logger.info("%d partial-history stocks below 24 obs — excluded.", n_skipped)

    if not rows:
        return pd.DataFrame()

    tickers   = [c for c in ret.columns if c in rows]
    beta_mat  = np.column_stack([rows[t][0] for t in tickers])   # (K, N)
    se_alpha  = np.array([rows[t][1] for t in tickers])
    r2        = np.array([rows[t][2] for t in tickers])
    n_obs_arr = np.array([rows[t][3] for t in tickers])

    alpha_m = beta_mat[0]                     # (N,) monthly
    alpha_a = (1 + alpha_m) ** 12 - 1        # annualised

    # Degenerate columns (halted/stale prices → se ≈ 0) → NaN, not |t| = ∞
    t_stats = np.where(se_alpha > 1e-7, alpha_m / se_alpha, np.nan)

    out = {
        "alpha_monthly": alpha_m,
        "alpha_annual":  alpha_a,
        "alpha_tstat":   t_stats,
        "beta_mkt":  beta_mat[1],
        "beta_smb":  beta_mat[2],
        "beta_hml":  beta_mat[3],
        "beta_rmw":  beta_mat[4],
        "beta_cma":  beta_mat[5],
        "r_squared": r2,
        "n_obs":     n_obs_arr,
    }
    if use_umd:
        out["beta_umd"] = beta_mat[6]

    result = pd.DataFrame(out, index=tickers)

    # ── Signal: |t| ≥ per-stock critical value with df = n_obs − K ─────── #
    # The sampling distribution of an OLS t-stat has df = T − K residual
    # degrees of freedom.  (Previously min(df_resid, ν=5) was used — fat-tailed
    # RETURNS do not change the t-stat's reference distribution; that misuse
    # raised the threshold to 2.57 and suppressed legitimate signals.)
    from scipy.stats import t as _t
    df_resid   = np.maximum(n_obs_arr - K, 1)
    t_crit_arr = _t.ppf(0.975, df=df_resid)

    result["signal"]     = np.abs(result["alpha_tstat"].values) >= t_crit_arr
    result["alpha_sign"] = np.sign(result["alpha_monthly"])

    logger.info(
        "%s regression: %d stocks (%d full, %d partial), %d with |t|≥crit "
        "[t(df=n_obs−%d), p<5%% two-sided] (%.0f%%).",
        model_label, len(tickers), len(complete_cols),
        len(tickers) - len(complete_cols),
        int(result["signal"].sum()), K, 100 * result["signal"].mean(),
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
    # fill_method=None: default pad-fill would turn price gaps into phantom
    # 0% returns before the regression ever sees them.
    returns = prices.pct_change(fill_method=None).dropna(how="all")
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
        # NaN handled per-stock inside compute_ff5_alpha (no imputation)
        window_ret = window_ret[valid_cols].dropna(how="all")

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
