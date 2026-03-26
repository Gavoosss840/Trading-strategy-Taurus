"""
Taurus – Factor model: CAPM + Fama-French 5 Factors → SML alpha.

Key design decisions
─────────────────────
• Vectorised OLS:  All N stocks are regressed in a single matrix solve.
  X  (T × 6) = [1, Mkt-RF, SMB, HML, RMW, CMA]
  Y  (T × N) = excess stock returns
  β  (6 × N) = (X'X)⁻¹ X'Y  – solved once for all stocks simultaneously.

• Heteroskedasticity-consistent (HC1) standard errors for the alpha t-stat,
  implemented without statsmodels to stay vectorised.

• Rolling window: the helper `rolling_alpha_signal` slides the estimation
  window forward one month at a time, reusing cached XtX_inv for the
  constant-factor block.

Outputs
────────
• alpha        : float array (N,) – annualised intercept
• alpha_tstat  : float array (N,) – t-statistic of the intercept
• betas        : DataFrame (N × 5) – market, SMB, HML, RMW, CMA loadings
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
    X: np.ndarray,       # (T, K)
    residuals: np.ndarray,  # (T, N)
    XtX_inv: np.ndarray, # (K, K)
) -> np.ndarray:
    """
    HC1 (heteroskedasticity-consistent) standard error for the intercept
    coefficient (first row of beta), for each of N stocks.

    HC1 sandwich: V = (X'X)⁻¹ [Σᵢ eᵢ² xᵢ xᵢ'] (X'X)⁻¹  × T/(T-K)

    Returns se_alpha : (N,)
    """
    T, K = X.shape
    N = residuals.shape[1]
    df_correction = T / (T - K)

    # Meat = Σ eᵢ² xᵢ xᵢ'  for each stock n → (K, K, N) but we only need
    # the [0,0] element of V, so we can simplify.
    # Meat[0,0,n] = Σᵢ eᵢₙ² xᵢ₀²  where x₀ = 1 (intercept column)
    # → Σᵢ eᵢₙ² × 1² = eₙ . eₙ element-wise
    # V[0,0,n] = (XtX_inv @ Meat_n @ XtX_inv)[0,0]
    # We compute the full K×K meat per stock vectorised over N.

    # e²: (T, N)
    e2 = residuals ** 2

    # Build meat matrices: (K, K, N)
    # meat[:, :, n] = X.T @ diag(e2[:, n]) @ X
    # = (X * e2[:, n:n+1]).T @ X   (broadcast)
    # Vectorised: meat = einsum('ti,tj,tn->ijn', X, X, e2)
    # Too memory-heavy; instead compute only the 0-th row/col we need.
    # row0 of (XtX_inv @ meat @ XtX_inv) = XtX_inv[0, :] @ meat @ XtX_inv
    # scalar V[0,0,n] = XtX_inv[0,:] @ meat_n @ XtX_inv[:,0]
    #                 = (XtX_inv[0,:] @ X.T) @ diag(e2[:,n]) @ (X @ XtX_inv[:,0])
    q = X @ XtX_inv[:, 0]        # (T,)   – leverage vector for intercept
    # V[0,0,n] = q.T @ diag(e2[:,n]) @ q = sum_t q_t² e_tn²
    V00 = (q[:, None] ** 2 * e2).sum(axis=0) * df_correction  # (N,)
    se_alpha = np.sqrt(np.maximum(V00, 1e-16))
    return se_alpha


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def compute_ff5_alpha(
    returns: pd.DataFrame,        # (T, N) monthly stock excess returns
    factors: pd.DataFrame,        # (T, 6) FF5 + RF  columns
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Run the FF5 regression for every stock and return a result DataFrame.

    Parameters
    ----------
    returns : monthly stock returns (NOT excess; RF is subtracted here)
    factors : DataFrame with columns Mkt-RF, SMB, HML, RMW, CMA, RF
              (decimal, month-end index)

    Returns
    -------
    DataFrame indexed by ticker with columns:
        alpha_monthly, alpha_annual, alpha_tstat,
        beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma,
        r_squared, signal (bool)
    """
    # ── Align ──────────────────────────────────────────────────────────── #
    aligned_idx = returns.index.intersection(factors.index)
    if len(aligned_idx) < cfg.min_obs:
        logger.warning(
            "Only %d overlapping observations (need %d).",
            len(aligned_idx), cfg.min_obs,
        )
        return pd.DataFrame()

    ret = returns.loc[aligned_idx]
    fac = factors.loc[aligned_idx]

    rf = fac["RF"].values                    # (T,)
    mkt_rf = fac["Mkt-RF"].values
    smb    = fac["SMB"].values
    hml    = fac["HML"].values
    rmw    = fac["RMW"].values
    cma    = fac["CMA"].values

    # ── Excess returns ──────────────────────────────────────────────────── #
    Y = ret.values - rf[:, None]             # (T, N)

    # ── Design matrix ──────────────────────────────────────────────────── #
    T = len(aligned_idx)
    X = np.column_stack([
        np.ones(T),
        mkt_rf, smb, hml, rmw, cma,
    ])                                        # (T, 6)

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

    # ── Assemble results ─────────────────────────────────────────────────── #
    tickers = ret.columns.tolist()
    result = pd.DataFrame(
        {
            "alpha_monthly": alpha_m,
            "alpha_annual":  alpha_a,
            "alpha_tstat":   t_stats,
            "beta_mkt":  beta[1],
            "beta_smb":  beta[2],
            "beta_hml":  beta[3],
            "beta_rmw":  beta[4],
            "beta_cma":  beta[5],
            "r_squared": r2,
        },
        index=tickers,
    )

    result["signal"] = result["alpha_tstat"].abs() >= cfg.alpha_tstat_threshold
    result["alpha_sign"] = np.sign(result["alpha_monthly"])   # +1 / -1

    logger.info(
        "FF5 regression: %d stocks, %d with |t|≥%.1f (%.0f%%).",
        len(tickers),
        result["signal"].sum(),
        cfg.alpha_tstat_threshold,
        100 * result["signal"].mean(),
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
    Compute the FF5 alpha signal at every rebalance date.

    Returns a DataFrame with a MultiIndex (date, ticker) and the same
    columns as `compute_ff5_alpha`.
    """
    returns = prices.pct_change().dropna(how="all")
    dates = returns.index
    window = cfg.lookback_months

    frames = []
    rebalance_dates = pd.date_range(
        start=dates[window],
        end=dates[-1],
        freq=cfg.rebalance_freq,
    ).intersection(dates)

    for reb_date in rebalance_dates:
        loc = dates.get_loc(reb_date)
        start_loc = max(0, loc - window + 1)
        window_ret = returns.iloc[start_loc : loc + 1]
        # Keep only stocks with full history in window
        min_obs = cfg.min_obs
        valid_cols = window_ret.columns[window_ret.notna().sum() >= min_obs]
        window_ret = window_ret[valid_cols].dropna()

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
