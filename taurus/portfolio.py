"""
Taurus – Portfolio construction.

Two stages
──────────
1.  CML / Maximum-Sharpe optimisation
    ─────────────────────────────────
    Given the candidate long and short universes, find the portfolio on the
    Capital Market Line (tangency portfolio) that maximises the Sharpe ratio.

    Problem:
        max  (μ − rf) / σ_p
        s.t. Σ wᵢ = 1,  wᵢ ≥ 0,  wᵢ ≤ w_max

    Equivalent unconstrained form via substitution z = w/σ_p gives the
    classic Markowitz tangency.  We solve with scipy L-BFGS-B so bounds and
    linear constraints are handled cleanly.

    Covariance is regularised with Ledoit-Wolf shrinkage for small-T regimes.

2.  Beta neutralisation
    ────────────────────
    Scale the long and short legs so that:
        β_long_leg = β_short_leg
    meaning the net portfolio beta is ≈ 0 (pure alpha exposure).

    Algorithm:
        β_leg = w_leg · β_stocks
        Scale short leg by factor k = β_long / β_short
        Renormalise both legs to sum to 1.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import LinAlgError

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Covariance estimation                                                       #
# --------------------------------------------------------------------------- #

def estimate_covariance(
    returns: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Estimate the covariance matrix with optional Ledoit-Wolf shrinkage.

    Returns a (N, N) PSD numpy array.
    """
    R = returns.dropna().values
    N = R.shape[1]

    if cfg.cov_shrinkage and _HAS_SKLEARN and R.shape[0] > N:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(R)
        cov = lw.covariance_
    else:
        cov = np.cov(R.T, ddof=1)

    # Enforce PSD: clip eigenvalues at floor
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, cfg.cov_min_eigenvalue)
    cov_psd = (eigvecs * eigvals) @ eigvecs.T

    return cov_psd


# --------------------------------------------------------------------------- #
#  Maximum-Sharpe (tangency) optimisation                                     #
# --------------------------------------------------------------------------- #

def max_sharpe_weights(
    expected_returns: pd.Series,
    cov: np.ndarray,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    Find the maximum-Sharpe-ratio portfolio weights (long-only, bounded).

    Parameters
    ----------
    expected_returns : annualised expected returns indexed by ticker
    cov              : (N × N) covariance matrix in monthly units

    Returns
    -------
    weights : Series indexed by ticker, summing to 1.
    """
    tickers = expected_returns.index.tolist()
    N = len(tickers)
    rf_m = cfg.rf_monthly                       # monthly risk-free

    # Annualised → monthly mean
    mu_m = (1 + expected_returns.values) ** (1 / 12) - 1   # (N,)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = w @ mu_m - rf_m
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e9
        return -port_ret / np.sqrt(port_var)

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        port_ret = w @ mu_m - rf_m
        port_var = w @ cov @ w
        sigma    = np.sqrt(max(port_var, 1e-16))
        d_ret    = mu_m
        d_sigma  = (cov @ w) / sigma
        grad = -(d_ret * sigma - port_ret * d_sigma) / (sigma ** 2)
        return grad

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Bounds: each weight ∈ [w_min, w_max]
    w_min = cfg.min_position_weight
    w_max = cfg.max_position_weight
    bounds = [(w_min, w_max)] * N

    # Warm start: equal weight
    w0 = np.full(N, 1.0 / N)

    result = minimize(
        neg_sharpe,
        w0,
        jac=neg_sharpe_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.optimizer_max_iter, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("Max-Sharpe optimisation did not converge: %s", result.message)
        # Fall back to equal weight
        w = np.full(N, 1.0 / N)
    else:
        w = result.x

    # Enforce positivity and renormalise
    w = np.maximum(w, 0)
    w /= w.sum()

    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------- #
#  Sector constraint enforcement                                               #
# --------------------------------------------------------------------------- #

def apply_sector_constraints(
    weights: pd.Series,
    sector_map: Dict[str, str],
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    Trim sector weights that exceed the max_sector_weight cap.

    Excess weight is redistributed proportionally to other sectors.
    """
    w = weights.copy()
    for _ in range(20):   # iterative until all constraints satisfied
        changed = False
        for sector in set(sector_map.values()):
            members = [t for t, s in sector_map.items() if s == sector and t in w.index]
            if not members:
                continue
            sector_w = w[members].sum()
            if sector_w > cfg.max_sector_weight + 1e-8:
                # Scale down sector proportionally
                scale = cfg.max_sector_weight / sector_w
                w[members] *= scale
                changed = True
        if not changed:
            break

    w = w.clip(lower=0)
    if w.sum() > 0:
        w /= w.sum()
    return w


# --------------------------------------------------------------------------- #
#  Beta estimation                                                             #
# --------------------------------------------------------------------------- #

def estimate_betas(
    returns: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.Series:
    """
    OLS beta of each stock against the market.

    Returns beta Series indexed by ticker.
    """
    R = returns.dropna(how="all")
    common_idx = R.index.intersection(market_returns.index)
    R = R.loc[common_idx].dropna(axis=1)
    mkt = market_returns.loc[common_idx].values

    X = np.column_stack([np.ones(len(mkt)), mkt])  # (T, 2)
    Y = R.values                                     # (T, N)

    try:
        beta_all = (np.linalg.lstsq(X, Y, rcond=None)[0])[1]  # (N,)
    except LinAlgError:
        beta_all = np.ones(Y.shape[1])

    return pd.Series(beta_all, index=R.columns)


# --------------------------------------------------------------------------- #
#  Beta neutralisation                                                         #
# --------------------------------------------------------------------------- #

def beta_neutralise(
    long_weights: pd.Series,
    short_weights: pd.Series,
    betas: pd.Series,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> Tuple[pd.Series, pd.Series]:
    """
    Rescale long and short leg weights so that:
        β_long_leg ≈ β_short_leg  (net β ≈ 0)

    Algorithm
    ---------
    1. Compute β_L = Σ w_L_i × β_i   (weighted avg beta of long leg)
    2. Compute β_S = Σ w_S_i × β_i   (weighted avg beta of short leg)
    3. We want k_L × β_L = k_S × β_S  with k_L + k_S = 2 (keep notional)
       → k_S = 2 × β_L / (β_L + β_S)
          k_L = 2 × β_S / (β_L + β_S)
    4. Renormalise each leg to sum to 1.

    Returns
    -------
    long_weights_neutralised, short_weights_neutralised
    (each sums to 1; caller scales to desired notional)
    """
    long_tickers  = long_weights.index.intersection(betas.index)
    short_tickers = short_weights.index.intersection(betas.index)

    if long_tickers.empty or short_tickers.empty:
        logger.warning("Beta neutralisation skipped: missing betas.")
        return long_weights, short_weights

    beta_L = (long_weights[long_tickers]  * betas[long_tickers]).sum()
    beta_S = (short_weights[short_tickers] * betas[short_tickers]).sum()

    if abs(beta_L + beta_S) < 1e-8:
        logger.warning("Beta sum near zero; returning equal-weighted legs.")
        return long_weights, short_weights

    k_L = 2.0 * abs(beta_S) / (abs(beta_L) + abs(beta_S))
    k_S = 2.0 * abs(beta_L) / (abs(beta_L) + abs(beta_S))

    lw = long_weights  * k_L
    sw = short_weights * k_S

    # Renormalise each leg to sum to 1
    lw = lw / lw.sum() if lw.sum() > 0 else lw
    sw = sw / sw.sum() if sw.sum() > 0 else sw

    net_beta = (lw[long_tickers] * betas[long_tickers]).sum() - \
               (sw[short_tickers] * betas[short_tickers]).sum()

    logger.info(
        "Beta neutralisation: β_L=%.3f, β_S=%.3f → net β=%.4f (target=%.2f).",
        beta_L, beta_S, net_beta, cfg.target_net_beta,
    )
    return lw, sw


# --------------------------------------------------------------------------- #
#  Full leg construction                                                       #
# --------------------------------------------------------------------------- #

def build_leg(
    candidates: pd.Index,
    alpha_scores: pd.Series,
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    cfg: TaurusConfig = DEFAULT_CONFIG,
    n_stocks: Optional[int] = None,
) -> pd.Series:
    """
    Build a single leg (long or short) using CML optimisation.

    Parameters
    ----------
    candidates    : tickers that passed all signal filters for this leg
    alpha_scores  : annualised alpha for each candidate ticker
    returns       : historical monthly returns (all stocks)
    sector_map    : {ticker: sector}
    n_stocks      : override config.n_longs / n_shorts

    Returns
    -------
    weights : Series (sums to 1, indexed by ticker)
    """
    n = n_stocks or cfg.n_longs
    tickers = candidates.tolist()

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    # Rank by |alpha| within this leg, take top-N
    ranked = alpha_scores[tickers].abs().nlargest(min(n, len(tickers)))
    tickers = ranked.index.tolist()

    # Historical returns for selected stocks
    ret_sub = returns[tickers].dropna(how="all")
    valid   = ret_sub.columns[ret_sub.notna().sum() >= max(cfg.min_obs, len(tickers) + 5)]
    tickers = valid.tolist()

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    ret_sub  = returns[tickers].dropna()
    mu       = alpha_scores[tickers].fillna(0.0)
    cov      = estimate_covariance(ret_sub, cfg) * 12  # annualise

    weights = max_sharpe_weights(mu, cov, cfg)
    weights = apply_sector_constraints(weights, sector_map, cfg)
    return weights
