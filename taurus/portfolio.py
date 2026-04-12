"""
Taurus – Portfolio construction.

Optimizer pipeline
──────────────────
1.  Covariance estimation
    ─────────────────────
    • EWMA (half-life configurable, default 36 months) when cov_halflife > 0.
      Gives more weight to recent data → captures regime changes.
    • Falls back to Ledoit-Wolf shrinkage when EWMA is disabled.
    • Student-t variance inflation ν/(ν-2) applied on top for fat tails.

2.  Portfolio optimisation
    ──────────────────────
    "min_variance" (default, AQR-style):
        min  w'Σw  −  λ_α × α'w  +  λ_t × |w − w_prev|₁
        s.t. Σw = 1,  w_min ≤ w ≤ w_max

        Minimises risk while tilting towards high-alpha stocks and penalising
        turnover.  Stable out-of-sample (no input-sensitive max-Sharpe issues).

    "max_sharpe" (legacy):
        Tangency portfolio — input-sensitive, kept for comparison.

3.  Beta estimation (Blume-adjusted)
    ──────────────────────────────
    β_adjusted = 0.67 × β_OLS + 0.33 × 1.0
    Betas mean-revert toward 1; Blume adjustment improves beta-neutralisation.

4.  Beta neutralisation
    ──────────────────
    Scale long/short legs so β_long_leg = β_short_leg (net β ≈ 0).
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
#  EWMA covariance                                                              #
# --------------------------------------------------------------------------- #

def _ewma_cov(R: np.ndarray, halflife: int) -> np.ndarray:
    """
    Exponentially weighted covariance matrix.

    More weight to recent observations → captures vol regime changes.
    λ decay: (1 − α)^(T−1−t) where α = 1 − exp(−ln2 / halflife).
    """
    T, N = R.shape
    alpha  = 1.0 - np.exp(-np.log(2.0) / halflife)
    # Decay weights: oldest → weight (1−α)^(T−1), newest → weight 1
    decay  = np.array([(1.0 - alpha) ** i for i in range(T - 1, -1, -1)])
    decay /= decay.sum()

    mean  = (decay[:, None] * R).sum(axis=0)
    R_c   = R - mean
    cov   = (R_c * decay[:, None]).T @ R_c
    return cov


# --------------------------------------------------------------------------- #
#  Covariance estimation                                                        #
# --------------------------------------------------------------------------- #

def estimate_covariance(
    returns: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Estimate the covariance matrix with optional EWMA or Ledoit-Wolf shrinkage.

    Priority:
        1. EWMA (when cfg.cov_halflife > 0)
        2. Ledoit-Wolf shrinkage (when cfg.cov_shrinkage=True and sklearn available)
        3. Sample covariance (fallback)

    Student-t variance inflation applied post-estimation:
        cov × ν/(ν-2)   where ν = cfg.return_df

    Returns a (N, N) PSD numpy array.
    """
    R = returns.dropna().values
    N = R.shape[1]

    halflife = getattr(cfg, "cov_halflife", 0)

    if halflife > 0 and R.shape[0] >= 12:
        cov = _ewma_cov(R, halflife)
        logger.debug("Covariance: EWMA (halflife=%d months).", halflife)
    elif cfg.cov_shrinkage and _HAS_SKLEARN and R.shape[0] > N:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(R)
        cov = lw.covariance_
        logger.debug("Covariance: Ledoit-Wolf shrinkage.")
    else:
        cov = np.cov(R.T, ddof=1)
        logger.debug("Covariance: sample (no shrinkage).")

    # Enforce PSD: clip eigenvalues at floor
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals  = np.maximum(eigvals, cfg.cov_min_eigenvalue)
    cov_psd  = (eigvecs * eigvals) @ eigvecs.T

    # Student-t fat-tail variance inflation: Var(t_ν) = σ² × ν/(ν-2)
    return_df = getattr(cfg, "return_df", None)
    if return_df is not None and return_df > 2:
        inflation = return_df / (return_df - 2.0)
        cov_psd   = cov_psd * inflation
        logger.debug("Covariance inflated by ν/(ν-2)=%.4f (Student-t, ν=%.1f).", inflation, return_df)

    return cov_psd


# --------------------------------------------------------------------------- #
#  Min-variance + alpha tilt + turnover penalty (AQR-style)                    #
# --------------------------------------------------------------------------- #

def min_variance_weights(
    tickers: List[str],
    cov: np.ndarray,
    alpha_scores: np.ndarray,    # composite signal or alpha annual (N,)
    cfg: TaurusConfig = DEFAULT_CONFIG,
    prev_weights: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Minimum-variance portfolio with two overlays:
      • Alpha tilt   : pulls weight toward high-composite-score stocks
      • Turnover penalty: penalises |w_new − w_prev|₁ to reduce trading costs

    Objective:
        min  (1−λ_α) × w'Σw  −  λ_α × ā'w  +  λ_t × |w − w_prev|₁
        s.t. Σw = 1,  w_min ≤ w ≤ w_max

    Parameters
    ----------
    tickers      : ordered ticker list (must match cov and alpha_scores)
    cov          : (N, N) covariance matrix (annualised)
    alpha_scores : (N,) continuous signal (higher = better for this leg)
    prev_weights : (N,) previous weights for turnover penalty (None = no penalty)
    """
    N = len(tickers)
    if N == 0:
        return pd.Series(dtype=float)

    # Normalise alpha scores to [0, 1] for numerical stability in objective
    a = alpha_scores.copy().astype(float)
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a_norm = (a - a_min) / (a_max - a_min)
    else:
        a_norm = np.full(N, 0.5)

    tilt_strength    = float(getattr(cfg, "alpha_tilt_strength", 0.30))
    turnover_penalty = float(getattr(cfg, "turnover_penalty", 0.002))
    w_prev = prev_weights if prev_weights is not None else np.full(N, 1.0 / N)

    def objective(w: np.ndarray) -> float:
        var_term      = (1.0 - tilt_strength) * float(w @ cov @ w)
        alpha_term    = -tilt_strength * float(w @ a_norm)
        turnover_term = turnover_penalty * float(np.abs(w - w_prev).sum())
        return var_term + alpha_term + turnover_term

    def objective_grad(w: np.ndarray) -> np.ndarray:
        d_var      = 2.0 * (1.0 - tilt_strength) * (cov @ w)
        d_alpha    = -tilt_strength * a_norm
        # L1 gradient (subgradient at 0 → 0)
        d_turnover = turnover_penalty * np.sign(w - w_prev)
        return d_var + d_alpha + d_turnover

    # Per-position caps: dynamic so small portfolios don't concentrate
    actual_max_w = max(cfg.max_position_weight, 1.0 / max(N, 1))
    actual_max_w = min(actual_max_w, 0.25)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(cfg.min_position_weight, actual_max_w)] * N
    w0          = np.full(N, 1.0 / N)

    result = minimize(
        objective, w0, jac=objective_grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": cfg.optimizer_max_iter, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("Min-variance optimisation did not converge: %s", result.message)
        w = w0.copy()
    else:
        w = result.x

    w = np.maximum(w, 0.0)
    total = w.sum()
    if total > 0:
        w /= total
    else:
        w = np.full(N, 1.0 / N)

    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------- #
#  Maximum-Sharpe (tangency) optimisation  [legacy]                            #
# --------------------------------------------------------------------------- #

def max_sharpe_weights(
    expected_returns: pd.Series,
    cov: np.ndarray,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    Find the maximum-Sharpe-ratio portfolio weights (long-only, bounded).
    Kept as legacy option; min_variance_weights preferred for stability.
    """
    tickers = expected_returns.index.tolist()
    N       = len(tickers)
    rf_m    = cfg.rf_monthly
    mu_m    = (1 + expected_returns.values) ** (1 / 12) - 1   # (N,)

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
        return -(d_ret * sigma - port_ret * d_sigma) / (sigma ** 2)

    actual_max_w = max(cfg.max_position_weight, 1.0 / max(N, 1))
    actual_max_w = min(actual_max_w, 0.25)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(cfg.min_position_weight, actual_max_w)] * N
    w0     = np.full(N, 1.0 / N)

    result = minimize(
        neg_sharpe, w0, jac=neg_sharpe_grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": cfg.optimizer_max_iter, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("Max-Sharpe optimisation did not converge: %s", result.message)
        w = w0.copy()
    else:
        w = result.x

    w = np.maximum(w, 0.0)
    w = np.minimum(w, actual_max_w)
    total = w.sum()
    if total > 0:
        w /= total
    else:
        w = np.full(N, 1.0 / N)

    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------- #
#  Sector constraint enforcement                                               #
# --------------------------------------------------------------------------- #

def apply_sector_constraints(
    weights: pd.Series,
    sector_map: Dict[str, str],
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Trim sector weights exceeding max_sector_weight; redistribute remainder."""
    w = weights.copy()
    for _ in range(20):
        changed = False
        for sector in set(sector_map.values()):
            members  = [t for t, s in sector_map.items() if s == sector and t in w.index]
            if not members:
                continue
            sector_w = w[members].sum()
            if sector_w > cfg.max_sector_weight + 1e-8:
                scale       = cfg.max_sector_weight / sector_w
                w[members] *= scale
                changed     = True
        if not changed:
            break

    w = w.clip(lower=0)
    if w.sum() > 0:
        w /= w.sum()
    return w


# --------------------------------------------------------------------------- #
#  Beta estimation (Blume-adjusted)                                            #
# --------------------------------------------------------------------------- #

def estimate_betas(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    OLS beta of each stock against the market, with Blume (1975) shrinkage.

    Blume adjustment: β_adj = 0.67 × β_OLS + 0.33 × 1.0
    Betas mean-revert toward 1.0 in the cross-section; shrinkage reduces
    beta-neutralisation error and improves out-of-sample hedge quality.
    """
    R          = returns.dropna(how="all")
    common_idx = R.index.intersection(market_returns.index)
    R          = R.loc[common_idx].dropna(axis=1)
    mkt        = market_returns.loc[common_idx].values

    X = np.column_stack([np.ones(len(mkt)), mkt])   # (T, 2)
    Y = R.values                                      # (T, N)

    try:
        beta_all = (np.linalg.lstsq(X, Y, rcond=None)[0])[1]   # (N,)
    except LinAlgError:
        beta_all = np.ones(Y.shape[1])

    # Blume shrinkage: pulls extreme betas toward the market mean (1.0)
    if getattr(cfg, "blume_shrinkage", True):
        beta_all = 0.67 * beta_all + 0.33
        logger.debug("Blume beta shrinkage applied.")

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
    Rescale long and short leg weights so β_long_leg ≈ β_short_leg (net β ≈ 0).
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
#  Futures beta hedge overlay                                                  #
# --------------------------------------------------------------------------- #

def futures_beta_hedge(
    long_weights: pd.Series,
    short_weights: pd.Series,
    betas: pd.Series,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> float:
    """Compute ES/SPY futures overlay weight to zero net portfolio beta."""
    long_tickers  = long_weights.index.intersection(betas.index)
    short_tickers = short_weights.index.intersection(betas.index)
    half = cfg.gross_leverage / 2.0
    beta_L   = (long_weights[long_tickers]  * betas[long_tickers]).sum()  * half
    beta_S   = (short_weights[short_tickers] * betas[short_tickers]).sum() * half
    beta_net = beta_L - beta_S
    futures_weight = -beta_net
    logger.info(
        "Futures hedge: β_long=%.3f, β_short=%.3f → β_net=%.4f → futures=%.4f × NAV",
        beta_L, beta_S, beta_net, futures_weight,
    )
    return float(futures_weight)


# --------------------------------------------------------------------------- #
#  Full leg construction                                                        #
# --------------------------------------------------------------------------- #

def build_leg(
    candidates: pd.Index,
    alpha_scores: pd.Series,       # higher = better for this leg
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    cfg: TaurusConfig = DEFAULT_CONFIG,
    n_stocks: Optional[int] = None,
    prev_weights: Optional[pd.Series] = None,  # for turnover penalty
) -> pd.Series:
    """
    Build a single leg (long or short) using the configured optimizer.

    Parameters
    ----------
    candidates    : tickers that passed all signal filters for this leg
    alpha_scores  : continuous signal score (composite z-score or alpha_annual)
                    — higher = better for this leg (sign already adjusted for short)
    returns       : historical monthly returns (all stocks)
    sector_map    : {ticker: sector}
    n_stocks      : override config.n_longs / n_shorts
    prev_weights  : previous leg weights for turnover penalty

    Returns
    -------
    weights : Series (sums to 1, indexed by ticker)
    """
    n       = n_stocks or cfg.n_longs
    tickers = candidates.tolist()

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    # Rank by signal score, take top-N
    ranked  = alpha_scores[tickers].nlargest(min(n, len(tickers)))
    tickers = ranked.index.tolist()

    # Filter to tickers with sufficient return history
    ret_sub   = returns[tickers].dropna(how="all")
    valid     = ret_sub.columns[ret_sub.notna().sum() >= max(cfg.min_obs, len(tickers) + 5)]
    tickers   = valid.tolist()

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    ret_sub = returns[tickers].dropna()
    scores  = alpha_scores[tickers].fillna(0.0).values
    cov     = estimate_covariance(ret_sub, cfg) * 12   # annualise

    # Previous weights aligned to current tickers
    if prev_weights is not None:
        prev_w = prev_weights.reindex(tickers, fill_value=0.0).values
    else:
        prev_w = None

    optimizer = getattr(cfg, "optimizer_method", "min_variance")

    if optimizer == "min_variance":
        weights = min_variance_weights(tickers, cov, scores, cfg, prev_w)
    else:
        # Legacy max-sharpe
        mu      = pd.Series(alpha_scores[tickers].fillna(0.0).values, index=tickers)
        weights = max_sharpe_weights(mu, cov, cfg)

    # Hard per-position cap then renormalise
    actual_max_w = max(cfg.max_position_weight, 1.0 / max(len(tickers), 1))
    actual_max_w = min(actual_max_w, 0.25)
    weights      = weights.clip(upper=actual_max_w)
    if weights.sum() > 0:
        weights /= weights.sum()

    weights = apply_sector_constraints(weights, sector_map, cfg)
    return weights
