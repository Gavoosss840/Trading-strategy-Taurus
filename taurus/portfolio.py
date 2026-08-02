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
    elif cfg.cov_shrinkage and _HAS_SKLEARN and R.shape[0] >= 12:
        # Ledoit-Wolf is designed precisely for T ≤ N (ill-conditioned) cases —
        # do NOT gate it on T > N.
        lw = LedoitWolf(assume_centered=False)
        lw.fit(R)
        cov = lw.covariance_
        logger.debug("Covariance: Ledoit-Wolf shrinkage.")
    else:
        cov = np.cov(R.T, ddof=1)
        logger.debug("Covariance: sample (no shrinkage).")

    cov = np.atleast_2d(cov)   # N=1 → np.cov returns a 0-d scalar

    # Enforce PSD with a floor RELATIVE to the average variance: an absolute
    # 1e-6 floor creates near-riskless phantom directions that a variance
    # minimiser will exploit.
    mean_var = float(np.mean(np.diag(cov))) if cov.size else 0.0
    floor    = max(cfg.cov_min_eigenvalue, 1e-4 * mean_var)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals  = np.maximum(eigvals, floor)
    cov_psd  = (eigvecs * eigvals) @ eigvecs.T

    # NOTE: no Student-t ν/(ν-2) inflation here — the empirical covariance of
    # realised returns already contains fat-tail variance; inflating it again
    # double-counts tails (+67% at ν=5).

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
    # Clip to avoid a negative base under the fractional power: a score ≤ -1
    # would produce NaN and silently break the optimisation.
    er      = np.clip(expected_returns.values.astype(float), -0.90, 5.0)
    mu_m    = (1 + er) ** (1 / 12) - 1   # (N,)

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
    """
    Trim sector weights exceeding the cap, iterating trim → renormalise to a
    fixed point (a single trim followed by one global renormalisation pushes
    just-capped sectors back above the limit).

    If the cap is infeasible (n_sectors × cap < 1), it is relaxed to 1/n_sectors.
    """
    w = weights.clip(lower=0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()

    sectors_present = {sector_map[t] for t in w.index if t in sector_map}
    n_sec = max(len(sectors_present), 1)
    cap   = max(cfg.max_sector_weight, 1.0 / n_sec + 1e-6)

    for _ in range(50):
        violated = False
        for sector in sectors_present:
            members  = [t for t, s in sector_map.items() if s == sector and t in w.index]
            if not members:
                continue
            sector_w = w[members].sum()
            if sector_w > cap + 1e-8:
                w[members] *= cap / sector_w
                violated    = True
        if not violated:
            break
        total = w.sum()
        if total > 0:
            w = w / total   # renormalise, then re-check caps next iteration

    if w.sum() > 0:
        w = w / w.sum()
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
    # Drop NaN market months FIRST (FF factors publish with a lag; a NaN row
    # would poison the whole lstsq and produce all-NaN betas).
    mkt_clean  = market_returns.dropna()
    common_idx = R.index.intersection(mkt_clean.index)
    R          = R.loc[common_idx]
    valid_cols = R.columns[R.notna().all()]
    Rv         = R[valid_cols]
    mkt        = mkt_clean.loc[common_idx].values

    if len(common_idx) < 12 or len(valid_cols) == 0:
        logger.warning("estimate_betas: insufficient clean data — defaulting to β=1.")
        return pd.Series(1.0, index=returns.columns)

    X = np.column_stack([np.ones(len(mkt)), mkt])   # (T, 2)
    Y = Rv.values                                     # (T, N)

    try:
        beta_all = (np.linalg.lstsq(X, Y, rcond=None)[0])[1]   # (N,)
    except LinAlgError:
        beta_all = np.ones(Y.shape[1])

    # Blume shrinkage: pulls extreme betas toward the market mean (1.0)
    if getattr(cfg, "blume_shrinkage", True):
        beta_all = 0.67 * beta_all + 0.33
        logger.debug("Blume beta shrinkage applied.")

    betas = pd.Series(beta_all, index=valid_cols)
    # Stocks with gaps get β=1.0 (market beta) instead of being dropped —
    # a dropped ticker would silently be hedged as β=0 downstream.
    return betas.reindex(returns.columns, fill_value=1.0)


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

    The returned legs are intentionally NOT renormalised to sum to 1: the long
    leg sums to k_L and the short leg to k_S (k_L + k_S = 2, gross preserved).
    Downstream consumers (backtest P&L and execution sizing) multiply weights
    by gross_leverage/2, so the scaling flows into actual position sizes.
    Renormalising here would cancel the scalars exactly and leave net beta
    unchanged.
    """
    long_tickers  = long_weights.index.intersection(betas.index)
    short_tickers = short_weights.index.intersection(betas.index)

    if long_tickers.empty or short_tickers.empty:
        logger.warning("Beta neutralisation skipped: missing betas.")
        return long_weights, short_weights

    beta_L = (long_weights[long_tickers]  * betas[long_tickers]).sum()
    beta_S = (short_weights[short_tickers] * betas[short_tickers]).sum()

    denom = abs(beta_L) + abs(beta_S)
    if denom < 1e-8:
        logger.info("Both leg betas ≈ 0; no neutralisation needed.")
        return long_weights, short_weights

    k_L = 2.0 * abs(beta_S) / denom
    k_S = 2.0 * abs(beta_L) / denom

    # Bound the leg scalars so noisy beta estimates cannot produce extreme
    # gross tilts (trade exact neutrality for bounded leverage per leg).
    k_L = float(np.clip(k_L, 0.6, 1.4))
    k_S = float(np.clip(k_S, 0.6, 1.4))

    lw = long_weights  * k_L
    sw = short_weights * k_S

    net_beta = (lw[long_tickers] * betas[long_tickers]).sum() - \
               (sw[short_tickers] * betas[short_tickers]).sum()

    logger.info(
        "Beta neutralisation: β_L=%.3f, β_S=%.3f → k_L=%.3f, k_S=%.3f, net β=%.4f (target=%.2f).",
        beta_L, beta_S, k_L, k_S, net_beta, cfg.target_net_beta,
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

    # Fill per-stock gaps with the column median (covariance estimation only)
    # so one gappy ticker doesn't truncate the sample for the whole leg.
    ret_sub = returns[tickers]
    ret_sub = ret_sub.fillna(ret_sub.median())
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

    # Hard per-position cap, iterated: a single clip + renormalise can push
    # weights back above the cap, so repeat until the cap holds.
    actual_max_w = max(cfg.max_position_weight, 1.0 / max(len(tickers), 1))
    actual_max_w = min(actual_max_w, 0.25)
    for _ in range(20):
        if weights.sum() > 0:
            weights = weights / weights.sum()
        if (weights <= actual_max_w + 1e-9).all():
            break
        weights = weights.clip(upper=actual_max_w)

    weights = apply_sector_constraints(weights, sector_map, cfg)
    return weights
