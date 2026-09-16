"""
Taurus Strategy – Central configuration.

All tuneable hyper-parameters live here so that every module imports from
a single source of truth rather than scattering magic numbers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UniverseConfig:
    """Per-universe metadata for data sourcing and IBKR execution."""
    name:                 str              # "sp500" | "nasdaq100" | "cac40" | ...
    display_name:         str              # "S&P 500"
    region:               str              # "US" | "Europe" | "Japan" | "Asia" | "MiddleEast"
    currency:             str              # "USD" | "EUR" | "GBP" | "JPY" | "HKD" | "SAR"
    ff5_dataset:          str              # Ken French dataset name
    futures_symbol:       str              # "ES" | "NQ" | "CAC" | "Z" | "NK225" | ...
    futures_exchange:     str              # "CME" | "EUREX" | "MONEP" | "LIFFE" | "OSE" | "HKEX"
    futures_currency:     str             # "USD" | "EUR" | "GBP" | "JPY" | "HKD"
    futures_multiplier:   float            # contract multiplier
    ibkr_exchange:        str              # "SMART" | "SBF" | "IBIS" | "LSE" | "TSEJ" | "SEHK"
    wikipedia_url:        str              # URL for constituent scraping
    wikipedia_table_id:   str              # HTML table id on Wikipedia page
    ticker_suffix:        str   = ""       # appended after scrape: ".T" Japan, ".HK" HK, ".SR" Saudi
    ticker_col_index:     int   = 0        # column index of ticker in Wikipedia table
    min_market_cap_usd:   float = 2e9
    min_lot_size:         int   = 1        # minimum order size (TSE=100, most others=1)
    nav_scale:            float = 1.0     # multiply NAV to match price units (LSE=100: GBP→GBp pence)
    fractional_shares:    bool  = False   # True for US markets (IBKR supports MKT fractions)
    supports_trail_stop:  bool  = True     # False for Euronext/LSE: MKT transmits alone, STP sent separately
    use_wikipedia_scrape: bool  = False   # True for non-US universes whose Wikipedia tables have numeric codes
    n_longs:              int   = 25
    n_shorts:             int   = 25


@dataclass
class TaurusConfig:
    # ------------------------------------------------------------------ #
    #  Universe                                                            #
    # ------------------------------------------------------------------ #
    index: str = "sp500"                 # "sp500" | "nasdaq100"
    min_market_cap_usd: float = 2e9      # Drop micro-caps (<$2 B)

    # ------------------------------------------------------------------ #
    #  Factor model  (FF5/FF6 + SML alpha)                                #
    # ------------------------------------------------------------------ #
    lookback_months: int = 60            # Rolling OLS window
    min_obs: int = 36                    # Minimum observations for fit
    alpha_tstat_threshold: float = 2.0   # |t| > threshold (Student-t when return_df set)
    use_umd_factor: bool = False         # FF6: include UMD momentum factor in regression
                                         # When True, removes momentum from alpha (weaker signal)

    # ------------------------------------------------------------------ #
    #  Return distribution  (Student-t fat tails)                         #
    # ------------------------------------------------------------------ #
    # ν = 5: common conservative choice for monthly equity returns.
    # Effects:
    #   • factors.py          — t_crit = t.ppf(0.975, ν) instead of 1.96
    #   • capital_structure   — Merton P(default) uses t.cdf(-d2, ν)
    #   • portfolio.py        — covariance inflated by ν/(ν-2)
    return_df: float = 5.0              # Student-t degrees of freedom (None → Normal)

    # ------------------------------------------------------------------ #
    #  Capital structure  (MM screen)                                     #
    # ------------------------------------------------------------------ #
    leverage_gap_threshold: float = 0.25     # 25 % divergence to flag
    min_interest_coverage:  float = 1.5      # IC below → always flag overleveraged
    # Les financières échappent à l'écran MM dans les deux sens : leurs
    # intérêts sont un coût d'exploitation (pas une charge de financement), et
    # l'APV n'a pas de firme non endettée à valoriser quand le levier EST
    # l'activité. Elles restent négociables via l'alpha et le momentum.
    mm_skip_financials:     bool  = True     # divergence = NaN pour les financières
    industry_distress_costs: bool = False    # Sector-specific distress rates (vs flat 20%)
    variable_credit_spread:  bool = False    # Leverage-based spread (vs flat +2%)

    # ── Unlevered firm value (APV) ──────────────────────────────────────── #
    # VU is discounted from fundamentals, not backed out of market cap — see
    # the module docstring of capital_structure.py for why the previous
    # formulation was circular.  These three exogenous inputs drive the
    # perpetuity; the fair value is highly sensitive to them.
    equity_risk_premium:  float = 0.05   # US long-run equity risk premium
    terminal_growth:      float = 0.025  # growth after convergence, forever
    min_discount_spread:  float = 0.02   # floor on (r_U − g); keeps VU finite
    default_unlevered_beta: float = 1.0  # fallback when beta is unavailable

    # Two-stage valuation. A single perpetual growth rate applied to every
    # company mechanically undervalues any that grows faster, and the screen
    # then shorts precisely the fastest growers: measured on 20 US large caps,
    # growth and divergence correlated at −0.49 and 100% of names above 8%
    # revenue growth were flagged as short candidates, while the long leg went
    # to the telecoms in decline.
    #
    # Stage 1: the company's own growth, fading linearly to the terminal rate.
    # Stage 2: perpetuity at the terminal rate.
    explicit_growth_years:  int   = 10    # length of the fade
    max_initial_growth:     float = 0.15  # nothing grows at 20% for a decade
    min_initial_growth:     float = -0.05 # a declining business
    default_initial_growth: float = 0.03  # when no history is available

    # ------------------------------------------------------------------ #
    #  Momentum filter                                                    #
    # ------------------------------------------------------------------ #
    momentum_months: int = 12            # Lookback (includes skip)
    momentum_skip:   int = 1             # Skip most-recent N months
    vol_adjust_momentum:   bool = True   # Sharpe momentum: raw / trailing_vol
                                         # Reduces momentum crashes (Barroso & Santa-Clara 2015)
    momentum_crash_dampen: bool = True   # Halve momentum weight when mkt vol > 2× avg

    # ------------------------------------------------------------------ #
    #  Signal combination                                                 #
    # ------------------------------------------------------------------ #
    # "binary": AND-filter cascade (alpha AND MM AND momentum — original proven strategy).
    # "composite": continuous z-score blend (experimental, kept for comparison).
    signal_method: str  = "binary"
    w_alpha:       float = 0.40          # Weight for FF alpha t-stat z-score
    w_mm:          float = 0.30          # Weight for MM divergence z-score
    # Le biais de niveau d'un modèle d'actualisation est presque entièrement
    # sectoriel : le marché paie des multiples élevés pour le logiciel et bas
    # pour les télécoms, si bien qu'un classement absolu achète des secteurs et
    # non des sociétés. Comparer chaque titre à son propre secteur retire ce
    # niveau commun et ne garde que ce que le pilier sait vraiment.
    mm_sector_neutral:     bool = True   # z(divergence) calculé au sein du secteur
    mm_sector_min_members: int  = 4      # en deçà, le secteur est classé contre l'univers
    w_momentum:    float = 0.30          # Weight for momentum z-score

    # ------------------------------------------------------------------ #
    #  Multi-universe NAV allocation (Sharpe-weighted)                    #
    # ------------------------------------------------------------------ #
    # Weights are proportional to max(Sharpe, 0) computed on each universe's
    # stored monthly_returns.csv.
    #
    # sharpe_window_months = 0  → SINCE INCEPTION (expanding window): every
    #   month of history counts, and each new live rebalance permanently
    #   enriches the estimate.  A short rolling window (e.g. 12) makes the
    #   Sharpe estimate too noisy (SE ≈ ±1.2 at 12 obs) and lets one lucky
    #   streak capture nearly the whole book.
    sharpe_window_months: int   = 0      # 0 = since inception; >0 = rolling
    sharpe_min_months:    int   = 6      # below this → equal-weight
    max_universe_weight:  float = 0.35   # safety cap per universe (0 = off)

    # ------------------------------------------------------------------ #
    #  Portfolio construction                                             #
    # ------------------------------------------------------------------ #
    n_longs:  int   = 25
    n_shorts: int   = 25
    max_position_weight: float = 0.08    # 8 % cap per leg
    min_position_weight: float = 0.005   # 0.5 % floor
    max_sector_weight:   float = 0.30    # 30 % sector cap per leg

    # Optimizer:
    #   "max_sharpe"   — classical tangency portfolio (original Taurus optimizer).
    #   "min_variance" — minimises portfolio variance with alpha tilt (AQR-style).
    optimizer_method:  str   = "max_sharpe"
    alpha_tilt_strength: float = 0.30   # Fraction of weight driven by composite score
    turnover_penalty:  float = 0.002    # λ penalising |w_new - w_old|₁ in optimizer

    # ------------------------------------------------------------------ #
    #  Risk / optimisation                                                #
    # ------------------------------------------------------------------ #
    risk_free_rate_annual: float = 0.045
    target_net_beta: float = 0.0         # Beta-neutral by default
    beta_tolerance:  float = 0.05        # Acceptable residual beta
    blume_shrinkage: bool  = False       # Shrink OLS betas: 0.67×β_raw + 0.33×1.0
    cov_shrinkage:   bool  = True        # Ledoit-Wolf shrinkage (when EWMA off)
    cov_halflife:    int   = 0           # EWMA half-life months (0 → Ledoit-Wolf)
    cov_min_eigenvalue: float = 1e-6     # Floor eigenvalue (PSD fix)
    optimizer_max_iter: int = 1_000

    # ------------------------------------------------------------------ #
    #  Leverage                                                           #
    # ------------------------------------------------------------------ #
    # Gross exposure as a multiple of NAV: each leg is sized at
    # gross_leverage/2 × NAV (execution.py) — this IS the leverage; nothing is
    # configured on the IBKR side, the margin account simply funds the larger
    # orders.  Calibrated on 10y backtests of the 5-universe book:
    #   1.00 → 14.89% ann, Sharpe 1.25, maxDD  -8.30%, Calmar 1.79
    #   1.25 → 16.59% ann, Sharpe 1.22, maxDD  -8.72%, Calmar 1.90  ← retained
    #   1.50 → 18.64% ann, Sharpe 1.19, maxDD -10.76%, Calmar 1.73
    #   2.00 → 22.75% ann, Sharpe 1.14, maxDD -14.85%, Calmar 1.53
    # 1.25 is the only level that improves Calmar over 1.0 (drawdown barely
    # moves, +1.70pt of return) and still survives a degraded scenario
    # (vol ×1.3, alpha ×0.75 → maxDD -12.64%, inside the -15% circuit breaker).
    # Reg-T initial margin at 1.25 = 62.5% of NAV → 37.5% free buffer.
    gross_leverage:       float = 1.25
    margin_cost_annual:   float = 0.058  # 5.8%/an (fed funds + spread)
    borrow_cost_annual:   float = 0.010  # 1.0%/an avg stock borrow fee

    # ------------------------------------------------------------------ #
    #  Futures beta hedge (Phase 2)                                       #
    # ------------------------------------------------------------------ #
    use_futures_hedge:            bool  = False
    futures_roll_cost_quarterly:  float = 0.0015   # 0.15%/quarter = 0.6%/an

    # ------------------------------------------------------------------ #
    #  Execution / back-test                                              #
    # ------------------------------------------------------------------ #
    rebalance_freq:        str   = "ME"    # pandas offset alias
    transaction_cost_bps:  float = 10.0   # One-way cost in basis points
    slippage_bps:          float = 5.0

    # ------------------------------------------------------------------ #
    #  IBKR live trading                                                  #
    # ------------------------------------------------------------------ #
    ibkr_host:      str   = "127.0.0.1"
    ibkr_port:      int   = 7497          # 7497=paper, 7496=live
    ibkr_client_id: int   = 10
    ibkr_account:   str   = ""            # "" = default account
    live_trading:   bool  = False         # False=paper, True=live
    dry_run:        bool  = True          # True=log only, no real orders
    nav_usd:        float = 100_000.0     # total NAV for position sizing

    # ------------------------------------------------------------------ #
    #  Cache                                                              #
    # ------------------------------------------------------------------ #
    cache_dir:       str   = ".cache"
    cache_ttl_hours: float = 12.0        # Re-fetch after N hours

    # ------------------------------------------------------------------ #
    #  Convenience properties                                             #
    # ------------------------------------------------------------------ #
    @property
    def rf_monthly(self) -> float:
        return (1 + self.risk_free_rate_annual) ** (1 / 12) - 1

    @property
    def total_cost_bps(self) -> float:
        return self.transaction_cost_bps + self.slippage_bps


# Module-level default instance – import and override as needed.
DEFAULT_CONFIG = TaurusConfig()
