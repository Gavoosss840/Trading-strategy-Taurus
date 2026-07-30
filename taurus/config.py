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
    industry_distress_costs: bool = False    # Sector-specific distress rates (vs flat 20%)
    variable_credit_spread:  bool = False    # Leverage-based spread (vs flat +2%)

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
    w_momentum:    float = 0.30          # Weight for momentum z-score

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
    gross_leverage:       float = 1.0
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
