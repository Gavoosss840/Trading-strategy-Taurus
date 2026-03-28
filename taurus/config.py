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
    #  Factor model  (FF5 + SML alpha)                                    #
    # ------------------------------------------------------------------ #
    lookback_months: int = 60            # Rolling OLS window
    min_obs: int = 36                    # Minimum observations for fit
    alpha_tstat_threshold: float = 2.0   # |t| > 2 to keep stock

    # ------------------------------------------------------------------ #
    #  Capital structure  (MM screen)                                     #
    # ------------------------------------------------------------------ #
    # Deviation of actual vs target leverage required to flag a stock.
    # >+threshold → overleveraged  |  <-threshold → underleveraged
    leverage_gap_threshold: float = 0.25     # 25 %
    # Interest-coverage guard: ratio below this always flags overleveraged
    min_interest_coverage: float = 1.5

    # ------------------------------------------------------------------ #
    #  Momentum filter                                                    #
    # ------------------------------------------------------------------ #
    momentum_months: int = 12            # Lookback (includes skip)
    momentum_skip: int = 1               # Skip most-recent N months

    # ------------------------------------------------------------------ #
    #  Portfolio construction                                             #
    # ------------------------------------------------------------------ #
    n_longs: int = 25
    n_shorts: int = 25
    max_position_weight: float = 0.08    # 8 % cap per leg
    min_position_weight: float = 0.005   # 0.5 % floor
    max_sector_weight: float = 0.30      # 30 % sector cap per leg

    # ------------------------------------------------------------------ #
    #  Risk / optimisation                                                #
    # ------------------------------------------------------------------ #
    risk_free_rate_annual: float = 0.045
    target_net_beta: float = 0.0         # Beta-neutral by default
    beta_tolerance: float = 0.05         # Acceptable residual beta
    cov_shrinkage: bool = True           # Ledoit-Wolf shrinkage
    cov_min_eigenvalue: float = 1e-6     # Floor eigenvalue (PSD fix)
    optimizer_max_iter: int = 1_000

    # ------------------------------------------------------------------ #
    #  Leverage                                                           #
    # ------------------------------------------------------------------ #
    # gross_leverage: 1.0 = no leverage, 1.5 = 150% gross (75L/75S)
    # margin_cost_annual: annual cost of borrowed capital (IBKR ~5.8%)
    # borrow_cost_annual: stock borrow fee for short positions (~0.5-2%)
    gross_leverage: float = 1.0
    margin_cost_annual: float = 0.058    # 5.8%/an (fed funds + spread)
    borrow_cost_annual: float = 0.010    # 1.0%/an avg stock borrow fee

    # ------------------------------------------------------------------ #
    #  Futures beta hedge (Phase 2)                                       #
    # ------------------------------------------------------------------ #
    # When True, replaces weight-rescaling beta neutralisation with a
    # clean futures overlay (ES/SPY) that leaves alpha weights intact.
    # futures_roll_cost: quarterly roll cost as fraction of notional (~0.15%)
    use_futures_hedge: bool = False
    futures_roll_cost_quarterly: float = 0.0015   # 0.15%/quarter = 0.6%/an

    # ------------------------------------------------------------------ #
    #  Execution / back-test                                              #
    # ------------------------------------------------------------------ #
    rebalance_freq: str = "ME"           # pandas offset alias
    transaction_cost_bps: float = 10.0   # One-way cost in basis points
    slippage_bps: float = 5.0

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
    cache_dir: str = ".cache"
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
