"""
Taurus – Risk management layer.

Runs DAILY (not monthly) to protect positions between rebalances.

3 niveaux de protection :
─────────────────────────
1. Stop-loss par position  : coupe si -10% depuis entrée
2. Trailing stop           : coupe si -10% depuis le pic (protège les gains)
3. Circuit breaker global  : coupe TOUT si portfolio -15% depuis dernier rebalancement

En cas de déclenchement → exit immédiat via IBKR, rapport sauvegardé.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Paramètres de risque (overridables via RiskConfig)                         #
# --------------------------------------------------------------------------- #

@dataclass
class RiskConfig:
    stop_loss_pct:       float = 0.10   # -10% depuis entrée → STP ordre
    take_profit_pct:     float = 0.20   # +20% depuis entrée → LMT ordre (take profit)
    trailing_stop_pct:   float = 0.10   # -10% depuis le pic → coupe (risk monitor)
    circuit_breaker_pct: float = 0.15   # -15% portfolio global → tout couper
    min_hold_days:       int   = 3      # ne pas couper avant 3 jours (évite faux signaux)
    state_file:          str   = ".cache/risk_state.json"


# --------------------------------------------------------------------------- #
#  État d'une position (persisté sur disque)                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PositionState:
    ticker:      str
    side:        str          # "LONG" | "SHORT"
    entry_price: float
    entry_date:  str          # ISO date
    peak_price:  float        # plus haut atteint depuis entrée (LONG) ou plus bas (SHORT)
    shares:      int
    universe:    str

    def update_peak(self, current_price: float) -> None:
        if self.side == "LONG":
            self.peak_price = max(self.peak_price, current_price)
        else:
            self.peak_price = min(self.peak_price, current_price)

    def pnl_pct(self, current_price: float) -> float:
        """P&L % depuis l'entrée."""
        if self.side == "LONG":
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def drawdown_from_peak(self, current_price: float) -> float:
        """Recul depuis le meilleur prix atteint."""
        if self.side == "LONG":
            if self.peak_price <= 0:
                return 0.0
            return (self.peak_price - current_price) / self.peak_price
        else:
            if self.peak_price <= 0:
                return 0.0
            return (current_price - self.peak_price) / self.peak_price

    def days_held(self) -> int:
        try:
            entry = date.fromisoformat(self.entry_date)
            return (date.today() - entry).days
        except Exception:
            return 999


# --------------------------------------------------------------------------- #
#  État persisté du risk manager                                              #
# --------------------------------------------------------------------------- #

class RiskState:
    """Charge/sauvegarde les positions et la NAV de référence sur disque."""

    def __init__(self, config: RiskConfig):
        self.config    = config
        self.positions: Dict[str, PositionState] = {}
        self.ref_nav:   float = 0.0   # NAV au dernier rebalancement
        Path(".cache").mkdir(exist_ok=True)
        self._load()

    def _load(self) -> None:
        try:
            with open(self.config.state_file, "r") as f:
                data = json.load(f)
            self.ref_nav = data.get("ref_nav", 0.0)
            for t, p in data.get("positions", {}).items():
                self.positions[t] = PositionState(**p)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        tmp = self.config.state_file + ".tmp"
        data = {
            "ref_nav":    self.ref_nav,
            "saved_at":   datetime.now().isoformat(),
            "positions":  {
                t: {k: v for k, v in p.__dict__.items()}
                for t, p in self.positions.items()
            },
        }
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(self.config.state_file)  # replace() works on Windows (rename() fails if dest exists)

    def register_rebalance(
        self,
        snapshot,
        prices:  Dict[str, float],
        nav:     float,
        universe: str,
    ) -> None:
        """
        Appelé après chaque rebalancement mensuel.
        Enregistre les nouvelles positions et reset la NAV de référence.
        """
        self.ref_nav = nav

        # Nouvelles positions LONG
        for ticker, w in snapshot.long_weights.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                shares = max(1, int(w * nav * 0.5 / price))
                self.positions[ticker] = PositionState(
                    ticker=ticker, side="LONG",
                    entry_price=price, entry_date=date.today().isoformat(),
                    peak_price=price, shares=shares, universe=universe,
                )

        # Nouvelles positions SHORT
        for ticker, w in snapshot.short_weights.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                shares = max(1, int(w * nav * 0.5 / price))
                self.positions[ticker] = PositionState(
                    ticker=ticker, side="SHORT",
                    entry_price=price, entry_date=date.today().isoformat(),
                    peak_price=price, shares=shares, universe=universe,
                )

        # Supprimer positions qui ne sont plus dans le portfolio
        current = set(snapshot.long_weights.index) | set(snapshot.short_weights.index)
        self.positions = {t: p for t, p in self.positions.items() if t in current}
        self.save()
        logger.info("RiskState updated: %d positions, NAV=%.0f", len(self.positions), nav)

    def remove(self, ticker: str) -> None:
        self.positions.pop(ticker, None)
        self.save()


# --------------------------------------------------------------------------- #
#  Rapport de déclenchement                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class RiskEvent:
    ticker:       str
    side:         str
    reason:       str           # "stop_loss" | "trailing_stop" | "circuit_breaker"
    entry_price:  float
    current_price: float
    pnl_pct:      float
    drawdown_pct: float
    triggered_at: str = field(default_factory=lambda: datetime.now().isoformat())


# --------------------------------------------------------------------------- #
#  Risk Manager principal                                                     #
# --------------------------------------------------------------------------- #

class RiskManager:
    """
    Vérifie chaque position quotidiennement.
    Déclenche des exits d'urgence si les seuils sont franchis.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.state  = RiskState(self.config)

    def check_positions(
        self,
        current_prices: Dict[str, float],
        current_nav:    float,
    ) -> Tuple[List[RiskEvent], bool]:
        """
        Vérifie toutes les positions contre les seuils de risque.

        Retourne :
          events       : liste des positions à couper
          circuit_open : True si le circuit breaker global est déclenché
        """
        events:        List[RiskEvent] = []
        circuit_open:  bool            = False

        # ── 1. Circuit breaker global ─────────────────────────────────── #
        if self.state.ref_nav > 0:
            portfolio_drawdown = (self.state.ref_nav - current_nav) / self.state.ref_nav
            if portfolio_drawdown >= self.config.circuit_breaker_pct:
                logger.warning(
                    "🔴 CIRCUIT BREAKER: portfolio down %.1f%% from ref NAV %.0f",
                    portfolio_drawdown * 100, self.state.ref_nav,
                )
                circuit_open = True
                # Créer un event pour CHAQUE position
                for ticker, pos in self.state.positions.items():
                    price = current_prices.get(ticker, pos.entry_price)
                    events.append(RiskEvent(
                        ticker=ticker, side=pos.side,
                        reason="circuit_breaker",
                        entry_price=pos.entry_price,
                        current_price=price,
                        pnl_pct=pos.pnl_pct(price),
                        drawdown_pct=portfolio_drawdown,
                    ))
                return events, circuit_open

        # ── 2. Stop-loss et trailing stop par position ────────────────── #
        for ticker, pos in list(self.state.positions.items()):
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                continue

            # Trop tôt pour couper
            if pos.days_held() < self.config.min_hold_days:
                continue

            pos.update_peak(price)
            pnl         = pos.pnl_pct(price)
            drawdown    = pos.drawdown_from_peak(price)

            # Stop-loss : -10% depuis l'entrée
            if pnl <= -self.config.stop_loss_pct:
                logger.warning(
                    "🔴 STOP-LOSS %s %s: P&L=%.1f%% (entry=%.2f, now=%.2f)",
                    pos.side, ticker, pnl * 100, pos.entry_price, price,
                )
                events.append(RiskEvent(
                    ticker=ticker, side=pos.side, reason="stop_loss",
                    entry_price=pos.entry_price, current_price=price,
                    pnl_pct=pnl, drawdown_pct=drawdown,
                ))

            # Trailing stop : -10% depuis le pic (uniquement si position profitable)
            elif drawdown >= self.config.trailing_stop_pct and pnl > 0.02:
                logger.warning(
                    "🟡 TRAILING STOP %s %s: drawdown=%.1f%% from peak=%.2f (P&L was +%.1f%%)",
                    pos.side, ticker, drawdown * 100, pos.peak_price, pnl * 100,
                )
                events.append(RiskEvent(
                    ticker=ticker, side=pos.side, reason="trailing_stop",
                    entry_price=pos.entry_price, current_price=price,
                    pnl_pct=pnl, drawdown_pct=drawdown,
                ))

        self.state.save()
        return events, circuit_open

    def execute_exits(
        self,
        events:       List[RiskEvent],
        conn,         # IBKRConnection
        universe_cfg,  # UniverseConfig
        dry_run:      bool = True,
        output_dir:   str  = "output",
    ) -> None:
        """
        Place les ordres de sortie pour chaque RiskEvent.
        LONG → SELL | SHORT → BUY (rachat)
        """
        from .execution import OrderManager

        if not events:
            return

        mgr = OrderManager()
        report = {
            "date":    date.today().isoformat(),
            "events":  [],
            "dry_run": dry_run,
        }

        for event in events:
            pos = self.state.positions.get(event.ticker)
            if pos is None:
                continue

            # LONG → vendre les actions | SHORT → racheter (delta positif)
            delta = -pos.shares if pos.side == "LONG" else pos.shares

            order = mgr.place_order(
                conn=conn,
                ticker=event.ticker,
                delta=delta,
                universe_cfg=universe_cfg,
                dry_run=dry_run,
            )

            report["events"].append({
                "ticker":       event.ticker,
                "side":         event.side,
                "reason":       event.reason,
                "pnl_pct":      round(event.pnl_pct * 100, 2),
                "drawdown_pct": round(event.drawdown_pct * 100, 2),
                "order":        order,
            })

            # Retirer de l'état des positions
            self.state.remove(event.ticker)

        # Sauvegarder le rapport
        Path(output_dir).mkdir(exist_ok=True)
        fname = f"{output_dir}/risk_exits_{date.today()}.json"
        with open(fname, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Risk exits report → %s (%d positions coupées)", fname, len(events))
