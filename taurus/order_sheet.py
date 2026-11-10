"""
Taurus — manual order sheet.

Builds the month's positions as an Excel workbook instead of sending them to
IBKR: ticker, company name, side, entry price, stop, take-profit, share count
and notional, one sheet per universe plus a combined sheet.

Needs no IBKR connection — prices come from market data and the NAV is given on
the command line, so the sheet can be produced from anywhere and read before any
order is placed.

Two things in here are deliberately explicit rather than hidden:

  * Share counts are computed through the same per-line currency conversion the
    order path uses (universe.ticker_market: quote currency, minor unit, primary
    exchange).  A London line quotes in pence, a Paris line in euros, and the
    NAV is in one currency — getting that wrong is what sized CPG.L and IHG.L
    74x too large.

  * The take-profit is reported TWICE, because the live system and the config
    disagree.  execute_rebalance derives it from the MM divergence with a 5%
    floor, and the MM screen returns a divergence of exactly 0 for ~92% of
    names, so what actually gets placed is +5%.  RiskConfig.take_profit_pct
    says 20%.  Both columns are given; pick one knowingly.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TaurusConfig, DEFAULT_CONFIG
from .risk import RiskConfig
from .universe import REGISTRY, ticker_market
from .valuation import load_calibration, valuation_screen

logger = logging.getLogger(__name__)

_FX_CACHE: Dict[Tuple[str, str], Optional[float]] = {}
_NAME_CACHE: Dict[str, str] = {}


def _fx(from_ccy: str, to_ccy: str) -> Optional[float]:
    """Spot rate from_ccy → to_ccy, from market data."""
    if from_ccy == to_ccy:
        return 1.0
    k = (from_ccy, to_ccy)
    if k in _FX_CACHE:
        return _FX_CACHE[k]
    rate = None
    try:
        import yfinance as yf
        raw = yf.download(f"{from_ccy}{to_ccy}=X", period="5d", interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        if raw is not None and not raw.empty:
            s = raw["Close"] if "Close" in raw else raw.iloc[:, [0]]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            if len(s):
                rate = float(s.iloc[-1])
    except Exception as e:
        logger.warning("FX %s->%s unavailable: %s", from_ccy, to_ccy, e)
    _FX_CACHE[k] = rate
    return rate


def _company_names(tickers: List[str]) -> Dict[str, str]:
    """
    Company name per ticker, from market data.

    Only the selected positions are looked up — a few dozen names, not the
    whole universe — so this stays cheap.  A ticker whose name cannot be
    fetched keeps the ticker itself rather than an empty cell.
    """
    import yfinance as yf
    out: Dict[str, str] = {}
    for t in tickers:
        if t in _NAME_CACHE:
            out[t] = _NAME_CACHE[t]
            continue
        nom = t
        try:
            info = yf.Ticker(t).info or {}
            nom = (info.get("longName") or info.get("shortName") or t).strip() or t
        except Exception as e:
            logger.debug("Name lookup failed for %s: %s", t, e)
        _NAME_CACHE[t] = nom
        out[t] = nom
    return out


def _last_prices(tickers: List[str]) -> Dict[str, float]:
    """Most recent close per ticker, in the ticker's own quote currency."""
    import yfinance as yf
    out: Dict[str, float] = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            raw = yf.download(chunk, period="5d", interval="1d",
                              auto_adjust=True, progress=False, threads=False)
            if raw is None or raw.empty:
                continue
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if not isinstance(raw.columns, pd.MultiIndex) and len(chunk) == 1:
                close = close.copy()
                close.columns = chunk
            for t in close.columns:
                s = close[t].dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
        except Exception as e:
            logger.warning("Price download failed for %s: %s", chunk[:3], e)
    return out


def load_current_positions(
    path: Optional[str] = None,
    universe: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Signed share count per ticker for what is CURRENTLY held, and where it
    came from.

    Without this the sheet cannot tell a new position from an adjustment, and —
    worse — a position that has dropped out of the target is simply ABSENT from
    the sheet, with nothing telling you to close it. Silence is the one thing a
    manual order sheet must never do.

    Sources, in order of trust:

      1. a CSV you supply (--positions): columns ticker and quantity, signed
         (positive long, negative short). Export it from TWS, or write it by
         hand. The most reliable, because it is what you actually hold.
      2. .cache/risk_state.json, the register the system keeps itself. Needs no
         connection, but it drifts from reality if an order was placed or
         cancelled outside the algo.
      3. nothing — every line is then reported as NEW, and the sheet says so.

    Positions are keyed on the BARE symbol (suffix stripped), because that is
    how IBKR reports them and how RiskState stores them.
    """
    def _bare(t: str) -> str:
        return t.split(".")[0] if "." in t else t

    if path:
        try:
            df = pd.read_csv(path)
            cols = {c.lower().strip(): c for c in df.columns}
            tc = next((cols[k] for k in ("ticker", "symbol", "symbole") if k in cols), None)
            qc = next((cols[k] for k in ("quantity", "qty", "shares", "quantite",
                                         "titres", "position") if k in cols), None)
            if tc is None or qc is None:
                raise ValueError(
                    f"needs a ticker column and a quantity column, found {list(df.columns)}")
            pos = {_bare(str(t)): float(q) for t, q in zip(df[tc], df[qc])
                   if str(t).strip() and q == q}
            logger.info("Current positions: %d line(s) from %s.", len(pos), path)
            return pos, f"file {path}"
        except Exception as e:
            logger.error("Could not read positions from %s (%s) — the sheet will "
                         "report every line as NEW and will NOT tell you what to "
                         "close.", path, e)
            return {}, "unreadable file"

    try:
        from .risk import RiskManager, RiskConfig as _RC
        rm = RiskManager(_RC())
        pos = {}
        for t, ps in rm.state.positions.items():
            if universe and getattr(ps, "universe", None) != universe:
                continue
            q = float(getattr(ps, "shares", 0) or 0)
            if q == 0:
                continue
            pos[_bare(t)] = q if getattr(ps, "side", "LONG") == "LONG" else -abs(q)
        if pos:
            logger.info("Current positions: %d line(s) from the internal register "
                        "(.cache/risk_state.json)%s.", len(pos),
                        f" for {universe}" if universe else "")
            return pos, "internal register"
    except Exception as e:
        logger.debug("Internal register unavailable: %s", e)

    logger.warning(
        "NO current positions found — every line will be reported as NEW and the "
        "sheet will NOT tell you what to close. Pass --positions <file.csv> with "
        "ticker and quantity columns to get adjustments and closures.")
    return {}, "none"


def _vol_stop_pct(returns: pd.DataFrame, ticker: str, rcfg: RiskConfig) -> float:
    """
    The level the risk monitor would cut at, which is NOT the level the resting
    STP order sits at.  RiskManager.check_positions uses
    clip(2 x monthly sigma, 6%, 20%); the order placed at IBKR uses a flat
    stop_loss_pct.  Both are reported so the gap is visible.
    """
    if ticker not in returns.columns:
        return rcfg.stop_loss_pct
    s = returns[ticker].dropna()
    if len(s) < 12:
        return rcfg.stop_loss_pct
    ann = float(s.std() * np.sqrt(12))
    monthly = ann / np.sqrt(12)
    return float(np.clip(rcfg.stop_vol_multiplier * monthly,
                         rcfg.min_stop_pct, rcfg.max_stop_pct))


def build_rows(
    universe: str,
    snapshot,
    returns: pd.DataFrame,
    nav_base: float,
    base_currency: str,
    cfg: TaurusConfig,
    rcfg: RiskConfig,
    prices: Optional[Dict[str, float]] = None,
    fundamentals: Optional[pd.DataFrame] = None,
    factors: Optional[pd.DataFrame] = None,
    as_of=None,
    current: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """One row per position for a single universe."""
    ucfg = REGISTRY.get(universe).config
    half = cfg.gross_leverage / 2.0

    legs = [("LONG", snapshot.long_weights), ("SHORT", snapshot.short_weights)]
    tickers = [t for _, w in legs for t in w.index]
    if not tickers:
        return []

    px = prices if prices is not None else _last_prices(tickers)
    noms = _company_names(tickers)

    # ── Target price from the valuation leg ──────────────────────────────── #
    # The take-profit sits at the TARGET PRICE, not at a fixed percentage. That
    # needs a valuation that does not take the price as an input, which is what
    # taurus/valuation.py provides; the MM divergence it replaces ran from
    # -0.00% to -1.32% on current data and would have put the target on top of
    # the entry. Names the model refuses to value get NO take-profit and a
    # stated reason — a fabricated target is worse than none.
    val = pd.DataFrame()
    if fundamentals is not None and not fundamentals.empty:
        calib = load_calibration(as_of)
        if calib:
            # The WHOLE universe is passed, not just the selected leg: the
            # sector asset betas are medians over sector members, and a leg of
            # a dozen names has too few per sector to estimate them — one
            # sector would be measured and the rest would silently fall back
            # to a global median.  Only the rows needed are read back out.
            val = valuation_screen(
                fundamentals, calib, returns=returns,
                market_returns=((factors["Mkt-RF"] + factors["RF"])
                                if (factors is not None
                                    and "Mkt-RF" in factors.columns) else None),
                factors=factors, cfg=cfg,
            )
        else:
            logger.warning(
                "[%s] No calibration available — no target price this run. "
                "Run `--mode calibrate` first.", universe,
            )

    rows: List[dict] = []
    for side, weights in legs:
        for ticker, w in weights.items():
            sym, ccy, minor, primary = ticker_market(ticker, ucfg)
            price = px.get(ticker, 0.0)
            if not price or price <= 0:
                rows.append({
                    "Ticker": ticker, "Societe": noms.get(ticker, ticker),
                    "Univers": universe, "Sens": side,
                    "Alerte": "AUCUN PRIX — position non dimensionnee",
                })
                continue

            rate = _fx(ccy, base_currency)
            if rate is None:
                rows.append({
                    "Ticker": ticker, "Societe": noms.get(ticker, ticker),
                    "Univers": universe, "Sens": side,
                    "Alerte": f"AUCUN TAUX {ccy}->{base_currency} — non dimensionnee",
                })
                continue

            price_base    = price / float(minor) * rate
            notional_base = float(w) * nav_base * half
            lot  = max(int(getattr(ucfg, "min_lot_size", 1) or 1), 1)
            brut = notional_base / price_base
            shares = float(int(brut) // lot * lot)

            vr = val.loc[ticker] if (len(val) and ticker in val.index) else None
            cible = (float(vr["target_factor"]) if (vr is not None
                     and vr.get("eligible") and vr.get("target_factor") == vr.get("target_factor"))
                     else np.nan)
            motif_val = "" if cible == cible else (
                str(vr["reason"]) if vr is not None else "not valued")

            # A take-profit on the WRONG SIDE of the entry is not a
            # take-profit: on a long it would sit below the stop and close the
            # position at a loss the moment it filled.  It happens here for a
            # structural reason, not a bug — a book-anchored residual income
            # model cannot reach the multiples the market pays for
            # intangible-heavy companies (it tops out near 1.2-1.6x book with
            # the measured persistence, against a CAC 40 median of 1.87x and
            # 8.6x for Safran), because R&D, brands and certifications are
            # expensed rather than capitalised and so are absent from the
            # capital base.  The level is therefore suppressed and the reason
            # stated, rather than handed over as an order.
            if cible == cible:
                marge = (cible - 1.0) if side == "LONG" else (1.0 - cible)
                if marge <= 0:
                    motif_val = (
                        f"target {('below' if side == 'LONG' else 'above')} entry "
                        f"({(cible - 1) * 100:+.0f}%) — book-anchored value cannot "
                        f"reach the market multiple; not usable as a take-profit"
                    )
                    cible = np.nan
            stop_order = rcfg.stop_loss_pct
            stop_mon   = _vol_stop_pct(returns, ticker, rcfg)

            sens = 1.0 if side == "LONG" else -1.0

            # Cible signee : un short est une quantite negative, comme IBKR la
            # rapporte, pour que le delta se calcule sans ambiguite de signe.
            cible_titres = shares if side == "LONG" else -shares
            actuel = float((current or {}).get(sym, 0.0))
            delta = cible_titres - actuel
            if actuel == 0:
                action = "NOUVELLE"
            elif abs(delta) < 1:
                action = "INCHANGEE"
            elif (cible_titres > 0) == (actuel > 0):
                action = "RENFORCER" if abs(cible_titres) > abs(actuel) else "REDUIRE"
            else:
                # Le signe s'inverse : ce n'est pas un ajustement mais un
                # retournement, et il passe par zero. A signaler comme tel.
                action = "RETOURNEMENT"
            sens_ordre = "" if abs(delta) < 1 else ("ACHAT" if delta > 0 else "VENTE")

            rows.append({
                "Ticker":            ticker,
                "Societe":           noms.get(ticker, ticker),
                "Univers":           universe,
                "Sens":              side,
                "Symbole IBKR":      sym,
                "Place":             primary or ucfg.ibkr_exchange,
                "Devise":            ccy,
                "Prix entree":       round(price, 4),
                f"Prix en {base_currency}": round(price_base, 4),
                "Titres":            shares,
                f"Notionnel {base_currency}": round(shares * price_base, 2),
                "Poids jambe %":     round(float(w) * 100, 3),
                "Stop ordre":        round(price * (1 - sens * stop_order), 4),
                "Stop ordre %":      round(stop_order * 100, 1),
                "Stop moniteur":     round(price * (1 - sens * stop_mon), 4),
                "Stop moniteur %":   round(stop_mon * 100, 1),
                "Prix cible":        round(price * cible, 4) if cible == cible else None,
                "Potentiel %":       round((cible - 1.0) * 100, 1) if cible == cible else None,
                "Take profit":       round(price * cible, 4) if cible == cible else None,
                "Valorisation":      (round(float(vr["intrinsic_equity"]), 0)
                                      if (cible == cible and vr is not None) else None),
                "Metrique":          (str(vr["metric"]).upper()
                                      if (vr is not None and vr.get("metric")) else ""),
                "Cout du capital %": (round(float(vr["cost_of_capital"]) * 100, 2)
                                      if (vr is not None
                                          and vr.get("cost_of_capital") == vr.get("cost_of_capital"))
                                      else None),
                "Sans prix cible":   motif_val,
                # Ce qu'il faut REELLEMENT passer comme ordre, pas seulement la
                # cible.  Sans ca il faut recalculer a la main l'ecart avec
                # l'existant, ligne par ligne, et c'est la que les erreurs
                # arrivent.
                "Action":            action,
                "Titres actuels":    actuel,
                "Titres cibles":     cible_titres,
                "A passer":          delta,
                "Ordre":             sens_ordre,
                "Alerte":            "" if shares > 0 else "sous le lot minimum",
            })

    # ── Positions a SOLDER ───────────────────────────────────────────────── #
    # Ce que le fichier passait silencieusement sous silence : une position
    # detenue qui n'est plus dans la cible disparaissait simplement du tableau.
    # Rien ne disait de la fermer, donc elle restait ouverte indefiniment — le
    # meme defaut que l'absence de mode flatten au niveau de l'univers, mais
    # ligne par ligne.
    if current:
        vises = {ticker_market(t, ucfg)[0] for _, w in legs for t in w.index}
        sortants = {sym: q for sym, q in current.items()
                    if sym not in vises and abs(q) >= 1}
        if sortants:
            # Le nom ne peut PAS etre cherche sur le symbole nu. Une position
            # IBKR ne porte que "AI", et interroger "AI" renvoie C3.ai, une
            # societe americaine, au lieu d'Air Liquide — la meme ambiguite que
            # SAN pour Sanofi et Banco Santander. Sur une ligne "a fermer
            # entierement", un mauvais nom peut faire solder la mauvaise valeur.
            # On remonte donc au ticker complet via la liste de l'univers, et
            # faute de correspondance on n'invente pas de nom.
            from .universe import _FALLBACKS
            vers_complet = {}
            for t in _FALLBACKS.get(universe, []):
                vers_complet.setdefault(ticker_market(t, ucfg)[0], t)
            complets = {sym: vers_complet.get(sym) for sym in sortants}
            a_chercher = [c for c in complets.values() if c]
            noms_s = _company_names(a_chercher) if a_chercher else {}
            inconnus = [s_ for s_, c in complets.items() if not c]
            if inconnus:
                logger.warning(
                    "[%s] %d position(s) to close are not in this universe's "
                    "list, so their company name cannot be resolved safely and "
                    "is left blank: %s. Check what they are before closing.",
                    universe, len(inconnus), ", ".join(sorted(inconnus)))
            logger.info("[%s] %d position(s) to CLOSE — out of the target: %s",
                        universe, len(sortants), ", ".join(sorted(sortants)))
            for sym, q in sorted(sortants.items()):
                plein = complets.get(sym)
                rows.append({
                    "Ticker":         plein or sym,
                    "Societe":        noms_s.get(plein, "") if plein else "",
                    "Univers":        universe,
                    "Sens":           "LONG" if q > 0 else "SHORT",
                    "Symbole IBKR":   sym,
                    "Action":         "SOLDER",
                    "Titres actuels": q,
                    "Titres cibles":  0.0,
                    "A passer":       -q,
                    "Ordre":          "VENTE" if q > 0 else "ACHAT",
                    "Alerte":         "sortie du signal — a fermer entierement",
                })
    return rows


def write_workbook(
    rows_by_universe: Dict[str, List[dict]],
    path: Path,
    base_currency: str,
    nav_base: float,
    weights: Dict[str, float],
) -> Path:
    """Write one sheet per universe, a combined sheet and a summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    toutes = [r for rs in rows_by_universe.values() for r in rs]
    combined = pd.DataFrame(toutes)

    col_notional = f"Notionnel {base_currency}"

    def _leg(rs, sens):
        return sum(abs(r.get(col_notional, 0.0) or 0.0)
                   for r in rs if r.get("Sens") == sens)

    lignes = []
    for u, rs in rows_by_universe.items():
        nav_u = nav_base * weights.get(u, 0.0)
        longs, shorts = _leg(rs, "LONG"), _leg(rs, "SHORT")
        brut = longs + shorts
        lignes.append({
            "Univers": u,
            "Poids NAV %": round(weights.get(u, 0.0) * 100, 2),
            f"NAV allouee {base_currency}": round(nav_u, 2),
            "Positions": len(rs),
            "Longs": sum(1 for r in rs if r.get("Sens") == "LONG"),
            "Shorts": sum(1 for r in rs if r.get("Sens") == "SHORT"),
            "Non dimensionnees": sum(1 for r in rs if r.get("Alerte")),
            f"Jambe longue {base_currency}": round(longs, 2),
            f"Jambe courte {base_currency}": round(shorts, 2),
            f"Brut deploye {base_currency}": round(brut, 2),
            "Brut / NAV allouee": round(brut / nav_u, 3) if nav_u > 0 else None,
            # What is deliberately NOT invested.  With deploy_only_what_fits a
            # leg holding few candidates stops at n x max_position_weight and
            # the balance stays in cash rather than being inflated back to
            # fully invested — so this figure is expected to be positive, not
            # a sign that something failed.
            f"Cash conserve {base_currency}": round(max(nav_u - brut, 0.0), 2),
        })

    total_brut = sum(l[f"Brut deploye {base_currency}"] for l in lignes)
    lignes.append({
        "Univers": "TOTAL",
        "Poids NAV %": round(sum(l["Poids NAV %"] for l in lignes), 2),
        f"NAV allouee {base_currency}": round(nav_base, 2),
        "Positions": sum(l["Positions"] for l in lignes),
        "Longs": sum(l["Longs"] for l in lignes),
        "Shorts": sum(l["Shorts"] for l in lignes),
        "Non dimensionnees": sum(l["Non dimensionnees"] for l in lignes),
        f"Jambe longue {base_currency}": round(
            sum(l[f"Jambe longue {base_currency}"] for l in lignes), 2),
        f"Jambe courte {base_currency}": round(
            sum(l[f"Jambe courte {base_currency}"] for l in lignes), 2),
        f"Brut deploye {base_currency}": round(total_brut, 2),
        "Brut / NAV allouee": round(total_brut / nav_base, 3) if nav_base > 0 else None,
        f"Cash conserve {base_currency}": round(max(nav_base - total_brut, 0.0), 2),
    })
    resume = pd.DataFrame(lignes)

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        resume.to_excel(xl, sheet_name="Resume", index=False)
        if not combined.empty:
            combined.to_excel(xl, sheet_name="Toutes positions", index=False)
        for u, rs in rows_by_universe.items():
            if rs:
                pd.DataFrame(rs).to_excel(xl, sheet_name=u[:31], index=False)

        # Column widths, so the file is readable without resizing by hand
        for ws in xl.book.worksheets:
            for col in ws.columns:
                longueur = max((len(str(c.value)) for c in col if c.value is not None),
                               default=8)
                ws.column_dimensions[col[0].column_letter].width = min(longueur + 2, 42)
            ws.freeze_panes = "A2"

    return path
