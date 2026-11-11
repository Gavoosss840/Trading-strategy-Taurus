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


def load_positions_from_ibkr(
    cfg: TaurusConfig,
    universe: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, str], str]:
    """
    Read what is held straight from IBKR. READ-ONLY: it reads positions and
    disconnects, it never places, modifies or cancels anything.

    More accurate than any export, and for one reason beyond freshness: an
    IBKR position carries its full CONTRACT, so currency and primary exchange
    come with it. A CSV gives only "SAN", which is Sanofi in Paris and Banco
    Santander in Madrid; the contract says which. That matters most on a line
    telling you to close a position entirely.

    Returns {bare symbol: signed quantity}, {bare symbol: full Yahoo ticker}
    for the ones that could be resolved, and a description of the source.
    """
    from .universe import REGISTRY, ticker_market, _FALLBACKS

    try:
        from .execution import IBKRConnection
    except Exception as e:
        logger.error("IBKR layer unavailable (%s) — cannot read positions.", e)
        return {}, {}, "IBKR unavailable"

    conn = IBKRConnection(cfg)
    if not conn.ensure_connected():
        logger.error("Cannot reach IBKR TWS/Gateway — is it running, and is the "
                     "API enabled? No positions read.")
        return {}, {}, "IBKR unreachable"

    try:
        brutes = conn.ib.positions(account=conn.cfg.ibkr_account or "")
    except Exception as e:
        logger.error("Reading positions from IBKR failed: %s", e)
        brutes = []
    finally:
        try:
            conn.ib.disconnect()
        except Exception:
            pass

    brutes = [p for p in brutes if getattr(p, "position", 0)]
    if not brutes:
        logger.info("IBKR reports no open position.")
        return {}, {}, "IBKR (no position)"

    ucfg = REGISTRY.get(universe).config if universe else None
    # Candidats de cet univers, indexes par (symbole nu, devise de cotation)
    candidats: Dict[Tuple[str, str], List[str]] = {}
    if universe:
        for t in _FALLBACKS.get(universe, []):
            sym, ccy, _m, _px = ticker_market(t, ucfg)
            candidats.setdefault((sym, ccy), []).append(t)

    # Les positions sont indexees sur le TICKER COMPLET, jamais sur le symbole
    # nu. Deux cotations du meme symbole coexistent dans un portefeuille —
    # Sanofi (SAN/EUR/SBF) et Banco Santander (SAN/EUR/BM) — et une cle nue
    # fait que la seconde ECRASE la premiere : la feuille annoncerait la
    # quantite de l'une sous le nom de l'autre, et l'autre disparaitrait sans
    # un mot. C'est le defaut qui a failli etre livre.
    pos: Dict[str, float] = {}
    plein: Dict[str, str] = {}
    hors, ambigus, collisions = [], [], []
    for p in brutes:
        c = p.contract
        sym = str(getattr(c, "symbol", "") or "")
        ccy = str(getattr(c, "currency", "") or "")
        place = str(getattr(c, "primaryExchange", "") or "")
        if not sym:
            continue

        if universe:
            choix = candidats.get((sym, ccy), [])
            if not choix:
                hors.append(f"{sym}/{ccy}")
                continue                      # appartient a un autre univers
            if len(choix) > 1:
                # Deux cotations du meme symbole dans la meme devise : seule la
                # place les separe. Sans elle on n'invente pas.
                par_place = [t for t in choix
                             if ticker_market(t, ucfg)[3] == place]
                if len(par_place) == 1:
                    choix = par_place
                else:
                    ambigus.append(f"{sym}/{ccy}" + (f"@{place}" if place else ""))
                    cle = f"{sym}@{place}" if place else sym
                    if cle in pos:
                        collisions.append(cle)
                    pos[cle] = pos.get(cle, 0.0) + float(p.position)
                    continue
            # Un seul candidat ne suffit PAS a conclure : si la position vient
            # d'une autre place, c'est une autre societe qui porte le meme
            # symbole dans la meme devise. Sans ce controle, Banco Santander
            # (SAN/EUR/BM) etait agrege dans Sanofi (SAN/EUR/SBF) des que
            # l'univers ne listait que Sanofi — et la quantite affichee devenait
            # la somme des deux.
            attendue = ticker_market(choix[0], ucfg)[3]
            if place and attendue and place != attendue:
                hors.append(f"{sym}/{ccy}@{place}")
                continue
            cle = choix[0]
            plein[cle] = choix[0]
        else:
            cle = f"{sym}@{place}" if place else sym

        if cle in pos:
            # Deux lignes IBKR sur le MEME contrat : on additionne plutot que
            # d'en perdre une, et on le signale.
            collisions.append(cle)
            pos[cle] = pos[cle] + float(p.position)
        else:
            pos[cle] = float(p.position)

    logger.info("IBKR: %d position(s) read%s, %d resolved to a full ticker.",
                len(pos), f" for {universe}" if universe else "", len(plein))
    if hors:
        logger.debug("Not in %s: %s", universe, ", ".join(sorted(hors)[:12]))
    if collisions:
        logger.warning(
            "%d contract(s) appeared more than once in the IBKR positions and "
            "their quantities were SUMMED: %s. Check them — a contract should "
            "normally appear once.", len(collisions), ", ".join(sorted(set(collisions))))
    if ambigus:
        logger.warning(
            "%d position(s) match several listings of the same symbol in the "
            "same currency and the primary exchange did not separate them, so "
            "their company name is left blank: %s. Check what they are before "
            "acting on their line.", len(ambigus), ", ".join(sorted(ambigus)))
    return pos, plein, "IBKR"


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
    current_full: Optional[Dict[str, str]] = None,
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

    # ── Lot rounding leaves capital on the table — put it back to work ───── #
    plan_titres = _repartir_restant(legs, px, ucfg, cfg, nav_base, half,
                                    base_currency, universe)

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
            # Share count comes from the leg-level plan, which rounds to whole
            # lots AND redistributes what the rounding leaves over.
            shares = float(plan_titres.get((side, ticker), 0.0))

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
            # Par TICKER COMPLET d'abord : c'est la cle que la lecture IBKR
            # produit, et la seule qui distingue deux cotations du meme
            # symbole. Le symbole nu reste accepte pour un CSV ou le registre
            # interne, qui ne portent pas la place de cotation.
            cur = current or {}
            actuel = float(cur.get(ticker, cur.get(sym, 0.0)))
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
                "Alerte":            "" if shares > 0 else "lot minimum au-dessus du plafond par ligne",
            })

    # ── Positions a SOLDER ───────────────────────────────────────────────── #
    # Ce que le fichier passait silencieusement sous silence : une position
    # detenue qui n'est plus dans la cible disparaissait simplement du tableau.
    # Rien ne disait de la fermer, donc elle restait ouverte indefiniment — le
    # meme defaut que l'absence de mode flatten au niveau de l'univers, mais
    # ligne par ligne.
    if current:
        # On compare sur les DEUX formes, parce que `current` peut etre indexe
        # par ticker complet (lecture IBKR) ou par symbole nu (CSV, registre).
        vises = set()
        for _, w in legs:
            for t in w.index:
                vises.add(t)
                vises.add(ticker_market(t, ucfg)[0])
        sortants = {k: q for k, q in current.items()
                    if k not in vises and abs(q) >= 1}
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
            # La resolution faite par IBKR a partir du contrat prime sur celle
            # deduite de la liste : elle vient de la devise et de la place
            # reelles de la position, pas d'une correspondance de symbole.
            def _resoudre(k: str) -> Optional[str]:
                # La cle est deja un ticker complet quand elle vient d'IBKR.
                if "." in k and k in vers_complet.values():
                    return k
                base = k.split("@")[0]
                return (current_full or {}).get(k) or vers_complet.get(base)
            complets = {k: _resoudre(k) for k in sortants}
            a_chercher = [c for c in complets.values() if c]
            noms_s = _company_names(a_chercher) if a_chercher else {}
            inconnus = [k for k, c in complets.items() if not c]
            if inconnus:
                logger.warning(
                    "[%s] %d position(s) to close are not in this universe's "
                    "list, so their company name cannot be resolved safely and "
                    "is left blank: %s. Check what they are before closing.",
                    universe, len(inconnus), ", ".join(sorted(inconnus)))
            logger.info("[%s] %d position(s) to CLOSE — out of the target: %s",
                        universe, len(sortants), ", ".join(sorted(sortants)))
            for cle, q in sorted(sortants.items()):
                plein = complets.get(cle)
                sym = cle.split("@")[0]
                rows.append({
                    "Ticker":         plein or cle,
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


def _repartir_restant(
    legs,
    px: Dict[str, float],
    ucfg,
    cfg: TaurusConfig,
    nav_base: float,
    half: float,
    base_currency: str,
    universe: str,
) -> Dict[tuple, float]:
    """
    Share count per line, after putting the lot-rounding remainder back to work.

    Every target notional is rounded DOWN to a whole lot, so each line leaves a
    fraction of a lot in cash and a line whose target is smaller than one lot
    is dropped outright.  On the Tokyo book, where the lot is 100 shares, that
    was 17 of 51 lines unsized and only 55.6% of the allocation deployed,
    against 88% elsewhere — the capital was not held back by choice, it simply
    could not be spent.

    The remainder is handed out one lot at a time to the line with the largest
    unmet demand, and the per-position cap is enforced throughout: a line is
    only offered another lot if that lot keeps it under max_position_weight.
    So the leftover of a name the lot size prices out of the book moves to the
    names that can carry it, and no position grows past the risk limit that
    the cap exists to impose.
    """
    plan: Dict[tuple, float] = {}
    cap_w = float(getattr(cfg, "max_position_weight", 0.08) or 0.08)
    lot = max(int(getattr(ucfg, "min_lot_size", 1) or 1), 1)

    for side, weights in legs:
        budget = nav_base * half
        if budget <= 0 or not len(weights):
            for t in weights:
                plan[(side, t)] = 0.0
            continue

        infos: Dict[str, dict] = {}
        for ticker, w in weights.items():
            price = px.get(ticker, 0.0)
            if not price or price <= 0:
                continue
            sym, ccy, minor, _primary = ticker_market(ticker, ucfg)
            rate = _fx(ccy, base_currency)
            if rate is None:
                continue
            pb = price / float(minor) * rate
            if pb <= 0:
                continue
            vise = float(w) * budget
            cap_titres = float(int((cap_w * budget) / pb) // lot * lot)
            base = float(int(vise / pb) // lot * lot)
            base = min(base, cap_titres)
            infos[ticker] = {"pb": pb, "vise": vise, "titres": base,
                             "cap": cap_titres}
            plan[(side, ticker)] = base

        if not infos:
            continue

        restant = budget - sum(i["titres"] * i["pb"] for i in infos.values())
        ajoutes = 0
        while restant > 0:
            # Largest shortfall against the intended notional, among the lines
            # that can still take a lot without breaching the position cap and
            # without overspending what is left.
            cand = [
                (i["vise"] - i["titres"] * i["pb"], t)
                for t, i in infos.items()
                if i["titres"] + lot <= i["cap"] and lot * i["pb"] <= restant
            ]
            if not cand:
                break
            _manque, t = max(cand)
            infos[t]["titres"] += lot
            plan[(side, t)] = infos[t]["titres"]
            restant -= lot * infos[t]["pb"]
            ajoutes += 1
            if ajoutes > 10_000:            # never spin on a pathological price
                break

        if ajoutes:
            deploye = sum(i["titres"] * i["pb"] for i in infos.values())
            logger.info(
                "[%s] %s leg: %d extra lot(s) placed from the rounding "
                "remainder — deployed %.1f%% of the leg budget (cap %.0f%% "
                "per line respected).",
                universe, side, ajoutes, 100.0 * deploye / budget, cap_w * 100,
            )
    return plan


def consolider_entre_univers(rows_by_universe: Dict[str, List[dict]]) -> None:
    """
    Net the orders of a ticker that several universes hold at once.

    The S&P 500 contains most of the NASDAQ 100, so the same company is
    selected by two universes and legitimately held twice: the combined book
    is the Sharpe-weighted sum of the universe books, so two target lines ARE
    the intended exposure and must be added, not deduplicated.

    What cannot be added is the POSITION ALREADY HELD.  Each universe reads the
    broker independently and sees the whole holding, so each subtracted it in
    full.  Measured on a real sheet: MU was held 3 shares and both universes
    ordered 3 sold — six for a three-share line.  HON was held 9.5 with a
    combined target of -57; one sheet ordered -66.5 and the other -9.5, so
    executing both would have sold 76.  PEP and PYPL were worse than wrong in
    size, they were opposite in direction: one sheet said buy to close, the
    other said sell to deepen.

    The holding is therefore attributed ONCE, to the line carrying the largest
    target, and the delta of every line of that ticker is recomputed against
    the consolidated figures.  The other lines keep their target for the record
    and carry no order, with the reason written in the alert column so nobody
    re-adds by hand what was deliberately removed.
    """
    par_ticker: Dict[str, List[dict]] = {}
    for rs in rows_by_universe.values():
        for r in rs:
            par_ticker.setdefault(str(r.get("Ticker", "")), []).append(r)

    for ticker, lignes in par_ticker.items():
        if len(lignes) < 2:
            continue

        cible_totale = sum(float(r.get("Titres cibles", 0.0) or 0.0) for r in lignes)
        # The holding is the same number read several times, not several
        # holdings: take the largest magnitude rather than the sum.
        detenu = max((float(r.get("Titres actuels", 0.0) or 0.0) for r in lignes),
                     key=abs, default=0.0)
        delta_total = cible_totale - detenu

        porteur = max(lignes, key=lambda r: abs(float(r.get("Titres cibles", 0.0) or 0.0)))
        univers_tous = ", ".join(sorted(str(r.get("Univers", "")) for r in lignes))

        for r in lignes:
            if r is porteur:
                r["Titres actuels"] = detenu
                r["A passer"] = delta_total
                r["Ordre"] = ("" if abs(delta_total) < 1
                              else ("ACHAT" if delta_total > 0 else "VENTE"))
                if abs(cible_totale) < 1e-9 and abs(detenu) > 0:
                    r["Action"] = "SOLDER"
                elif detenu == 0:
                    r["Action"] = "NOUVELLE"
                elif cible_totale * detenu < 0:
                    r["Action"] = "RETOURNEMENT"
                elif abs(cible_totale) > abs(detenu):
                    r["Action"] = "RENFORCER"
                elif abs(cible_totale) < abs(detenu):
                    r["Action"] = "REDUIRE"
                else:
                    r["Action"] = "INCHANGEE"
                note = (f"ordre consolide sur {univers_tous} — "
                        f"cible totale {cible_totale:+.0f}, detenu {detenu:+.0f}")
                r["Alerte"] = f"{r.get('Alerte') or ''} {note}".strip()
            else:
                r["Titres actuels"] = 0.0
                r["A passer"] = 0.0
                r["Ordre"] = ""
                r["Action"] = "VOIR " + str(porteur.get("Univers", "")).upper()
                note = (f"titre partage ({univers_tous}) — ordre porte par la "
                        f"ligne {porteur.get('Univers', '')}, ne pas passer deux fois")
                r["Alerte"] = f"{r.get('Alerte') or ''} {note}".strip()


def write_workbook(
    rows_by_universe: Dict[str, List[dict]],
    path: Path,
    base_currency: str,
    nav_base: float,
    weights: Dict[str, float],
) -> Path:
    """Write one sheet per universe, a combined sheet and a summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    consolider_entre_univers(rows_by_universe)
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
            # Lines that carry NO position, not lines that carry a note.  The
            # cross-universe consolidation writes its reasoning into the alert
            # column, so counting alerts made this read 70 when only 5 lines
            # were actually unsized — the column said the sheet was broken when
            # it was explaining itself.
            "Non dimensionnees": sum(
                1 for r in rs
                if not float(r.get("Titres", 0.0) or 0.0)
                and str(r.get("Action", "")) not in ("SOLDER",)
                and not str(r.get("Action", "")).startswith("VOIR")
            ),
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
