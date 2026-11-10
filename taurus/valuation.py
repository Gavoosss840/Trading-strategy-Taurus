"""
Taurus — valuation leg, rebuilt.

The MM module could never value a business: VU is backed out of the market
price, so its divergence collapses to -(distress + agency)/market_cap and is
always <= 0.  This module estimates operating value WITHOUT taking the price as
an input, so the result can disagree with the market in either direction — and
that is what gives a take-profit an actual target price.

Model: residual income in autoregressive form (Ohlson / Penman).  A company is
worth its capital base plus the present value of the returns it earns ABOVE its
cost of capital, with that excess decaying towards zero at a measured rate:

    V = B + B x (R0 - r) x q / (1 - q),      q = phi / (1 + r)

B is the capital base, R0 the current return on it, r the cost of that capital
and phi the measured persistence of the excess return.  Because phi < 1 < 1 + r,
q < 1 and the infinite sum converges in closed form — so there is NO horizon
parameter to choose.  That matters: every free parameter is a surface for
overfitting, and this model has exactly one that is not measured.

Which capital base, per sector
------------------------------
Assigned from data, not from a textbook, on four criteria measured per sector
per metric: coverage, persistence phi, contamination by leverage, and
implausible values.  Persistence decides, because the model projects the metric
forward — a metric that does not predict itself cannot support a projection.

  ROIC  = EBIT x (1 - effective tax) / invested capital, discounted at the
          UNLEVERED cost of capital.  Immune to buybacks: a buyback funded from
          cash moves equity -X and cash -X, so invested capital (D + E - C) is
          unchanged; funded by debt, +X and -X, also unchanged.  Measured:
          Home Depot's return drops from 122% on equity to 20.8% on invested
          capital, Boeing's from 164% to 11.8%, and McDonald's, Lowe's and
          Starbucks — which have NEGATIVE book equity and no ROE at all —
          come back with 19-25%.  Cross-sectional dispersion falls from 57.8
          points to 19.7: two thirds of the ROE's spread was capital-structure
          noise.

  ROE   = net income to common shareholders / average common equity, discounted
          at the levered cost of equity.  Kept where the data says so, which
          includes lenders: invested capital is not defined for a bank (its
          interest expense is revenue, not a financing cost), and the ROIC
          coverage measured there is 32.8% against 82.4% for ROE.

Utilities get NEITHER: ROE comes out at phi = -0.21 with R2 = 0.065 and ROIC at
R2 = 0.147, i.e. no persistence on either side.  Economically coherent — a
regulated return is set by the regulator, so cross-sectional deviations do not
persist in a way any projection can use.  Those names are still tradable on the
other two signal legs; they simply get no target price.

Calibration
-----------
The PROCEDURE is frozen; the NUMBERS are re-estimated quarterly and archived
with the date they were produced.  Updating estimates under an unchanged rule
is maintenance; changing the rule because the estimates displease is
overfitting, and they must not be confused.

Dating the archive also fixes the look-ahead going forward: load_calibration
returns the most recent calibration NOT LATER than as_of, so from the first
stored calibration onward the parameters a rebalance sees are the ones that
existed then.  The historical backtest stays contaminated — phi is estimated on
2022-2026 statements, the only ones yfinance exposes — and that cannot be
undone, only disclosed.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TaurusConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent / "calibration"

# ── The frozen procedure ─────────────────────────────────────────────────── #
# These are the estimation RULES.  They do not change between calibrations.
WINSOR = (-0.50, 1.00)      # a return on capital outside this is a collapsed denominator
COUV_MIN = 0.70             # a metric must be computable for 70% of the sector
R2_MIN = 0.25               # below this, phi is noise whatever its value
PHI_BOUNDS = (0.0, 1.0)     # outside ]0,1[ the process explodes or oscillates
R2_TIE = 0.05               # within this, pick the metric less contaminated by leverage
MIN_FIRMS = 4               # fewer firms than this and the sector is not estimated
MIN_TRANS = 15              # fewer transitions than this and phi is not estimated
JUMP_ALERT = 0.25           # a phi moving more than this between calibrations is flagged

# The one parameter that is NOT measured.  The realised Mkt-RF over the
# available window is 13.83% annualised for the US, 8.31% for Europe and 7.75%
# for Japan — realised returns over a long bull market, not expected premia.
# Discounting at 15% would send every valuation to zero.  Forward estimates in
# the literature cluster at 4-6%; the sensitivity to this choice is reported
# rather than hidden.
EQUITY_RISK_PREMIUM = 0.050

# Sectors where invested capital is not a meaningful denominator, whatever the
# measured persistence says: interest expense is revenue for a lender.
LENDERS = frozenset({"Financial Services", "Financials", "Banks", "Insurance"})

# ── Documented overrides of the rule ─────────────────────────────────────── #
# Applied ON TOP of the estimation, never by bending the procedure, so the
# rule's own answer stays visible in the archive next to the override and its
# reason. Each entry is a judgement call that has to justify itself.
#
# Technology: the rule picks ROE on an R2 of 0.723 against 0.667, a gap of
# 0.056 that sits just above the 0.05 tie-break and is well inside the noise of
# a 144-transition estimate. Meanwhile its ROE correlates 0.31 with leverage
# against -0.02 for ROIC, and technology is the sector where buybacks are
# heaviest — exactly the distortion the capital base was chosen to avoid. The
# cost is a slightly less well estimated persistence; the gain is a denominator
# that cannot be hollowed out by capital returns.
METRIC_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "Technology": ("roic",
                   "rule picked ROE on a 0.056 R2 gap (inside the noise) while "
                   "its ROE is 10x more leverage-contaminated (0.31 vs -0.02) "
                   "in the sector with the heaviest buybacks"),
}


# --------------------------------------------------------------------------- #
#  Per-company metrics                                                        #
# --------------------------------------------------------------------------- #

def invested_capital(row) -> float:
    """Debt + equity - cash.  Unchanged by a buyback, however it is funded."""
    d = float(row.get("total_debt", 0.0) or 0.0)
    e = float(row.get("total_equity", np.nan))
    c = float(row.get("cash", 0.0) or 0.0)
    if not (e == e):
        return np.nan
    return d + e - c


def roic(row) -> float:
    """After-tax operating return on invested capital."""
    ic = invested_capital(row)
    ebit = float(row.get("ebit", np.nan))
    tax = float(row.get("effective_tax_rate", row.get("tax_rate", np.nan)))
    if not (ic == ic and ic > 0 and ebit == ebit and tax == tax):
        return np.nan
    return ebit * (1.0 - tax) / ic


def roe(row) -> float:
    """Return on average COMMON equity, as computed in the fundamentals."""
    v = row.get("roe_normalized", np.nan)
    if v == v:
        return float(v)
    v = row.get("roe", np.nan)
    return float(v) if v == v else np.nan


# --------------------------------------------------------------------------- #
#  Panel + calibration                                                       #
# --------------------------------------------------------------------------- #

def build_panel(tickers: List[str], max_workers: int = 6) -> pd.DataFrame:
    """
    Annual ROE and ROIC per company, from the statements, for calibration.

    yfinance exposes four annual periods, so three transitions per company —
    thin individually, ample pooled across a sector.
    """
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _row(frame, names):
        if frame is None or getattr(frame, "empty", True):
            return None
        idx = {str(r).strip().lower(): r for r in frame.index}
        for n in names:
            r = idx.get(n.strip().lower())
            if r is not None and frame.loc[r].notna().any():
                return frame.loc[r].astype(float)
        return None

    def _one(t: str):
        try:
            tk = yf.Ticker(t)
            inc, bs = tk.income_stmt, tk.balance_sheet
            sector = (tk.info or {}).get("sector", "Unknown") or "Unknown"
        except Exception:
            return []
        ebit = _row(inc, ["EBIT", "Operating Income"])
        ni   = _row(inc, ["Net Income Common Stockholders", "Net Income"])
        taxp = _row(inc, ["Tax Provision"])
        ptx  = _row(inc, ["Pretax Income"])
        dette = _row(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
        fpc  = _row(bs, ["Common Stock Equity", "Stockholders Equity"])
        fpt  = _row(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
        tres = _row(bs, ["Cash And Cash Equivalents",
                         "Cash Cash Equivalents And Short Term Investments"])
        ref = fpc if fpc is not None else fpt
        if ref is None:
            return []

        periodes = sorted(set(ref.index))
        out = []
        for k, per in enumerate(periodes):
            def g(s, d=np.nan):
                if s is None:
                    return d
                v = s.get(per, np.nan)
                return float(v) if v == v else d
            e, n = g(ebit), g(ni)
            fc, ft = g(fpc), g(fpt)
            d, c = g(dette, 0.0), g(tres, 0.0)
            tp, px = g(taxp), g(ptx)
            tx = (float(np.clip(tp / px, 0.05, 0.45))
                  if (tp == tp and px == px and px > 0) else 0.25)
            fc_prev = np.nan
            if k > 0 and fpc is not None:
                v = fpc.get(periodes[k - 1], np.nan)
                fc_prev = float(v) if v == v else np.nan
            fc_moy = (fc + fc_prev) / 2.0 if (fc == fc and fc_prev == fc_prev) else fc
            ci = (0.0 if d != d else d) + (ft if ft == ft else np.nan) - (0.0 if c != c else c)
            out.append({
                "ticker": t, "secteur": sector, "periode": pd.Timestamp(per),
                "roe": (n / fc_moy) if (n == n and fc_moy == fc_moy and fc_moy > 0) else np.nan,
                "roic": (e * (1 - tx) / ci) if (e == e and ci == ci and ci > 0) else np.nan,
                "levier": (d / fc) if (fc == fc and fc > 0) else np.nan,
            })
        return out

    lignes: List[dict] = []
    logger.info("Building the calibration panel over %d tickers…", len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                lignes.extend(fut.result(timeout=120) or [])
            except Exception as e:
                logger.debug("Panel row failed for %s: %s", futs[fut], e)
            if i % 100 == 0:
                logger.info("  %d/%d tickers", i, len(tickers))
    df = pd.DataFrame(lignes)
    logger.info("Panel: %d companies, %d annual observations",
                df["ticker"].nunique() if len(df) else 0, len(df))
    return df


def _phi(sub: pd.DataFrame, ancre: float) -> Tuple[float, float, int]:
    x = (sub["t"] - ancre).values
    y = (sub["t1"] - ancre).values
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < MIN_TRANS or np.var(x, ddof=1) <= 0:
        return np.nan, np.nan, len(x)
    phi = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return phi, r2, len(x)


def estimate_calibration(panel: pd.DataFrame) -> dict:
    """
    Run the frozen procedure over a panel and return the per-sector parameters.

    Nothing here looks at returns, prices or Sharpe ratios: the assignment is
    made on the statistical properties of the accounting metrics alone, so it
    cannot be tuned towards a backtest result.
    """
    if panel.empty:
        return {"sectors": {}, "n_firms": 0, "n_obs": 0}

    p = panel.copy()
    hors = {m: int(((p[m] < WINSOR[0]) | (p[m] > WINSOR[1])).sum())
            for m in ("roe", "roic")}
    for m in ("roe", "roic"):
        p[m + "_w"] = p[m].clip(*WINSOR)
    logger.info("Winsorised to [%.0f%%, %.0f%%]: %d ROE, %d ROIC values",
                WINSOR[0] * 100, WINSOR[1] * 100, hors["roe"], hors["roic"])

    trans = []
    for t, g in p.sort_values("periode").groupby("ticker"):
        for m in ("roe", "roic"):
            v = g[m + "_w"].values
            for k in range(len(v) - 1):
                trans.append({"secteur": g["secteur"].iloc[0], "metrique": m,
                              "t": v[k], "t1": v[k + 1]})
    tr = pd.DataFrame(trans)

    sectors: Dict[str, dict] = {}
    for sect, g in p.groupby("secteur"):
        if g["ticker"].nunique() < MIN_FIRMS:
            continue
        cands = {}
        for m in ("roe", "roic"):
            fini = g[m + "_w"].dropna()
            couv = float(g[m].notna().mean())
            ancre = float(fini.median()) if len(fini) else np.nan
            sub = tr[(tr["secteur"] == sect) & (tr["metrique"] == m)]
            phi, r2, n = _phi(sub, ancre) if len(sub) else (np.nan, np.nan, 0)
            corr = np.nan
            both = g[[m + "_w", "levier"]].dropna()
            if len(both) >= 10 and both["levier"].std() > 0 and both[m + "_w"].std() > 0:
                corr = float(np.corrcoef(both[m + "_w"], both["levier"])[0, 1])
            # Invested capital is not a denominator for a lender, whatever the
            # measured persistence: its interest expense is revenue.
            interdit = (m == "roic" and sect in LENDERS)
            valide = (couv >= COUV_MIN and phi == phi
                      and PHI_BOUNDS[0] < phi < PHI_BOUNDS[1]
                      and r2 == r2 and r2 >= R2_MIN and not interdit)
            cands[m] = {"couverture": round(couv, 4), "ancre": round(ancre, 6),
                        "phi": None if phi != phi else round(phi, 6),
                        "r2": None if r2 != r2 else round(r2, 6),
                        "n_trans": int(n),
                        "corr_levier": None if corr != corr else round(corr, 4),
                        "valide": bool(valide), "exclu_preteur": bool(interdit)}

        valides = [(m, c) for m, c in cands.items() if c["valide"]]
        if not valides:
            choix, motif = None, "no metric passed coverage, phi and R2"
        elif len(valides) == 1:
            choix, motif = valides[0][0], "only metric that passed"
        else:
            valides.sort(key=lambda x: -x[1]["r2"])
            if abs(valides[0][1]["r2"] - valides[1][1]["r2"]) < R2_TIE:
                cl = {m: abs(c["corr_levier"]) if c["corr_levier"] is not None else 9.0
                      for m, c in valides}
                choix = min(cl, key=cl.get)
                motif = f"R2 within {R2_TIE}, less contaminated by leverage"
            else:
                choix = valides[0][0]
                motif = f"higher R2 ({valides[0][1]['r2']:.3f})"

        # Documented override, applied after the rule has spoken so both
        # answers stay in the archive.  It can only move to a metric the rule
        # itself found VALID — an override is a choice between defensible
        # options, never a way to resurrect one the data rejected.
        rule_choice, rule_reason = choix, motif
        override = METRIC_OVERRIDES.get(sect)
        if override is not None:
            want, why = override
            if want == choix:
                pass                                  # already there
            elif cands.get(want, {}).get("valide"):
                choix = want
                motif = f"OVERRIDE of the rule ({rule_reason}): {why}"
                logger.info("Sector %s: override %s -> %s. %s",
                            sect, (rule_choice or "none").upper(), want.upper(), why)
            else:
                logger.warning(
                    "Sector %s: override to %s ignored — the rule found it "
                    "INVALID (coverage %.0f%%, phi %s, R2 %s). An override "
                    "chooses between defensible metrics, it cannot revive a "
                    "rejected one.", sect, want.upper(),
                    cands.get(want, {}).get("couverture", 0) * 100,
                    cands.get(want, {}).get("phi"), cands.get(want, {}).get("r2"),
                )

        sectors[sect] = {"metric": choix, "reason": motif,
                         "rule_metric": rule_choice,
                         "rule_reason": rule_reason,
                         "overridden": bool(choix != rule_choice),
                         "n_firms": int(g["ticker"].nunique()),
                         "candidates": cands,
                         "phi": cands[choix]["phi"] if choix else None,
                         "anchor": cands[choix]["ancre"] if choix else None}

    return {"sectors": sectors,
            "n_firms": int(p["ticker"].nunique()),
            "n_obs": int(len(p)),
            "procedure": {"winsor": list(WINSOR), "couv_min": COUV_MIN,
                          "r2_min": R2_MIN, "phi_bounds": list(PHI_BOUNDS),
                          "r2_tie": R2_TIE, "min_firms": MIN_FIRMS,
                          "min_trans": MIN_TRANS,
                          "equity_risk_premium": EQUITY_RISK_PREMIUM}}


def _quarter(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def save_calibration(calib: dict, as_of: Optional[date] = None,
                     directory: Optional[Path] = None,
                     force: bool = False) -> Optional[Path]:
    """
    Archive a calibration under its quarter, after comparing it to the previous
    one.

    A phi that moves more than JUMP_ALERT between two quarters is not the
    economy changing in three months — it is a data outage, a sector relabelling
    upstream, or an outlier that escaped winsorisation. Every bug this codebase
    has produced was silent, so a jump is reported loudly and, unless forced,
    refuses to be written.
    """
    as_of = as_of or date.today()
    directory = directory or CALIBRATION_DIR
    directory.mkdir(parents=True, exist_ok=True)

    prev_path, prev = _latest(directory, as_of, exclusive=True)
    alertes: List[str] = []
    if prev:
        for sect, cur in calib.get("sectors", {}).items():
            old = prev.get("sectors", {}).get(sect)
            if not old:
                continue
            if old.get("metric") != cur.get("metric"):
                alertes.append(f"{sect}: metric {old.get('metric')} -> {cur.get('metric')}")
            a, b = old.get("phi"), cur.get("phi")
            if a is not None and b is not None and abs(b - a) > JUMP_ALERT:
                alertes.append(f"{sect}: phi {a:.3f} -> {b:.3f}")

    calib = dict(calib)
    calib["as_of"] = as_of.isoformat()
    calib["quarter"] = _quarter(as_of)
    calib["previous"] = prev_path.name if prev_path else None
    calib["alerts"] = alertes

    if alertes and not force:
        logger.error(
            "Calibration NOT written — %d parameter(s) moved more than expected "
            "between quarters, which is a data problem far more often than an "
            "economic one. Review, then re-run with force=True to accept:\n  %s",
            len(alertes), "\n  ".join(alertes),
        )
        return None

    for a in alertes:
        logger.warning("Calibration change accepted under force: %s", a)

    path = directory / f"valuation_{_quarter(as_of)}.json"
    path.write_text(json.dumps(calib, indent=2, ensure_ascii=False))
    n_assigned = sum(1 for s in calib["sectors"].values() if s.get("metric"))
    logger.info("Calibration written to %s — %d/%d sectors assigned a metric.",
                path.name, n_assigned, len(calib["sectors"]))
    return path


def _latest(directory: Path, as_of: date, exclusive: bool = False):
    """Most recent archived calibration not later than as_of."""
    if not directory.exists():
        return None, None
    best, best_d = None, None
    for p in sorted(directory.glob("valuation_*.json")):
        try:
            d = date.fromisoformat(json.loads(p.read_text())["as_of"])
        except Exception:
            continue
        if d > as_of or (exclusive and d == as_of):
            continue
        if best_d is None or d > best_d:
            best, best_d = p, d
    if best is None:
        return None, None
    return best, json.loads(best.read_text())


def load_calibration(as_of=None, directory: Optional[Path] = None) -> Optional[dict]:
    """
    The calibration a rebalance on `as_of` is entitled to see: the most recent
    one NOT LATER than that date.  Point-in-time by construction, so once a
    history of quarterly calibrations exists the parameters can no longer leak
    backwards.
    """
    # Normalised through pd.Timestamp for every input type.  Testing
    # `isinstance(x, date)` to decide whether to convert does NOT work:
    # pd.Timestamp subclasses datetime.datetime which subclasses datetime.date,
    # so a Timestamp passes that test, skips the conversion, and then raises on
    # the comparison in _latest. strategy.run passes a Timestamp, so that path
    # failed on every call.
    if as_of is None:
        as_of = date.today()
    else:
        try:
            as_of = pd.Timestamp(as_of).date()
        except Exception:
            logger.error("Cannot read %r as a date — no calibration loaded.", as_of)
            return None
    path, calib = _latest(directory or CALIBRATION_DIR, as_of)
    if calib is None:
        logger.warning("No calibration dated on or before %s — the valuation leg "
                       "is unavailable for this run.", as_of)
        return None
    logger.info("Calibration %s (produced %s), %d sectors.",
                path.name, calib.get("as_of"), len(calib.get("sectors", {})))
    return calib


# --------------------------------------------------------------------------- #
#  Cost of capital                                                            #
# --------------------------------------------------------------------------- #

def sector_asset_betas(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    fundamentals: pd.DataFrame,
    cfg: TaurusConfig = DEFAULT_CONFIG,
) -> Dict[str, float]:
    """
    Unlevered (asset) beta per sector, by Hamada.

    A beta fitted on one stock's own returns is noisy and carries that stock's
    leverage history. Estimating at SECTOR level pools the observations, and
    de-levering with the sector's median debt-to-equity strips the financial
    risk out, leaving business risk:

        beta_asset = beta_equity / [1 + (1 - tau) x D/E]

    It is re-levered per company later, so a company's own leverage enters its
    cost of equity through its own D/E rather than through a noisy regression.
    """
    from .portfolio import estimate_betas

    betas = estimate_betas(returns, market_returns, cfg)
    if betas is None or len(betas) == 0 or "sector" not in fundamentals.columns:
        return {}

    out: Dict[str, float] = {}
    sect = fundamentals["sector"].reindex(betas.index)
    lev = (fundamentals.get("total_debt") / fundamentals.get("total_equity")
           ).reindex(betas.index) if "total_debt" in fundamentals.columns else None
    tax = (fundamentals.get("effective_tax_rate").reindex(betas.index)
           if "effective_tax_rate" in fundamentals.columns else None)

    for s, idx in sect.groupby(sect).groups.items():
        b = betas.reindex(idx).dropna()
        if len(b) < 3:
            continue
        be = float(b.median())
        de = 0.0
        if lev is not None:
            d = lev.reindex(idx).replace([np.inf, -np.inf], np.nan).dropna()
            d = d[(d >= 0) & (d < 10)]
            if len(d):
                de = float(d.median())
        tau = 0.25
        if tax is not None:
            tv = tax.reindex(idx).dropna()
            if len(tv):
                tau = float(tv.median())
        ba = be / (1.0 + (1.0 - tau) * de) if (1.0 + (1.0 - tau) * de) > 0 else be
        # A negative or absurd asset beta is an estimation failure, not a
        # finding: clip to a range a real business can occupy.
        out[str(s)] = float(np.clip(ba, 0.20, 2.00))
    return out


def _risk_free(factors: Optional[pd.DataFrame], cfg: TaurusConfig) -> float:
    """
    Annual risk-free rate, measured rather than assumed.

    The Fama-French files carry a monthly RF column; its LAST observation is
    the current rate. The mean over the window would be wrong — it spans the
    zero-rate decade (1.38% annualised against 3.72% today).
    """
    if factors is not None and "RF" in getattr(factors, "columns", []):
        rf = factors["RF"].dropna()
        if len(rf):
            return float(rf.iloc[-1]) * 12.0
    return float(getattr(cfg, "risk_free_rate", 0.045))


# --------------------------------------------------------------------------- #
#  Intrinsic value                                                            #
# --------------------------------------------------------------------------- #

def intrinsic_equity_value(row, sector_cfg: dict, rf: float, beta_asset: float,
                           erp: float = EQUITY_RISK_PREMIUM) -> Tuple[float, str]:
    """
    Intrinsic EQUITY value for one company, or (nan, reason it is ineligible).

    Closed form, no horizon to choose:

        V = B + B x (R0 - r) x q / (1 - q),     q = phi / (1 + r)

    ROIC sectors value the FIRM off invested capital at the unlevered cost of
    capital, then subtract net debt. ROE sectors value equity directly off
    common equity at the levered cost of equity.

    A known simplification: the capital base is held CONSTANT, where in reality
    it grows with retained earnings. That understates growing companies. It is
    the standard form for a relative-value signal, which is what this is, but
    it biases the absolute level — and therefore the target price.
    """
    metric = sector_cfg.get("metric")
    if not metric:
        return np.nan, "sector has no usable metric"
    phi = sector_cfg.get("phi")
    if phi is None or not (0.0 < float(phi) < 1.0):
        return np.nan, "no valid persistence for the sector"
    phi = float(phi)

    tax = float(row.get("effective_tax_rate", row.get("tax_rate", 0.25)) or 0.25)
    debt = float(row.get("total_debt", 0.0) or 0.0)
    cash = float(row.get("cash", 0.0) or 0.0)
    net_debt = debt - cash

    if metric == "roic":
        base = invested_capital(row)
        r0 = roic(row)
        if not (base == base and base > 0):
            return np.nan, "invested capital not positive"
        r = rf + beta_asset * erp                     # unlevered cost of capital
    else:
        base = float(row.get("equity_common", row.get("total_equity", np.nan)))
        r0 = roe(row)
        if not (base == base and base > 0):
            return np.nan, "common equity not positive"
        de = debt / base if base > 0 else 0.0
        de = float(np.clip(de, 0.0, 10.0))
        beta_eq = beta_asset * (1.0 + (1.0 - tax) * de)   # Hamada, re-levered
        r = rf + float(np.clip(beta_eq, 0.20, 3.00)) * erp

    if not (r0 == r0):
        return np.nan, f"{metric.upper()} not computable"
    if not (WINSOR[0] <= r0 <= WINSOR[1]):
        # Outside this band the denominator has collapsed; the number is an
        # artefact and projecting it would produce an arbitrary target price.
        return np.nan, f"{metric.upper()} of {r0:.0%} outside the plausible band"
    if r <= 0:
        return np.nan, "non-positive cost of capital"

    q = phi / (1.0 + r)
    if not (0.0 < q < 1.0):
        return np.nan, "divergent series"
    value = base + base * (r0 - r) * q / (1.0 - q)

    if metric == "roic":
        value -= net_debt                             # firm value -> equity value
    if not (value == value):
        return np.nan, "value not computable"
    return float(value), ""


def valuation_screen(
    fundamentals: pd.DataFrame,
    calibration: dict,
    returns: Optional[pd.DataFrame] = None,
    market_returns: Optional[pd.Series] = None,
    factors: Optional[pd.DataFrame] = None,
    cfg: TaurusConfig = DEFAULT_CONFIG,
    erp: float = EQUITY_RISK_PREMIUM,
) -> pd.DataFrame:
    """
    Intrinsic value, target price and divergence per company.

    divergence = (intrinsic equity value - market cap) / market cap, so it is
    signed: positive means the market is below our estimate (a long's target is
    above the price), negative means the opposite (a short's target is below).
    That signedness is the whole point — the retired MM divergence could only
    ever be <= 0.

    A company the model cannot value gets NaN and a stated reason, never a
    filled-in plausible number.
    """
    cols = ["metric", "intrinsic_equity", "divergence_pct", "target_factor",
            "cost_of_capital", "metric_value", "phi", "eligible", "reason"]
    if fundamentals is None or fundamentals.empty or not calibration:
        return pd.DataFrame(columns=cols)

    rf = _risk_free(factors, cfg)
    betas = ({} if (returns is None or market_returns is None)
             else sector_asset_betas(returns, market_returns, fundamentals, cfg))
    if betas:
        logger.info("Asset betas by sector (Hamada de-levered): %s",
                    "  ".join(f"{k}={v:.2f}" for k, v in sorted(betas.items())))
    beta_defaut = float(np.median(list(betas.values()))) if betas else 1.0
    secteurs = calibration.get("sectors", {})

    lignes = []
    for t, row in fundamentals.iterrows():
        sect = str(row.get("sector", "Unknown") or "Unknown")
        scfg = secteurs.get(sect, {})
        ba = betas.get(sect, beta_defaut)
        val, raison = intrinsic_equity_value(row, scfg, rf, ba, erp)
        mc = float(row.get("market_cap", np.nan))
        metric = scfg.get("metric")
        mv = (roic(row) if metric == "roic" else roe(row)) if metric else np.nan
        if metric == "roic":
            r = rf + ba * erp
        elif metric == "roe":
            base = float(row.get("equity_common", row.get("total_equity", np.nan)))
            tax = float(row.get("effective_tax_rate", 0.25) or 0.25)
            de = float(np.clip((row.get("total_debt", 0.0) or 0.0) / base, 0, 10)) \
                if (base == base and base > 0) else 0.0
            r = rf + float(np.clip(ba * (1 + (1 - tax) * de), 0.20, 3.00)) * erp
        else:
            r = np.nan

        if val == val and mc == mc and mc > 0:
            div = (val - mc) / mc
            lignes.append({"metric": metric, "intrinsic_equity": val,
                           "divergence_pct": div * 100.0,
                           "target_factor": val / mc, "cost_of_capital": r,
                           "metric_value": mv, "phi": scfg.get("phi"),
                           "eligible": True, "reason": ""})
        else:
            if val == val and not (mc == mc and mc > 0):
                raison = raison or "no market cap to compare against"
            lignes.append({"metric": metric, "intrinsic_equity": np.nan,
                           "divergence_pct": np.nan, "target_factor": np.nan,
                           "cost_of_capital": r, "metric_value": mv,
                           "phi": scfg.get("phi"), "eligible": False,
                           "reason": raison or "not valued"})

    out = pd.DataFrame(lignes, index=fundamentals.index)[cols]
    n_ok = int(out["eligible"].sum())
    logger.info("Valuation: %d/%d companies valued (rf %.2f%%, ERP %.1f%%).",
                n_ok, len(out), rf * 100, erp * 100)
    if n_ok < len(out):
        motifs = out.loc[~out["eligible"], "reason"].value_counts()
        for m, n in motifs.head(6).items():
            logger.info("  not valued — %s: %d", m, n)
    return out
