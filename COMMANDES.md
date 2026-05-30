# Taurus Strategy — Commandes

## Rebalancement mensuel (commande principale)

```bash
python main.py --mode force-rebalance --universes sp500 nasdaq100 ftse100 cac40 nikkei225 --no-dry-run
```

> **À lancer le 1er du mois** (ex: 1er juin pour les signaux de mai).
> Lance une fois, place les ordres, s'arrête automatiquement.
> Connexion paper par défaut (port 7497). Ajouter `--live` pour le compte réel (port 7496).

**Pourquoi le 1er du mois ?**
Le calcul des signaux utilise `as_of` = dernier jour ouvrable du mois précédent complété.
- Lancé le **1er juin** → `as_of = 29 mai` (signaux MAI complets) ✓
- Lancé le **31 mai** → `as_of = 30 avril` (signaux AVRIL — même que le mois dernier) ✗

---

## Correction des ordres STP/LMT après rebalancement

```bash
# Prévisualisation (dry-run)
python main.py --mode refresh-protective \
  --universes sp500 nasdaq100 ftse100 cac40 nikkei225

# Application
python main.py --mode refresh-protective \
  --universes sp500 nasdaq100 ftse100 cac40 nikkei225 \
  --no-dry-run
```

> Recalcule les niveaux stop-loss et take-profit depuis le `avg_cost` IBKR réel.
> Ne touche **jamais** aux ordres MKT (entrées déjà placées).
> À lancer si des STP/LMT semblent mal positionnés dans TWS.

---

## Rapport de performance

```bash
python main.py --mode report \
  --universes sp500 nasdaq100 ftse100 cac40 nikkei225
```

> Génère tous les charts en ~3 secondes depuis les données stockées.
> Fusionne automatiquement backtest + données live si `output/live/nav_history.csv` existe.
> Aucune connexion IBKR ni téléchargement nécessaire.

**Fichiers générés dans `output/` :**

| Fichier | Description |
|---------|-------------|
| `report.png` | Rapport complet 1 page (table + equity curve + heatmap) |
| `equity_curve.png` | Courbe de performance avec ligne de transition backtest→live |
| `monthly_heatmap.png` | Heatmap des returns mensuels (backtest + cases live) |
| `rolling_sharpe.png` | Sharpe ratio glissant 12 mois |
| `positions.csv` | Positions actuelles tous univers |
| `combined_analytics.json` | Stats détaillées JSON |
| `{univers}/monthly_returns.csv` | Returns mensuels par univers |
| `live/nav_history.csv` | Historique NAV live (1 ligne par rebalancement) |
| `execution_{univers}_{date}.json` | Rapport d'exécution des ordres |

---

## Backtest complet

```bash
python main.py --mode backtest \
  --start 2020-01-01 --end 2026-05-29 \
  --universes sp500 nasdaq100 ftse100 cac40 nikkei225
```

> Durée : ~10-15 min (téléchargement EDGAR + yfinance + calcul).
> Génère les `monthly_returns.csv` nécessaires pour `--mode report`.

---

## Vérifier les ordres de protection en cours

```bash
python main.py --mode check-protective \
  --universes sp500 nasdaq100 ftse100 cac40 nikkei225
```

> Liste tous les STP/LMT actifs et leur statut sans rien modifier.

---

## Checklist mensuelle (1er du mois)

| Étape | Commande / Action |
|-------|-------------------|
| 1. Ouvrir TWS | Paper trading → vérifier connexion (port 7497) |
| 2. Git pull | `git pull origin claude/optimize-python-algorithm-dWMxq` |
| 3. Rebalancement | `python main.py --mode force-rebalance --universes sp500 nasdaq100 ftse100 cac40 nikkei225 --no-dry-run` |
| 4. Vérifier TWS | Contrôler les ordres MKT dans le blotter |
| 5. Corriger STP/LMT | `python main.py --mode refresh-protective --universes sp500 nasdaq100 ftse100 cac40 nikkei225 --no-dry-run` |
| 6. Rapport | `python main.py --mode report --universes sp500 nasdaq100 ftse100 cac40 nikkei225` |
| 7. Ouvrir | `output/report.png` |

---

## Ajouter une NAV manuelle dans l'historique live

Si une NAV mensuelle est manquante (ex: mois de démarrage), l'ajouter dans
`output/live/nav_history.csv` :

```csv
date,nav,currency,note,recorded_at
2026-03-31,100000.00,USD,manual-baseline,2026-04-01T09:00:00
2026-04-30,XXXXX.XX,USD,force-rebalance,2026-04-30T11:...
```

> La valeur NAV du 31 mars se trouve dans TWS → Account → Reports → Account Statement.

---

## Vérifications avant lancement

```bash
# Tester la connexion IBKR + prix live (ex: LVMH)
python -c "
from ib_insync import IB, Stock
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=99)
contract = Stock('MC.PA', 'EURONEXT', 'EUR')
ib.qualifyContracts(contract)
ticker = ib.reqMktData(contract)
ib.sleep(2)
print('LVMH last price:', ticker.last)
ib.disconnect()
"

# Tester yfinance
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info.get('marketCap', 'N/A'))"
```

---

## Git

```bash
# Récupérer les dernières mises à jour
git pull origin claude/optimize-python-algorithm-dWMxq

# Pousser ses modifications
git add -A && git commit -m "message" && git push -u origin claude/optimize-python-algorithm-dWMxq
```

---

## Modes disponibles (référence complète)

| Mode | Usage | Boucle infinie |
|------|-------|:--------------:|
| `force-rebalance` | **Rebalancement mensuel manuel** | Non — s'arrête seul |
| `refresh-protective` | Corriger STP/LMT après rebalancement | Non — s'arrête seul |
| `check-protective` | Inspecter les ordres de protection | Non — s'arrête seul |
| `report` | Générer les charts (backtest + live) | Non — s'arrête seul |
| `live-report` | Rapport live uniquement (NAV history) | Non — s'arrête seul |
| `backtest` | Backtest complet (données historiques) | Non — s'arrête seul |
| `snapshot` | Snapshot du portefeuille actuel | Non — s'arrête seul |
| `live` | Daemon 24/7 automatique (pas utile en manuel) | **Oui — Ctrl+C pour stop** |

---

## Paramètres clés (config.py)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `ibkr_port` | 7497 | Paper trading (live = 7496) |
| `stop_loss_pct` | 10% | Stop-loss par position (vol-ajusté si `vol_adjusted_stops=True`) |
| `take_profit_pct` | 20% | Take-profit par position (ou divergence MM si disponible) |
| `trailing_stop_pct` | 10% | Trailing stop depuis le pic |
| `circuit_breaker_pct` | 15% | Coupe tout si portefeuille -15% depuis dernier rebalancement |
| `lookback_months` | 60 | Fenêtre régression FF5/FF6 |
| `optimizer_method` | `min_variance` | Optimiseur : min-variance + alpha tilt |
| `signal_method` | `composite` | Signal : z-score composite (alpha + MM + momentum) |
| `use_umd_factor` | `True` | FF6 : ajoute le facteur momentum dans la régression |
| `vol_adjust_momentum` | `True` | Momentum ajusté par la volatilité (Sharpe-momentum) |
| `cov_halflife` | 36 | Demi-vie EWMA covariance (mois) |
