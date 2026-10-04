# Taurus Strategy — Commandes

## ⭐ LA COMMANDE — rebalancement mensuel

Deux lignes, **identiques à chaque fois**, depuis la racine du projet (PowerShell) :

```powershell
git pull
```

```powershell
python main.py --mode force-rebalance --universes sp500 nasdaq100 cac40 ftse100 nikkei225 --no-dry-run | Tee-Object -FilePath "run_$(Get-Date -Format yyyy-MM-dd).log"
```

C'est tout. Rien d'autre à lancer, aucun scheduler à laisser tourner.

### Ce que fait cette commande

1. Enregistre la performance réalisée du mois écoulé pour les 5 univers
   → ajoutée à l'historique, **jamais écrasée**
2. Recalcule la pondération Sharpe sur **tout** l'historique depuis l'inception, ce nouveau mois inclus
3. Construit les portefeuilles : FF5 régional → screen MM → momentum → max-Sharpe → neutralisation bêta
4. Place les ordres au levier **1.25** (exposition brute 125 % du NAV)
5. Archive les positions dans `output/{univers}/snapshots/` (archive permanente datée)
6. Sauvegarde l'intégralité du log dans `run_AAAA-MM-JJ.log`

### Quand la lancer

**N'importe quel jour.** L'algo détermine seul le mois à traiter :

| Jour du lancement | Mois enregistré |
|---|---|
| du 1er au 5 | le mois **précédent** |
| du 6 à la fin | le mois **en cours** |

Lancer début de mois reste préférable : les signaux portent alors sur un mois complet.

### Les 3 lignes à vérifier dans le log

```
NAV allocation (global weights, 5 active universes): SP500=33.8%  NIKKEI225=26.7%  ...
[sp500] Monthly return appended: 2026-09-30 → +X.XX%
Beta neutralisation: β_L=..., β_S=... → net β=0.0000
```

⚠️ Si l'allocation affiche **20 % partout**, la pondération Sharpe n'a pas trouvé l'historique
(`output/{univers}/monthly_returns.csv`) et est retombée sur l'équipondération de secours.

### Points de vigilance

| | |
|---|---|
| **`--universes` est obligatoire** | sans lui, le défaut est `sp500` seul — les 4 autres univers resteraient figés |
| **Pas de `--live`** | son absence = port 7497 = **compte démo**. L'ajouter bascule sur le compte réel (port 7496) |
| **`--no-dry-run`** | sans lui, l'algo simule et ne passe aucun ordre |
| **TWS / IB Gateway ouvert** | API activée, port 7497 |

### Vérification dans IBKR après le lancement

- **Exposition brute** ≈ 125 % du NAV
- **Exposition nette** ≈ 0 % (bêta-neutre)
- **Excess Liquidity** largement positif (marge Reg-T requise : 62,5 % du NAV)

### Univers tradés

`sp500` · `nasdaq100` · `cac40` · `ftse100` · `nikkei225`

`hangseng` et `tadawul` sont déclarés dans le registre mais **jamais tradés** : aucun historique,
donc Sharpe = 0 et allocation 0 % pendant au moins 6 mois. Ne pas les ajouter sans backtest préalable.

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

## Checklist mensuelle

| Étape | Commande / Action |
|-------|-------------------|
| 1. Ouvrir TWS | Paper trading → vérifier la connexion (port 7497) |
| 2. Mettre à jour | `git pull` |
| 3. **Rebalancement** | `python main.py --mode force-rebalance --universes sp500 nasdaq100 cac40 ftse100 nikkei225 --no-dry-run \| Tee-Object -FilePath "run_$(Get-Date -Format yyyy-MM-dd).log"` |
| 4. Contrôler le log | allocation Sharpe ≠ 20 % partout · `Monthly return appended` présent · `net β=0.0000` |
| 5. Vérifier TWS | ordres MKT dans le blotter · brut ≈ 125 % · net ≈ 0 % |
| 6. Corriger STP/LMT | `python main.py --mode refresh-protective --universes sp500 nasdaq100 cac40 ftse100 nikkei225 --no-dry-run` |
| 7. Rapport | `python main.py --mode report --universes sp500 nasdaq100 cac40 ftse100 nikkei225` |
| 8. Ouvrir | `output/report.png` |

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
| `optimizer_method` | `max_sharpe` | Optimiseur : tangency portfolio (max-Sharpe) |
| `signal_method` | `binary` | Signal : AND-filter (alpha AND MM AND momentum) |
| `use_umd_factor` | `False` | FF5 uniquement (pas de facteur UMD) |
| `vol_adjust_momentum` | `True` | Momentum ajusté par la volatilité (Sharpe-momentum) |
| `cov_halflife` | 0 | Covariance Ledoit-Wolf (0 = pas d'EWMA) |
| `gross_leverage` | **1.25** | Exposition brute = 125 % du NAV (chaque jambe à 62,5 %) |
| `sharpe_window_months` | **0** | Pondération Sharpe **depuis l'inception** (0 = fenêtre expansive) |
| `sharpe_min_months` | 6 | En dessous → équipondération de secours |
| `max_universe_weight` | 0.35 | Plafond par univers (garde-fou, ne mord pas aujourd'hui) |

### Pondération des univers

Les poids sont proportionnels à `max(Sharpe, 0)`, calculés sur **tout** l'historique
stocké dans `output/{univers}/monthly_returns.csv` — backtest 10 ans + chaque mois live
ajouté depuis. Le mois écoulé est enregistré **avant** le calcul des poids, donc il compte
dès le rebalancement en cours.

Plus l'historique s'allonge, plus l'allocation devient stable : un mois de plus sur une
base de 126 ne déplace les poids qu'à la marge.

### Facteurs Fama-French par univers

| Univers | Dataset Kenneth French |
|---|---|
| sp500 · nasdaq100 | `F-F_Research_Data_5_Factors_2x3` (US) |
| cac40 · ftse100 | `Europe_5_Factors` |
| nikkei225 | `Japan_5_Factors` |
| hangseng | `Asia_Pacific_ex_Japan_5_Factors` |
| tadawul | `Global_5_Factors` |

Le routage est automatique selon la région déclarée dans `UniverseConfig`.
