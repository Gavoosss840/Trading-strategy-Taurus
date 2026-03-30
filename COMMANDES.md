# Taurus Strategy — Commandes

## Backtest

```bash
python main.py --mode backtest --start 2022-01-01 --end 2026-03-31 --universes sp500 nasdaq100 ftse100 cac40 nikkei225
```
> Lance un backtest complet sur tous les univers et génère les rapports dans `output/`

---

## Live (rebalance mensuel)

```bash
python main.py --mode live --universes sp500 nasdaq100 ftse100 cac40 nikkei225
```
> À lancer le **30 ou 31 du mois**. Se connecte au TWS (paper par défaut, port 7497).  
> L'algo tourne toute la nuit et exécute les ordres à la clôture US (~22h30 CET).  
> Les stops natifs IBKR sont posés automatiquement → tu peux éteindre le PC le 1er du mois.

---

## Rapport live (à tout moment)

```bash
python main.py --mode report --universes sp500 nasdaq100 ftse100 cac40 nikkei225
```
> Génère `output/report.png` en ~3 secondes depuis les données déjà stockées.  
> Aucune connexion IBKR ni téléchargement nécessaire.

---

## Git

```bash
# Récupérer les dernières modifs
git pull origin claude/optimize-python-algorithm-dWMxq

# Pousser tes modifs
git add -A && git commit -m "ton message" && git push -u origin claude/optimize-python-algorithm-dWMxq
```

---

## Checklist mensuelle (J-1 = 30 ou 31 du mois)

| Heure | Action |
|-------|--------|
| Matin | Ouvrir TWS (paper) → vérifier connexion |
| Matin | `python main.py --mode live --universes sp500 nasdaq100 ftse100 cac40 nikkei225` |
| 22h30 | Vérifier les ordres dans TWS (blotter) |
| 1er du mois | `python main.py --mode report --universes sp500 nasdaq100 ftse100 cac40 nikkei225` |
| 1er du mois | Vérifier `output/report.png` → fermer TWS → éteindre PC |

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

# Tester yfinance (données fondamentaux)
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info.get('marketCap', 'N/A'))"
```

---

## Paramètres clés (config.py)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `ibkr_port` | 7497 | Paper trading (live = 7496) |
| `trailing_stop_pct` | 10% | Stop trailing natif IBKR |
| `circuit_breaker_pct` | 15% | Coupe tout si -15% portefeuille |
| `stop_loss_pct` | 10% | Stop-loss par position |
| `lookback_months` | 60 | Fenêtre régression FF5 |
| `n_longs` | 10 | Nombre de positions long |
| `n_shorts` | 10 | Nombre de positions short |

---

## Fichiers de sortie (`output/`)

| Fichier | Description |
|---------|-------------|
| `report.png` | Rapport complet 1 page |
| `equity_curve.png` | Courbe de performance |
| `monthly_heatmap.png` | Heatmap des returns mensuels |
| `rolling_sharpe.png` | Sharpe ratio glissant 12 mois |
| `positions.csv` | Positions actuelles tous univers |
| `combined_analytics.json` | Stats détaillées JSON |
| `{univers}/monthly_returns.csv` | Returns mensuels par univers |
| `execution_{univers}_{date}.json` | Rapport d'exécution des ordres |
