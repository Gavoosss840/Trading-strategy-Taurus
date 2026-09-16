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

## Valorisation Modigliani-Miller — changement de comportement

L'écran MM calculait la valeur de la firme non endettée à partir de la
capitalisation boursière :

```
VU = capitalisation + dette nette − bouclier fiscal
VL = VU + bouclier fiscal − détresse − agence
```

Le bouclier fiscal s'annulant entre les deux lignes, il restait
`VL_equity = capitalisation − détresse − agence`, donc une divergence égale à
`−(détresse + agence) / capitalisation` : **toujours négative ou nulle**. Le
signal d'achat `divergence > +25 %` était inatteignable.

Conséquences, désormais corrigées :

| Étage | Avant | Après |
|---|---|---|
| `underleveraged` | jamais vrai | atteignable |
| `_binary_signal`, jambe longue | repli sur le quantile d'alpha à chaque rebalancement | branche MM utilisée |
| `_composite_signal`, `z_mm` | mesure surtout le risque de faillite | mesure la valorisation |
| Take-profit (`execution.py`) | divergence ≈ 0 → plancher de 5 % partout | niveau propre à chaque titre, 5 % à 80 % |

La valeur non endettée est maintenant actualisée depuis les fondamentaux
(NOPAT en perpétuité, bêta dé-leviérisé issu de la régression FF5). Les trois
frottements — bouclier fiscal, détresse de Merton, agence — sont inchangés.

> **À faire avant le prochain rebalancement réel** : relancer un back-test.
> La composition de la jambe longue et les niveaux de take-profit changent
> nettement ; les performances historiques publiées ont été mesurées sous
> l'ancienne formulation.

Une société au résultat d'exploitation négatif ou nul n'est pas valorisable par
perpétuité : sa divergence vaut `NaN` (et non 0, qui se lirait « au juste
prix »). Elle n'est retenue dans aucune des deux jambes au titre de la
valorisation, reste notée neutre sur ce pilier dans le mode composite, et les
deux garde-fous de solvabilité — couverture des intérêts sous 1,5× et EBIT
négatif avec dette — continuent de s'appliquer.

---

## Lecture des comptes EDGAR — correction de période

La lecture XBRL prenait, pour chaque agrégat, **le fait le plus récent quelle
que soit la période qu'il couvre**. Un dépôt trimestriel étant plus récent que
le dernier exercice, le chiffre retenu était souvent celui d'un trimestre — et
l'écran MM le capitalisait ensuite à l'infini comme s'il s'agissait d'une
année.

| Titre | Chiffre d'affaires lu (avant) | Réel sur 12 mois |
|---|---|---|
| Coca-Cola | 12,5 Md$ | 47,9 Md$ |
| Johnson & Johnson | 18,5 Md$ | 94,2 Md$ |
| Apple | 364 Md$ (exercice 2023) | 416 Md$ |

Le flux était donc divisé par quatre pour une partie des titres — **lesquels
dépendait du concept XBRL le plus frais**, c'est-à-dire du hasard des dépôts,
pas de l'économie de l'entreprise. Coca-Cola ressortait « surévalué de 82 % »
et passait en VENTE pour cette seule raison.

Trois corrections :

- **flux sur douze mois glissants** : somme de quatre trimestres consécutifs
  et disjoints, à défaut le dernier exercice, à défaut le cumulé annualisé ;
- **bilan lu sur les faits instantanés** uniquement (une valeur de bilan n'a
  pas de période) ;
- **choix du concept par fraîcheur** et non par ordre de priorité : Microsoft
  n'alimente plus `Revenues` depuis 2010, Johnson & Johnson depuis 2014 ; la
  priorité seule renvoyait un chiffre vieux de dix ans.

Le nombre d'actions, cherché en dollars, revenait `NaN` pour tout le monde :
il est désormais lu dans l'unité `shares`.

Mesure sur 20 grandes capitalisations américaines (EDGAR réel, bêtas FF5 réels,
capitalisations réelles) :

| | Avant | Après |
|---|---|---|
| Verdicts VENTE | 15 / 20 | 10 / 20 |
| Verdicts ACHAT | **0 / 20** | 4 / 20 |
| Verdicts modifiés | — | **8 / 20** |

Zéro achat sur vingt signifiait que `_binary_signal` restait sous son seuil
`MM_MIN_RATIO = 0,10` et **ignorait purement et simplement le pilier MM** pour
la jambe longue, à chaque rebalancement.

---

## Valorisation en deux étapes — le biais anti-croissance

Le NOPAT était capitalisé en perpétuité à un taux unique de 2,5 % pour
**toutes** les sociétés. Une entreprise qui croît à 15 % par an était donc
valorisée comme si elle croissait à 2,5 % — et ressortait mécaniquement
surévaluée.

Mesure sur les mêmes 20 titres : croissance du chiffre d'affaires et divergence
corrélées à **−0,49**, 100 % des titres au-dessus de 8 % de croissance classés
en VENTE, et la jambe longue peuplée des télécoms en déclin (T à −5,7 % de
croissance annuelle). L'algorithme vendait les sociétés en croissance et
achetait celles qui reculent — un pari que personne n'avait choisi.

La valorisation se fait désormais en deux étapes :

1. **dix exercices** dont la croissance décroît linéairement de la croissance
   propre de l'entreprise vers le taux terminal ;
2. **perpétuité** au taux terminal (2,5 %).

La croissance de départ est estimée sur **huit exercices de chiffre
d'affaires** (le chiffre d'affaires plutôt que le résultat : ses marges
fluctuent moins, et une marge exceptionnelle ne se confond pas avec une
trajectoire). Les extrémités sont lissées sur deux ans. Elle est bornée à
±15 % / −5 %, mais **pas** par le taux d'actualisation : la contrainte `g < r`
ne s'applique qu'à la perpétuité terminale.

Sans historique exploitable, le repli est `default_initial_growth = 3 %` — la
valorisation reste calculable, jamais estimée sur deux points.

> **Ce qui subsiste, et qui est assumé** : après correction, la corrélation
> croissance / divergence reste négative (−0,55 sur les 20 titres). C'est le
> biais *value* d'un modèle d'actualisation : le marché price les sociétés en
> forte croissance à des multiples que dix ans de fondu à 15 % ne rattrapent
> pas. Ce n'est plus un artefact de lecture des données, c'est un choix de
> méthode — il se corrige en relevant `max_initial_growth`, ou en neutralisant
> `z_mm` par secteur.

> **À faire avant le prochain rebalancement réel** : relancer un back-test.
> Ces deux corrections changent la composition des deux jambes.

---

## Le secteur n'était jamais renseigné

`sec_edgar.get_fundamentals` renvoyait `sector = "Unknown"` pour **toutes** les
sociétés, et yfinance ne complétait que les titres dont la capitalisation
manquait. Quatre mécanismes tournaient donc à vide :

| Mécanisme | Effet du secteur manquant |
|---|---|
| Taux de détresse MM (`_SECTOR_DISTRESS_RATE`) | 20 % forfaitaire pour tout le monde |
| Plafond sectoriel (`max_sector_weight` = 30 %) | un seul secteur observé ⇒ plafond relevé à 100 %, **inerte** |
| Neutralisation sectorielle du pilier valorisation | impossible |
| Exemption des financières | impossible |

Le secteur est désormais lu depuis le **code SIC** du dépôt SEC
(`data.sec.gov/submissions/CIK…`) et traduit en nomenclature GICS par
`taurus/sectors.py`. Vérifié : JPM → Financials, KO → Consumer Staples,
NVDA → Information Technology, T → Communication Services, PLD → Real Estate.

---

## Neutralisation sectorielle du pilier valorisation

Le biais de niveau d'un modèle d'actualisation est presque entièrement
sectoriel : le marché paie des multiples élevés pour le logiciel et bas pour
les télécoms. Classer un titre contre l'univers entier **achète donc des
secteurs, pas des sociétés** — et c'est ce qui produisait la corrélation
négative entre croissance et divergence.

`z_mm` est maintenant calculé **au sein de chaque secteur** (`mm_sector_neutral`,
actif par défaut). Un secteur de moins de `mm_sector_min_members = 4` titres est
classé contre l'univers : une médiane sur deux titres est du bruit.

Mesure sur 30 grandes capi US, données réelles :

| | Classement absolu | Neutralisé par secteur |
|---|---|---|
| Corrélation croissance / `z_mm` | −0,49 | **−0,10** |
| Côté achat | VZ, TMUS, COP, MRK, T, PEP | MRK, ADBE, COP, META, MSFT, JNJ |
| Côté vente | AVGO, NVDA, AAPL, ORCL, AMZN, GOOGL | T, LLY, AVGO, NVDA, KO, AMZN |

AT&T passe du côté achat au côté vente : dans l'absolu son cours paraît bas,
mais **face aux autres télécoms** c'est la moins bonne. C'est exactement
l'information que le pilier est censé porter.

> ⚠️ `z_mm` n'est utilisé que par `_composite_signal`. Avec
> `signal_method = "binary"` — la valeur par défaut, donc votre mode actif —
> le pilier MM passe par les seuils absolus `divergence > ±25 %` et la
> neutralisation n'a **aucun effet**. Pour en bénéficier, passer
> `signal_method = "composite"` dans `config.py`.

---

## Financières : l'écran MM s'abstient

Le garde-fou de couverture des intérêts (`EBIT / intérêts < 1,5×`) classait
**toutes les banques** en VENTE. Pour une banque les intérêts versés ne sont pas
une charge de financement mais le coût de la matière première : elle se finance
par les dépôts et prête le produit. Ratios mesurés : JPM 0,89× · BAC 0,51× ·
GS 0,33× · WFC 0,77×.

Pire, la valorisation mettait JPM 47 % sous son prix : le titre était
simultanément `underleveraged = True` **et** `overleveraged = True`, donc
candidat aux **deux jambes en même temps**.

Trois corrections :

- **les financières sont exemptées du garde-fou de couverture** ; les foncières
  (Real Estate) le conservent, leur dette est un vrai levier ;
- **l'écran MM ne les valorise plus du tout** (`mm_skip_financials`, actif par
  défaut) : l'APV sépare des actifs d'exploitation d'un choix de financement,
  or pour une banque le levier **est** l'activité — il n'y a pas de firme non
  endettée à valoriser, et l'« EBIT » n'est pas un flux capitalisable. Le pilier
  s'abstient, comme il le fait déjà pour un EBIT négatif. Les financières
  restent négociables via l'alpha et le momentum ;
- **un titre ne peut plus être candidat aux deux jambes** : en cas de conflit,
  les garde-fous de solvabilité l'emportent sur une valorisation attractive.

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
| `equity_risk_premium` | 5,0% | Prime de risque actions — valorisation MM (APV) |
| `terminal_growth` | 2,5% | Croissance du NOPAT après fondu — perpétuité terminale |
| `explicit_growth_years` | 10 | Durée du fondu entre croissance propre et taux terminal |
| `max_initial_growth` | 15% | Plafond de la croissance de départ |
| `min_initial_growth` | −5% | Plancher (activité en déclin) |
| `default_initial_growth` | 3% | Repli quand l'historique de CA est absent |
| `min_discount_spread` | 2,0% | Écart plancher entre r_U et g (perpétuité finie) |
| `optimizer_method` | `max_sharpe` | Optimiseur : tangency portfolio (max-Sharpe) |
| `signal_method` | `binary` | Signal : AND-filter (alpha AND MM AND momentum) — `composite` pour activer `z_mm` |
| `mm_sector_neutral` | `True` | `z_mm` calculé au sein du secteur (mode composite) |
| `mm_sector_min_members` | 4 | En deçà, le secteur est classé contre l'univers |
| `mm_skip_financials` | `True` | L'écran MM s'abstient sur les financières |
| `use_umd_factor` | `False` | FF5 uniquement (pas de facteur UMD) |
| `vol_adjust_momentum` | `True` | Momentum ajusté par la volatilité (Sharpe-momentum) |
| `cov_halflife` | 0 | Covariance Ledoit-Wolf (0 = pas d'EWMA) |
