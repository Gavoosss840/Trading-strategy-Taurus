# Taurus Strategy — Commandes

> Document de référence du rebalancement mensuel.
> Mis à jour pour l'**exécution manuelle** : l'algo produit une feuille d'ordres,
> vous passez les ordres. Plus prudent que l'automatique, et c'est un filet que
> le mode automatique n'a pas.

---

## ⭐ LE REBALANCEMENT MENSUEL — 3 étapes

TWS ou IB Gateway doit tourner, API activée.

### 1. Mettre à jour le code

```powershell
git pull
```

### 2. Produire la feuille d'ordres

```powershell
python main.py --mode orders --nav 1000000 --from-ibkr --universes sp500 nasdaq100 nikkei225 stoxx_eur stoxx_uk | Tee-Object -FilePath "orders_$(Get-Date -Format yyyy-MM-dd).log"
```

→ écrit `output/ordres_AAAA-MM-JJ.xlsx`

`--nav` est la valeur de votre compte, dans la devise de `--base-currency`
(EUR par défaut). `--from-ibkr` lit vos positions réelles : **lecture seule**,
aucun ordre n'est passé.

### 3. Lire la feuille, puis passer les ordres dans TWS

Une feuille par univers, plus une feuille `Toutes positions` et un `Resume`.

---

## La feuille d'ordres, colonne par colonne

| colonne | à quoi elle sert |
|---|---|
| `Ticker` · `Societe` | le ticker Yahoo et le **vrai nom** de la société |
| `Symbole IBKR` · `Place` | ce que vous cherchez dans TWS, et sur quelle place |
| `Sens` | LONG ou SHORT |
| **`Action`** | **NOUVELLE · RENFORCER · REDUIRE · INCHANGEE · RETOURNEMENT · SOLDER** |
| **`A passer`** | **le nombre exact de titres à acheter (+) ou vendre (−)** |
| `Ordre` | ACHAT ou VENTE |
| `Titres actuels` · `Titres cibles` | l'existant et la cible, signés |
| `Prix entree` · `Prix en EUR` | prix de marché, devise locale puis devise de base |
| `Stop ordre` · `Stop ordre %` | **le stop à déposer** (−10 % de l'entrée) |
| `Stop moniteur` | le niveau qu'utiliserait le moniteur automatique — **information**, pas un ordre |
| `Prix cible` · `Potentiel %` | prix cible de la valorisation, **souvent vide** (voir plus bas) |
| `Sans prix cible` | pourquoi il n'y a pas de cible sur cette ligne |
| `Alerte` | ce qui demande votre attention |

### Les six actions

- **NOUVELLE** — rien de détenu, acheter la cible entière
- **RENFORCER** / **REDUIRE** — même sens, ajuster de `A passer`
- **INCHANGEE** — à moins d'un titre près, aucun ordre
- **RETOURNEMENT** — le sens s'inverse : l'ordre passe par zéro et est **plus
  gros que chacune des deux positions**. Signalé à part exprès
- **SOLDER** — détenu mais sorti du signal : **fermer entièrement**

### Le `Resume`

Donne par univers le poids Sharpe, la NAV allouée, le brut déployé et le
**cash conservé**. Un cash positif est **normal** : quand une contrainte mord
(plafond de 8 % par position, plafond sectoriel, candidats insuffisants), le
capital non déployé reste en cash plutôt que d'être réparti de force.

---

## Avant la première exécution : solder le CAC 40 et le FTSE 100

Ces deux univers ont été retirés — ils rapportaient **moins que le cash** sur
dix ans (Sharpe −0,44 et −0,23 contre un taux sans risque de 4,5 %). Leurs
positions ne se ferment pas d'elles-mêmes.

```powershell
# 1. lire le plan, sans rien envoyer
python main.py --mode flatten --universes cac40 ftse100

# 2. exécuter
python main.py --mode flatten --universes cac40 ftse100 --no-dry-run --live
```

> **À faire AVANT que `stoxx_eur` ne trade.** Les deux univers partagent des
> noms (AI.PA, BNP.PA, AIR.PA…) et seule l'attribution dans `RiskState`
> permet de les distinguer. Après, l'ambiguïté s'installe.

Le mode épargne les lignes attribuées à un autre univers et signale celles
qu'il n'attribue à personne sans les fermer (`--include-untracked` pour les
inclure).

---

## Recalibration trimestrielle

```powershell
python main.py --mode calibrate --universes sp500 nasdaq100 nikkei225 stoxx_eur stoxx_uk
```

→ écrit `taurus/calibration/valuation_AAAATQ.json`

Ré-estime les paramètres de la jambe de valorisation (persistance du ROIC/ROE
par secteur) sur les états financiers frais. **La procédure est figée dans le
code, seuls les nombres bougent.**

Chaque calibration est **datée** : un rebalancement n'utilise que la plus
récente antérieure ou égale à sa date, donc les paramètres ne peuvent pas
fuiter en arrière. Un paramètre qui bouge de plus de 0,25 en un trimestre
**refuse de s'écrire** et affiche pourquoi — c'est presque toujours une panne
de données, pas l'économie. `--force-calibration` pour accepter.

---

## Les univers

`sp500` · `nasdaq100` · `nikkei225` · `stoxx_eur` · `stoxx_uk`

| univers | valeurs | positions/jambe | ce qui limite |
|---|---|---|---|
| sp500 | 495 | 35 | la cible |
| stoxx_eur | 289 | 35 | la cible |
| nikkei225 | 129 | 32 | la taille |
| stoxx_uk | 127 | 31 | la taille |
| nasdaq100 | 95 | 23 | la taille |

Le nombre de positions vient d'**une seule règle** :
`min(n_target, taille × max_universe_fraction / 2)` = `min(35, taille × 0,25)`.
Un univers ne diffère que par sa **largeur**, pas par un modèle propre.

Disponible mais non activé : `stoxx_europe` (416 valeurs, EUR + Londres en un
seul book multi-devises). **Ne pas le lancer en même temps que `stoxx_eur` /
`stoxx_uk`** — mêmes noms, les books se disputeraient les lignes.

Conservés dans le registre pour pouvoir être soldés : `cac40`, `ftse100`.

---

## Ce que fait le signal

1. **Alpha FF5 régional** — régression 5 facteurs, t-stat de l'intercept (HC1)
2. **Momentum** — 12 mois ajusté de la volatilité, saut du dernier mois
3. **Véto crédit** — retire un candidat **long** sur : couverture d'intérêts
   < 1,5 · perte d'exploitation avec dette · probabilité de défaut de Merton
   > 20 %. Les prêteurs (banques, assureurs, foncières) sont exemptés des deux
   premiers : les intérêts sont leur matière première
4. **Dimensionnement max-Sharpe**, plafond absolu de 8 % par position
5. **Neutralisation bêta par réduction** — la jambe la plus exposée est
   réduite, jamais l'autre gonflée

Bêta réalisé mesuré sur le book combiné : **−0,029** (t = −0,67), R² marché
0,004.

### La jambe de valorisation existe mais est désactivée

Un modèle de résultat résiduel (Ohlson/Penman) est construit et calibré, avec
une métrique par secteur choisie sur les données (ROIC ou ROE selon la
persistance mesurée). **Il n'entre ni dans la sélection ni dans le
take-profit**, pour deux raisons mesurées :

- **en sélection** il coûte **−0,586 de Sharpe**, dégradation sur les cinq
  univers. Il inverse le biais value/croissance du book, et la croissance a
  dominé la décennie
- **en prix cible** il est inutilisable : ancré sur les fonds propres
  comptables, il ne peut pas atteindre les multiples que le marché paie pour
  les sociétés à forte composante immatérielle (plafond ~1,2-1,6× le book
  contre une médiane de 1,87× sur le CAC 40, 8,6× pour Safran). Les cibles
  sortaient à −78 % sous l'entrée sur les longs

C'est pourquoi la colonne `Prix cible` est souvent vide, avec le motif indiqué.

---

## Risk management

### Appliqué à la construction

| contrôle | valeur |
|---|---|
| levier brut | 1.00 (chaque jambe = 0,50 du NAV alloué) |
| plafond par position | **8 %, absolu**, sans renormalisation |
| plafond sectoriel | 80 %, honoré, le reste en cash |
| plafond par univers | 35 % du NAV |
| positions par jambe | règle de largeur ci-dessus |

### Sorties — ⚠️ lire attentivement en exécution manuelle

| protection | en manuel |
|---|---|
| stop −10 % par position | ✅ **si vous le déposez** depuis la colonne `Stop ordre` |
| take-profit | ⚠️ +5 % seulement (défaut connu, rapport gain/risque inversé) |
| trailing stop depuis le pic | ❌ **absent** |
| stop ajusté de la volatilité | ❌ absent (le fichier l'indique, l'ordre est à −10 % fixe) |
| **circuit breaker −15 % du portefeuille** | ❌ **absent** |

Le moniteur de risque (trailing stop, circuit breaker, stop vol-ajusté) ne
tourne que dans `--mode live`, la boucle permanente. **En exécution manuelle,
vous n'avez que les stops individuels à −10 % et aucune protection au niveau
du portefeuille.**

---

## Réserves sur les chiffres de backtest

Performance combinée mesurée : **+17,35 %/an, Sharpe 1,99, maxDD −5,42 %,
Calmar 3,20**. À lire avec trois réserves :

1. **Biais du survivant** — les listes de constituants sont celles
   d'aujourd'hui. La jambe zone euro affiche zéro année négative sur onze et un
   Sortino presque double de son Sharpe : la queue de perte est tronquée.
   **Attendez-vous à nettement moins que 1,99 en réel.**
2. **Le backtest ne modélise ni stop ni take-profit** — il tient chaque
   position un mois plein, alors que le système réel pose un stop à −10 %. Ce
   sont deux stratégies différentes, et c'est le candidat le plus solide pour
   expliquer un écart backtest/réel.
3. **Biais factoriel non déclaré** — la sélection est longue croissance /
   courte value (P/B médian 8,80 contre 2,29). Le Sharpe dépend d'un régime.

---

## Autres commandes

### Rapport de performance

```powershell
python main.py --mode report --universes sp500 nasdaq100 nikkei225 stoxx_eur stoxx_uk
```

### Vérifier les ordres de protection en cours

```powershell
python main.py --mode check-protective
```

### Corriger les STP/LMT après exécution

```powershell
# prévisualisation
python main.py --mode refresh-protective --universes sp500 nasdaq100 nikkei225 stoxx_eur stoxx_uk
# application
python main.py --mode refresh-protective --universes sp500 nasdaq100 nikkei225 stoxx_eur stoxx_uk --no-dry-run
```

Recalcule stop-loss et take-profit depuis l'`avg_cost` IBKR réel. Ne touche
jamais aux ordres MKT.

### Backtest complet

```powershell
python main.py --mode backtest --universes sp500 nasdaq100 nikkei225 stoxx_eur stoxx_uk --start 2016-01-01 --end 2026-10-31
```

### Feuille d'ordres sans IBKR

```powershell
python main.py --mode orders --nav 1000000 --positions mes_positions.csv --universes ...
```

CSV à deux colonnes, `ticker` et `quantity` **signée** (positive = long,
négative = short). Moins précis que `--from-ibkr` : un export ne porte pas la
place de cotation, donc `SAN` est ambigu entre Sanofi (Paris) et Banco
Santander (Madrid).

---

## Points de vigilance

| | |
|---|---|
| **`--universes` est obligatoire** | sans lui, le défaut est `sp500` seul |
| **`--nav` est obligatoire** en mode `orders` | sinon rien n'est dimensionné |
| **Pas de `--live`** | son absence = port 7497 = compte démo. L'ajouter bascule sur le réel (7496) |
| **`--no-dry-run`** | nécessaire pour que `flatten` passe réellement les ordres |
| **TWS / IB Gateway ouvert** | API activée. Sans lui, `--from-ibkr` **interrompt** le run plutôt que de produire une feuille qui dirait « tout est nouveau » |
| **Codes de place IBKR européens non vérifiés** | `SBF`, `IBIS`, `BVME`, `BM`, `AEB`, `ENEXT.BE`, `BVL`, `ISE`, `HEX`, `VSE`, `ATH` viennent de la documentation IBKR, non qualifiés contre un TWS réel. À vérifier avant le premier ordre sur un marché non encore tradé |

---

## Vérification dans IBKR après exécution

- **exposition brute** ≈ 100 % du NAV moins le cash conservé (voir `Resume`)
- **exposition nette** ≈ 0 % (bêta-neutre)
- **Excess Liquidity** largement positif (marge Reg-T : 50 % du NAV)

---

## Git

```powershell
git pull
git log --oneline -10
```
