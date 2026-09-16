"""
Taurus – Correspondance code SIC (SEC) → secteur GICS.

Le secteur pilote quatre mécanismes : le taux de coûts de détresse de l'écran
MM, le plafond sectoriel de `portfolio.py`, la neutralisation sectorielle du
pilier valorisation, et l'exemption du garde-fou de couverture des intérêts
pour les financières.  Sans lui, les quatre sont inertes.

SEC EDGAR publie un code SIC pour chaque déclarant ; ce module le traduit dans
la nomenclature sectorielle attendue par le moteur.

La table suit les plages SIC officielles ; seules les plages qui changent de
secteur GICS sont listées, l'ordre de lecture allant du plus spécifique au plus
général.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# (borne_basse, borne_haute, secteur) — bornes incluses, lues dans l'ordre.
_SIC_RANGES: List[Tuple[int, int, str]] = [
    # ── Agriculture, mines, énergie ──────────────────────────────────────
    (100,   999,  "Consumer Staples"),        # production agricole
    (1000,  1099, "Materials"),               # mines métalliques
    (1200,  1299, "Energy"),                  # charbon
    (1300,  1399, "Energy"),                  # pétrole et gaz
    (1400,  1499, "Materials"),               # carrières, minéraux non métalliques
    (1500,  1799, "Industrials"),             # construction

    # ── Industrie manufacturière ─────────────────────────────────────────
    (2000,  2199, "Consumer Staples"),        # alimentaire, tabac
    (2200,  2399, "Consumer Discretionary"),  # textile, habillement
    (2400,  2499, "Materials"),               # bois
    (2500,  2599, "Consumer Discretionary"),  # ameublement
    (2600,  2699, "Materials"),               # papier
    (2700,  2799, "Communication Services"),  # édition, imprimerie
    (2800,  2829, "Materials"),               # produits chimiques
    (2830,  2836, "Health Care"),             # pharmacie, biotechnologies
    (2840,  2844, "Consumer Staples"),        # savons, cosmétiques, parfums
    (2845,  2899, "Materials"),               # chimie de spécialité
    (2900,  2999, "Energy"),                  # raffinage
    (3000,  3099, "Consumer Discretionary"),  # caoutchouc, plastique
    (3100,  3199, "Consumer Discretionary"),  # cuir
    (3200,  3299, "Materials"),               # verre, ciment
    (3300,  3399, "Materials"),               # métallurgie
    (3400,  3499, "Industrials"),             # produits métalliques
    (3500,  3558, "Industrials"),             # machines industrielles
    (3559,  3559, "Information Technology"),  # équipement de semi-conducteurs (ASML)
    (3560,  3569, "Industrials"),             # machines générales
    (3570,  3579, "Information Technology"),  # matériel informatique
    (3580,  3599, "Industrials"),             # machines diverses
    (3600,  3639, "Industrials"),             # équipement électrique
    (3640,  3659, "Consumer Discretionary"),  # électronique grand public
    (3660,  3669, "Communication Services"),  # équipement de télécommunication
    (3670,  3679, "Information Technology"),  # semi-conducteurs, composants
    (3680,  3699, "Information Technology"),  # ordinateurs, périphériques
    (3700,  3716, "Consumer Discretionary"),  # automobile
    (3720,  3799, "Industrials"),             # aéronautique, ferroviaire, naval
    (3800,  3826, "Information Technology"),  # instruments de mesure
    (3827,  3899, "Health Care"),             # dispositifs médicaux, optique
    (3900,  3999, "Consumer Discretionary"),  # industries diverses

    # ── Transport, services publics, télécoms ────────────────────────────
    (4000,  4499, "Industrials"),             # transport
    (4500,  4599, "Industrials"),             # transport aérien
    (4600,  4699, "Energy"),                  # oléoducs et gazoducs
    (4700,  4799, "Industrials"),             # services de transport
    (4800,  4829, "Communication Services"),  # télécommunications
    (4830,  4899, "Communication Services"),  # radio, télévision, câble
    (4900,  4999, "Utilities"),               # services publics

    # ── Distribution ─────────────────────────────────────────────────────
    (5000,  5099, "Industrials"),             # négoce de gros durables
    (5100,  5199, "Consumer Staples"),        # négoce de gros non durables
    (5200,  5399, "Consumer Discretionary"),  # grande distribution
    (5400,  5499, "Consumer Staples"),        # alimentation de détail
    (5500,  5599, "Consumer Discretionary"),  # concessionnaires automobiles
    (5600,  5699, "Consumer Discretionary"),  # habillement de détail
    (5700,  5899, "Consumer Discretionary"),  # ameublement, restauration
    (5900,  5911, "Consumer Staples"),        # pharmacies
    (5912,  5999, "Consumer Discretionary"),  # détail divers

    # ── Finance et immobilier ────────────────────────────────────────────
    (6000,  6499, "Financials"),              # banques, assurances
    (6500,  6599, "Real Estate"),             # immobilier
    (6798,  6798, "Real Estate"),             # REIT
    (6600,  6999, "Financials"),              # holdings, fonds

    # ── Services ─────────────────────────────────────────────────────────
    (7000,  7099, "Consumer Discretionary"),  # hôtellerie
    (7200,  7299, "Consumer Discretionary"),  # services aux particuliers
    (7300,  7369, "Industrials"),             # services aux entreprises
    (7370,  7379, "Information Technology"),  # services informatiques, logiciels
    (7380,  7399, "Industrials"),             # services divers aux entreprises
    (7500,  7599, "Consumer Discretionary"),  # réparation automobile
    (7600,  7699, "Consumer Discretionary"),  # réparations diverses
    (7800,  7899, "Communication Services"),  # cinéma, production
    (7900,  7999, "Communication Services"),  # loisirs
    (8000,  8099, "Health Care"),             # services de santé
    (8100,  8199, "Industrials"),             # services juridiques
    (8200,  8299, "Consumer Discretionary"),  # enseignement
    (8300,  8399, "Health Care"),             # services sociaux
    (8700,  8799, "Industrials"),             # ingénierie, conseil, comptabilité
]


def sector_from_sic(sic: Optional[str | int]) -> str:
    """Traduit un code SIC en secteur GICS, ou "Unknown" si indéterminable."""
    if sic is None or sic == "":
        return "Unknown"
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return "Unknown"

    for low, high, sector in _SIC_RANGES:
        if low <= code <= high:
            return sector
    return "Unknown"
