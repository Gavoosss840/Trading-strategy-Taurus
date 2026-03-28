"""
Test de connexion FMP et fondamentaux.

Usage:
    python test_fmp.py
"""

import os
from dotenv import load_dotenv

load_dotenv()   # charge .env → FMP_API_KEY

from taurus.fmp import get_fundamentals, get_earnings_calendar

def test_fundamentals(ticker: str = "AAPL"):
    print(f"\n--- Fondamentaux FMP : {ticker} ---")
    data = get_fundamentals(ticker)
    for k, v in data.items():
        if isinstance(v, float):
            print(f"  {k:<25} : {v:>20,.0f}" if v == v else f"  {k:<25} : {'nan':>20}")
        else:
            print(f"  {k:<25} : {v}")

def test_earnings(tickers: list = ["AAPL", "MSFT", "NVDA"]):
    print(f"\n--- Earnings calendar : {tickers} ---")
    events = get_earnings_calendar(tickers)
    if not events:
        print("  Aucun événement trouvé (hors fenêtre calendrier)")
        return
    for e in events[:10]:
        print(f"  {e['ticker']:<8} {e['date']}  EPS={e['eps']}  Est={e['eps_est']}")

if __name__ == "__main__":
    test_fundamentals("AAPL")
    test_fundamentals("MSFT")
    test_earnings(["AAPL", "MSFT", "NVDA", "JPM", "XOM"])
    print("\n✓ FMP opérationnel.")
