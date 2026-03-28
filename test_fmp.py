"""
Test fondamentaux SEC EDGAR (gratuit, source officielle).

Usage:
    python test_fmp.py
"""

from taurus.sec_edgar import get_fundamentals

def test_fundamentals(ticker: str):
    print(f"\n--- Fondamentaux SEC EDGAR : {ticker} ---")
    data = get_fundamentals(ticker)
    for k, v in data.items():
        if isinstance(v, float) and v == v:
            print(f"  {k:<25} : {v:>20,.0f}")
        elif isinstance(v, float):
            print(f"  {k:<25} : {'nan':>20}")
        else:
            print(f"  {k:<25} : {v}")

if __name__ == "__main__":
    for ticker in ["AAPL", "MSFT", "JPM"]:
        test_fundamentals(ticker)
    print("\n✓ SEC EDGAR opérationnel.")
