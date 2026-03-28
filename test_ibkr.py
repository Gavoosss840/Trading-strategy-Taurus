"""
Test de connexion IBKR TWS API.
Prérequis : TWS ouvert sur le port 7497 (paper trading).

Usage:
    python test_ibkr.py
"""

from ib_insync import IB, Stock, Future, util

def test_connection():
    ib = IB()
    try:
        ib.connect("127.0.0.1", 7497, clientId=1)
        print(f"✓ Connecté à TWS  |  version={ib.client.serverVersion()}")
        return ib
    except Exception as e:
        print(f"✗ Connexion échouée : {e}")
        print("  → Vérifie que TWS est ouvert et que l'API est activée (port 7497)")
        return None


def test_stock_price(ib: IB, ticker: str = "AAPL"):
    print(f"\n--- Prix (données différées) : {ticker} ---")
    # Type 3 = données différées gratuites (15 min de délai)
    # Suffisant pour Taurus qui rebalance mensuellement
    ib.reqMarketDataType(3)
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    mkt = ib.reqMktData(contract, "", False, False)
    ib.sleep(3)
    print(f"  Last  : {mkt.last}")
    print(f"  Bid   : {mkt.bid}")
    print(f"  Ask   : {mkt.ask}")
    print(f"  Volume: {mkt.volume}")
    ib.cancelMktData(contract)


def test_historical_data(ib: IB, ticker: str = "AAPL"):
    print(f"\n--- Historique mensuel : {ticker} (12 derniers mois) ---")
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    # ADJUSTED_LAST non supporté en mensuel → on prend journalier TRADES
    # et on resample en mensuel (fin de mois)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="12 M",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        keepUpToDate=False,
    )
    if not bars:
        print("  ⚠ Aucune donnée retournée (abonnement ou heure de marché)")
        return
    import pandas as pd
    df = util.df(bars)[["date", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").last().reset_index()
    print(df.tail(6).to_string(index=False))


def _front_month_es() -> str:
    """
    Retourne l'échéance front-month du contrat ES.
    ES expire le 3ème vendredi de mars/juin/sept/déc (codes H/M/U/Z).
    On prend le prochain contrat non expiré.
    """
    from datetime import date
    import calendar
    today = date.today()
    expiry_months = [3, 6, 9, 12]
    for year in [today.year, today.year + 1]:
        for m in expiry_months:
            # 3ème vendredi du mois
            first_day = date(year, m, 1)
            first_friday = first_day + __import__("datetime").timedelta(
                days=(4 - first_day.weekday()) % 7
            )
            third_friday = first_friday + __import__("datetime").timedelta(weeks=2)
            if third_friday > today:
                return f"{year}{m:02d}"
    return "202606"   # fallback


def test_es_futures(ib: IB):
    print("\n--- Futures ES Mini (S&P 500) ---")
    try:
        expiry = _front_month_es()
        contract = Future("ES", lastTradeDateOrContractMonth=expiry,
                          exchange="CME", currency="USD")
        ib.qualifyContracts(contract)
        print(f"  Contrat front-month : {contract.localSymbol} (expiry {expiry})")
        ib.reqMarketDataType(3)   # données différées si pas d'abonnement CME
        mkt = ib.reqMktData(contract, "", False, False)
        ib.sleep(3)
        print(f"  ES last  : {mkt.last}")
        print(f"  ES bid   : {mkt.bid}")
        print(f"  ES ask   : {mkt.ask}")
        ib.cancelMktData(contract)
    except Exception as e:
        print(f"  ⚠ ES futures : {e}")


def test_account_summary(ib: IB):
    print("\n--- Résumé du compte Paper ---")
    summary = ib.accountSummary()
    keys = ["NetLiquidation", "TotalCashValue", "BuyingPower", "GrossPositionValue"]
    for item in summary:
        if item.tag in keys:
            print(f"  {item.tag:<25} : {float(item.value):>15,.2f} {item.currency}")


if __name__ == "__main__":
    ib = test_connection()
    if ib and ib.isConnected():
        test_stock_price(ib)
        test_historical_data(ib)
        test_es_futures(ib)
        test_account_summary(ib)
        ib.disconnect()
        print("\n✓ Tous les tests terminés. Connexion IBKR opérationnelle.")
