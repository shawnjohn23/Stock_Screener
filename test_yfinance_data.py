import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'V']

def test_price_data(ticker):
    try:
        df = yf.download(ticker, period="3y", interval="1d", progress=False)
        print(f"\n===== {ticker} Price Data =====")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Head:", df.head(2).to_dict())
        closes = df['Close'].dropna()
        print("Close count:", len(closes))
        returns = closes.pct_change().dropna()
        print("Returns count:", len(returns))
        print("Returns stats: mean = {:.6f}, std = {:.6f}".format(returns.mean(), returns.std()))
        volatility = returns.std() * np.sqrt(252) if not returns.empty else None
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else None
        print("Volatility:", volatility)
        print("Sharpe:", sharpe)
    except Exception as e:
        print(f"Error with price data for {ticker}: {repr(e)}")

def test_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        print(f"\n===== {ticker} Fundamentals =====")
        print("Info keys:", list(info.keys()))
        keys = ["trailingPE", "returnOnEquity", "debtToEquity", "heldPercentInsiders", "revenueGrowth", "earningsGrowth"]
        for k in keys:
            print(f"{k}: {info.get(k, 'MISSING')}")
    except Exception as e:
        print(f"Error with fundamentals for {ticker}: {repr(e)}")

def test_beta(ticker):
    try:
        data = yf.download(ticker, period="3y", interval="1d", progress=False)
        closes = data['Close'].dropna()
        returns = closes.pct_change().dropna()
        spy = yf.download("SPY", period="3y", interval="1d", progress=False)['Close'].dropna()
        spy_returns = spy.pct_change().dropna()
        min_len = min(len(returns), len(spy_returns))
        print(f"\n===== {ticker} Beta Calculation =====")
        if min_len > 0:
            beta = np.cov(returns[-min_len:], spy_returns[-min_len:])[0][1] / np.var(spy_returns[-min_len:])
            print("Beta:", beta)
        else:
            print("Insufficient data for beta calculation.")
    except Exception as e:
        print(f"Error with beta calculation for {ticker}: {repr(e)}")

def main():
    print("yfinance version:", yf.__version__)
    for ticker in TICKERS:
        test_price_data(ticker)
        test_fundamentals(ticker)
        test_beta(ticker)
        print("\n------------------------\n")

if __name__ == "__main__":
    main()