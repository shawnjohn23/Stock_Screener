import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

TICKERS_CSV = "sp500_tickers.csv"
LOOKBACK_YEARS = 3
RISK_AVERSION_LAMBDA = 1.0

# --- LOAD TICKERS ---
sp500 = pd.read_csv(TICKERS_CSV)
tickers = sp500['Symbol'].tolist()

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "pe": pd.to_numeric(info.get("trailingPE", np.nan), errors='coerce'),
            "roe": pd.to_numeric(info.get("returnOnEquity", np.nan), errors='coerce'),
            "debt_equity": pd.to_numeric(info.get("debtToEquity", np.nan), errors='coerce'),
            "insider_own": pd.to_numeric(info.get("heldPercentInsiders", np.nan), errors='coerce'),
            "revenue_growth": pd.to_numeric(info.get("revenueGrowth", np.nan), errors='coerce'),
            "eps_growth": pd.to_numeric(info.get("earningsGrowth", np.nan), errors='coerce'),
        }
    except Exception:
        return {k: np.nan for k in ["pe", "roe", "debt_equity", "insider_own", "revenue_growth", "eps_growth"]}

def get_statistical_features(ticker, years=LOOKBACK_YEARS):
    try:
        data = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
        # --- FLATTEN MULTIINDEX COLUMNS ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col]).strip('_') for col in data.columns]

        # Find correct 'Close' column
        close_col = None
        for c in data.columns:
            if str(c).startswith('Close'):
                close_col = c
                break
        if close_col is None:
            return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}

        closes = data[close_col].dropna()
        if len(closes) < 60:
            return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}

        returns = closes.pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}

        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # Beta calculation (vs SPY)
        spy = yf.download("SPY", period=f"{years}y", interval="1d", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = ['_'.join([str(i) for i in col]).strip('_') for col in spy.columns]
        spy_close_col = None
        for c in spy.columns:
            if str(c).startswith('Close'):
                spy_close_col = c
                break
        if spy_close_col is not None:
            spy_closes = spy[spy_close_col].dropna()
            spy_returns = spy_closes.pct_change().dropna()
            min_len = min(len(returns), len(spy_returns))
            if min_len > 0:
                r = returns[-min_len:].values
                sr = spy_returns[-min_len:].values
                beta = np.cov(r, sr)[0][1] / np.var(sr) if np.var(sr) != 0 else np.nan
            else:
                beta = np.nan
        else:
            beta = np.nan

        momentum_1m = closes[-21:].pct_change().sum() if len(closes) >= 21 else np.nan
        momentum_3m = closes[-63:].pct_change().sum() if len(closes) >= 63 else np.nan
        drawdown = (closes / closes.cummax() - 1).min()
        return {
            "volatility": volatility,
            "sharpe": sharpe,
            "momentum_1m": momentum_1m,
            "momentum_3m": momentum_3m,
            "drawdown": drawdown,
            "beta": beta,
        }
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}

# --- BUILD FEATURE MATRIX ---
features = []
for ticker in tickers:
    try:
        f = get_fundamentals(ticker)
        s = get_statistical_features(ticker)
        f.update(s)
        f['ticker'] = ticker
        features.append(f)
    except Exception as e:
        print(f"Error for {ticker}: {e}")

df = pd.DataFrame(features)
df = df.dropna(subset=['sharpe', 'volatility'])

print(f"Total stocks with valid sharpe & volatility: {len(df)}")
if df.empty:
    raise ValueError("No stocks with valid sharpe and volatility. Try different tickers or check your internet connection.")

# Ensure 'sharpe' and 'volatility' are numeric and present
for col in ['sharpe', 'volatility']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Build numeric_cols list, always including 'sharpe' and 'volatility'
numeric_cols = ['sharpe', 'volatility'] + [
    col for col in df.select_dtypes(include=[np.number]).columns 
    if col not in ['sharpe', 'volatility']
]
print("Numeric columns to standardize:", numeric_cols)

df_std = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
print("Columns in df_std:", df_std.columns)

if 'sharpe' not in df_std.columns or 'volatility' not in df_std.columns:
    print("df_std columns:", df_std.columns)
    print("df head:", df.head())
    raise ValueError("Column 'sharpe' or 'volatility' missing after standardization.")

# Objective function
df['score'] = df_std['score'] = df_std['sharpe'] - RISK_AVERSION_LAMBDA * df_std['volatility']
df_sorted = df.sort_values('score', ascending=False)
top_bets = df_sorted.head(10)[['ticker', 'sharpe', 'volatility', 'score']]

print("Top 10 Strongest Bets:")
print(top_bets.to_string(index=False))