import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

TICKERS_CSV = "sp500_tickers.csv"
LOOKBACK_YEARS = 3
RISK_AVERSION_LAMBDA = 1.0
MAX_PE = 30  # You can change this if you want to adjust the PE threshold for bargains

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
        # Flatten MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col]).strip('_') for col in data.columns]
        close_col = next((c for c in data.columns if str(c).startswith('Close')), None)
        if close_col is None:
            return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}
        closes = data[close_col].dropna()
        if len(closes) < 60: return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}
        returns = closes.pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return {k: np.nan for k in ["volatility", "sharpe", "momentum_1m", "momentum_3m", "drawdown", "beta"]}
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        spy = yf.download("SPY", period=f"{years}y", interval="1d", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = ['_'.join([str(i) for i in col]).strip('_') for col in spy.columns]
        spy_close_col = next((c for c in spy.columns if str(c).startswith('Close')), None)
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
# Only keep stocks with all relevant metrics present
df = df.dropna(subset=['sharpe', 'volatility', 'pe', 'roe', 'revenue_growth', 'eps_growth', 'momentum_1m', 'momentum_3m', 'drawdown'])

# Remove stocks that are not bargains by PE
df = df[df['pe'] < MAX_PE]

print(f"Total bargain candidates with valid dip metrics: {len(df)}")
if df.empty:
    raise ValueError("No bargain candidates found. Try a higher PE threshold or check your data.")

# Ensure numeric columns
for col in ['sharpe', 'volatility', 'pe', 'roe', 'revenue_growth', 'eps_growth', 'momentum_1m', 'momentum_3m', 'drawdown', 'beta']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- QUALITY METRIC ---
df['quality'] = df[['roe', 'revenue_growth', 'eps_growth']].apply(lambda x: (x - x.mean()) / x.std()).sum(axis=1)

# --- VALUE METRIC ---
df['value'] = -((df['pe'] - df['pe'].mean()) / df['pe'].std())

# --- DIP METRICS ---
df['dip'] = -((df['momentum_1m'] + df['momentum_3m']) / 2)
df['off_high'] = -df['drawdown']

# --- COMPOSITE SCORE ---
# You can adjust the weights below to emphasize dip more or less
score_weights = {
    'quality': 1.0,
    'value': 1.0,
    'dip': 0.7,
    'off_high': 0.7,
    'sharpe': 0.5,
    'volatility': -RISK_AVERSION_LAMBDA,
}
df['score'] = (
    score_weights['quality'] * df['quality'] +
    score_weights['value'] * df['value'] +
    score_weights['dip'] * df['dip'] +
    score_weights['off_high'] * df['off_high'] +
    score_weights['sharpe'] * df['sharpe'] +
    score_weights['volatility'] * df['volatility']
)

df_sorted = df.sort_values('score', ascending=False)
top_dip_bargains = df_sorted.head(10)[['ticker', 'score', 'pe', 'roe', 'revenue_growth', 'eps_growth', 'sharpe', 'volatility', 'momentum_1m', 'momentum_3m', 'drawdown']]

print("Top 10 Bargain Dip Candidates (Quality + Dip):")
print(top_dip_bargains.to_string(index=False))