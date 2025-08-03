import yfinance as yf
import pandas as pd

ticker = "QCOM"
LOOKBACK_YEARS = 3

# Download price data
data = yf.download(ticker, period=f"{LOOKBACK_YEARS}y", interval="1d", progress=False)

# Flatten MultiIndex columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join([str(i) for i in col]).strip('_') for col in data.columns]

close_col = next((c for c in data.columns if str(c).startswith('Close')), None)
closes = data[close_col].dropna() if close_col is not None else pd.Series(dtype=float)

# Calculate price/return metrics
returns = closes.pct_change().dropna()
volatility = returns.std() * (252**0.5) if not returns.empty else None
sharpe = returns.mean() / returns.std() * (252**0.5) if returns.std() != 0 else None
momentum_1m = closes[-21:].pct_change().sum() if len(closes) >= 21 else None
momentum_3m = closes[-63:].pct_change().sum() if len(closes) >= 63 else None
drawdown = (closes / closes.cummax() - 1).min() if not closes.empty else None

# Get fundamental info
stock = yf.Ticker(ticker)
info = stock.info

fundamentals = {
    "Ticker": ticker,
    "Company Name": info.get("longName"),
    "Sector": info.get("sector"),
    "Industry": info.get("industry"),
    "Market Cap": info.get("marketCap"),
    "PE Ratio": info.get("trailingPE"),
    "EPS": info.get("trailingEps"),
    "ROE": info.get("returnOnEquity"),
    "Debt/Equity": info.get("debtToEquity"),
    "Insider Ownership": info.get("heldPercentInsiders"),
    "Revenue Growth": info.get("revenueGrowth"),
    "Earnings Growth": info.get("earningsGrowth"),
    "Dividend Yield": info.get("dividendYield"),
    "Beta": info.get("beta"),
    "52 Week High": info.get("fiftyTwoWeekHigh"),
    "52 Week Low": info.get("fiftyTwoWeekLow"),
    "Current Price": info.get("currentPrice"),
    "Volatility (Annualized)": volatility,
    "Sharpe Ratio": sharpe,
    "Momentum 1M": momentum_1m,
    "Momentum 3M": momentum_3m,
    "Max Drawdown": drawdown,
}

# Create DataFrame for fundamentals
fund_df = pd.DataFrame([fundamentals])

# Save full price history in a separate sheet
with pd.ExcelWriter('QCOM_stock_report.xlsx') as writer:
    fund_df.to_excel(writer, index=False, sheet_name='Summary')
    closes.to_frame(name='Close').to_excel(writer, sheet_name='Close History')
    data.to_excel(writer, sheet_name='Full OHLCV')

print("Exported QCOM data to QCOM_stock_report.xlsx")