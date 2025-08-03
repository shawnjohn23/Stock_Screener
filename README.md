# Stock_Screener
in this repository I had GitHub code a stock screener and added features that would make the screener look for strong buys at a good price
Here below I coppied the most recent work journal. 
Stock Screener Project Journal  
================================

**Project Start:** 2024-07  
**User:** shawnjohn23  
**Assistant:** GitHub Copilot

---

## Initial Request

User wanted a Python stock screener to rank S&P 500 stocks by risk/return, using metrics like Sharpe ratio and volatility.  
First versions built basic data pipeline with yfinance, pandas, and simple scoring.

---

## Major Updates & Issues

- **Early Issues:**  
  - Data columns missing after standardization (`ValueError`).  
  - DataFrame not defined (`NameError`).  
  - All result columns were NaN: yfinance API changes, MultiIndex columns, formatting errors.

- **Debugging & Testing:**  
  - Added diagnostics to print columns, dtypes, sample data.  
  - Created test script for yfinance, fundamentals, price data, and beta calculation.  
  - Identified MultiIndex columns in yfinance DataFrames as root cause for many errors.

- **Breakthrough:**  
  - Flattened DataFrame columns and robust column selection.  
  - Ensured all key metrics are numeric and present.  
  - First successful output: Top 10 stocks by risk/return, e.g. Walmart, JPMorgan.

---

## Screener Versions and Capabilities

- **v.6:**  
  - Stable screening by risk/return (Sharpe, volatility, and fundamental metrics).  
  - Handles yfinance MultiIndex columns.  
  - Output: Top "strongest bets" by composite score.

- **v.7:**  
  - Added "quality" and "value" metrics (ROE, growth, PE).  
  - Penalized high PE stocks, highlighted bargains with strong fundamentals.  
  - Output: Top 10 "bargain quality stocks".

- **v.8:**  
  - Focused on "dip" strategy: find quality companies with undervalued prices and recent price drops.  
  - Filters out high PE stocks.  
  - Dip metrics: negative recent momentum, distance from highs (drawdown).  
  - Composite score includes quality, value, dip, and risk.  
  - Output: Top 10 "bargain dip candidates" for buy-the-dip investing.

---

## Strategy Discussion

- v.7/v.8 allow user to tune for value, risk, and dip exposure.
- Highlighted risk of "value traps" and importance of fundamental strength.
- Discussed use of technical indicators (momentum, drawdown) and possible future additions (RSI, sector comparison).

---

## Next Steps & Suggestions

- Add additional technicals: RSI, moving averages.
- Compare PE to sector or historical average.
- Export results to CSV for further analysis.
- Add plotting for visual analysis of candidates.

---

**Current Version:** v.8  
**Capabilities:**  
- Screens S&P 500 for undervalued, quality companies in a dip.
- Easily tunable scoring.
- Robust error handling for yfinance quirks.
