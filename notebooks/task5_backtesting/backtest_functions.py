import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# ----------------------------
# Download Backtest Data
# ----------------------------
def get_backtest_data():
    tickers = ["TSLA", "BND", "SPY"]
    
    # Use auto_adjust=True to avoid the 'Adj Close' vs 'Close' headache
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01", auto_adjust=True)
    
    # Since auto_adjust is True, we just grab 'Close'
    # This is safer for multi-ticker dataframes
    if 'Close' in data.columns:
        returns = data['Close'].pct_change().dropna()
    else:
        # If the download failed entirely, this prevents a crash
        print("Warning: No data found for these tickers/dates.")
        return None
        
    return returns


# ----------------------------
# Portfolio Returns
# ----------------------------
def portfolio_returns(returns, weights):
    weights_array = np.array(list(weights.values()))
    port_returns = returns.dot(weights_array)
    return port_returns


# ----------------------------
# Metrics
# ----------------------------
def performance_metrics(returns):
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252/len(returns)) - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    max_drawdown = (returns.cumsum().expanding().max() - returns.cumsum()).max()
    return total_return, annual_return, sharpe, max_drawdown
