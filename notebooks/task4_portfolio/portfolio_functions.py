import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns

# -----------------------------
# Download Data
# -----------------------------
def get_data():
    tickers = ["TSLA", "BND", "SPY"]
    
    # 1. Download the full data first
    raw_data = yf.download(tickers, start="2020-01-01")
    
    # 2. Check if 'Adj Close' exists; if not, fall back to 'Close'
    # This handles changes in different yfinance versions
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']
    
    # 3. Calculate daily returns
    returns = data.pct_change().dropna()
    
    return data, returns


# -----------------------------
# Expected Returns
# -----------------------------
def expected_returns_calc(returns, tsla_forecast_annual):
    mu = returns.mean() * 252
    mu["TSLA"] = tsla_forecast_annual
    return mu


# -----------------------------
# Covariance Matrix
# -----------------------------
def covariance_matrix(returns):
    cov = returns.cov() * 252
    return cov


import os

def plot_covariance(cov):
    # This line is the magic fix:
    # It creates 'outputs' folder in your CURRENT working directory
    os.makedirs("outputs", exist_ok=True) 
        
    plt.figure(figsize=(6,4))
    sns.heatmap(cov, annot=True, cmap="coolwarm")
    plt.title("Covariance Matrix")
    
    save_path = "outputs/covariance_heatmap.png"
    plt.savefig(save_path)
    print(f"Saved to: {os.path.abspath(save_path)}") # This will tell you exactly where it went
    plt.close()


# -----------------------------
# Efficient Frontier
# -----------------------------
def optimize_portfolio(mu, cov):
    ef = EfficientFrontier(mu, cov)

    weights_sharpe = ef.max_sharpe()
    cleaned_sharpe = ef.clean_weights()
    perf_sharpe = ef.portfolio_performance()

    ef2 = EfficientFrontier(mu, cov)
    weights_minvol = ef2.min_volatility()
    cleaned_minvol = ef2.clean_weights()
    perf_minvol = ef2.portfolio_performance()

    return cleaned_sharpe, perf_sharpe, cleaned_minvol, perf_minvol
