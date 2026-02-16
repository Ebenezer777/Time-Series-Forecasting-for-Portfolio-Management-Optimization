import yfinance as yf
import pandas as pd
from pathlib import Path

ASSETS = ["TSLA", "BND", "SPY"]

def download_data(start="2015-01-01", end="2026-01-15"):
    data = yf.download(
        ASSETS,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False
    )
    return data

def save_processed(data):
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    data.ffill(inplace=True)
    for asset in ASSETS:
        df = data[asset].copy()
        
        df.to_csv(out_dir / f"{asset.lower()}.csv")

if __name__ == "__main__":
    data = download_data()
    save_processed(data)
