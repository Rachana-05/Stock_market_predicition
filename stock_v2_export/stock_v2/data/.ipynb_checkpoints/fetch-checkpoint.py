# data/fetch.py
import os, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
import joblib
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *

def fetch_ticker(ticker: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            if len(df) > 100:
                print(f"  ✓ {ticker}: {len(df)} rows")
                return df
            else:
                print(f"  ✗ {ticker}: only {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {ticker} attempt {attempt+1}: {e}")
            time.sleep(2)
    return pd.DataFrame()

def fetch_all_stocks() -> dict:
    cache = os.path.join(DATA_DIR, "raw_data.pkl")
    if os.path.exists(cache):
        print("  Using cached data...")
        return joblib.load(cache)

    print("Downloading NSE data (8 years, 10 stocks + Nifty50)...")
    data = {}
    for ticker in TICKERS:
        df = fetch_ticker(ticker)
        if not df.empty:
            data[ticker] = df
        time.sleep(0.3)

    print("Downloading Nifty50...")
    nifty = fetch_ticker(BENCHMARK)
    if not nifty.empty:
        data["NIFTY"] = nifty

    joblib.dump(data, cache)
    print(f"Cached {len(data)} datasets.")
    return data
