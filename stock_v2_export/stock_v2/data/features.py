# data/features.py — 45-feature engineering + walk-forward validation
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler
import joblib, os, sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *

FEATURE_COLS = [
    # Trend (12)
    "sma_10","sma_20","sma_50","sma_200","ema_9","ema_21",
    "macd","macd_sig","macd_hist","adx","adx_pos","adx_neg",
    # Momentum (7)
    "rsi_7","rsi_14","rsi_21","stoch_k","stoch_d","cci","williams",
    # Volatility (5)
    "bb_upper","bb_lower","bb_width","bb_pct","atr",
    # Volume (5)
    "obv","mfi","cmf","vol_ratio_5","vol_ratio_20",
    # Returns (10)
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20",
    "vol_5d","vol_20d","zscore_20","trend_consistency",
    # Relative (6)
    "price_vs_sma20","price_vs_sma50","body","hl_range",
    "nifty_ret_5","beta_vs_nifty",
]
# Exactly 45
assert len(FEATURE_COLS) == 45, f"Got {len(FEATURE_COLS)}"


def compute_features(df: pd.DataFrame, nifty: pd.DataFrame = None) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"].squeeze()
    h = d["High"].squeeze()
    l = d["Low"].squeeze()
    v = d["Volume"].squeeze()
    o = d["Open"].squeeze()

    d["sma_10"]    = ta.trend.sma_indicator(c, 10)
    d["sma_20"]    = ta.trend.sma_indicator(c, 20)
    d["sma_50"]    = ta.trend.sma_indicator(c, 50)
    d["sma_200"]   = ta.trend.sma_indicator(c, 200)
    d["ema_9"]     = ta.trend.ema_indicator(c, 9)
    d["ema_21"]    = ta.trend.ema_indicator(c, 21)
    d["macd"]      = ta.trend.macd(c)
    d["macd_sig"]  = ta.trend.macd_signal(c)
    d["macd_hist"] = ta.trend.macd_diff(c)
    d["adx"]       = ta.trend.adx(h, l, c)
    d["adx_pos"]   = ta.trend.adx_pos(h, l, c)
    d["adx_neg"]   = ta.trend.adx_neg(h, l, c)

    d["rsi_7"]     = ta.momentum.rsi(c, 7)
    d["rsi_14"]    = ta.momentum.rsi(c, 14)
    d["rsi_21"]    = ta.momentum.rsi(c, 21)
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c)
    d["stoch_k"]   = stoch.stoch()
    d["stoch_d"]   = stoch.stoch_signal()
    d["cci"]       = ta.trend.cci(h, l, c)
    d["williams"]  = ta.momentum.williams_r(h, l, c)

    bb = ta.volatility.BollingerBands(c, window=20)
    d["bb_upper"]  = bb.bollinger_hband()
    d["bb_lower"]  = bb.bollinger_lband()
    d["bb_width"]  = bb.bollinger_wband()
    d["bb_pct"]    = bb.bollinger_pband()
    d["atr"]       = ta.volatility.average_true_range(h, l, c)

    d["obv"]          = ta.volume.on_balance_volume(c, v)
    d["mfi"]          = ta.volume.money_flow_index(h, l, c, v)
    d["cmf"]          = ta.volume.chaikin_money_flow(h, l, c, v)
    d["vol_ratio_5"]  = v / (v.rolling(5).mean()  + 1e-8)
    d["vol_ratio_20"] = v / (v.rolling(20).mean() + 1e-8)

    d["ret_1"]  = c.pct_change(1)
    d["ret_2"]  = c.pct_change(2)
    d["ret_3"]  = c.pct_change(3)
    d["ret_5"]  = c.pct_change(5)
    d["ret_10"] = c.pct_change(10)
    d["ret_20"] = c.pct_change(20)
    d["vol_5d"] = c.pct_change(1).rolling(5).std()
    d["vol_20d"]= c.pct_change(1).rolling(20).std()
    d["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)
    d["trend_consistency"] = (c > c.shift(1)).rolling(10).mean()

    d["price_vs_sma20"] = (c - d["sma_20"]) / (d["sma_20"] + 1e-8)
    d["price_vs_sma50"] = (c - d["sma_50"]) / (d["sma_50"] + 1e-8)
    d["body"]     = (c - o) / (c + 1e-8)
    d["hl_range"] = (h - l)  / (c + 1e-8)

    if nifty is not None and len(nifty) > 10:
        nc = nifty["Close"].squeeze()
        na = nc.reindex(d.index, method="ffill")
        d["nifty_ret_5"]   = na.pct_change(5)
        d["beta_vs_nifty"] = c.pct_change(1).rolling(20).corr(na.pct_change(1))
    else:
        d["nifty_ret_5"]   = 0.0
        d["beta_vs_nifty"] = 1.0

    # Target: next-day direction
    d["target"] = (c.shift(-1) > c).astype(int)
    d.dropna(inplace=True)
    return d


def build_multi_stock_dataset(raw_data: dict):
    all_flat, all_seq, all_y = [], [], []
    nifty = raw_data.get("NIFTY", None)
    scalers = {}

    for ticker in TICKERS:
        if ticker not in raw_data:
            continue
        df = raw_data[ticker]
        df_feat = compute_features(df, nifty)
        if len(df_feat) < SEQUENCE_LEN + 50:
            print(f"  Skipping {ticker}: only {len(df_feat)} rows after features")
            continue

        X_raw = df_feat[FEATURE_COLS].values.astype(np.float32)
        y     = df_feat["target"].values.astype(np.int64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        scalers[ticker] = scaler

        for i in range(SEQUENCE_LEN, len(X_scaled)):
            all_flat.append(X_scaled[i])
            all_y.append(y[i])
            all_seq.append(X_scaled[i - SEQUENCE_LEN:i])

    os.makedirs(SAVED_MODELS, exist_ok=True)
    joblib.dump(scalers, os.path.join(SAVED_MODELS, "scalers.pkl"))

    X_flat = np.array(all_flat, dtype=np.float32)
    X_seq  = np.array(all_seq,  dtype=np.float32)
    y_arr  = np.array(all_y,    dtype=np.int64)

    print(f"Dataset: {X_flat.shape[0]} samples, {X_flat.shape[1]} features, UP={y_arr.mean()*100:.1f}%")
    return X_flat, X_seq, y_arr, scalers


def train_test_split_time(X_flat, X_seq, y, test_ratio=0.20):
    n = len(y)
    split = int(n * (1 - test_ratio))
    return (X_flat[:split], X_flat[split:],
            X_seq[:split],  X_seq[split:],
            y[:split],      y[split:])
