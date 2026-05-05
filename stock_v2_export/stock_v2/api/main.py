# api/main.py — FastAPI backend
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/stock_v2")

import numpy as np
import pandas as pd
import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import ta

from config import *
from data.features import compute_features, FEATURE_COLS

app = FastAPI(title="NSE Stock Prediction API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── serve frontend
FRONTEND = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

PARADIGM_MAP = {
    "LSTM":"ML","GRU":"ML","XGBoost":"ML","RandomForest":"ML","SVM":"ML",
    "DNN":"DL","Transformer":"DL","CNNLSTM":"DL","BiLSTM":"DL","TCN":"DL",
    "QSVM_ZZFeatureMap":"QML","VQC_ZZFeatureMap":"QML",
    "HybridQNN":"QNN",
    "StackingEnsemble":"Ensemble","WeightedVote":"Ensemble",
}

# ── cache scalers at startup
_scalers = None
def get_scalers():
    global _scalers
    if _scalers is None:
        sp = os.path.join(SAVED_MODELS, "scalers.pkl")
        if os.path.exists(sp):
            _scalers = joblib.load(sp)
    return _scalers or {}


@app.get("/")
def root():
    return {"status": "ok", "message": "NSE Stock Prediction API v2",
            "models": 13, "stocks": len(TICKERS)}


@app.get("/ui")
def ui():
    idx = os.path.join(FRONTEND, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"error": "Frontend not found"}


@app.get("/results")
def get_results():
    csv = os.path.join(RESULTS_DIR, "all_results.csv")
    if not os.path.exists(csv):
        return []
    df = pd.read_csv(csv)
    df["paradigm"] = df["model"].map(PARADIGM_MAP).fillna("Unknown")
    return df.fillna("").to_dict(orient="records")


@app.get("/models")
def list_models():
    files = []
    if os.path.exists(SAVED_MODELS):
        files = sorted(os.listdir(SAVED_MODELS))
    return {"models": files, "count": len(files)}


@app.get("/summary")
def get_summary():
    """Best model, paradigm breakdown, overall stats."""
    csv = os.path.join(RESULTS_DIR, "all_results.csv")
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv)
    df["paradigm"] = df["model"].map(PARADIGM_MAP).fillna("Unknown")
    clf = df[~df["model"].isin(["ARIMA","WeightedVote"])]

    best_row = clf.loc[clf["predict_score"].idxmax()] if len(clf) else None
    paradigm_best = clf.groupby("paradigm")["predict_score"].max().to_dict()

    return {
        "total_models":    len(df),
        "best_model":      best_row["model"] if best_row is not None else "N/A",
        "best_accuracy":   round(float(best_row["predict_score"]), 4) if best_row is not None else 0,
        "paradigm_best":   paradigm_best,
        "mean_accuracy":   round(float(clf["predict_score"].mean()), 4) if len(clf) else 0,
        "models_above_60": int((clf["predict_score"] > 0.60).sum()),
        "models_above_55": int((clf["predict_score"] > 0.55).sum()),
    }


@app.get("/predict/{ticker}/{model_name}")
def predict(ticker: str, model_name: str):
    """Live prediction: fetch latest data → compute features → run model."""
    nse_ticker = f"{ticker.upper()}.NS" if not ticker.endswith(".NS") else ticker.upper()

    try:
        df = yf.download(nse_ticker, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        df.dropna(inplace=True)
        if len(df) < 50:
            raise HTTPException(400, f"Not enough data for {nse_ticker}")
    except Exception as e:
        raise HTTPException(400, f"Data fetch failed: {e}")

    try:
        nifty = yf.download("^NSEI", period="1y", interval="1d",
                             auto_adjust=True, progress=False)
        df_feat = compute_features(df, nifty)
    except:
        df_feat = compute_features(df, None)

    if len(df_feat) < 1:
        raise HTTPException(400, "Feature engineering failed")

    x_raw = df_feat[FEATURE_COLS].values[-1:].astype(np.float32)

    # Load scaler (use closest ticker or global)
    scalers = get_scalers()
    scaler  = scalers.get(nse_ticker) or scalers.get(list(scalers.keys())[0]) if scalers else None
    if scaler:
        x_scaled = scaler.transform(x_raw)
    else:
        from sklearn.preprocessing import StandardScaler
        x_scaled = x_raw  # no scaler available

    pred = _run_single_model(model_name, x_scaled, x_raw)

    return {
        "ticker":     nse_ticker,
        "model":      model_name,
        "paradigm":   PARADIGM_MAP.get(model_name, "Unknown"),
        "prediction": "UP" if pred == 1 else "DOWN",
        "raw":        int(pred),
        "date":       str(df_feat.index[-1].date()),
        "close":      round(float(df["Close"].squeeze().iloc[-1]), 2),
    }


@app.get("/ensemble/{ticker}")
def ensemble_predict(ticker: str):
    """Run ensemble + all available models for one ticker."""
    nse_ticker = f"{ticker.upper()}.NS" if not ticker.endswith(".NS") else ticker.upper()

    try:
        df = yf.download(nse_ticker, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        nifty = yf.download("^NSEI", period="1y", interval="1d",
                             auto_adjust=True, progress=False)
        df_feat = compute_features(df, nifty)
    except Exception as e:
        raise HTTPException(400, str(e))

    x_raw    = df_feat[FEATURE_COLS].values[-1:].astype(np.float32)
    scalers  = get_scalers()
    scaler   = scalers.get(nse_ticker) or (list(scalers.values())[0] if scalers else None)
    x_scaled = scaler.transform(x_raw) if scaler else x_raw

    individual = {}
    all_preds  = []

    pkl_models = ["XGBoost", "RandomForest", "SVM"]
    pt_flat    = ["DNN", "HybridQNN"]
    pt_seq     = ["LSTM", "GRU", "Transformer", "CNNLSTM", "BiLSTM", "TCN"]

    for name in pkl_models:
        p = _run_single_model(name, x_scaled, x_raw)
        individual[name] = {"prediction": "UP" if p==1 else "DOWN", "raw": int(p)}
        all_preds.append(p)

    for name in pt_flat + pt_seq:
        p = _run_single_model(name, x_scaled, x_raw)
        individual[name] = {"prediction": "UP" if p==1 else "DOWN", "raw": int(p)}
        all_preds.append(p)

    # Majority vote
    vote = 1 if np.mean(all_preds) >= 0.5 else 0
    votes_up   = int(sum(all_preds))
    votes_down = len(all_preds) - votes_up

    # Stacking ensemble
    ens_path = os.path.join(SAVED_MODELS, "StackingEnsemble.pkl")
    ens_pred = vote  # fallback
    if os.path.exists(ens_path):
        try:
            bundle = joblib.load(ens_path)
            meta   = bundle["meta"]
            names  = bundle["model_names"]
            x_meta = np.array([[individual.get(n, {"raw": 0})["raw"] for n in names]])
            ens_pred  = int(meta.predict(x_meta)[0])
            ens_proba = meta.predict_proba(x_meta)[0]
        except:
            ens_proba = [0.5, 0.5]
    else:
        ens_proba = [0.5, 0.5]

    return {
        "ticker":           nse_ticker,
        "date":             str(df_feat.index[-1].date()),
        "close":            round(float(df["Close"].squeeze().iloc[-1]), 2),
        "ensemble_prediction": "UP" if ens_pred == 1 else "DOWN",
        "ensemble_confidence": round(float(max(ens_proba)), 4),
        "vote_prediction":  "UP" if vote == 1 else "DOWN",
        "votes_up":         votes_up,
        "votes_down":       votes_down,
        "total_models":     len(all_preds),
        "individual_models": individual,
    }


@app.get("/feature_importance")
def feature_importance():
    """Return XGBoost feature importances."""
    path = os.path.join(SAVED_MODELS, "XGBoost.pkl")
    if not os.path.exists(path):
        return {}
    model = joblib.load(path)
    imp = model.feature_importances_
    return {
        "features": FEATURE_COLS[:len(imp)],
        "importances": [round(float(v), 6) for v in imp],
    }


def _run_single_model(name: str, x_scaled: np.ndarray, x_raw: np.ndarray) -> int:
    """Load model and predict for one sample. Returns 0 or 1."""
    import torch
    pkl = os.path.join(SAVED_MODELS, f"{name}.pkl")
    pt  = os.path.join(SAVED_MODELS, f"{name}.pt")

    try:
        if os.path.exists(pkl):
            obj = joblib.load(pkl)
            if hasattr(obj, "predict"):
                return int(obj.predict(x_scaled)[0])
            elif isinstance(obj, dict) and "model" in obj:
                m = obj["model"]
                if hasattr(m, "predict"):
                    return int(m.predict(x_scaled)[0])
        elif os.path.exists(pt):
            # Lazy import architectures
            from models.ml_models  import LSTMModel, GRUModel
            from models.dl_models  import (DNNModel, StockTransformer,
                                            CNNLSTMModel, BiLSTMModel, TCNModel)
            from models.qnn_models import HybridQNN

            n = x_raw.shape[1]
            arch_map = {
                "LSTM": (LSTMModel, {"n_feat": n}),
                "GRU":  (GRUModel,  {"n_feat": n}),
                "DNN":  (DNNModel,  {"n_feat": n}),
                "Transformer": (StockTransformer, {"n_feat": n}),
                "CNNLSTM":     (CNNLSTMModel,     {"n_feat": n}),
                "BiLSTM":      (BiLSTMModel,       {"n_feat": n}),
                "TCN":         (TCNModel,           {"n_feat": n}),
                "HybridQNN":   (HybridQNN,          {"n_feat": n}),
            }
            if name in arch_map:
                cls, kwargs = arch_map[name]
                model = cls(**kwargs)
                model.load_state_dict(torch.load(pt, map_location="cpu",
                                                  weights_only=False))
                model.eval()
                with torch.no_grad():
                    inp = torch.tensor(x_scaled, dtype=torch.float32)
                    if name in ["LSTM","GRU","Transformer","CNNLSTM","BiLSTM","TCN"]:
                        inp = inp.unsqueeze(0).repeat(1, SEQUENCE_LEN, 1)
                    return int(model(inp).argmax(1).item())
    except Exception as e:
        pass
    return 0  # default DOWN if model unavailable
