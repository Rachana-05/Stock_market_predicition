# models/ml_models.py — LSTM, GRU, XGBoost, Random Forest, SVM
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from xgboost import XGBClassifier
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *

os.makedirs(SAVED_MODELS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────
# METRICS HELPER
# ────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_train, y_pred_train, name):
    fit_score     = float(accuracy_score(y_train, y_pred_train))
    predict_score = float(accuracy_score(y_true, y_pred))
    f1            = float(f1_score(y_true, y_pred, average="weighted"))
    cm            = confusion_matrix(y_true, y_pred).tolist()
    report        = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "model":         name,
        "fit_score":     round(fit_score, 4),
        "predict_score": round(predict_score, 4),
        "f1":            round(f1, 4),
        "precision":     round(report["weighted avg"]["precision"], 4),
        "recall":        round(report["weighted avg"]["recall"], 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    print(f"  [{name}] fit={fit_score:.4f}  predict={predict_score:.4f}  f1={f1:.4f}")
    return metrics


# ────────────────────────────────────────────────────
# PYTORCH TRAINING LOOP
# ────────────────────────────────────────────────────
def train_torch(model, X_tr, y_tr, X_te, y_te, name, seq=True):
    model = model.to(DEVICE)
    dtype = torch.float32
    Xt = torch.tensor(X_tr, dtype=dtype)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xe = torch.tensor(X_te, dtype=dtype)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val, patience_cnt, best_state = 0, 0, None
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xe.to(DEVICE)).argmax(1).cpu().numpy()
            val_acc = accuracy_score(y_te, val_pred)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING // 5:
                print(f"    Early stop at epoch {epoch+1}")
                break
            if (epoch + 1) % 20 == 0:
                print(f"    [{name}] epoch {epoch+1}  val_acc={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    train_time = round(time.time() - t0, 1)

    # Final predictions
    model.eval()
    with torch.no_grad():
        y_pred_tr = model(Xt.to(DEVICE)).argmax(1).cpu().numpy()
        y_pred_te = model(Xe.to(DEVICE)).argmax(1).cpu().numpy()

    metrics = compute_metrics(y_te, y_pred_te, y_tr, y_pred_tr, name)
    metrics["train_time_s"] = train_time

    path = os.path.join(SAVED_MODELS, f"{name}.pt")
    torch.save(model.state_dict(), path)
    return metrics, path, y_pred_te


# ────────────────────────────────────────────────────
# LSTM
# ────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_feat=45, hidden=256, layers=3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=0.3, bidirectional=False)
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.fc(self.drop(out))


def run_lstm(X_tr_seq, y_tr, X_te_seq, y_te):
    print("\n--- LSTM ---")
    model = LSTMModel(n_feat=X_tr_seq.shape[2])
    return train_torch(model, X_tr_seq, y_tr, X_te_seq, y_te, "LSTM")


# ────────────────────────────────────────────────────
# GRU
# ────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, n_feat=45, hidden=256, layers=3):
        super().__init__()
        self.gru  = nn.GRU(n_feat, hidden, layers, batch_first=True, dropout=0.3)
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.bn(out[:, -1, :])
        return self.fc(self.drop(out))


def run_gru(X_tr_seq, y_tr, X_te_seq, y_te):
    print("\n--- GRU ---")
    model = GRUModel(n_feat=X_tr_seq.shape[2])
    return train_torch(model, X_tr_seq, y_tr, X_te_seq, y_te, "GRU")


# ────────────────────────────────────────────────────
# XGBOOST
# ────────────────────────────────────────────────────
def run_xgboost(X_tr, y_tr, X_te, y_te):
    print("\n--- XGBoost ---")
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=RANDOM_SEED, verbosity=0, eval_metric="logloss",
        early_stopping_rounds=20
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    metrics = compute_metrics(y_te, y_pred_te, y_tr, y_pred_tr, "XGBoost")
    metrics["train_time_s"] = round(time.time() - t0, 1)
    path = os.path.join(SAVED_MODELS, "XGBoost.pkl")
    joblib.dump(model, path)
    return metrics, path, y_pred_te


# ────────────────────────────────────────────────────
# RANDOM FOREST
# ────────────────────────────────────────────────────
def run_random_forest(X_tr, y_tr, X_te, y_te):
    print("\n--- Random Forest ---")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED
    )
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    metrics = compute_metrics(y_te, y_pred_te, y_tr, y_pred_tr, "RandomForest")
    metrics["train_time_s"] = round(time.time() - t0, 1)
    path = os.path.join(SAVED_MODELS, "RandomForest.pkl")
    joblib.dump(model, path)
    return metrics, path, y_pred_te


# ────────────────────────────────────────────────────
# SVM
# ────────────────────────────────────────────────────
def run_svm(X_tr, y_tr, X_te, y_te):
    print("\n--- SVM ---")
    t0 = time.time()
    # Subsample for speed — SVM is O(n²)
    n_sample = min(4000, len(X_tr))
    idx = np.random.choice(len(X_tr), n_sample, replace=False)
    model = SVC(kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=RANDOM_SEED)
    model.fit(X_tr[idx], y_tr[idx])
    y_pred_tr = model.predict(X_tr[idx])
    y_pred_te = model.predict(X_te)
    metrics = compute_metrics(y_te, y_pred_te, y_tr[idx], y_pred_tr, "SVM")
    metrics["train_time_s"] = round(time.time() - t0, 1)
    path = os.path.join(SAVED_MODELS, "SVM.pkl")
    joblib.dump(model, path)
    return metrics, path, y_pred_te
