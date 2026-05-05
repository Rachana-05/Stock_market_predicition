# models/ensemble.py — stacking ensemble + weighted vote + confidence
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *
from models.ml_models import compute_metrics

os.makedirs(SAVED_MODELS, exist_ok=True)


def run_stacking_ensemble(model_preds_train: dict, y_train: np.ndarray,
                           model_preds_test:  dict, y_test:  np.ndarray,
                           model_probas_test: dict = None):
    """
    Stacking ensemble:
    - Input: predictions from all 13 models on train and test
    - Meta-learner: Logistic Regression
    - Output: final prediction + confidence + per-model weights

    Also returns:
    - weighted_vote prediction (simple majority)
    - confidence-filtered prediction (only when confidence >= threshold)
    """
    print("\n--- Stacking Ensemble (all 13 models) ---")
    t0 = time.time()

    model_names = list(model_preds_train.keys())
    print(f"  Base models: {model_names}")

    # ── Build meta-features
    X_meta_tr = np.stack([model_preds_train[m] for m in model_names], axis=1).astype(float)
    X_meta_te = np.stack([model_preds_test[m]  for m in model_names], axis=1).astype(float)

    # ── Meta-learner
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED,
                               solver="lbfgs")
    meta.fit(X_meta_tr, y_train)

    y_pred_tr  = meta.predict(X_meta_tr)
    y_pred_te  = meta.predict(X_meta_te)
    y_proba_te = meta.predict_proba(X_meta_te)

    # ── Model weights (coefficients)
    weights = dict(zip(model_names, meta.coef_[0].tolist()))

    # ── Confidence
    confidence = np.max(y_proba_te, axis=1)
    high_conf_mask = confidence >= CONFIDENCE_THRESHOLD
    if high_conf_mask.sum() >= 10:
        acc_conf = float(accuracy_score(y_test[high_conf_mask], y_pred_te[high_conf_mask]))
        coverage = float(high_conf_mask.mean())
    else:
        acc_conf = float(accuracy_score(y_test, y_pred_te))
        coverage = 1.0

    metrics = compute_metrics(y_test, y_pred_te, y_train, y_pred_tr, "StackingEnsemble")
    metrics["train_time_s"]          = round(time.time() - t0, 1)
    metrics["base_models"]           = model_names
    metrics["model_weights"]         = weights
    metrics["confidence_threshold"]  = CONFIDENCE_THRESHOLD
    metrics["high_conf_accuracy"]    = round(acc_conf, 4)
    metrics["coverage"]              = round(coverage, 4)
    metrics["mean_confidence"]       = round(float(confidence.mean()), 4)

    # ── Weighted vote
    vote_pred = _weighted_vote(model_preds_test, y_test)

    # ── Save
    path = os.path.join(SAVED_MODELS, "StackingEnsemble.pkl")
    joblib.dump({"meta": meta, "model_names": model_names,
                  "weights": weights, "threshold": CONFIDENCE_THRESHOLD}, path)

    print(f"  Stacking accuracy:          {metrics['predict_score']:.4f}")
    print(f"  High-confidence accuracy:   {acc_conf:.4f} (coverage {coverage*100:.1f}%)")
    print(f"  Weighted vote accuracy:     {vote_pred['accuracy']:.4f}")

    return metrics, path, y_pred_te, y_proba_te, vote_pred


def _weighted_vote(model_preds_test: dict, y_test: np.ndarray) -> dict:
    """Simple majority vote across all models."""
    preds = np.stack(list(model_preds_test.values()), axis=1)
    vote  = (preds.mean(axis=1) >= 0.5).astype(int)
    acc   = float(accuracy_score(y_test, vote))
    return {
        "predictions": vote,
        "accuracy":    round(acc, 4),
        "name":        "WeightedVote"
    }


def predict_single_day(features_flat: np.ndarray, scaler,
                        selected_model_name: str = None) -> dict:
    """
    Run ensemble + individual model prediction on a single day's features.
    Used by FastAPI for live /predict endpoint.
    """
    x = scaler.transform(features_flat.reshape(1, -1))

    results = {}
    ensemble_path = os.path.join(SAVED_MODELS, "StackingEnsemble.pkl")

    if os.path.exists(ensemble_path):
        bundle = joblib.load(ensemble_path)
        meta   = bundle["meta"]
        names  = bundle["model_names"]

        # Get predictions from all available base models
        base_preds = {}
        for name in names:
            pred = _load_and_predict(name, x, features_flat)
            if pred is not None:
                base_preds[name] = np.array([pred])

        if base_preds:
            X_meta = np.array([[base_preds.get(n, [0.5])[0] for n in names]])
            ens_pred  = int(meta.predict(X_meta)[0])
            ens_proba = meta.predict_proba(X_meta)[0]
            results["ensemble"] = {
                "prediction":  "UP" if ens_pred == 1 else "DOWN",
                "confidence":  round(float(ens_proba.max()), 4),
                "raw":         ens_pred,
                "base_preds":  {k: ("UP" if v[0]==1 else "DOWN") for k, v in base_preds.items()}
            }

    # Selected model individually
    if selected_model_name:
        pred = _load_and_predict(selected_model_name, x, features_flat)
        if pred is not None:
            results["selected"] = {
                "model":      selected_model_name,
                "prediction": "UP" if pred == 1 else "DOWN",
                "raw":        int(pred)
            }

    return results


def _load_and_predict(name: str, x_scaled: np.ndarray,
                       x_raw: np.ndarray) -> int:
    """Load a saved model and return prediction for one sample."""
    import torch
    try:
        pkl_path = os.path.join(SAVED_MODELS, f"{name}.pkl")
        pt_path  = os.path.join(SAVED_MODELS, f"{name}.pt")

        if os.path.exists(pkl_path):
            obj = joblib.load(pkl_path)
            if hasattr(obj, "predict"):
                return int(obj.predict(x_scaled)[0])
            elif isinstance(obj, dict) and "meta" in obj:
                return None
            elif isinstance(obj, dict) and "model" in obj:
                m = obj["model"]
                if hasattr(m, "predict"):
                    # QSVM — needs kernel
                    return None  # skip for live prediction
        elif os.path.exists(pt_path):
            # PyTorch model — need architecture
            return None  # architecture not available here
    except:
        pass
    return None
