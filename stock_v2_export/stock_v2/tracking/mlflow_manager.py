# tracking/mlflow_manager.py — MLflow experiment tracking
import os, json, warnings
warnings.filterwarnings("ignore")
import mlflow
import mlflow.sklearn
import numpy as np
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

PARADIGM_MAP = {
    "LSTM": "ML", "GRU": "ML", "XGBoost": "ML",
    "RandomForest": "ML", "SVM": "ML",
    "DNN": "DL", "Transformer": "DL", "CNNLSTM": "DL",
    "BiLSTM": "DL", "TCN": "DL",
    "QSVM_ZZFeatureMap": "QML", "VQC_ZZFeatureMap": "QML",
    "HybridQNN": "QNN",
    "StackingEnsemble": "Ensemble", "WeightedVote": "Ensemble",
}


def log_run(metrics: dict, artifact_path: str = None,
            fold: int = None, extra_tags: dict = None):
    """Log one model training run to MLflow."""
    model_name = metrics.get("model", "Unknown")
    run_name   = f"{model_name}_fold{fold}" if fold else model_name

    with mlflow.start_run(run_name=run_name):
        # ── Tags
        mlflow.set_tag("model",    model_name)
        mlflow.set_tag("paradigm", PARADIGM_MAP.get(model_name, "Unknown"))
        if fold is not None:
            mlflow.set_tag("fold", str(fold))
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, str(v))

        # ── Parameters
        mlflow.log_params({
            "epochs":       EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr":           LEARNING_RATE,
            "seq_len":      SEQUENCE_LEN,
            "n_features":   N_FEATURES,
            "n_qubits":     N_QUBITS,
            "n_layers":     N_LAYERS,
            "target":       PREDICT_TARGET,
            "n_stocks":     len(TICKERS),
        })

        # ── Core metrics (professor focus)
        for key in ["fit_score", "predict_score", "f1", "precision",
                    "recall", "train_time_s", "high_conf_accuracy",
                    "coverage", "mean_confidence"]:
            if key in metrics and isinstance(metrics[key], (int, float)):
                mlflow.log_metric(key, metrics[key])

        # ── Quantum-specific
        for key in ["n_qubits", "n_layers"]:
            if key in metrics:
                mlflow.log_metric(key, metrics[key])

        # ── Confusion matrix as JSON artifact
        if "confusion_matrix" in metrics:
            cm_path = f"/tmp/{model_name}_cm.json"
            with open(cm_path, "w") as f:
                json.dump({"confusion_matrix": metrics["confusion_matrix"],
                            "classification_report": metrics.get("classification_report", {})}, f, indent=2)
            mlflow.log_artifact(cm_path)

        # ── Model file
        if artifact_path and os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)

    print(f"  [MLflow] {run_name}  "
          f"fit={metrics.get('fit_score','?'):.4f}  "
          f"predict={metrics.get('predict_score','?'):.4f}")
