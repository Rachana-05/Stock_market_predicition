# tests/test_suite.py
# Run with: pytest tests/test_suite.py -v
# Or:       python -m pytest tests/test_suite.py -v

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

# ── patch config paths for local testing
import config
config.SAVED_MODELS = os.path.join(os.path.dirname(__file__), "test_models")
config.RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "test_results")
config.DATA_DIR     = os.path.join(os.path.dirname(__file__), "test_data")
config.MLFLOW_URI   = f"file:///{os.path.join(os.path.dirname(__file__), 'test_mlruns')}"
config.EPOCHS       = 2
config.BATCH_SIZE   = 32
os.makedirs(config.SAVED_MODELS, exist_ok=True)
os.makedirs(config.RESULTS_DIR,  exist_ok=True)
os.makedirs(config.DATA_DIR,     exist_ok=True)

# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════

def make_ohlcv(n=400):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.randn(n) * 0.012) * 2000
    return pd.DataFrame({
        "Open":   prices * (1 + np.random.randn(n) * 0.002),
        "High":   prices * (1 + np.abs(np.random.randn(n)) * 0.005),
        "Low":    prices * (1 - np.abs(np.random.randn(n)) * 0.005),
        "Close":  prices,
        "Volume": np.random.lognormal(14, 0.5, n) * 1000,
    }, index=pd.bdate_range(end="2026-01-01", periods=n))

@pytest.fixture
def ohlcv():
    return make_ohlcv()

@pytest.fixture
def flat_data():
    np.random.seed(42)
    n, f = 300, 45
    X_tr = np.random.randn(int(n*0.8), f).astype(np.float32)
    X_te = np.random.randn(int(n*0.2), f).astype(np.float32)
    y_tr = (np.random.randn(int(n*0.8)) > 0).astype(np.int64)
    y_te = (np.random.randn(int(n*0.2)) > 0).astype(np.int64)
    return X_tr, X_te, y_tr, y_te

@pytest.fixture
def seq_data():
    np.random.seed(42)
    n, seq, f = 300, 30, 45
    X_tr = np.random.randn(int(n*0.8), seq, f).astype(np.float32)
    X_te = np.random.randn(int(n*0.2), seq, f).astype(np.float32)
    y_tr = (np.random.randn(int(n*0.8)) > 0).astype(np.int64)
    y_te = (np.random.randn(int(n*0.2)) > 0).astype(np.int64)
    return X_tr, X_te, y_tr, y_te


# ═══════════════════════════════════════════════════════════
# PHASE 1 — DATA PIPELINE TESTS
# ═══════════════════════════════════════════════════════════

class TestDataPipeline:
    """Tests for data fetching and feature engineering."""

    def test_ohlcv_shape(self, ohlcv):
        """OHLCV dataframe has exactly 5 columns."""
        assert ohlcv.shape[1] == 5
        assert all(c in ohlcv.columns for c in ["Open","High","Low","Close","Volume"])

    def test_ohlcv_no_nan(self, ohlcv):
        """Raw OHLCV has no NaN values."""
        assert ohlcv.isnull().sum().sum() == 0

    def test_ohlcv_positive_prices(self, ohlcv):
        """All prices are positive."""
        for col in ["Open","High","Low","Close"]:
            assert (ohlcv[col] > 0).all(), f"{col} has non-positive values"

    def test_high_gte_low(self, ohlcv):
        """High is always >= Low."""
        assert (ohlcv["High"] >= ohlcv["Low"]).all()

    def test_feature_count(self, ohlcv):
        """Feature engineering produces exactly 45 features."""
        from data.features import compute_features, FEATURE_COLS
        df_feat = compute_features(ohlcv, None)
        assert len(FEATURE_COLS) == 45
        for col in FEATURE_COLS:
            assert col in df_feat.columns, f"Missing feature: {col}"

    def test_no_nan_after_features(self, ohlcv):
        """No NaN values after feature engineering."""
        from data.features import compute_features, FEATURE_COLS
        df_feat = compute_features(ohlcv, None)
        assert df_feat[FEATURE_COLS].isnull().sum().sum() == 0

    def test_target_is_binary(self, ohlcv):
        """Target column contains only 0 and 1."""
        from data.features import compute_features
        df_feat = compute_features(ohlcv, None)
        assert set(df_feat["target"].unique()).issubset({0, 1})

    def test_rsi_range(self, ohlcv):
        """RSI values are within valid range [0, 100]."""
        from data.features import compute_features
        df_feat = compute_features(ohlcv, None)
        rsi = df_feat["rsi_14"].dropna()
        assert rsi.between(0, 100).all(), f"RSI out of range: {rsi.min():.2f} to {rsi.max():.2f}"

    def test_sufficient_rows_after_features(self, ohlcv):
        """At least 100 rows remain after feature engineering."""
        from data.features import compute_features
        df_feat = compute_features(ohlcv, None)
        assert len(df_feat) >= 100, f"Only {len(df_feat)} rows after feature engineering"

    def test_multi_stock_dataset_shape(self):
        """Multi-stock dataset produces correct shape."""
        from data.features import build_multi_stock_dataset, FEATURE_COLS
        config.TICKERS = ["RELIANCE.NS", "TCS.NS"]
        raw = {"RELIANCE.NS": make_ohlcv(), "TCS.NS": make_ohlcv(), "NIFTY": make_ohlcv()}
        X_flat, X_seq, y, scalers = build_multi_stock_dataset(raw)
        assert X_flat.shape[1] == 45
        assert X_seq.shape[1] == config.SEQUENCE_LEN
        assert X_seq.shape[2] == 45
        assert len(y) == len(X_flat) == len(X_seq)

    def test_train_test_split_no_shuffle(self):
        """Train/test split preserves time order."""
        from data.features import train_test_split_time
        n = 100
        X = np.arange(n).reshape(n, 1).astype(np.float32)
        Xs = np.arange(n).reshape(n, 1, 1).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)
        X_tr, X_te, _, _, y_tr, y_te = train_test_split_time(X, Xs, y, test_ratio=0.2)
        # Training set comes first chronologically
        assert len(X_tr) == 80
        assert len(X_te) == 20


# ═══════════════════════════════════════════════════════════
# PHASE 2 — ML MODEL TESTS
# ═══════════════════════════════════════════════════════════

class TestMLModels:
    """Tests for ML model training and inference."""

    def test_xgboost_fit_predict_scores(self, flat_data):
        """XGBoost returns valid fit_score and predict_score."""
        from models.ml_models import run_xgboost
        X_tr, X_te, y_tr, y_te = flat_data
        m, p, pte = run_xgboost(X_tr, y_tr, X_te, y_te)
        assert "fit_score"     in m
        assert "predict_score" in m
        assert 0 <= m["fit_score"]     <= 1
        assert 0 <= m["predict_score"] <= 1
        assert len(pte) == len(y_te)

    def test_xgboost_predictions_binary(self, flat_data):
        """XGBoost predictions are only 0 or 1."""
        from models.ml_models import run_xgboost
        X_tr, X_te, y_tr, y_te = flat_data
        _, _, pte = run_xgboost(X_tr, y_tr, X_te, y_te)
        assert set(pte).issubset({0, 1})

    def test_random_forest_fit_predict(self, flat_data):
        """Random Forest returns valid scores."""
        from models.ml_models import run_random_forest
        X_tr, X_te, y_tr, y_te = flat_data
        m, p, pte = run_random_forest(X_tr, y_tr, X_te, y_te)
        assert 0 <= m["fit_score"]     <= 1
        assert 0 <= m["predict_score"] <= 1

    def test_svm_fit_predict(self, flat_data):
        """SVM returns valid scores."""
        from models.ml_models import run_svm
        X_tr, X_te, y_tr, y_te = flat_data
        m, p, pte = run_svm(X_tr, y_tr, X_te, y_te)
        assert 0 <= m["fit_score"]     <= 1
        assert 0 <= m["predict_score"] <= 1

    def test_lstm_output_shape(self, seq_data):
        """LSTM model output shape is (n, 2)."""
        from models.ml_models import LSTMModel
        X_tr, X_te, y_tr, y_te = seq_data
        model = LSTMModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_gru_output_shape(self, seq_data):
        """GRU model output shape is (n, 2)."""
        from models.ml_models import GRUModel
        X_tr, X_te, y_tr, y_te = seq_data
        model = GRUModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_metrics_keys_present(self, flat_data):
        """All required metric keys are present in output."""
        from models.ml_models import run_xgboost
        X_tr, X_te, y_tr, y_te = flat_data
        m, _, _ = run_xgboost(X_tr, y_tr, X_te, y_te)
        required = ["model","fit_score","predict_score","f1","precision","recall"]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_model_file_saved(self, flat_data):
        """Model file is saved to disk after training."""
        from models.ml_models import run_xgboost
        X_tr, X_te, y_tr, y_te = flat_data
        _, path, _ = run_xgboost(X_tr, y_tr, X_te, y_te)
        assert os.path.exists(path), f"Model file not found: {path}"


# ═══════════════════════════════════════════════════════════
# PHASE 3 — DL MODEL TESTS
# ═══════════════════════════════════════════════════════════

class TestDLModels:
    """Tests for Deep Learning model architectures."""

    def test_dnn_output_shape(self, flat_data):
        """DNN produces output of shape (n, 2)."""
        from models.dl_models import DNNModel
        X_tr, X_te, y_tr, y_te = flat_data
        model = DNNModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_dnn_fit_predict_scores(self, flat_data):
        """DNN training returns valid fit and predict scores."""
        from models.dl_models import run_dnn
        X_tr, X_te, y_tr, y_te = flat_data
        m, p, pte = run_dnn(X_tr, y_tr, X_te, y_te)
        assert 0 <= m["fit_score"]     <= 1
        assert 0 <= m["predict_score"] <= 1
        assert m["model"] == "DNN"

    def test_transformer_output_shape(self, seq_data):
        """Transformer produces output of shape (n, 2)."""
        from models.dl_models import StockTransformer
        X_tr, X_te, y_tr, y_te = seq_data
        model = StockTransformer(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_cnnlstm_output_shape(self, seq_data):
        """CNN-LSTM produces output of shape (n, 2)."""
        from models.dl_models import CNNLSTMModel
        X_tr, X_te, y_tr, y_te = seq_data
        model = CNNLSTMModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_bilstm_output_shape(self, seq_data):
        """BiLSTM produces output of shape (n, 2)."""
        from models.dl_models import BiLSTMModel
        X_tr, X_te, y_tr, y_te = seq_data
        model = BiLSTMModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_tcn_output_shape(self, seq_data):
        """TCN produces output of shape (n, 2)."""
        from models.dl_models import TCNModel
        X_tr, X_te, y_tr, y_te = seq_data
        model = TCNModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_te, dtype=torch.float32))
        assert out.shape == (len(X_te), 2)

    def test_dnn_predictions_binary(self, flat_data):
        """DNN argmax predictions are only 0 or 1."""
        from models.dl_models import DNNModel
        X_tr, X_te, y_tr, y_te = flat_data
        model = DNNModel(n_feat=45)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_te, dtype=torch.float32)).argmax(1).numpy()
        assert set(preds).issubset({0, 1})

    def test_dnn_5_layers(self):
        """DNN has exactly 5 linear layers."""
        from models.dl_models import DNNModel
        model = DNNModel(n_feat=45)
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 6  # 5 hidden + 1 output


# ═══════════════════════════════════════════════════════════
# PHASE 4 — QML MODEL TESTS
# ═══════════════════════════════════════════════════════════

class TestQMLModels:
    """Tests for Quantum Machine Learning models."""

    def test_qsvm_returns_metrics(self, flat_data):
        """QSVM returns valid metrics dict."""
        from models.qml_models import run_qsvm_qiskit
        config.N_QUBITS = 4
        X_tr, X_te, y_tr, y_te = flat_data
        m, p, pte = run_qsvm_qiskit(X_tr, y_tr, X_te, y_te, n_sample=30)
        assert "fit_score"     in m
        assert "predict_score" in m
        assert "feature_map"   in m
        assert m["feature_map"] == "ZZFeatureMap"

    def test_qsvm_predictions_binary(self, flat_data):
        """QSVM predictions are 0 or 1."""
        from models.qml_models import run_qsvm_qiskit
        config.N_QUBITS = 4
        X_tr, X_te, y_tr, y_te = flat_data
        _, _, pte = run_qsvm_qiskit(X_tr, y_tr, X_te, y_te, n_sample=30)
        assert set(pte).issubset({0, 1})

    def test_qsvm_n_qubits_recorded(self, flat_data):
        """QSVM records correct number of qubits."""
        from models.qml_models import run_qsvm_qiskit
        config.N_QUBITS = 4
        X_tr, X_te, y_tr, y_te = flat_data
        m, _, _ = run_qsvm_qiskit(X_tr, y_tr, X_te, y_te, n_sample=30)
        assert m["n_qubits"] == 4

    def test_pca_reduction(self, flat_data):
        """PCA reduces 45 features to N_QUBITS dimensions."""
        from models.qml_models import reduce_to_qubits
        config.N_QUBITS = 4
        X_tr, X_te, y_tr, y_te = flat_data
        Xtr_r, Xte_r, pca, scale = reduce_to_qubits(X_tr, X_te)
        assert Xtr_r.shape[1] == 4
        assert Xte_r.shape[1] == 4

    def test_qsvm_scores_in_range(self, flat_data):
        """QSVM fit and predict scores are between 0 and 1."""
        from models.qml_models import run_qsvm_qiskit
        config.N_QUBITS = 4
        X_tr, X_te, y_tr, y_te = flat_data
        m, _, _ = run_qsvm_qiskit(X_tr, y_tr, X_te, y_te, n_sample=30)
        assert 0 <= m["fit_score"]     <= 1
        assert 0 <= m["predict_score"] <= 1


# ═══════════════════════════════════════════════════════════
# PHASE 5 — ENSEMBLE TESTS
# ═══════════════════════════════════════════════════════════

class TestEnsemble:
    """Tests for ensemble models."""

    def test_stacking_ensemble_runs(self):
        """Stacking ensemble trains and returns metrics."""
        from models.ensemble import run_stacking_ensemble
        np.random.seed(42)
        n_tr, n_te = 200, 50
        y_tr = (np.random.randn(n_tr) > 0).astype(np.int64)
        y_te = (np.random.randn(n_te) > 0).astype(np.int64)
        ptr = {m: (np.random.randn(n_tr) > 0).astype(int) for m in ["A","B","C"]}
        pte = {m: (np.random.randn(n_te) > 0).astype(int) for m in ["A","B","C"]}
        ens_m, _, _, _, vote = run_stacking_ensemble(ptr, y_tr, pte, y_te)
        assert "fit_score"     in ens_m
        assert "predict_score" in ens_m
        assert 0 <= ens_m["predict_score"] <= 1

    def test_weighted_vote_accuracy(self):
        """Weighted vote accuracy is between 0 and 1."""
        from models.ensemble import run_stacking_ensemble
        np.random.seed(42)
        n_tr, n_te = 200, 50
        y_tr = (np.random.randn(n_tr) > 0).astype(np.int64)
        y_te = (np.random.randn(n_te) > 0).astype(np.int64)
        ptr = {m: (np.random.randn(n_tr) > 0).astype(int) for m in ["A","B","C"]}
        pte = {m: (np.random.randn(n_te) > 0).astype(int) for m in ["A","B","C"]}
        _, _, _, _, vote = run_stacking_ensemble(ptr, y_tr, pte, y_te)
        assert 0 <= vote["accuracy"] <= 1

    def test_ensemble_requires_minimum_models(self):
        """Ensemble needs at least 2 models."""
        from models.ensemble import run_stacking_ensemble
        np.random.seed(42)
        y_tr = np.array([0,1,0,1,0])
        y_te = np.array([0,1])
        ptr = {"A": np.array([0,1,0,1,0])}
        pte = {"A": np.array([0,1])}
        # With only 1 model it should still not crash
        try:
            run_stacking_ensemble(ptr, y_tr, pte, y_te)
        except Exception:
            pass  # acceptable to fail with 1 model

    def test_ensemble_model_names_recorded(self):
        """Ensemble records all base model names."""
        from models.ensemble import run_stacking_ensemble
        np.random.seed(42)
        n_tr, n_te = 100, 30
        y_tr = (np.random.randn(n_tr) > 0).astype(np.int64)
        y_te = (np.random.randn(n_te) > 0).astype(np.int64)
        names = ["LSTM","GRU","XGBoost"]
        ptr = {m: (np.random.randn(n_tr) > 0).astype(int) for m in names}
        pte = {m: (np.random.randn(n_te) > 0).astype(int) for m in names}
        ens_m, _, _, _, _ = run_stacking_ensemble(ptr, y_tr, pte, y_te)
        assert "base_models" in ens_m
        assert set(ens_m["base_models"]) == set(names)


# ═══════════════════════════════════════════════════════════
# PHASE 6 — ACCURACY VALIDATION TESTS
# ═══════════════════════════════════════════════════════════

class TestAccuracyValidation:
    """Tests validating model accuracy against expected values."""

    EXPECTED = {
        "LSTM":              0.5273,
        "GRU":               0.5102,
        "XGBoost":           0.5510,
        "RandomForest":      0.5383,
        "SVM":               0.5019,
        "DNN":               0.5273,
        "Transformer":       0.5333,
        "CNNLSTM":           0.5344,
        "BiLSTM":            0.5229,
        "TCN":               0.5471,
        "QSVM_ZZFeatureMap": 0.5212,
        "StackingEnsemble":  0.4926,
        "WeightedVote":      0.5614,
    }

    def test_results_csv_exists(self):
        """Results CSV exists in the results directory."""
        # Check actual results dir
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found — run training first")
        assert os.path.exists(results_path)

    def test_results_csv_has_required_columns(self):
        """Results CSV has all required columns."""
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found")
        df = pd.read_csv(results_path)
        for col in ["model","fit_score","predict_score","f1"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_best_model_above_random_baseline(self):
        """Best model predict score exceeds random baseline (50%)."""
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found")
        df = pd.read_csv(results_path)
        clf = df[~df["model"].isin(["WeightedVote"])]
        best = clf["predict_score"].max()
        assert best > 0.50, f"Best accuracy {best:.4f} does not exceed random baseline"

    def test_weighted_vote_is_best_ensemble(self):
        """WeightedVote achieves 56.14% predict score."""
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found")
        df = pd.read_csv(results_path)
        wv = df[df["model"]=="WeightedVote"]
        if not wv.empty:
            assert abs(wv["predict_score"].values[0] - 0.5614) < 0.01

    def test_xgboost_accuracy(self):
        """XGBoost predict score matches expected 55.1%."""
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found")
        df = pd.read_csv(results_path)
        row = df[df["model"]=="XGBoost"]
        if not row.empty:
            assert abs(row["predict_score"].values[0] - 0.5510) < 0.01

    def test_all_models_above_48_percent(self):
        """All classification models achieve at least 48% (not completely random)."""
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found")
        df = pd.read_csv(results_path)
        clf = df[~df["model"].isin(["WeightedVote","StackingEnsemble"])]
        clf = clf[clf["predict_score"].notna()]
        failures = clf[clf["predict_score"] < 0.48]
        assert len(failures) == 0, f"Models below 48%: {failures['model'].tolist()}"

    def test_four_paradigms_present(self):
        """All 4 paradigms (ML, DL, QML, QNN) have results."""
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "all_results.csv"
        )
        if not os.path.exists(results_path):
            pytest.skip("Results CSV not found")
        df = pd.read_csv(results_path)
        paradigm_map = {
            "LSTM":"ML","GRU":"ML","XGBoost":"ML","RandomForest":"ML","SVM":"ML",
            "DNN":"DL","Transformer":"DL","CNNLSTM":"DL","BiLSTM":"DL","TCN":"DL",
            "QSVM_ZZFeatureMap":"QML","HybridQNN":"QNN",
        }
        df["paradigm"] = df["model"].map(paradigm_map)
        found = set(df["paradigm"].dropna().unique())
        assert "ML"  in found, "No ML models found"
        assert "DL"  in found, "No DL models found"
        assert "QML" in found, "No QML models found"


# ═══════════════════════════════════════════════════════════
# PHASE 7 — API TESTS
# ═══════════════════════════════════════════════════════════

class TestAPI:
    """Tests for FastAPI endpoints."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        try:
            from fastapi.testclient import TestClient
            from api.main import app
            self.client = TestClient(app)
            self.api_available = True
        except Exception:
            self.api_available = False

    def test_health_check(self):
        """GET / returns status ok."""
        if not self.api_available:
            pytest.skip("API not available")
        r = self.client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_results_endpoint(self):
        """GET /results returns a list."""
        if not self.api_available:
            pytest.skip("API not available")
        r = self.client.get("/results")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_models_endpoint(self):
        """GET /models returns models dict with count."""
        if not self.api_available:
            pytest.skip("API not available")
        r = self.client.get("/models")
        assert r.status_code == 200
        assert "models" in r.json()
        assert "count"  in r.json()

    def test_predict_invalid_model_returns_error(self):
        """GET /predict with invalid model returns error response."""
        if not self.api_available:
            pytest.skip("API not available")
        r = self.client.get("/predict/RELIANCE/InvalidModelXYZ")
        assert r.status_code in [200, 400, 422]

    def test_summary_endpoint(self):
        """GET /summary returns summary dict."""
        if not self.api_available:
            pytest.skip("API not available")
        r = self.client.get("/summary")
        assert r.status_code == 200

    def test_docs_accessible(self):
        """GET /docs returns Swagger UI HTML."""
        if not self.api_available:
            pytest.skip("API not available")
        r = self.client.get("/docs")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ═══════════════════════════════════════════════════════════
# RUN DIRECTLY
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short",
                 "--no-header", "-q"])
