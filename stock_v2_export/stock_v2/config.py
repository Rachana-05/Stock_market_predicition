# config.py — central configuration for all modules
import os

# ── Stocks (10 NSE large-caps for universal model)
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS",
    "SBIN.NS", "HINDUNILVR.NS"
]
BENCHMARK = "^NSEI"          # Nifty50 index for relative features

# ── Data
PERIOD          = "5y"        # 8 years daily data
INTERVAL        = "1d"
PREDICT_TARGET  = "direction" # next-day UP/DOWN classification
RANDOM_SEED     = 42

# ── Walk-forward validation
WF_TRAIN_DAYS   = 365         # rolling 1-year train window
WF_TEST_DAYS    = 30          # predict 30 days at a time
WF_MIN_FOLDS    = 3           # minimum folds to run

# ── Feature engineering
SEQUENCE_LEN    = 30          # lookback window for recurrent models
N_FEATURES      = 45          # total features per day

# ── Quantum settings
N_QUBITS        = 6           # 6 qubits as approved
N_LAYERS        = 2           # variational layers

# ── Training
EPOCHS          = 80
BATCH_SIZE      = 64
LEARNING_RATE   = 5e-4
EARLY_STOPPING  = 15          # patience epochs

# ── Confidence threshold for ensemble
CONFIDENCE_THRESHOLD = 0.58   # only report predictions above this

# ── Paths
BASE_DIR        = "C:/Users/racha/Downloads/stock_v2_export/stock_v2"
SAVED_MODELS    = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
DATA_DIR        = os.path.join(BASE_DIR, "data_cache")
MLFLOW_URI      = f"file:///{BASE_DIR}/mlruns"
EXPERIMENT_NAME = "stock_prediction_v2"

# ── MLflow
MLFLOW_TRACKING_URI = MLFLOW_URI

for d in [SAVED_MODELS, RESULTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
