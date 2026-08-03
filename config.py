# config.py — V2 Pipeline Configuration
# Probabilistic EPF: PJM + ERCOT (2019–2025)
# ============================================================

import os
import numpy as np

# ── Reproducibility ──────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Directory Layout ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for d in [RAW_DIR, PROC_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Raw Data Paths ────────────────────────────────────────────
PJM_RAW_PATH         = os.path.join(RAW_DIR, "pjm_da_lmp.parquet")
ERCOT_RAW_PATH       = os.path.join(RAW_DIR, "ercot_da_spp.parquet")
WEATHER_PJM_PATH     = os.path.join(RAW_DIR, "weather_pjm.parquet")
WEATHER_ERCOT_PATH   = os.path.join(RAW_DIR, "weather_ercot.parquet")
EIA_GEN_PATH         = os.path.join(RAW_DIR, "eia_generation.parquet")
EIA_GEN_ERCOT_PATH   = os.path.join(RAW_DIR, "eia_generation_ercot.parquet")
EIA_GAS_PATH         = os.path.join(RAW_DIR, "eia_gas_price.parquet")

# ── Processed Data Paths ──────────────────────────────────────
PJM_TRAIN_PATH = os.path.join(PROC_DIR, "pjm_train.parquet")
PJM_CAL_PATH   = os.path.join(PROC_DIR, "pjm_calibration.parquet")
PJM_VAL_PATH   = os.path.join(PROC_DIR, "pjm_val.parquet")
PJM_TEST_PATH  = os.path.join(PROC_DIR, "pjm_test.parquet")

ERCOT_TRAIN_PATH = os.path.join(PROC_DIR, "ercot_train.parquet")
ERCOT_CAL_PATH   = os.path.join(PROC_DIR, "ercot_calibration.parquet")
ERCOT_VAL_PATH   = os.path.join(PROC_DIR, "ercot_val.parquet")
ERCOT_TEST_PATH  = os.path.join(PROC_DIR, "ercot_test.parquet")

# ── Target & Date Configuration ───────────────────────────────
TARGET_COL       = "price"       # Column name after preprocessing
DATA_START       = "2019-01-01"
DATA_END         = "2025-12-31"

# Chronological 4-way split boundaries
TRAIN_END        = "2021-12-31"
CALIBRATION_END  = "2022-12-31"   # Calibration set: for conformal prediction only
VAL_END          = "2023-12-31"
# TEST: 2024-01-01 → 2025-12-31  (2 full years of out-of-sample evaluation)

# Volatility regime labels (for stress-test analysis)
REGIMES = {
    "stable_baseline": ("2019-01-01", "2019-12-31"),
    "covid_collapse":  ("2020-01-01", "2021-01-31"),
    "uri_crisis":      ("2021-02-01", "2021-02-28"),   # Winter Storm Uri
    "gas_shock":       ("2021-03-01", "2022-12-31"),
    "new_normal":      ("2023-01-01", "2025-12-31"),   # Full post-shock period
}

# ── Feature Engineering ───────────────────────────────────────
TARGET_LAGS     = [24, 48, 72, 96, 168, 336]               # hours (only day-ahead available)
ROLLING_WINDOWS = [24, 48, 168]                            # hours
ROLLING_FUNCS   = ["mean", "std", "min", "max"]

# Weather cities
PJM_CITIES   = ["Philadelphia", "Chicago", "Pittsburgh", "Detroit", "Columbus"]
ERCOT_CITIES = ["Houston", "Dallas", "Austin", "San_Antonio", "Amarillo"]

# Top-N features to keep after SHAP selection (prevents 200+ feature bloat)
SHAP_TOP_N_FEATURES = 50

# ── Model Sequence Configuration ─────────────────────────────
SEQ_LEN_DEFAULT   = 168    # 7 days (primary window)
SEQ_LEN_SHORT     = 48     # 2 days
SEQ_LEN_LONG      = 336    # 14 days
PRED_LEN          = 24     # 24-step day-ahead (unified for all models)
BATCH_SIZE        = 64
MAX_EPOCHS        = 100
LEARNING_RATE     = 3e-4
L2_REG            = 1e-4
PATIENCE          = 15     # Early stopping patience

# ── LightGBM Parameters (point forecast) ────────────────────
LGBM_POINT_PARAMS = {
    "objective":        "regression_l1",
    "metric":           "mae",
    "n_estimators":     2000,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "min_child_samples": 20,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             RANDOM_SEED,
    "boosting_type":    "gbdt",
}

# ── LightGBM Quantile Parameters ─────────────────────────────
LGBM_QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

def get_lgbm_quantile_params(alpha: float) -> dict:
    p = LGBM_POINT_PARAMS.copy()
    p["objective"] = "quantile"
    p["alpha"]     = alpha
    p["metric"]    = "quantile"
    return p

# ── XGBoost Parameters ────────────────────────────────────────
XGB_PARAMS = {
    "objective":       "reg:absoluteerror",
    "n_estimators":    2000,
    "learning_rate":   0.03,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "tree_method":     "hist",
    "seed":            RANDOM_SEED,
    "n_jobs":          -1,
}

# ── Bayesian Bi-LSTM Parameters ──────────────────────────────
BILSTM_UNITS        = 128
BILSTM_DENSE_UNITS  = 64
BILSTM_DROPOUT_RATE = 0.2    # Default; will be swept {0.1,0.2,0.3,0.4}
MC_SAMPLES          = 100    # Monte Carlo forward passes for uncertainty

# ── Conformal Prediction ──────────────────────────────────────
CONFORMAL_ALPHA     = 0.10   # Target: 90% coverage (1 - alpha)

# ── Probabilistic Evaluation ──────────────────────────────────
NOMINAL_COVERAGE    = 0.90   # ALL probabilistic models must use this
LOWER_QUANTILE      = 0.05   # For 90% CI: p05 to p95
UPPER_QUANTILE      = 0.95

# ── Economic Simulation Parameters ───────────────────────────
TRANSACTION_COST_PER_MWH = 0.50    # $/MWh (realistic US power market)
TRADE_VOLUME_MWH         = 1.0     # MWh per trade
SLIPPAGE_STD_FACTOR      = 0.30    # Execution at predicted ± 0.3 × pred std

# ── Stacked Ensemble ──────────────────────────────────────────
# Base models: one primary tree, one secondary tree, one DL.
# iTransformer excluded: NeuralForecast cross_validation only produces
# test-set predictions — including it as a meta-feature would leak test data.
ENSEMBLE_BASE_MODELS = ["lgbm", "xgboost", "bilstm"]
ENSEMBLE_META_PARAMS = {
    "objective":     "regression",
    "metric":        "rmse",
    "n_estimators":  300,
    "learning_rate": 0.05,
    "num_leaves":    25,
    "max_depth":     5,
    "verbose":       -1,
    "n_jobs":        -1,
    "seed":          RANDOM_SEED,
}

# ── API Configurations ──────────────────────────────────────────
# Keys MUST be set as environment variables. Hardcoded fallbacks are
# REMOVED to prevent accidental exposure in git history.
# Set before running download scripts:
#   export PJM_API_KEY="<your-key>"
#   export EIA_API_KEY="<your-key>"
PJM_API_KEY = os.environ.get("PJM_API_KEY") or None
EIA_API_KEY = os.environ.get("EIA_API_KEY") or None

EIA_BASE_URL         = "https://api.eia.gov/v2"
EIA_PJM_RESPONDENT   = "PJM"
EIA_ERCOT_RESPONDENT = "ERCO"

def _require_api_key(name: str, value) -> str:
    """Raise a clear error if a required API key is missing."""
    if not value:
        raise EnvironmentError(
            f"\n  Missing API key: {name}\n"
            f"  Set it with:  export {name}=\"<your-key>\"\n"
            f"  Keys are not stored in source code for security.\n"
        )
    return value


# ── NOAA API Configuration ────────────────────────────────────
# Station IDs for major cities (WBAN / GHCND station IDs)
NOAA_STATIONS = {
    # PJM Cities
    "Philadelphia": "72408",   # PHL Airport
    "Chicago":      "94846",   # ORD Airport
    "Pittsburgh":   "94823",   # PIT Airport
    "Detroit":      "94847",   # DTW Airport
    "Columbus":     "14821",   # CMH Airport
    # ERCOT Cities
    "Houston":      "12960",   # IAH Airport
    "Dallas":       "03927",   # DFW Airport
    "Austin":       "13958",   # AUS Airport
    "San_Antonio":  "12921",   # SAT Airport
    "Amarillo":     "23047",   # AMA Airport
}

# ── Publication Aesthetics ─────────────────────────────────────
PLOT_DPI = 300
PLOT_STYLE = "seaborn-v0_8-whitegrid"

MODEL_COLORS = {
    "Actual":          "#000000",
    "Seasonal Naive":  "#aec7e8",
    "AutoARIMA":       "#8c564b",
    "MSTL":            "#c49c94",
    "LightGBM":        "#2ca02c",
    "XGBoost":         "#98df8a",
    "Bayesian Bi-LSTM":"#ff7f0e",
    "PatchTST":        "#9467bd",
    "iTransformer":    "#1f77b4",
    "N-HiTS":          "#17becf",
    "Chronos-Bolt":    "#e377c2",
    "CQR":             "#d62728",
    "QRF":             "#bcbd22",
    "Ensemble":        "#7f7f7f",
}

MODEL_MARKERS = {
    "LightGBM":        "s",
    "Bayesian Bi-LSTM":"o",
    "PatchTST":        "^",
    "iTransformer":    "D",
    "Chronos-Bolt":    "*",
    "CQR":             "P",
    "Ensemble":        "X",
}
