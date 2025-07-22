# config.py

import os
import matplotlib.pyplot as plt

# --- Base Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- File Paths ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_ENERGY_DATA_PATH = os.path.join(DATA_DIR, 'energy_dataset.csv')
RAW_WEATHER_DATA_PATH = os.path.join(DATA_DIR, 'weather_features.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.parquet')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

# --- Data Preprocessing ---
TARGET_FEATURE = 'price actual'

# --- Model & Training Parameters ---
# For sequential models
SEQ_LENGTH = 72
SEQ_LENGTH_LONG = 336 # For Autoformer
SEQ_LENGTH_CONSTRAINED = 72
PRED_LENGTH = 24 # Prediction length, e.g., 24 hours
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
L2_REGULARIZATION_FACTOR = 0.001
L2_REG_FACTOR = 0.001
DECODER_SEQ_LEN = SEQ_LENGTH_LONG // 2 # Standard Autoformer setting

# For sequential models



# Constrained Transformer Model Parameters
CONSTRAINED_TRANSFORMER_HEAD_SIZE = 128
CONSTRAINED_TRANSFORMER_NUM_HEADS = 2
CONSTRAINED_TRANSFORMER_FF_DIM = 4
CONSTRAINED_TRANSFORMER_NUM_BLOCKS = 2
CONSTRAINED_TRANSFORMER_DROPOUT = 0.2

# Transformer (Constrained) Model Parameters
TRANSFORMER_HEAD_SIZE = 128
TRANSFORMER_NUM_HEADS = 2
TRANSFORMER_FF_DIM = 4
TRANSFORMER_NUM_BLOCKS = 2
TRANSFORMER_DROPOUT = 0.2 # Increased for regularization

# Autoformer (Long-Sequence) Model Parameters
AUTOFORMER_D_MODEL = 64
AUTOFORMER_NUM_HEADS = 4
AUTOFORMER_D_FF = 128
AUTOFORMER_ENCODER_LAYERS = 2 # Adjusted for efficiency
AUTOFORMER_DECODER_LAYERS = 1 # Adjusted for efficiency
AUTOFORMER_DROPOUT = 0.2
AUTOFORMER_MOVING_AVG = 25

# Bayesian LSTM Model Parameters
BAYESIAN_LSTM_UNITS = 64
BAYESIAN_DENSE_UNITS = 32

# =========================================================================
# === ADD THIS SECTION ====================================================
# =========================================================================
# --- Tree-Based Model Parameters (LightGBM) ---
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}


ENSEMBLE_LGBM_PARAMS = {
    'objective': 'regression', # Standard L2 loss for the meta-learner
    'metric': 'rmse',
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 20,
    'max_depth': 5,
    'feature_fraction': 1.0, # Use all features (i.e., all base model predictions)
    'bagging_fraction': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

PUBLICATION_PALETTE = {
    'Actual': '#000000', # Black
    'Ensemble': '#d62728', # Red
    'Autoformer': '#1f77b4', # Blue
    'LightGBM': '#2ca02c', # Green
    'Bayesian Bi-LSTM': '#ff7f0e', # Orange
    'Constrained Transformer': '#9467bd', # Purple
    'SARIMA': '#8c564b', # Brown
    'Oracle': '#17becf', # Cyan
    'Naive (Autoformer)': '#1f77b4', # Blue
    'Risk-Aware (Bayesian)': '#ff7f0e', # Orange
}

PUBLICATION_LINESTYLES = {
    'Actual': '-',
    'Ensemble': '--',
    'Autoformer': ':',
    'LightGBM': '-.',
    'Bayesian Bi-LSTM': '--',
    'Constrained Transformer': ':',
    'SARIMA': '-.'
}

# General font settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.weight': 'normal',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
})
# =========================================================================
# === END OF SECTION TO ADD ===============================================
# =========================================================================

# --- Evaluation & Analysis ---
MONTE_CARLO_SAMPLES = 100

# --- Plotting ---
PLOT_STYLE = 'whitegrid'
PLOT_PALETTE = 'viridis'
PLOT_DPI = 300