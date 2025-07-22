# 09_inspect_models.py

# --- BOILERPLATE TO FIX MODULE PATHS ---
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# -----------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K # Import Keras backend
import joblib
import lightgbm as lgb
import pmdarima as pm
import gc # Import garbage collector

# --- Local Imports ---
import config
from utils import (
    ProbabilisticHead, nll,
    AutoCorrelation, AutoCorrelationLayer, SeasonalTrendDecompositionBlock
)

def inspect_keras_model(model_name, model_path, custom_objects):
    """Loads a Keras model and prints its summary."""
    print("\n" + "="*80)
    print(f"INSPECTING KERAS MODEL: {model_name}")
    print("="*80)
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
        model.summary()
    except Exception as e:
        print(f"Could not load or inspect {model_name}: {e}")

def inspect_lightgbm_model(model_name, model_path):
    """Loads a LightGBM model and prints its key parameters."""
    print("\n" + "="*80)
    print(f"INSPECTING LIGHTGBM MODEL: {model_name}")
    print("="*80)
    try:
        model = joblib.load(model_path)
        params = model.get_params()
        print(f"--- Key Parameters for {model_name} ---")
        print(f"Objective: {params.get('objective')}")
        if params.get('objective') == 'quantile':
            print(f"Alpha (Quantile): {params.get('alpha')}")
        print(f"Boosting Type: {params.get('boosting_type')}")
        print(f"Number of Estimators (Max): {params.get('n_estimators')}")
        # Use .best_iteration_ if it exists (from early stopping), otherwise it's None
        best_iter = getattr(model, 'best_iteration_', 'N/A (no early stopping)')
        print(f"Actual Estimators Used: {best_iter}")
        print(f"Learning Rate: {params.get('learning_rate')}")
        print(f"Number of Leaves: {params.get('num_leaves')}")
    except Exception as e:
        print(f"Could not load or inspect {model_name}: {e}")

def inspect_sarima_model(model_name, model_path):
    """Loads a SARIMA model and prints its summary."""
    print("\n" + "="*80)
    print(f"INSPECTING SARIMA MODEL: {model_name}")
    print("="*80)
    try:
        model = joblib.load(model_path)
        print(model.summary())
    except Exception as e:
        print(f"Could not load or inspect {model_name}: {e}")

def main():
    """
    Main function to run all model inspections, clearing memory
    between each deep learning model to prevent OOM errors.
    """
    custom_objects = {
        'AutoCorrelation': AutoCorrelation, 'AutoCorrelationLayer': AutoCorrelationLayer,
        'SeasonalTrendDecompositionBlock': SeasonalTrendDecompositionBlock,
        'ProbabilisticHead': ProbabilisticHead, 'nll': nll
    }

    # --- Inspect Deep Learning Models Sequentially with Memory Clearing ---
    
    inspect_keras_model(
        'Constrained Transformer',
        os.path.join(config.MODEL_DIR, 'constrained_transformer_model.keras'),
        custom_objects
    )
    # --- FIX: Clear GPU memory after inspecting the model ---
    print("\n--- Clearing GPU memory before next model ---")
    K.clear_session()
    gc.collect()

    inspect_keras_model(
        'Autoformer',
        os.path.join(config.MODEL_DIR, 'autoformer_model.keras'),
        custom_objects
    )
    # --- FIX: Clear GPU memory after inspecting the model ---
    print("\n--- Clearing GPU memory before next model ---")
    K.clear_session()
    gc.collect()

    inspect_keras_model(
        'Bayesian Bi-LSTM',
        os.path.join(config.MODEL_DIR, 'bayesian_model.keras'),
        custom_objects
    )
    # --- FIX: Clear GPU memory after inspecting the model ---
    print("\n--- Clearing GPU memory before next model ---")
    K.clear_session()
    gc.collect()

    # --- Inspect CPU-Based Models (No memory clearing needed) ---
    
    inspect_lightgbm_model(
        'LightGBM (Point)',
        os.path.join(config.MODEL_DIR, 'lightgbm_point_model.joblib')
    )
    inspect_lightgbm_model(
        'LightGBM (Quantile p50)',
        os.path.join(config.MODEL_DIR, 'lightgbm_quantile_p50_model.joblib')
    )
    
    inspect_sarima_model(
        'SARIMA',
        os.path.join(config.MODEL_DIR, 'sarima_model.joblib')
    )
    
    print("\n\n" + "="*80)
    print(" " * 28 + "ALL MODELS INSPECTED")
    print("="*80)

if __name__ == '__main__':
    main()