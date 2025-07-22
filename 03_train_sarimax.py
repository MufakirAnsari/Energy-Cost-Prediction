# 03_train_sarima.py

import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import model_selection
import joblib
import os
import time

# --- Local Imports ---
import config

def train_sarima():
    """
    Trains a SARIMA model using auto_arima to find the best parameters.
    To handle memory constraints with high-frequency data, the time series is
    resampled to a 3-hour frequency before training.
    """
    print("="*80)
    print(" " * 20 + "TRAINING STATISTICAL BASELINE: SARIMA")
    print("="*80)

    # --- 1. Load Data ---
    print("\n--- [ 1. LOADING & PREPARING DATA ] ---")
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        price_series = df[config.TARGET_FEATURE]
        print(f"Full hourly data loaded successfully. Series length: {len(price_series):,}")
    except FileNotFoundError:
        print(f"ERROR: Processed data not found at {config.PROCESSED_DATA_PATH}")
        print("Please run '02_preprocess.py' first.")
        return

    # --- 2. Resample and Split Data ---
    # --- METHODOLOGICAL CHANGE: Resample data to 3-hour frequency ---
    # This reduces memory usage and focuses the model on core daily patterns.
    print("\nResampling data to 3H frequency to ensure computational feasibility...")
    price_series_resampled = price_series.resample('3H').mean().dropna()
    print(f"Resampled data length: {len(price_series_resampled):,}")

    n = len(price_series_resampled)
    train_end_index = int(n * 0.7)

    # Use a recent 1-year slice of the RESAMPLED data for maximum relevance and speed.
    # 2920 points = 365 days * 8 (3-hour periods per day)
    train_start_index = train_end_index - 2920
    if train_start_index < 0:
        train_start_index = 0

    train_data = price_series_resampled[train_start_index:train_end_index]

    print(f"\nUsing a recent subset of resampled data for training.")
    print(f"Training data size: {len(train_data):,} points")
    print(f"Training period: From {train_data.index.min()} to {train_data.index.max()}")


    # --- 3. Find Best SARIMA Model using auto_arima ---
    print("\n--- [ 2. SEARCHING FOR OPTIMAL SARIMA PARAMETERS ] ---")
    print("This may take a few minutes...")

    start_time = time.time()

    # --- KEY CHANGE: The seasonal period 'm' is now 8 because 24 hours / 3 hours = 8 periods per day ---
    arima_model = pm.auto_arima(
        train_data,
        start_p=1, start_q=1,
        test='adf',
        max_p=3, max_q=3,
        m=8,              # The new seasonal period
        d=None,
        seasonal=True,
        start_P=0,
        D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    duration = time.time() - start_time
    print(f"\nAuto-ARIMA search complete in {duration/60:.2f} minutes.")
    print("\n--- Best Model Summary ---")
    print(arima_model.summary())

    # --- 4. Save the Model ---
    print("\n--- [ 3. SAVING THE FITTED MODEL ] ---")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, 'sarima_model.joblib')
    joblib.dump(arima_model, model_save_path)
    print(f"Best SARIMA model saved to: {model_save_path}")
    print("\n" + "="*80)
    print(" " * 25 + "SARIMA TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    train_sarima()