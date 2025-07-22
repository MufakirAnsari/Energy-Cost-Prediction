#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 23:55:31 2025

@author: ansari
"""

# 03_train_sarimax_resampled.py

import pandas as pd
import pmdarima as pm
import joblib
import os
import time

# --- Local Imports ---
import config


def train_sarimax_resampled():
    """
    Trains a computationally feasible SARIMAX model by resampling the data.

    METHODOLOGICAL COMPROMISE FOR MEMORY CONSTRAINTS:
    Due to extreme memory requirements for fitting a SARIMAX model with daily
    seasonality (m=24) on hourly data, this script implements a pragmatic
    compromise:
    1. The time series (target and features) is resampled to a 3-hour ('3H')
       frequency. This reduces the seasonal period to m=8 (24/3) and
       significantly cuts the number of data points.
    2. The model is trained on the most recent 1 year of this resampled data.
    3. Feature selection is still applied to the resampled features.

    This approach allows the model to train successfully on consumer-grade
    hardware while still capturing sub-daily seasonal patterns.
    """
    print("="*80)
    print(" " * 8 + "TRAINING PRAGMATIC & FEASIBLE SARIMAX (3-HOURLY RESAMPLED)")
    print("="*80)

    # --- 1. Load Full Hourly Data ---
    print("\n--- [ 1. LOADING & PREPARING HOURLY DATA ] ---")
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        print(f"Full hourly data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Processed data not found at {config.PROCESSED_DATA_PATH}. Run preprocess script.")
        return

    # --- 2. Resample Data to 3-Hour Frequency ---
    print("\n--- [ 2. RESAMPLING DATA TO 3H FREQUENCY FOR MEMORY EFFICIENCY ] ---")
    df_resampled = df.resample('3H').mean().dropna()
    print(f"Data resampled from hourly to 3-hourly. New shape: {df_resampled.shape}")

    # --- 3. Prepare Resampled Data for Training ---
    # Use 1 year of resampled data (365 days * 8 periods/day = 2920)
    train_data_points = 365 * 8
    n = len(df_resampled)
    train_end_idx = int(n * 0.7)
    train_start_idx = max(0, train_end_idx - train_data_points)

    train_slice = df_resampled.iloc[train_start_idx:train_end_idx]
    y_train = train_slice[config.TARGET_FEATURE]
    X_train_full = train_slice.drop(columns=[config.TARGET_FEATURE])
    print(f"\nUsing a recent subset of resampled data: {len(y_train)} points.")

    # --- 4. Perform Feature Selection on Resampled Data ---
    print("\n--- [ 3. PERFORMING FEATURE SELECTION ] ---")
    try:
        lgbm_model = joblib.load(os.path.join(config.MODEL_DIR, 'lightgbm_point_model.joblib'))
        print("Loaded LightGBM model for feature importance.")
    except FileNotFoundError:
        print("ERROR: LightGBM model not found. Please run '03_train_lightgbm.py' first.")
        return

    n_top_features = 10
    feature_importances = pd.Series(
        lgbm_model.feature_importances_,
        index=df.drop(columns=[config.TARGET_FEATURE]).columns
    )
    top_features = feature_importances.nlargest(n_top_features).index
    X_train = X_train_full[top_features]

    print(f"Selected the top {n_top_features} most important features.")
    print(f"Number of exogenous features: {X_train.shape[1]}.")

    # --- 5. Find Best SARIMAX Model using auto_arima on Resampled Data ---
    print("\n--- [ 4. SEARCHING FOR OPTIMAL SARIMAX PARAMETERS (m=8) ] ---")
    print("This should now be computationally feasible.")
    start_time = time.time()

    sarimax_model = pm.auto_arima(
        y=y_train,
        X=X_train,
        start_p=1, start_q=1,
        test='adf',
        max_p=3, max_q=3,
        m=8,  # The crucial change: seasonal period is now 8
        d=1,
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
    print(sarimax_model.summary())

    # --- 6. Save the Fitted Model ---
    print("\n--- [ 5. SAVING THE FITTED MODEL ] ---")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, 'sarimax_resampled_model.joblib')
    joblib.dump(sarimax_model, model_save_path)
    print(f"Best Resampled SARIMAX model saved to: {model_save_path}")
    print("\n" + "="*80)
    print(" " * 20 + "FEASIBLE SARIMAX TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    train_sarimax_resampled()
