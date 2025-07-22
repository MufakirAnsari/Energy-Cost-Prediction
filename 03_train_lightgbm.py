#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 15:15:41 2025

@author: ansari
"""

# 03_train_lightgbm.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import time

# --- Local Imports ---
import config

def train_lightgbm_models():
    """
    Trains two types of LightGBM models:
    1. A standard LGBMRegressor for point forecasting (powerful baseline).
    2. Three separate LGBMRegressors for quantile regression (p10, p50, p90)
       to serve as a probabilistic forecasting alternative.
    """
    print("="*80)
    print(" " * 15 + "TRAINING TREE-BASED BASELINE & QUANTILE MODELS: LightGBM")
    print("="*80)

    # --- 1. Load Data ---
    print("\n--- [ 1. LOADING & PREPARING DATA ] ---")
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Processed data not found at {config.PROCESSED_DATA_PATH}")
        print("Please run '02_preprocess.py' first.")
        return

    # --- 2. Prepare Data for Tabular Learning ---
    # For tree-based models, we treat the data as a simple feature matrix.
    # The target is 'price actual', and all other columns are features.
    X = df.drop(columns=[config.TARGET_FEATURE])
    y = df[config.TARGET_FEATURE]

    # Use the same split percentages as the other models.
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    X_train, y_train = X.iloc[0:train_end], y.iloc[0:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    # Test set will be loaded in the evaluation script, we only need train/val here.

    print(f"Training features shape:   {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # --- 3. Train Standard LightGBM for Point Forecasting ---
    print("\n--- [ 2. TRAINING POINT FORECAST MODEL (LGBM REGRESSOR) ] ---")
    start_time = time.time()

    # Use early stopping to prevent overfitting
    early_stopping_callback = lgb.early_stopping(stopping_rounds=10, verbose=True)

    point_model = lgb.LGBMRegressor(**config.LGBM_PARAMS)

    point_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[early_stopping_callback]
    )

    duration = time.time() - start_time
    print(f"Point forecast model trained in {duration:.2f} seconds.")

    # Save the model
    point_model_path = os.path.join(config.MODEL_DIR, 'lightgbm_point_model.joblib')
    joblib.dump(point_model, point_model_path)
    print(f"Point forecast model saved to: {point_model_path}")

    # --- 4. Train LightGBM for Quantile Regression ---
    print("\n--- [ 3. TRAINING PROBABILISTIC MODELS (LGBM QUANTILE REGRESSION) ] ---")
    quantiles = [0.10, 0.50, 0.90] # p10, p50 (median), p90

    for q in quantiles:
        print(f"\n--- Training for quantile: {q:.2f} ---")
        start_time = time.time()

        # Update parameters for quantile regression
        quantile_params = config.LGBM_PARAMS.copy()
        quantile_params['objective'] = 'quantile'
        quantile_params['alpha'] = q

        quantile_model = lgb.LGBMRegressor(**quantile_params)

        quantile_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='quantile',
            callbacks=[early_stopping_callback]
        )

        duration = time.time() - start_time
        print(f"Quantile model (p{int(q*100)}) trained in {duration:.2f} seconds.")

        # Save the model
        quantile_model_path = os.path.join(config.MODEL_DIR, f'lightgbm_quantile_p{int(q*100)}_model.joblib')
        joblib.dump(quantile_model, quantile_model_path)
        print(f"Quantile model saved to: {quantile_model_path}")

    print("\n" + "="*80)
    print(" " * 20 + "ALL LightGBM TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    train_lightgbm_models()