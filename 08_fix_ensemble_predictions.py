#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 12:14:15 2025

@author: ansari
"""

# 08_fix_ensemble_predictions.py

# --- BOILERPLATE TO FIX MODULE PATHS ---
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# -----------------------------------------

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb # Required for loading the model

# --- Local Imports ---
import config
from utils import inverse_transform

def fix_ensemble_predictions():
    """
    This script loads the previously generated results and corrects the
    predictions for the Ensemble model, which were generated using incorrectly
    scaled meta-features.
    """
    print("="*80)
    print(" " * 15 + "CORRECTING ENSEMBLE MODEL PREDICTIONS")
    print("="*80)

    # --- 1. Load Pre-computed Results and Models ---
    print("\n--- [ 1. LOADING DATA, SCALER, AND MODELS ] ---")
    try:
        # Load the results dataframe containing the base model predictions
        results_df = pd.read_csv(os.path.join(config.REPORT_DIR, 'results_df.csv'), index_col=0, parse_dates=True)
        
        # Load the original dataset to refit the scaler
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        train_df = df[0:int(len(df)*0.7)]
        
        # This scaler is essential to transform predictions back to the [0, 1] range
        scaler = MinMaxScaler().fit(train_df)
        
        # Load the trained ensemble model
        ensemble_model = joblib.load(os.path.join(config.MODEL_DIR, 'enhanced_ensemble_model.joblib'))
        
        print("All necessary files loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file: {e}")
        print("Please ensure '05_evaluate_and_report.py' has been run successfully first.")
        return

    # --- 2. Create the CORRECT Meta-Features ---
    print("\n--- [ 2. RE-CREATING META-FEATURES WITH CORRECT SCALING ] ---")
    
    # Isolate the predictions from the three base models
    autoformer_preds = results_df['Autoformer'].values
    bayesian_preds = results_df['Bayesian Bi-LSTM'].values
    lightgbm_preds = results_df['LightGBM'].values
    
    # Get information needed for scaling/inverse scaling
    target_idx = list(df.columns).index(config.TARGET_FEATURE)
    n_features = len(df.columns)

    # To scale these predictions, we must place them in a matrix of the original
    # data's shape, then use the scaler's `transform` method.
    
    def scale_predictions(preds, scaler, target_idx, n_features):
        """Helper function to scale a 1D prediction array."""
        # Create a dummy array with the same number of features as the original data
        dummy_array = np.zeros((len(preds), n_features))
        # Place the predictions in the target feature column
        dummy_array[:, target_idx] = preds
        # Use the fitted scaler to transform the data
        scaled_preds = scaler.transform(dummy_array)[:, target_idx]
        return scaled_preds

    # Scale each base model's predictions back to the [0, 1] range
    autoformer_scaled = scale_predictions(autoformer_preds, scaler, target_idx, n_features)
    bayesian_scaled = scale_predictions(bayesian_preds, scaler, target_idx, n_features)
    lightgbm_scaled = scale_predictions(lightgbm_preds, scaler, target_idx, n_features)

    # Combine the scaled predictions into the meta-feature matrix
    # This matrix now has the exact same scale and structure as the data
    # the ensemble model was trained on.
    meta_features_scaled = np.column_stack([
        autoformer_scaled,
        bayesian_scaled,
        lightgbm_scaled
    ])
    
    print(f"Correctly scaled meta-features created with shape: {meta_features_scaled.shape}")

    # --- 3. Generate and Save Corrected Predictions ---
    print("\n--- [ 3. GENERATING & SAVING CORRECTED PREDICTIONS ] ---")
    
    # Predict using the correctly scaled features
    ensemble_preds_scaled = ensemble_model.predict(meta_features_scaled)
    
    # Inverse transform the scaled output to get the final price forecast
    ensemble_preds_corrected = inverse_transform(ensemble_preds_scaled, scaler, target_idx, n_features)
    
    # Overwrite the old, incorrect column in the results dataframe
    results_df['Ensemble'] = ensemble_preds_corrected
    
    # Save the fully corrected dataframe
    results_df.to_csv(os.path.join(config.REPORT_DIR, 'results_df_corrected.csv'))
    
    print("Corrected 'Ensemble' predictions have been generated.")
    print(f"The updated results have been saved to 'results_df_corrected.csv'")

    # --- 4. Recalculate and Display Final Point Metrics ---
    print("\n--- [ 4. UPDATED POINT FORECAST PERFORMANCE ] ---")
    
    def calculate_point_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        return {'MAE': mae, 'RMSE': rmse, 'sMAPE (%)': smape}

    point_metrics = []
    # Load the original inference times so we don't lose that data
    point_metrics_df_old = pd.read_csv(os.path.join(config.REPORT_DIR, 'table_01_point_metrics.csv')).set_index('Model')
    
    point_models = ['Constrained Transformer', 'Autoformer', 'Bayesian Bi-LSTM', 'LightGBM', 'Ensemble', 'SARIMA']
    for name in point_models:
        metrics = calculate_point_metrics(results_df['Actual'], results_df[name])
        metrics['Model'] = name
        metrics['Inference Time (ms/pred)'] = point_metrics_df_old.loc[name, 'Inference Time (ms/pred)']
        point_metrics.append(metrics)
    
    point_results_df_new = pd.DataFrame(point_metrics).set_index('Model')
    
    print("\nUPDATED Point Forecast Metrics on Test Set:")
    print(point_results_df_new.round(4))
    
    # Save the new table
    point_results_df_new.to_csv(os.path.join(config.REPORT_DIR, 'table_01_point_metrics_corrected.csv'))
    print("\nUpdated metrics table saved to 'table_01_point_metrics_corrected.csv'")
    
    print("\n" + "="*80)
    print(" " * 25 + "ENSEMBLE FIX COMPLETE")
    print("="*80)

if __name__ == '__main__':
    fix_ensemble_predictions()