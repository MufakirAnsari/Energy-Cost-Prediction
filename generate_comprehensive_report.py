# generate_comprehensive_report.py

import os
import sys
import platform
import pandas as pd
import numpy as np
import tensorflow as tf
import lightgbm as lgb
import sklearn
import joblib
from datetime import datetime
from contextlib import redirect_stdout
import importlib.metadata

# --- Local Imports ---
import config
from utils import ProbabilisticHead, nll

def get_environment_details():
    """Captures details about the execution environment and library versions."""
    details = [
        "--- [ 1. EXECUTION ENVIRONMENT ] ---",
        f"Report Generated on:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Operating System:       {platform.system()} {platform.release()}",
        f"Python Version:         {platform.python_version()}",
        "\n--- Library Versions ---",
    ]
    libraries_to_check = [
        'tensorflow', 'keras', 'tensorflow-probability', 'pandas',
        'numpy', 'scikit-learn', 'lightgbm', 'joblib'
    ]
    for lib in libraries_to_check:
        try:
            version = importlib.metadata.version(lib)
            details.append(f"{lib.replace('-', ' ').title():<25}: {version}")
        except importlib.metadata.PackageNotFoundError:
            details.append(f"{lib.replace('-', ' ').title():<25}: Not found")
    details.append("-" * 30 + "\n")
    return "\n".join(details)

def get_config_details():
    """Captures all parameters from the config.py file."""
    details = [
        "--- [ 2. PROJECT CONFIGURATION PARAMETERS ] ---"
    ]
    for key, value in config.__dict__.items():
        if key.isupper():
            details.append(f"{key:<25}: {value}")
    details.append("-" * 40 + "\n")
    return "\n".join(details)

def analyze_dataset():
    """Performs and reports a deep analysis of the processed dataset."""
    details = [
        "--- [ 3. DATASET ANALYSIS (processed_data.parquet) ] ---"
    ]
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        details.append("Dataset loaded successfully.\n")
        details.append("--- General Information ---")
        details.append(f"Number of Samples (Rows):   {df.shape[0]:,}")
        details.append(f"Number of Features (Cols):  {df.shape[1]}")
        details.append(f"Total Data Points:          {(df.shape[0] * df.shape[1]):,}")
        details.append(f"Memory Usage:               {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        details.append(f"Missing Values Check:       {df.isnull().sum().sum()} (should be 0)")
        start_date, end_date = df.index.min(), df.index.max()
        duration = end_date - start_date
        details.append("\n--- Time Period ---")
        details.append(f"Start Date:                 {start_date}")
        details.append(f"End Date:                   {end_date}")
        details.append(f"Total Duration:             {duration}")
        n = len(df)
        train_end, val_end = int(n*0.7), int(n*0.9)
        details.append("\n--- Data Split Counts (Approximate) ---")
        details.append(f"Training Set Samples:       {train_end:,}")
        details.append(f"Validation Set Samples:     {val_end - train_end:,}")
        details.append(f"Test Set Samples:           {n - val_end:,}")
        details.append("\n--- Full Feature List ---")
        details.extend(df.columns.tolist())
        details.append("\n\n" + "="*80)
        details.append("          DETAILED DESCRIPTIVE STATISTICS FOR ALL FEATURES")
        details.append("="*80)
        details.append(df.describe().T.to_string())
        details.append("="*80 + "\n")
    except FileNotFoundError:
        details.append("ERROR: Processed data file not found. Please run '02_preprocess.py' first.")
    return "\n".join(map(str, details))

def analyze_model(model_name, model_type):
    """Analyzes a given model file and returns its details and summary."""
    title_name = model_name.replace('_', ' ').upper()
    details = [f"\n--- [ {title_name} ANALYSIS ] ---"]
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.keras" if model_type == 'keras' else f"{model_name}.joblib")
    try:
        file_size = os.path.getsize(model_path) / (1024**2)
        last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
        details.append(f"File Path:          {model_path}")
        details.append(f"File Size:          {file_size:.2f} MB")
        details.append(f"Last Modified:      {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        if model_type == 'keras':
            custom_objects = {'ProbabilisticHead': ProbabilisticHead, 'nll': nll} if 'bayesian' in model_name else None
            # --- CORRECTED LINE: Added the 'tf.' prefix ---
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            # --- END CORRECTION ---
            summary_path = os.path.join(config.REPORT_DIR, f'summary_{model_name}.txt')
            with open(summary_path, 'w') as f:
                with redirect_stdout(f):
                    model.summary(expand_nested=True)
            details.append(f"Architecture Summary: See '{summary_path}'")
        elif model_type == 'lightgbm':
            model = joblib.load(model_path)
            details.append("\n--- Hyperparameters ---")
            params = model.get_params()
            for key, value in params.items():
                details.append(f"  {key:<20}: {value}")
    except FileNotFoundError:
        details.append(f"\nERROR: Model file not found at {model_path}. Please ensure it has been trained.")
    return "\n".join(details)

def main():
    """Generates a single, comprehensive text report of the entire project."""
    report_path = os.path.join(config.REPORT_DIR, 'comprehensive_project_report.txt')
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    print(f"--- Generating comprehensive project report... ---")
    print(f"Output will be saved to: {report_path}")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("      COMPREHENSIVE THESIS PROJECT REPORT: DEEP LEARNING FOR PRICE FORECASTING\n")
        f.write("="*80 + "\n\n")
        f.write(get_environment_details())
        f.write(get_config_details())
        f.write(analyze_dataset())
        f.write("\n" + "="*40 + " MODEL DETAILS " + "="*40 + "\n")
        f.write(analyze_model('transformer_model', 'keras'))
        f.write(analyze_model('bayesian_model', 'keras'))
        f.write(analyze_model('ensemble_model', 'lightgbm'))
        f.write("\n\n" + "="*80 + "\n")
        f.write("                          END OF REPORT\n")
        f.write("="*80 + "\n")
    print("--- Report generation complete. ---")

if __name__ == '__main__':
    main()