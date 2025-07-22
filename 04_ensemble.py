
# 04_ensemble.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time

# --- Local Imports ---
import config
from utils import (
    ProbabilisticHead, nll, create_sequences, inverse_transform,
    AutoCorrelation, AutoCorrelationLayer, SeasonalTrendDecompositionBlock
)

def load_base_models():
    """
    Loads the diverse base models for the ensemble.
    """
    print("\n--- [ 1. LOADING DIVERSE BASE MODELS ] ---")
    autoformer_path = os.path.join(config.MODEL_DIR, 'autoformer_model.keras')
    bayesian_path = os.path.join(config.MODEL_DIR, 'bayesian_model.keras')
    # The Naive strategy should use the best point forecast model, which is LightGBM.
    lightgbm_path = os.path.join(config.MODEL_DIR, 'lightgbm_point_model.joblib')

    custom_objects = {
        'AutoCorrelation': AutoCorrelation, 'AutoCorrelationLayer': AutoCorrelationLayer,
        'SeasonalTrendDecompositionBlock': SeasonalTrendDecompositionBlock,
        'ProbabilisticHead': ProbabilisticHead, 'nll': nll
    }
    
    autoformer_model = keras.models.load_model(autoformer_path, custom_objects=custom_objects)
    print("Autoformer model loaded successfully.")
    bayesian_model_dist = keras.models.load_model(bayesian_path, custom_objects=custom_objects)
    print("Bayesian model loaded successfully.")

    b_inputs = keras.Input(shape=bayesian_model_dist.input_shape[1:])
    b_dist = bayesian_model_dist(b_inputs)
    b_mean = layers.Lambda(lambda t: t.mean())(b_dist)
    bayesian_model_mean = keras.Model(inputs=b_inputs, outputs=b_mean, name="bayesian_mean_model")
    
    lightgbm_model = joblib.load(lightgbm_path)
    print("LightGBM model loaded successfully.")

    return {'autoformer': autoformer_model, 'bayesian': bayesian_model_mean, 'lightgbm': lightgbm_model}


def generate_meta_features(models, val_df, scaler, target_idx, n_features):
    """
    Generates predictions from base models on the validation set to create
    features for the meta-learner.
    """
    print("\n--- [ 2. GENERATING META-FEATURES FROM BASE MODELS ] ---")
    
    val_scaled = scaler.transform(val_df)
    
    # --- For Sequential Models (Autoformer, Bayesian) ---
    print("Generating predictions for sequential models...")
    # All sequential models now use the same sequence length.
    X_val_seq, _ = create_sequences(val_scaled, config.SEQ_LENGTH, target_idx)
    
    bayesian_preds_scaled = models['bayesian'].predict(X_val_seq, batch_size=config.BATCH_SIZE, verbose=1)
    
    # Autoformer requires encoder and decoder inputs.
    decoder_input_padding = np.zeros((len(X_val_seq), config.PRED_LENGTH, n_features))
    decoder_input_known = X_val_seq[:, -config.DECODER_SEQ_LEN:, :]
    X_val_decoder = np.concatenate([decoder_input_known, decoder_input_padding], axis=1)

    autoformer_preds_scaled = models['autoformer'].predict([X_val_seq, X_val_decoder], batch_size=config.BATCH_SIZE, verbose=1)
    autoformer_preds_final = autoformer_preds_scaled[:, 0].flatten()
    
    # --- For Tabular Model (LightGBM) ---
    print("Generating predictions for tabular model...")
    # Align the tabular data with the predictions from the sequential models.
    # Predictions from sequential models start after the first SEQ_LENGTH window.
    X_val_tabular_aligned = val_df.iloc[config.SEQ_LENGTH:].drop(columns=[config.TARGET_FEATURE])
    lightgbm_preds = models['lightgbm'].predict(X_val_tabular_aligned)
    
    # Re-scale LGBM predictions to match the DL models' scaled output
    dummy_lgbm = np.zeros((len(lightgbm_preds), n_features))
    dummy_lgbm[:, target_idx] = lightgbm_preds
    lightgbm_preds_scaled = scaler.transform(dummy_lgbm)[:, target_idx]

    # --- Align and Combine ---
    # With unified sequence lengths, alignment is straightforward.
    min_length = min(len(autoformer_preds_final), len(bayesian_preds_scaled.flatten()), len(lightgbm_preds_scaled))
    
    meta_features = np.column_stack((
        autoformer_preds_final[:min_length],
        bayesian_preds_scaled.flatten()[:min_length],
        lightgbm_preds_scaled[:min_length]
    ))
    
    # The target `y` must also be aligned to the start of the predictions.
    y_meta_train_scaled = val_scaled[config.SEQ_LENGTH : config.SEQ_LENGTH + min_length, target_idx]

    return meta_features, y_meta_train_scaled


def run_ensemble_training():
    """Main function to train and save the ensemble meta-learner."""
    print("\n" + "="*80 + "\n" + " " * 18 + "TRAINING STACKED ENSEMBLE MODEL\n" + "="*80)
    
    models = load_base_models()
    
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    train_df = df[0:int(len(df)*0.7)]
    val_df = df[int(len(df)*0.7):int(len(df)*0.9)]
    
    scaler = MinMaxScaler().fit(train_df)
    target_idx = list(df.columns).index(config.TARGET_FEATURE)
    n_features = len(df.columns)

    X_meta_train, y_meta_train = generate_meta_features(models, val_df, scaler, target_idx, n_features)
    
    print("\n--- [ 3. TRAINING THE LIGHTGBM META-LEARNER ] ---")
    # NOTE: These parameters could also be tuned if desired.
    meta_learner_params = {
        'objective': 'regression', 'metric': 'rmse', 'n_estimators': 250,
        'learning_rate': 0.05, 'num_leaves': 25, 'max_depth': 6,
        'feature_fraction': 1.0, 'bagging_fraction': 1.0,
        'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    meta_learner = lgb.LGBMRegressor(**meta_learner_params)
    
    meta_learner.fit(
        X_meta_train, y_meta_train,
        feature_name=['autoformer_pred', 'bayesian_pred', 'lightgbm_pred']
    )
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    ensemble_model_path = os.path.join(config.MODEL_DIR, 'enhanced_ensemble_model.joblib')
    joblib.dump(meta_learner, ensemble_model_path)
    
    print(f"\nEnsemble meta-learner trained and saved to {ensemble_model_path}")
    print("\n" + "="*80 + "\n" + " " * 22 + "ENSEMBLE TRAINING COMPLETE\n" + "="*80)

if __name__ == '__main__':
    run_ensemble_training()