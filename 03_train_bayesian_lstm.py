# 03_train_bayesian_lstm.py

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import time
import os

# --- Local Imports ---
import config
from utils import create_sequences, ProbabilisticHead, nll

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- GPU DETECTED ---")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("--- NO GPU DETECTED, using CPU ---")

def load_and_prepare_data():
    """Loads preprocessed data, splits, scales, and creates sequences."""
    print("Step 1: Loading, splitting, and scaling data...")
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    n = len(df)
    
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    
    scaler = MinMaxScaler().fit(train_df)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    
    target_idx = list(df.columns).index(config.TARGET_FEATURE)
    
    print(f"Step 2: Creating sequences with length {config.SEQ_LENGTH_CONSTRAINED}...")
    X_train, y_train = create_sequences(train_scaled, config.SEQ_LENGTH_CONSTRAINED, target_idx)
    X_val, y_val = create_sequences(val_scaled, config.SEQ_LENGTH_CONSTRAINED, target_idx)
    
    return X_train, y_train, X_val, y_val

def build_bayesian_lstm_model(input_shape, kl_weight):
    """Builds the Bayesian Bi-LSTM model."""
    print("Building Bayesian Bi-LSTM model...")
    l2_reg = keras.regularizers.l2(config.L2_REG_FACTOR)
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(config.BAYESIAN_LSTM_UNITS, return_sequences=False, kernel_regularizer=l2_reg)),
        layers.Dense(config.BAYESIAN_DENSE_UNITS, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.2), # Added dropout for regularization
        ProbabilisticHead(kl_weight)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss=nll)
    return model

def train():
    """Main function to run the model training pipeline."""
    check_gpu()
    X_train, y_train, X_val, y_val = load_and_prepare_data()

    print("\nStep 3: Building Bayesian Bi-LSTM model...")
    # The KL weight balances the complexity of the distribution against the data fit.
    kl_weight = 1.0 / len(X_train)
    model = build_bayesian_lstm_model(X_train.shape[1:], kl_weight)
    model.summary()
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, "bayesian_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    ]

    print(f"\nStep 4: Training Bayesian Bi-LSTM model...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    duration = time.time() - start_time
    print(f"\nTraining complete for Bayesian Bi-LSTM model.")
    print(f"Total Training Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Best model saved to {model_save_path}")

if __name__ == '__main__':
    train()