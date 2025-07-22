# 03_train_constrained_transformer.py
#
# Trains a standard Transformer model on a short sequence length (72 hours).
# This model serves as a case study for performance under computational
# and contextual constraints.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import time
import os
import argparse

# --- Local Imports ---
import config
from utils import create_sequences

def check_gpu():
    """Checks for GPU availability and sets memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- GPU DETECTED ---")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("--- NO GPU DETECTED, using CPU ---")

def load_and_prepare_data():
    """Loads preprocessed data, splits, scales, and creates short sequences."""
    print("Step 1: Loading, splitting, and scaling data for CONSTRAINED model...")
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

def build_constrained_transformer_model(input_shape):
    """Builds the constrained Transformer model based on parameters in config.py."""
    print("Building Constrained Transformer model...")
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(config.CONSTRAINED_TRANSFORMER_NUM_BLOCKS):
        x_norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            key_dim=config.CONSTRAINED_TRANSFORMER_HEAD_SIZE, 
            num_heads=config.CONSTRAINED_TRANSFORMER_NUM_HEADS, 
            dropout=config.CONSTRAINED_TRANSFORMER_DROPOUT
        )(x_norm1, x_norm1)
        x = layers.Add()([x, attention_output])
        x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = keras.Sequential([
            layers.Dense(config.CONSTRAINED_TRANSFORMER_FF_DIM, activation="relu"), 
            layers.Dense(input_shape[-1])
        ])(x_norm2)
        x = layers.Add()([x, ff_output])
        
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )
    return model

def train():
    """Main function to run the model training pipeline."""
    check_gpu()
    X_train, y_train, X_val, y_val = load_and_prepare_data()

    print("\nStep 3: Building Constrained Transformer model...")
    model = build_constrained_transformer_model(X_train.shape[1:])
    model.summary()
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, "constrained_transformer_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    print(f"\nStep 4: Training Constrained Transformer model...")
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
    print(f"\nTraining complete.")
    print(f"Total Training Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Best model saved to {model_save_path}")

if __name__ == '__main__':
    train()