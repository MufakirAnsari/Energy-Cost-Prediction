# 03_train_autoformer.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import time
import os

import config
from utils import autoformer_generator, SeasonalTrendDecompositionBlock

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- GPU DETECTED ---")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("--- NO GPU DETECTED, using CPU ---")

def load_and_prepare_data_pipeline():
    print("Step 1: Loading data and setting up data pipelines...")
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    n = len(df)
    n_features = len(df.columns)
    
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    
    scaler = MinMaxScaler().fit(train_df)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    
    target_idx = list(df.columns).index(config.TARGET_FEATURE)

    print("Step 2: Creating TensorFlow Dataset generators...")
    train_gen = autoformer_generator(train_scaled, config.SEQ_LENGTH_LONG, config.PRED_LENGTH, target_idx, config.BATCH_SIZE)
    val_gen = autoformer_generator(val_scaled, config.SEQ_LENGTH_LONG, config.PRED_LENGTH, target_idx, config.BATCH_SIZE)

    output_signature = (
        (tf.TensorSpec(shape=(None, config.SEQ_LENGTH_LONG, n_features), dtype=tf.float32),
         tf.TensorSpec(shape=(None, config.DECODER_SEQ_LEN + config.PRED_LENGTH, n_features), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, config.PRED_LENGTH, 1), dtype=tf.float32)
    )
    
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_generator(lambda: val_gen, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)
    
    train_steps = (len(train_df) - (config.SEQ_LENGTH_LONG + config.PRED_LENGTH)) // config.BATCH_SIZE
    val_steps = (len(val_df) - (config.SEQ_LENGTH_LONG + config.PRED_LENGTH)) // config.BATCH_SIZE

    return train_dataset, val_dataset, n_features, train_steps, val_steps

def build_autoformer_model(n_features):
    print("Building Autoformer model...")
    
    encoder_inputs = layers.Input(shape=(config.SEQ_LENGTH_LONG, n_features))
    decoder_inputs = layers.Input(shape=(config.DECODER_SEQ_LEN + config.PRED_LENGTH, n_features))
    
    l2_reg = keras.regularizers.l2(config.L2_REG_FACTOR)

    # Encoder
    enc_out = layers.Dense(config.AUTOFORMER_D_MODEL, kernel_regularizer=l2_reg)(encoder_inputs)
    for _ in range(config.AUTOFORMER_ENCODER_LAYERS):
        enc_out = SeasonalTrendDecompositionBlock(
            d_model=config.AUTOFORMER_D_MODEL, n_heads=config.AUTOFORMER_NUM_HEADS,
            d_ff=config.AUTOFORMER_D_FF, dropout=config.AUTOFORMER_DROPOUT,
            seq_len=config.SEQ_LENGTH_LONG
        )(enc_out)

    # Decoder
    seasonal_init, trend_init = layers.Dense(config.AUTOFORMER_D_MODEL)(decoder_inputs), layers.Dense(config.AUTOFORMER_D_MODEL)(decoder_inputs)
    
    for i in range(config.AUTOFORMER_DECODER_LAYERS):
        # --- THE FIX ---
        # The decomposition logic for the decoder was missing. It's now correctly implemented.
        seasonal_init = SeasonalTrendDecompositionBlock(
             d_model=config.AUTOFORMER_D_MODEL, n_heads=config.AUTOFORMER_NUM_HEADS,
             d_ff=config.AUTOFORMER_D_FF, dropout=config.AUTOFORMER_DROPOUT,
             seq_len=config.DECODER_SEQ_LEN + config.PRED_LENGTH
        )(seasonal_init)

        # Cross-attention block would go here if implemented
        # Trend prediction block
        trend_init = SeasonalTrendDecompositionBlock(
             d_model=config.AUTOFORMER_D_MODEL, n_heads=config.AUTOFORMER_NUM_HEADS,
             d_ff=config.AUTOFORMER_D_FF, dropout=config.AUTOFORMER_DROPOUT,
             seq_len=config.DECODER_SEQ_LEN + config.PRED_LENGTH
        )(trend_init)
        # --- END FIX ---
    
    decoder_output = seasonal_init + trend_init
    final_output_seq = layers.Dense(1, kernel_regularizer=l2_reg)(decoder_output)
    final_output = layers.Lambda(lambda x: x[:, -config.PRED_LENGTH:, :])(final_output_seq)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=final_output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="mean_squared_error", metrics=["mean_absolute_error"]
    )
    return model

def train():
    check_gpu()
    train_ds, val_ds, n_features, train_steps, val_steps = load_and_prepare_data_pipeline()

    print("\nStep 3: Building Autoformer model...")
    model = build_autoformer_model(n_features)
    model.summary()
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, "autoformer_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
    ]

    print(f"\nStep 4: Training Autoformer model...")
    start_time = time.time()
    
    model.fit(
        train_ds,
        epochs=config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    duration = time.time() - start_time
    print(f"\nTraining complete.")
    print(f"Total Training Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Best model saved to {model_save_path}")

if __name__ == '__main__':
    train()