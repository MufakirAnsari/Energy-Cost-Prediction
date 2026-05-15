"""
step_05_train_bilstm.py
=======================
Trains a Bayesian Bidirectional LSTM using MC Dropout for uncertainty.

Key methodological decisions:
- MC Dropout (Gal & Ghahramani 2016): dropout active at inference time
  → approximates Bayesian posterior over weights
- NO TensorFlow Probability required — pure Keras implementation
- Dropout rate SWEPT over {0.1, 0.2, 0.3, 0.4} on validation set
- Best rate selected by Expected Calibration Error (ECE)
- 100 MC forward passes at inference time for uncertainty quantification
- Loss: MAE (robust to ERCOT spikes)

Run:
    python step_05_train_bilstm.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

# Disable XLA JIT to avoid 'libdevice not found' on GTX 1650
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"]             = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    raise ImportError("pip install tensorflow")

# Suppress verbose TF output
tf.get_logger().setLevel("ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS (add to config.py if missing)
# ─────────────────────────────────────────────────────────────────────────────

BILSTM_UNITS      = getattr(config, "BILSTM_UNITS",      64)
BILSTM_DENSE_UNITS = getattr(config, "BILSTM_DENSE_UNITS", 32)
SEQ_LEN           = getattr(config, "SEQ_LEN_DEFAULT",   168)
BATCH_SIZE        = getattr(config, "BATCH_SIZE",         256)
MAX_EPOCHS        = getattr(config, "MAX_EPOCHS",         50)
PATIENCE          = getattr(config, "PATIENCE",           10)
LR                = getattr(config, "LEARNING_RATE",      1e-3)
L2_REG            = getattr(config, "L2_REG",             1e-4)
MC_SAMPLES        = getattr(config, "MC_SAMPLES",         100)
RANDOM_SEED       = getattr(config, "RANDOM_SEED",        42)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_scale_sequences(market: str, seq_len: int):
    """
    Loads train/val splits, fits MinMax scaler on train only, builds sequences.
    Returns (X_tr, y_tr), (X_v, y_v), n_features, scaler_stats
    """
    tr_df = pd.read_parquet(
        config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH
    )
    v_df = pd.read_parquet(
        config.PJM_VAL_PATH if market == "PJM" else config.ERCOT_VAL_PATH
    )

    # Fit scaler on TRAIN only (no leakage)
    arr_tr   = tr_df.values.astype(np.float32)
    col_min  = arr_tr.min(axis=0)
    col_max  = arr_tr.max(axis=0)
    denom    = np.where(col_max - col_min == 0, 1.0, col_max - col_min)

    def scale(df):
        return (df.values.astype(np.float32) - col_min) / denom

    target_idx = list(tr_df.columns).index(config.TARGET_COL)
    n_features = tr_df.shape[1]

    def make_seqs(scaled):
        X, y = [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i: i + seq_len])
            y.append(scaled[i + seq_len, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_tr, y_tr = make_seqs(scale(tr_df))
    X_v,  y_v  = make_seqs(scale(v_df))

    scaler_stats = (col_min, col_max, denom, target_idx)
    return (X_tr, y_tr), (X_v, y_v), n_features, scaler_stats


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — Pure Keras, no TFP
# ─────────────────────────────────────────────────────────────────────────────

def build_bilstm(input_shape, dropout_rate: float):
    """
    Bidirectional LSTM with MC Dropout.
    Dropout(training=True) keeps dropout active at inference → Bayesian approx.
    """
    l2 = keras.regularizers.l2(L2_REG)
    inputs = keras.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS, return_sequences=True,
                    kernel_regularizer=l2, recurrent_regularizer=l2)
    )(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS // 2, return_sequences=False,
                    kernel_regularizer=l2)
    )(x)
    x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout

    x = layers.Dense(BILSTM_DENSE_UNITS, activation="relu",
                      kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout

    # Output: single scalar (scaled price)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="mae",
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MC INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def mc_predict(model, X, n_mc: int = MC_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_mc stochastic forward passes with dropout active.
    Returns (mean, std) over MC samples — shape: [n_samples]
    """
    preds = np.stack([
        model(X, training=True).numpy().flatten()
        for _ in range(n_mc)
    ])  # [n_mc, n_samples]
    return preds.mean(axis=0), preds.std(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# ECE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(model, X_val, y_val, n_bins: int = 10) -> float:
    """
    Expected Calibration Error via MC Dropout intervals.
    For each confidence level α, check if empirical coverage ≈ α.
    Lower ECE = better calibrated uncertainty.
    """
    n_mc = min(30, MC_SAMPLES)  # Fewer samples for speed during sweep
    preds = np.stack([
        model(X_val, training=True).numpy().flatten()
        for _ in range(n_mc)
    ])  # [n_mc, n_val]

    ece = 0.0
    alphas = np.linspace(0.1, 0.9, n_bins)
    for alpha in alphas:
        lower = np.percentile(preds, (1 - alpha) / 2 * 100, axis=0)
        upper = np.percentile(preds, (1 + alpha) / 2 * 100, axis=0)
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        ece += abs(coverage - alpha)
    return ece / n_bins


# ─────────────────────────────────────────────────────────────────────────────
# DROPOUT SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sweep_dropout(X_tr, y_tr, X_v, y_v, n_features: int, market: str) -> float:
    """
    Train quick models with different dropout rates, select by ECE.
    Returns the best dropout rate.
    """
    dropout_rates = [0.1, 0.2, 0.3, 0.4]
    input_shape   = (X_tr.shape[1], n_features)
    results       = {}

    print(f"\n  [DROPOUT SWEEP] Evaluating rates: {dropout_rates}")

    for dr in dropout_rates:
        print(f"    dropout={dr} ...", end=" ", flush=True)
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_SEED)
        model = build_bilstm(input_shape, dr)

        model.fit(
            X_tr, y_tr,
            validation_data=(X_v, y_v),
            batch_size=BATCH_SIZE * 2,  # Larger batch for sweep speed
            epochs=15,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
            verbose=0,
        )
        ece = compute_ece(model, X_v[:500], y_v[:500])
        results[dr] = ece
        print(f"ECE={ece:.4f}")

    best_dr = min(results, key=results.get)
    print(f"\n  Best dropout rate: {best_dr} (ECE={results[best_dr]:.4f})")

    # Save sweep results
    sweep_df = pd.DataFrame.from_dict(results, orient="index", columns=["ECE"])
    sweep_df.index.name = "dropout_rate"
    sweep_df.reset_index(inplace=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    sweep_df.to_csv(
        os.path.join(config.REPORT_DIR, f"dropout_sweep_{market.lower()}.csv"),
        index=False
    )
    return best_dr


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def predict_on_test(model, market: str, scaler_stats, seq_len: int) -> pd.DataFrame:
    """
    Runs MC Dropout inference on the test set.
    Returns DataFrame with columns: actual, mean_pred, std_pred, q05, q95
    """
    col_min, col_max, denom, target_idx = scaler_stats

    # Load calibration set for CQR (not used here, but keeps data flow clean)
    te_df = pd.read_parquet(
        config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
    )
    tr_df = pd.read_parquet(
        config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH
    )
    # Include calibration year so history is chronologically continuous.
    # Original omission of cal_df created a 1-year gap (train→2021, val→2023)
    # in the lookback window — fixed here.
    cal_df = pd.read_parquet(
        config.PJM_CAL_PATH if market == "PJM" else config.ERCOT_CAL_PATH
    )
    v_df = pd.read_parquet(
        config.PJM_VAL_PATH if market == "PJM" else config.ERCOT_VAL_PATH
    )

    # Build test sequences using train+cal+val as history (chronologically continuous)
    history = pd.concat([tr_df, cal_df, v_df])
    all_data = pd.concat([history, te_df])
    scaled   = (all_data.values.astype(np.float32) - col_min) / denom

    n_hist = len(history)
    n_test = len(te_df)

    X_te, y_te = [], []
    for i in range(n_hist - seq_len, n_hist - seq_len + n_test):
        X_te.append(scaled[i: i + seq_len])
        y_te.append(scaled[i + seq_len, target_idx])

    X_te = np.array(X_te, dtype=np.float32)
    y_te = np.array(y_te, dtype=np.float32)

    print(f"  Running MC inference ({MC_SAMPLES} samples) on test set ({len(X_te):,} steps)...")
    # Run in batches to avoid OOM on GTX 1650
    batch = 512
    all_preds = []
    for i in range(0, len(X_te), batch):
        Xb = X_te[i:i+batch]
        mc = np.stack([model(Xb, training=True).numpy().flatten()
                       for _ in range(MC_SAMPLES)])
        all_preds.append(mc)
    all_preds = np.concatenate(all_preds, axis=1)  # [MC_SAMPLES, n_test]

    mean_s = all_preds.mean(axis=0)
    std_s  = all_preds.std(axis=0)
    q05_s  = np.percentile(all_preds, 5, axis=0)
    q95_s  = np.percentile(all_preds, 95, axis=0)

    # Inverse-scale to original units
    price_min   = col_min[target_idx]
    price_range = denom[target_idx]

    def inv(x): return x * price_range + price_min

    results = pd.DataFrame({
        "actual":    inv(y_te),
        "mean_pred": inv(mean_s),
        "std_pred":  std_s * price_range,
        "q05":       inv(q05_s),
        "q95":       inv(q95_s),
    }, index=te_df.index[-n_test:])

    mask = ~np.isnan(results["actual"])
    mae  = np.mean(np.abs(results.loc[mask, "actual"] - results.loc[mask, "mean_pred"]))
    picp = np.mean((results.loc[mask, "actual"] >= results.loc[mask, "q05"]) &
                   (results.loc[mask, "actual"] <= results.loc[mask, "q95"])) * 100
    print(f"  Test MAE: {mae:.4f} $/MWh  |  PICP (90% MC): {picp:.2f}%")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_bilstm(market: str = "PJM", seq_len: int = SEQ_LEN):
    print(f"\n{'='*65}")
    print(f"  Bayesian Bi-LSTM (MC Dropout): {market} | seq_len={seq_len}")
    print(f"{'='*65}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"  GPU detected: {len(gpus)} device(s)")
    else:
        print("  No GPU detected — running on CPU")

    (X_tr, y_tr), (X_v, y_v), n_features, scaler_stats = load_and_scale_sequences(
        market, seq_len
    )
    print(f"  X_train: {X_tr.shape} | X_val: {X_v.shape}")

    # Step 1: Dropout sweep
    best_dropout = sweep_dropout(X_tr, y_tr, X_v, y_v, n_features, market)

    # Step 2: Full training
    print(f"\n  [FULL TRAINING] dropout={best_dropout}, epochs={MAX_EPOCHS}")
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    input_shape = (seq_len, n_features)
    model = build_bilstm(input_shape, best_dropout)

    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    save_path = os.path.join(config.MODEL_DIR, f"bilstm_{market.lower()}.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True, monitor="val_loss", mode="min", verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    best_val = min(history.history["val_loss"])
    print(f"\n  Training time: {elapsed/60:.1f} min | Best val_loss: {best_val:.6f}")

    # Step 3: Predict on test set
    preds = predict_on_test(model, market, scaler_stats, seq_len)
    pred_path = os.path.join(config.REPORT_DIR, f"bilstm_preds_{market.lower()}.csv")
    preds.to_csv(pred_path)

    # Step 4: Save model + meta
    meta = {
        "scaler_min":   scaler_stats[0],
        "scaler_max":   scaler_stats[1],
        "scaler_denom": scaler_stats[2],
        "target_idx":   scaler_stats[3],
        "dropout_rate": best_dropout,
        "seq_len":      seq_len,
        "n_features":   n_features,
        "market":       market,
        "val_loss":     best_val,
    }
    meta_path = os.path.join(config.MODEL_DIR, f"bilstm_{market.lower()}_meta.joblib")
    joblib.dump(meta, meta_path)

    print(f"\n  ✅ Saved model: {save_path}")
    print(f"  ✅ Saved preds: {pred_path}")
    print(f"  ✅ Saved meta:  {meta_path}")
    return model, meta


if __name__ == "__main__":
    train_bilstm("PJM")
    train_bilstm("ERCOT")
