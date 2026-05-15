"""
step_05b_retrain_bilstm_ercot.py
=================================
ERCOT-only BiLSTM retrain with fixes for extreme price distribution.

Root cause of ERCOT val_loss=15.5:
  - MinMax scaler sees $8,996/MWh (Uri crisis) → normal $40 prices scale to ~0.004
  - Gradient explosion with standard Adam (no clipping)
  - 50 epoch limit insufficient for volatile ERCOT distribution

Fixes applied:
  1. Log1p price transform BEFORE MinMax scaling → compresses spikes
  2. gradient clipping: clipnorm=1.0 in Adam
  3. Increase patience to 15 (from 10)
  4. Increase max_epochs to 100 (from 50)
  5. Reduce LR to 5e-4 (from 1e-3)

Run:
    python step_05b_retrain_bilstm_ercot.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import joblib
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"]         = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"]            = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

# ── Hyperparams (ERCOT-tuned) ────────────────────────────────────────────────
BILSTM_UNITS      = getattr(config, "BILSTM_UNITS", 64)
BILSTM_DENSE_UNITS = getattr(config, "BILSTM_DENSE_UNITS", 32)
SEQ_LEN           = getattr(config, "SEQ_LEN_DEFAULT", 168)
BATCH_SIZE        = getattr(config, "BATCH_SIZE", 256)
MAX_EPOCHS        = 100          # ↑ from 50
PATIENCE          = 15           # ↑ from 10
LR                = 5e-4         # ↓ from 1e-3 (more stable)
L2_REG            = getattr(config, "L2_REG", 1e-4)
MC_SAMPLES        = getattr(config, "MC_SAMPLES", 100)
RANDOM_SEED       = getattr(config, "RANDOM_SEED", 42)
CLIP_NORM         = 1.0          # NEW: gradient clipping


def log1p_scale(df, col_min, col_max, denom):
    """Log1p-transform then MinMax-scale. Compresses ERCOT spike distribution."""
    arr = df.values.astype(np.float32)
    arr_log = np.log1p(np.maximum(arr, 0))          # log1p(x) — handles negatives with clip
    return (arr_log - col_min) / denom


def load_and_scale_sequences_ercot(seq_len: int):
    tr_df = pd.read_parquet(config.ERCOT_TRAIN_PATH)
    v_df  = pd.read_parquet(config.ERCOT_VAL_PATH)

    arr_tr     = tr_df.values.astype(np.float32)
    arr_tr_log = np.log1p(np.maximum(arr_tr, 0))   # log1p in TRAIN space
    col_min    = arr_tr_log.min(axis=0)
    col_max    = arr_tr_log.max(axis=0)
    denom      = np.where(col_max - col_min == 0, 1.0, col_max - col_min)

    target_idx = list(tr_df.columns).index(config.TARGET_COL)
    n_features = tr_df.shape[1]

    def scale(df_):
        arr = df_.values.astype(np.float32)
        return (np.log1p(np.maximum(arr, 0)) - col_min) / denom

    def make_seqs(scaled):
        X, y = [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i: i + seq_len])
            y.append(scaled[i + seq_len, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_tr, y_tr = make_seqs(scale(tr_df))
    X_v,  y_v  = make_seqs(scale(v_df))

    scaler_stats = (col_min, col_max, denom, target_idx, "log1p+minmax")
    return (X_tr, y_tr), (X_v, y_v), n_features, scaler_stats


def build_bilstm(input_shape, dropout_rate: float):
    l2 = keras.regularizers.l2(L2_REG)
    inputs = keras.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS, return_sequences=True,
                    kernel_regularizer=l2, recurrent_regularizer=l2)
    )(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS // 2, return_sequences=False,
                    kernel_regularizer=l2)
    )(x)
    x = layers.Dropout(dropout_rate)(x, training=True)

    x = layers.Dense(BILSTM_DENSE_UNITS, activation="relu",
                     kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NORM),
        loss="mae",
    )
    return model


def mc_predict(model, X, n_mc: int = MC_SAMPLES):
    batch = 512
    all_preds = []
    for i in range(0, len(X), batch):
        Xb = X[i:i+batch]
        mc = np.stack([model(Xb, training=True).numpy().flatten()
                       for _ in range(n_mc)])
        all_preds.append(mc)
    all_preds = np.concatenate(all_preds, axis=1)  # [MC, n_test]
    return all_preds.mean(axis=0), all_preds.std(axis=0), all_preds


def compute_ece(model, X_val, y_val, n_bins=10):
    n_mc = 30
    preds = np.stack([model(X_val, training=True).numpy().flatten()
                      for _ in range(n_mc)])
    ece = 0.0
    for alpha in np.linspace(0.1, 0.9, n_bins):
        lower = np.percentile(preds, (1-alpha)/2*100, axis=0)
        upper = np.percentile(preds, (1+alpha)/2*100, axis=0)
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        ece += abs(coverage - alpha)
    return ece / n_bins


def sweep_dropout(X_tr, y_tr, X_v, y_v, n_features):
    dropout_rates = [0.1, 0.2, 0.3, 0.4]
    input_shape = (X_tr.shape[1], n_features)
    results = {}
    print(f"\n  [DROPOUT SWEEP — ERCOT tuned]")
    for dr in dropout_rates:
        print(f"    dropout={dr} ...", end=" ", flush=True)
        tf.keras.backend.clear_session(); tf.random.set_seed(RANDOM_SEED)
        m = build_bilstm(input_shape, dr)
        m.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              batch_size=BATCH_SIZE*2, epochs=20,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=5, restore_best_weights=True)],
              verbose=0)
        ece = compute_ece(m, X_v[:500], y_v[:500])
        results[dr] = ece
        print(f"ECE={ece:.4f}")
    best_dr = min(results, key=results.get)
    print(f"\n  Best dropout: {best_dr}  (ECE={results[best_dr]:.4f})")
    sweep_df = pd.DataFrame.from_dict(results, orient="index", columns=["ECE"])
    sweep_df.index.name = "dropout_rate"
    sweep_df.reset_index().to_csv(
        os.path.join(config.REPORT_DIR, "dropout_sweep_ercot.csv"), index=False)
    return best_dr


def predict_on_test(model, scaler_stats, seq_len: int):
    col_min, col_max, denom, target_idx, _ = scaler_stats

    te_df = pd.read_parquet(config.ERCOT_TEST_PATH)
    tr_df = pd.read_parquet(config.ERCOT_TRAIN_PATH)
    v_df  = pd.read_parquet(config.ERCOT_VAL_PATH)
    cal_df = pd.read_parquet(config.ERCOT_CAL_PATH)

    def scale(df_):
        arr = df_.values.astype(np.float32)
        return (np.log1p(np.maximum(arr, 0)) - col_min) / denom

    history = pd.concat([tr_df, cal_df, v_df])
    all_data = pd.concat([history, te_df])
    scaled = scale(all_data)

    n_hist = len(history)
    n_test = len(te_df)

    X_te, y_te = [], []
    for i in range(n_hist - seq_len, n_hist - seq_len + n_test):
        X_te.append(scaled[i: i + seq_len])
        y_te.append(scaled[i + seq_len, target_idx])
    X_te = np.array(X_te, dtype=np.float32)
    y_te = np.array(y_te, dtype=np.float32)

    print(f"  MC inference ({MC_SAMPLES} passes) on {len(X_te):,} test steps...")
    mean_s, std_s, all_mc = mc_predict(model, X_te, MC_SAMPLES)

    # Compute percentile intervals from MC samples
    q05_s  = np.percentile(all_mc, 5, axis=0)
    q25_s  = np.percentile(all_mc, 25, axis=0)
    q75_s  = np.percentile(all_mc, 75, axis=0)
    q95_s  = np.percentile(all_mc, 95, axis=0)

    # Inverse: undo log1p+minmax
    price_min   = col_min[target_idx]
    price_range = denom[target_idx]

    def inv(x):
        return np.expm1(x * price_range + price_min)  # inverse of log1p+minmax

    results = pd.DataFrame({
        "actual":    inv(y_te),
        "mean_pred": inv(mean_s),
        "std_pred":  std_s * price_range,   # std in log space
        "q05":       inv(q05_s),
        "q25":       inv(q25_s),
        "q75":       inv(q75_s),
        "q95":       inv(q95_s),
    }, index=te_df.index[-n_test:])

    mask = ~np.isnan(results["actual"])
    mae  = np.mean(np.abs(results.loc[mask,"actual"] - results.loc[mask,"mean_pred"]))
    picp = np.mean((results.loc[mask,"actual"] >= results.loc[mask,"q05"]) &
                   (results.loc[mask,"actual"] <= results.loc[mask,"q95"])) * 100
    print(f"  ERCOT BiLSTM  MAE={mae:.4f} $/MWh  PICP(90% MC)={picp:.2f}%")
    return results


def retrain_ercot():
    print(f"\n{'='*65}")
    print(f"  Bayesian Bi-LSTM (MC Dropout): ERCOT [Log1p+Clip Fix]")
    print(f"{'='*65}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)

    (X_tr, y_tr), (X_v, y_v), n_features, scaler_stats = \
        load_and_scale_sequences_ercot(SEQ_LEN)
    print(f"  X_train: {X_tr.shape} | X_val: {X_v.shape}")
    print(f"  y_train: min={y_tr.min():.4f} max={y_tr.max():.4f}  (log1p+minmax space)")

    best_dropout = sweep_dropout(X_tr, y_tr, X_v, y_v, n_features)

    print(f"\n  [FULL TRAINING] dropout={best_dropout}  lr={LR}  "
          f"clipnorm={CLIP_NORM}  epochs={MAX_EPOCHS}")
    tf.keras.backend.clear_session(); tf.random.set_seed(RANDOM_SEED)
    input_shape = (SEQ_LEN, n_features)
    model = build_bilstm(input_shape, best_dropout)

    save_path = os.path.join(config.MODEL_DIR, "bilstm_ercot.keras")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True, monitor="val_loss", mode="min", verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
                     batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
                     callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0
    best_val = min(hist.history["val_loss"])
    print(f"\n  Training: {elapsed/60:.1f} min | Best val_loss: {best_val:.6f}")

    preds    = predict_on_test(model, scaler_stats, SEQ_LEN)
    pred_path = os.path.join(config.REPORT_DIR, "bilstm_preds_ercot.csv")
    preds.to_csv(pred_path)

    meta = {
        "scaler_min":    scaler_stats[0],
        "scaler_max":    scaler_stats[1],
        "scaler_denom":  scaler_stats[2],
        "target_idx":    scaler_stats[3],
        "scaler_type":   scaler_stats[4],
        "dropout_rate":  best_dropout,
        "seq_len":       SEQ_LEN,
        "n_features":    n_features,
        "market":        "ERCOT",
        "val_loss":      best_val,
        "lr":            LR,
        "clip_norm":     CLIP_NORM,
    }
    joblib.dump(meta, os.path.join(config.MODEL_DIR, "bilstm_ercot_meta.joblib"))
    print(f"\n  ✅ Saved model:  {save_path}")
    print(f"  ✅ Saved preds:  {pred_path}")


if __name__ == "__main__":
    retrain_ercot()
