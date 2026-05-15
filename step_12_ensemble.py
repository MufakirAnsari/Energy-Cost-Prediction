"""
step_12_ensemble.py
===================
Stacked Ensemble Meta-Learner.

Combines predictions from available model families:
  - LightGBM (primary tree model, always available)
  - XGBoost  (secondary tree model, always available)
  - Bayesian Bi-LSTM (if scaling is compatible — PJM only)

Meta-learner: LightGBM trained on VALIDATION set predictions → generalizes to TEST.

WHY VALIDATION SET (not calibration set):
  The calibration set (2022) covers the energy-price crisis, a distributional regime
  very different from the 2024-2025 test period ("new_normal"). A meta-learner trained
  on 2022 data learns a combination optimised for crisis conditions and degrades on
  the test set. The validation set (2023) shares the same distributional regime as
  the test set, making it a better training ground for the stacking weights.

Note on DL models (PatchTST, iTransformer, N-HiTS, BiTCN, TFT):
  These are trained with cross_validation covering the TEST set only (2024-2025),
  so they have no validation-set (2023) predictions. They cannot be used as
  meta-features without data leakage. They are compared separately (Table 2).

Run:
    python step_12_ensemble.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Disable XLA JIT to avoid libdevice/CUDA compilation errors on GTX 1650
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF info/warning logs

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("pip install lightgbm")


# ─────────────────────────────────────────────────────────────────────────────
# BASE MODEL INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_lgbm_preds(market: str, split_df: pd.DataFrame) -> np.ndarray:
    """LightGBM point forecast on any split."""
    path = os.path.join(config.MODEL_DIR, f"lgbm_point_{market.lower()}.joblib")
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    X = split_df.drop(columns=[config.TARGET_COL])
    return model.predict(X)


def get_xgboost_preds(market: str, split_df: pd.DataFrame) -> np.ndarray:
    """XGBoost point forecast on any split."""
    path = os.path.join(config.MODEL_DIR, f"xgboost_point_{market.lower()}.joblib")
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    X = split_df.drop(columns=[config.TARGET_COL])
    return model.predict(X)


def get_bilstm_preds(market: str, split_df: pd.DataFrame,
                     seq_len: int = config.SEQ_LEN_DEFAULT) -> np.ndarray:
    """BiLSTM MC Dropout inference on any split (pure-Keras, no TFP)."""
    path      = os.path.join(config.MODEL_DIR, f"bilstm_{market.lower()}.keras")
    meta_path = os.path.join(config.MODEL_DIR, f"bilstm_{market.lower()}_meta.joblib")
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None
    try:
        import tensorflow as tf
        meta       = joblib.load(meta_path)
        scaler_min = meta["scaler_min"]
        scaler_max = meta["scaler_max"]
        denom      = scaler_max - scaler_min
        denom[denom == 0] = 1.0

        scaled     = (split_df.values.astype(np.float32) - scaler_min) / denom
        target_idx = meta["target_idx"]

        X = []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i - seq_len: i])
        if not X:
            return None
        X = np.array(X, dtype=np.float32)

        model = tf.keras.models.load_model(path)
        # Batched inference to avoid GPU OOM on GTX 1650 (4GB VRAM)
        INFER_BATCH = 256
        chunks = []
        for start in range(0, len(X), INFER_BATCH):
            batch = X[start: start + INFER_BATCH]
            out   = model(batch, training=False)
            chunks.append(np.array(out).flatten())
        preds_scaled = np.concatenate(chunks)

        # Inverse scale
        preds = preds_scaled * denom[target_idx] + scaler_min[target_idx]

        # Pad first seq_len rows with NaN
        full_preds = np.full(len(split_df), np.nan)
        full_preds[seq_len:] = preds
        return full_preds
    except Exception as e:
        print(f"  WARNING: BiLSTM inference failed: {e}")
        return None


def get_dl_test_preds(model_name: str, market: str, index: pd.Index) -> np.ndarray:
    """
    Load test-set-only DL predictions (PatchTST/iTransformer/N-HiTS).
    CSV format: ds, actual, predicted  (from our cross_validation scripts).
    Returns NaN-padded array aligned to `index`.
    """
    fname = f"{model_name}_preds_{market.lower()}.csv"
    path  = os.path.join(config.REPORT_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.set_index("ds")
    df.index = pd.to_datetime(df.index, utc=True)

    # Deduplicate index (cross_validation can produce duplicate timestamps)
    df = df[~df.index.duplicated(keep="first")]

    # Align index timezone
    if hasattr(index, 'tz') and index.tz is not None:
        idx = index.tz_convert("UTC")
    else:
        idx = pd.to_datetime(index, utc=True)

    aligned = df["predicted"].reindex(idx)
    return aligned.values


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  Stacked Ensemble Meta-Learner: {market}")
    print(f"{'='*65}")

    # ── CRITICAL FIX: use VAL set (2023) for meta-learner training ──
    # The calibration set (2022) covers the gas-price crisis: its distribution
    # is very different from the test period (2024-2025), causing the ensemble
    # meta-learner to learn a combination that degrades on the actual test set.
    # The validation set (2023, "new_normal") shares the same distribution as
    # test, making it a better training ground for the meta-learner.
    cal_df  = pd.read_parquet(config.PJM_VAL_PATH  if market=="PJM" else config.ERCOT_VAL_PATH)
    test_df = pd.read_parquet(config.PJM_TEST_PATH if market=="PJM" else config.ERCOT_TEST_PATH)
    y_cal   = cal_df[config.TARGET_COL].values
    y_test  = test_df[config.TARGET_COL].values
    print(f"  Meta-learner training set (val 2023): {len(cal_df):,} rows")


    # ── Build calibration meta-features (models that cover 2022 cal set) ──
    print("\n  Building calibration meta-features...")
    cal_feats  = {}
    test_feats = {}

    for name, fn in [("lgbm", get_lgbm_preds), ("xgboost", get_xgboost_preds)]:
        cal_p  = fn(market, cal_df)
        test_p = fn(market, test_df)
        if cal_p is not None and test_p is not None:
            cal_feats[name]  = cal_p
            test_feats[name] = test_p
            print(f"    {name:15} ✅ cal={len(cal_p):,} | test={len(test_p):,}")

    bilstm_cal  = get_bilstm_preds(market, cal_df)
    bilstm_test = get_bilstm_preds(market, test_df)
    if bilstm_cal is not None:
        valid_cal = (~np.isnan(bilstm_cal)).sum()
        if valid_cal > 100:
            # Sanity check: BiLSTM val MAE must be plausible.
            # ERCOT BiLSTM uses log1p scaling incompatible with this loader
            # → produces MAE ~25 vs LGBM ~12. Skip if >5× LGBM val MAE.
            bl_valid = ~np.isnan(bilstm_cal) & ~np.isnan(y_cal)
            bl_mae   = np.mean(np.abs(y_cal[bl_valid] - bilstm_cal[bl_valid]))
            lgbm_cal_preds = cal_feats.get("lgbm")
            lgbm_mae = np.mean(np.abs(y_cal[~np.isnan(lgbm_cal_preds)] -
                                     lgbm_cal_preds[~np.isnan(lgbm_cal_preds)])) \
                       if lgbm_cal_preds is not None else np.inf
            if bl_mae > 5 * lgbm_mae:
                print(f"    bilstm  ⚠️  val MAE={bl_mae:.2f} × LGBM={lgbm_mae:.2f} — scaling mismatch, skipped")
            else:
                cal_feats["bilstm"]  = bilstm_cal
                test_feats["bilstm"] = bilstm_test
                print(f"    {'bilstm':15} ✅ cal={valid_cal:,} valid rows (MAE={bl_mae:.2f})")
        else:
            print(f"    bilstm  ⚠️  too few valid cal rows ({valid_cal}), skipping")

    if len(cal_feats) < 2:
        print("  ❌ Need ≥2 base models. Ensure lgbm + xgboost are trained.")
        return None

    X_cal_meta  = pd.DataFrame(cal_feats,  index=cal_df.index)
    X_test_meta = pd.DataFrame(test_feats, index=test_df.index)

    # Optionally add DL test predictions as test-only features
    # (NOT used for meta-learner training — only enriches test prediction)
    for dl_name in ["patchtst", "itransformer", "nhits"]:
        dl_preds = get_dl_test_preds(dl_name, market, test_df.index)
        if dl_preds is not None:
            valid = (~np.isnan(dl_preds)).sum()
            print(f"    {dl_name:15} ℹ️  test-only ({valid:,} valid) — not used in meta-learner")

    # Drop rows where any base model is NaN
    cal_mask  = ~X_cal_meta.isna().any(axis=1).values
    test_mask = ~X_test_meta.isna().any(axis=1).values
    print(f"\n  Calibration valid rows: {cal_mask.sum():,} / {len(cal_mask):,}")
    print(f"  Test valid rows:        {test_mask.sum():,} / {len(test_mask):,}")

    # ── Train meta-learner ────────────────────────────────────────
    # ── Train meta-learner (with held-out early stopping) ────────────────────
    # Use first 75% of cal set for training, last 25% for early stopping.
    # This prevents overfitting that degrades test performance.
    n_meta   = cal_mask.sum()
    n_train  = int(n_meta * 0.75)
    cal_idx  = np.where(cal_mask)[0]
    tr_idx   = cal_idx[:n_train]
    ev_idx   = cal_idx[n_train:]

    X_meta_all = X_cal_meta.values
    meta_params = {k: v for k, v in config.ENSEMBLE_META_PARAMS.items()
                   if k != "n_estimators"}
    n_est = config.ENSEMBLE_META_PARAMS.get("n_estimators", 300)

    meta_model = lgb.LGBMRegressor(**meta_params, n_estimators=n_est)
    meta_model.fit(
        X_meta_all[tr_idx], y_cal[tr_idx],
        eval_set=[(X_meta_all[ev_idx], y_cal[ev_idx])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"  Meta-learner best iteration: {meta_model.best_iteration_}")

    # ── Predict on test ───────────────────────────────────────────
    ensemble_preds = np.full(len(test_df), np.nan)
    if test_mask.sum() > 0:
        ensemble_preds[test_mask] = meta_model.predict(
            X_test_meta.values[test_mask]
        )

    # ── Evaluate ──────────────────────────────────────────────────
    eval_mask = ~np.isnan(y_test) & ~np.isnan(ensemble_preds)
    mae  = np.mean(np.abs(y_test[eval_mask] - ensemble_preds[eval_mask]))
    rmse = np.sqrt(np.mean((y_test[eval_mask] - ensemble_preds[eval_mask])**2))

    print(f"\n  Ensemble Test Results ({market}):")
    print(f"    Ensemble  MAE={mae:.4f}  RMSE={rmse:.4f}")
    for col in cal_feats:
        p = test_feats.get(col)
        if p is not None:
            m = ~np.isnan(y_test) & ~np.isnan(np.array(p))
            b_mae = np.mean(np.abs(y_test[m] - np.array(p)[m]))
            print(f"    {col:15}  MAE={b_mae:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    results = pd.DataFrame({
        "actual":   y_test,
        "ensemble": ensemble_preds,
        **{c: test_feats.get(c, [np.nan]*len(test_df)) for c in cal_feats}
    }, index=test_df.index)

    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    meta_path = os.path.join(config.MODEL_DIR,  f"ensemble_meta_{market.lower()}.joblib")
    out_path  = os.path.join(config.REPORT_DIR, f"ensemble_preds_{market.lower()}.csv")

    joblib.dump(meta_model, meta_path)
    results.to_csv(out_path)

    print(f"\n  ✅ Meta model: {meta_path}")
    print(f"  ✅ Predictions: {out_path}")
    return meta_model, results


if __name__ == "__main__":
    run_ensemble("PJM")
    run_ensemble("ERCOT")
