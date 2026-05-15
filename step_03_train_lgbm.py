"""
step_03_train_lgbm.py
=====================
Trains LightGBM models for the V2 EPF pipeline:
  1. Point forecast model (MAE objective)
  2. Multi-quantile models at p05, p10, p25, p50, p75, p90, p95
     → enables both 80% CI (p10-p90) AND 90% CI (p05-p95) comparison

All models use the full engineered feature set (tabular format).
The scaler is NOT applied here — LightGBM is scale-invariant.

Training uses early stopping on the validation set.

Run:
    python step_03_train_lgbm.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.dirname(__file__))
import config


def load_splits(market: str):
    paths = {
        "PJM":   (config.PJM_TRAIN_PATH, config.PJM_VAL_PATH),
        "ERCOT": (config.ERCOT_TRAIN_PATH, config.ERCOT_VAL_PATH),
    }
    train_df = pd.read_parquet(paths[market][0])
    val_df   = pd.read_parquet(paths[market][1])
    X_tr = train_df.drop(columns=[config.TARGET_COL])
    y_tr = train_df[config.TARGET_COL]
    X_v  = val_df.drop(columns=[config.TARGET_COL])
    y_v  = val_df[config.TARGET_COL]
    return X_tr, y_tr, X_v, y_v


def train_point_model(X_tr, y_tr, X_v, y_v, market: str) -> lgb.LGBMRegressor:
    print(f"\n  [LGBM POINT] Training {market}...")
    t0 = time.time()
    model = lgb.LGBMRegressor(**config.LGBM_POINT_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t0
    val_preds = model.predict(X_v)
    mae = np.mean(np.abs(y_v.values - val_preds))
    print(f"    Val MAE: {mae:.4f} $/MWh | Time: {elapsed:.1f}s | "
          f"Best iter: {model.best_iteration_}")
    return model


def train_quantile_model(
    X_tr, y_tr, X_v, y_v,
    alpha: float, market: str
) -> lgb.LGBMRegressor:
    params = config.get_lgbm_quantile_params(alpha)
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ],
    )
    # Pinball loss for reporting
    q_pred = model.predict(X_v)
    pinball = np.mean(
        np.where(y_v.values >= q_pred,
                 alpha * (y_v.values - q_pred),
                 (1 - alpha) * (q_pred - y_v.values))
    )
    print(f"    q={alpha:.2f}  Val Pinball: {pinball:.4f}  "
          f"Best iter: {model.best_iteration_}")
    return model


def run_lgbm_training(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  LightGBM Training: {market}")
    print(f"{'='*65}")

    X_tr, y_tr, X_v, y_v = load_splits(market)
    print(f"  Train: {X_tr.shape} | Val: {X_v.shape}")
    print(f"  Features: {list(X_tr.columns[:5])} ...")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    saved = {}

    # Point model
    point_model = train_point_model(X_tr, y_tr, X_v, y_v, market)
    path = os.path.join(config.MODEL_DIR, f"lgbm_point_{market.lower()}.joblib")
    joblib.dump(point_model, path)
    saved["point"] = path
    print(f"  Saved: {path}")

    # Quantile models
    print(f"\n  [LGBM QUANTILE] Training {len(config.LGBM_QUANTILE_LEVELS)} quantiles...")
    t0 = time.time()
    for alpha in config.LGBM_QUANTILE_LEVELS:
        q_model = train_quantile_model(X_tr, y_tr, X_v, y_v, alpha, market)
        q_name  = f"q{int(alpha*100):02d}"
        q_path  = os.path.join(config.MODEL_DIR, f"lgbm_{q_name}_{market.lower()}.joblib")
        joblib.dump(q_model, q_path)
        saved[q_name] = q_path

    print(f"\n  All quantile models trained in {time.time()-t0:.1f}s")
    print(f"  ✅ LightGBM training complete for {market}.")
    return saved


if __name__ == "__main__":
    run_lgbm_training("PJM")
    run_lgbm_training("ERCOT")
