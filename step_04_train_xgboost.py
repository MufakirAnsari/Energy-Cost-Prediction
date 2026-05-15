"""
step_04_train_xgboost.py
========================
Trains XGBoost point forecast model for both PJM and ERCOT markets.
Uses the same engineered feature matrix as LightGBM.
XGBoost is scale-invariant so no scaler is needed.

Run:
    python step_04_train_xgboost.py
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

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("pip install xgboost")


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


def run_xgboost_training(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  XGBoost Training: {market}")
    print(f"{'='*65}")

    X_tr, y_tr, X_v, y_v = load_splits(market)
    print(f"  Train: {X_tr.shape} | Val: {X_v.shape}")

    params = config.XGB_PARAMS.copy()
    params.pop("n_estimators", None)  # handled via early stopping

    model = xgb.XGBRegressor(
        **params,
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric="mae",
    )

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        verbose=100,
    )
    elapsed = time.time() - t0

    val_preds = model.predict(X_v)
    mae  = np.mean(np.abs(y_v.values - val_preds))
    rmse = np.sqrt(np.mean((y_v.values - val_preds) ** 2))
    print(f"\n  Val MAE: {mae:.4f} $/MWh | RMSE: {rmse:.4f}")
    print(f"  Best iter: {model.best_iteration} | Time: {elapsed:.1f}s")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    path = os.path.join(config.MODEL_DIR, f"xgboost_point_{market.lower()}.joblib")
    joblib.dump(model, path)
    print(f"  ✅ Saved: {path}")
    return model


if __name__ == "__main__":
    run_xgboost_training("PJM")
    run_xgboost_training("ERCOT")
