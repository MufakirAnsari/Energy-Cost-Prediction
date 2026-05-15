"""
step_11_qrf.py
==============
Trains a Quantile Regression Forest (QRF) for probabilistic forecasting.

QRF (Meinshausen, 2006) extends Random Forests to produce full conditional
quantile distributions. It is a strong non-parametric probabilistic baseline
that doesn't require distribution assumptions.

Advantages over LightGBM quantile:
  - Single model produces ALL quantiles simultaneously
  - Exact finite-sample coverage on training data
  - No quantile crossing by construction

Run:
    python step_11_qrf.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    raise ImportError("pip install quantile-forest")


def load_splits(market: str):
    tr = pd.read_parquet(config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH)
    ca = pd.read_parquet(config.PJM_CAL_PATH   if market == "PJM" else config.ERCOT_CAL_PATH)
    va = pd.read_parquet(config.PJM_VAL_PATH   if market == "PJM" else config.ERCOT_VAL_PATH)
    te = pd.read_parquet(config.PJM_TEST_PATH  if market == "PJM" else config.ERCOT_TEST_PATH)
    return tr, ca, va, te


def run_qrf(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  Quantile Regression Forest: {market}")
    print(f"{'='*65}")

    tr, ca, va, te = load_splits(market)

    # FAIRNESS FIX: train QRF on TRAIN set only, same as LightGBM/XGBoost.
    # Previous: pd.concat([tr, ca, va]) gave QRF ~3 extra years of data vs LGBM.
    # That made direct MAE comparisons methodologically invalid.
    # QRF is non-parametric (no calibration set needed), so train-only is correct.
    # NOTE: re-run this script to regenerate qrf_preds_*.csv after this change.
    X_tr = tr.drop(columns=[config.TARGET_COL])
    y_tr = tr[config.TARGET_COL].values
    X_te = te.drop(columns=[config.TARGET_COL])
    y_te = te[config.TARGET_COL].values

    print(f"  Train: {X_tr.shape} | Test: {X_te.shape}")

    t0 = time.time()
    qrf = RandomForestQuantileRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )
    qrf.fit(X_tr, y_tr)
    print(f"  Training: {time.time()-t0:.1f}s")

    # Predict quantiles on test set
    quantiles = config.LGBM_QUANTILE_LEVELS  # [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    t0 = time.time()
    q_preds = qrf.predict(X_te, quantiles=quantiles)  # shape: [n_test, n_quantiles]
    print(f"  Inference: {time.time()-t0:.1f}s")

    # Build results DataFrame
    results = pd.DataFrame(
        q_preds,
        columns=[f"q{int(q*100):02d}" for q in quantiles],
        index=te.index,
    )
    results["actual"] = y_te
    results["point"]  = results["q50"]  # median as point forecast

    # Evaluate 90% CI (q05–q95)
    mask = ~np.isnan(y_te)
    lower = results["q05"].values
    upper = results["q95"].values
    picp = np.mean((y_te[mask] >= lower[mask]) & (y_te[mask] <= upper[mask])) * 100
    mpiw = np.mean(upper[mask] - lower[mask])
    mae  = np.mean(np.abs(y_te[mask] - results["q50"].values[mask]))

    print(f"\n  QRF Test Results ({market}):")
    print(f"    MAE (median):  {mae:.4f} $/MWh")
    print(f"    PICP (90% CI): {picp:.2f}%  (target: ≥90%)")
    print(f"    MPIW (90% CI): {mpiw:.4f} $/MWh")

    # Pinball losses
    for q in quantiles:
        col = f"q{int(q*100):02d}"
        pb = np.mean(np.where(
            y_te[mask] >= results[col].values[mask],
            q * (y_te[mask] - results[col].values[mask]),
            (1 - q) * (results[col].values[mask] - y_te[mask])
        ))
        print(f"    Pinball q={q:.2f}: {pb:.4f}")

    # Save model + predictions
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, f"qrf_{market.lower()}.joblib")
    joblib.dump(qrf, model_path)

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out = os.path.join(config.REPORT_DIR, f"qrf_preds_{market.lower()}.csv")
    results.to_csv(out)

    print(f"\n  ✅ Saved model: {model_path}")
    print(f"  ✅ Saved preds: {out}")
    return qrf, results


if __name__ == "__main__":
    run_qrf("PJM")
    run_qrf("ERCOT")
