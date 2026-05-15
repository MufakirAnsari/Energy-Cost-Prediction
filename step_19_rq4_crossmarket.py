"""
step_19_rq4_crossmarket.py — Cross-Market Generalizability (RQ4)
================================================================
Contribution #4: "Train on PJM → evaluate on ERCOT (zero-shot transfer)"

Tests whether PJM-trained models generalize to ERCOT market dynamics.
Uses shared-feature intersection to ensure fair comparison.

Strategy: Both markets share the same feature engineering pipeline
(EIA generation mix, gas price, calendar features). Only shared
feature columns are used for cross-market inference.

Outputs:
  reports/table_rq4_crossmarket.csv
  reports/table_rq4_crossmarket_detail.csv

Run:
    python step_19_rq4_crossmarket.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config


def mae(y, yhat):
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    return np.nanmean(np.abs(y[mask] - yhat[mask]))

def rmse(y, yhat):
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    return np.sqrt(np.nanmean((y[mask] - yhat[mask])**2))

def smape(y, yhat):
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    denom = (np.abs(y[mask]) + np.abs(yhat[mask])) / 2
    denom[denom == 0] = np.nan
    return np.nanmean(np.abs(y[mask] - yhat[mask]) / denom) * 100


def run_rq4():
    print(f"\n{'='*65}")
    print(f"  RQ4: Cross-Market Generalizability")
    print(f"  PJM-trained models → ERCOT test set (zero-shot transfer)")
    print(f"{'='*65}")

    # Load both test sets
    pjm_test   = pd.read_parquet(config.PJM_TEST_PATH)
    ercot_test = pd.read_parquet(config.ERCOT_TEST_PATH)

    y_ercot = ercot_test[config.TARGET_COL].values

    # Determine shared feature columns
    pjm_feats   = set(pjm_test.drop(columns=[config.TARGET_COL]).columns)
    ercot_feats = set(ercot_test.drop(columns=[config.TARGET_COL]).columns)
    shared_feats = sorted(pjm_feats & ercot_feats)

    print(f"\n  PJM features:   {len(pjm_feats)}")
    print(f"  ERCOT features: {len(ercot_feats)}")
    print(f"  Shared features: {len(shared_feats)}")

    # PJM-only features (not in ERCOT)
    pjm_only = pjm_feats - ercot_feats
    print(f"  PJM-only (dropped for cross-market): {sorted(pjm_only)[:5]}...")

    X_ercot_shared = ercot_test[shared_feats]

    rows = []
    models_to_test = [
        ("LightGBM",  "lgbm_point_pjm.joblib"),
        ("XGBoost",   "xgboost_point_pjm.joblib"),
        ("QRF (q50)", "qrf_pjm.joblib"),
    ]

    for model_name, fname in models_to_test:
        path = os.path.join(config.MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠️  {model_name}: model not found — skipping")
            continue

        model = joblib.load(path)

        # Get model's expected feature names
        if hasattr(model, "feature_name_"):
            expected_feats = model.feature_name_   # list attribute, not callable
        elif hasattr(model, "feature_names_in_"):
            expected_feats = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):  # XGBoost
            expected_feats = model.get_booster().feature_names
        else:
            expected_feats = None

        if expected_feats is not None:
            # Align: use shared features that the PJM model knows about
            avail = [f for f in expected_feats if f in ercot_feats]
            missing = [f for f in expected_feats if f not in ercot_feats]
            print(f"\n  {model_name}: {len(avail)}/{len(expected_feats)} features available "
                  f"({len(missing)} missing — filled with 0)")

            X_aligned = pd.DataFrame(0.0, index=X_ercot_shared.index, columns=expected_feats)
            for f in avail:
                X_aligned[f] = ercot_test[f].values
        else:
            X_aligned = X_ercot_shared
            print(f"\n  {model_name}: using {len(shared_feats)} shared features")

        try:
            if "QRF" in model_name:
                preds = model.predict(X_aligned, quantiles=0.5)
            else:
                preds = model.predict(X_aligned)

            m_mae   = mae(y_ercot, preds)
            m_rmse  = rmse(y_ercot, preds)
            m_smape = smape(y_ercot, preds)
            print(f"  PJM→ERCOT {model_name}: MAE={m_mae:.4f} RMSE={m_rmse:.4f}")
            rows.append({
                "Direction": "PJM→ERCOT (cross-market)",
                "Model": model_name,
                "MAE":   round(m_mae, 4),
                "RMSE":  round(m_rmse, 4),
                "sMAPE": round(m_smape, 4),
            })
        except Exception as e:
            print(f"  ⚠️  {model_name} cross-market failed: {e}")

    # In-market ERCOT baselines (for degradation comparison)
    print(f"\n  In-market ERCOT baselines:")
    ercot_models = [
        ("LightGBM",  "lgbm_point_ercot.joblib"),
        ("XGBoost",   "xgboost_point_ercot.joblib"),
        ("QRF (q50)", "qrf_ercot.joblib"),
    ]
    X_ercot_full = ercot_test.drop(columns=[config.TARGET_COL])

    for model_name, fname in ercot_models:
        path = os.path.join(config.MODEL_DIR, fname)
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        try:
            if "QRF" in model_name:
                preds = model.predict(X_ercot_full, quantiles=0.5)
            else:
                preds = model.predict(X_ercot_full)
            m_mae   = mae(y_ercot, preds)
            m_rmse  = rmse(y_ercot, preds)
            m_smape = smape(y_ercot, preds)
            print(f"  ERCOT→ERCOT {model_name}: MAE={m_mae:.4f} RMSE={m_rmse:.4f}")
            rows.append({
                "Direction": "ERCOT→ERCOT (in-market)",
                "Model": model_name,
                "MAE":   round(m_mae, 4),
                "RMSE":  round(m_rmse, 4),
                "sMAPE": round(m_smape, 4),
            })
        except Exception as e:
            print(f"  ⚠️  {model_name} in-market failed: {e}")

    df = pd.DataFrame(rows)

    # Compute degradation %
    pivot = df.groupby(["Model", "Direction"])["MAE"].first().unstack()
    if "PJM→ERCOT (cross-market)" in pivot.columns and "ERCOT→ERCOT (in-market)" in pivot.columns:
        pivot["MAE_Degradation_%"] = ((pivot["PJM→ERCOT (cross-market)"] -
                                       pivot["ERCOT→ERCOT (in-market)"]) /
                                      pivot["ERCOT→ERCOT (in-market)"]) * 100
        print(f"\n  Cross-market MAE Degradation:")
        print(pivot["MAE_Degradation_%"].to_string())

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out_path = os.path.join(config.REPORT_DIR, "table_rq4_crossmarket.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved: {out_path}")

    if not pivot.empty:
        pivot.to_csv(os.path.join(config.REPORT_DIR, "table_rq4_pivot.csv"))

    return df


if __name__ == "__main__":
    run_rq4()
