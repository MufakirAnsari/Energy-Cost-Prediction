"""
step_15_ablation.py
===================
Ablation study — quantifies the contribution of each design choice.

Three ablation axes:
  1. Sequence length: [48h, 96h, 168h (default), 336h]
     → Tests whether longer context improves accuracy
  2. Feature set: [price-only, price+calendar, price+calendar+exogenous (default)]
     → Tests whether exogenous features add value
  3. Dropout rate (BiLSTM): [0.1, 0.2, 0.3, 0.4]
     → Already done in step_05; results loaded from CSV here

Results are saved as a table compatible with the paper's Table 4.

Run:
    python step_15_ablation.py
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
    import lightgbm as lgb
except ImportError:
    raise ImportError("pip install lightgbm")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SETS = {
    "price_only": lambda cols: [c for c in cols
                                if c.startswith("price_lag") or c == "price"],
    "price_calendar": lambda cols: [c for c in cols
                                    if any(c.startswith(p) for p in
                                           ["price_lag", "hour_", "dow_", "month_",
                                            "is_weekend", "price"])],
    "full": lambda cols: cols,   # All features (default)
}


def quick_lgbm_eval(X_tr, y_tr, X_v, y_v, X_te, y_te, label: str) -> dict:
    """Train a quick LightGBM and evaluate on test set (capped for speed)."""
    params = config.LGBM_POINT_PARAMS.copy()
    params["n_estimators"] = 300   # Hard cap — ablation needs speed not optimality
    params["num_leaves"]   = 31    # Smaller tree
    params["n_jobs"]       = -1    # Use all CPU cores
    params["verbose"]      = -1
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    pred = model.predict(X_te)
    mask = ~np.isnan(y_te)
    mae  = np.mean(np.abs(y_te[mask] - pred[mask]))
    rmse = np.sqrt(np.mean((y_te[mask] - pred[mask])**2))
    return {"config": label, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "n_features": X_tr.shape[1]}


def run_ablation(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY: {market}")
    print(f"{'='*65}")

    tr_df = pd.read_parquet(config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH)
    va_df = pd.read_parquet(config.PJM_VAL_PATH   if market == "PJM" else config.ERCOT_VAL_PATH)
    te_df = pd.read_parquet(config.PJM_TEST_PATH  if market == "PJM" else config.ERCOT_TEST_PATH)

    all_cols  = list(tr_df.columns)
    feat_cols = [c for c in all_cols if c != config.TARGET_COL]
    y_tr = tr_df[config.TARGET_COL].values
    y_v  = va_df[config.TARGET_COL].values
    y_te = te_df[config.TARGET_COL].values

    results = []

    # ── Ablation 1: Feature Set ────────────────────────────────────
    print(f"\n  [Ablation 1] Feature Set")
    for fs_name, fs_fn in FEATURE_SETS.items():
        selected = [c for c in fs_fn(feat_cols)]
        if len(selected) == 0:
            print(f"    {fs_name}: no features — skipping")
            continue
        X_tr = tr_df[selected].values
        X_v  = va_df[selected].values
        X_te = te_df[selected].values
        label = f"Feature={fs_name} ({len(selected)} feats)"
        print(f"    {label} ...", end=" ", flush=True)
        r = quick_lgbm_eval(X_tr, y_tr, X_v, y_v, X_te, y_te, label)
        results.append({"Ablation": "Feature Set", **r})
        print(f"MAE={r['MAE']:.4f}")

    # ── Ablation 2: Sequence Length (lag window) ───────────────────
    # We proxy "context length" by including lags only up to that horizon
    print(f"\n  [Ablation 2] Lag Window (Context Length)")
    lag_windows = [48, 96, 168, 336]
    for max_lag in lag_windows:
        # Include lags up to max_lag only
        selected = [c for c in feat_cols
                    if not c.startswith("price_lag_") or
                    int(c.split("_")[-1].replace("h", "")) <= max_lag]
        if len(selected) == 0:
            continue
        X_tr = tr_df[selected].values
        X_v  = va_df[selected].values
        X_te = te_df[selected].values
        label = f"LagWindow={max_lag}h ({len(selected)} feats)"
        print(f"    {label} ...", end=" ", flush=True)
        r = quick_lgbm_eval(X_tr, y_tr, X_v, y_v, X_te, y_te, label)
        results.append({"Ablation": "Lag Window", **r})
        print(f"MAE={r['MAE']:.4f}")

    # ── Ablation 3: BiLSTM Dropout (load from sweep CSV) ──────────
    print(f"\n  [Ablation 3] BiLSTM Dropout Rate (from sweep results)")
    sweep_path = os.path.join(
        config.REPORT_DIR, f"dropout_sweep_{market.lower()}.csv"
    )
    if os.path.exists(sweep_path):
        sweep_df = pd.read_csv(sweep_path)
        sweep_df["Ablation"] = "Dropout Rate"
        sweep_df["config"] = "dropout=" + sweep_df["dropout_rate"].astype(str)
        sweep_df = sweep_df.rename(columns={"ECE": "ECE"})
        print(sweep_df.to_string(index=False))
        results_dropout = sweep_df[["Ablation", "config", "ECE"]].copy()
    else:
        print(f"    Sweep file not found: {sweep_path}")
        print(f"    Run step_05_train_bilstm.py first.")
        results_dropout = pd.DataFrame()

    # ── Ablation 4: Ensemble Composition (A4) ─────────────────────
    # Uses existing prediction CSVs — no retraining needed.
    print(f"\n  [Ablation 4] Ensemble Composition (7 subsets)")
    m = market.lower()
    cal_df = pd.read_parquet(config.PJM_CAL_PATH if market=="PJM" else config.ERCOT_CAL_PATH)
    y_cal  = cal_df[config.TARGET_COL].values
    cal_idx = cal_df.index

    def load_cal_pred(fname, col):
        path = os.path.join(config.REPORT_DIR, fname)
        if not os.path.exists(path): return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="first")]
        if col not in df.columns: return None
        idx_utc = cal_idx.tz_localize("UTC") if cal_idx.tz is None else cal_idx.tz_convert("UTC")
        return df[col].reindex(idx_utc).values

    def load_test_pred_csv(fname, col):
        path = os.path.join(config.REPORT_DIR, fname)
        if not os.path.exists(path): return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="first")]
        if col not in df.columns: return None
        idx_utc = y_te_idx = te_df.index
        idx_utc = idx_utc.tz_localize("UTC") if idx_utc.tz is None else idx_utc.tz_convert("UTC")
        return df[col].reindex(idx_utc).values

    # Load cal-set predictions for meta-learner training
    lgbm_model = joblib.load(os.path.join(config.MODEL_DIR, f"lgbm_point_{m}.joblib")) \
                 if os.path.exists(os.path.join(config.MODEL_DIR, f"lgbm_point_{m}.joblib")) else None
    xgb_model  = joblib.load(os.path.join(config.MODEL_DIR, f"xgboost_point_{m}.joblib")) \
                 if os.path.exists(os.path.join(config.MODEL_DIR, f"xgboost_point_{m}.joblib")) else None

    X_cal = cal_df.drop(columns=[config.TARGET_COL])
    X_te  = te_df.drop(columns=[config.TARGET_COL])

    base_preds_cal  = {}
    base_preds_test = {}

    if lgbm_model:
        base_preds_cal["LGBM"]  = lgbm_model.predict(X_cal)
        base_preds_test["LGBM"] = lgbm_model.predict(X_te)
    if xgb_model:
        base_preds_cal["XGB"]   = xgb_model.predict(X_cal)
        base_preds_test["XGB"]  = xgb_model.predict(X_te)

    # BiLSTM cal-set predictions: generate on-the-fly from saved model
    # bilstm_preds CSV only covers the TEST set, so we cannot reindex onto cal_idx.
    # Instead, load the saved .keras model and run a single forward pass on cal data.
    bilstm_model_path = os.path.join(config.MODEL_DIR, f"bilstm_{m}.keras")
    bilstm_meta_path  = os.path.join(config.MODEL_DIR, f"bilstm_{m}_meta.joblib")
    if os.path.exists(bilstm_model_path) and os.path.exists(bilstm_meta_path):
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            bl_model = tf.keras.models.load_model(bilstm_model_path)
            bl_meta  = joblib.load(bilstm_meta_path)

            # Reconstruct scaling from saved meta
            col_min, col_max = bl_meta["scaler_min"], bl_meta["scaler_max"]
            denom  = bl_meta["scaler_denom"]
            t_idx  = bl_meta["target_idx"]
            sl     = bl_meta["seq_len"]

            # Build cal sequences using train+cal as history
            tr_df_bl = pd.read_parquet(config.PJM_TRAIN_PATH if market=="PJM" else config.ERCOT_TRAIN_PATH)
            history_bl = pd.concat([tr_df_bl, cal_df])
            scaled_bl  = (history_bl.values.astype(np.float32) - col_min) / denom
            n_hist = len(tr_df_bl)

            X_cal_bl, y_cal_bl = [], []
            for i in range(n_hist - sl, n_hist - sl + len(cal_df)):
                if i + sl < len(scaled_bl):
                    X_cal_bl.append(scaled_bl[i: i + sl])
                    y_cal_bl.append(scaled_bl[i + sl, t_idx])
            X_cal_bl = np.array(X_cal_bl, dtype=np.float32)

            # Single forward pass (no MC dropout — just point prediction for meta-learner)
            cal_preds_scaled = bl_model(X_cal_bl, training=False).numpy().flatten()
            cal_preds_price  = cal_preds_scaled * denom[t_idx] + col_min[t_idx]

            # Truncate y_cal to match (sequences may be shorter than cal_df)
            n_bl = len(cal_preds_price)
            base_preds_cal["BiLSTM"] = cal_preds_price[:len(y_cal)]

            # Load test predictions from CSV
            bl_test_col = None
            bl_test_path = os.path.join(config.REPORT_DIR, f"bilstm_preds_{m}.csv")
            if os.path.exists(bl_test_path):
                bl_cols = pd.read_csv(bl_test_path, nrows=0).columns
                bl_test_col = next((c for c in bl_cols if "pred" in c.lower() or "mean" in c.lower()), None)
            if bl_test_col:
                bl_test = load_test_pred_csv(f"bilstm_preds_{m}.csv", bl_test_col)
                if bl_test is not None:
                    base_preds_test["BiLSTM"] = bl_test

            print(f"    BiLSTM loaded: cal={n_bl} preds, test={'yes' if 'BiLSTM' in base_preds_test else 'no'}")
        except Exception as e:
            print(f"    BiLSTM ablation skipped: {e}")

    # Define subsets
    base_names = [b for b in ["LGBM","XGB","BiLSTM"] if b in base_preds_cal]
    subsets = []
    for i in range(1, 2**len(base_names)):
        subset = [base_names[j] for j in range(len(base_names)) if i & (1 << j)]
        subsets.append(subset)

    results_ens = []
    try:
        import lightgbm as lgb
        for subset in subsets:
            X_meta_cal  = np.column_stack([base_preds_cal[b]  for b in subset])
            X_meta_test = np.column_stack([base_preds_test[b] for b in subset])
            valid_cal   = ~np.any(np.isnan(X_meta_cal), axis=1) & ~np.isnan(y_cal)
            if valid_cal.sum() < 100:
                continue
            meta = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                     n_jobs=-1, verbose=-1, random_state=42)
            meta.fit(X_meta_cal[valid_cal], y_cal[valid_cal])
            ens_pred = meta.predict(X_meta_test)
            valid_te = ~np.any(np.isnan(X_meta_test), axis=1) & ~np.isnan(y_te)
            e_mae  = np.mean(np.abs(y_te[valid_te] - ens_pred[valid_te]))
            e_rmse = np.sqrt(np.mean((y_te[valid_te] - ens_pred[valid_te])**2))
            label  = "+".join(subset)
            print(f"    [{label:25}]  MAE={e_mae:.4f}  RMSE={e_rmse:.4f}")
            results_ens.append({"Ablation": "Ensemble Composition",
                                 "config": label,
                                 "MAE": round(e_mae, 4),
                                 "RMSE": round(e_rmse, 4),
                                 "n_features": len(subset)})
    except Exception as e:
        print(f"    Ensemble ablation failed: {e}")
    results_ens_df = pd.DataFrame(results_ens) if results_ens else pd.DataFrame()


    # ── Save combined results ────────────────────────────────
    abl_df = pd.DataFrame(results)
    print(f"\n  Ablation Results (Feature Set + Lag Window):")
    print(abl_df.to_string(index=False))

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out = os.path.join(config.REPORT_DIR, f"table_ablation_{market.lower()}.csv")
    abl_df.to_csv(out, index=False)

    if not results_dropout.empty:
        drop_out = os.path.join(config.REPORT_DIR,
                                f"table_ablation_dropout_{market.lower()}.csv")
        results_dropout.to_csv(drop_out, index=False)

    if not results_ens_df.empty:
        ens_out = os.path.join(config.REPORT_DIR,
                               f"table_ablation_ensemble_{market.lower()}.csv")
        results_ens_df.to_csv(ens_out, index=False)
        print(f"\n  Ensemble Composition Ablation:")
        print(results_ens_df.to_string(index=False))

    print(f"\n  ✅ Ablation complete. Saved: {out}")
    return abl_df


if __name__ == "__main__":
    run_ablation("PJM")
    run_ablation("ERCOT")
