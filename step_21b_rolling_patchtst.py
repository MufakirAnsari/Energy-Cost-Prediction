"""
step_21b_rolling_patchtst.py
=============================
Expanding-window retraining for PatchTST to fairly compare against rolling LightGBM.

Protocol:
  - Initial training: all data up to 2023-12-31
  - For each month M in [2024-01 ... 2025-12]:
      1. Train set = all data up to (M - 1 month)
      2. Test set = month M only
      3. Train fresh PatchTST (reduced max_steps for speed)
      4. Predict all hours in month M
"""

import os, sys, time
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE

def load_full_dataset(market: str) -> pd.DataFrame:
    is_pjm = market.upper() == "PJM"
    paths = [
        config.PJM_TRAIN_PATH if is_pjm else config.ERCOT_TRAIN_PATH,
        config.PJM_CAL_PATH   if is_pjm else config.ERCOT_CAL_PATH,
        config.PJM_VAL_PATH   if is_pjm else config.ERCOT_VAL_PATH,
        config.PJM_TEST_PATH  if is_pjm else config.ERCOT_TEST_PATH,
    ]
    dfs = [pd.read_parquet(p) for p in paths]
    full = pd.concat(dfs, axis=0)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return full

def to_nf(df):
    return pd.DataFrame({
        "unique_id": "price",
        "ds": pd.to_datetime(df.index, utc=True),
        "y":  df[config.TARGET_COL].values,
    })

def run_rolling_patchtst(market: str = "PJM"):
    print(f"\n{'='*65}\n  ROLLING PatchTST: {market}\n{'='*65}")

    full_df = load_full_dataset(market)
    if full_df.index.tz is None:
        full_df.index = full_df.index.tz_localize("UTC")

    months = pd.date_range(start="2024-01", end="2025-12", freq="MS", tz="UTC")
    results = []
    
    # Load static model predictions if available
    m = market.lower()
    static_preds_path = os.path.join(config.REPORT_DIR, f"patchtst_preds_{m}.csv")
    static_preds = None
    if os.path.exists(static_preds_path):
        static_preds = pd.read_csv(static_preds_path)
        static_preds["ds"] = pd.to_datetime(static_preds["ds"], utc=True)
        static_preds = static_preds.set_index("ds")
    
    all_rolling_preds = []
    t0 = time.time()
    
    for i, month_start in enumerate(months):
        if month_start.month == 12:
            month_end = pd.Timestamp(f"{month_start.year + 1}-01-01", tz="UTC")
        else:
            month_end = pd.Timestamp(f"{month_start.year}-{month_start.month + 1:02d}-01", tz="UTC")

        train_data = full_df[full_df.index < month_start]
        test_data = full_df[(full_df.index >= month_start) & (full_df.index < month_end)]

        if len(test_data) < 24 or len(train_data) < 1000:
            continue

        val_size = max(int(len(train_data) * 0.1), 168)
        
        # Prepare NF dataframe for training
        nf_train = to_nf(train_data)
        
        # Train fresh model for this window (using 1000 steps instead of 1500 for speed in rolling)
        model = PatchTST(
            h=24,
            input_size=config.SEQ_LEN_DEFAULT,
            patch_len=24,
            stride=12,
            hidden_size=64,
            linear_hidden_size=128,
            n_heads=4,
            encoder_layers=2,
            dropout=0.2,
            head_dropout=0.1,
            learning_rate=config.LEARNING_RATE,
            max_steps=1000,
            batch_size=config.BATCH_SIZE,
            loss=MAE(),
            valid_loss=MAE(),
            val_monitor="train_loss",
            scaler_type="standard",
            random_seed=config.RANDOM_SEED,
        )
        
        nf = NeuralForecast(models=[model], freq="h")
        
        # Temporarily suppress NF output to keep logs clean
        nf.fit(df=nf_train, val_size=val_size)
        
        # To predict exactly the test month, we need train_data + test_data
        nf_pred_input = to_nf(pd.concat([train_data, test_data]))
        # Cross validation for the test month only
        n_windows = len(test_data) // 24
        cv = nf.cross_validation(df=nf_pred_input, n_windows=n_windows, step_size=24)
        
        pred_col = [c for c in cv.columns if "PatchTST" in c][0]
        preds = cv[["ds", pred_col]].rename(columns={pred_col: "predicted"}).set_index("ds")
        preds.index = pd.to_datetime(preds.index, utc=True)
        # Handle duplicates from CV overlapping
        preds = preds[~preds.index.duplicated(keep="first")]
        
        # Extract subset matching test_data index exactly
        preds = preds.reindex(test_data.index)
        y_test = test_data[config.TARGET_COL].values
        pred_vals = preds["predicted"].values
        
        all_rolling_preds.append(preds)
        
        # Metrics
        mask = ~np.isnan(y_test) & ~np.isnan(pred_vals)
        mae = np.mean(np.abs(y_test[mask] - pred_vals[mask]))
        rmse = np.sqrt(np.mean((y_test[mask] - pred_vals[mask])**2))
        
        # Get static metrics for this month if available
        static_mae = np.nan
        static_rmse = np.nan
        if static_preds is not None:
            static_month = static_preds.reindex(test_data.index)
            stat_y = static_month["actual"].values
            stat_pred = static_month["predicted"].values
            stat_mask = ~np.isnan(stat_y) & ~np.isnan(stat_pred)
            if np.sum(stat_mask) > 0:
                static_mae = np.mean(np.abs(stat_y[stat_mask] - stat_pred[stat_mask]))
                static_rmse = np.sqrt(np.mean((stat_y[stat_mask] - stat_pred[stat_mask])**2))
        
        row = {
            "Market": market,
            "Month": month_start.strftime("%Y-%m"),
            "Rolling_PatchTST_MAE": round(mae, 4),
            "Rolling_PatchTST_RMSE": round(rmse, 4),
            "Static_PatchTST_MAE": round(static_mae, 4),
            "Static_PatchTST_RMSE": round(static_rmse, 4),
        }
        results.append(row)
        
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(months)}] {row['Month']} | "
              f"Train: {len(train_data):,}h | "
              f"MAE: {mae:.2f} (roll) vs {static_mae:.2f} (static) | {elapsed:.0f}s")
              
    results_df = pd.DataFrame(results)
    out_path = os.path.join(config.REPORT_DIR, f"table_rolling_patchtst_{m}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved: {out_path}")
    
    return results_df

if __name__ == "__main__":
    run_rolling_patchtst("PJM")
    run_rolling_patchtst("ERCOT")
