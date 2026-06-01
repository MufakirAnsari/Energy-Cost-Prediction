"""
step_21_rolling_window.py
=========================
Expanding-window retraining experiment for LightGBM and XGBoost.

Addresses the key reviewer concern: "Why didn't you use rolling/expanding
window retraining like standard EPF literature?"

Protocol:
  - Initial training: all data up to 2023-12-31 (train + cal + val)
  - For each month M in [2024-01 ... 2025-12]:
      1. Train set = all data from start up to (M - 1 month)
      2. Test set  = month M only
      3. Train fresh LightGBM + XGBoost with config params
      4. Predict all hours in month M
      5. Record per-month MAE, RMSE
  - Compare vs. single-split (static model trained once)

Run:
    python step_21_rolling_window.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_full_dataset(market: str) -> pd.DataFrame:
    """Load and concatenate all splits into one chronological DataFrame."""
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


def train_lgbm(X_tr, y_tr, X_val, y_val):
    """Train LightGBM with early stopping."""
    model = lgb.LGBMRegressor(**config.LGBM_POINT_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ],
    )
    return model


def train_xgb(X_tr, y_tr, X_val, y_val):
    """Train XGBoost with early stopping."""
    params = config.XGB_PARAMS.copy()
    params.pop("n_estimators", None)
    model = xgb.XGBRegressor(
        **params,
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )
    return model


def run_rolling_window(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  EXPANDING-WINDOW RETRAINING: {market}")
    print(f"{'='*65}")

    full_df = load_full_dataset(market)
    target = config.TARGET_COL
    print(f"  Full dataset: {len(full_df):,} hours | "
          f"{full_df.index.min().date()} → {full_df.index.max().date()}")

    # Ensure timezone-aware
    if full_df.index.tz is None:
        full_df.index = full_df.index.tz_localize("UTC")

    # Test period: 2024-01-01 → 2025-12-31
    test_start = pd.Timestamp("2024-01-01", tz="UTC")
    test_end   = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")

    # Generate monthly windows
    months = pd.date_range(start="2024-01", end="2025-12", freq="MS", tz="UTC")

    # Also load the static (single-split) model for comparison
    m = market.lower()
    static_lgbm_path = os.path.join(config.MODEL_DIR, f"lgbm_point_{m}.joblib")
    static_xgb_path  = os.path.join(config.MODEL_DIR, f"xgboost_point_{m}.joblib")

    static_lgbm = joblib.load(static_lgbm_path) if os.path.exists(static_lgbm_path) else None
    static_xgb  = joblib.load(static_xgb_path)  if os.path.exists(static_xgb_path)  else None

    results = []
    all_rolling_lgbm_preds = []
    all_rolling_xgb_preds  = []

    t0 = time.time()
    for i, month_start in enumerate(months):
        # Month boundaries
        if month_start.month == 12:
            month_end = pd.Timestamp(f"{month_start.year + 1}-01-01", tz="UTC")
        else:
            month_end = pd.Timestamp(f"{month_start.year}-{month_start.month + 1:02d}-01", tz="UTC")

        # Training: everything before this month
        train_data = full_df[full_df.index < month_start]
        # Test: this month only
        test_data = full_df[(full_df.index >= month_start) & (full_df.index < month_end)]

        if len(test_data) < 24 or len(train_data) < 1000:
            continue

        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        X_test  = test_data.drop(columns=[target])
        y_test  = test_data[target]

        # Use last 10% of training data as validation for early stopping
        val_size = max(int(len(X_train) * 0.1), 168)
        X_tr_sub = X_train.iloc[:-val_size]
        y_tr_sub = y_train.iloc[:-val_size]
        X_val    = X_train.iloc[-val_size:]
        y_val    = y_train.iloc[-val_size:]

        month_label = month_start.strftime("%Y-%m")

        # Train fresh models for this window
        lgbm_model = train_lgbm(X_tr_sub, y_tr_sub, X_val, y_val)
        xgb_model  = train_xgb(X_tr_sub, y_tr_sub, X_val, y_val)

        # Rolling predictions
        lgbm_preds = lgbm_model.predict(X_test)
        xgb_preds  = xgb_model.predict(X_test)

        # Static predictions (single-split model)
        static_lgbm_preds = static_lgbm.predict(X_test) if static_lgbm else np.full(len(X_test), np.nan)
        static_xgb_preds  = static_xgb.predict(X_test)  if static_xgb  else np.full(len(X_test), np.nan)

        y = y_test.values

        # Store rolling predictions for aggregate metrics
        all_rolling_lgbm_preds.append(pd.Series(lgbm_preds, index=test_data.index))
        all_rolling_xgb_preds.append(pd.Series(xgb_preds, index=test_data.index))

        # Per-month metrics
        row = {
            "Market": market,
            "Month": month_label,
            "N_hours": len(y),
            "Train_hours": len(X_train),
            # Rolling LightGBM
            "Rolling_LGBM_MAE":  round(np.mean(np.abs(y - lgbm_preds)), 4),
            "Rolling_LGBM_RMSE": round(np.sqrt(np.mean((y - lgbm_preds)**2)), 4),
            # Static LightGBM
            "Static_LGBM_MAE":   round(np.mean(np.abs(y - static_lgbm_preds)), 4),
            "Static_LGBM_RMSE":  round(np.sqrt(np.mean((y - static_lgbm_preds)**2)), 4),
            # Rolling XGBoost
            "Rolling_XGB_MAE":   round(np.mean(np.abs(y - xgb_preds)), 4),
            "Rolling_XGB_RMSE":  round(np.sqrt(np.mean((y - xgb_preds)**2)), 4),
            # Static XGBoost
            "Static_XGB_MAE":    round(np.mean(np.abs(y - static_xgb_preds)), 4),
            "Static_XGB_RMSE":   round(np.sqrt(np.mean((y - static_xgb_preds)**2)), 4),
        }
        results.append(row)

        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(months)}] {month_label} | "
              f"Train: {len(X_train):,}h | Test: {len(X_test)}h | "
              f"LGBM MAE: {row['Rolling_LGBM_MAE']:.2f} (roll) vs {row['Static_LGBM_MAE']:.2f} (static) | "
              f"{elapsed:.0f}s")

    results_df = pd.DataFrame(results)

    # Aggregate metrics
    print(f"\n  {'─'*55}")
    print(f"  AGGREGATE RESULTS ({market}):")

    # Concatenate all rolling predictions
    all_r_lgbm = pd.concat(all_rolling_lgbm_preds)
    all_r_xgb  = pd.concat(all_rolling_xgb_preds)

    # Get corresponding actuals
    test_df = full_df[(full_df.index >= test_start) & (full_df.index <= test_end)]
    y_all = test_df[target].reindex(all_r_lgbm.index).values
    mask = ~np.isnan(y_all)

    agg = {
        "Rolling_LGBM_MAE":  np.mean(np.abs(y_all[mask] - all_r_lgbm.values[mask])),
        "Static_LGBM_MAE":   results_df["Static_LGBM_MAE"].mean(),
        "Rolling_XGB_MAE":   np.mean(np.abs(y_all[mask] - all_r_xgb.values[mask])),
        "Static_XGB_MAE":    results_df["Static_XGB_MAE"].mean(),
    }

    for k, v in agg.items():
        print(f"    {k}: {v:.4f}")

    lgbm_improvement = (agg["Static_LGBM_MAE"] - agg["Rolling_LGBM_MAE"]) / agg["Static_LGBM_MAE"] * 100
    xgb_improvement  = (agg["Static_XGB_MAE"]  - agg["Rolling_XGB_MAE"])  / agg["Static_XGB_MAE"]  * 100
    print(f"\n    LGBM: Rolling {'improves' if lgbm_improvement > 0 else 'worsens'} MAE by {abs(lgbm_improvement):.1f}%")
    print(f"    XGB:  Rolling {'improves' if xgb_improvement > 0 else 'worsens'} MAE by {abs(xgb_improvement):.1f}%")

    # Save
    out_path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{m}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved: {out_path}")

    return results_df


def generate_figure():
    """Generate comparison figure for both markets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, market in enumerate(["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{market}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        # Top row: LightGBM
        ax = axes[0, col_idx]
        ax.plot(df["Month"], df["Rolling_LGBM_MAE"], "o-", color="#2ca02c",
               label="Rolling (monthly retrain)", linewidth=1.5, markersize=4)
        ax.plot(df["Month"], df["Static_LGBM_MAE"], "s--", color="#aec7e8",
               label="Static (single split)", linewidth=1.5, markersize=4)
        ax.set_ylabel("MAE ($/MWh)")
        ax.set_title(f"{market.upper()}: LightGBM", fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

        # Bottom row: XGBoost
        ax = axes[1, col_idx]
        ax.plot(df["Month"], df["Rolling_XGB_MAE"], "o-", color="#ff7f0e",
               label="Rolling (monthly retrain)", linewidth=1.5, markersize=4)
        ax.plot(df["Month"], df["Static_XGB_MAE"], "s--", color="#aec7e8",
               label="Static (single split)", linewidth=1.5, markersize=4)
        ax.set_ylabel("MAE ($/MWh)")
        ax.set_xlabel("Month")
        ax.set_title(f"{market.upper()}: XGBoost", fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("Expanding-Window Retraining vs Static Single-Split\n"
                 "Monthly retraining with growing training set",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"FigW6_Rolling_Window.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: FigW6_Rolling_Window")


if __name__ == "__main__":
    run_rolling_window("PJM")
    run_rolling_window("ERCOT")
    generate_figure()
    print("\n  Done! Rolling window experiment complete.")
