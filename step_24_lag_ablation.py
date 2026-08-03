"""
step_24_lag_ablation.py
=======================
Addresses reviewer concern regarding the exclusion of short lags (<24h).
Demonstrates that if the 1h lag is included, it dominates the predictions,
thus verifying the paper's claim that we excluded them to prevent leakage/persistence domination
at the day-ahead forecast origin.

It performs a lag ablation study on three feature sets:
  A: Current (lags >= 24h)
  B: With Short Lags (+ 1, 2, 4, 6, 12h)
  C: Price-Only Short (1h lag + calendar features)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import config
from utils import mae, rmse, dm_test

def reconstruct_and_split(market: str):
    print(f"\n[INFO] Loading {market.upper()} datasets...")
    
    if market == 'pjm':
        train_path = config.PJM_TRAIN_PATH
        cal_path = config.PJM_CAL_PATH
        val_path = config.PJM_VAL_PATH
        test_path = config.PJM_TEST_PATH
    else:
        train_path = config.ERCOT_TRAIN_PATH
        cal_path = config.ERCOT_CAL_PATH
        val_path = config.ERCOT_VAL_PATH
        test_path = config.ERCOT_TEST_PATH
        
    df_train = pd.read_parquet(train_path)
    df_cal = pd.read_parquet(cal_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    # Store indices to split back later
    idx_train = df_train.index
    idx_cal = df_cal.index
    idx_val = df_val.index
    idx_test = df_test.index
    
    # Concatenate all by time to compute continuous short lags
    df_all = pd.concat([df_train, df_cal, df_val, df_test]).sort_index()
    
    # Lags to reconstruct
    short_lags = [1, 2, 4, 6, 12]
    short_lag_cols = []
    for lag in short_lags:
        col = f"price_lag_{lag}h"
        short_lag_cols.append(col)
        df_all[col] = df_all[config.TARGET_COL].shift(lag)
        
    # Re-split (ensuring strict chronological bounds match original)
    df_train_new = df_all.loc[idx_train].copy()
    df_val_new = df_all.loc[idx_val].copy()
    df_test_new = df_all.loc[idx_test].copy()
    
    # Drop NaNs in train caused by shift (very first 12 hours)
    df_train_new = df_train_new.dropna(subset=short_lag_cols)
    
    return df_train_new, df_val_new, df_test_new, short_lag_cols

def train_and_eval(X_tr, y_tr, X_v, y_v, X_ts, y_ts, name: str):
    print(f"  Training Feature Set: {name} ({X_tr.shape[1]} features)")
    t0 = time.time()
    
    # Instantiate the model with LightGBM point params
    model = lgb.LGBMRegressor(**config.LGBM_POINT_PARAMS)
    
    # Fit with early stopping on the validation set
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.PATIENCE, verbose=False)
        ]
    )
    
    # Predict and evaluate on the test set
    preds = model.predict(X_ts)
    m_mae = mae(y_ts.values, preds)
    m_rmse = rmse(y_ts.values, preds)
    
    print(f"  -> Test MAE: {m_mae:.3f} | Test RMSE: {m_rmse:.3f} | Time: {time.time()-t0:.1f}s")
    
    return preds, m_mae, m_rmse

def main():
    # Ensure reporting directories exist
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.REPORT_DIR, "figures"), exist_ok=True)
    
    results = []
    
    for market in ['pjm', 'ercot']:
        print(f"\n{'='*50}")
        print(f" MARKET: {market.upper()}")
        print(f"{'='*50}")
        
        # Load data and reconstruct short-term lag features
        df_train, df_val, df_test, short_lag_cols = reconstruct_and_split(market)
        
        y_train = df_train[config.TARGET_COL]
        y_val = df_val[config.TARGET_COL]
        y_test = df_test[config.TARGET_COL]
        
        X_train = df_train.drop(columns=[config.TARGET_COL])
        X_val = df_val.drop(columns=[config.TARGET_COL])
        X_test = df_test.drop(columns=[config.TARGET_COL])
        
        # Define the three distinct feature sets
        
        # A: Current (all original features, explicitly exclude the newly created short lags)
        features_a = [c for c in X_train.columns if c not in short_lag_cols]
        
        # B: With short lags (original + newly created short lags)
        features_b = list(X_train.columns)
        
        # C: Price-Only Short (Only the 1h lag + standard calendar features)
        calendar_cols = [c for c in X_train.columns if any(cal in c for cal in ['hour', 'dow', 'week', 'month', 'year', 'day'])]
        features_c = ['price_lag_1h'] + calendar_cols
        
        feature_sets = {
            'A_Current': features_a,
            'B_Short_Lags': features_b,
            'C_Price_1h_Only': features_c
        }
        
        preds_dict = {}
        
        # Train and evaluate for each feature set
        for name, feats in feature_sets.items():
            preds, m_mae, m_rmse = train_and_eval(
                X_train[feats], y_train,
                X_val[feats], y_val,
                X_test[feats], y_test,
                name
            )
            preds_dict[name] = preds
            
            results.append({
                'Market': market.upper(),
                'Feature Set': name,
                'Test MAE': m_mae,
                'Test RMSE': m_rmse
            })
            
        # Perform Diebold-Mariano Test between Set A and Set B
        print("\n  [DM Test] Comparing A (Current) vs B (With Short Lags)")
        try:
            # We use h=24 because this is a day-ahead forecasting task
            dm_stat, dm_pval = dm_test(y_test.values, preds_dict['A_Current'], preds_dict['B_Short_Lags'], h=24)
            print(f"  DM Statistic: {dm_stat:.3f}, p-value: {dm_pval:.3e}")
        except Exception as e:
            print(f"  DM Test failed: {e}")
            
    # ── Save Results to CSV ──────────────────────────────────────
    df_results = pd.DataFrame(results)
    results_path = os.path.join(config.REPORT_DIR, 'table_lag_ablation.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\n[INFO] Saved numerical results to {results_path}")
    
    # ── Generate Aesthetic Figure ──────────────────────────────────
    print("[INFO] Generating ablation figure...")
    plt.style.use(config.PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    colors = ['#2ca02c', '#d62728', '#1f77b4'] # Green, Red, Blue
    labels = ["A: Current\n(Lags >= 24h)", "B: + Short Lags\n({1,2,4,6,12}h)", "C: 1h Lag + Cal\n(Pure Persistence)"]
    
    for i, market in enumerate(['PJM', 'ERCOT']):
        ax = axes[i]
        market_res = df_results[df_results['Market'] == market.upper()]
        
        # Ensure strict ordering for plotting
        maes = [
            market_res[market_res['Feature Set'] == 'A_Current']['Test MAE'].values[0],
            market_res[market_res['Feature Set'] == 'B_Short_Lags']['Test MAE'].values[0],
            market_res[market_res['Feature Set'] == 'C_Price_1h_Only']['Test MAE'].values[0]
        ]
        
        bars = ax.bar(labels, maes, color=colors, edgecolor='black', linewidth=1.0, width=0.6)
        
        ax.set_title(f"{market.upper()} Market", fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Test MAE ($/MWh)", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Annotate MAE values cleanly on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), 
                    f"{yval:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
            
    # Add an overarching title for context
    plt.suptitle("Impact of Short-Term Lags (<24h) on Forecast Error\n(Addressing reviewer concern on lag exclusion)", 
                 fontsize=15, y=1.05)
    plt.tight_layout()
    
    # Save high-res figure
    fig_path = os.path.join(config.REPORT_DIR, 'figures', 'fig_lag_ablation.png')
    plt.savefig(fig_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved figure to {fig_path}")

if __name__ == '__main__':
    main()
