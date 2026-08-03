"""
step_25_train_lear.py
=====================
Implements the LEAR (LASSO Estimated AutoRegressive) model 
from Uniejewski et al. (2019).

Trains 24 separate LASSO models (one per hour) using CV for penalty selection.
Features:
- Lags: 24h, 48h, 168h
- Yesterday's daily min, max, mean
- Day-of-week dummies
- Holiday indicator

Run:
    python step_25_train_lear.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import holidays
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
import config
from utils import mae, rmse, smape

def create_lear_features(df_price):
    """
    Reconstruct the LEAR-specific features from the target column.
    df_price must have a DatetimeIndex and one column 'price'.
    """
    df = df_price.copy()
    target = config.TARGET_COL
    
    # 1. Autoregressive lags
    df['price_lag_24'] = df[target].shift(24)
    df['price_lag_48'] = df[target].shift(48)
    df['price_lag_168'] = df[target].shift(168)
    
    # 2. Yesterday's daily stats
    df['date'] = df.index.date
    daily_stats = df.groupby('date')[target].agg(['min', 'max', 'mean'])
    daily_stats = daily_stats.shift(1) # yesterday
    daily_stats.columns = ['yest_min', 'yest_max', 'yest_mean']
    
    df = df.join(daily_stats, on='date')
    df = df.drop(columns=['date'])
    
    # 3. Day of week dummies
    dow = df.index.dayofweek
    for i in range(1, 7):
        df[f'dow_{i}'] = (dow == i).astype(int)
        
    # 4. Holiday indicator
    us_holidays = holidays.US(years=df.index.year.unique().tolist())
    df['is_holiday'] = df.index.map(lambda d: int(d in us_holidays)).values
    
    return df.dropna()

def run_lear_for_market(market: str):
    print(f"\n" + "="*50)
    print(f" RUNNING LEAR MODEL FOR {market}")
    print("="*50)
    
    if market == "PJM":
        train_path = config.PJM_TRAIN_PATH
        val_path   = config.PJM_VAL_PATH
        test_path  = config.PJM_TEST_PATH
    else:
        train_path = config.ERCOT_TRAIN_PATH
        val_path   = config.ERCOT_VAL_PATH
        test_path  = config.ERCOT_TEST_PATH
        
    # Load just the price column
    df_train = pd.read_parquet(train_path)[[config.TARGET_COL]]
    df_val   = pd.read_parquet(val_path)[[config.TARGET_COL]]
    df_test  = pd.read_parquet(test_path)[[config.TARGET_COL]]
    
    df_all = pd.concat([df_train, df_val, df_test])
    df_feat = create_lear_features(df_all)
    
    # We train on train + val
    train_val_end = df_val.index[-1]
    
    df_train_val = df_feat.loc[:train_val_end]
    df_test_feats = df_feat.loc[df_test.index[0]:]
    
    # Will store predictions
    preds = np.full(len(df_test_feats), np.nan)
    
    # We need to train 24 separate models
    models = {}
    
    features = [c for c in df_feat.columns if c != config.TARGET_COL]
    
    t0 = time.time()
    
    for h in range(24):
        # Subset data for hour h
        hour_train_val = df_train_val[df_train_val.index.hour == h]
        hour_test = df_test_feats[df_test_feats.index.hour == h]
        
        if len(hour_train_val) == 0 or len(hour_test) == 0:
            print(f"  Hour {h}: missing data!")
            continue
            
        X_tr = hour_train_val[features].values
        y_tr = hour_train_val[config.TARGET_COL].values
        X_te = hour_test[features].values
        
        # LassoCV with 5-fold CV
        # Use a scaler because Lasso is sensitive to scale
        # KFold for standard cross validation
        cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LassoCV(cv=cv, random_state=config.RANDOM_SEED, max_iter=10000, n_jobs=-1))
        ])
        
        model.fit(X_tr, y_tr)
        
        # Predict
        hour_preds = model.predict(X_te)
        
        # Place predictions back in their original index
        # We find the integer positions of hour_test.index in df_test_feats.index
        # An easier way is boolean indexing
        mask = df_test_feats.index.hour == h
        preds[mask] = hour_preds
        
        # Optional: Print progress
        if h % 6 == 0:
            print(f"  Trained hour {h:02d} ... alpha={model.named_steps['lasso'].alpha_:.4f}")
            
    t1 = time.time()
    print(f"  Training & Prediction completed in {t1 - t0:.1f} seconds")
    
    # Compile results
    df_results = pd.DataFrame(index=df_test_feats.index)
    df_results['Actual'] = df_test_feats[config.TARGET_COL]
    df_results['LEAR'] = preds
    
    # Save predictions
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out_csv = os.path.join(config.REPORT_DIR, f"lear_preds_{market.lower()}.csv")
    df_results.to_csv(out_csv)
    print(f"  Saved predictions to {out_csv}")
    
    # Metrics
    m_mae = mae(df_results['Actual'], df_results['LEAR'])
    m_rmse = rmse(df_results['Actual'], df_results['LEAR'])
    m_smape = smape(df_results['Actual'], df_results['LEAR'])
    
    print(f"  Metrics for {market}:")
    print(f"    MAE:   {m_mae:.2f}")
    print(f"    RMSE:  {m_rmse:.2f}")
    print(f"    sMAPE: {m_smape:.2f}%")
    
    return {
        "Market": market,
        "MAE": m_mae,
        "RMSE": m_rmse,
        "sMAPE": m_smape
    }

if __name__ == "__main__":
    results = []
    for mkt in ["PJM", "ERCOT"]:
        res = run_lear_for_market(mkt)
        results.append(res)
        
    df_metrics = pd.DataFrame(results)
    out_table = os.path.join(config.REPORT_DIR, "table_lear_results.csv")
    df_metrics.to_csv(out_table, index=False)
    print(f"\nSaved combined metrics to {out_table}")
    print(df_metrics)

