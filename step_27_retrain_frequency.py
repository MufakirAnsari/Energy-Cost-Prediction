"""
step_27_retrain_frequency.py

Tests different retraining frequencies (Weekly, Monthly, Quarterly, Semi-Annual, Static)
for LightGBM to justify the chosen monthly retraining protocol.
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config

def train_and_predict_lgb(X_tr, y_tr, X_val, y_val, X_te):
    model = lgb.LGBMRegressor(**config.LGBM_POINT_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model.predict(X_te)

def run_retrain_frequency_experiment(market, df, freqs, test_start, test_end):
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    df_test = df[test_mask]
    
    results = {}
    
    for freq_name, freq_rule in freqs.items():
        print(f"  Testing {freq_name} ({freq_rule})")
        if freq_rule == 'static':
            # Train once on all data prior to test_start
            train_mask = df.index < test_start
            df_train_full = df[train_mask]
            
            val_size = int(len(df_train_full) * 0.1)
            df_tr = df_train_full.iloc[:-val_size]
            df_val = df_train_full.iloc[-val_size:]
            
            X_tr, y_tr = df_tr.drop(columns=[config.TARGET_COL]), df_tr[config.TARGET_COL]
            X_val, y_val = df_val.drop(columns=[config.TARGET_COL]), df_val[config.TARGET_COL]
            X_te = df_test.drop(columns=[config.TARGET_COL])
            
            preds = train_and_predict_lgb(X_tr, y_tr, X_val, y_val, X_te)
            
            mae = mean_absolute_error(df_test[config.TARGET_COL], preds)
            rmse = np.sqrt(mean_squared_error(df_test[config.TARGET_COL], preds))
            results[freq_name] = {'MAE': mae, 'RMSE': rmse}
        else:
            # Generate retraining periods based on freq_rule inside test_end
            test_dates = df_test.resample(freq_rule).first().index
            
            preds_all = []
            actuals_all = []
            
            for i in range(len(test_dates)):
                window_start = test_dates[i]
                if i + 1 < len(test_dates):
                    window_end = test_dates[i+1]
                else:
                    window_end = pd.to_datetime(test_end).tz_localize('UTC') + pd.Timedelta(days=1)
                
                # Get window test data
                window_mask = (df_test.index >= window_start) & (df_test.index < window_end)
                df_window_test = df_test[window_mask]
                
                if len(df_window_test) == 0:
                    continue
                    
                # Train data is everything before window_start
                train_mask = df.index < window_start
                df_train_full = df[train_mask]
                
                val_size = int(len(df_train_full) * 0.1)
                df_tr = df_train_full.iloc[:-val_size]
                df_val = df_train_full.iloc[-val_size:]
                
                X_tr, y_tr = df_tr.drop(columns=[config.TARGET_COL]), df_tr[config.TARGET_COL]
                X_val, y_val = df_val.drop(columns=[config.TARGET_COL]), df_val[config.TARGET_COL]
                X_te = df_window_test.drop(columns=[config.TARGET_COL])
                
                preds = train_and_predict_lgb(X_tr, y_tr, X_val, y_val, X_te)
                
                preds_all.extend(preds)
                actuals_all.extend(df_window_test[config.TARGET_COL].values)
                
            mae = mean_absolute_error(actuals_all, preds_all)
            rmse = np.sqrt(mean_squared_error(actuals_all, preds_all))
            results[freq_name] = {'MAE': mae, 'RMSE': rmse}
            
    return results

def main():
    print("Starting step 27: Retrain Frequency Analysis")
    
    freqs = {
        'Weekly': 'W',
        'Monthly': 'MS',
        'Quarterly': 'QS',
        'Semi-Annual': '6MS',
        'Static': 'static'
    }
    
    freq_mapping = {
        'Weekly': 52,
        'Monthly': 12,
        'Quarterly': 4,
        'Semi-Annual': 2,
        'Static': 0
    }
    
    all_results = []
    
    for market, prefix in [('PJM', 'pjm'), ('ERCOT', 'ercot')]:
        print(f"\nProcessing {market}...")
        
        # Load all data
        df_train = pd.read_parquet(os.path.join(config.PROC_DIR, f"{prefix}_train.parquet"))
        df_cal = pd.read_parquet(os.path.join(config.PROC_DIR, f"{prefix}_calibration.parquet"))
        df_val = pd.read_parquet(os.path.join(config.PROC_DIR, f"{prefix}_val.parquet"))
        df_test = pd.read_parquet(os.path.join(config.PROC_DIR, f"{prefix}_test.parquet"))
        
        df_full = pd.concat([df_train, df_cal, df_val, df_test])
        df_full.sort_index(inplace=True)
        
        test_start = "2024-01-01"
        test_end = "2025-12-31"
        
        res = run_retrain_frequency_experiment(market, df_full, freqs, test_start, test_end)
        
        for k, v in res.items():
            all_results.append({
                'Market': market,
                'Frequency': k,
                'Windows_per_Year': freq_mapping[k],
                'MAE': v['MAE'],
                'RMSE': v['RMSE']
            })
            
    df_res = pd.DataFrame(all_results)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out_csv = os.path.join(config.REPORT_DIR, "table_retrain_frequency.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    
    # Plotting
    try:
        plt.style.use(config.PLOT_STYLE)
    except:
        pass
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for market in ['PJM', 'ERCOT']:
        df_m = df_res[df_res['Market'] == market].sort_values('Windows_per_Year')
        ax.plot(df_m['Windows_per_Year'], df_m['MAE'], marker='o', label=f'{market} MAE')
        
    ax.set_xlabel("Retraining Windows per Year (0 = Static)")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Impact of Retraining Frequency on Model Performance (LightGBM)")
    ax.set_xticks([0, 2, 4, 12, 52])
    ax.set_xticklabels(['Static (0)', 'Semi-Annual (2)', 'Quarterly (4)', 'Monthly (12)', 'Weekly (52)'])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig_dir = os.path.join(config.REPORT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "Fig_Retrain_Frequency.png")
    plt.savefig(fig_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to {fig_path}")

if __name__ == "__main__":
    main()
