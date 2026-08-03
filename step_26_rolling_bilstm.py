import os
import gc
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import subprocess

import config

def create_sequences(features, targets, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(targets[i + lookback])
    return np.array(X), np.array(y)

def load_and_combine_data(market):
    if market == 'PJM':
        train = pd.read_parquet(config.PJM_TRAIN_PATH)
        cal = pd.read_parquet(config.PJM_CAL_PATH)
        val = pd.read_parquet(config.PJM_VAL_PATH)
        test = pd.read_parquet(config.PJM_TEST_PATH)
    else:
        train = pd.read_parquet(config.ERCOT_TRAIN_PATH)
        cal = pd.read_parquet(config.ERCOT_CAL_PATH)
        val = pd.read_parquet(config.ERCOT_VAL_PATH)
        test = pd.read_parquet(config.ERCOT_TEST_PATH)
    
    df = pd.concat([train, cal, val, test])
    if df.index.name is not None and df.index.name.startswith('datetime'):
        df = df.rename_axis('datetime').reset_index()
    elif 'datetime' not in df.columns:
        df = df.reset_index(names='datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def run_single_window(market, current_month_str):
    # This runs in a separate process to avoid TF memory leaks
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass

    def build_bilstm_model(lookback, n_features):
        model = Sequential([
            Input(shape=(lookback, n_features)),
            Bidirectional(LSTM(config.BILSTM_UNITS, return_sequences=True, kernel_regularizer=l2(config.L2_REG))),
            Dropout(config.BILSTM_DROPOUT_RATE),
            Bidirectional(LSTM(config.BILSTM_DENSE_UNITS, return_sequences=False, kernel_regularizer=l2(config.L2_REG))),
            Dropout(config.BILSTM_DROPOUT_RATE),
            Dense(config.BILSTM_DENSE_UNITS, activation='relu', kernel_regularizer=l2(config.L2_REG)),
            Dropout(config.BILSTM_DROPOUT_RATE),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='mae')
        return model

    df = load_and_combine_data(market)
    features_cols = [c for c in df.columns if c not in ['datetime', config.TARGET_COL]]
    
    current_month = pd.Timestamp(current_month_str)
    next_month = current_month + relativedelta(months=1)
    
    train_mask = df['datetime'] < current_month
    train_df = df[train_mask].copy()
    
    full_mask = df['datetime'] < next_month
    full_df = df[full_mask].copy()
    
    if len(train_df) == 0:
        return json.dumps({"error": "No training data"})
        
    X_train_raw = train_df[features_cols].values
    y_train_raw = train_df[config.TARGET_COL].values
    
    X_full_raw = full_df[features_cols].values
    y_full_raw = full_df[config.TARGET_COL].values
    
    y_train_arcsinh = np.arcsinh(y_train_raw)
    y_full_arcsinh = np.arcsinh(y_full_raw)
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    y_train_scaled = target_scaler.fit_transform(y_train_arcsinh.reshape(-1, 1))
    
    X_full_scaled = feature_scaler.transform(X_full_raw)
    y_full_scaled = target_scaler.transform(y_full_arcsinh.reshape(-1, 1))
    
    lookback = 168
    X_seq_full, y_seq_full = create_sequences(X_full_scaled, y_full_scaled, lookback)
    
    target_datetimes = pd.to_datetime(full_df['datetime'].values[lookback:])
    train_seq_mask = target_datetimes < current_month
    test_seq_mask = (target_datetimes >= current_month) & (target_datetimes < next_month)
    
    X_train_seq = X_seq_full[train_seq_mask]
    y_train_seq = y_seq_full[train_seq_mask]
    X_test_seq = X_seq_full[test_seq_mask]
    y_test_seq = y_seq_full[test_seq_mask]
    
    if len(X_test_seq) == 0:
        return json.dumps({"error": "No test data"})
        
    val_split_idx = int(len(X_train_seq) * 0.9)
    X_t, y_t = X_train_seq[:val_split_idx], y_train_seq[:val_split_idx]
    X_v, y_v = X_train_seq[val_split_idx:], y_train_seq[val_split_idx:]
    
    model = build_bilstm_model(lookback, len(features_cols))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=30, batch_size=config.BATCH_SIZE, callbacks=[es], verbose=0)
    preds_scaled = model.predict(X_test_seq, batch_size=config.BATCH_SIZE, verbose=0)
    
    preds_arcsinh = target_scaler.inverse_transform(preds_scaled)
    preds = np.sinh(preds_arcsinh).flatten()
    
    actuals_arcsinh = target_scaler.inverse_transform(y_test_seq)
    actuals = np.sinh(actuals_arcsinh).flatten()
    
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    
    return json.dumps({"MAE": float(mae), "RMSE": float(rmse)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--market', type=str)
    parser.add_argument('--month', type=str)
    args = parser.parse_args()

    if args.worker:
        result = run_single_window(args.market, args.month)
        print(f"WORKER_RESULT:{result}")
        sys.exit(0)
        
    # Controller mode
    results = []
    
    for market in ['PJM', 'ERCOT']:
        print(f"--- Running Rolling BiLSTM for {market} ---")
        start_month = pd.Timestamp('2024-01-01')
        end_month = pd.Timestamp('2025-12-01')
        current_month = start_month
        
        while current_month <= end_month:
            month_str = current_month.strftime('%Y-%m-%d')
            display_month = current_month.strftime('%Y-%m')
            print(f"[{market}] Training up to {display_month}, Testing on {display_month}")
            
            # Call itself as a subprocess
            cmd = [sys.executable, __file__, '--worker', '--market', market, '--month', month_str]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = proc.stdout
                
                # Parse output
                res_dict = None
                for line in output.split('\n'):
                    if line.startswith("WORKER_RESULT:"):
                        res_dict = json.loads(line.replace("WORKER_RESULT:", ""))
                        break
                
                if res_dict and "MAE" in res_dict:
                    print(f"[{market}] {display_month} MAE: {res_dict['MAE']:.2f}, RMSE: {res_dict['RMSE']:.2f}")
                    results.append({
                        'Market': market,
                        'Month': display_month,
                        'MAE': res_dict['MAE'],
                        'RMSE': res_dict['RMSE']
                    })
                elif res_dict and "error" in res_dict:
                    print(f"[{market}] {display_month} Error: {res_dict['error']}")
                else:
                    print(f"[{market}] {display_month} Failed to parse worker output. Stdout:\n{output}\nStderr:\n{proc.stderr}")
                    
            except subprocess.CalledProcessError as e:
                print(f"[{market}] {display_month} Process crashed with code {e.returncode}. Stderr:\n{e.stderr}")
                
            current_month += relativedelta(months=1)
            
    all_results = pd.DataFrame(results)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out_path = os.path.join(config.REPORT_DIR, 'rolling_bilstm_results.csv')
    all_results.to_csv(out_path, index=False)
    
    print("\n--- Aggregate Results ---")
    agg = all_results.groupby('Market')[['MAE', 'RMSE']].mean()
    print(agg)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
