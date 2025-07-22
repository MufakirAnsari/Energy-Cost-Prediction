# 05_evaluate_and_report.py

# --- BOILERPLATE TO FIX MODULE PATHS ---
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# -----------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import lightgbm as lgb
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# --- Local Imports ---
import config
from utils import (
    create_sequences, inverse_transform,
    ProbabilisticHead, nll,
    AutoCorrelation, AutoCorrelationLayer, SeasonalTrendDecompositionBlock
)
import properscoring as ps

# =========================================================================
# === SETUP & HELPER FUNCTIONS ============================================
# =========================================================================

def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- GPU DETECTED, setting memory growth ---")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    sns.set_theme(style=config.PLOT_STYLE, palette=config.PLOT_PALETTE)
    plt.rcParams['figure.dpi'] = config.PLOT_DPI
    print("Environment setup complete.")

def calculate_point_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'sMAPE (%)': smape}

def calculate_probabilistic_metrics(y_true, lower_bound, upper_bound, y_dist_samples=None):
    picp = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
    mpiw = np.mean(upper_bound - lower_bound)
    crps = np.nan
    if y_dist_samples is not None:
        crps = ps.crps_ensemble(y_true, y_dist_samples).mean()
    return {'PICP (%)': picp, 'MPIW': mpiw, 'CRPS': crps}

def calculate_economic_metrics(daily_pnl):
    if daily_pnl.std() == 0: return (0, 0, 0)
    sharpe_ratio = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
    downside_std = daily_pnl[daily_pnl < 0].std()
    sortino_ratio = daily_pnl.mean() / downside_std * np.sqrt(365) if downside_std > 0 else np.inf
    cumulative_pnl = daily_pnl.cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - running_max) / running_max
    max_drawdown = drawdown.min() * 100 if not running_max.eq(0).all() else 0
    return {'Sharpe Ratio': sharpe_ratio, 'Sortino Ratio': sortino_ratio, 'Max Drawdown (%)': max_drawdown}

# =========================================================================
# === MAIN EVALUATION FUNCTION ============================================
# =========================================================================

def run_evaluation_and_reporting():
    setup_environment()
    
    print("\n" + "="*80)
    print(" " * 15 + "PHASE 2 & 3: RIGOROUS EVALUATION & REPORTING")
    print("="*80)

    print("\n--- [ 1. LOADING DATA & PREPARING TEST SET ] ---")
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    n = len(df)
    train_df = df[0:int(n*0.7)]
    test_df = df[int(n*0.9):]
    
    scaler = MinMaxScaler().fit(train_df)
    test_scaled = scaler.transform(test_df)
    
    target_idx = list(df.columns).index(config.TARGET_FEATURE)
    n_features = len(df.columns)
    
    X_test_constrained, y_test_scaled = create_sequences(test_scaled, config.SEQ_LENGTH_CONSTRAINED, target_idx)
    X_test_long, _ = create_sequences(test_scaled, config.SEQ_LENGTH_LONG, target_idx)
    
    max_seq_len = config.SEQ_LENGTH_LONG
    y_true = inverse_transform(test_scaled[max_seq_len:, target_idx], scaler, target_idx, n_features)
    results_index = test_df.index[max_seq_len:]
    
    print(f"Test set prepared. Number of evaluation points: {len(y_true):,}")

    print("\n--- [ 2. LOADING ALL TRAINED MODELS ] ---")
    custom_objects = {
        'AutoCorrelation': AutoCorrelation, 'AutoCorrelationLayer': AutoCorrelationLayer,
        'SeasonalTrendDecompositionBlock': SeasonalTrendDecompositionBlock,
        'ProbabilisticHead': ProbabilisticHead, 'nll': nll
    }
    
    models = {
        'Constrained Transformer': keras.models.load_model(os.path.join(config.MODEL_DIR, 'constrained_transformer_model.keras')),
        'Autoformer': keras.models.load_model(os.path.join(config.MODEL_DIR, 'autoformer_model.keras'), custom_objects=custom_objects, safe_mode=False),
        'Bayesian Bi-LSTM': keras.models.load_model(os.path.join(config.MODEL_DIR, 'bayesian_model.keras'), custom_objects=custom_objects, safe_mode=False),
        'LightGBM': joblib.load(os.path.join(config.MODEL_DIR, 'lightgbm_point_model.joblib')),
        'LGBM Quantile p10': joblib.load(os.path.join(config.MODEL_DIR, 'lightgbm_quantile_p10_model.joblib')),
        'LGBM Quantile p90': joblib.load(os.path.join(config.MODEL_DIR, 'lightgbm_quantile_p90_model.joblib')),
        'Ensemble': joblib.load(os.path.join(config.MODEL_DIR, 'enhanced_ensemble_model.joblib')),
        'SARIMA': joblib.load(os.path.join(config.MODEL_DIR, 'sarima_model.joblib'))
    }
    print("All models loaded successfully.")

    print("\n--- [ 3. GENERATING PREDICTIONS ON TEST SET ] ---")
    predictions = {'Actual': y_true}
    inference_times = {}

    offset_constrained = max_seq_len - config.SEQ_LENGTH_CONSTRAINED
    
    for name, model in models.items():
        if isinstance(model, keras.Model):
            print(f"Predicting with {name}...")
            start_time = time.time()
            if 'Autoformer' in name:
                decoder_known = X_test_long[:, -config.DECODER_SEQ_LEN:, :]
                decoder_padding = np.zeros((len(X_test_long), config.PRED_LENGTH, n_features))
                X_test_decoder = np.concatenate([decoder_known, decoder_padding], axis=1)
                preds_scaled = model.predict([X_test_long, X_test_decoder], batch_size=config.BATCH_SIZE)
                preds_scaled = preds_scaled[:, 0, 0]
            elif 'Bayesian' in name:
                preds_dist = model(X_test_constrained[offset_constrained:])
                preds_scaled = preds_dist.mean().numpy().flatten()
            else:
                preds_scaled = model.predict(X_test_constrained[offset_constrained:], batch_size=config.BATCH_SIZE).flatten()

            inference_times[name] = (time.time() - start_time) * 1000 / len(y_true)
            predictions[name] = inverse_transform(preds_scaled, scaler, target_idx, n_features)

    X_test_tabular = test_df.drop(columns=[config.TARGET_FEATURE]).iloc[max_seq_len:]
    for name in ['LightGBM', 'LGBM Quantile p10', 'LGBM Quantile p90']:
        if name in models:
            print(f"Predicting with {name}...")
            start_time = time.time()
            predictions[name] = models[name].predict(X_test_tabular)
            inference_times[name] = (time.time() - start_time) * 1000 / len(y_true)

    print("Predicting with SARIMA...")
    start_time = time.time()
    price_series_resampled = df[config.TARGET_FEATURE].resample('3h').mean().dropna()
    train_end_index = int(len(price_series_resampled) * 0.7)
    n_periods = len(price_series_resampled) - train_end_index
    sarima_preds_resampled = models['SARIMA'].predict(n_periods=n_periods)
    sarima_preds_hourly = sarima_preds_resampled.reindex(results_index, method='ffill')
    sarima_preds_hourly = sarima_preds_hourly.fillna(method='bfill')
    predictions['SARIMA'] = sarima_preds_hourly.values
    inference_times['SARIMA'] = (time.time() - start_time) * 1000 / len(y_true)
    
    print("Predicting with Ensemble...")
    start_time = time.time()
    meta_features = np.column_stack([
        predictions['Autoformer'],
        predictions['Bayesian Bi-LSTM'],
        predictions['LightGBM']
    ])
    ensemble_preds = models['Ensemble'].predict(meta_features)
    predictions['Ensemble'] = ensemble_preds
    inference_times['Ensemble'] = (time.time() - start_time) * 1000 / len(y_true)

    results_df = pd.DataFrame(predictions, index=results_index)
    
    print("\n--- [ 4. POINT FORECAST PERFORMANCE (RQ1) ] ---")
    point_metrics = []
    point_models = ['Constrained Transformer', 'Autoformer', 'Bayesian Bi-LSTM', 'LightGBM', 'Ensemble', 'SARIMA']
    for name in point_models:
        metrics = calculate_point_metrics(results_df['Actual'], results_df[name])
        metrics['Model'] = name
        metrics['Inference Time (ms/pred)'] = inference_times.get(name, 0)
        point_metrics.append(metrics)
    
    point_results_df = pd.DataFrame(point_metrics).set_index('Model')
    print("\nPoint Forecast Metrics on Test Set:")
    print(point_results_df.round(4))
    point_results_df.to_csv(os.path.join(config.REPORT_DIR, 'table_01_point_metrics.csv'))
    
    print("\n--- [ 5. PROBABILISTIC FORECAST QUALITY (RQ2) ] ---")
    prob_metrics = []

    print("Generating uncertainty for Bayesian model...")
    mc_samples = []
    for _ in tqdm(range(50), desc="Bayesian MC Sampling"):
        mc_samples.append(models['Bayesian Bi-LSTM'](X_test_constrained[offset_constrained:]).sample().numpy())
    mc_samples = np.squeeze(np.array(mc_samples))
    mc_samples_actual = inverse_transform(mc_samples.T.flatten(), scaler, target_idx, n_features).reshape(len(y_true), -1)
    
    bayesian_lower = np.percentile(mc_samples_actual, 5, axis=1)
    bayesian_upper = np.percentile(mc_samples_actual, 95, axis=1)
    
    bayesian_metrics = calculate_probabilistic_metrics(y_true, bayesian_lower, bayesian_upper, mc_samples_actual)
    bayesian_metrics['Model'] = 'Bayesian Bi-LSTM (90% CI)'
    prob_metrics.append(bayesian_metrics)

    lgbm_lower = results_df['LGBM Quantile p10']
    lgbm_upper = results_df['LGBM Quantile p90']
    lgbm_metrics = calculate_probabilistic_metrics(y_true, lgbm_lower, lgbm_upper)
    lgbm_metrics['Model'] = 'LGBM Quantile (p10-p90)'
    prob_metrics.append(lgbm_metrics)

    prob_results_df = pd.DataFrame(prob_metrics).set_index('Model')
    print("\nProbabilistic Forecast Metrics on Test Set:")
    print(prob_results_df.round(4))
    prob_results_df.to_csv(os.path.join(config.REPORT_DIR, 'table_02_probabilistic_metrics.csv'))

    results_df['Bayesian Lower'] = bayesian_lower
    results_df['Bayesian Upper'] = bayesian_upper
    results_df['LGBM Lower'] = lgbm_lower
    results_df['LGBM Upper'] = lgbm_upper

    print("\n--- [ 6. ECONOMIC UTILITY & RISK ANALYSIS (RQ4) ] ---")
    daily_groups = results_df.groupby(results_df.index.date)
    
    strategies = {
        'Oracle': [], 'Naive (Autoformer)': [], 'Risk-Aware (Bayesian)': []
    }

    for day, group in daily_groups:
        if len(group) < 24: continue
        
        strategies['Oracle'].append(group['Actual'].max() - group['Actual'].min())
        
        buy_price_naive = group.loc[group['Autoformer'].idxmin(), 'Actual']
        sell_price_naive = group.loc[group['Autoformer'].idxmax(), 'Actual']
        strategies['Naive (Autoformer)'].append(sell_price_naive - buy_price_naive)
        
        buy_price_risk = group.loc[group['Bayesian Upper'].idxmin(), 'Actual']
        sell_price_risk = group.loc[group['Bayesian Lower'].idxmax(), 'Actual']
        strategies['Risk-Aware (Bayesian)'].append(sell_price_risk - buy_price_risk)

    pnl_df = pd.DataFrame(strategies)
    
    economic_metrics = []
    for name, daily_pnl in pnl_df.items():
        metrics = {'Strategy': name, 'Total Profit (€)': daily_pnl.sum()}
        metrics.update(calculate_economic_metrics(daily_pnl))
        economic_metrics.append(metrics)
        
    economic_results_df = pd.DataFrame(economic_metrics).set_index('Strategy')
    print("\nEconomic Performance Metrics:")
    print(economic_results_df.round(4))
    economic_results_df.to_csv(os.path.join(config.REPORT_DIR, 'table_03_economic_metrics.csv'))

    # --- 7. Generate All Thesis Plots ---
    print("\n--- [ 7. GENERATING ALL THESIS PLOTS ] ---")
    
    # Find a volatile week for plotting
    results_df['volatility'] = results_df['Actual'].rolling(24*7).std()
    volatile_day = results_df['volatility'].idxmax()
    start_date = volatile_day - pd.Timedelta(days=3)
    end_date = volatile_day + pd.Timedelta(days=4)
    
    # --- FIX: Create a sliced DataFrame first, then plot from it. ---
    volatile_df = results_df.loc[start_date:end_date]

    # Plot 1: Time Series Overlay on Volatile Week
    plt.figure(figsize=(20, 10))
    plot_models = ['Autoformer', 'Bayesian Bi-LSTM', 'LightGBM', 'Ensemble']
    plt.plot(volatile_df.index, volatile_df['Actual'], 'o-', label='Actual Price', color='black', linewidth=2, markersize=4)
    for name in plot_models:
        plt.plot(volatile_df.index, volatile_df[name], '--', label=name)
    plt.title('Model Predictions vs. Actual Price (Highly Volatile Week)', fontsize=20, fontweight='bold')
    plt.ylabel('Price (€/MWh)', fontsize=14); plt.legend(fontsize=12)
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_01_prediction_comparison.png'))
    plt.show()

    # Plot 2: Error Distribution Violin Plot
    errors = pd.DataFrame({name: results_df[name] - results_df['Actual'] for name in point_models})
    plt.figure(figsize=(18, 9))
    sns.violinplot(data=errors, inner='quartile', cut=0)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Distribution of Prediction Errors (Residuals) by Model', fontsize=20, fontweight='bold')
    plt.xlabel('Model', fontsize=14); plt.ylabel('Prediction Error (€/MWh)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_02_error_distribution.png'))
    plt.show()

    # Plot 3: Time Series with Uncertainty Bands
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
    # Bayesian
    ax1.plot(volatile_df.index, volatile_df['Actual'], 'o-', color='black', label='Actual')
    ax1.plot(volatile_df.index, volatile_df['Bayesian Bi-LSTM'], '-', color='red', label='Predicted Mean')
    ax1.fill_between(volatile_df.index, volatile_df['Bayesian Lower'], volatile_df['Bayesian Upper'], color='red', alpha=0.2, label='90% Interval')
    ax1.set_title('Bayesian Bi-LSTM Probabilistic Forecast', fontsize=16)
    ax1.legend(); ax1.set_ylabel('Price (€/MWh)')
    # LGBM
    ax2.plot(volatile_df.index, volatile_df['Actual'], 'o-', color='black', label='Actual')
    ax2.plot(volatile_df.index, volatile_df['LightGBM'], '-', color='blue', label='Predicted Mean')
    ax2.fill_between(volatile_df.index, volatile_df['LGBM Lower'], volatile_df['LGBM Upper'], color='blue', alpha=0.2, label='p10-p90 Interval')
    ax2.set_title('LightGBM Quantile Regression Forecast', fontsize=16)
    ax2.legend(); ax2.set_ylabel('Price (€/MWh)')
    fig.suptitle('Probabilistic Forecast Comparison on Volatile Week', fontsize=20, fontweight='bold', y=0.95)
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_03_probabilistic_comparison.png'))
    plt.show()

    # Plot 5: Cumulative Profit "Equity Curve"
    plt.figure(figsize=(14, 8))
    pnl_df.cumsum().plot(ax=plt.gca(), linewidth=2.5)
    plt.title('Cumulative Profit & Loss (Equity Curve) of Trading Strategies', fontsize=20, fontweight='bold')
    plt.xlabel('Date', fontsize=14); plt.ylabel('Cumulative Profit (€)', fontsize=14)
    plt.legend(title='Strategy', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_05_equity_curve.png'))
    plt.show()

    # Plot 6: Daily P&L Distribution
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=pnl_df, inner='quartile', cut=0)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Distribution of Daily Profit & Loss by Strategy', fontsize=20, fontweight='bold')
    plt.xlabel('Strategy', fontsize=14); plt.ylabel('Daily P&L (€)', fontsize=14)
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_06_pnl_distribution.png'))
    plt.show()

    # Plot 7: Base Model Error Correlation Matrix
    base_model_errors = pd.DataFrame({
        'Autoformer': results_df['Autoformer'] - results_df['Actual'],
        'Bayesian Bi-LSTM': results_df['Bayesian Bi-LSTM'] - results_df['Actual'],
        'LightGBM': results_df['LightGBM'] - results_df['Actual'],
    })
    plt.figure(figsize=(10, 8))
    sns.heatmap(base_model_errors.corr(), annot=True, cmap='coolwarm', fmt=".3f")
    plt.title('Correlation Matrix of Base Model Prediction Errors', fontsize=18, fontweight='bold')
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_07_error_correlation.png'))
    plt.show()

    # Plot 8: New Ensemble Feature Importance
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(models['Ensemble'], importance_type='gain', figsize=(10, 6), title='Ensemble Meta-Learner Feature Importance (by Gain)', grid=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'plot_08_ensemble_importance.png'))
    plt.show()
    
    results_df.to_csv(os.path.join(config.REPORT_DIR, 'results_df.csv'))
    pnl_df.to_csv(os.path.join(config.REPORT_DIR, 'pnl_df.csv'))    
    print("\n\n" + "="*80)
    print(" " * 28 + "REPORT GENERATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    run_evaluation_and_reporting()