# 07_final_report_generator.py

# --- BOILERPLATE TO FIX MODULE PATHS ---
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib

# --- Local Imports ---
import config

# =========================================================================
# === MAIN SCRIPT EXECUTION ===============================================
# =========================================================================

def generate_all_final_plots():
    """
    Main function to load pre-computed results and generate the complete suite
    of 11 publication-quality visualizations for the thesis.
    """
    print("="*80)
    print(" " * 18 + "GENERATING ALL FINAL THESIS VISUALIZATIONS")
    print("="*80)

    # --- Load Pre-computed Results ---
    print("\n--- [ 1. LOADING COMPUTED RESULTS ] ---")
    try:
        results_df = pd.read_csv(os.path.join(config.REPORT_DIR, 'results_df.csv'), index_col=0, parse_dates=True)
        pnl_df = pd.read_csv(os.path.join(config.REPORT_DIR, 'pnl_df.csv'), index_col=0, parse_dates=True)
        point_metrics_df = pd.read_csv(os.path.join(config.REPORT_DIR, 'table_01_point_metrics.csv')).set_index('Model')
        ensemble_model = joblib.load(os.path.join(config.MODEL_DIR, 'enhanced_ensemble_model.joblib'))
        print("Result files loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find result file: {e}")
        print("Please run '05_evaluate_and_report.py' first to generate the result CSVs.")
        return
        
    # Define model groups for plotting
    point_models_all = ['Constrained Transformer', 'Autoformer', 'Bayesian Bi-LSTM', 'LightGBM', 'Ensemble', 'SARIMA']
    point_models_main = ['Autoformer', 'Bayesian Bi-LSTM', 'LightGBM', 'Ensemble']
    
    # --- [ PLOT 1: Time Series Overlay ] ---
    print("Generating Plot 1: Time Series Overlay...")
    volatile_day = results_df['Actual'].rolling('7d').std().idxmax()
    start_date = volatile_day - pd.Timedelta(days=3)
    end_date = volatile_day + pd.Timedelta(days=4)
    volatile_df = results_df.loc[start_date:end_date]
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(volatile_df.index, volatile_df['Actual'], color=config.PUBLICATION_PALETTE['Actual'],
            label='Actual Price', linewidth=3, marker='o', markersize=5, zorder=10)
    for name in point_models_main:
        ax.plot(volatile_df.index, volatile_df[name], color=config.PUBLICATION_PALETTE.get(name),
                linestyle=config.PUBLICATION_LINESTYLES.get(name, 'dashed'), label=name, linewidth=2.5)
    ax.set_title(f'Model Forecasts vs. Actual Price During High Volatility ({start_date.date()} to {end_date.date()})')
    ax.set_ylabel('Price (€/MWh)'); ax.legend(title='Model')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_01_prediction_comparison.png'))
    plt.show()

    # --- [ PLOT 2: Error Distribution ] ---
    print("Generating Plot 2: Error Distribution...")
    errors = pd.DataFrame({name: results_df[name] - results_df['Actual'] for name in point_models_all})
    errors_melted = errors.melt(var_name='Model', value_name='Error (€/MWh)')
    
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.violinplot(data=errors_melted, x='Model', y='Error (€/MWh)', palette=config.PUBLICATION_PALETTE,
                   inner='quartile', cut=0, ax=ax)
    ax.axhline(0, color='black', linestyle='-.', linewidth=2)
    ax.set_title('Distribution of Prediction Errors (Residuals) by Model')
    ax.set_xlabel(None)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_02_error_distribution.png'))
    plt.show()
    
    # --- [ PLOT 3: Uncertainty Bands ] ---
    print("Generating Plot 3: Uncertainty Bands...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
    fig.suptitle('Probabilistic Forecast Comparison on Volatile Week', y=0.97, fontsize=24)
    ax1.plot(volatile_df.index, volatile_df['Actual'], 'o-', color=config.PUBLICATION_PALETTE['Actual'], label='Actual', markersize=4)
    ax1.plot(volatile_df.index, volatile_df['Bayesian Bi-LSTM'], '-', color=config.PUBLICATION_PALETTE['Bayesian Bi-LSTM'], label='Predicted Mean')
    ax1.fill_between(volatile_df.index, volatile_df['Bayesian Lower'], volatile_df['Bayesian Upper'], color=config.PUBLICATION_PALETTE['Bayesian Bi-LSTM'], alpha=0.25, label='90% Interval')
    ax1.set_title('Bayesian Bi-LSTM Probabilistic Forecast'); ax1.legend(loc='upper left'); ax1.set_ylabel('Price (€/MWh)')
    ax2.plot(volatile_df.index, volatile_df['Actual'], 'o-', color=config.PUBLICATION_PALETTE['Actual'], label='Actual', markersize=4)
    ax2.plot(volatile_df.index, volatile_df['LightGBM'], '-', color=config.PUBLICATION_PALETTE['LightGBM'], label='Predicted Median')
    ax2.fill_between(volatile_df.index, volatile_df['LGBM Lower'], volatile_df['LGBM Upper'], color=config.PUBLICATION_PALETTE['LightGBM'], alpha=0.25, label='p10-p90 Interval')
    ax2.set_title('LightGBM Quantile Regression Forecast'); ax2.legend(loc='upper left'); ax2.set_ylabel('Price (€/MWh)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_03_probabilistic_comparison.png'))
    plt.show()

    # --- [ PLOT 4: Reliability Diagram ] ---
    print("Generating Plot 4: Reliability Diagram...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    obs_freq_b = np.mean((results_df['Actual'] >= results_df['Bayesian Lower']) & (results_df['Actual'] <= results_df['Bayesian Upper']))
    ax.plot([0.9], [obs_freq_b], 'o', color=config.PUBLICATION_PALETTE['Bayesian Bi-LSTM'], label=f'Bayesian Bi-LSTM (PICP: {obs_freq_b*100:.1f}%)', markersize=15)
    obs_freq_l = np.mean((results_df['Actual'] >= results_df['LGBM Lower']) & (results_df['Actual'] <= results_df['LGBM Upper']))
    ax.plot([0.8], [obs_freq_l], 's', color=config.PUBLICATION_PALETTE['LightGBM'], label=f'LGBM Quantile (PICP: {obs_freq_l*100:.1f}%)', markersize=15)
    ax.set_xlabel('Forecasted Probability (Confidence/Quantile Range)'); ax.set_ylabel('Observed Frequency (Actual Coverage)')
    ax.set_title('Calibration via Reliability Diagram'); ax.legend(); ax.grid(True)
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_04_reliability_diagram.png'))
    plt.show()

    # --- [ PLOT 5: Equity Curve ] ---
    print("Generating Plot 5: Equity Curve...")
    fig, ax = plt.subplots(figsize=(16, 9))
    cumulative_pnl = pnl_df.cumsum()
    for col in cumulative_pnl.columns:
        ax.plot(cumulative_pnl.index, cumulative_pnl[col], label=col, color=config.PUBLICATION_PALETTE.get(col), linewidth=3)
    ax.set_title('Cumulative Profit & Loss (Equity Curve) of Trading Strategies')
    ax.set_ylabel('Cumulative Profit (€)'); ax.set_xlabel('Date')
    ax.legend(title='Strategy'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    for col in cumulative_pnl.columns:
        final_profit = cumulative_pnl[col].iloc[-1]
        ax.text(cumulative_pnl.index[-1], final_profit, f' €{final_profit:,.0f}', va='center', color=config.PUBLICATION_PALETTE.get(col), fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_05_equity_curve.png'))
    plt.show()

    # --- [ PLOT 6: P&L Distribution ] ---
    print("Generating Plot 6: P&L Distribution...")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(data=pnl_df, palette=config.PUBLICATION_PALETTE, inner='quartile', cut=0, ax=ax)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('Distribution of Daily Profit & Loss by Strategy')
    ax.set_xlabel('Strategy'); ax.set_ylabel('Daily P&L (€)')
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_06_pnl_distribution.png'))
    plt.show()
    
    # --- [ PLOT 7: Error Correlation Matrix ] ---
    print("Generating Plot 7: Error Correlation Matrix...")
    base_model_errors = pd.DataFrame({
        'Autoformer': results_df['Autoformer'] - results_df['Actual'],
        'Bayesian Bi-LSTM': results_df['Bayesian Bi-LSTM'] - results_df['Actual'],
        'LightGBM': results_df['LightGBM'] - results_df['Actual'],
    })
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(base_model_errors.corr(), annot=True, cmap='vlag', fmt=".3f", linewidths=.5, annot_kws={"size": 16}, ax=ax)
    ax.set_title('Correlation Matrix of Base Model Prediction Errors')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_07_error_correlation.png'))
    plt.show()

    # --- [ PLOT 8: Ensemble Feature Importance ] ---
    print("Generating Plot 8: Ensemble Feature Importance...")
    fig, ax = plt.subplots(figsize=(12, 7))
    lgb.plot_importance(ensemble_model, ax=ax, importance_type='gain', title='Ensemble Meta-Learner Feature Importance (by Gain)', grid=False)
    fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_08_ensemble_importance.png'))
    plt.show()

    # --- [ PLOT 9: MAE vs. Inference Time ] ---
    print("Generating Plot 9: MAE vs. Inference Time...")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=point_metrics_df, x='Inference Time (ms/pred)', y='MAE', hue='Model', s=250, style='Model', palette=config.PUBLICATION_PALETTE, ax=ax)
    ax.set_title('Model Performance vs. Computational Cost'); ax.set_xlabel('Inference Time per Prediction (ms) [Log Scale]'); ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_xscale('log'); ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left'); ax.grid(True, which='both', linestyle='--')
    fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_09_tradeoff_scatter.png'))
    plt.show()
    
    # --- [ PLOT 10: Error by Hour ] ---
    print("Generating Plot 10: Error by Hour...")
    errors_by_hour = pd.DataFrame({name: results_df[name] - results_df['Actual'] for name in point_models_main})
    errors_by_hour['hour'] = results_df.index.hour
    errors_melted_hour = errors_by_hour.melt(id_vars='hour', var_name='Model', value_name='Error')
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(data=errors_melted_hour, x='hour', y='Error', hue='Model', palette=config.PUBLICATION_PALETTE, fliersize=2, ax=ax)
    ax.axhline(0, color='black', linestyle='--'); ax.set_title('Model Error Distribution by Hour of Day')
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Prediction Error (€/MWh)')
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left'); fig.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_10_error_by_hour.png'))
    plt.show()

    # --- [ PLOT 13: Rolling Sharpe Ratio ] ---
    print("Generating Plot 13: Rolling Sharpe Ratio...")
    fig, ax = plt.subplots(figsize=(16, 8))
    for col in pnl_df.columns:
        if col != 'Oracle':
            rolling_sharpe = (pnl_df[col].rolling(window=30).mean() / pnl_df[col].rolling(window=30).std()) * np.sqrt(365)
            ax.plot(rolling_sharpe.index, rolling_sharpe, label=f'{col} (30-Day Rolling)', color=config.PUBLICATION_PALETTE.get(col), linewidth=2.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5); ax.set_title('Consistency of Risk-Adjusted Returns (Rolling 30-Day Sharpe Ratio)')
    ax.set_ylabel('Annualized Sharpe Ratio'); ax.set_xlabel('Date')
    ax.legend(fontsize=12); ax.grid(True, which='both', linestyle='--')
    plt.savefig(os.path.join(config.REPORT_DIR, 'pub_plot_13_rolling_sharpe.png'))
    plt.show()

    print("\n" + "="*80)
    print(" " * 24 + "FINAL PLOT GENERATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    generate_all_final_plots()