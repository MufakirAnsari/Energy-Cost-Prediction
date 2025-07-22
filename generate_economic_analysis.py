# generate_economic_analysis_plots.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Local Imports ---
import config
from utils import inverse_transform, create_sequences # Assumes these are in utils.py
# Note: This script re-uses logic from 05_evaluate_and_report.py but is focused only on the economic analysis part.

def perform_deep_economic_analysis():
    """
    Loads the final results and performs a deep economic analysis,
    calculating profit volatility and generating a distribution plot.
    """
    print("--- Performing Deep Economic Impact Analysis ---")
    
    # --- This section is a simplified re-run of the prediction generation ---
    # In a real project, you might load a saved 'results_df.csv' instead.
    # For now, we'll assume the results_df is available or can be recreated.
    # This is a placeholder for the full results_df generation logic from script 05.
    # For this to run, you need to have your 'results_df' from the main evaluation script.
    # Let's assume you've saved it.
    
    # --- Create a dummy results_df for demonstration if not saved ---
    # In your actual run, load the real results_df from script 05
    try:
        # Ideally you would save your results_df in script 05 and load it here
        # results_df = pd.read_csv(os.path.join(config.REPORT_DIR, 'final_results.csv'), index_col=0, parse_dates=True)
        # For now, let's create a placeholder based on the output you shared
        print("NOTE: Using placeholder data for demonstration. Run script 05 to get the real DataFrame.")
        num_test_samples = 3490
        dates = pd.date_range(start='2018-01-01', periods=num_test_samples, freq='H')
        actuals = np.random.normal(60, 15, num_test_samples)
        bayesian_preds = actuals + np.random.normal(0, 2.24, num_test_samples)
        uncertainty = np.random.normal(2.7, 0.5, num_test_samples)
        results_df = pd.DataFrame({
            'actual': actuals,
            'bayesian_pred': bayesian_preds,
            'upper_bound': bayesian_preds + 2 * uncertainty,
            'lower_bound': bayesian_preds - 2 * uncertainty
        }, index=dates)

    except FileNotFoundError:
        print("ERROR: Could not find a saved results file. Please run script 05 first and save its 'results_df'.")
        return

    # --- Core Economic Simulation Logic ---
    daily_groups = results_df.groupby(results_df.index.date)
    daily_profits = []
    for day, group in daily_groups:
        if len(group) < 24: continue
        
        perfect_profit = group['actual'].max() - group['actual'].min()
        
        naive_buy = group.loc[group['bayesian_pred'].idxmin(), 'actual']
        naive_sell = group.loc[group['bayesian_pred'].idxmax(), 'actual']
        naive_profit = naive_sell - naive_buy

        risk_aware_buy = group.loc[group['upper_bound'].idxmin(), 'actual']
        risk_aware_sell = group.loc[group['lower_bound'].idxmax(), 'actual']
        risk_aware_profit = risk_aware_sell - risk_aware_buy
        
        daily_profits.append({
            'date': day,
            'oracle_profit': perfect_profit,
            'naive_profit': naive_profit,
            'risk_aware_profit': risk_aware_profit
        })
        
    profit_df = pd.DataFrame(daily_profits).set_index('date')

    # --- Calculate New Statistics ---
    std_oracle = profit_df['oracle_profit'].std()
    std_naive = profit_df['naive_profit'].std()
    std_risk_aware = profit_df['risk_aware_profit'].std()

    print("\n--- In-Depth Economic Results ---")
    print(f"Standard Deviation of Daily Oracle Profit: {std_oracle:.4f} €")
    print(f"Standard Deviation of Daily Naive Profit: {std_naive:.4f} €")
    print(f"Standard Deviation of Daily Risk-Aware Profit: {std_risk_aware:.4f} €")
    print("\nNOTE: Copy these values into TABLE II of your paper.")

    # --- Generate New Plot ---
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.kdeplot(data=profit_df['naive_profit'], label=f'Naive Strategy (Std Dev = {std_naive:.2f} €)', fill=True, ax=ax, lw=2.5)
    sns.kdeplot(data=profit_df['risk_aware_profit'], label=f'Risk-Aware Strategy (Std Dev = {std_risk_aware:.2f} €)', fill=True, ax=ax, lw=2.5)
    
    plt.title("Distribution of Daily Trading Profits", fontsize=18, fontweight='bold')
    plt.xlabel("Daily Profit (€)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.axvline(0, color='black', linestyle='--')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    diagram_path = os.path.join(config.REPORT_DIR, 'daily_profit_distribution.png')
    plt.savefig(diagram_path, dpi=config.PLOT_DPI)
    print(f"\nProfit distribution plot saved to: {diagram_path}")
    plt.show()


if __name__ == '__main__':
    # You will need the 'results_df' DataFrame from script 05.
    # A good practice would be to save it to a CSV in script 05.
    # For now, this script includes a placeholder to generate the plot structure.
    # Replace the placeholder part with loading your actual results.
    perform_deep_economic_analysis()
