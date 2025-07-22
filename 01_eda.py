# 01_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import config # Import the configuration file
import os

def run_eda():
    """Performs a comprehensive Exploratory Data Analysis (EDA) on the datasets."""
    print("--- Running Exploratory Data Analysis ---")
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # --- 1. Data Loading ---
    print("Step 1: Loading datasets...")
    try:
        energy_df = pd.read_csv(config.RAW_ENERGY_DATA_PATH)
        weather_df = pd.read_csv(config.RAW_WEATHER_DATA_PATH)
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the 'data' directory.")
        return

    # --- 2. Time Series Analysis of Target Variable ---
    print("\nStep 2: Analyzing the target variable 'price actual'...")
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True)
    energy_df.set_index('time', inplace=True)

    # Set plot style for a professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(config.PLOT_PALETTE)

    # Plot the full time series
    fig, ax = plt.subplots(figsize=(18, 7))
    energy_df['price actual'].plot(ax=ax, title='Actual Electricity Price Over Time',
                                   xlabel='Time', ylabel='Price (€/MWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, "eda_01_full_timeseries.png"), dpi=config.PLOT_DPI)
    plt.show()

    # STL Decomposition
    print("\nStep 3: Decomposing the time series...")
    daily_price = energy_df['price actual'].resample('D').mean().dropna()
    decomposition = seasonal_decompose(daily_price, model='additive', period=365)
    
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle('Seasonal-Trend-Residual Decomposition of Daily Average Price', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(config.REPORT_DIR, "eda_02_decomposition.png"), dpi=config.PLOT_DPI)
    plt.show()

    print("\n--- EDA Complete ---")

if __name__ == '__main__':
    run_eda()