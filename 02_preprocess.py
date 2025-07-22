# 02_preprocess.py

import pandas as pd
import numpy as np
import holidays
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import os
import config # Import the configuration file

def run_preprocessing():
    """Main function to run the data processing pipeline."""
    print("\n--- Starting Data Preprocessing and Feature Engineering ---")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print("Step 1: Loading and Initial Preparation...")
    energy_df = pd.read_csv(config.RAW_ENERGY_DATA_PATH)
    weather_df = pd.read_csv(config.RAW_WEATHER_DATA_PATH)

    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True).dt.tz_convert('UTC')
    weather_df['time'] = pd.to_datetime(weather_df['dt_iso'], utc=True).dt.tz_convert('UTC')

    energy_df = energy_df.set_index('time')
    weather_df = weather_df.set_index('time')
    
    print("Step 2: Cleaning Energy Data...")
    cols_to_drop_energy = [
        'generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead',
        'forecast solar day ahead', 'forecast wind onshore day ahead',
        'total load forecast', 'price day ahead',
        'generation fossil coal-derived gas', 'generation fossil oil shale',
        'generation fossil peat', 'generation geothermal', 'generation marine',
        'generation wind offshore'
    ]
    energy_df = energy_df.drop(columns=cols_to_drop_energy)
    # <<< FIX: Changed df to energy_df to use the correct dataframe for the operation.
    energy_df['generation_fossil_total'] = energy_df['generation fossil hard coal'] + energy_df['generation fossil brown coal/lignite']
    energy_df = energy_df.drop(columns=['generation fossil hard coal', 'generation fossil brown coal/lignite'])

    print("Step 3: Cleaning and Pivoting Weather Data...")
    weather_df['city_name'] = weather_df['city_name'].str.strip()
    cols_to_drop_weather = [
        'dt_iso', 'weather_id', 'weather_main', 'weather_description',
        'weather_icon', 'temp_min', 'temp_max'
    ]
    weather_df = weather_df.drop(columns=cols_to_drop_weather)
    weather_df = weather_df.reset_index().drop_duplicates(subset=['time', 'city_name'], keep='first').set_index('time')
    weather_pivoted = weather_df.pivot(columns='city_name')
    weather_pivoted.columns = ['_'.join(col).strip() for col in weather_pivoted.columns.values]
    
    print("Step 4: Merging energy and weather data...")
    # <<< FIX: Changed energy_cleaned to energy_df, which is the correct variable name.
    df_merged = energy_df.join(weather_pivoted, how='outer').asfreq('h')

    print("Step 5: Engineering Time Series Features...")
    df_feat = df_merged.copy()
    holiday_dates = list(holidays.Spain(years=range(df_feat.index.year.min(), df_feat.index.year.max() + 1)).keys())
    df_feat['is_holiday'] = df_feat.index.normalize().isin(holiday_dates).astype(int)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat.index.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat.index.hour / 24)
    df_feat['day_of_week_sin'] = np.sin(2 * np.pi * df_feat.index.dayofweek / 7)
    df_feat['day_of_week_cos'] = np.cos(2 * np.pi * df_feat.index.dayofweek / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat.index.month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat.index.month / 12)

    target_cols = ['price actual', 'total load actual']
    lags = [1, 2, 3, 24, 48, 168]
    windows = [6, 12, 24]
    for col in target_cols:
        for lag in lags:
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
        for window in windows:
            df_feat[f'{col}_roll_mean_{window}'] = df_feat[col].shift(1).rolling(window=window).mean()
            df_feat[f'{col}_roll_std_{window}'] = df_feat[col].shift(1).rolling(window=window).std()

    df_featured = df_feat.dropna(subset=[f'{config.TARGET_FEATURE}_lag_168'])

    print(f"Step 6: Imputing remaining missing values... This may take a moment.")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_featured)
    imputer = KNNImputer(n_neighbors=5)
    imputed_scaled_data = imputer.fit_transform(scaled_data)
    imputed_data = scaler.inverse_transform(imputed_scaled_data)
    df_final = pd.DataFrame(imputed_data, columns=df_featured.columns, index=df_featured.index)

    df_final.to_parquet(config.PROCESSED_DATA_PATH)
    
    print(f"\nPreprocessing complete. Final dataset shape: {df_final.shape}")
    print(f"Processed data saved to {config.PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    run_preprocessing()