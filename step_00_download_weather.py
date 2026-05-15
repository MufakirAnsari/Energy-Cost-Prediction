"""
step_00_download_weather.py
===========================
Downloads historical hourly weather data for major PJM and ERCOT load centers.
Uses the Open-Meteo Historical API (free, no API key required).

Features:
  - Temperature (2m)
  - Relative Humidity (2m)
  - Wind Speed (10m)

Run:
    python step_00_download_weather.py
"""

import os
import time
import requests
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config

# Lat/Lon coordinates for major load centers
CITIES = {
    "PJM": {
        "Philadelphia": (39.9526, -75.1652),
        "Chicago":      (41.8781, -87.6298),
        "Pittsburgh":   (40.4406, -79.9959),
        "Detroit":      (42.3314, -83.0458),
        "Columbus":     (39.9612, -82.9988),
    },
    "ERCOT": {
        "Houston":      (29.7604, -95.3698),
        "Dallas":       (32.7767, -96.7970),
        "Austin":       (30.2672, -97.7431),
        "San_Antonio":  (29.4241, -98.4936),
        "Amarillo":     (35.2220, -101.8313),
    }
}

def fetch_city_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "UTC"
    }
    
    retries = 0
    while retries < 3:
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(5)
                retries += 1
                continue
            resp.raise_for_status()
            data = resp.json()
            
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.rename(columns={
                "time": "datetime_utc",
                "temperature_2m": "temp_c",
                "relative_humidity_2m": "rh_pct",
                "wind_speed_10m": "wind_kmh"
            })
            return df.set_index("datetime_utc")
        except Exception as e:
            print(f"      Error: {e}")
            retries += 1
            time.sleep(2)
            
    return pd.DataFrame()


def download_market_weather(market: str, out_path: str):
    print(f"\n[Weather] Fetching data for {market} Region...")
    cities = CITIES[market]
    
    # Open-Meteo requires YYYY-MM-DD
    start_str = pd.Timestamp(config.DATA_START).strftime("%Y-%m-%d")
    end_str   = pd.Timestamp(config.DATA_END).strftime("%Y-%m-%d")
    
    all_city_dfs = []
    
    for city, (lat, lon) in cities.items():
        print(f"  → {city} ({lat}, {lon})")
        df = fetch_city_weather(lat, lon, start_str, end_str)
        if not df.empty:
            # Prefix columns with city name
            df.columns = [f"{city.lower()}_{col}" for col in df.columns]
            all_city_dfs.append(df)
        time.sleep(1) # Be polite to the free API
        
    if not all_city_dfs:
        print(f"  ❌ Failed to download {market} weather.")
        return
        
    # Merge all cities on time index
    final_df = pd.concat(all_city_dfs, axis=1)
    
    # Compute regional averages
    temp_cols = [c for c in final_df.columns if "temp_c" in c]
    rh_cols   = [c for c in final_df.columns if "rh_pct" in c]
    wind_cols = [c for c in final_df.columns if "wind_kmh" in c]
    
    final_df["regional_temp_c"] = final_df[temp_cols].mean(axis=1)
    final_df["regional_rh_pct"] = final_df[rh_cols].mean(axis=1)
    final_df["regional_wind_kmh"] = final_df[wind_cols].mean(axis=1)
    
    final_df.to_parquet(out_path)
    print(f"  ✅ Saved {market} weather: {final_df.shape} to {out_path}")


def main():
    print("=" * 65)
    print("  Weather Downloader (Open-Meteo Archive)")
    print(f"  Period: {config.DATA_START} → {config.DATA_END}")
    print("=" * 65)
    os.makedirs(config.RAW_DIR, exist_ok=True)
    
    download_market_weather("PJM", config.WEATHER_PJM_PATH)
    download_market_weather("ERCOT", config.WEATHER_ERCOT_PATH)

if __name__ == "__main__":
    main()
