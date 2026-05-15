"""
step_00_download_eia.py
=======================
Downloads from EIA Open Data API v2:
  1. Hourly electricity generation by fuel type (solar, wind, gas, nuclear)
     for PJM and ERCOT respondents
  2. Hourly net generation and demand (total load)
  3. Henry Hub natural gas spot price (daily → forward-filled to hourly)

Requires EIA API key set as environment variable:
    export EIA_API_KEY="your_key_here"
    (Register free at: https://www.eia.gov/opendata/)

Run:
    python step_00_download_eia.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config


# ── EIA API Helper ────────────────────────────────────────────────────────────

def eia_fetch(endpoint: str, params: dict, max_records: int = 5000) -> pd.DataFrame:
    """
    Fetch all pages from EIA API v2 endpoint, handling 5000-record pagination.
    """
    if not config.EIA_API_KEY:
        raise ValueError(
            "EIA_API_KEY not set. Export it: export EIA_API_KEY='your_key'"
        )
    base_url = f"{config.EIA_BASE_URL}/{endpoint}"
    params["api_key"] = config.EIA_API_KEY
    params["offset"]  = 0
    params["length"]  = max_records

    all_data = []
    while True:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("response", {}).get("data", [])
        if not records:
            break
        all_data.extend(records)

        total = int(payload.get("response", {}).get("total", 0))
        params["offset"] += len(records)
        if params["offset"] >= total:
            break
        time.sleep(0.5)

    return pd.DataFrame(all_data)


# ── Fuel Type Generation ─────────────────────────────────────────────────────

def download_generation(respondent: str, label: str) -> pd.DataFrame:
    """
    Download hourly generation by fuel type for a given respondent (PJM or ERCO).
    Returns wide-format DataFrame with columns: solar_mw, wind_mw, gas_mw, nuclear_mw
    """
    print(f"  Fetching {label} generation mix...")
    fuel_types = {
        "SUN": "solar_mw",
        "WND": "wind_mw",
        "NG":  "gas_mw",
        "NUC": "nuclear_mw",
    }

    frames = []
    for fuel_code, col_name in fuel_types.items():
        try:
            df = eia_fetch(
                endpoint="electricity/rto/fuel-type-data/data/",
                params={
                    "frequency": "hourly",
                    "data[0]": "value",
                    f"facets[respondent][]": respondent,
                    f"facets[fueltype][]":   fuel_code,
                    "start": config.DATA_START + "T00",
                    "end":   config.DATA_END   + "T23",
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                }
            )
            if df.empty:
                print(f"    {fuel_code}: no data")
                continue
            df["period"] = pd.to_datetime(df["period"], utc=True)
            df = df.set_index("period")[["value"]].rename(columns={"value": col_name})
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            frames.append(df)
            print(f"    {col_name}: {len(df):,} rows")
        except Exception as e:
            print(f"    ERROR fetching {fuel_code}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).resample("h").mean()
    # Compute renewable fraction
    total = result.sum(axis=1).replace(0, float("nan"))
    result["renewable_fraction"] = (
        result.get("solar_mw", 0) + result.get("wind_mw", 0)
    ) / total
    return result


# ── System Load / Demand ─────────────────────────────────────────────────────

def download_demand(respondent: str, label: str) -> pd.DataFrame:
    print(f"  Fetching {label} system demand...")
    try:
        df = eia_fetch(
            endpoint="electricity/rto/region-data/data/",
            params={
                "frequency": "hourly",
                "data[0]": "value",
                f"facets[respondent][]": respondent,
                f"facets[type][]": "D",    # D = demand
                "start": config.DATA_START + "T00",
                "end":   config.DATA_END   + "T23",
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
        )
        df["period"] = pd.to_datetime(df["period"], utc=True)
        df = (df.set_index("period")[["value"]]
              .rename(columns={"value": "demand_mw"}))
        df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
        return df.resample("h").mean()
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


# ── Henry Hub Gas Price ──────────────────────────────────────────────────────

def download_gas_price() -> pd.DataFrame:
    """Daily Henry Hub spot price, forward-filled to hourly."""
    print("  Fetching Henry Hub natural gas price (daily)...")
    try:
        df = eia_fetch(
            endpoint="natural-gas/pri/fut/data/",
            params={
                "frequency":     "daily",
                "data[0]":       "value",
                "facets[series][]": "RNGWHHD",   # Henry Hub daily spot
                "start": config.DATA_START,
                "end":   config.DATA_END,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
        )
        df["period"] = pd.to_datetime(df["period"], utc=True)
        df = (df.set_index("period")[["value"]]
              .rename(columns={"value": "gas_price_mmBtu"}))
        df["gas_price_mmBtu"] = pd.to_numeric(df["gas_price_mmBtu"], errors="coerce")
        # Resample to hourly via forward fill (gas price is constant within a day)
        idx = pd.date_range(
            start=config.DATA_START, end=config.DATA_END, freq="h", tz="UTC"
        )
        df = df.reindex(idx).ffill().bfill()
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  EIA API v2 Data Downloader")
    print(f"  Period: {config.DATA_START} → {config.DATA_END}")
    print("=" * 65)
    os.makedirs(config.RAW_DIR, exist_ok=True)

    # PJM generation + demand
    print("\n[1/4] PJM Generation Mix")
    pjm_gen  = download_generation(config.EIA_PJM_RESPONDENT, "PJM")
    print("[2/4] PJM System Demand")
    pjm_dem  = download_demand(config.EIA_PJM_RESPONDENT, "PJM")

    pjm_eia = pd.concat([pjm_gen, pjm_dem], axis=1)
    pjm_eia.to_parquet(config.EIA_GEN_PATH)
    print(f"  → Saved PJM EIA features: {pjm_eia.shape}")

    # ERCOT generation + demand
    print("\n[3/4] ERCOT Generation Mix")
    erc_gen = download_generation(config.EIA_ERCOT_RESPONDENT, "ERCOT")
    print("[4/4] ERCOT System Demand")
    erc_dem = download_demand(config.EIA_ERCOT_RESPONDENT, "ERCOT")

    erc_eia = pd.concat([erc_gen, erc_dem], axis=1)
    erc_path = config.EIA_GEN_PATH.replace("eia_generation", "eia_generation_ercot")
    erc_eia.to_parquet(erc_path)
    print(f"  → Saved ERCOT EIA features: {erc_eia.shape}")

    # Henry Hub gas price (same for both markets)
    print("\n[5/5] Henry Hub Natural Gas Price")
    gas_df = download_gas_price()
    gas_df.to_parquet(config.EIA_GAS_PATH)
    print(f"  → Saved gas price: {gas_df.shape}")

    print("\n✅ All EIA downloads complete.")


if __name__ == "__main__":
    main()
