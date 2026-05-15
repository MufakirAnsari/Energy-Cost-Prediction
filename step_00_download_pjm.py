"""
step_00_download_pjm.py  (V2 — multi-source)
=============================================
Downloads PJM Day-Ahead Hourly LMP for the Western Hub node
from 2019-01-01 through 2025-12-31.

Data source priority (auto-selects best available):
  1. gridstatus + PJM_API_KEY env var (fastest, recommended)
  2. EIA Open Data API + EIA_API_KEY env var (free, 2-min registration)
  3. PJM Data Miner 2 direct HTTP (browser-compatible, no key required)

API Key registration (both free):
  PJM: https://apiportal.pjm.com/
  EIA: https://www.eia.gov/opendata/  (instant approval)

Set before running:
  export PJM_API_KEY="your_pjm_key"   # Option 1
  export EIA_API_KEY="your_eia_key"   # Option 2 (fallback)

Run:
  python step_00_download_pjm.py
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.dirname(__file__))
import config


# ─────────────────────────────────────────────────────────────
# SOURCE 1: gridstatus (requires PJM_API_KEY)
# ─────────────────────────────────────────────────────────────

def download_via_gridstatus(api_key: str) -> pd.DataFrame:
    """Use gridstatus library with PJM API key, with strict rate limiting."""
    import gridstatus
    pjm = gridstatus.PJM(api_key=api_key)
    all_chunks = []

    start   = pd.Timestamp(config.DATA_START, tz="UTC")
    end     = pd.Timestamp(config.DATA_END,   tz="UTC")
    current = start

    while current < end:
        chunk_end = min(
            current + relativedelta(months=1) - pd.Timedelta(hours=1), end
        )
        print(f"  [{current.strftime('%Y-%m')}] Fetching ...", end=" ", flush=True)
        t0 = time.time()
        
        # We will manually retry if gridstatus fails
        success = False
        retries = 0
        while retries < 5:
            try:
                df = pjm.get_lmp(
                    start=current.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    market="DAY_AHEAD_HOURLY",
                    location_type="HUB",
                )
                # Filter Western Hub
                for loc_col in ["Location", "location", "pnode_name"]:
                    if loc_col in df.columns:
                        df = df[df[loc_col].str.upper().str.contains("WESTERN", na=False)]
                        break
                        
                all_chunks.append(df)
                print(f"✓ {len(df):,} rows ({time.time()-t0:.1f}s)")
                success = True
                break
            except Exception as e:
                retries += 1
                sleep_time = 5 * retries
                print(f"\n    [Rate Limit/Error] Retrying in {sleep_time}s... ({e})", end=" ")
                time.sleep(sleep_time)
                
        if not success:
            print(f"✗ Failed to download {current.strftime('%Y-%m')}")
            
        current += relativedelta(months=1)
        
        # PJM allows 6 requests per minute. Gridstatus makes ~2 requests per get_lmp.
        # Sleeping for 10 seconds guarantees we stay under the limit.
        time.sleep(10.0)

    if not all_chunks:
        return pd.DataFrame()
        
    return pd.concat(all_chunks, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# SOURCE 2: EIA API v2 — hourly day-ahead price for PJM
# (Free key: https://www.eia.gov/opendata/ — takes 2 min)
# ─────────────────────────────────────────────────────────────

def download_via_eia(api_key: str) -> pd.DataFrame:
    """
    EIA API v2 — Day-Ahead hourly price for PJM region.
    Endpoint: /v2/electricity/rto/region-data/data/
    type=DF = Day-Ahead Forecast LMP ($/MWh)

    Note: EIA provides regional aggregate DA price, not Western Hub nodal.
    This is slightly different from hub-level LMP but acceptable for research.
    """
    BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_chunks = []

    start   = pd.Timestamp(config.DATA_START)
    end     = pd.Timestamp(config.DATA_END)
    current = start

    print("  Using EIA API for PJM Day-Ahead prices...")
    print("  (Free key at eia.gov/opendata — regional aggregate, not Western Hub nodal)")

    while current < end:
        chunk_end = min(current + relativedelta(months=3), end)

        params = {
            "api_key":              api_key,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "PJM",
            "facets[type][]":       "DF",   # Day-Ahead Forecast
            "start":   current.strftime("%Y-%m-%dT%H"),
            "end":     chunk_end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]":    "period",
            "sort[0][direction]": "asc",
            "length":             5000,
            "offset":             0,
        }

        chunk_rows = []
        while True:
            try:
                resp = requests.get(BASE, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("response", {})
                records = data.get("data", [])
                if not records:
                    break
                chunk_rows.extend(records)
                total = data.get("total", 0)
                params["offset"] += len(records)
                if params["offset"] >= total:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"\n    EIA fetch error: {e}")
                break

        if chunk_rows:
            df_chunk = pd.DataFrame(chunk_rows)
            all_chunks.append(df_chunk)
            print(f"  [{current.strftime('%Y-%m')}→{chunk_end.strftime('%Y-%m')}] "
                  f"✓ {len(chunk_rows):,} rows")

        current = chunk_end
        time.sleep(0.5)

    if not all_chunks:
        raise RuntimeError("EIA returned no data. Check API key and parameters.")

    df = pd.concat(all_chunks, ignore_index=True)
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df = df.rename(columns={"period": "datetime_utc", "value": "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.set_index("datetime_utc")[["price"]].sort_index()
    df = df.resample("h").mean()
    return df


# ─────────────────────────────────────────────────────────────
# SOURCE 3: PJM Data Miner 2 direct HTTP (no key, browser-style)
# Uses the legacy public feed endpoint with session cookies
# ─────────────────────────────────────────────────────────────

def download_via_dataminer(start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetches DA LMP from PJM Data Miner 2 using the public HTTP feed.
    PJM WESTERN HUB pnode_id = 51217.

    This uses the same endpoint as the Data Miner 2 web browser interface —
    no API key required, just a browser-compatible request.
    """
    WESTERN_HUB_PNODE = 51217

    BASE_URL = (
        "https://dataminer2.pjm.com/feed/da_hrl_lmps/data"
        "?startRow=1&numRows=100000"
        f"&pnodeId={WESTERN_HUB_PNODE}"
        "&fields=datetime_beginning_utc,total_lmp_da"
        f"&startDate={start_str}&endDate={end_str}&download=true"
    )

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://dataminer2.pjm.com/",
    }

    session = requests.Session()
    # First hit the main page to get session cookies
    session.get("https://dataminer2.pjm.com/", headers=HEADERS, timeout=15)

    resp = session.get(BASE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    return df


def download_via_dataminer_chunked() -> pd.DataFrame:
    """
    Download from PJM Data Miner 2 in monthly chunks (no API key).
    Formats date as MM/DD/YYYY as required by Data Miner 2.
    """
    all_chunks = []
    start   = pd.Timestamp(config.DATA_START)
    end     = pd.Timestamp(config.DATA_END)
    current = start

    print("  Using PJM Data Miner 2 direct HTTP (no API key required)...")

    while current < end:
        chunk_end = min(current + relativedelta(months=1), end)

        start_str = current.strftime("%m/%d/%Y")
        end_str   = chunk_end.strftime("%m/%d/%Y")

        print(f"  [{current.strftime('%Y-%m')}] Fetching ...", end=" ", flush=True)
        t0 = time.time()
        try:
            df = download_via_dataminer(start_str, end_str)
            all_chunks.append(df)
            print(f"✓ {len(df):,} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"✗ {e}")

        current += relativedelta(months=1)
        time.sleep(2.0)  # Be polite — Data Miner 2 is a free public service

    if not all_chunks:
        raise RuntimeError("Data Miner 2 returned no data.")

    df = pd.concat(all_chunks, ignore_index=True)

    # Standardize
    df.columns = df.columns.str.strip().str.lower()
    ts_col = next((c for c in df.columns if "utc" in c or "datetime" in c), None)
    price_col = next((c for c in df.columns if "lmp" in c or "price" in c), None)

    if not ts_col or not price_col:
        raise ValueError(f"Unexpected columns from Data Miner 2: {list(df.columns)}")

    df = df.rename(columns={ts_col: "datetime_utc", price_col: "price"})
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.set_index("datetime_utc")[["price"]].sort_index()
    df = df.resample("h").mean()
    return df


# ─────────────────────────────────────────────────────────────
# STANDARDIZE OUTPUT (same schema regardless of source)
# ─────────────────────────────────────────────────────────────

def finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Ensure output has a clean UTC DatetimeIndex and a single 'price' column.
    Clip PJM DA prices above $3,000/MWh (PJM real-world cap is ~$2,000/MWh).
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex after cleaning")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df[["price"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.resample("h").mean()
    df["price"] = df["price"].clip(upper=3_000.0)

    # Fill short gaps (≤ 3h) with linear interpolation
    df["price"] = df["price"].interpolate(method="linear", limit=3)

    missing_pct = df["price"].isna().mean() * 100
    print(f"\n  Source: {source}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Range:   {df.index.min()} → {df.index.max()}")
    print(f"  Missing: {missing_pct:.2f}%")
    print(f"  Price:   min={df['price'].min():.2f}  "
          f"mean={df['price'].mean():.2f}  "
          f"max={df['price'].max():.2f}  $/MWh")
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PJM Day-Ahead LMP Downloader (V2 multi-source)")
    print(f"  Period: {config.DATA_START} → {config.DATA_END}")
    print("=" * 65)

    os.makedirs(config.RAW_DIR, exist_ok=True)
    out_path = config.PJM_RAW_PATH

    if os.path.exists(out_path):
        df_existing = pd.read_parquet(out_path)
        print(f"\n  ⚠ File exists: {out_path}")
        print(f"    Shape: {df_existing.shape} | "
              f"Range: {df_existing.index.min().date()} → {df_existing.index.max().date()}")
        ans = input("  Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("  Skipping download.")
            return

    # ── Auto-select source ────────────────────────────────────
    pjm_key = os.environ.get("PJM_API_KEY", getattr(config, "PJM_API_KEY", ""))
    eia_key = os.environ.get("EIA_API_KEY", getattr(config, "EIA_API_KEY", ""))

    print()
    if pjm_key:
        print("  ✅ PJM_API_KEY found → using gridstatus")
        raw = download_via_gridstatus(pjm_key)
        if not raw.empty:
            df = _clean_gridstatus_raw(raw)
            df = finalize(df, "gridstatus/PJM API")
        else:
            print("  ❌ Failed to download from PJM API.")
            return

    elif eia_key:
        print("  ✅ EIA_API_KEY found → using EIA API v2")
        print("  (Regional DA price — slightly different from Western Hub nodal LMP)")
        print("  (Register free in 2 min at: https://www.eia.gov/opendata/)")
        df = download_via_eia(eia_key)
        df = finalize(df, "EIA API v2 (PJM regional DA)")

    else:
        print("  ℹ No API keys found → using PJM Data Miner 2 direct HTTP")
        print("  (This works without any registration)")
        print("  ⚠ Slower (2s/month); Data Miner 2 may block automated access.")
        print("  Tip: Set EIA_API_KEY for faster, more reliable downloads:")
        print("       export EIA_API_KEY='your_key'  # free at eia.gov/opendata")
        print()
        df = download_via_dataminer_chunked()
        df = finalize(df, "PJM Data Miner 2 direct HTTP")

    df.to_parquet(out_path)
    print(f"\n  ✅ Saved: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def _clean_gridstatus_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw gridstatus output to {datetime_utc index, price}."""
    df = raw.copy()
    
    # 1. Find the timestamp column
    ts_col = None
    for cand in ["Interval Start", "interval_start", "Time", "time"]:
        if cand in df.columns:
            ts_col = cand
            break
            
    # 2. Find the price column
    price_col = None
    for cand in ["LMP", "lmp", "Total LMP", "total_lmp", "price", "Price"]:
        if cand in df.columns:
            price_col = cand
            break
            
    if not ts_col or not price_col:
        raise ValueError(f"Could not find required columns in: {list(df.columns)}")
        
    # Isolate just those two columns to prevent duplicate key errors
    df = df[[ts_col, price_col]].copy()
    df.columns = ["datetime_utc", "price"]
    
    # Convert types
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Set index
    df = df.set_index("datetime_utc").sort_index()
    return df

if __name__ == "__main__":
    main()
