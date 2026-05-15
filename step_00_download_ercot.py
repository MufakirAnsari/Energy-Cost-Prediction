"""
step_00_download_ercot.py  (V2 — FINAL)
=========================================
Downloads ERCOT Day-Ahead Market (DAM) Settlement Point Prices
for HB_HOUSTON and HB_WEST, 2018-01-01 → 2025-12-31.

ERCOT DATA ARCHITECTURE:
  We use ERCOT's official "Historical DAM Load Zone and Hub Prices" 
  (ReportTypeID 13060) which provides consolidated annual ZIP files 
  containing all hourly SPPs. This is the fastest, most robust method,
  avoiding rate limits and parsing issues with daily XMLs.

Run: python step_00_download_ercot.py
"""

import os
import sys
import io
import zipfile
import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
})

TARGET_HUBS = {"HB_HOUSTON", "HB_WEST"}


def get_annual_doc_ids() -> dict:
    """
    Fetch the list of annual ZIP files for 'Historical DAM Load Zone and Hub Prices'.
    Returns a dict mapping year (int) to ERCOT DocID (str).
    """
    url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060"
    print("  Fetching ERCOT historical report index (Report 13060)...")
    try:
        resp = SESSION.get(url, timeout=30)
        data = resp.json()
        docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
        
        doc_map = {}
        for d in docs:
            doc = d.get("Document", {})
            name = doc.get("FriendlyName", "")
            doc_id = doc.get("DocID", "")
            if "DAMLZHBSPP_" in name and doc_id:
                try:
                    year = int(name.split("_")[-1][:4])
                    doc_map[year] = doc_id
                except ValueError:
                    pass
        return doc_map
    except Exception as e:
        print(f"  ✗ Error fetching ERCOT doc list: {e}")
        return {}


def download_and_parse_annual_zip(year: int, doc_id: str) -> pd.DataFrame:
    """Download the ERCOT annual ZIP and parse its contents."""
    url = f"https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc_id}"
    print(f"  [{year}] Downloading ...", end=" ", flush=True)
    
    try:
        resp = SESSION.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"✗ HTTP {resp.status_code}")
            return pd.DataFrame()
            
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # An annual zip usually contains multiple monthly files
            data_files = [f for f in zf.namelist() if f.endswith(".xlsx") or f.endswith(".csv")]
            if not data_files:
                print("✗ No data files in ZIP")
                return pd.DataFrame()
                
            yearly_chunks = []
            for filename in data_files:
                with zf.open(filename) as f:
                    if filename.endswith(".csv"):
                        df_dict = {"sheet1": pd.read_csv(f, low_memory=False)}
                    else:
                        # ERCOT puts each month in a separate sheet
                        df_dict = pd.read_excel(f, sheet_name=None)
                        
                for sheet_name, df in df_dict.items():
                    if df.empty: continue
                    
                    # Parse the dataframe
                    col_map = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
                    
                    date_col = col_map.get("deliverydate")
                    hour_col = col_map.get("hourending")
                    loc_col  = col_map.get("settlementpoint") or col_map.get("settlementpointname")
                    pr_col   = col_map.get("settlementpointprice")
                    
                    if not all([date_col, hour_col, loc_col, pr_col]):
                        continue
                        
                    df = df[[date_col, hour_col, loc_col, pr_col]].copy()
                    df.columns = ["date", "hour_ending", "location", "price"]
                    
                    # Filter locations
                    df["location"] = df["location"].astype(str).str.upper().str.strip()
                    df = df[df["location"].isin(TARGET_HUBS)].copy()
                    
                    if df.empty:
                        continue
                        
                    # Parse Dates and Hours
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["price"] = pd.to_numeric(df["price"], errors="coerce")
                    
                    def parse_hour(h):
                        try:
                            val = str(h).strip().replace(":", "")
                            if val == "0200": return 1
                            return int(float(val)) - 1
                        except:
                            return 0
                    
                    df["hour"] = df["hour_ending"].apply(parse_hour)
                    df["datetime_cst"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
                    df["datetime_utc"] = df["datetime_cst"].dt.tz_localize(
                        "US/Central", ambiguous="NaT", nonexistent="NaT"
                    ).dt.tz_convert("UTC")
                    
                    df = df.dropna(subset=["datetime_utc", "price"])
                    yearly_chunks.append(df[["datetime_utc", "location", "price"]])
            
            if not yearly_chunks:
                print("✗ Failed to parse any files/sheets in ZIP")
                return pd.DataFrame()
                
            combined = pd.concat(yearly_chunks, ignore_index=True)
            print(f"✓ {len(combined):,} rows")
            return combined
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return pd.DataFrame()


def normalize_and_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw multi-hub DataFrame to hourly average price series."""
    df = raw.copy()
    
    # Average HB_HOUSTON + HB_WEST per hour
    hourly = (
        df.groupby("datetime_utc")["price"]
        .mean()
        .to_frame()
        .sort_index()
        .resample("h")
        .mean()
    )

    # Preserve ERCOT extreme prices (Uri crisis ~$9,000/MWh is legitimate)
    hourly["price"] = hourly["price"].clip(upper=15_000.0)
    hourly["price"] = hourly["price"].interpolate(method="linear", limit=6)

    # Filter to requested date range
    start_ts = pd.Timestamp(config.DATA_START, tz="UTC")
    end_ts = pd.Timestamp(config.DATA_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    hourly = hourly[(hourly.index >= start_ts) & (hourly.index <= end_ts)]
    return hourly


def print_summary(df: pd.DataFrame):
    missing = df["price"].isna().mean() * 100
    print(f"\n  ── Data Summary ───────────────────────────────────")
    print(f"  Total rows:   {len(df):,} hourly observations")
    print(f"  Date range:   {df.index.min()} → {df.index.max()}")
    print(f"  Missing:      {missing:.2f}%")
    print(f"  Price ($/MWh): min={df['price'].min():.2f}  "
          f"mean={df['price'].mean():.2f}  max={df['price'].max():.2f}")
    if df['price'].max() > 500:
        pct_extreme = (df['price'] > 500).mean() * 100
        print(f"  Spike >$500:  {pct_extreme:.2f}% of hours")


def main():
    print("=" * 65)
    print("  ERCOT Day-Ahead SPP Downloader (V2 — ANNUAL BULK)")
    print(f"  Hubs:   HB_HOUSTON + HB_WEST (averaged)")
    print(f"  Period: {config.DATA_START} → {config.DATA_END}")
    print("=" * 65)
    print()

    os.makedirs(config.RAW_DIR, exist_ok=True)
    out_path = config.ERCOT_RAW_PATH

    if os.path.exists(out_path):
        df_ex = pd.read_parquet(out_path)
        print(f"  ⚠ File exists: {out_path}")
        print(f"    {df_ex.shape[0]:,} rows | "
              f"{df_ex.index.min().date()} → {df_ex.index.max().date()}")
        ans = input("  Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("  Skipping.")
            return

    # 1. Get ERCOT index of annual files
    doc_map = get_annual_doc_ids()
    if not doc_map:
        return

    # 2. Determine required years based on config
    start_year = pd.Timestamp(config.DATA_START).year
    end_year   = pd.Timestamp(config.DATA_END).year
    target_years = list(range(start_year, end_year + 1))
    
    print(f"\n  Downloading {len(target_years)} annual datasets...")
    
    all_rows = []
    for year in target_years:
        doc_id = doc_map.get(year)
        if not doc_id:
            print(f"  [{year}] ✗ No ERCOT annual file found for this year")
            continue
            
        df = download_and_parse_annual_zip(year, doc_id)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("\n  ❌ Failed to download any data.")
        return

    # 3. Aggregate
    raw = pd.concat(all_rows, ignore_index=True)
    
    print("\n  Normalizing and aggregating to hourly series...")
    df = normalize_and_aggregate(raw)
    print_summary(df)

    df.to_parquet(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  ✅ Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
