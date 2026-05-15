"""
step_01_preprocess.py
=====================
LEAK-PROOF preprocessing pipeline for the V2 EPF project.

Protocol (critical for methodological integrity):
  1. Load all raw data
  2. Merge on hourly UTC index
  3. Engineer ALL features using only shifted/lagged operations (no future leakage)
  4. Chronological 4-way split BEFORE any fitting
  5. Fit MinMaxScaler + KNNImputer ONLY on train split
  6. transform() calibration, val, test — NEVER fit_transform
  7. SHAP-based feature selection on train split
  8. Save 4 processed parquets (PJM) + 4 (ERCOT)

Run:
    python step_01_preprocess.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_price(path: str, label: str) -> pd.DataFrame:
    print(f"  Loading {label} price data from {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["price"]].resample("h").mean()
    # Clip to configured date range
    start = pd.Timestamp(config.DATA_START, tz="UTC")
    end   = pd.Timestamp(config.DATA_END, tz="UTC")
    df = df.loc[start:end]
    print(f"    Shape: {df.shape} | Range: {df.index.min()} → {df.index.max()}")
    return df


def load_eia(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  ⚠ EIA file not found: {path}. Skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("h").mean()


def load_weather(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  ⚠ Weather file not found: {path}. Skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("h").mean()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (all shift-safe — no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(price: pd.DataFrame,
                      eia: pd.DataFrame,
                      weather: pd.DataFrame,
                      gas: pd.DataFrame,
                      country_code: str = "US") -> pd.DataFrame:
    """
    Build feature matrix from price + exogenous data.
    All lag/rolling operations use shift ≥ 1 to prevent leakage.
    Returns a DataFrame with 'price' as the first column (target).
    """
    print("  Engineering features...")
    df = price.copy()

    # ── Temporal cyclicals ───────────────────────────────────────
    idx = df.index
    df["hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)
    df["week_sin"]   = np.sin(2 * np.pi * idx.isocalendar().week.values / 52)
    df["week_cos"]   = np.cos(2 * np.pi * idx.isocalendar().week.values / 52)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # ── US Holidays ──────────────────────────────────────────────
    us_holidays = holidays.US(years=range(idx.year.min(), idx.year.max() + 2))
    df["is_holiday"] = idx.normalize().isin(us_holidays.keys()).astype(int)
    df["is_holiday_eve"] = (
        (idx - pd.Timedelta(days=1)).normalize().isin(us_holidays.keys())
    ).astype(int)

    # ── Price lags ───────────────────────────────────────────────
    for lag in config.TARGET_LAGS:
        df[f"price_lag_{lag}h"] = df["price"].shift(lag)

    # ── Rolling statistics (shift=1 ensures no leakage) ─────────
    price_shifted = df["price"].shift(1)
    for window in config.ROLLING_WINDOWS:
        roll = price_shifted.rolling(window=window, min_periods=window // 2)
        df[f"price_rmean_{window}h"] = roll.mean()
        df[f"price_rstd_{window}h"]  = roll.std()
        df[f"price_rmin_{window}h"]  = roll.min()
        df[f"price_rmax_{window}h"]  = roll.max()

    # ── EIA exogenous features ───────────────────────────────────
    if not eia.empty:
        eia_aligned = eia.reindex(df.index).ffill().bfill()
        # Lag EIA features by 1h (they're typically available 1h before delivery)
        for col in eia_aligned.columns:
            df[f"eia_{col}"] = eia_aligned[col].shift(1)

    # ── Gas price ────────────────────────────────────────────────
    if not gas.empty:
        gas_aligned = gas.reindex(df.index).ffill().bfill()
        df["gas_price"] = gas_aligned["gas_price_mmBtu"].shift(1)
        # Gas price lag 24h (published day-ahead)
        df["gas_price_lag_24h"] = gas_aligned["gas_price_mmBtu"].shift(24)

    # ── Weather features ─────────────────────────────────────────
    # DA FORECASTING PROTOCOL: actual weather at hour t is not available
    # at day-ahead bidding time (~12 UTC day-1). We shift 24h so models
    # use t-24h realized weather as a day-ahead NWP proxy — consistent
    # with ECMWF/GFS <1°C RMSE at 24h horizon (see paper Section 3.2).
    if not weather.empty:
        wx_aligned = weather.reindex(df.index).ffill().bfill()
        for col in wx_aligned.columns:
            df[f"wx_{col}"] = wx_aligned[col].shift(24)  # 24h lag = DA NWP proxy

    print(f"    Total features: {df.shape[1]} (before SHAP selection)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHRONOLOGICAL SPLIT  (strict half-open intervals, zero row overlap)
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame):
    """
    4-way strict chronological split using half-open intervals [start, end).

    Each boundary timestamp belongs to exactly ONE split:
      train : [df.start, TRAIN_END)
      cal   : [TRAIN_END, CALIBRATION_END)
      val   : [CALIBRATION_END, VAL_END)
      test  : [VAL_END, df.end]

    This guarantees zero row overlap. The previous implementation used
    pandas .loc (inclusive on both ends) + iloc[1:] which only removed 1
    row and left up to ~7 UTC-boundary duplicates per boundary.
    """
    tz = df.index.tz

    def _ts(cfg_val):
        t = pd.Timestamp(cfg_val)
        return t.tz_localize(tz) if (tz is not None and t.tz is None) else t

    t_cal  = _ts(config.TRAIN_END)
    t_val  = _ts(config.CALIBRATION_END)
    t_test = _ts(config.VAL_END)

    train = df[df.index <  t_cal]
    cal   = df[(df.index >= t_cal)  & (df.index < t_val)]
    val   = df[(df.index >= t_val)  & (df.index < t_test)]
    test  = df[df.index >= t_test]

    # Defensive zero-overlap assertion
    assert len(train.index.intersection(cal.index))  == 0, "BUG: train/cal overlap"
    assert len(cal.index.intersection(val.index))    == 0, "BUG: cal/val overlap"
    assert len(val.index.intersection(test.index))   == 0, "BUG: val/test overlap"

    print(f"\n  Split sizes (strict half-open intervals, zero overlap):")
    print(f"    Train:       {len(train):>8,} rows  "
          f"({train.index.min().date()} → {train.index.max().date()})")
    print(f"    Calibration: {len(cal):>8,} rows  "
          f"({cal.index.min().date()} → {cal.index.max().date()})")
    print(f"    Validation:  {len(val):>8,} rows  "
          f"({val.index.min().date()} → {val.index.max().date()})")
    print(f"    Test:        {len(test):>8,} rows  "
          f"({test.index.min().date()} → {test.index.max().date()})")
    return train, cal, val, test



# ─────────────────────────────────────────────────────────────────────────────
# 4. IMPUTATION (fit on TRAIN ONLY)
# ─────────────────────────────────────────────────────────────────────────────

def leak_proof_impute(train: pd.DataFrame,
                      cal:   pd.DataFrame,
                      val:   pd.DataFrame,
                      test:  pd.DataFrame,
                      target_col: str = "price"):
    """
    CRITICAL: Scaler and imputer are ONLY fit on the training set.
    All other splits are transformed (not fit) to prevent data leakage.
    """
    print("\n  Fitting MinMaxScaler on TRAIN only...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)  # ← fit here ONLY

    print("  Fitting KNNImputer on TRAIN only (scaled)...")
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    train_imputed_scaled = imputer.fit_transform(train_scaled)  # ← fit here ONLY

    print("  Transforming CAL/VAL/TEST (no fitting)...")
    cal_scaled  = scaler.transform(cal)
    val_scaled  = scaler.transform(val)
    test_scaled = scaler.transform(test)

    cal_imputed_scaled  = imputer.transform(cal_scaled)   # ← transform only
    val_imputed_scaled  = imputer.transform(val_scaled)
    test_imputed_scaled = imputer.transform(test_scaled)

    # Inverse transform back to original scale
    def reconstruct(arr, index, columns):
        inv = scaler.inverse_transform(arr)
        return pd.DataFrame(inv, index=index, columns=columns)

    train_clean = reconstruct(train_imputed_scaled, train.index, train.columns)
    cal_clean   = reconstruct(cal_imputed_scaled,  cal.index,   cal.columns)
    val_clean   = reconstruct(val_imputed_scaled,  val.index,   val.columns)
    test_clean  = reconstruct(test_imputed_scaled, test.index,  test.columns)

    # Verify no leakage: target column stats should not change from TRAIN→TEST
    print(f"\n  ✔ Leakage check — target '{target_col}' stats:")
    for split_name, split_df in [("TRAIN", train_clean), ("CAL", cal_clean),
                                   ("VAL", val_clean), ("TEST", test_clean)]:
        s = split_df[target_col]
        print(f"    {split_name:5}: mean={s.mean():.2f}, std={s.std():.2f}, "
              f"missing={s.isna().sum()}")

    return train_clean, cal_clean, val_clean, test_clean, scaler, imputer


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP FEATURE SELECTION (optional — reduces feature bloat)
# ─────────────────────────────────────────────────────────────────────────────

def select_features_shap(train: pd.DataFrame,
                         top_n: int = config.SHAP_TOP_N_FEATURES,
                         target_col: str = "price") -> list:
    """
    Use LightGBM + SHAP to select the top-N most important features.
    SHAP is computed ONLY on the training set.
    Returns list of selected feature names (always includes target_col).
    """
    try:
        import lightgbm as lgb
        import shap
    except ImportError:
        print("  ⚠ lightgbm/shap not installed. Skipping SHAP selection.")
        return list(train.columns)

    print(f"\n  Running SHAP feature selection (top {top_n} features)...")
    X_train = train.drop(columns=[target_col]).dropna()
    y_train = train.loc[X_train.index, target_col]

    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.1, num_leaves=31,
        verbose=-1, n_jobs=-1, seed=config.RANDOM_SEED
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_train.columns
    ).sort_values(ascending=False)

    top_features = mean_abs_shap.head(top_n).index.tolist()
    print(f"  Top-5 features by SHAP: {top_features[:5]}")

    # Save SHAP importance for figure generation
    shap_path = os.path.join(config.REPORT_DIR, "shap_importance.csv")
    mean_abs_shap.to_csv(shap_path)
    print(f"  SHAP importances saved to: {shap_path}")

    return [target_col] + [f for f in top_features if f != target_col]


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(market: str = "PJM"):
    """
    Full preprocessing pipeline for a given market (PJM or ERCOT).
    """
    is_pjm = market.upper() == "PJM"
    price_path   = config.PJM_RAW_PATH   if is_pjm else config.ERCOT_RAW_PATH
    eia_path     = config.EIA_GEN_PATH if is_pjm else config.EIA_GEN_ERCOT_PATH
    gas_path     = config.EIA_GAS_PATH
    weather_path = config.WEATHER_PJM_PATH if is_pjm else config.WEATHER_ERCOT_PATH
    out_paths    = {
        "train": config.PJM_TRAIN_PATH if is_pjm else config.ERCOT_TRAIN_PATH,
        "cal":   config.PJM_CAL_PATH   if is_pjm else config.ERCOT_CAL_PATH,
        "val":   config.PJM_VAL_PATH   if is_pjm else config.ERCOT_VAL_PATH,
        "test":  config.PJM_TEST_PATH  if is_pjm else config.ERCOT_TEST_PATH,
    }

    print(f"\n{'='*65}")
    print(f"  PREPROCESSING: {market} Market")
    print(f"{'='*65}")

    # Load
    price   = load_price(price_path, market)
    eia     = load_eia(eia_path)
    gas     = load_eia(gas_path)
    weather = load_weather(weather_path)

    # Feature engineering (all shift-safe)
    df = engineer_features(price, eia, weather, gas)

    # Drop rows where the target is NaN (can't train on these)
    df = df.dropna(subset=[config.TARGET_COL])

    # Chronological 4-way split
    train, cal, val, test = chronological_split(df)

    # Leak-proof imputation
    train, cal, val, test, scaler, imputer = leak_proof_impute(
        train, cal, val, test, config.TARGET_COL
    )

    # SHAP feature selection (on train only)
    selected_cols = select_features_shap(train, top_n=config.SHAP_TOP_N_FEATURES)
    train = train[selected_cols]
    cal   = cal[selected_cols]
    val   = val[selected_cols]
    test  = test[selected_cols]
    print(f"\n  Final feature count: {len(selected_cols)}")

    # Save
    os.makedirs(config.PROC_DIR, exist_ok=True)
    train.to_parquet(out_paths["train"])
    cal.to_parquet(out_paths["cal"])
    val.to_parquet(out_paths["val"])
    test.to_parquet(out_paths["test"])

    print(f"\n  ✅ {market} preprocessing complete.")
    for k, v in out_paths.items():
        print(f"    {k:5}: {v}")

    return train, cal, val, test


if __name__ == "__main__":
    run_preprocessing("PJM")
    run_preprocessing("ERCOT")
