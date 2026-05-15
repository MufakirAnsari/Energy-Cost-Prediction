"""
step_07b_train_tft.py — Temporal Fusion Transformer (TFT)
==========================================================
GPU-optimized for GTX 1650 (3.63 GB VRAM):
  - hidden_size: 32  (was 64 — halves attention memory)
  - n_head: 2        (was 4)
  - hist_exog: 12 top features (was 40 — main OOM cause)
  - batch_size: 64   (was 256)
  - windows_batch_size: 64
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Output format is identical: ds, actual, predicted columns.

Run: python step_07b_train_tft.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

# Must set BEFORE importing torch/pytorch_lightning
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(__file__))
import config
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MAE

# Top-12 features using ACTUAL column names from the parquet files
# (verified from: price, price_lag_1h, price_lag_24h, hour_cos, hour_sin,
#  price_lag_336h, price_lag_168h, wx_columbus_temp_c, gas_price, ...)
TOP_HIST_EXOG = [
    "price_lag_1h",
    "price_lag_24h",
    "price_lag_168h",
    "price_lag_336h",
    "gas_price",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "wx_columbus_temp_c",
    "price_rstd_6h",
    "eia_nuclear_mw",
]


def prepare_nf_df(df, market, hist_exog):
    nf = pd.DataFrame()
    nf["ds"]        = df.index.tz_localize(None) if df.index.tz else df.index
    nf["y"]         = df[config.TARGET_COL].values
    nf["unique_id"] = market
    for col in hist_exog:
        if col in df.columns:
            nf[col] = df[col].values
    return nf


def train_tft(market="PJM"):
    print(f"\n{'='*65}\n  TFT: {market}  [GTX-1650 memory-optimized]\n{'='*65}")

    tr_df  = pd.read_parquet(config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH)
    val_df = pd.read_parquet(config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH)
    te_df  = pd.read_parquet(config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH)

    # Use intersection of TOP_HIST_EXOG with available columns;
    # fall back to first 12 non-target cols if fewer than 4 matched
    hist_exog = [c for c in TOP_HIST_EXOG if c in tr_df.columns]
    if len(hist_exog) < 4:
        hist_exog = [c for c in tr_df.columns if c != config.TARGET_COL][:12]
    print(f"  hist_exog ({len(hist_exog)}): {hist_exog}")

    train_val = pd.concat([tr_df, val_df])
    val_size  = len(val_df)
    # n_windows covers the full test set: test_size / horizon
    n_windows = len(te_df) // 24
    print(f"  Train+Val: {len(train_val):,} | Test: {len(te_df):,} | n_windows: {n_windows}")

    model = TFT(
        h=24,
        input_size=72,              # 3-day context (was 168 — big memory saver)
        hidden_size=32,             # 32 instead of 64 — halves attention map memory
        n_head=2,                   # 2 instead of 4 — fits 32-dim hidden
        dropout=0.1,
        hist_exog_list=hist_exog,
        loss=MAE(),
        valid_loss=MAE(),
        learning_rate=config.LEARNING_RATE,
        max_steps=1500,
        batch_size=64,              # 64 instead of 256 — critical for VRAM
        windows_batch_size=64,      # match batch_size
        inference_windows_batch_size=32,   # conservative for predict pass
        val_monitor="train_loss",
        scaler_type="standard",
        random_seed=config.RANDOM_SEED,
    )

    nf = NeuralForecast(models=[model], freq="h")

    t0 = time.time()
    nf.fit(df=prepare_nf_df(train_val, market, hist_exog), val_size=val_size)
    print(f"  Fit: {(time.time()-t0)/60:.1f} min")

    m_key = market.lower()
    model_dir = os.path.join(config.MODEL_DIR, f"tft_{m_key}")
    nf.save(model_dir, overwrite=True)

    # Cross-validation for OOS test predictions (same as all other NF models)
    # n_windows covers exactly the test set, processed inference_windows_batch_size at a time
    nf_all = prepare_nf_df(pd.concat([train_val, te_df]), market, hist_exog)
    t0 = time.time()
    cv = nf.cross_validation(df=nf_all, n_windows=n_windows, step_size=24)
    print(f"  CV: {(time.time()-t0)/60:.1f} min | rows: {len(cv):,}")

    # Extract predicted column (same output format as all other NF models)
    pred_col = next((c for c in cv.columns
                     if "TFT" in c and "lo" not in c and "hi" not in c
                     and c not in ["unique_id","ds","cutoff","y"]), None)
    if pred_col is None:
        pred_col = [c for c in cv.columns
                    if c not in ["unique_id","ds","cutoff","y"]][0]

    out = pd.DataFrame({
        "ds":        pd.to_datetime(cv["ds"].values, utc=True),
        "actual":    cv["y"].values,
        "predicted": cv[pred_col].values,
    }).set_index("ds")
    out = out[~out.index.duplicated(keep="first")]

    mask = ~np.isnan(out["actual"])
    mae_  = np.mean(np.abs(out.loc[mask,"actual"] - out.loc[mask,"predicted"]))
    rmse_ = np.sqrt(np.mean((out.loc[mask,"actual"] - out.loc[mask,"predicted"])**2))
    print(f"\n  TFT {market}  MAE={mae_:.4f}  RMSE={rmse_:.4f} $/MWh")

    pred_path = os.path.join(config.REPORT_DIR, f"tft_preds_{m_key}.csv")
    out.to_csv(pred_path)
    print(f"  ✅ Model: {model_dir}/")
    print(f"  ✅ Preds: {pred_path}")


if __name__ == "__main__":
    train_tft("PJM")
    train_tft("ERCOT")

