"""
step_08b_nhits_quantile.py — N-HiTS with Quantile Loss (MQLoss)
================================================================
Trains N-HiTS with MQLoss to produce native quantile outputs:
  q10, q50 (median), q90  →  80% prediction interval

This satisfies the "Deep Ensemble Quantile" paradigm from Section 3.6
of the implementation plan. Enables CRPS computation for RQ2.

neuralforecast 3.1.8 | h=24 day-ahead | cross_validation on test set.

Run:
    python step_08b_nhits_quantile.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss


def to_nf(df):
    return pd.DataFrame({
        "unique_id": "price",
        "ds": pd.to_datetime(df.index, utc=True),
        "y":  df[config.TARGET_COL].values,
    })


def train_nhits_quantile(market="PJM"):
    print(f"\n{'='*65}\n  N-HiTS Quantile (80% CI): {market}\n{'='*65}")

    tr  = pd.read_parquet(config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH)
    val = pd.read_parquet(config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH)
    te  = pd.read_parquet(config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH)

    train_val = pd.concat([to_nf(tr), to_nf(val)], ignore_index=True)
    all_df    = pd.concat([to_nf(tr), to_nf(val), to_nf(te)], ignore_index=True)
    val_size  = len(val)
    n_windows = len(te) // 24
    print(f"  Train+Val: {len(train_val):,} | Test: {len(te):,} | CV windows: {n_windows}")
    print(f"  Quantiles: q10 / q50 / q90  (80% nominal CI)")

    model = NHITS(
        h=24,
        input_size=config.SEQ_LEN_DEFAULT,
        stack_types=["identity", "identity", "identity"],
        n_blocks=[1, 1, 1],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        n_pool_kernel_size=[2, 2, 1],
        n_freq_downsample=[4, 2, 1],
        dropout_prob_theta=0.2,
        # MQLoss with level=[80] → outputs q10 (lo-80), median, q90 (hi-80)
        loss=MQLoss(level=[80]),
        learning_rate=config.LEARNING_RATE,
        max_steps=1500,
        batch_size=config.BATCH_SIZE,
        val_monitor="train_loss",
        scaler_type="standard",
        random_seed=config.RANDOM_SEED,
    )

    nf = NeuralForecast(models=[model], freq="h")

    t0 = time.time()
    nf.fit(df=train_val, val_size=val_size)
    print(f"  Fit: {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    cv = nf.cross_validation(df=all_df, n_windows=n_windows, step_size=24)
    print(f"  CV:  {(time.time()-t0)/60:.1f} min | CV rows: {len(cv):,}")

    print(f"\n  CV columns: {list(cv.columns)}")

    # Column naming: NHITS-median, NHITS-lo-80, NHITS-hi-80
    median_col = next((c for c in cv.columns if "median" in c.lower()), None)
    lo_col     = next((c for c in cv.columns if "lo-80"  in c.lower()), None)
    hi_col     = next((c for c in cv.columns if "hi-80"  in c.lower()), None)

    if median_col is None:
        # Fallback: first non-standard column
        extra_cols = [c for c in cv.columns if c not in ["ds", "unique_id", "y", "cutoff"]]
        print(f"  Fallback columns: {extra_cols}")
        median_col = extra_cols[len(extra_cols)//2] if extra_cols else None
        lo_col     = extra_cols[0]               if len(extra_cols) > 1 else None
        hi_col     = extra_cols[-1]              if len(extra_cols) > 1 else None

    out_df = cv[["ds", "y"]].rename(columns={"y": "actual"})
    if median_col: out_df["q50"] = cv[median_col].values
    if lo_col:     out_df["q10"] = cv[lo_col].values
    if hi_col:     out_df["q90"] = cv[hi_col].values

    # Metrics
    mask = ~out_df["actual"].isna()
    if "q50" in out_df.columns:
        mae  = np.nanmean(np.abs(out_df.loc[mask, "actual"].values - out_df.loc[mask, "q50"].values))
        print(f"\n  N-HiTS-Q {market}  Median MAE={mae:.4f} $/MWh")
    if "q10" in out_df.columns and "q90" in out_df.columns:
        picp = np.nanmean(
            (out_df.loc[mask, "actual"].values >= out_df.loc[mask, "q10"].values) &
            (out_df.loc[mask, "actual"].values <= out_df.loc[mask, "q90"].values)
        ) * 100
        mpiw = np.nanmean(out_df.loc[mask, "q90"].values - out_df.loc[mask, "q10"].values)
        print(f"  PICP (80% CI): {picp:.2f}%  MPIW: {mpiw:.4f} $/MWh")

    # Save
    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    nf.save(os.path.join(config.MODEL_DIR, f"nhits_quantile_{market.lower()}"), overwrite=True)

    out_path = os.path.join(config.REPORT_DIR, f"nhits_quantile_preds_{market.lower()}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  ✅ Model: models/nhits_quantile_{market.lower()}/")
    print(f"  ✅ Preds: {out_path}")
    return out_df


if __name__ == "__main__":
    train_nhits_quantile("PJM")
    train_nhits_quantile("ERCOT")
