"""
step_07_train_itransformer.py — iTransformer (Liu et al., ICLR 2024)
neuralforecast 3.1.8 compatible. h=24 day-ahead, cross_validation.
"""
import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer
from neuralforecast.losses.pytorch import MAE


def to_nf(df):
    return pd.DataFrame({
        "unique_id": "price",
        "ds": pd.to_datetime(df.index, utc=True),
        # Variance Stabilizing Transformation to fix scale bias
        "y":  np.arcsinh(df[config.TARGET_COL].values),
    })


def check_overfitting(train_l, val_l, model_name, market):
    ratio = val_l / train_l if train_l > 0 else float("inf")
    status = "✅ OK" if ratio < 3.0 else ("⚠️  MILD" if ratio < 6.0 else "❌ SEVERE")
    print(f"\n  [Overfitting Check] {model_name} {market}")
    print(f"    Train: {train_l:.4f} | Val: {val_l:.4f} | Ratio: {ratio:.2f}x  {status}")


def train_itransformer(market="PJM"):
    print(f"\n{'='*65}\n  iTransformer: {market}\n{'='*65}")

    tr  = pd.read_parquet(config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH)
    val = pd.read_parquet(config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH)
    te  = pd.read_parquet(config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH)

    train_val = pd.concat([to_nf(tr), to_nf(val)], ignore_index=True)
    all_df    = pd.concat([to_nf(tr), to_nf(val), to_nf(te)], ignore_index=True)
    val_size  = len(val)
    n_windows = len(te) // 24
    print(f"  Train+Val: {len(train_val):,} | Test: {len(te):,} | CV windows: {n_windows}")

    model = iTransformer(
        h=24,
        input_size=config.SEQ_LEN_DEFAULT,
        n_series=1,
        hidden_size=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.2,
        learning_rate=config.LEARNING_RATE,
        max_steps=1500,
        batch_size=config.BATCH_SIZE,
        loss=MAE(),
        valid_loss=MAE(),
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
    # cv already contains 'y' (actual) - inverse transform VST
    pred_col = [c for c in cv.columns if "iTransformer" in c][0]
    out_df = cv[["ds", "y", pred_col]].rename(columns={"y": "actual", pred_col: "predicted"})
    out_df["actual"] = np.sinh(out_df["actual"])
    out_df["predicted"] = np.sinh(out_df["predicted"])

    mask = ~out_df["actual"].isna() & ~out_df["predicted"].isna()
    mae  = np.mean(np.abs(out_df.loc[mask, "actual"] - out_df.loc[mask, "predicted"]))
    rmse = np.sqrt(np.mean((out_df.loc[mask, "actual"] - out_df.loc[mask, "predicted"])**2))
    print(f"\n  iTransformer {market}  MAE={mae:.4f} $/MWh  RMSE={rmse:.4f} $/MWh")

    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    nf.save(os.path.join(config.MODEL_DIR, f"itransformer_{market.lower()}"), overwrite=True)

    out_path = os.path.join(config.REPORT_DIR, f"itransformer_preds_{market.lower()}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  ✅ Saved: {out_path}")


if __name__ == "__main__":
    train_itransformer("PJM")
    train_itransformer("ERCOT")
