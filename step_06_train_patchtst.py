"""
step_06_train_patchtst.py — PatchTST (Nie et al., ICLR 2023)
neuralforecast 3.1.8 compatible. h=24 day-ahead, cross_validation.
"""
import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE


def to_nf(df):
    return pd.DataFrame({
        "unique_id": "price",
        "ds": pd.to_datetime(df.index, utc=True),
        "y":  df[config.TARGET_COL].values,
    })


def check_overfitting(train_loss, val_loss, model_name, market):
    """Warn if val/train loss ratio indicates overfitting."""
    ratio = val_loss / train_loss if train_loss > 0 else float("inf")
    status = "✅ OK" if ratio < 3.0 else ("⚠️  MILD" if ratio < 6.0 else "❌ SEVERE")
    print(f"\n  [Overfitting Check] {model_name} {market}")
    print(f"    Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Ratio: {ratio:.2f}x  {status}")
    if ratio >= 3.0:
        print(f"    → Tip: increase dropout or reduce max_steps")
    return ratio


def train_patchtst(market="PJM"):
    print(f"\n{'='*65}\n  PatchTST: {market}\n{'='*65}")

    tr  = pd.read_parquet(config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH)
    val = pd.read_parquet(config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH)
    te  = pd.read_parquet(config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH)

    train_val = pd.concat([to_nf(tr), to_nf(val)], ignore_index=True)
    all_df    = pd.concat([to_nf(tr), to_nf(val), to_nf(te)], ignore_index=True)
    val_size  = len(val)
    n_windows = len(te) // 24
    print(f"  Train+Val: {len(train_val):,} | Test: {len(te):,} | CV windows: {n_windows}")

    model = PatchTST(
        h=24,
        input_size=config.SEQ_LEN_DEFAULT,
        patch_len=24,
        stride=12,
        hidden_size=64,
        linear_hidden_size=128,
        n_heads=4,
        encoder_layers=2,
        dropout=0.2,              # increased from 0.1 to reduce overfitting
        head_dropout=0.1,
        learning_rate=config.LEARNING_RATE,
        max_steps=1500,
        batch_size=config.BATCH_SIZE,
        loss=MAE(),
        valid_loss=MAE(),
        val_monitor="train_loss",  # cross_validation has no val set
        scaler_type="standard",
        random_seed=config.RANDOM_SEED,
    )

    nf = NeuralForecast(models=[model], freq="h")

    t0 = time.time()
    nf.fit(df=train_val, val_size=val_size)
    elapsed_fit = time.time() - t0
    # NOTE: NF 3.x train_loss is normalized (0-1 scale), valid_loss is in $/MWh
    # They are NOT comparable — overfitting is assessed via test MAE below
    print(f"  Fit: {elapsed_fit/60:.1f} min")

    # Rolling day-ahead cross-validation on test set
    t0 = time.time()
    cv = nf.cross_validation(df=all_df, n_windows=n_windows, step_size=24)
    print(f"  CV:  {(time.time()-t0)/60:.1f} min | CV rows: {len(cv):,}")

    # cv already contains 'y' (actual) — no positional slicing needed
    pred_col = [c for c in cv.columns if "PatchTST" in c][0]
    out_df = cv[["ds", "y", pred_col]].rename(columns={"y": "actual", pred_col: "predicted"})

    mask = ~out_df["actual"].isna() & ~out_df["predicted"].isna()
    mae  = np.mean(np.abs(out_df.loc[mask, "actual"] - out_df.loc[mask, "predicted"]))
    rmse = np.sqrt(np.mean((out_df.loc[mask, "actual"] - out_df.loc[mask, "predicted"])**2))
    picp_lo = out_df.loc[mask, "actual"].quantile(0.05)
    print(f"\n  PatchTST {market}  MAE={mae:.4f} $/MWh  RMSE={rmse:.4f} $/MWh")

    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    nf.save(os.path.join(config.MODEL_DIR, f"patchtst_{market.lower()}"), overwrite=True)

    out_path = os.path.join(config.REPORT_DIR, f"patchtst_preds_{market.lower()}.csv")
    out_df = out_df.drop_duplicates(subset=["ds"], keep="first")  # NF cv window dedup
    out_df.to_csv(out_path, index=False)
    print(f"  ✅ Saved: {out_path}")


if __name__ == "__main__":
    train_patchtst("PJM")
    train_patchtst("ERCOT")
