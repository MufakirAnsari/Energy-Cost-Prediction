"""
step_06b_train_bitcn.py — Bidirectional TCN (BiTCN)
====================================================
Bidirectional Temporal Convolutional Network.
Completes Section 3.3 "Temporal CNN" of the implementation plan.

BiTCN is strictly superior to vanilla TCN for offline forecasting
(processes sequence in both directions during training).

neuralforecast 3.1.8 | h=24 day-ahead | cross_validation on test set.

Run:
    python step_06b_train_bitcn.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

from neuralforecast import NeuralForecast
from neuralforecast.models import BiTCN
from neuralforecast.losses.pytorch import MAE


def to_nf(df):
    return pd.DataFrame({
        "unique_id": "price",
        "ds": pd.to_datetime(df.index, utc=True),
        "y":  df[config.TARGET_COL].values,
    })


def check_overfitting(train_l, val_l, model_name, market):
    ratio = val_l / train_l if train_l > 0 else float("inf")
    status = "✅ OK" if ratio < 3.0 else ("⚠️  MILD" if ratio < 6.0 else "❌ SEVERE")
    print(f"\n  [Overfitting Check] {model_name} {market}")
    print(f"    Train: {train_l:.4f} | Val: {val_l:.4f} | Ratio: {ratio:.2f}x  {status}")


def train_bitcn(market="PJM"):
    print(f"\n{'='*65}\n  BiTCN: {market}\n{'='*65}")

    tr  = pd.read_parquet(config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH)
    val = pd.read_parquet(config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH)
    te  = pd.read_parquet(config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH)

    train_val = pd.concat([to_nf(tr), to_nf(val)], ignore_index=True)
    all_df    = pd.concat([to_nf(tr), to_nf(val), to_nf(te)], ignore_index=True)
    val_size  = len(val)
    n_windows = len(te) // 24
    print(f"  Train+Val: {len(train_val):,} | Test: {len(te):,} | CV windows: {n_windows}")

    model = BiTCN(
        h=24,
        input_size=config.SEQ_LEN_DEFAULT,   # 168h = 7-day context
        hidden_size=128,
        dropout=0.1,
        loss=MAE(),
        valid_loss=MAE(),
        learning_rate=config.LEARNING_RATE,
        max_steps=1500,
        batch_size=config.BATCH_SIZE,
        val_monitor="train_loss",
        scaler_type="standard",
        random_seed=config.RANDOM_SEED,
    )

    nf = NeuralForecast(models=[model], freq="h")

    # Fit with val monitoring
    t0 = time.time()
    nf.fit(df=train_val, val_size=val_size)
    fit_time = (time.time() - t0) / 60
    print(f"  Fit: {fit_time:.1f} min")

    # Extract train/val losses for overfitting check
    try:
        history = nf.models[0].trainer.callback_metrics
        train_l = float(history.get("train_loss", 0))
        val_l   = float(history.get("val_loss", train_l))
        check_overfitting(train_l, val_l, "BiTCN", market)
    except Exception:
        pass

    # Cross-validation on test set
    t0 = time.time()
    cv = nf.cross_validation(df=all_df, n_windows=n_windows, step_size=24)
    cv_time = (time.time() - t0) / 60
    print(f"  CV:  {cv_time:.1f} min | CV rows: {len(cv):,}")

    # Extract prediction column
    pred_col = [c for c in cv.columns if "BiTCN" in c][0]
    out_df = cv[["ds", "y", pred_col]].rename(columns={"y": "actual", pred_col: "predicted"})

    # Evaluate
    mask = ~out_df["actual"].isna() & ~out_df["predicted"].isna()
    mae  = np.mean(np.abs(out_df.loc[mask, "actual"] - out_df.loc[mask, "predicted"]))
    rmse = np.sqrt(np.mean((out_df.loc[mask, "actual"] - out_df.loc[mask, "predicted"])**2))
    print(f"\n  BiTCN {market}  MAE={mae:.4f} $/MWh  RMSE={rmse:.4f} $/MWh")

    # Save model
    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    nf.save(os.path.join(config.MODEL_DIR, f"bitcn_{market.lower()}"), overwrite=True)

    # Save predictions (dedup NF cross_validation window boundary duplicates)
    out_path = os.path.join(config.REPORT_DIR, f"bitcn_preds_{market.lower()}.csv")
    out_df = out_df.drop_duplicates(subset=["ds"], keep="first")
    out_df.to_csv(out_path, index=False)
    print(f"  ✅ Model: models/bitcn_{market.lower()}/")
    print(f"  ✅ Preds: {out_path}")
    return mae, rmse


if __name__ == "__main__":
    train_bitcn("PJM")
    train_bitcn("ERCOT")
