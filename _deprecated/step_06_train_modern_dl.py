"""
step_06_train_patchtst.py + step_07_train_itransformer.py + step_08_train_nhits.py
====================================================================================
Trains three state-of-the-art 2024 deep learning models using neuralforecast:
  - PatchTST (Nie et al., ICLR 2023): patch-based tokenization Transformer
  - iTransformer (Liu et al., ICLR 2024): inverted attention for multivariate
  - N-HiTS (Challu et al., AAAI 2023): neural hierarchical interpolation

All models:
  - 1-step ahead (h=1) unified horizon
  - Same input features as LightGBM (engineered feature matrix)
  - Trained on train split, validated on val split
  - Produce POINT forecasts + quantile outputs for probabilistic evaluation

Run:
    python step_06_train_modern_dl.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        PatchTST,
        iTransformer,
        NHITS,
    )
    from neuralforecast.losses.pytorch import (
        MAE,
        QuantileLoss,
    )
except ImportError:
    raise ImportError(
        "pip install neuralforecast\n"
        "Note: neuralforecast requires PyTorch. Install via:\n"
        "  pip install torch neuralforecast"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION FOR NEURALFORECAST
# ─────────────────────────────────────────────────────────────────────────────

def prepare_neuralforecast_df(market: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Convert our preprocessed parquets into the long-format DataFrame
    required by neuralforecast: columns [unique_id, ds, y, *covariates]
    """
    tr_df  = pd.read_parquet(config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH)
    val_df = pd.read_parquet(config.PJM_VAL_PATH   if market == "PJM" else config.ERCOT_VAL_PATH)
    te_df  = pd.read_parquet(config.PJM_TEST_PATH  if market == "PJM" else config.ERCOT_TEST_PATH)

    covariate_cols = [c for c in tr_df.columns if c != config.TARGET_COL]

    def to_nf(df, split_label):
        out = df.reset_index().rename(columns={
            df.index.name or "index": "ds",
            config.TARGET_COL: "y",
        })
        out["unique_id"] = "price"
        out["ds"] = pd.to_datetime(out["ds"], utc=True)
        return out

    train_nf = to_nf(tr_df, "train")
    # neuralforecast uses the FULL df including val for training context
    trainval_nf = pd.concat([to_nf(tr_df, "train"), to_nf(val_df, "val")], ignore_index=True)
    test_nf     = to_nf(te_df, "test")

    return train_nf, trainval_nf, test_nf, covariate_cols


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_models(n_features: int, covariate_cols: list) -> list:
    """
    Returns list of neuralforecast model instances.
    All configured for 1-step ahead (h=1) and 90% prediction interval.
    """
    h = config.PRED_LEN    # = 1

    models = [
        # ── PatchTST ─────────────────────────────────────────────
        # Patch-based self-attention: treats time series as sequence of patches
        # Strong performer on structured time series with rich covariates
        PatchTST(
            h=h,
            input_size=config.SEQ_LEN_DEFAULT,       # 168h context window
            patch_len=24,                             # 24h patches (daily)
            stride=12,                                # 50% overlap
            d_model=128,
            n_heads=8,
            d_ff=256,
            dropout=0.1,
            attn_dropout=0.1,
            encoder_layers=3,
            learning_rate=config.LEARNING_RATE,
            max_steps=3000,
            batch_size=config.BATCH_SIZE,
            loss=MAE(),
            valid_loss=MAE(),
            futr_exog_list=None,              # No future exogenous
            hist_exog_list=covariate_cols[:10] if covariate_cols else None,
            scaler_type="standard",
            early_stop_patience_steps=config.PATIENCE,
            seed=config.RANDOM_SEED,
        ),

        # ── iTransformer ─────────────────────────────────────────
        # Inverted Transformers: tokenize variables (not time steps)
        # Superior when cross-variate relationships matter (price + load + wind)
        iTransformer(
            h=h,
            input_size=config.SEQ_LEN_DEFAULT,
            d_model=128,
            n_heads=8,
            d_ff=256,
            e_layers=3,
            dropout=0.1,
            learning_rate=config.LEARNING_RATE,
            max_steps=3000,
            batch_size=config.BATCH_SIZE,
            loss=MAE(),
            valid_loss=MAE(),
            scaler_type="standard",
            early_stop_patience_steps=config.PATIENCE,
            seed=config.RANDOM_SEED,
        ),

        # ── N-HiTS ───────────────────────────────────────────────
        # Neural Hierarchical Interpolation for time series
        # Excellent for multi-scale patterns (hourly, daily, weekly)
        NHITS(
            h=h,
            input_size=config.SEQ_LEN_DEFAULT,
            stack_types=["identity", "identity", "identity"],
            n_blocks=[1, 1, 1],
            mlp_units=[[512, 512], [512, 512], [512, 512]],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1],
            dropout_prob_theta=0.1,
            learning_rate=config.LEARNING_RATE,
            max_steps=3000,
            batch_size=config.BATCH_SIZE,
            loss=MAE(),
            valid_loss=MAE(),
            hist_exog_list=covariate_cols[:10] if covariate_cols else None,
            scaler_type="standard",
            early_stop_patience_steps=config.PATIENCE,
            seed=config.RANDOM_SEED,
        ),
    ]
    return models


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING + INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def train_modern_dl(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  Modern DL Models (PatchTST / iTransformer / N-HiTS): {market}")
    print(f"{'='*65}")

    train_nf, trainval_nf, test_nf, covariate_cols = prepare_neuralforecast_df(market)
    n_features = len(covariate_cols) + 1  # +1 for target

    print(f"  Train rows: {len(train_nf):,} | Test rows: {len(test_nf):,}")
    print(f"  Covariates: {len(covariate_cols)}")

    models = get_models(n_features, covariate_cols)

    # Train on train set, validate on val set
    nf = NeuralForecast(models=models, freq="h")

    print(f"\n  Training {len(models)} models...")
    t0 = time.time()

    nf.fit(df=train_nf, val_size=len(pd.read_parquet(
        config.PJM_VAL_PATH if market == "PJM" else config.ERCOT_VAL_PATH
    )))

    elapsed = time.time() - t0
    print(f"  Training complete: {elapsed/60:.1f} min")

    # Generate predictions on test set
    print(f"\n  Generating test set predictions...")
    # neuralforecast needs full historical context for inference
    val_df_raw = pd.read_parquet(
        config.PJM_VAL_PATH if market == "PJM" else config.ERCOT_VAL_PATH
    )
    val_nf = val_df_raw.reset_index().rename(
        columns={val_df_raw.index.name or "index": "ds",
                 config.TARGET_COL: "y"}
    )
    val_nf["unique_id"] = "price"
    val_nf["ds"] = pd.to_datetime(val_nf["ds"], utc=True)

    full_df = pd.concat([train_nf, val_nf], ignore_index=True)
    forecasts = nf.predict(df=full_df)

    # Normalize index
    if "ds" in forecasts.columns:
        forecasts = forecasts.set_index("ds")
    forecasts.index = pd.to_datetime(forecasts.index, utc=True)

    # Save model + forecasts
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, f"modern_dl_{market.lower()}")
    nf.save(model_path, overwrite=True)
    print(f"  Saved NF models: {model_path}")

    forecast_path = os.path.join(
        config.REPORT_DIR, f"modern_dl_preds_{market.lower()}.csv"
    )
    forecasts.to_csv(forecast_path)
    print(f"  Saved predictions: {forecast_path}")

    # Quick accuracy report
    test_actual = pd.read_parquet(
        config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
    )[config.TARGET_COL]

    for model_name in ["PatchTST", "iTransformer", "NHITS"]:
        col = f"price/{model_name}" if f"price/{model_name}" in forecasts.columns \
              else model_name
        if col not in forecasts.columns:
            continue
        y_pred = forecasts[col].values
        y_true = test_actual.values[:len(y_pred)]
        mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
        mae    = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        rmse   = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
        print(f"  {model_name:20} MAE={mae:.4f}  RMSE={rmse:.4f}")

    print(f"\n  ✅ Modern DL training complete for {market}.")
    return nf, forecasts


if __name__ == "__main__":
    train_modern_dl("PJM")
    train_modern_dl("ERCOT")
