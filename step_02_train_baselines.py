"""
step_02_train_baselines.py
==========================
Trains three classical statistical baselines using statsforecast:
  1. Seasonal Naïve (SNaive) — predicts last week's same-hour price
  2. AutoARIMA (hourly, season_length=24) — rolling window walk-forward
  3. MSTL (Multi-Seasonal Trend decomposition) — captures 24h + 168h seasonality

All baselines use HOURLY data (not resampled) and walk-forward validation
to match real-world forecasting conditions.

CHECKPOINT / RESUME:
  - After every batch, partial results are saved to <out_path>.ckpt.pkl
  - On restart, the script detects the checkpoint and resumes from the
    last completed batch automatically.
  - Completed splits (final CSV already saved) are skipped entirely.

Run:
    python step_02_train_baselines.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA,
        MSTL,
        SeasonalNaive,
        AutoETS,
    )
except ImportError:
    raise ImportError("pip install statsforecast")


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(out_path: str) -> str:
    return out_path + ".ckpt.pkl"


def save_checkpoint(out_path: str, state: dict):
    """Save batch state to checkpoint file."""
    with open(ckpt_path(out_path), "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(out_path: str) -> dict | None:
    """Load checkpoint if it exists, else return None."""
    cp = ckpt_path(out_path)
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            state = pickle.load(f)
        print(f"  📂 Checkpoint found — resuming from batch "
              f"{state['batch_num_done'] + 1} / {state['n_windows']} "
              f"({state['offset']} / {state['n_test']} steps done)")
        return state
    return None


def delete_checkpoint(out_path: str):
    cp = ckpt_path(out_path)
    if os.path.exists(cp):
        os.remove(cp)


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD EVALUATION WITH CHECKPOINT / RESUME
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_predict(
    train_series: pd.Series,
    test_series: pd.Series,
    models: list,
    out_path: str,
    retrain_freq_hours: int = 168,
) -> dict:
    """
    Walk-forward (expanding window) prediction in WEEKLY BATCHES.
    Fits once per week, predicts h=retrain_freq_hours steps ahead,
    then appends true values and moves to the next window.

    Checkpoints after every batch — safe to interrupt and resume.
    """
    n_test    = len(test_series)
    n_windows = (n_test + retrain_freq_hours - 1) // retrain_freq_hours
    model_names = [m.__class__.__name__ for m in models]

    print(f"    Walk-forward: {n_test:,} steps in {n_windows} weekly batches")

    # ── Try to resume from checkpoint ────────────────────────────
    state = load_checkpoint(out_path)
    if state is not None:
        history    = state["history"]
        preds      = state["preds"]
        timestamps = state["timestamps"]
        offset     = state["offset"]
        batch_num  = state["batch_num_done"]
        t_elapsed  = state["elapsed_seconds"]
    else:
        history    = train_series.copy()
        preds      = {name: [] for name in model_names}
        timestamps = []
        offset     = 0
        batch_num  = 0
        t_elapsed  = 0.0

    t_start = time.time()

    while offset < n_test:
        batch_num += 1
        h = min(retrain_freq_hours, n_test - offset)

        # Fit on expanding history
        sf = StatsForecast(
            models=models,
            freq="h",
            n_jobs=1,
        )
        sf_df = pd.DataFrame({
            "unique_id": "price",
            "ds": history.index,
            "y": history.values,
        })
        sf.fit(sf_df)

        # Predict h steps ahead
        fc = sf.predict(h=h)

        # Extract predictions for each model
        for name in model_names:
            try:
                vals = fc.loc[fc["unique_id"] == "price", name].values[:h]
            except (KeyError, IndexError):
                vals = np.full(h, np.nan)
            preds[name].extend(vals.tolist())

        batch_ts = test_series.index[offset:offset + h].tolist()
        timestamps.extend(batch_ts)

        # Add true observations to history (expanding window)
        history = pd.concat([history, test_series.iloc[offset:offset + h]])
        offset += h

        wall_elapsed = t_elapsed + (time.time() - t_start)
        print(f"      Batch {batch_num:>3}/{n_windows}  "
              f"(step {offset:>6}/{n_test}, {wall_elapsed/60:.1f} min)")

        # ── Save checkpoint after every batch ────────────────────
        save_checkpoint(out_path, {
            "history":        history,
            "preds":          preds,
            "timestamps":     timestamps,
            "offset":         offset,
            "batch_num_done": batch_num,
            "n_windows":      n_windows,
            "n_test":         n_test,
            "elapsed_seconds": wall_elapsed,
        })

    # Done — clean up checkpoint
    delete_checkpoint(out_path)

    return {
        name: pd.Series(vals[:n_test], index=timestamps[:n_test])
        for name, vals in preds.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_statistical_baselines(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  STATISTICAL BASELINES: {market}")
    print(f"{'='*65}")

    train_path = config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH
    val_path   = config.PJM_VAL_PATH   if market == "PJM" else config.ERCOT_VAL_PATH
    test_path  = config.PJM_TEST_PATH  if market == "PJM" else config.ERCOT_TEST_PATH

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    price_train = train_df[config.TARGET_COL]
    price_val   = val_df[config.TARGET_COL]
    price_test  = test_df[config.TARGET_COL]

    models = [
        SeasonalNaive(season_length=168),
        AutoARIMA(
            season_length=24,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            d=None, D=1,
            stepwise=True, approximation=True,
            nmodels=20,
        ),
        MSTL(
            season_length=[24, 168],
            trend_forecaster=AutoETS(model="ZZN"),
        ),
    ]

    os.makedirs(config.MODEL_DIR,  exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    for split_name, price_history, price_eval in [
        ("val",  price_train,                      price_val),
        ("test", pd.concat([price_train, price_val]), price_test),
    ]:
        out_path = os.path.join(
            config.REPORT_DIR,
            f"baseline_preds_{market.lower()}_{split_name}.csv"
        )

        # ── Skip if final CSV already exists (and no checkpoint) ──
        if os.path.exists(out_path) and not os.path.exists(ckpt_path(out_path)):
            print(f"\n  --- {market} {split_name.upper()} SET ---")
            print(f"  ✅ Already complete — skipping. ({out_path})")
            continue

        print(f"\n  --- {market} {split_name.upper()} SET ---")
        preds = walk_forward_predict(
            train_series=price_history,
            test_series=price_eval,
            models=models,
            out_path=out_path,
            retrain_freq_hours=168,
        )

        # Save final CSV
        pred_df = pd.DataFrame(preds, index=price_eval.index)
        pred_df["actual"] = price_eval.values
        pred_df.to_csv(out_path)
        print(f"  Saved: {out_path}")

        # Print accuracy
        for model_name in preds:
            y_true = price_eval.values
            y_pred = preds[model_name].values
            mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
            mae    = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            rmse   = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
            print(f"    {model_name:20} MAE={mae:.3f}  RMSE={rmse:.3f}")

    print(f"\n  ✅ Statistical baselines complete for {market}.")


if __name__ == "__main__":
    train_statistical_baselines("PJM")
    train_statistical_baselines("ERCOT")
