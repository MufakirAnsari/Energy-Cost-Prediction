"""
step_09_chronos_inference.py
============================
Zero-shot inference using Amazon Chronos-Bolt (small variant).
No fine-tuning — purely zero-shot foundation model baseline.

PERFORMANCE FIX: Uses batched h=24 (day-ahead) prediction instead of
rolling h=1 (which would require 17k+ individual model calls on CPU).
~710 PJM + ~731 ERCOT batched calls → completes in ~15-30 min on CPU.

Reference: Ansari et al. (2024). Chronos: Learning the Language of
           Time Series. arXiv:2403.07815.

Run:
    python step_09_chronos_inference.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    import torch
    from chronos import ChronosPipeline
    # ChronosBoltPipeline added in chronos-forecasting >= 1.4.0
    try:
        from chronos import ChronosBoltPipeline
        _HAS_BOLT = True
    except ImportError:
        _HAS_BOLT = False
except ImportError:
    raise ImportError("pip install --upgrade chronos-forecasting torch")


CONTEXT_LEN = 168   # 7-day context window (same as other models)
PRED_HORIZON = 24   # predict 24h at a time (day-ahead)
N_SAMPLES    = 20   # probabilistic samples for CI


def run_chronos_inference(market: str = "PJM") -> pd.DataFrame:
    """
    Runs Chronos-Bolt (small) in zero-shot mode on the test set.
    Uses batched day-ahead (h=24) prediction for efficiency.
    Returns DataFrame with point forecast and 90% CI columns.
    """
    print(f"\n{'='*65}")
    print(f"  Chronos-Bolt Zero-Shot: {market} | h={PRED_HORIZON} | ctx={CONTEXT_LEN}h")
    print(f"{'='*65}")

    tr_df  = pd.read_parquet(config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH)
    val_df = pd.read_parquet(config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH)
    te_df  = pd.read_parquet(config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH)

    # Full price history for context
    history = pd.concat([tr_df[config.TARGET_COL], val_df[config.TARGET_COL]])
    price_test = te_df[config.TARGET_COL]
    n_test = len(price_test)
    n_days = n_test // PRED_HORIZON   # number of 24h batches

    # Checkpoint: skip if already done
    ckpt_path = os.path.join(config.REPORT_DIR, f"chronos_preds_{market.lower()}.csv")
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    ckpt_tmp  = ckpt_path.replace(".csv", "_ckpt.csv")

    # Load partial checkpoint if exists
    done_days = 0
    existing  = []
    if os.path.exists(ckpt_tmp):
        try:
            ck = pd.read_csv(ckpt_tmp)
            done_days = len(ck) // PRED_HORIZON
            existing  = [ck]
            print(f"  ♻️  Resuming from checkpoint: {done_days}/{n_days} days done")
        except Exception:
            pass

    print(f"  Test: {n_test:,} hours | Batches: {n_days} | Remaining: {n_days-done_days}")

    print(f"  Loading Chronos model (small)...")
    t_load = time.time()
    _is_bolt = False
    if _HAS_BOLT:
        try:
            pipeline = ChronosBoltPipeline.from_pretrained(
                "amazon/chronos-bolt-small",
                device_map="cpu",
                dtype=torch.float32,
            )
            _is_bolt = True
            print(f"  Using: Chronos-Bolt-Small (deterministic quantile, zero-shot)")
        except Exception as e:
            print(f"  Chronos-Bolt failed ({e}), falling back to Chronos-T5-Small")
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cpu",
                dtype=torch.float32,
            )
            print(f"  Using: Chronos-T5-Small (sampling, zero-shot)")
    else:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            dtype=torch.float32,
        )
        print(f"  Using: Chronos-T5-Small (sampling, zero-shot)")
    print(f"  Model loaded in {time.time()-t_load:.1f}s")

    def chronos_predict(ctx_tensor):
        """Unified predict: returns (median, q10, q90) as numpy arrays [PRED_HORIZON]."""
        with torch.no_grad():
            if _is_bolt:
                # Bolt: quantile regression, trained on [0.1..0.9] only
                # Returns shape: [batch, horizon, n_quantiles]
                quantiles, mean = pipeline.predict_quantiles(
                    ctx_tensor,
                    prediction_length=PRED_HORIZON,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                q10 = quantiles[0, :, 0].numpy()   # [horizon] — 10th pct
                med = quantiles[0, :, 1].numpy()   # [horizon] — median
                q90 = quantiles[0, :, 2].numpy()   # [horizon] — 90th pct
            else:
                # T5: sample-based — returns [batch, n_samples, horizon]
                fc   = pipeline.predict(ctx_tensor, PRED_HORIZON, num_samples=N_SAMPLES)
                samp = fc[0].numpy()               # [n_samples, horizon]
                q10  = np.percentile(samp, 10, axis=0)
                med  = np.median(samp,         axis=0)
                q90  = np.percentile(samp, 90, axis=0)
        return med, q10, q90


    # Rolling day-ahead batched inference
    context_buffer = history.copy()
    all_rows = existing.copy()

    t0 = time.time()
    for day_idx in range(done_days, n_days):
        start = day_idx * PRED_HORIZON
        end   = start + PRED_HORIZON

        # Context: last CONTEXT_LEN observations before this day
        ctx = context_buffer.iloc[-CONTEXT_LEN:].values.astype(np.float32)
        ctx_tensor = torch.tensor(ctx).unsqueeze(0)  # [1, context_length]

        med, q10, q90 = chronos_predict(ctx_tensor)  # each: [PRED_HORIZON]

        for h in range(PRED_HORIZON):
            idx = start + h
            if idx >= n_test:
                break
            all_rows.append({
                "ds":              price_test.index[idx],
                "actual":          price_test.iloc[idx],
                "chronos_point":   float(med[h]),
                "chronos_lower80": float(q10[h]),   # 80% CI (q10-q90 is Bolt's range)
                "chronos_upper80": float(q90[h]),
            })

        # Extend context with true observations for next batch
        context_buffer = pd.concat([
            context_buffer,
            price_test.iloc[start:end]
        ])

        # Progress + checkpoint every 50 batches
        if (day_idx + 1) % 50 == 0 or day_idx == n_days - 1:
            elapsed = time.time() - t0
            rate = (day_idx - done_days + 1) / elapsed  # batches/sec
            remaining = (n_days - day_idx - 1) / rate / 60 if rate > 0 else 0
            print(f"    Day {day_idx+1:>4}/{n_days} | {elapsed/60:.1f}min | "
                  f"~{remaining:.1f}min left")
            # Save checkpoint
            pd.DataFrame(all_rows).to_csv(ckpt_tmp, index=False)

    results = pd.DataFrame(all_rows)
    if "ds" in results.columns:
        results = results.set_index("ds")

    # Metrics
    mask = ~results["actual"].isna() & ~results["chronos_point"].isna()
    mae  = np.mean(np.abs(results.loc[mask,"actual"] - results.loc[mask,"chronos_point"]))
    rmse = np.sqrt(np.mean((results.loc[mask,"actual"] - results.loc[mask,"chronos_point"])**2))
    picp = np.mean(
        (results.loc[mask,"actual"] >= results.loc[mask,"chronos_lower80"]) &
        (results.loc[mask,"actual"] <= results.loc[mask,"chronos_upper80"])
    ) * 100
    mpiw = np.mean(results.loc[mask,"chronos_upper80"] - results.loc[mask,"chronos_lower80"])

    print(f"\n  Chronos-Bolt Results ({market}):")
    print(f"    MAE:  {mae:.4f} $/MWh")
    print(f"    RMSE: {rmse:.4f} $/MWh")
    print(f"    PICP: {picp:.2f}% (80% CI, target ≥80%)")
    print(f"    MPIW: {mpiw:.4f} $/MWh")

    results.to_csv(ckpt_path)
    # Remove checkpoint temp file on success
    if os.path.exists(ckpt_tmp):
        os.remove(ckpt_tmp)

    print(f"  ✅ Saved: {ckpt_path}")
    return results


if __name__ == "__main__":
    run_chronos_inference("PJM")
    run_chronos_inference("ERCOT")
