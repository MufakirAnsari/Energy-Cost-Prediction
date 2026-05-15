"""
step_07_chronos_inference.py  [DEPRECATED — use step_09_chronos_inference.py]
=================================================================================
This is a SUPERSEDED stub. The active Chronos inference script is:
    step_09_chronos_inference.py

Do not run this file directly. It exists for historical reference only.
=================================================================================

ORIGINAL DOCSTRING (archived):
Zero-shot inference using Amazon Chronos-Bolt (small variant).
No fine-tuning is performed — this is purely zero-shot evaluation.

Purpose: Establishes a "pre-trained foundation model" baseline.
If Chronos-Bolt matches domain-specific models without any training,
that is a significant finding. If it underperforms, that validates
the value of domain-specific training.

Reference: Ansari et al. (2024). Chronos: Learning the Language of
           Time Series. arXiv:2403.07815.

Run:
    python step_07_chronos_inference.py
    (No GPU required — runs on CPU for small model)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    import torch
    from chronos import ChronosPipeline
except ImportError:
    raise ImportError(
        "pip install chronos-forecasting\n"
        "Also requires: pip install torch"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ZERO-SHOT INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_chronos_inference(
    market: str = "PJM",
    context_length: int = 168,    # 7-day context window (same as other models)
    n_samples: int = 20,          # Probabilistic samples for uncertainty
) -> pd.DataFrame:
    """
    Runs Chronos-Bolt (small) in zero-shot mode on the test set.
    Returns DataFrame with point forecast and 90% CI columns.
    """
    print(f"\n{'='*65}")
    print(f"  Chronos-Bolt Zero-Shot Inference: {market}")
    print(f"  Context: {context_length}h | Samples: {n_samples}")
    print(f"{'='*65}")

    # Load test data (and preceding context from val/train)
    tr_df  = pd.read_parquet(config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH)
    val_df = pd.read_parquet(config.PJM_VAL_PATH   if market == "PJM" else config.ERCOT_VAL_PATH)
    te_df  = pd.read_parquet(config.PJM_TEST_PATH  if market == "PJM" else config.ERCOT_TEST_PATH)

    # Full history up to test (for context)
    history = pd.concat([
        tr_df[config.TARGET_COL],
        val_df[config.TARGET_COL],
    ])

    price_test = te_df[config.TARGET_COL]
    n_test = len(price_test)

    print(f"  Loading Chronos-Bolt (small)...")
    t0 = time.time()
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",   # ~200M parameters
        device_map="cpu",              # CPU-only for reliability
        torch_dtype=torch.float32,
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Rolling inference: predict 1 step at a time using preceding context
    point_preds = []
    lower_90    = []
    upper_90    = []
    timestamps  = []

    print(f"\n  Running rolling 1-step inference ({n_test:,} steps)...")
    t0 = time.time()

    # Build an expanding context buffer
    context_buffer = history.copy()

    for i in range(n_test):
        # Use last `context_length` observations as context
        ctx = context_buffer.iloc[-context_length:].values.astype(np.float32)
        ctx_tensor = torch.tensor(ctx).unsqueeze(0)  # [1, context_length]

        with torch.no_grad():
            fc = pipeline.predict(
                context=ctx_tensor,
                prediction_length=1,
                num_samples=n_samples,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
        # fc shape: [1, n_samples, 1]
        samples = fc[0, :, 0].numpy()   # [n_samples]

        point_preds.append(np.median(samples))
        lower_90.append(np.percentile(samples, 5))    # 5th percentile → 90% CI lower
        upper_90.append(np.percentile(samples, 95))   # 95th percentile → 90% CI upper
        timestamps.append(price_test.index[i])

        # Update context with true observation (expanding window)
        context_buffer = pd.concat([
            context_buffer,
            pd.Series([price_test.iloc[i]], index=[price_test.index[i]])
        ])

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_test - i - 1) / rate / 60
            print(f"    Step {i+1:>6}/{n_test} | {elapsed/60:.1f}min elapsed | "
                  f"~{remaining:.1f}min remaining")

    elapsed = time.time() - t0
    print(f"\n  Inference complete: {elapsed/60:.1f} min total")

    # Build result DataFrame
    results = pd.DataFrame({
        "actual":               price_test.values[:len(point_preds)],
        "chronos_point":        point_preds,
        "chronos_lower_90":     lower_90,
        "chronos_upper_90":     upper_90,
    }, index=timestamps)

    # Point accuracy
    mask = ~results["actual"].isna() & ~results["chronos_point"].isna()
    mae  = np.mean(np.abs(results.loc[mask, "actual"] - results.loc[mask, "chronos_point"]))
    rmse = np.sqrt(np.mean((results.loc[mask, "actual"] - results.loc[mask, "chronos_point"])**2))
    picp = np.mean(
        (results.loc[mask, "actual"] >= results.loc[mask, "chronos_lower_90"]) &
        (results.loc[mask, "actual"] <= results.loc[mask, "chronos_upper_90"])
    ) * 100
    mpiw = np.mean(results.loc[mask, "chronos_upper_90"] - results.loc[mask, "chronos_lower_90"])

    print(f"\n  Chronos-Bolt Results ({market}):")
    print(f"    MAE:  {mae:.4f} $/MWh")
    print(f"    RMSE: {rmse:.4f} $/MWh")
    print(f"    PICP: {picp:.2f}% (target: ≥90%)")
    print(f"    MPIW: {mpiw:.4f} $/MWh")

    # Save
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out_path = os.path.join(config.REPORT_DIR, f"chronos_preds_{market.lower()}.csv")
    results.to_csv(out_path)
    print(f"\n  ✅ Saved: {out_path}")

    return results


if __name__ == "__main__":
    run_chronos_inference("PJM")
    run_chronos_inference("ERCOT")
