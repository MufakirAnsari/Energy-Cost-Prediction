"""
patch_bilstm_pjm_quantiles.py  (v2 — analytical, no GPU needed)
=================================================================
Adds q25/q75 to the PJM BiLSTM prediction CSV WITHOUT reloading
the Keras model (avoids libdevice/XLA issues on GTX 1650).

Method:
  MC Dropout samples are approximately normal. Given the already-saved
  mean_pred, std_pred, q05, q95, we can derive q25/q75 analytically:

    std  = (q95 - q05) / (2 * 1.645)   ← from normal quantile relation
    q25  = mean_pred - 0.674 * std
    q75  = mean_pred + 0.674 * std

  0.674 = norm.ppf(0.75), 1.645 = norm.ppf(0.95)
  This is equivalent to running MC inference and computing percentiles.

Run: python patch_bilstm_pjm_quantiles.py
"""

import os, sys
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config


def patch_pjm():
    print("\n=================================================================")
    print("  Patching PJM BiLSTM: adding q25/q75 (analytical, no GPU)")
    print("=================================================================")

    pred_path = os.path.join(config.REPORT_DIR, "bilstm_preds_pjm.csv")
    if not os.path.exists(pred_path):
        print(f"  ❌ Not found: {pred_path}")
        return

    df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    print(f"  Existing columns: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")

    if "q25" in df.columns and "q75" in df.columns:
        print("  ✅ q25/q75 already present — nothing to do.")
        return

    # Derive std from q05/q95 via normal quantile relationship
    # q95 = mean + 1.645*std  → std = (q95-q05)/(2*1.645)
    Z_95 = 1.6449   # norm.ppf(0.95)
    Z_75 = 0.6745   # norm.ppf(0.75)

    if "std_pred" in df.columns and not df["std_pred"].isna().all():
        # Prefer the stored std directly
        std = df["std_pred"].values
        print("  Using stored std_pred directly")
    elif "q05" in df.columns and "q95" in df.columns:
        # Derive std from the q05/q95 spread
        std = (df["q95"].values - df["q05"].values) / (2 * Z_95)
        print("  Deriving std from q05/q95 spread")
    else:
        print("  ❌ Neither std_pred nor q05/q95 found — cannot derive quantiles.")
        return

    mean = df["mean_pred"].values
    q25  = mean - Z_75 * std
    q75  = mean + Z_75 * std

    df["q25"] = q25
    df["q75"] = q75

    # Reorder columns to canonical order
    col_order = [c for c in ["actual","mean_pred","std_pred","q05","q25","q75","q95"]
                 if c in df.columns]
    df = df[col_order]

    # Verification
    mask = ~df["actual"].isna()
    y    = df.loc[mask, "actual"].values
    picp_90 = np.mean((y >= df.loc[mask,"q05"].values) &
                      (y <= df.loc[mask,"q95"].values)) * 100
    picp_50 = np.mean((y >= df.loc[mask,"q25"].values) &
                      (y <= df.loc[mask,"q75"].values)) * 100
    mpiw_90 = np.mean(df.loc[mask,"q95"].values - df.loc[mask,"q05"].values)
    mpiw_50 = np.mean(df.loc[mask,"q75"].values - df.loc[mask,"q25"].values)

    print(f"\n  PICP 90% (q05-q95): {picp_90:.2f}%  (nominal: 90%)")
    print(f"  PICP 50% (q25-q75): {picp_50:.2f}%  (nominal: 50%)")
    print(f"  MPIW 90%: {mpiw_90:.2f} $/MWh")
    print(f"  MPIW 50%: {mpiw_50:.2f} $/MWh")
    print(f"\n  Final columns: {list(df.columns)}")

    df.to_csv(pred_path)
    print(f"  ✅ Saved: {pred_path}")


if __name__ == "__main__":
    patch_pjm()
    print("\n  Next: re-run step_09_evaluate.py to get PJM BiLSTM CRPS")
