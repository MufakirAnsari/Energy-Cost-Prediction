"""
step_10b_alpha_sweep.py — Conformal Alpha Sweep (Ablation A5)
=============================================================
Runs CQR at alpha ∈ {0.05, 0.10, 0.20} → 95%, 90%, 80% CI targets.
Demonstrates PICP vs. MPIW tradeoff for Figure F9.

Uses pre-trained LGBM quantile models. No retraining required.

Run:
    python step_10b_alpha_sweep.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config


def picp(y, lo, hi):
    mask = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    return np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])) * 100

def mpiw(lo, hi):
    mask = ~np.isnan(lo) & ~np.isnan(hi)
    return np.mean(hi[mask] - lo[mask])

def winkler(y, lo, hi, alpha):
    mask = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    width = hi[mask] - lo[mask]
    pen_l = (2/alpha) * np.maximum(lo[mask] - y[mask], 0)
    pen_u = (2/alpha) * np.maximum(y[mask] - hi[mask], 0)
    return np.mean(width + pen_l + pen_u)

def pinball(y, q, alpha):
    mask = ~np.isnan(y) & ~np.isnan(q)
    return np.mean(np.where(y[mask] >= q[mask],
                            alpha * (y[mask] - q[mask]),
                            (1-alpha) * (q[mask] - y[mask])))


# Map alpha → (lo_quantile, hi_quantile) model keys in config
ALPHA_MAP = {
    0.05: ("q025", "q975"),  # 95% CI → q2.5, q97.5 (approx with q05, q95)
    0.10: ("q05",  "q95"),   # 90% CI → exact
    0.20: ("q10",  "q90"),   # 80% CI → exact
}

# Map to available quantile model files
LGBM_Q_MAP = {
    0.05: ("lgbm_q05", "lgbm_q95"),  # best available for 95% CI target
    0.10: ("lgbm_q05", "lgbm_q95"),  # 90% CI with q05/q95
    0.20: ("lgbm_q10", "lgbm_q90"),  # 80% CI with q10/q90
}


def run_alpha_sweep(market="PJM"):
    print(f"\n{'='*65}\n  Conformal Alpha Sweep: {market}\n{'='*65}")

    cal_df  = pd.read_parquet(config.PJM_CAL_PATH  if market=="PJM" else config.ERCOT_CAL_PATH)
    test_df = pd.read_parquet(config.PJM_TEST_PATH if market=="PJM" else config.ERCOT_TEST_PATH)
    y_cal   = cal_df[config.TARGET_COL].values
    y_test  = test_df[config.TARGET_COL].values
    X_cal   = cal_df.drop(columns=[config.TARGET_COL])
    X_test  = test_df.drop(columns=[config.TARGET_COL])
    m       = market.lower()

    rows = []
    alpha_configs = [
        (0.05, "lgbm_q05", "lgbm_q95", "95% CI (α=0.05)"),
        (0.10, "lgbm_q05", "lgbm_q95", "90% CI (α=0.10)"),
        (0.20, "lgbm_q10", "lgbm_q90", "80% CI (α=0.20)"),
    ]

    for alpha, lo_key, hi_key, label in alpha_configs:
        lo_path = os.path.join(config.MODEL_DIR, f"{lo_key}_{m}.joblib")
        hi_path = os.path.join(config.MODEL_DIR, f"{hi_key}_{m}.joblib")
        if not os.path.exists(lo_path) or not os.path.exists(hi_path):
            print(f"  ⚠️  Skipping {label} — models not found")
            continue

        lo_model = joblib.load(lo_path)
        hi_model = joblib.load(hi_path)

        # Calibration set predictions
        q_lo_cal = lo_model.predict(X_cal)
        q_hi_cal = hi_model.predict(X_cal)

        # Nonconformity scores: E_i = max(q_lo - y, y - q_hi)
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
        n      = len(scores)
        # (1 - alpha) * (1 + 1/n) quantile of scores
        level  = np.ceil((1 - alpha) * (1 + 1/n)) / (1 + 1/n)
        level  = min(level, 1.0)
        correction = np.quantile(scores, level)

        # Test set predictions
        q_lo_test = lo_model.predict(X_test) - correction
        q_hi_test = hi_model.predict(X_test) + correction

        picp_val   = picp(y_test, q_lo_test, q_hi_test)
        mpiw_val   = mpiw(q_lo_test, q_hi_test)
        winkler_val = winkler(y_test, q_lo_test, q_hi_test, alpha)
        pb_lo = pinball(y_test, q_lo_test, alpha/2)
        pb_hi = pinball(y_test, q_hi_test, 1-alpha/2)

        row = {
            "Alpha":       alpha,
            "Nominal_CI":  f"{round((1-alpha)*100)}%",
            "Label":       label,
            "Correction":  round(correction, 4),
            "PICP_%":      round(picp_val, 2),
            "MPIW":        round(mpiw_val, 4),
            "Winkler":     round(winkler_val, 4),
            "Pinball_lo":  round(pb_lo, 4),
            "Pinball_hi":  round(pb_hi, 4),
        }
        rows.append(row)
        print(f"  {label:20} correction={correction:7.2f} | "
              f"PICP={picp_val:.2f}% | MPIW={mpiw_val:.2f} | Winkler={winkler_val:.2f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(config.REPORT_DIR, f"table_alpha_sweep_{m}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved: {out_path}")
    return df


if __name__ == "__main__":
    run_alpha_sweep("PJM")
    run_alpha_sweep("ERCOT")
