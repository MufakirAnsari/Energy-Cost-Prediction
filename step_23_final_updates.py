"""
step_23_final_updates.py
========================
1. Updates DM tests for new models (Chronos-Base v2, rolling LGBM/XGB)
2. Generates graphical abstract for Applied Energy submission

Run:
    python step_23_final_updates.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. DM TESTS FOR NEW MODELS
# ═══════════════════════════════════════════════════════════════

def dm_test_hln(y, pred_a, pred_b, h=1):
    """Harvey-Leybourne-Newbold small-sample DM test."""
    from scipy import stats
    try:
        e1 = np.abs(y - pred_a)
        e2 = np.abs(y - pred_b)
        d  = e1 - e2
        n  = len(d)
        d_mean = np.mean(d)
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
        var_d = np.var(d, ddof=1)
        for k in range(1, max_lag + 1):
            w   = 1 - k / (max_lag + 1)
            cov = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
            var_d += 2 * w * cov
        var_d = max(var_d, 1e-10)
        dm_stat = d_mean / np.sqrt(var_d / n)
        dm_corr = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n) * dm_stat
        p_val = 2 * stats.t.sf(np.abs(dm_corr), df=n-1)
        return float(dm_corr), float(p_val)
    except Exception:
        return np.nan, np.nan


def update_dm_tests():
    print("\n[1] Updating DM Tests for new models...")

    for market in ["pjm", "ercot"]:
        # Load test actuals
        test_path = config.PJM_TEST_PATH if market == "pjm" else config.ERCOT_TEST_PATH
        te_df = pd.read_parquet(test_path)
        y_true = te_df[config.TARGET_COL].values
        te_idx = te_df.index
        if te_idx.tz is None:
            te_idx = te_idx.tz_localize("UTC")

        # Load reference model predictions (best static = LightGBM for ERCOT, XGBoost for PJM)
        ens_path = os.path.join(config.REPORT_DIR, f"ensemble_preds_{market}.csv")
        ens_df = pd.read_csv(ens_path, index_col=0, parse_dates=True)
        ens_df.index = pd.to_datetime(ens_df.index, utc=True)

        # Reference: use the best static model
        ref_model = "xgboost" if market == "pjm" else "lgbm"
        if ref_model not in ens_df.columns:
            ref_model = "lgbm"
        ref_preds = ens_df[ref_model].reindex(te_idx).values

        # Load existing DM table
        dm_path = os.path.join(config.REPORT_DIR, f"table_dm_tests_{market}.csv")
        dm_df = pd.read_csv(dm_path)

        new_rows = []

        # --- Chronos-Base v2 (univariate) ---
        c2_path = os.path.join(config.REPORT_DIR, f"chronos2_preds_{market}.csv")
        if os.path.exists(c2_path):
            c2_df = pd.read_csv(c2_path, index_col=0, parse_dates=True)
            c2_df.index = pd.to_datetime(c2_df.index, utc=True)

            # Univariate
            c2_uni = c2_df["c2_uni_point"].reindex(te_idx).values
            valid = ~np.isnan(ref_preds) & ~np.isnan(c2_uni) & ~np.isnan(y_true)
            if valid.sum() > 100:
                dm_s, p_v = dm_test_hln(y_true[valid], ref_preds[valid], c2_uni[valid])
                mae = np.mean(np.abs(y_true[valid] - c2_uni[valid]))
                new_rows.append({
                    "Model": "Chronos-Bolt-Base",
                    "MAE": round(mae, 4),
                    "DM_stat": round(dm_s, 4),
                    "p_value": round(p_v, 6),
                    "significant": p_v < 0.05,
                    "is_reference": False,
                    "MAE_vs_ref_%": 0,
                })

            # Covariate-enhanced
            if "c2_cov_point" in c2_df.columns:
                c2_cov = c2_df["c2_cov_point"].reindex(te_idx).values
                valid2 = ~np.isnan(ref_preds) & ~np.isnan(c2_cov) & ~np.isnan(y_true)
                if valid2.sum() > 100:
                    dm_s2, p_v2 = dm_test_hln(y_true[valid2], ref_preds[valid2], c2_cov[valid2])
                    mae2 = np.mean(np.abs(y_true[valid2] - c2_cov[valid2]))
                    new_rows.append({
                        "Model": "Chronos-Base+Cov",
                        "MAE": round(mae2, 4),
                        "DM_stat": round(dm_s2, 4),
                        "p_value": round(p_v2, 6),
                        "significant": p_v2 < 0.05,
                        "is_reference": False,
                        "MAE_vs_ref_%": 0,
                    })

        # Remove duplicates and append
        new_names = [r["Model"] for r in new_rows]
        dm_df = dm_df[~dm_df["Model"].isin(new_names)]
        dm_df = pd.concat([dm_df, pd.DataFrame(new_rows)], ignore_index=True)

        # Recalculate MAE_vs_ref_%
        ref_row = dm_df[dm_df["is_reference"] == True]
        if not ref_row.empty:
            ref_mae = ref_row.iloc[0]["MAE"]
            dm_df["MAE_vs_ref_%"] = ((dm_df["MAE"] - ref_mae) / ref_mae * 100).round(1)

        dm_df.to_csv(dm_path, index=False)
        print(f"  ✅ Updated: table_dm_tests_{market}.csv ({len(dm_df)} models)")
        print(dm_df[["Model", "MAE", "DM_stat", "p_value", "significant"]].to_string(index=False))
        print()


# ═══════════════════════════════════════════════════════════════
# 2. GRAPHICAL ABSTRACT
# ═══════════════════════════════════════════════════════════════

def create_graphical_abstract():
    print("\n[2] Creating Graphical Abstract...")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # ── Title Banner ──
    ax.add_patch(FancyBboxPatch((0.3, 7.8), 15.4, 1.0, boxstyle="round,pad=0.15",
                                facecolor="#1565C0", edgecolor="none", alpha=0.95))
    ax.text(8, 8.3, "Trees Still Beat Transformers for Day-Ahead Electricity Price Forecasting",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white",
            family="sans-serif")

    # ── Pipeline Boxes ──
    box_style = "round,pad=0.2"
    box_h = 1.6

    # Box 1: Data
    ax.add_patch(FancyBboxPatch((0.3, 4.8), 3.0, box_h, boxstyle=box_style,
                                facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5))
    ax.text(1.8, 6.05, "DATA", ha="center", va="center", fontsize=11, fontweight="bold", color="#1565C0")
    ax.text(1.8, 5.6, "PJM + ERCOT", ha="center", va="center", fontsize=9, color="#333")
    ax.text(1.8, 5.3, "2019\u20132025 (60K hrs)", ha="center", va="center", fontsize=8, color="#666")
    ax.text(1.8, 5.0, "50 features", ha="center", va="center", fontsize=8, color="#666")

    # Box 2: Models
    ax.add_patch(FancyBboxPatch((4.0, 4.8), 3.5, box_h, boxstyle=box_style,
                                facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5))
    ax.text(5.75, 6.05, "18 CONFIGURATIONS", ha="center", va="center", fontsize=11, fontweight="bold", color="#E65100")
    ax.text(5.75, 5.6, "Trees \u00b7 DL \u00b7 Transformers", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(5.75, 5.3, "Chronos-Bolt v1 & v2", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(5.75, 5.0, "+ Rolling Retraining", ha="center", va="center", fontsize=8.5, color="#333")

    # Box 3: Evaluation
    ax.add_patch(FancyBboxPatch((8.2, 4.8), 3.5, box_h, boxstyle=box_style,
                                facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5))
    ax.text(9.95, 6.05, "EVALUATION", ha="center", va="center", fontsize=11, fontweight="bold", color="#2E7D32")
    ax.text(9.95, 5.6, "Point \u00b7 Probabilistic", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(9.95, 5.3, "Economic (PnL, Sharpe)", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(9.95, 5.0, "DM Tests + BH Correction", ha="center", va="center", fontsize=8.5, color="#333")

    # Box 4: UQ
    ax.add_patch(FancyBboxPatch((12.4, 4.8), 3.3, box_h, boxstyle=box_style,
                                facecolor="#FCE4EC", edgecolor="#C62828", linewidth=1.5))
    ax.text(14.05, 6.05, "UNCERTAINTY", ha="center", va="center", fontsize=11, fontweight="bold", color="#C62828")
    ax.text(14.05, 5.6, "CQR \u00b7 QRF \u00b7 MC Dropout", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(14.05, 5.3, "Conformal fails under", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(14.05, 5.0, "regime shift (PICP 82.6%)", ha="center", va="center", fontsize=8.5, color="#333")

    # Arrows between boxes
    for x_start, x_end in [(3.3, 4.0), (7.5, 8.2), (11.7, 12.4)]:
        ax.annotate("", xy=(x_end, 5.6), xytext=(x_start, 5.6),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    # ── Key Findings (Bottom Section) ──
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 15.4, 4.0, boxstyle="round,pad=0.2",
                                facecolor="white", edgecolor="#BDBDBD", linewidth=1))

    ax.text(8, 4.0, "KEY FINDINGS", ha="center", va="center", fontsize=13,
            fontweight="bold", color="#333")

    # Finding 1
    ax.add_patch(FancyBboxPatch((0.6, 2.3), 3.6, 1.4, boxstyle="round,pad=0.15",
                                facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.2))
    ax.text(2.4, 3.35, "LightGBM Dominates", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2E7D32")
    ax.text(2.4, 2.95, "MAE: 3.05 (PJM) / 2.58 (ERCOT)", ha="center", va="center",
            fontsize=8, color="#333")
    ax.text(2.4, 2.65, "Beats all DL by 40\u2013130%", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1B5E20")

    # Finding 2
    ax.add_patch(FancyBboxPatch((4.5, 2.3), 3.6, 1.4, boxstyle="round,pad=0.15",
                                facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=1.2))
    ax.text(6.3, 3.35, "Covariates > Architecture", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#E65100")
    ax.text(6.3, 2.95, "Chronos-2 + covariates: 3.48", ha="center", va="center",
            fontsize=8, color="#333")
    ax.text(6.3, 2.65, "49% improvement over univar.", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#BF360C")

    # Finding 3
    ax.add_patch(FancyBboxPatch((8.4, 2.3), 3.6, 1.4, boxstyle="round,pad=0.15",
                                facecolor="#E3F2FD", edgecolor="#1E88E5", linewidth=1.2))
    ax.text(10.2, 3.35, "Rolling Retrain: +24%", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1565C0")
    ax.text(10.2, 2.95, "Monthly retraining essential", ha="center", va="center",
            fontsize=8, color="#333")
    ax.text(10.2, 2.65, "4.03 \u2192 3.05 MAE (PJM)", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#0D47A1")

    # Finding 4
    ax.add_patch(FancyBboxPatch((12.3, 2.3), 3.3, 1.4, boxstyle="round,pad=0.15",
                                facecolor="#FCE4EC", edgecolor="#E53935", linewidth=1.2))
    ax.text(13.95, 3.35, "CQR Fails Under Shift", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#C62828")
    ax.text(13.95, 2.95, "QRF: 91.2% PICP (on target)", ha="center", va="center",
            fontsize=8, color="#333")
    ax.text(13.95, 2.65, "CQR: 82.6% (misses by 7pp)", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#B71C1C")

    # Bottom bar with headline stat
    ax.add_patch(FancyBboxPatch((0.3, 0.5), 15.4, 1.4, boxstyle="round,pad=0.1",
                                facecolor="#1565C0", edgecolor="none", alpha=0.9))
    ax.text(8, 1.45, "Rolling LightGBM achieves MAE = 3.05 \$/MWh (PJM) and 2.58 \$/MWh (ERCOT)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(8, 0.95, "outperforming 17 alternatives including Chronos-2, PatchTST, iTransformer, and TFT across 2 US markets (2024\u20132025)",
            ha="center", va="center", fontsize=9, color="#E3F2FD")

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"Graphical_Abstract.{ext}"),
                   dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  \u2705 Saved: Graphical_Abstract")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  FINAL UPDATES: DM TESTS + GRAPHICAL ABSTRACT")
    print("=" * 65)

    update_dm_tests()
    create_graphical_abstract()

    print("\n" + "=" * 65)
    print("  ALL DONE!")
    print("=" * 65)
