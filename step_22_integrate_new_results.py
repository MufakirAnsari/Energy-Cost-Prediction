"""
step_22_integrate_new_results.py
=================================
Integrates Rolling Window + Chronos-2 results into the publication tables
and generates new comparison figures.

Run:
    python step_22_integrate_new_results.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
})

def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ═══════════════════════════════════════════════════════════════
# 1. Chronos-2 Comparison Figure
# ═══════════════════════════════════════════════════════════════
def generate_chronos2_figure():
    print("\n[1] Generating Chronos-2 Comparison Figure...")

    # Gather all model MAEs for comparison
    data = {}
    for market in ["pjm", "ercot"]:
        acc_path = os.path.join(config.REPORT_DIR, f"table_point_accuracy_{market}.csv")
        acc_df = pd.read_csv(acc_path)

        # Get Chronos-Bolt v1 MAE
        bolt_row = acc_df[acc_df["Model"] == "Chronos-Bolt"]
        bolt_mae = bolt_row.iloc[0]["MAE"] if not bolt_row.empty else np.nan

        # Get LightGBM MAE (static)
        lgbm_row = acc_df[acc_df["Model"] == "LightGBM"]
        lgbm_mae = lgbm_row.iloc[0]["MAE"] if not lgbm_row.empty else np.nan

        # Get Chronos-2 MAEs
        c2_path = os.path.join(config.REPORT_DIR, f"chronos2_preds_{market}.csv")
        c2_df = pd.read_csv(c2_path, index_col=0)
        mask = ~c2_df["actual"].isna() & ~c2_df["c2_uni_point"].isna()
        c2_uni_mae = np.mean(np.abs(c2_df.loc[mask, "actual"] - c2_df.loc[mask, "c2_uni_point"]))

        if "c2_cov_point" in c2_df.columns:
            mask2 = ~c2_df["actual"].isna() & ~c2_df["c2_cov_point"].isna()
            c2_cov_mae = np.mean(np.abs(c2_df.loc[mask2, "actual"] - c2_df.loc[mask2, "c2_cov_point"]))
        else:
            c2_cov_mae = np.nan

        # Get rolling LGBM MAE
        roll_path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{market}.csv")
        if os.path.exists(roll_path):
            roll_df = pd.read_csv(roll_path)
            roll_lgbm_mae = roll_df["Rolling_LGBM_MAE"].mean()
        else:
            roll_lgbm_mae = np.nan

        data[market] = {
            "Chronos-Bolt v1\n(univariate)": bolt_mae,
            "Chronos-Bolt-Base v2\n(univariate)": c2_uni_mae,
            "Chronos-Bolt-Base v2\n(+covariates)": c2_cov_mae,
            "LightGBM\n(static)": lgbm_mae,
            "LightGBM\n(rolling)": roll_lgbm_mae,
        }

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#E53935", "#FF7F0E", "#FFC107", "#2196F3", "#4CAF50"]

    for ax, market in zip(axes, ["pjm", "ercot"]):
        d = data[market]
        models = list(d.keys())
        maes = [d[m] for m in models]

        bars = ax.barh(range(len(models)), maes, color=colors,
                      edgecolor="black", linewidth=0.5, height=0.6)

        # Add value labels
        for i, (bar, mae) in enumerate(zip(bars, maes)):
            if not np.isnan(mae):
                ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                       f"{mae:.2f}", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("MAE ($/MWh) — lower is better")
        ax.set_title(f"{market.upper()}", fontweight="bold", fontsize=13)
        ax.invert_yaxis()

        # Highlight best
        best_idx = np.nanargmin(maes)
        bars[best_idx].set_edgecolor("#1B5E20")
        bars[best_idx].set_linewidth(2.5)

    fig.suptitle("Foundation Model vs Tree-Based: Full Comparison\n"
                 "Chronos-Bolt-Base v2 with covariates closes the gap but rolling LightGBM still wins",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "Fig7_Chronos2_Comparison")


# ═══════════════════════════════════════════════════════════════
# 2. Updated Accuracy Table (with all new models)
# ═══════════════════════════════════════════════════════════════
def update_accuracy_tables():
    print("\n[2] Updating accuracy tables with new models...")

    for market in ["pjm", "ercot"]:
        # Load existing
        acc_path = os.path.join(config.REPORT_DIR, f"table_point_accuracy_{market}.csv")
        df = pd.read_csv(acc_path)

        # Chronos-2 results
        c2_path = os.path.join(config.REPORT_DIR, f"chronos2_preds_{market}.csv")
        c2_df = pd.read_csv(c2_path, index_col=0)
        mask = ~c2_df["actual"].isna() & ~c2_df["c2_uni_point"].isna()
        y = c2_df.loc[mask, "actual"].values
        p_uni = c2_df.loc[mask, "c2_uni_point"].values

        new_rows = []

        # Chronos-2 univariate
        new_rows.append({
            "Model": "Chronos-Bolt-Base",
            "MAE": round(np.mean(np.abs(y - p_uni)), 4),
            "RMSE": round(np.sqrt(np.mean((y - p_uni)**2)), 4),
            "sMAPE": round(np.mean(2 * np.abs(y - p_uni) / (np.abs(y) + np.abs(p_uni) + 1e-8)) * 100, 4),
        })

        # Chronos-2 + covariates
        if "c2_cov_point" in c2_df.columns:
            mask2 = ~c2_df["actual"].isna() & ~c2_df["c2_cov_point"].isna()
            y2 = c2_df.loc[mask2, "actual"].values
            p_cov = c2_df.loc[mask2, "c2_cov_point"].values
            new_rows.append({
                "Model": "Chronos-Base+Cov",
                "MAE": round(np.mean(np.abs(y2 - p_cov)), 4),
                "RMSE": round(np.sqrt(np.mean((y2 - p_cov)**2)), 4),
                "sMAPE": round(np.mean(2 * np.abs(y2 - p_cov) / (np.abs(y2) + np.abs(p_cov) + 1e-8)) * 100, 4),
            })

        # Rolling LGBM
        roll_path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{market}.csv")
        if os.path.exists(roll_path):
            roll_df = pd.read_csv(roll_path)
            new_rows.append({
                "Model": "LightGBM (rolling)",
                "MAE": round(roll_df["Rolling_LGBM_MAE"].mean(), 4),
                "RMSE": round(roll_df["Rolling_LGBM_RMSE"].mean(), 4),
                "sMAPE": np.nan,  # need raw preds for sMAPE
            })
            new_rows.append({
                "Model": "XGBoost (rolling)",
                "MAE": round(roll_df["Rolling_XGB_MAE"].mean(), 4),
                "RMSE": round(roll_df["Rolling_XGB_RMSE"].mean(), 4),
                "sMAPE": np.nan,
            })

        # Remove any existing rows with same model names (idempotent)
        new_model_names = [r["Model"] for r in new_rows]
        df = df[~df["Model"].isin(new_model_names)]

        # Append
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values("MAE").reset_index(drop=True)

        out_path = os.path.join(config.REPORT_DIR, f"table_point_accuracy_full_{market}.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✅ Saved: table_point_accuracy_full_{market}.csv ({len(df)} models)")
        print(df[["Model", "MAE", "RMSE"]].to_string(index=False))
        print()


# ═══════════════════════════════════════════════════════════════
# 3. Rolling vs Static Summary Table
# ═══════════════════════════════════════════════════════════════
def generate_rolling_summary():
    print("\n[3] Generating rolling window summary...")

    rows = []
    for market in ["pjm", "ercot"]:
        roll_path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{market}.csv")
        if not os.path.exists(roll_path):
            continue
        df = pd.read_csv(roll_path)
        for model_prefix in ["LGBM", "XGB"]:
            rolling_mae = df[f"Rolling_{model_prefix}_MAE"].mean()
            static_mae  = df[f"Static_{model_prefix}_MAE"].mean()
            improvement = (static_mae - rolling_mae) / static_mae * 100
            rows.append({
                "Market": market.upper(),
                "Model": f"{'LightGBM' if model_prefix == 'LGBM' else 'XGBoost'}",
                "Static_MAE": round(static_mae, 4),
                "Rolling_MAE": round(rolling_mae, 4),
                "Improvement_%": round(improvement, 1),
                "N_windows": len(df),
            })

    summary = pd.DataFrame(rows)
    out_path = os.path.join(config.REPORT_DIR, "table_rolling_vs_static_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"  ✅ Saved: {out_path}")
    print(summary.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# 4. Chronos Evolution Figure (v1 → v2 → v2+cov)
# ═══════════════════════════════════════════════════════════════
def generate_chronos_evolution():
    print("\n[4] Generating Chronos Evolution Figure...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, market in zip(axes, ["pjm", "ercot"]):
        acc_path = os.path.join(config.REPORT_DIR, f"table_point_accuracy_{market}.csv")
        acc_df = pd.read_csv(acc_path)

        bolt_mae = acc_df[acc_df["Model"] == "Chronos-Bolt"].iloc[0]["MAE"]
        lgbm_mae = acc_df[acc_df["Model"] == "LightGBM"].iloc[0]["MAE"]

        c2_path = os.path.join(config.REPORT_DIR, f"chronos2_preds_{market}.csv")
        c2_df = pd.read_csv(c2_path, index_col=0)
        mask = ~c2_df["actual"].isna() & ~c2_df["c2_uni_point"].isna()
        c2_uni_mae = np.mean(np.abs(c2_df.loc[mask, "actual"] - c2_df.loc[mask, "c2_uni_point"]))

        c2_cov_mae = np.nan
        if "c2_cov_point" in c2_df.columns:
            mask2 = ~c2_df["actual"].isna() & ~c2_df["c2_cov_point"].isna()
            c2_cov_mae = np.mean(np.abs(c2_df.loc[mask2, "actual"] - c2_df.loc[mask2, "c2_cov_point"]))

        # Rolling LGBM
        roll_path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{market}.csv")
        roll_mae = pd.read_csv(roll_path)["Rolling_LGBM_MAE"].mean() if os.path.exists(roll_path) else np.nan

        # Bar chart: evolution
        labels = ["Chronos-Bolt\n(v1, univar.)", "Chronos-Base\n(v2, univar.)",
                  "Chronos-Base\n(v2 + cov.)", "LightGBM\n(static)", "LightGBM\n(rolling)"]
        values = [bolt_mae, c2_uni_mae, c2_cov_mae, lgbm_mae, roll_mae]
        colors = ["#EF9A9A", "#FF7043", "#FFC107", "#64B5F6", "#2E7D32"]

        bars = ax.bar(range(len(labels)), values, color=colors,
                     edgecolor="black", linewidth=0.5, width=0.7)

        # Value labels on top
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("MAE ($/MWh)")
        ax.set_title(f"{market.upper()}", fontweight="bold", fontsize=13)

        # Draw LightGBM reference line
        ax.axhline(lgbm_mae, color="#1565C0", ls="--", lw=1, alpha=0.5)

    fig.suptitle("TSFM Evolution: Covariates Close the Gap, Rolling Retraining Widens It\n"
                 "Chronos-Bolt-Base v2 + covariates approaches LightGBM, but rolling LGBM pulls ahead",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    save_fig(fig, "Fig8_Chronos_Evolution")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  INTEGRATING ROLLING WINDOW + CHRONOS-2 RESULTS")
    print("=" * 65)

    generate_chronos2_figure()
    update_accuracy_tables()
    generate_rolling_summary()
    generate_chronos_evolution()

    print("\n" + "=" * 65)
    print("  INTEGRATION COMPLETE")
    print("=" * 65)
