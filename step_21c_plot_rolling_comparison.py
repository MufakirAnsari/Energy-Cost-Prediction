import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def generate_comparison_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for col_idx, market in enumerate(["pjm", "ercot"]):
        # Load LightGBM/XGBoost rolling table
        tree_path = os.path.join(config.REPORT_DIR, f"table_rolling_window_{market}.csv")
        dl_path = os.path.join(config.REPORT_DIR, f"table_rolling_patchtst_{market}.csv")
        
        if not os.path.exists(tree_path) or not os.path.exists(dl_path):
            print(f"Missing data for {market}, skipping...")
            continue
            
        tree_df = pd.read_csv(tree_path)
        dl_df = pd.read_csv(dl_path)
        
        # Merge on Month
        df = pd.merge(tree_df, dl_df, on="Month", how="inner")
        
        # Top row: LightGBM (Tree Architecture)
        ax = axes[0, col_idx]
        ax.plot(df["Month"], df["Rolling_LGBM_MAE"], "o-", color="#2ca02c",
               label="LightGBM - Rolling Retrain", linewidth=1.5, markersize=4)
        ax.plot(df["Month"], df["Static_LGBM_MAE"], "s--", color="#2ca02c", alpha=0.4,
               label="LightGBM - Static", linewidth=1.5, markersize=4)
        ax.set_ylabel("MAE ($/MWh)")
        ax.set_title(f"{market.upper()}: LightGBM (Tree)", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Bottom row: PatchTST (Transformer Architecture)
        ax = axes[1, col_idx]
        ax.plot(df["Month"], df["Rolling_PatchTST_MAE"], "o-", color="#d62728",
               label="PatchTST - Rolling Retrain", linewidth=1.5, markersize=4)
        ax.plot(df["Month"], df["Static_PatchTST_MAE"], "s--", color="#d62728", alpha=0.4,
               label="PatchTST - Static", linewidth=1.5, markersize=4)
        ax.set_ylabel("MAE ($/MWh)")
        ax.set_xlabel("Test Month (2024-2025)")
        ax.set_title(f"{market.upper()}: PatchTST (Transformer)", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Expanding-Window Retraining: Trees vs Transformers\n"
                 "Monthly adaptation significantly helps LightGBM, but provides little-to-no benefit for PatchTST.",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_png = os.path.join(FIG_DIR, "FigW6_Rolling_Architecture_Comparison.png")
    out_pdf = os.path.join(FIG_DIR, "FigW6_Rolling_Architecture_Comparison.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"✅ Saved figure to {out_png}")

if __name__ == "__main__":
    generate_comparison_figure()
