"""
step_01_eda.py — Exploratory Data Analysis
===========================================
Generates publication-quality EDA figures:
  EDA1 — Price distribution violin plots per regime (PJM + ERCOT)
  EDA2 — ACF / PACF (24h + 168h seasonality) — both markets
  EDA3 — Feature correlation heatmap (top 20 features, LightGBM SHAP ordering)
  EDA4 — Price spike frequency heatmap (hour-of-day × month)

All figures saved to reports/figures/ as PNG (300 DPI) + PDF.

Run:
    python step_01_eda.py
"""
import os, sys
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "figure.dpi": 300, "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")

REGIME_COLORS = {
    "stable_baseline": "#4CAF50",
    "covid_collapse":  "#FF9800",
    "uri_crisis":      "#F44336",
    "gas_shock":       "#9C27B0",
    "new_normal":      "#2196F3",
}


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


def load_full_market(market):
    """Load train+cal+val+test concatenated."""
    splits = []
    for path in [
        config.PJM_TRAIN_PATH  if market=="PJM" else config.ERCOT_TRAIN_PATH,
        config.PJM_CAL_PATH    if market=="PJM" else config.ERCOT_CAL_PATH,
        config.PJM_VAL_PATH    if market=="PJM" else config.ERCOT_VAL_PATH,
        config.PJM_TEST_PATH   if market=="PJM" else config.ERCOT_TEST_PATH,
    ]:
        if os.path.exists(path):
            splits.append(pd.read_parquet(path))
    return pd.concat(splits).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# EDA1 — Violin Plots by Regime
# ─────────────────────────────────────────────────────────────────────────────
def fig_violin_regimes():
    print("  EDA1: Regime violin plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, market in zip(axes, ["PJM", "ERCOT"]):
        full = load_full_market(market)[config.TARGET_COL]
        regime_data, regime_labels, colors = [], [], []

        for regime_name, (r_start, r_end) in config.REGIMES.items():
            mask = (full.index >= pd.Timestamp(r_start, tz="UTC")) & \
                   (full.index <= pd.Timestamp(r_end,   tz="UTC"))
            vals = full[mask].dropna()
            if len(vals) < 100:
                continue
            # Clip to [-50, 200] for visibility
            regime_data.append(np.clip(vals.values, -50, 500))
            regime_labels.append(regime_name.replace("_", "\n").title())
            colors.append(REGIME_COLORS.get(regime_name, "gray"))

        parts = ax.violinplot(regime_data, showmedians=True, widths=0.7)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(1, len(regime_labels)+1))
        ax.set_xticklabels(regime_labels, fontsize=7)
        ax.set_ylabel("LMP ($/MWh)")
        ax.set_title(f"{market} — Price Distribution by Market Regime", fontweight="bold")
        if market == "ERCOT":
            ax.set_ylim(-50, 500)
            ax.text(0.97, 0.97, "Uri: $9,000 cap\nclipped for visibility",
                    transform=ax.transAxes, fontsize=6, ha="right", va="top",
                    color="red", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.7))

    fig.tight_layout()
    save_fig(fig, "EDA1_violin_regimes")


# ─────────────────────────────────────────────────────────────────────────────
# EDA2 — ACF / PACF
# ─────────────────────────────────────────────────────────────────────────────
def fig_acf_pacf():
    print("  EDA2: ACF / PACF...")
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        print("    statsmodels not available — skipping EDA2")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for col, market in enumerate(["PJM", "ERCOT"]):
        full = load_full_market(market)[config.TARGET_COL].dropna()
        # Use training portion only for ACF
        train_path = config.PJM_TRAIN_PATH if market=="PJM" else config.ERCOT_TRAIN_PATH
        train = pd.read_parquet(train_path)[config.TARGET_COL].dropna()

        plot_acf(train.values,  lags=48, ax=axes[0, col], alpha=0.05,
                 title=f"{market} — ACF (48 lags)", color="#1565C0")
        plot_pacf(train.values, lags=48, ax=axes[1, col], alpha=0.05, method="ywm",
                  title=f"{market} — PACF (48 lags)", color="#E53935")

        # Mark h=24 and h=168 seasonality
        for ax_r in [axes[0, col], axes[1, col]]:
            ax_r.axvline(24,  color="orange", ls="--", lw=1, alpha=0.7, label="24h (daily)")
            ax_r.axvline(168, color="green",  ls="--", lw=1, alpha=0.7, label="168h (weekly)")
            ax_r.legend(fontsize=6)

    fig.tight_layout()
    save_fig(fig, "EDA2_acf_pacf")


# ─────────────────────────────────────────────────────────────────────────────
# EDA3 — Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_correlation():
    print("  EDA3: Feature correlation heatmap...")
    # Use PJM training set, top features from SHAP importance
    train = pd.read_parquet(config.PJM_TRAIN_PATH)
    shap_path = os.path.join(config.REPORT_DIR, "shap_importance.csv")

    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path, index_col=0, header=0)
        shap_df.columns = ["importance"]
        top_feats = shap_df.nlargest(20, "importance").index.tolist()
        top_feats = [f for f in top_feats if f in train.columns][:16]
    else:
        # Fallback: pick numeric cols
        numeric = train.select_dtypes(include=[np.number]).columns
        top_feats = list(numeric[:16])

    corr = train[top_feats].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_xticks(range(len(top_feats)))
    ax.set_yticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(top_feats, fontsize=7)
    ax.set_title("PJM Top-16 Feature Correlation (SHAP-ranked)", fontweight="bold")
    for i in range(len(top_feats)):
        for j in range(len(top_feats)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if abs(val) < 0.7 else "white")
    fig.tight_layout()
    save_fig(fig, "EDA3_feature_correlation")


# ─────────────────────────────────────────────────────────────────────────────
# EDA4 — Price Spike Heatmap (Hour × Month)
# ─────────────────────────────────────────────────────────────────────────────
def fig_spike_heatmap():
    print("  EDA4: Price spike frequency heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, market in zip(axes, ["PJM", "ERCOT"]):
        full = load_full_market(market)[config.TARGET_COL].dropna()
        thresh = full.mean() + 2 * full.std()
        spikes = full[full > thresh]

        pivot = pd.DataFrame({
            "hour":  spikes.index.hour,
            "month": spikes.index.month,
        }).groupby(["hour", "month"]).size().unstack(fill_value=0)

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Spike Count (>2σ)", shrink=0.8)
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=7)
        ax.set_yticks(range(24))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(24)], fontsize=5)
        ax.set_title(f"{market} — Price Spike Frequency (>2σ = ${thresh:.0f}/MWh)",
                     fontweight="bold")
        ax.set_ylabel("Hour of Day")

    fig.tight_layout()
    save_fig(fig, "EDA4_spike_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Price Statistics Table
# ─────────────────────────────────────────────────────────────────────────────
def compute_price_stats():
    print("  Computing price statistics table...")
    rows = []
    for market in ["PJM", "ERCOT"]:
        full = load_full_market(market)[config.TARGET_COL].dropna()
        for regime_name, (r_start, r_end) in list(config.REGIMES.items()) + \
                [("full_dataset", (config.DATA_START, config.DATA_END))]:
            mask = (full.index >= pd.Timestamp(r_start, tz="UTC")) & \
                   (full.index <= pd.Timestamp(r_end,   tz="UTC"))
            vals = full[mask]
            if len(vals) < 10:
                continue
            thresh = full.mean() + 2 * full.std()
            rows.append({
                "Market": market, "Regime": regime_name,
                "N_hours": len(vals),
                "Mean": round(vals.mean(), 2),
                "Std":  round(vals.std(), 2),
                "Min":  round(vals.min(), 2),
                "Max":  round(vals.max(), 2),
                "Skew": round(float(vals.skew()), 3),
                "Kurt": round(float(vals.kurtosis()), 3),
                "Spike_pct>2sigma": round((vals > thresh).mean() * 100, 2),
                "Negative_pct":     round((vals < 0).mean() * 100, 2),
            })
    stats_df = pd.DataFrame(rows)
    out_path = os.path.join(config.REPORT_DIR, "table_price_statistics.csv")
    stats_df.to_csv(out_path, index=False)
    print(f"  ✅ Saved: {out_path}")
    print(stats_df[["Market","Regime","N_hours","Mean","Std","Max","Spike_pct>2sigma"]].to_string(index=False))
    return stats_df


if __name__ == "__main__":
    print(f"\n{'='*65}\n  EDA: Exploratory Data Analysis\n{'='*65}\n")
    compute_price_stats()
    fig_violin_regimes()
    fig_acf_pacf()
    fig_feature_correlation()
    fig_spike_heatmap()
    print(f"\n  ✅ All EDA outputs saved to: {FIG_DIR}")
