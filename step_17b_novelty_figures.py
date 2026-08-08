"""
step_17b_novelty_figures.py
===========================
Generates the 5 specific novelty figures proposed in the implementation plan:
  Figure A: Exogenous Feature Superiority (SHAP Bar Chart)
  Figure B: The Conformal Breakdown (CQR vs. QRF Interval Plot)
  Figure C: Cumulative PnL Trajectory
  Figure D: Error Complementarity Scatter (LGBM vs BiLSTM)
  Figure E: Cross-Market Degradation Topology
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi":        300,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, name: str):
    path_png = os.path.join(FIG_DIR, f"{name}.png")
    path_pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {name}")
    plt.close(fig)

def generate_figA_shap(market: str = "pjm"):
    print(f"Generating Figure A: SHAP Feature Superiority ({market.upper()})...")
    path = os.path.join(config.REPORT_DIR, f"shap_importance_{market.lower()}.csv")
    if not os.path.exists(path):
        print(f"  Missing {path}")
        return
    
    df = pd.read_csv(path, index_col=0, header=0)
    df.index.name = "feature"
    df.columns = ["importance"]
    df = df.nlargest(15, "importance").reset_index()

    # Color code exogenous vs autoregressive (heuristic)
    colors = []
    for f in df["feature"]:
        if any(kw in f.lower() for kw in ["lag", "roll", "shift"]):
            colors.append("#B0BEC5") # Gray for autoregressive
        else:
            colors.append("#FF8F00") # Orange for Exogenous (Load, Weather, Gas, etc.)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(df["feature"], df["importance"], color=colors, alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title(f"SHAP Feature Importance (LightGBM, {market.upper()})", fontsize=11)
    
    import matplotlib.patches as mpatches
    exo_patch = mpatches.Patch(color="#FF8F00", label="Exogenous (Load, Weather, Gas)")
    ar_patch = mpatches.Patch(color="#B0BEC5", label="Autoregressive (Lags)")
    ax.legend(handles=[exo_patch, ar_patch], loc="lower right")
    
    fig.tight_layout()
    save_fig(fig, f"FigA_SHAP_Superiority_{market.upper()}")

def generate_figB_cqr_vs_qrf():
    print("Generating Figure B: CQR vs QRF Interval Breakdown...")
    market = "pjm"
    qrf_path = os.path.join(config.REPORT_DIR, f"qrf_preds_{market}.csv")
    cqr_path = os.path.join(config.REPORT_DIR, f"cqr_preds_{market}.csv")
    test_path = config.PJM_TEST_PATH
    
    if not (os.path.exists(qrf_path) and os.path.exists(cqr_path) and os.path.exists(test_path)):
        print("  Missing files for Figure B")
        return
        
    te_df = pd.read_parquet(test_path)
    qrf_df = pd.read_csv(qrf_path, index_col=0, parse_dates=True)
    cqr_df = pd.read_csv(cqr_path, index_col=0, parse_dates=True)
    
    te_df.index = pd.to_datetime(te_df.index, utc=True)
    qrf_df.index = pd.to_datetime(qrf_df.index, utc=True)
    cqr_df.index = pd.to_datetime(cqr_df.index, utc=True)
    
    # Align indices
    common_idx = te_df.index.intersection(qrf_df.index).intersection(cqr_df.index)
    te_df = te_df.loc[common_idx]
    qrf_df = qrf_df.loc[common_idx]
    cqr_df = cqr_df.loc[common_idx]
    
    # Find a highly volatile week to show breakdown
    # e.g., highest actual price spike
    max_spike_idx = te_df[config.TARGET_COL].argmax()
    spike_time = te_df.index[max_spike_idx]
    window_start = spike_time - pd.Timedelta(days=3)
    window_end = spike_time + pd.Timedelta(days=4)
    
    mask = (te_df.index >= window_start) & (te_df.index <= window_end)
    y_act = te_df.loc[mask, config.TARGET_COL]
    q_lo = qrf_df.loc[mask, "q05"]
    q_hi = qrf_df.loc[mask, "q95"]
    c_lo = cqr_df.loc[mask, "cqr_lower"]
    c_hi = cqr_df.loc[mask, "cqr_upper"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot intervals
    ax.fill_between(y_act.index, c_lo, c_hi, color="#26C6DA", alpha=0.3, label="CQR 90% Nominal (Failed)")
    ax.plot(y_act.index, c_lo, color="#26C6DA", ls="--", lw=1)
    ax.plot(y_act.index, c_hi, color="#26C6DA", ls="--", lw=1)
    
    ax.fill_between(y_act.index, q_lo, q_hi, color="#66BB6A", alpha=0.4, label="QRF 90% Actual (Robust)")
    ax.plot(y_act.index, q_lo, color="#2E7D32", ls="-", lw=1)
    ax.plot(y_act.index, q_hi, color="#2E7D32", ls="-", lw=1)
    
    # Plot Actual
    ax.plot(y_act.index, y_act.values, color="black", lw=1.5, marker=".", label="Actual Price")
    
    ax.set_title("CQR vs. QRF Prediction Intervals (PJM)", fontsize=11)
    ax.set_ylabel("Price ($/MWh)")
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    save_fig(fig, "FigB_CQR_vs_QRF")

def generate_figC_pnl():
    print("Generating Figure C: Cumulative PnL Trajectory...")
    path = os.path.join(config.REPORT_DIR, "pnl_daily_pjm.csv")
    if not os.path.exists(path):
        print("  Missing pnl_daily_pjm.csv")
        return
        
    pnl = pd.read_csv(path, index_col=0, parse_dates=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = {
        "LightGBM": "#FF8F00",
        "Seasonal Naive": "#90A4AE",
        "Risk-Aware CQR": "#26C6DA",
        "Risk-Aware Bayesian": "#42A5F5",
        "Oracle": "black"
    }
    
    for col in pnl.columns:
        c = colors.get(col, "gray")
        ls = "--" if col == "Oracle" else "-"
        ax.plot(pnl.index, pnl[col].cumsum(), label=col, lw=1.5, color=c, linestyle=ls)
        
    ax.set_title("Cumulative Trading P&L (PJM, 2024–2025)", fontsize=11)
    ax.set_ylabel("Cumulative Profit ($)")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)
    
    fig.tight_layout()
    save_fig(fig, "FigC_PnL_Trajectory")

def generate_figD_error_scatter():
    print("Generating Figure D: Error Complementarity Scatter...")
    market = "pjm"
    path = os.path.join(config.REPORT_DIR, f"ensemble_preds_{market}.csv")
    
    if not os.path.exists(path):
        print(f"  Missing {path} for Figure D")
        return
        
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    
    if "lgbm" not in df.columns or "bilstm" not in df.columns or "actual" not in df.columns:
        print("  Missing required columns in ensemble preds")
        return
        
    # Drop rows where predictions might be NaN
    df = df.dropna(subset=["actual", "lgbm", "bilstm"])
    
    y_act = df["actual"]
    y_lgbm = df["lgbm"]
    y_lstm = df["bilstm"]
    
    err_lgbm = y_lgbm - y_act
    err_lstm = y_lstm - y_act
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Color by regime (simple heuristc: magnitude of actual price)
    regime = np.where(y_act > np.percentile(y_act, 95), "Spike (>95th %ile)", "Normal")
    colors = {"Spike (>95th %ile)": "#E53935", "Normal": "#1E88E5"}
    
    for r in ["Normal", "Spike (>95th %ile)"]:
        mask = (regime == r)
        ax.scatter(err_lgbm[mask], err_lstm[mask], alpha=0.4, s=15, 
                   color=colors[r], label=r)
                   
    # Perfect complementarity line
    max_val = max(err_lgbm.abs().max(), err_lstm.abs().max())
    ax.axline((-max_val, -max_val), (max_val, max_val), color="black", ls="--", lw=1, alpha=0.5, label="y=x (Same Error)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    
    ax.set_xlabel("LightGBM Error ($/MWh)")
    ax.set_ylabel("BiLSTM Error ($/MWh)")
    ax.set_title("LightGBM vs. BiLSTM Error Correlation (PJM)", fontsize=11)
    ax.legend()
    
    fig.tight_layout()
    save_fig(fig, "FigD_Error_Complementarity")

def generate_figE_cross_market():
    print("Generating Figure E: Cross-Market Degradation Topology...")
    path = os.path.join(config.REPORT_DIR, "table_rq4_crossmarket.csv")
    if not os.path.exists(path):
        print("  Missing table_rq4_crossmarket.csv")
        return
        
    df = pd.read_csv(path)
    
    # Filter to only LightGBM and XGBoost
    models = ["LightGBM", "XGBoost"]
    df = df[df["Model"].isin(models)]
    
    in_market = df[df["Direction"].str.contains("ERCOT→ERCOT")]
    cross_market = df[df["Direction"].str.contains("PJM→ERCOT")]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    mae_in = []
    mae_cross = []
    for m in models:
        mae_in.append(in_market[in_market["Model"] == m]["MAE"].values[0])
        mae_cross.append(cross_market[cross_market["Model"] == m]["MAE"].values[0])
        
    rects1 = ax.bar(x - width/2, mae_in, width, label='In-Market (ERCOT→ERCOT)', color="#4CAF50")
    rects2 = ax.bar(x + width/2, mae_cross, width, label='Cross-Market (PJM→ERCOT)', color="#EF5350")
    
    ax.set_ylabel('MAE ($/MWh)')
    ax.set_title('Cross-Market Transfer: In-Market vs. PJM→ERCOT', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    for r1, r2 in zip(rects1, rects2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax.annotate(f"{h1:.2f}", xy=(r1.get_x() + r1.get_width() / 2, h1),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        ax.annotate(f"{h2:.2f}", xy=(r2.get_x() + r2.get_width() / 2, h2),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
        degrad = ((h2 - h1)/h1) * 100
        ax.annotate(f"+{degrad:.1f}%\nDegradation", xy=(r2.get_x(), h2+1.5), color="red", fontsize=9, ha="center")
        
    ax.set_ylim(0, max(mae_cross) * 1.3)
    fig.tight_layout()
    save_fig(fig, "FigE_CrossMarket_Topology")

if __name__ == "__main__":
    generate_figA_shap("pjm")
    generate_figA_shap("ercot")
    generate_figB_cqr_vs_qrf()
    generate_figC_pnl()
    generate_figD_error_scatter()
    generate_figE_cross_market()
    print("Done generating novelty figures.")
