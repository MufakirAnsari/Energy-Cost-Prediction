"""
step_17_figures.py
==================
Generates all publication figures at 300 DPI.

Figures produced:
  F1 — Price time series with regime shading (PJM + ERCOT)
  F2 — Model accuracy comparison bar chart (MAE + RMSE)
  F3 — Probabilistic interval comparison (PICP vs MPIW scatter)
  F4 — Regime MAE heatmap
  F5 — Winkler score comparison
  F6 — Daily P&L cumulative curves (economic utility)
  F7 — SHAP feature importance (top 20 features)
  F8 — Ablation study: feature set vs lag window
  F9 — Chronos vs trained models (foundation model benchmark)
  F10 — Uri crisis detail (ERCOT Feb 2021 actual vs. forecasts)

All figures saved to reports/figures/ as both PNG (300 DPI) and PDF.

Run:
    python step_17_figures.py
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    import matplotlib
    matplotlib.use("Agg")   # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import AutoMinorLocator
except ImportError:
    raise ImportError("pip install matplotlib")

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        300,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

REGIME_COLORS = {
    "stable_baseline": "#4CAF50",
    "covid_collapse":  "#FF9800",
    "uri_crisis":      "#F44336",
    "gas_shock":       "#9C27B0",
    "new_normal":      "#2196F3",
}

MODEL_COLORS = {
    "SeasonalNaive":  "#90A4AE",
    "AutoARIMA":      "#78909C",
    "MSTL":           "#607D8B",
    "LightGBM":       "#FF8F00",
    "XGBoost":        "#F57C00",
    "BiLSTM":         "#42A5F5",
    "BiTCN":          "#29B6F6",
    "TFT":            "#1565C0",   # dark blue
    "PatchTST":       "#7E57C2",
    "iTransformer":   "#AB47BC",
    "N-HiTS":         "#26A69A",
    "N-HiTS Quantile": "#00897B",
    "Chronos-Bolt":   "#EF5350",
    "QRF":            "#66BB6A",
    "Ensemble":       "#EC407A",
    "CQR":            "#26C6DA",
    "BiLSTM MC Dropout": "#42A5F5",
}

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")


def save_fig(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


def add_regime_shading(ax, idx):
    for regime_name, (r_start, r_end) in config.REGIMES.items():
        ax.axvspan(
            pd.Timestamp(r_start, tz="UTC"),
            pd.Timestamp(r_end,   tz="UTC"),
            alpha=0.08,
            color=REGIME_COLORS.get(regime_name, "gray"),
            label=regime_name.replace("_", " ").title(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Price Time Series with Regime Shading
# ─────────────────────────────────────────────────────────────────────────────
def fig_price_timeseries():
    print("  F1: Price time series...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    for ax, market, path in [
        (axes[0], "PJM",   config.PJM_TRAIN_PATH),
        (axes[1], "ERCOT", config.ERCOT_TRAIN_PATH),
    ]:
        splits = []
        for p in [
            (config.PJM_TRAIN_PATH if market == "PJM" else config.ERCOT_TRAIN_PATH),
            (config.PJM_CAL_PATH   if market == "PJM" else config.ERCOT_CAL_PATH),
            (config.PJM_VAL_PATH   if market == "PJM" else config.ERCOT_VAL_PATH),
            (config.PJM_TEST_PATH  if market == "PJM" else config.ERCOT_TEST_PATH),
        ]:
            if os.path.exists(p):
                splits.append(pd.read_parquet(p)[config.TARGET_COL])
        price = pd.concat(splits).resample("D").mean()

        add_regime_shading(ax, price.index)
        ax.plot(price.index, price.values, lw=0.6, color="#1565C0", alpha=0.85)
        ax.set_title(f"{market} Day-Ahead LMP (Daily Average)", fontweight="bold")
        ax.set_ylabel("$/MWh")

        # Clip ERCOT for visibility (Uri spike dominates)
        if market == "ERCOT":
            ax.set_ylim(-50, 300)
            ax.annotate("Uri: $9,000/MWh →", xy=(pd.Timestamp("2021-02-15"), 280),
                        fontsize=7, color="#F44336")

    # Legend
    handles = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4,
                               label=r.replace("_", " ").title())
               for r in REGIME_COLORS]
    axes[0].legend(handles=handles, loc="upper left", ncol=3, fontsize=7)

    fig.tight_layout()
    save_fig(fig, "F1_price_timeseries")


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Model Accuracy Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_accuracy_comparison():
    print("  F2: Accuracy comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, market in zip(axes, ["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"table_point_accuracy_{market}.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Run step_13 first\n({market.upper()})",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        df = pd.read_csv(path, index_col=0).sort_values("MAE")
        colors = [MODEL_COLORS.get(m, "#78909C") for m in df.index]
        bars = ax.barh(df.index, df["MAE"], color=colors, alpha=0.85, height=0.6)
        ax.set_xlabel("MAE ($/MWh)")
        ax.set_title(f"{market.upper()} — Point Accuracy (MAE)", fontweight="bold")
        for bar, val in zip(bars, df["MAE"]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "F2_accuracy_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Probabilistic Quality: PICP vs MPIW
# ─────────────────────────────────────────────────────────────────────────────
def fig_prob_quality():
    print("  F3: Probabilistic quality scatter...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, market in zip(axes, ["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"table_probabilistic_{market}.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "Run step_13 first", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        df = pd.read_csv(path, index_col=0)
        if "PICP_%" not in df.columns or "MPIW" not in df.columns:
            continue

        # Clip x-axis to exclude extreme outliers (e.g. CQR over-correction)
        mpiw_vals = df["MPIW"].dropna().values
        x_clip = np.percentile(mpiw_vals, 90) * 1.5
        x_clip = max(x_clip, 30)   # at least 30 $/MWh range
        clipped_models = []

        for model_name in df.index:
            mpiw_val = df.loc[model_name, "MPIW"]
            picp_val = df.loc[model_name, "PICP_%"]
            color = MODEL_COLORS.get(model_name.split(" ")[0], "#78909C")
            if mpiw_val > x_clip:
                clipped_models.append(f"{model_name} (MPIW={mpiw_val:.0f})")
                continue
            ax.scatter(mpiw_val, picp_val, color=color, s=80, zorder=5, label=model_name)
            ax.annotate(model_name, (mpiw_val, picp_val),
                        textcoords="offset points", xytext=(5, 2), fontsize=6)

        ax.axhline(90, color="red", lw=1, ls="--", label="90% target")
        ax.set_xlim(left=0, right=x_clip)
        ax.set_xlabel("MPIW — Interval Width ($/MWh)\n[lower is sharper]")
        ax.set_ylabel("PICP (%) [higher is better coverage]")
        ax.set_title(f"{market.upper()} — PICP vs. MPIW", fontweight="bold")
        if clipped_models:
            note = "Clipped (off-chart):\n" + "\n".join(clipped_models)
            ax.text(0.97, 0.03, note, transform=ax.transAxes,
                    fontsize=6, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
        ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    save_fig(fig, "F3_probabilistic_quality")


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Regime MAE Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_regime_heatmap():
    print("  F4: Regime MAE heatmap...")
    try:
        import matplotlib.colors as mcolors
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, market in zip(axes, ["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"heatmap_regime_mae_{market}.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "Run step_16 first", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        df = pd.read_csv(path, index_col=0)
        im = ax.imshow(df.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels([r.replace("_", " ").title() for r in df.index], fontsize=7)
        ax.set_title(f"{market.upper()} — Regime MAE Heatmap ($/MWh)", fontweight="bold")
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=6, color="black")
        plt.colorbar(im, ax=ax, label="MAE ($/MWh)")

    fig.tight_layout()
    save_fig(fig, "F4_regime_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# F5 — Economic Utility: Cumulative P&L
# ─────────────────────────────────────────────────────────────────────────────
def fig_economic_pnl():
    print("  F5: Economic P&L curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, market in zip(axes, ["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"pnl_daily_{market}.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "Run step_13 first", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        pnl = pd.read_csv(path, index_col=0, parse_dates=True)
        for col in pnl.columns:
            color = MODEL_COLORS.get(col, "#78909C")
            ls    = "--" if col == "Oracle" else "-"
            ax.plot(pnl.index, pnl[col].cumsum(), label=col, lw=1.2,
                    color=color, linestyle=ls)
        ax.set_title(f"{market.upper()} — Cumulative P&L", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    save_fig(fig, "F5_economic_pnl")


# ─────────────────────────────────────────────────────────────────────────────
# F6 — SHAP Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
def fig_shap_importance():
    print("  F6: SHAP feature importance...")
    path = os.path.join(config.REPORT_DIR, "shap_importance.csv")
    if not os.path.exists(path):
        print("    shap_importance.csv not found — skipping.")
        return

    # CSV format: unnamed index = feature name, column '0' = importance value
    df = pd.read_csv(path, index_col=0, header=0)
    df.index.name = "feature"
    df.columns = ["importance"]
    df = df.nlargest(20, "importance").reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["importance"], color="#7E57C2", alpha=0.8)
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title("Top 20 Features by SHAP Importance (LightGBM, PJM)", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "F6_shap_importance")


# ─────────────────────────────────────────────────────────────────────────────
# F7 — Ablation Study
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    print("  F7: Ablation study...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, market in zip(axes, ["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"table_ablation_{market}.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "Run step_15 first", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        df = pd.read_csv(path)
        for ablation_type in df["Ablation"].unique():
            sub = df[df["Ablation"] == ablation_type].sort_values("MAE")
            ax.barh(sub["config"], sub["MAE"], alpha=0.75, label=ablation_type)
        ax.set_xlabel("MAE ($/MWh)")
        ax.set_title(f"{market.upper()} — Ablation Study", fontweight="bold")
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, "F7_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# F8 — Uri Crisis Detail (ERCOT)
# ─────────────────────────────────────────────────────────────────────────────
def fig_uri_crisis():
    print("  F8: Uri crisis detail...")
    # Load ERCOT full price series
    splits = []
    for p in [config.ERCOT_TRAIN_PATH, config.ERCOT_CAL_PATH,
              config.ERCOT_VAL_PATH, config.ERCOT_TEST_PATH]:
        if os.path.exists(p):
            splits.append(pd.read_parquet(p)[config.TARGET_COL])
    if not splits:
        print("    ERCOT data not found — skipping.")
        return

    full_price = pd.concat(splits).sort_index()   # must be monotonic for label-slice
    uri_start  = pd.Timestamp("2021-02-08", tz="UTC")
    uri_end    = pd.Timestamp("2021-02-22", tz="UTC")
    uri = full_price.loc[uri_start:uri_end]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(uri.index, uri.values, lw=1.5, color="#F44336", label="Actual ERCOT LMP")
    ax.axvspan(pd.Timestamp("2021-02-10"), pd.Timestamp("2021-02-19"),
               alpha=0.1, color="red", label="Uri Crisis")
    ax.set_title("Winter Storm Uri — ERCOT Price Spike (Feb 2021)", fontweight="bold")
    ax.set_ylabel("$/MWh")
    ax.set_xlabel("Date (UTC)")

    # ── In-sample retroactive overlay (tree models only) ─────────────────────
    # Prediction files (chronos, baseline_test) cover 2024-2025 and do NOT
    # include Feb 2021 — overlaying them here was incorrect.
    # Instead we use LGBM/XGBoost evaluated retroactively on training data,
    # clearly labelled [in-sample] per paper Section 4.4.
    try:
        tr_df = pd.read_parquet(config.ERCOT_TRAIN_PATH)
        uri_mask = (tr_df.index >= pd.Timestamp("2021-02-08", tz="UTC")) & \
                   (tr_df.index <= pd.Timestamp("2021-02-21", tz="UTC")) \
                   if tr_df.index.tz is not None else \
                   (tr_df.index >= pd.Timestamp("2021-02-08")) & \
                   (tr_df.index <= pd.Timestamp("2021-02-21"))
        uri_tr = tr_df[uri_mask]
        if len(uri_tr) > 0:
            X_uri = uri_tr.drop(columns=[config.TARGET_COL])
            for model_name, fname, color in [
                ("LightGBM [in-sample]",  "lgbm_point_ercot.joblib",    "#2196F3"),
                ("XGBoost [in-sample]",   "xgboost_point_ercot.joblib", "#4CAF50"),
            ]:
                mpath = os.path.join(config.MODEL_DIR, fname)
                if os.path.exists(mpath):
                    m_obj = joblib.load(mpath)
                    p = pd.Series(m_obj.predict(X_uri), index=uri_tr.index)
                    ax.plot(p.index, p.values, lw=1.2, ls="--",
                            color=color, label=model_name, alpha=0.8)
    except Exception:
        pass  # graceful skip if training data not present

    ax.legend()
    fig.tight_layout()
    save_fig(fig, "F8_uri_crisis")




# ───────────────────────────────────────────────────────────────────────────────
# F9 — Conformal Alpha Sweep
# ───────────────────────────────────────────────────────────────────────────────
def fig_alpha_sweep():
    print("  F9: Conformal alpha sweep (PICP vs. MPIW)...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, market in zip(axes, ["pjm", "ercot"]):
        path = os.path.join(config.REPORT_DIR, f"table_alpha_sweep_{market}.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Run step_10b_alpha_sweep.py first",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        df = pd.read_csv(path)
        alphas   = df["Alpha"].values
        picp_vals = df["PICP_%"].values
        mpiw_vals = df["MPIW"].values
        labels    = df["Label"].values
        nominals  = (1 - alphas) * 100

        ax2 = ax.twinx()
        ax.plot(nominals, picp_vals, "o-", color="#1565C0", lw=2, label="PICP (%)")
        ax2.plot(nominals, mpiw_vals, "s--", color="#E53935", lw=2, label="MPIW ($/MWh)")
        ax.axline((80, 80), slope=1, color="gray", ls=":", lw=1, label="Nominal = Actual")

        ax.set_xlabel("Nominal Coverage (%)")
        ax.set_ylabel("Actual PICP (%)", color="#1565C0")
        ax2.set_ylabel("MPIW ($/MWh)", color="#E53935")
        ax.set_title(f"{market.upper()} — CQR Coverage vs. Sharpness Tradeoff",
                     fontweight="bold")
        ax.tick_params(axis="y", labelcolor="#1565C0")
        ax2.tick_params(axis="y", labelcolor="#E53935")
        ax.set_xticks(nominals)
        ax.set_xticklabels([f"{n:.0f}%" for n in nominals])

        # Annotate each point
        for ni, pi, mi, lbl in zip(nominals, picp_vals, mpiw_vals, labels):
            ax.annotate(f"PICP={pi:.1f}%", (ni, pi), textcoords="offset points",
                        xytext=(0, 8), fontsize=6, ha="center", color="#1565C0")
            ax2.annotate(f"MPIW={mi:.1f}", (ni, mi), textcoords="offset points",
                         xytext=(0, -12), fontsize=6, ha="center", color="#E53935")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()
    save_fig(fig, "F9_alpha_sweep")


def fig_reliability_diagram():
    """F10 — Calibration reliability diagram for all probabilistic methods."""
    print("  F10: Reliability diagram (calibration curves)...")
    markets = ["PJM", "ERCOT"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    quantile_levels = np.array([0.05, 0.10, 0.20, 0.30, 0.40,
                                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])

    for ax, market in zip(axes, markets):
        m = market.lower()
        # Load actuals
        te_path = config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
        if not os.path.exists(te_path):
            continue
        te_df = pd.read_parquet(te_path)
        y_true = te_df[config.TARGET_COL].values
        idx    = te_df.index

        def emp_coverage(lo, hi, y):
            mask = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
            return np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask]))

        def load_col(fname, col):
            p = os.path.join(config.REPORT_DIR, fname)
            if not os.path.exists(p): return None
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            return df[col].reindex(idx_utc).values if col in df.columns else None

        # QRF — 5 quantile pairs → 4 intervals + 90% CI
        qrf_qs = {}
        qrf_path = os.path.join(config.REPORT_DIR, f"qrf_preds_{m}.csv")
        if os.path.exists(qrf_path):
            q_df = pd.read_csv(qrf_path, index_col=0, parse_dates=True)
            q_df.index = pd.to_datetime(q_df.index, utc=True)
            idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            q_df = q_df.reindex(idx_utc)
            pairs = [("q05","q95",0.90), ("q25","q75",0.50)]
            qrf_coverages, qrf_nominals = [], []
            for lo_c, hi_c, nom in pairs:
                if lo_c in q_df.columns and hi_c in q_df.columns:
                    cov = emp_coverage(q_df[lo_c].values, q_df[hi_c].values, y_true)
                    qrf_coverages.append(cov * 100)
                    qrf_nominals.append(nom * 100)
            if qrf_coverages:
                ax.plot(qrf_nominals, qrf_coverages, "o-",
                        color=MODEL_COLORS["QRF"], lw=1.8, label="QRF", zorder=3)

        # LGBM Quantile — load quantile models and predict on test set
        import joblib as jl
        lgbm_cvg = []
        X_te = te_df.drop(columns=[config.TARGET_COL])
        for lo_q, hi_q, nom in [("q05","q95",0.90), ("q25","q75",0.50)]:
            lo_p = os.path.join(config.MODEL_DIR, f"lgbm_{lo_q}_{m}.joblib")
            hi_p = os.path.join(config.MODEL_DIR, f"lgbm_{hi_q}_{m}.joblib")
            if os.path.exists(lo_p) and os.path.exists(hi_p):
                lv = jl.load(lo_p).predict(X_te)
                hv = jl.load(hi_p).predict(X_te)
                cov = emp_coverage(lv, hv, y_true)
                lgbm_cvg.append((nom * 100, cov * 100))

        if lgbm_cvg:
            ns, cs = zip(*lgbm_cvg)
            ax.plot(ns, cs, "s--", color=MODEL_COLORS["LightGBM"],
                    lw=1.8, label="LGBM Quantile", zorder=3)

        # CQR — single point (90%)
        cqr_path = os.path.join(config.REPORT_DIR, f"cqr_preds_{m}.csv")
        if os.path.exists(cqr_path):
            cqr_df = pd.read_csv(cqr_path, index_col=0, parse_dates=True)
            cqr_df.index = pd.to_datetime(cqr_df.index, utc=True)
            idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            cqr_df = cqr_df.reindex(idx_utc)
            if "cqr_lower" in cqr_df.columns:
                cov = emp_coverage(cqr_df["cqr_lower"].values,
                                   cqr_df["cqr_upper"].values, y_true)
                ax.scatter([90], [cov * 100], marker="*", s=100,
                           color=MODEL_COLORS["CQR"], label="CQR", zorder=5)

        # N-HiTS-Q — single point (80%)
        nhq_path = os.path.join(config.REPORT_DIR, f"nhits_quantile_preds_{m}.csv")
        if os.path.exists(nhq_path):
            nhq = pd.read_csv(nhq_path, parse_dates=["ds"]).set_index("ds")
            nhq.index = pd.to_datetime(nhq.index, utc=True)
            nhq = nhq[~nhq.index.duplicated(keep="first")]
            idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            nhq = nhq.reindex(idx_utc)
            if "q10" in nhq.columns and "q90" in nhq.columns:
                cov = emp_coverage(nhq["q10"].values, nhq["q90"].values, y_true)
                ax.scatter([80], [cov * 100], marker="^", s=80,
                           color=MODEL_COLORS["N-HiTS Quantile"],
                           label="N-HiTS-Q", zorder=5)

        # Diagonal (perfect calibration)
        ax.plot([0, 100], [0, 100], "k:", lw=1.2, label="Perfect calibration")
        ax.set_xlim(40, 100); ax.set_ylim(40, 105)
        ax.set_xlabel("Nominal Coverage (%)")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_title(f"{market} — Reliability Diagram", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.fill_between([40, 100], [40, 100], [105, 105], alpha=0.04,
                        color="green", label="Over-coverage zone")

    fig.suptitle("Calibration Reliability Diagram — All Probabilistic Methods",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "F10_reliability_diagram")


def fig_quantile_fan():
    """F11 — Quantile fan chart for QRF vs N-HiTS-Q vs CQR over one representative week."""
    print("  F11: Quantile fan chart...")
    market = "PJM"
    m = market.lower()

    te_df = pd.read_parquet(config.PJM_TEST_PATH)
    # Pick a representative 2-week window from the test set (avoid NaNs)
    y_all = te_df[config.TARGET_COL]
    window_start = y_all.dropna().index[168]   # start from 168h into test (skip warmup)
    window_end   = window_start + pd.Timedelta(hours=167)
    mask_w = (te_df.index >= window_start) & (te_df.index <= window_end)
    te_win = te_df.loc[mask_w]
    y_win  = te_win[config.TARGET_COL].values
    t_win  = te_win.index

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    titles = ["QRF (5-quantile fan)", "N-HiTS Quantile (3-quantile fan)", "CQR (conformal interval)"]

    def get_window(df, cols):
        df.index = pd.to_datetime(df.index, utc=True)
        w_utc = t_win.tz_localize("UTC") if t_win.tz is None else t_win.tz_convert("UTC")
        df = df[~df.index.duplicated(keep="first")]
        return {c: df[c].reindex(w_utc).values for c in cols if c in df.columns}

    # QRF fan
    qrf_path = os.path.join(config.REPORT_DIR, f"qrf_preds_{m}.csv")
    ax = axes[0]
    if os.path.exists(qrf_path):
        q = get_window(pd.read_csv(qrf_path, index_col=0, parse_dates=True),
                       ["q05","q25","q50","q75","q95"])
        if "q05" in q:
            ax.fill_between(range(len(y_win)), q["q05"], q["q95"],
                            alpha=0.15, color=MODEL_COLORS["QRF"], label="Q05–Q95")
        if "q25" in q:
            ax.fill_between(range(len(y_win)), q["q25"], q["q75"],
                            alpha=0.30, color=MODEL_COLORS["QRF"], label="Q25–Q75")
        if "q50" in q:
            ax.plot(q["q50"], color=MODEL_COLORS["QRF"], lw=1.5, label="Median")
    ax.plot(y_win, "k-", lw=0.8, label="Actual", zorder=5)
    ax.set_title(titles[0], fontweight="bold"); ax.legend(fontsize=7)

    # N-HiTS-Q fan
    nhq_path = os.path.join(config.REPORT_DIR, f"nhits_quantile_preds_{m}.csv")
    ax = axes[1]
    if os.path.exists(nhq_path):
        nhq = pd.read_csv(nhq_path, parse_dates=["ds"]).set_index("ds")
        q = get_window(nhq, ["q10","q50","q90"])
        if "q10" in q:
            ax.fill_between(range(len(y_win)), q["q10"], q["q90"],
                            alpha=0.20, color=MODEL_COLORS["N-HiTS Quantile"], label="Q10–Q90")
        if "q50" in q:
            ax.plot(q["q50"], color=MODEL_COLORS["N-HiTS Quantile"], lw=1.5, label="Median")
    ax.plot(y_win, "k-", lw=0.8, label="Actual", zorder=5)
    ax.set_title(titles[1], fontweight="bold"); ax.legend(fontsize=7)

    # CQR interval
    cqr_path = os.path.join(config.REPORT_DIR, f"cqr_preds_{m}.csv")
    ax = axes[2]
    if os.path.exists(cqr_path):
        cq = get_window(pd.read_csv(cqr_path, index_col=0, parse_dates=True),
                        ["cqr_lower","cqr_upper","q50"])
        if "cqr_lower" in cq:
            ax.fill_between(range(len(y_win)), cq["cqr_lower"], cq["cqr_upper"],
                            alpha=0.15, color=MODEL_COLORS["CQR"], label="CQR 90% CI")
        if "q50" in cq:
            ax.plot(cq["q50"], color=MODEL_COLORS["CQR"], lw=1.5, label="Midpoint")
    ax.plot(y_win, "k-", lw=0.8, label="Actual", zorder=5)
    ax.set_xlabel("Hour in Window")
    ax.set_title(titles[2], fontweight="bold"); ax.legend(fontsize=7)

    fig.suptitle(f"{market} — Quantile Interval Comparison (1 representative week)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "F11_quantile_fan")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_figures():
    print(f"\n{'='*65}")
    print(f"  Generating Publication Figures (300 DPI)")
    print(f"  Output: {FIG_DIR}")
    print(f"{'='*65}\n")

    os.makedirs(FIG_DIR, exist_ok=True)

    figure_fns = [
        fig_price_timeseries,
        fig_accuracy_comparison,
        fig_prob_quality,
        fig_regime_heatmap,
        fig_economic_pnl,
        fig_shap_importance,
        fig_ablation,
        fig_uri_crisis,
        fig_alpha_sweep,            # F9 — alpha sweep (Ablation A5)
        fig_reliability_diagram,    # F10 — calibration curves (new)
        fig_quantile_fan,           # F11 — quantile fan chart (new)
    ]

    for fn in figure_fns:
        try:
            fn()
        except Exception as e:
            print(f"  ⚠️  {fn.__name__} failed: {e}")

    print(f"\n  ✅ All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    generate_all_figures()
