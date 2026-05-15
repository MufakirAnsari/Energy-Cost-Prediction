"""
step_18_paper_tables.py — LaTeX Table Generator
================================================
Reads all result CSVs from reports/ and outputs publication-ready
LaTeX tables using booktabs formatting.

Tables generated:
  Table 1  — Dataset statistics (price stats per regime)
  Table 2  — Point accuracy: MAE, RMSE, sMAPE + DM significance (★)
  Table 3  — Probabilistic quality: PICP, MPIW, Winkler, CRPS (RQ2)
  Table 4  — Economic utility: P&L, Sharpe, Sortino, MDD (RQ3)
  Table 5  — Cross-market: PJM→ERCOT vs ERCOT→ERCOT (RQ4)
  Table A1 — Full hyperparameter table

All output to reports/tex/ as .tex files.

Run:
    python step_18_paper_tables.py
"""
import os, sys
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

TEX_DIR = os.path.join(config.REPORT_DIR, "tex")


def save_tex(content: str, fname: str):
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, fname)
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✅ {fname}")


def escape(s):
    """Escape LaTeX special characters in strings."""
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def tex_table(df, caption, label, col_fmt=None, bold_row=None, sig_col=None, sig_df=None):
    """
    Convert DataFrame to booktabs LaTeX table.
    bold_row: index label to bold (e.g. best model)
    sig_col: column name in sig_df with boolean significance markers
    """
    ncols = len(df.columns) + 1  # +1 for index
    if col_fmt is None:
        col_fmt = "l" + "r" * len(df.columns)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{escape(caption)}}}")
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Header
    header = [escape(df.index.name or "Model")] + [escape(c) for c in df.columns]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Rows
    for idx, row in df.iterrows():
        cells = [escape(idx)]
        for col in df.columns:
            val = row[col]
            if isinstance(val, float) and np.isnan(val):
                cells.append("—")
            elif isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(escape(val))
        row_str = " & ".join(cells) + r" \\"
        if bold_row is not None and str(idx) == str(bold_row):
            row_str = r"\textbf{" + " & ".join(cells) + r"} \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Dataset Statistics
# ─────────────────────────────────────────────────────────────────────────────
def table_dataset_stats():
    path = os.path.join(config.REPORT_DIR, "table_price_statistics.csv")
    if not os.path.exists(path):
        print("  ⚠️  table_price_statistics.csv not found — run step_01_eda.py first")
        return

    df = pd.read_csv(path)
    # Pivot: rows = Regime, cols = Market metrics
    pjm   = df[df["Market"]=="PJM"].set_index("Regime")[["N_hours","Mean","Std","Max","Spike_pct>2sigma"]]
    ercot = df[df["Market"]=="ERCOT"].set_index("Regime")[["Mean","Std","Max","Spike_pct>2sigma"]]
    pjm.columns   = ["N (hrs)", "PJM Mean", "PJM Std", "PJM Max", "PJM Spike%"]
    ercot.columns = ["ERCOT Mean", "ERCOT Std", "ERCOT Max", "ERCOT Spike%"]
    combined = pjm.join(ercot)
    combined.index.name = "Regime"

    cap = ("Price statistics by market regime (\\$/MWh). "
           "Spike\\% = fraction of hours with price $>\\mu+2\\sigma$.")
    tex = tex_table(combined, cap, "dataset_stats",
                    col_fmt="l" + "r"*len(combined.columns))
    save_tex(tex, "table1_dataset_stats.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Point Accuracy (both markets, DM significance)
# ─────────────────────────────────────────────────────────────────────────────
def table_point_accuracy():
    rows_all = []
    for market in ["pjm", "ercot"]:
        acc_path = os.path.join(config.REPORT_DIR, f"table_point_accuracy_{market}.csv")
        dm_path  = os.path.join(config.REPORT_DIR, f"dm_tests_vs_best_{market}.csv")
        if not os.path.exists(acc_path):
            continue

        acc = pd.read_csv(acc_path, index_col=0)
        sig_set = set()
        if os.path.exists(dm_path):
            dm_df = pd.read_csv(dm_path, index_col=0)
            sig_set = set(dm_df[dm_df["BH_reject"]==True].index.tolist())

        best_model = acc["MAE"].idxmin()
        for model in acc.index:
            sig_mark = "$^\\star$" if model in sig_set else ""
            rows_all.append({
                "Model": model,
                f"PJM MAE{sig_mark if market=='pjm' else ''}": acc.loc[model,"MAE"] if market=="pjm" else "",
                f"PJM RMSE": acc.loc[model,"RMSE"] if market=="pjm" else "",
                f"ERCOT MAE{sig_mark if market=='ercot' else ''}": acc.loc[model,"MAE"] if market=="ercot" else "",
                f"ERCOT RMSE": acc.loc[model,"RMSE"] if market=="ercot" else "",
            })

    # Cleaner approach: build a single combined table
    combined_rows = []
    pjm_acc   = pd.read_csv(os.path.join(config.REPORT_DIR, "table_point_accuracy_pjm.csv"), index_col=0) \
                if os.path.exists(os.path.join(config.REPORT_DIR, "table_point_accuracy_pjm.csv")) else None
    ercot_acc = pd.read_csv(os.path.join(config.REPORT_DIR, "table_point_accuracy_ercot.csv"), index_col=0) \
                if os.path.exists(os.path.join(config.REPORT_DIR, "table_point_accuracy_ercot.csv")) else None

    dm_pjm   = pd.read_csv(os.path.join(config.REPORT_DIR, "dm_tests_vs_best_pjm.csv"), index_col=0) \
               if os.path.exists(os.path.join(config.REPORT_DIR, "dm_tests_vs_best_pjm.csv")) else None
    dm_ercot = pd.read_csv(os.path.join(config.REPORT_DIR, "dm_tests_vs_best_ercot.csv"), index_col=0) \
               if os.path.exists(os.path.join(config.REPORT_DIR, "dm_tests_vs_best_ercot.csv")) else None

    if pjm_acc is None:
        return

    all_models = list(pjm_acc.index) if pjm_acc is not None else []

    table_rows = {}
    for model in all_models:
        row = {}
        # PJM
        if pjm_acc is not None and model in pjm_acc.index:
            star = "$^{\\star}$" if (dm_pjm is not None and model in dm_pjm.index
                                     and dm_pjm.loc[model,"BH_reject"]) else ""
            row["PJM MAE"]  = f"{pjm_acc.loc[model,'MAE']:.4f}{star}"
            row["PJM RMSE"] = f"{pjm_acc.loc[model,'RMSE']:.4f}"
        # ERCOT
        if ercot_acc is not None and model in ercot_acc.index:
            star = "$^{\\star}$" if (dm_ercot is not None and model in dm_ercot.index
                                     and dm_ercot.loc[model,"BH_reject"]) else ""
            row["ERCOT MAE"]  = f"{ercot_acc.loc[model,'MAE']:.4f}{star}"
            row["ERCOT RMSE"] = f"{ercot_acc.loc[model,'RMSE']:.4f}"
        table_rows[model] = row

    df = pd.DataFrame(table_rows).T
    df.index.name = "Model"

    # Sort by PJM MAE
    try:
        df["_sort"] = df["PJM MAE"].str.replace(r"\$.*", "", regex=True).astype(float)
        df = df.sort_values("_sort").drop(columns=["_sort"])
    except Exception:
        pass

    best = pjm_acc["MAE"].idxmin() if pjm_acc is not None else None
    cap = ("Point forecast accuracy. MAE and RMSE in \\$/MWh. "
           "$^{\\star}$ = significantly worse than best model "
           "(Diebold--Mariano test, BH-corrected $\\alpha=5\\%$).")
    tex = tex_table(df, cap, "point_accuracy",
                    col_fmt="l" + "r"*len(df.columns), bold_row=best)
    save_tex(tex, "table2_point_accuracy.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Probabilistic Quality
# ─────────────────────────────────────────────────────────────────────────────
def table_probabilistic():
    rows = []
    for market in ["pjm", "ercot"]:
        path = os.path.join(config.REPORT_DIR, f"table_probabilistic_{market}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        df["Market"] = market.upper()
        rows.append(df)

    if not rows:
        return
    combined = pd.concat(rows).reset_index()
    combined = combined.rename(columns={"Model": "Method"})
    combined = combined.set_index(["Market", "Method"])

    # Handle both old column names (Winkler_Score, Pinball_p05/p95)
    # and new standardized names (Winkler_α10, Pinball_lo, Pinball_hi)
    col_candidates = [
        "PICP_%",
        "MPIW",
        "Winkler_α10", "Winkler_Score",   # accept either name
        "CRPS",
        "Pinball_lo", "Pinball_p05",       # accept either name
        "Pinball_hi", "Pinball_p95",       # accept either name
    ]
    cols = [c for c in col_candidates if c in combined.columns]
    # Deduplicate: if both old+new exist, keep new
    seen_roles = set()
    final_cols = []
    for c in cols:
        role = c.replace("Winkler_Score","winkler").replace("Winkler_α10","winkler") \
                 .replace("Pinball_p05","pin_lo").replace("Pinball_lo","pin_lo") \
                 .replace("Pinball_p95","pin_hi").replace("Pinball_hi","pin_hi")
        if role not in seen_roles:
            seen_roles.add(role)
            final_cols.append(c)
    combined = combined[final_cols]

    cap = ("Probabilistic forecast quality. 90\\% nominal CI (80\\% for N-HiTS-Q & Chronos-Bolt). "
           "PICP = prediction interval coverage probability (\\%). "
           "MPIW = mean prediction interval width (\\$/MWh). "
           "Winkler score at $\\alpha=0.10$ (lower = better). "
           "CRPS = continuous ranked probability score (lower = better). "
           "Pinball loss at boundary quantiles.")
    tex = tex_table(combined, cap, "probabilistic",
                    col_fmt="ll" + "r"*len(final_cols))
    save_tex(tex, "table3_probabilistic.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 4 — Economic Utility
# ─────────────────────────────────────────────────────────────────────────────
def table_economic():
    rows = []
    for market in ["pjm", "ercot"]:
        path = os.path.join(config.REPORT_DIR, f"table_economic_{market}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        df["Market"] = market.upper()
        rows.append(df)

    if not rows:
        return
    combined = pd.concat(rows).reset_index()
    combined = combined.set_index(["Market", "Strategy"])
    cols = [c for c in ["Total_PnL_$","Sharpe","Sortino","Max_Drawdown_$","Win_Rate_pct"]
            if c in combined.columns]
    combined = combined[cols]

    cap = ("Economic utility simulation. TC=\\$0.50/MWh, slippage=$0.3\\sigma$, volume=1 MWh. "
           "Oracle = perfect foresight upper bound (labeled). "
           "P\\&L and MDD in USD.")
    tex = tex_table(combined, cap, "economic", col_fmt="ll" + "r"*len(cols))
    save_tex(tex, "table4_economic.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 5 — Cross-Market Generalizability (RQ4)
# ─────────────────────────────────────────────────────────────────────────────
def table_crossmarket():
    path = os.path.join(config.REPORT_DIR, "table_rq4_crossmarket.csv")
    if not os.path.exists(path):
        print("  ⚠️  table_rq4_crossmarket.csv not found — run step_19 first")
        return

    df = pd.read_csv(path)
    df = df.set_index(["Direction","Model"])

    cap = ("Cross-market generalizability (RQ4). "
           "PJM$\\to$ERCOT: PJM-trained models evaluated on ERCOT test set "
           "using shared feature intersection. "
           "ERCOT$\\to$ERCOT: in-market baseline. "
           "MAE and RMSE in \\$/MWh.")
    tex = tex_table(df, cap, "crossmarket", col_fmt="ll" + "r"*len(df.columns))
    save_tex(tex, "table5_crossmarket.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table A1 — Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def table_hyperparams():
    params = {
        "SeasonalNaive":         {"type": "Statistical", "params": "—", "season_length": 168},
        "AutoARIMA":             {"type": "Statistical", "params": "auto", "max_p": 5, "max_q": 5},
        "MSTL":                  {"type": "Statistical", "seasonal_periods": "[24, 168]", "params": "auto"},
        "LightGBM (point)":      {"type": "Tree", "n_estimators": 1000, "lr": 0.05, "num_leaves": 127},
        "LightGBM (quantile)":   {"type": "Tree", "n_estimators": 1000, "lr": 0.05, "quantiles": "q05–q95"},
        "XGBoost":               {"type": "Tree", "n_estimators": 1000, "lr": 0.05, "max_depth": 6},
        "QRF":                   {"type": "Tree", "n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt"},
        "BiLSTM (MC Dropout)":   {"type": "RNN", "BiLSTM_units": 64, "dense": 32, "dropout": "sweep {0.1–0.4}", "seq_len": 168, "lr": "5e-4", "clipnorm": 1.0},
        "BiTCN":                 {"type": "CNN", "hidden_size": 128, "max_steps": 1500, "dropout": 0.1, "lr": "1e-3"},
        "PatchTST":              {"type": "Transformer", "hidden_size": 128, "n_heads": 4, "patch_len": 16, "max_steps": 1500, "lr": "1e-3"},
        "iTransformer":          {"type": "Transformer", "hidden_size": 128, "n_heads": 8, "max_steps": 1500, "lr": "1e-3"},
        "TFT":                   {"type": "Transformer", "hidden_size": 32, "n_heads": 2, "input_size": 72, "max_steps": 1500, "batch_size": 64, "lr": "1e-3"},
        "N-HiTS (MAE)":          {"type": "MLP", "mlp_units": "3×[256,256]", "max_steps": 1500, "dropout": 0.2, "lr": "1e-3"},
        "N-HiTS (MQLoss 80%CI)": {"type": "MLP", "mlp_units": "3×[256,256]", "max_steps": 1500, "loss": "MQLoss[80]", "lr": "1e-3"},
        "Chronos-Bolt (Small)":  {"type": "Foundation", "params": "200M", "context": 168, "horizon": 24, "fine-tune": "None"},
        "CQR":                   {"type": "Conformal", "base": "LGBM-Q", "alpha": 0.10, "cal_set": "2022"},
        "Ensemble":              {"type": "Stacked", "base": "LGBM+XGB+BiLSTM", "meta": "LightGBM", "cal_set": "2022"},
    }
    df = pd.DataFrame(params).T
    df.index.name = "Model"
    df = df.fillna("—")

    cap = ("Hyperparameter summary. All NeuralForecast models: freq=hourly, h=24, "
           "scaler=standard, seed=42. "
           "BiLSTM: ERCOT retrained with log1p+MinMax scaling and clipnorm=1.0. "
           "TFT: memory-optimized for 4\\,GB VRAM with 12 hist\\_exog features.")
    tex = tex_table(df, cap, "hyperparams", col_fmt="l" + "c"*len(df.columns))
    save_tex(tex, "tableA1_hyperparameters.tex")


if __name__ == "__main__":
    print(f"\n{'='*65}\n  Generating LaTeX Tables\n  Output: {TEX_DIR}\n{'='*65}\n")

    table_dataset_stats()
    table_point_accuracy()
    table_probabilistic()
    table_economic()
    table_crossmarket()
    table_hyperparams()

    print(f"\n  ✅ All LaTeX tables saved to: {TEX_DIR}")
    print(f"  Usage: \\input{{reports/tex/table2_point_accuracy.tex}}")
