"""
step_14_dm_tests.py
===================
Dedicated Diebold-Mariano (DM) testing with Benjamini-Hochberg (BH)
false-discovery-rate correction for multiple comparisons.

Runs ALL pairwise DM tests between models and applies BH correction
to control FDR at 5%. Produces publication-ready tables.

Reference:
  - Diebold & Mariano (1995). Comparing predictive accuracy.
  - Harvey, Leybourne & Newbold (1997). Small sample correction.
  - Benjamini & Hochberg (1995). FDR control.

Run:
    python step_14_dm_tests.py
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config


def dm_test_hln(y, pred_a, pred_b, h=1):
    """Harvey-Leybourne-Newbold small-sample DM test."""
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

        from scipy import stats
        p_val = 2 * stats.t.sf(np.abs(dm_corr), df=n-1)
        return float(dm_corr), float(p_val)
    except Exception:
        return np.nan, np.nan


def bh_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean array of rejections."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    order  = np.argsort(p_values)
    ranks  = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    bh_threshold = (ranks / n) * alpha
    reject = p_values <= bh_threshold
    # Once we find the largest k where p_(k) <= k/n * alpha, all below are rejected
    if reject.any():
        max_reject_rank = ranks[reject].max()
        reject = ranks <= max_reject_rank
    return reject


def run_dm_tests(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  Diebold-Mariano Tests + BH Correction: {market}")
    print(f"{'='*65}")

    m = market.lower()
    test_df = pd.read_parquet(
        config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
    )
    y_true  = test_df[config.TARGET_COL].values
    y_index = test_df.index

    def load_col(fname, col):
        path = os.path.join(config.REPORT_DIR, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[col].reindex(y_index).values if col in df.columns else None

    # Load all predictions
    predictions = {}

    for model_name in ["SeasonalNaive", "AutoARIMA", "MSTL"]:
        col = load_col(f"baseline_preds_{m}_test.csv", model_name)
        if col is not None:
            predictions[model_name] = col

    for path_name, col_name in [
        (f"lgbm_point_{m}.joblib",      "LightGBM"),
        (f"xgboost_point_{m}.joblib",   "XGBoost"),
    ]:
        full_path = os.path.join(config.MODEL_DIR, path_name)
        if os.path.exists(full_path):
            X_test = test_df.drop(columns=[config.TARGET_COL])
            predictions[col_name] = joblib.load(full_path).predict(X_test)

    def _load_dl_pred(csv_path, col_name, ref_index):
        """Load a DL/foundation model prediction CSV, handle duplicates + timezone."""
        df_tmp = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df_tmp.index = pd.to_datetime(df_tmp.index, utc=True)
        df_tmp = df_tmp[~df_tmp.index.duplicated(keep="first")]
        if col_name not in df_tmp.columns:
            return None
        idx_utc = (ref_index.tz_localize("UTC") if ref_index.tz is None
                   else ref_index.tz_convert("UTC"))
        return df_tmp[col_name].reindex(idx_utc).values

    for model_key, csv_name, col_name in [
        ("PatchTST",     f"patchtst_preds_{m}.csv",     "predicted"),
        ("iTransformer", f"itransformer_preds_{m}.csv",  "predicted"),
        ("N-HiTS",       f"nhits_preds_{m}.csv",         "predicted"),
        ("BiTCN",        f"bitcn_preds_{m}.csv",         "predicted"),
        ("TFT",          f"tft_preds_{m}.csv",           "predicted"),
        ("Chronos-Bolt", f"chronos_preds_{m}.csv",       "chronos_point"),
        ("QRF",          f"qrf_preds_{m}.csv",           "q50"),
        ("BiLSTM",       f"bilstm_preds_{m}.csv",        None),   # auto-detect
        ("Ensemble",     f"ensemble_preds_{m}.csv",      "ensemble"),
    ]:
        csv_path = os.path.join(config.REPORT_DIR, csv_name)
        if not os.path.exists(csv_path):
            continue
        # BiLSTM: auto-detect point prediction column
        if col_name is None:
            df_t = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            col_name = next((c for c in df_t.columns
                             if "pred" in c.lower() or "mean" in c.lower()), None)
            if col_name is None and len(df_t.columns):
                col_name = df_t.columns[0]
        vals = _load_dl_pred(csv_path, col_name, y_index)
        if vals is not None and not np.all(np.isnan(vals)):
            predictions[model_key] = vals

    model_names = list(predictions.keys())
    n_models = len(model_names)
    print(f"  Models: {model_names}")

    if n_models < 2:
        print("  ⚠️  Need at least 2 models. Run earlier steps first.")
        return

    # ── All-pairs DM tests ─────────────────────────────────────────
    records = []
    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if i >= j:
                continue
            pred_a = predictions[name_a]
            pred_b = predictions[name_b]
            mask   = ~np.isnan(y_true) & ~np.isnan(pred_a) & ~np.isnan(pred_b)
            if mask.sum() < 50:
                continue
            stat, pval = dm_test_hln(y_true[mask], pred_a[mask], pred_b[mask])
            records.append({
                "Model_A":  name_a,
                "Model_B":  name_b,
                "DM_stat":  round(stat, 4) if not np.isnan(stat) else np.nan,
                "p_value":  round(pval, 6) if not np.isnan(pval) else np.nan,
            })

    dm_df = pd.DataFrame(records)

    # BH correction on all p-values
    valid = ~dm_df["p_value"].isna()
    p_vals = dm_df.loc[valid, "p_value"].values
    reject = bh_correction(p_vals, alpha=0.05)
    dm_df.loc[valid, "BH_reject_5pct"] = reject
    dm_df["BH_reject_5pct"] = dm_df["BH_reject_5pct"].fillna(False).astype(bool)
    dm_df["significant"] = dm_df["p_value"] < 0.05

    print(f"\n  Pairwise DM tests ({len(dm_df)} pairs):")
    print(dm_df.to_string(index=False))

    sig_count = dm_df["BH_reject_5pct"].sum()
    print(f"\n  Significant after BH correction: {sig_count} / {len(dm_df)}")

    # ── vs. Best model matrix ──────────────────────────────────────
    maes = {name: np.mean(np.abs(y_true[~np.isnan(y_true) & ~np.isnan(pred)] -
                                  pred[~np.isnan(y_true) & ~np.isnan(pred)]))
            for name, pred in predictions.items()}
    best = min(maes, key=maes.get)
    print(f"\n  Best model by MAE: {best} ({maes[best]:.4f})")

    vs_best = []
    for name, pred in predictions.items():
        if name == best:
            continue
        mask = ~np.isnan(y_true) & ~np.isnan(pred) & ~np.isnan(predictions[best])
        stat, pval = dm_test_hln(y_true[mask], pred[mask], predictions[best][mask])
        vs_best.append({
            "Model":    name,
            "MAE":      round(maes[name], 4),
            "DM_stat":  round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value":  round(pval, 6) if not np.isnan(pval) else np.nan,
            "sig_p05":  pval < 0.05 if not np.isnan(pval) else False,
        })

    vs_df = pd.DataFrame(vs_best).set_index("Model")
    vs_df["BH_reject"] = bh_correction(
        vs_df["p_value"].fillna(1.0).values, alpha=0.05
    )
    print(f"\n  All models vs. {best}:")
    print(vs_df.to_string())

    # Save
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    dm_df.to_csv(os.path.join(config.REPORT_DIR, f"dm_tests_pairwise_{m}.csv"), index=False)
    vs_df.to_csv(os.path.join(config.REPORT_DIR, f"dm_tests_vs_best_{m}.csv"))
    print(f"\n  ✅ Saved: dm_tests_pairwise_{m}.csv")
    print(f"  ✅ Saved: dm_tests_vs_best_{m}.csv")
    return dm_df, vs_df


if __name__ == "__main__":
    run_dm_tests("PJM")
    run_dm_tests("ERCOT")
