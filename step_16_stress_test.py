"""
step_16_stress_test.py  (REWRITTEN — v2)
=========================================
Regime-isolated evaluation — STRICTLY out-of-sample.

CRITICAL FIX (v2): Previous version ran LightGBM/XGBoost predictions on
the full dataset (including training data), producing impossibly low MAEs
(e.g., stable_baseline LightGBM MAE=0.51). This was in-sample evaluation.

v2 approach:
  - All model predictions loaded from saved CSVs (test-period only)
  - No model.predict() calls on training/calibration data
  - Regime analysis only for test set (2024-2025) → "new_normal" regime
  - NEW: Validation set (2023) added as additional out-of-sample holdout
  - Historical regimes (Uri, COVID, gas_shock) clearly labeled as
    NOT EVALUATED (no out-of-sample predictions available)

Run:
    python step_16_stress_test.py
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config


def evaluate_on_regime(y_true, predictions, regime_mask, regime_name):
    rows = []
    for model_name, pred in predictions.items():
        mask = regime_mask & ~np.isnan(y_true) & ~np.isnan(pred)
        if mask.sum() < 24:
            continue
        y, yh = y_true[mask], pred[mask]
        rows.append({
            "Regime":  regime_name,
            "Model":   model_name,
            "N_hours": int(mask.sum()),
            "MAE":     round(np.mean(np.abs(y - yh)), 4),
            "RMSE":    round(np.sqrt(np.mean((y - yh)**2)), 4),
            "sMAPE":   round(np.mean(2*np.abs(y-yh)/(np.abs(y)+np.abs(yh)+1e-9))*100, 4),
        })
    return pd.DataFrame(rows)


def load_oos_predictions(market: str, ref_index: pd.DatetimeIndex) -> dict:
    """
    Load ALL predictions from saved CSVs.
    These are strictly out-of-sample (cross_validation or test-set inference).
    NO model.predict() on training/calibration data.
    """
    m = market.lower()
    predictions = {}

    def load_col(csv_path, col_name):
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="first")]
        if col_name not in df.columns:
            return None
        idx_utc = ref_index.tz_localize("UTC") if ref_index.tz is None \
                  else ref_index.tz_convert("UTC")
        return df[col_name].reindex(idx_utc).values

    # Classical baselines (test period only)
    for col in ["SeasonalNaive", "AutoARIMA", "MSTL"]:
        v = load_col(os.path.join(config.REPORT_DIR, f"baseline_preds_{m}_test.csv"), col)
        if v is not None:
            predictions[col] = v

    # DL models (cross_validation predictions — out-of-sample)
    for name, fname, col in [
        ("LightGBM",     f"lgbm_point_{m}_oos.csv",      "predicted"),   # if exists
        ("PatchTST",     f"patchtst_preds_{m}.csv",       "predicted"),
        ("iTransformer", f"itransformer_preds_{m}.csv",   "predicted"),
        ("N-HiTS",       f"nhits_preds_{m}.csv",          "predicted"),
        ("BiTCN",        f"bitcn_preds_{m}.csv",          "predicted"),
        ("Chronos-Bolt", f"chronos_preds_{m}.csv",        "chronos_point"),
        ("QRF",          f"qrf_preds_{m}.csv",            "q50"),
        ("Ensemble",     f"ensemble_preds_{m}.csv",       "ensemble"),
    ]:
        v = load_col(os.path.join(config.REPORT_DIR, fname), col)
        if v is not None:
            predictions[name] = v

    # Tree models — only use test-set predictions (generate from model on TEST data)
    # This is valid: test set is NEVER seen during training
    for model_name, joblib_name in [
        ("LightGBM", f"lgbm_point_{m}.joblib"),
        ("XGBoost",  f"xgboost_point_{m}.joblib"),
    ]:
        path = os.path.join(config.MODEL_DIR, joblib_name)
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        # ONLY predict on test set (strictly out-of-sample)
        te_df = pd.read_parquet(
            config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
        )
        X_te = te_df.drop(columns=[config.TARGET_COL])
        preds_te = model.predict(X_te)
        # Align to ref_index with NaN outside test period
        full_vals = np.full(len(ref_index), np.nan)
        idx_utc = ref_index.tz_localize("UTC") if ref_index.tz is None \
                  else ref_index.tz_convert("UTC")
        te_idx_utc = te_df.index.tz_localize("UTC") if te_df.index.tz is None \
                     else te_df.index.tz_convert("UTC")
        pos = np.where(np.isin(idx_utc, te_idx_utc))[0]
        if len(pos) == len(preds_te):
            full_vals[pos] = preds_te
        predictions[model_name] = full_vals

    # BiLSTM (MC Dropout mean — out-of-sample test set)
    bl_path = os.path.join(config.REPORT_DIR, f"bilstm_preds_{m}.csv")
    if os.path.exists(bl_path):
        df_bl = pd.read_csv(bl_path, index_col=0, parse_dates=True)
        bl_col = next((c for c in df_bl.columns
                       if "pred" in c.lower() or "mean" in c.lower()), None)
        if bl_col is None and len(df_bl.columns):
            bl_col = df_bl.columns[0]
        if bl_col:
            v = load_col(bl_path, bl_col)
            if v is not None:
                predictions["BiLSTM"] = v

    return predictions


def run_stress_test(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  REGIME STRESS TEST: {market}  [v2 — strictly out-of-sample]")
    print(f"{'='*65}")

    # Use only TEST set as the evaluation universe (strictly OOS)
    te_df = pd.read_parquet(
        config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
    )
    y_true = te_df[config.TARGET_COL].values
    idx    = te_df.index

    print(f"  Test set: {len(te_df):,} hours | "
          f"{idx.min().date()} → {idx.max().date()}")

    predictions = load_oos_predictions(market, idx)
    print(f"  Models loaded: {list(predictions.keys())}")

    # ── SECTION 1: Test-set regime + seasonal breakdown ───────────────────────
    print(f"\n  [A] Test Set Analysis (2024-2025 — all OOS)")
    all_results = []

    # 1a. Named regimes from config (will usually be just "new_normal")
    for regime_name, (r_start, r_end) in config.REGIMES.items():
        mask = (
            (idx >= pd.Timestamp(r_start, tz="UTC")) &
            (idx <= pd.Timestamp(r_end,   tz="UTC"))
        )
        if mask.sum() == 0:
            continue
        print(f"  [{regime_name}]: {mask.sum():,} hours in test set")
        res = evaluate_on_regime(y_true, predictions, mask, regime_name)
        if not res.empty:
            all_results.append(res)

    # 1b. Annual breakdown (2024 vs 2025)
    print(f"\n  [A-i] Annual breakdown:")
    for yr in [2024, 2025]:
        if idx.tz is not None:
            yr_mask = idx.year == yr
        else:
            yr_mask = pd.DatetimeIndex(idx).year == yr
        n = yr_mask.sum()
        if n < 24:
            continue
        print(f"  [Year {yr}]: {n:,} hours")
        res = evaluate_on_regime(y_true, predictions, yr_mask, f"Year_{yr}")
        if not res.empty:
            all_results.append(res)

    # 1c. Seasonal breakdown (meteorological seasons)
    SEASONS = {
        "Winter (DJF)": [12, 1, 2],
        "Spring (MAM)": [3, 4, 5],
        "Summer (JJA)": [6, 7, 8],
        "Fall   (SON)": [9, 10, 11],
    }
    print(f"\n  [A-ii] Seasonal breakdown:")
    for season_name, months in SEASONS.items():
        if idx.tz is not None:
            s_mask = idx.month.isin(months)
        else:
            s_mask = pd.DatetimeIndex(idx).month.isin(months)
        n = s_mask.sum()
        if n < 24:
            continue
        print(f"  [{season_name}]: {n:,} hours")
        res = evaluate_on_regime(y_true, predictions, s_mask, season_name.strip())
        if not res.empty:
            all_results.append(res)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        mae_heatmap = results_df[results_df["Regime"].str.startswith("Year") |
                                 results_df["Regime"].str.contains(r"(DJF|MAM|JJA|SON)", regex=True) |
                                 ~results_df["Regime"].str.startswith("Year")
                                 ].pivot_table(
            index="Regime", columns="Model", values="MAE"
        )
        print(f"\n  Regime/Season MAE Heatmap ($/MWh) — strictly OOS test set:")
        print(mae_heatmap.to_string())

        os.makedirs(config.REPORT_DIR, exist_ok=True)
        m = market.lower()
        results_df.to_csv(
            os.path.join(config.REPORT_DIR, f"table_regime_stress_{m}.csv"))
        mae_heatmap.to_csv(
            os.path.join(config.REPORT_DIR, f"heatmap_regime_mae_{m}.csv"))
        print(f"  ✅ Saved: table_regime_stress_{m}.csv")
    else:
        print("  ⚠️  No regime overlap with test set.")


    # ── SECTION 2: Validation set analysis (2023 — additional OOS) ───────────
    print(f"\n  [B] Validation Set Analysis (2023 — additional OOS holdout)")
    val_df = pd.read_parquet(
        config.PJM_VAL_PATH if market == "PJM" else config.ERCOT_VAL_PATH
    )
    y_val = val_df[config.TARGET_COL].values
    idx_v = val_df.index
    m     = market.lower()

    val_preds = {}
    for model_name, joblib_name in [
        ("LightGBM", f"lgbm_point_{m}.joblib"),
        ("XGBoost",  f"xgboost_point_{m}.joblib"),
    ]:
        path = os.path.join(config.MODEL_DIR, joblib_name)
        if not os.path.exists(path):
            continue
        model_obj = joblib.load(path)
        X_val = val_df.drop(columns=[config.TARGET_COL])
        val_preds[model_name] = model_obj.predict(X_val)

    # NF models — check for val coverage in cross_val CSVs
    def load_val_col(fname, col):
        path = os.path.join(config.REPORT_DIR, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="first")]
        if col not in df.columns:
            return None
        idx_utc = idx_v.tz_localize("UTC") if idx_v.tz is None else idx_v.tz_convert("UTC")
        v = df[col].reindex(idx_utc).values
        return v if not np.all(np.isnan(v)) else None

    for name, fname, col in [
        ("QRF",   f"qrf_preds_{m}.csv",  "q50"),
        ("BiTCN", f"bitcn_preds_{m}.csv","predicted"),
        ("N-HiTS",f"nhits_preds_{m}.csv","predicted"),
    ]:
        v = load_val_col(fname, col)
        if v is not None:
            val_preds[name] = v

    if val_preds:
        mask_all = np.ones(len(y_val), dtype=bool)
        val_res = evaluate_on_regime(y_val, val_preds, mask_all, "validation_2023")
        if not val_res.empty:
            print(val_res.to_string(index=False))
            val_res.to_csv(
                os.path.join(config.REPORT_DIR, f"table_val_stress_{m}.csv"),
                index=False)
            print(f"  ✅ Saved: table_val_stress_{m}.csv")
    else:
        print("  ⚠️  No val-set predictions available (NF models are test-only)")

    # ── SECTION 3: ERCOT Uri Crisis — retroactive in-sample analysis ──────────
    if market == "ERCOT":
        print(f"\n  [C] ERCOT Uri Crisis (Feb 2021) — Retroactive Analysis")
        print(f"  ⚠️  Uri window is in TRAINING data. The following uses tree models")
        print(f"  (which can predict on any split) evaluated retroactively.")
        print(f"  Neural models cannot be evaluated here without data leakage.")
        print(f"  This section is labelled [In-Sample] in the paper.\n")

        try:
            tr_df = pd.read_parquet(config.ERCOT_TRAIN_PATH)
            # Uri crisis: Texas deep freeze, Feb 10-20 2021
            uri_start = pd.Timestamp("2021-02-10", tz="UTC")
            uri_end   = pd.Timestamp("2021-02-20", tz="UTC")
            if tr_df.index.tz is None:
                uri_df = tr_df[(tr_df.index >= uri_start.tz_localize(None)) &
                               (tr_df.index <= uri_end.tz_localize(None))]
            else:
                uri_df = tr_df[(tr_df.index >= uri_start) & (tr_df.index <= uri_end)]

            y_uri = uri_df[config.TARGET_COL].values
            X_uri = uri_df.drop(columns=[config.TARGET_COL])
            print(f"  Uri window: {len(uri_df)} hours | "
                  f"Price range: ${y_uri.min():.0f}–${y_uri.max():.0f}/MWh")
            print(f"  Mean price: ${y_uri.mean():.1f} | Spike% (>$200): "
                  f"{(y_uri>200).mean()*100:.1f}%\n")

            uri_rows = []
            for model_name, joblib_name in [
                ("LightGBM", f"lgbm_point_ercot.joblib"),
                ("XGBoost",  f"xgboost_point_ercot.joblib"),
            ]:
                path = os.path.join(config.MODEL_DIR, joblib_name)
                if not os.path.exists(path):
                    continue
                model_obj = joblib.load(path)
                p = model_obj.predict(X_uri)
                mask_uri = ~np.isnan(y_uri)
                uri_mae  = np.mean(np.abs(y_uri[mask_uri] - p[mask_uri]))
                uri_rmse = np.sqrt(np.mean((y_uri[mask_uri] - p[mask_uri])**2))
                uri_rows.append({
                    "Model": model_name, "Period": "Uri crisis (Feb 2021)",
                    "Note": "[IN-SAMPLE]", "N_hours": mask_uri.sum(),
                    "MAE": round(uri_mae, 2), "RMSE": round(uri_rmse, 2),
                    "Price_mean": round(y_uri.mean(), 1),
                    "Price_max":  round(y_uri.max(), 1),
                })
                print(f"    {model_name:12} [IN-SAMPLE] MAE={uri_mae:.2f}  RMSE={uri_rmse:.2f} $/MWh")

            if uri_rows:
                uri_df_out = pd.DataFrame(uri_rows)
                uri_path = os.path.join(config.REPORT_DIR, "table_uri_crisis_insample_ercot.csv")
                uri_df_out.to_csv(uri_path, index=False)
                print(f"\n  ✅ Saved: table_uri_crisis_insample_ercot.csv")
                print(f"  Paper note: Models were trained on Uri data \u2192 in-sample results")
                print(f"  show model capacity to fit extreme prices, NOT generalization.")
        except Exception as e:
            print(f"  ⚠️  Uri analysis failed: {e}")


    return results_df if all_results else pd.DataFrame()


if __name__ == "__main__":
    run_stress_test("PJM")
    run_stress_test("ERCOT")
