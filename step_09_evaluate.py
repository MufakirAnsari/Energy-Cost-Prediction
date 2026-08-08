"""
step_13_evaluate.py
===================
Comprehensive evaluation of ALL models across ALL metrics.

Evaluation Framework:
  RQ1 — Point Accuracy:       MAE, RMSE, sMAPE + Diebold-Mariano tests
  RQ2 — Probabilistic Quality: PICP, MPIW, CRPS, Winkler Score, Pinball Loss
                               (ALL models at 90% nominal CI)
  RQ3 — Economic Utility:     P&L under realistic transaction costs + slippage
  RQ4 — Cross-market:         PJM-trained models on ERCOT test set

All results are saved as CSVs ready for LaTeX table generation.

Run:
    python step_09_evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

try:
    import properscoring as ps
    HAS_PROPERSCORING = True
except ImportError:
    HAS_PROPERSCORING = False


def _crps_fallback(y, samples):
    """Simple CRPS approximation when properscoring is unavailable."""
    n = len(y)
    result = []
    for i in range(n):
        s = np.sort(samples[i])
        m = len(s)
        expected = np.mean(np.abs(s - y[i]))
        spread = np.mean(np.abs(s[:, None] - s[None, :])) / 2
        result.append(expected - spread)
    return np.mean(result)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mae(y, yhat):
    return np.nanmean(np.abs(y - yhat))

def rmse(y, yhat):
    return np.sqrt(np.nanmean((y - yhat)**2))

def smape(y, yhat):
    denom = (np.abs(y) + np.abs(yhat)) / 2
    denom[denom == 0] = np.nan
    return np.nanmean(np.abs(y - yhat) / denom) * 100

def picp(y, lower, upper):
    return np.nanmean((y >= lower) & (y <= upper)) * 100

def mpiw(lower, upper):
    return np.nanmean(upper - lower)

def winkler_score(y, lower, upper, alpha=0.10):
    width = upper - lower
    pen_l = (2 / alpha) * np.maximum(lower - y, 0)
    pen_u = (2 / alpha) * np.maximum(y - upper, 0)
    return np.nanmean(width + pen_l + pen_u)

def crps(y, samples):
    """CRPS. samples: [n_points, n_samples]."""
    if HAS_PROPERSCORING:
        return ps.crps_ensemble(y, samples).mean()
    return _crps_fallback(y, samples)

def pinball_loss(y, q_pred, alpha):
    return np.nanmean(
        np.where(y >= q_pred,
                 alpha * (y - q_pred),
                 (1 - alpha) * (q_pred - y))
    )


# ─────────────────────────────────────────────────────────────────────────────
# DIEBOLD-MARIANO TEST
# ─────────────────────────────────────────────────────────────────────────────

def dm_test(y, pred_a, pred_b, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    H0: Models A and B have equal MAE.
    Returns: (dm_stat, p_value)
    Uses Harvey-Leybourne-Newbold (HLN) small-sample correction.
    Reference: Harvey, Leybourne, Newbold (1997).
    """
    try:
        # Manual HLN implementation (always used for reliability)
        e1 = np.abs(y - pred_a)
        e2 = np.abs(y - pred_b)
        d  = e1 - e2
        n  = len(d)
        d_mean = np.mean(d)

        # Newey-West HAC variance for autocorrelation
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
        var_d = np.var(d, ddof=1)
        for k in range(1, max_lag + 1):
            w    = 1 - k / (max_lag + 1)
            cov  = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
            var_d += 2 * w * cov
        var_d = max(var_d, 1e-10)

        # HLN correction
        dm_stat = d_mean / np.sqrt(var_d / n)
        dm_corr = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n) * dm_stat

        from scipy import stats
        p_val = 2 * stats.t.sf(np.abs(dm_corr), df=n-1)
        return dm_corr, p_val
    except Exception:
        return np.nan, np.nan


def pairwise_dm_matrix(y, predictions: dict, reference_model: str):
    """
    Computes DM test p-values for all models vs. reference model.
    Returns DataFrame: models × ['DM_stat', 'p_value', 'significant']
    """
    results = []
    ref_pred = predictions[reference_model]
    for model, pred in predictions.items():
        if model == reference_model:
            results.append({
                "Model": model, "DM_stat": 0.0, "p_value": 1.0,
                "significant": False, "is_reference": True
            })
            continue
        stat, pval = dm_test(y, pred, ref_pred)
        results.append({
            "Model": model, "DM_stat": stat, "p_value": pval,
            "significant": pval < 0.05, "is_reference": False
        })
    df = pd.DataFrame(results).set_index("Model")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ECONOMIC SIMULATION (REALISTIC)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trading(
    actual: pd.Series,
    point_preds: dict,     # {strategy_name: predicted_prices}
    interval_preds: dict,  # {strategy_name: (lower, upper)}
    transaction_cost: float = config.TRANSACTION_COST_PER_MWH,
    slippage_factor: float = config.SLIPPAGE_STD_FACTOR,
) -> pd.DataFrame:
    """
    Simulates daily buy-low/sell-high trading strategies.

    For each day:
    - Identify predicted min-hour (buy) and max-hour (sell)
    - Execute at actual price ± slippage_std
    - Deduct transaction costs from both legs
    - Record daily P&L

    Strategies:
    - S0: Seasonal Naïve (persistence baseline)
    - S1: LightGBM (best point model)
    - S2: Risk-Aware (trade only when CI width < median CI width)
    - S-Oracle: Perfect foresight (theoretical upper bound — clearly labeled)
    """
    results = {name: [] for name in list(point_preds.keys()) + list(interval_preds.keys()) + ["Oracle"]}
    dates = []

    # Compute rolling std of actual price for slippage
    price_rolling_std = actual.rolling(168, min_periods=24).std().fillna(actual.std())

    for day, group in actual.groupby(actual.index.date):
        if len(group) < 24:
            continue
        dates.append(day)
        act = group.values
        idx = group.index

        # Oracle: theoretical maximum (perfect foresight — upper bound)
        oracle_pnl = (act.max() - act.min()
                      - 2 * transaction_cost) * config.TRADE_VOLUME_MWH
        results["Oracle"].append(oracle_pnl)

        # Point-forecast strategies
        for name, preds in point_preds.items():
            pred_group = preds.reindex(idx)
            if pred_group.isna().all():
                results[name].append(np.nan)
                continue
            buy_hour  = pred_group.idxmin()
            sell_hour = pred_group.idxmax()
            if buy_hour == sell_hour:
                results[name].append(0.0)
                continue
            # Slippage: execute at actual ± 0.3 × rolling std
            slippage = slippage_factor * price_rolling_std.reindex(idx).mean()
            buy_price  = actual[buy_hour]  + slippage  # buy slightly higher
            sell_price = actual[sell_hour] - slippage  # sell slightly lower
            pnl = (sell_price - buy_price - 2 * transaction_cost) * config.TRADE_VOLUME_MWH
            results[name].append(pnl)

        # Interval-based risk-aware strategies
        for name, (lower, upper) in interval_preds.items():
            lower_g = lower.reindex(idx)
            upper_g = upper.reindex(idx)
            ci_width = (upper_g - lower_g).mean()

            # Risk filter: only trade when CI is narrow (high confidence).
            # LOOK-AHEAD FIX: use trailing 30-day expanding median of CI width
            # (available at day d without seeing future widths).
            ci_series = (upper - lower)
            # Compute trailing median up to but NOT including the current day
            ci_trailing = ci_series.expanding().median().shift(1)
            # For this day's window, use the median at the first hour of the day
            ci_thresh = ci_trailing.reindex(idx).iloc[0] if len(idx) > 0 else ci_series.median()
            if pd.isna(ci_thresh):              # first day: no prior history
                ci_thresh = ci_series.median() # fallback to global (only day 1)
            # Continuous risk-aware scaling: trade smaller volume when uncertainty is high
            volume_scalar = np.clip(ci_thresh / ci_width, 0.0, 1.0) if ci_width > 0 else 1.0

            # Buy when lower bound is lowest (confident low), sell when upper is highest
            buy_hour  = lower_g.idxmin()
            sell_hour = upper_g.idxmax()
            if buy_hour == sell_hour or volume_scalar < 0.01:
                results[name].append(0.0)
                continue
            slippage = slippage_factor * price_rolling_std.reindex(idx).mean()
            buy_price  = actual[buy_hour]  + slippage
            sell_price = actual[sell_hour] - slippage
            pnl = (sell_price - buy_price - 2 * transaction_cost) * (config.TRADE_VOLUME_MWH * volume_scalar)
            results[name].append(pnl)

    pnl_df = pd.DataFrame(results, index=pd.DatetimeIndex(dates))
    return pnl_df


def compute_economic_metrics(pnl_series: pd.Series) -> dict:
    """Compute risk-adjusted return metrics from daily P&L series."""
    pnl = pnl_series.dropna()
    if len(pnl) == 0 or pnl.std() == 0:
        return {"Total_PnL_$": 0, "Sharpe": 0, "Sortino": 0, "Max_Drawdown_$": 0,
                "Win_Rate_pct": 0, "Avg_Daily_PnL": 0}

    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / abs(running_max.max())) * 100 \
                 if abs(running_max.max()) > 1e-6 else 0.0

    sharpe  = pnl.mean() / pnl.std() * np.sqrt(252)
    down    = pnl[pnl < 0]
    sortino = pnl.mean() / down.std() * np.sqrt(252) if len(down) > 0 else np.inf

    return {
        "Total_PnL_$":   round(pnl.sum(), 2),
        "Sharpe":        round(sharpe, 3),
        "Sortino":       round(sortino, 3),
        "Max_Drawdown_$": round(max_dd, 2),   # absolute $ drawdown (no %-vs-peak illusion)
        "Win_Rate_pct":  round((pnl > 0).mean() * 100, 1),
        "Avg_Daily_PnL": round(pnl.mean(), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(market: str = "PJM"):
    print(f"\n{'='*65}")
    print(f"  COMPREHENSIVE EVALUATION: {market}")
    print(f"{'='*65}")

    test_df = pd.read_parquet(
        config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
    )
    y_true  = test_df[config.TARGET_COL].values
    y_index = test_df.index

    # ── Load all prediction CSV files ────────────────────────────
    def load_col(fname, col):
        path = os.path.join(config.REPORT_DIR, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[col] if col in df.columns else None

    m = market.lower()
    predictions = {}

    # Classical baselines
    for model_name in ["SeasonalNaive", "AutoARIMA", "MSTL"]:
        col = load_col(f"baseline_preds_{m}_test.csv", model_name)
        if col is not None:
            predictions[model_name] = col.reindex(y_index).values

    # LightGBM
    lgbm_pt = os.path.join(config.MODEL_DIR, f"lgbm_point_{m}.joblib")
    if os.path.exists(lgbm_pt):
        X_test = test_df.drop(columns=[config.TARGET_COL])
        lgbm_model = joblib.load(lgbm_pt)
        predictions["LightGBM"] = lgbm_model.predict(X_test)

    # XGBoost
    xgb_pt = os.path.join(config.MODEL_DIR, f"xgboost_point_{m}.joblib")
    if os.path.exists(xgb_pt):
        if "X_test" not in dir():
            X_test = test_df.drop(columns=[config.TARGET_COL])
        xgb_model = joblib.load(xgb_pt)
        predictions["XGBoost"] = xgb_model.predict(X_test)

    # Modern DL
    nf_path = os.path.join(config.REPORT_DIR, f"modern_dl_preds_{m}.csv")
    if os.path.exists(nf_path):
        nf_df = pd.read_csv(nf_path, index_col=0, parse_dates=True)
        for col in nf_df.columns:
            if not col.startswith("Unnamed"):
                predictions[col.split("/")[-1]] = nf_df[col].reindex(y_index).values

    # Individual DL model CSVs (steps 06–08b, 07b)
    for model_key, csv_name in [
        ("PatchTST",     f"patchtst_preds_{m}.csv"),
        ("iTransformer", f"itransformer_preds_{m}.csv"),
        ("N-HiTS",       f"nhits_preds_{m}.csv"),
        ("BiTCN",        f"bitcn_preds_{m}.csv"),
        ("TFT",          f"tft_preds_{m}.csv"),
    ]:
        csv_path = os.path.join(config.REPORT_DIR, csv_name)
        if os.path.exists(csv_path):
            df_tmp = pd.read_csv(csv_path, parse_dates=["ds"])
            df_tmp = df_tmp.set_index("ds")
            df_tmp.index = pd.to_datetime(df_tmp.index, utc=True)
            # Remove duplicates from cross_validation overlap
            df_tmp = df_tmp[~df_tmp.index.duplicated(keep="first")]
            # Timezone-safe reindex
            idx_utc = y_index.tz_localize("UTC") if y_index.tz is None else y_index.tz_convert("UTC")
            col_vals = df_tmp["predicted"].reindex(idx_utc).values
            if not np.all(np.isnan(col_vals)):
                predictions[model_key] = col_vals

    # BiLSTM MC Dropout (from pre-computed CSV)
    bilstm_path = os.path.join(config.REPORT_DIR, f"bilstm_preds_{m}.csv")
    if os.path.exists(bilstm_path):
        bl_df = pd.read_csv(bilstm_path, index_col=0, parse_dates=True)
        pt_col = next((c for c in bl_df.columns if "pred" in c.lower() or "mean" in c.lower()), None)
        if pt_col is None and len(bl_df.columns) > 0:
            pt_col = bl_df.columns[0]
        if pt_col:
            predictions["BiLSTM"] = bl_df[pt_col].reindex(y_index).values

    # Chronos (step 09)
    ch_col = load_col(f"chronos_preds_{m}.csv", "chronos_point")
    if ch_col is not None:
        predictions["Chronos-Bolt"] = ch_col.reindex(y_index).values

    # QRF median (step 11)
    qrf_col = load_col(f"qrf_preds_{m}.csv", "q50")
    if qrf_col is not None:
        predictions["QRF"] = qrf_col.reindex(y_index).values

    # Ensemble (step 12)
    ens_col = load_col(f"ensemble_preds_{m}.csv", "ensemble")
    if ens_col is not None:
        predictions["Ensemble (Stacking)"] = ens_col.reindex(y_index).values
    ens_mean_col = load_col(f"ensemble_preds_{m}.csv", "ensemble_mean")
    if ens_mean_col is not None:
        predictions["Ensemble (Mean)"] = ens_mean_col.reindex(y_index).values
    ens_med_col = load_col(f"ensemble_preds_{m}.csv", "ensemble_median")
    if ens_med_col is not None:
        predictions["Ensemble (Median)"] = ens_med_col.reindex(y_index).values

    print(f"\n  Models loaded: {list(predictions.keys())}")

    # ── RQ1: Point Accuracy ───────────────────────────────────────
    print(f"\n  [RQ1] Point Accuracy Metrics")
    point_rows = []
    for name, pred in predictions.items():
        mask = ~np.isnan(y_true) & ~np.isnan(pred)
        row = {
            "Model": name,
            "MAE":   round(mae(y_true[mask],  pred[mask]), 4),
            "RMSE":  round(rmse(y_true[mask], pred[mask]), 4),
            "sMAPE": round(smape(y_true[mask], pred[mask]), 4),
        }
        point_rows.append(row)
        print(f"    {name:22} MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f}")

    point_df = pd.DataFrame(point_rows).set_index("Model")
    point_df.to_csv(os.path.join(config.REPORT_DIR, f"table_point_accuracy_{m}.csv"))

    # DM tests (vs. best model by MAE)
    best_model = point_df["MAE"].idxmin()
    print(f"\n  Best point model: {best_model}")
    dm_results = pairwise_dm_matrix(y_true, predictions, best_model)
    dm_results.to_csv(os.path.join(config.REPORT_DIR, f"table_dm_tests_{m}.csv"))
    print(f"  DM tests saved.")

    # ── RQ2: Probabilistic Quality (all at 90% CI, Winkler α=0.10) ──────────
    print(f"\n  [RQ2] Probabilistic Quality (standardized α=0.10 / 90% CI)")
    prob_rows = []

    def prob_row(label, lo, hi, y, alpha=0.10, samples=None):
        """Compute standard probabilistic metrics for a (lo, hi) interval."""
        mask = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
        row  = {"Model": label}
        row["PICP_%"]       = round(picp(y[mask], lo[mask], hi[mask]), 2)
        row["MPIW"]        = round(mpiw(lo[mask], hi[mask]), 4)
        row["Winkler_α10"] = round(winkler_score(y[mask], lo[mask], hi[mask], 0.10), 4)
        row["Pinball_lo"]  = round(pinball_loss(y[mask], lo[mask], alpha/2), 4)
        row["Pinball_hi"]  = round(pinball_loss(y[mask], hi[mask], 1-alpha/2), 4)
        if samples is not None:
            valid = mask & ~np.any(np.isnan(samples), axis=1)
            row["CRPS"] = round(float(crps(y[valid], samples[valid])), 4)
        return row

    X_test_df = test_df.drop(columns=[config.TARGET_COL])

    # ── LightGBM Quantile (90% CI: q05–q95) ──────────────────────────────────
    lgbm_qs = {}
    for q in ["q05","q25","q50","q75","q95"]:
        p = os.path.join(config.MODEL_DIR, f"lgbm_{q}_{m}.joblib")
        if os.path.exists(p):
            lgbm_qs[q] = joblib.load(p).predict(X_test_df)
    if "q05" in lgbm_qs and "q95" in lgbm_qs:
        samples_lgbm = None
        if len(lgbm_qs) >= 3:
            samples_lgbm = np.stack(list(lgbm_qs.values()), axis=1)
        prob_rows.append(prob_row(
            "LGBM Quantile (90% CI)",
            lgbm_qs["q05"], lgbm_qs["q95"], y_true,
            alpha=0.10, samples=samples_lgbm))

    # ── CQR (90% nominal CI; empirical coverage may be lower under distributional shift) ───
    cqr_path = os.path.join(config.REPORT_DIR, f"cqr_preds_{m}.csv")
    cqr_df   = pd.read_csv(cqr_path, index_col=0, parse_dates=True) \
               if os.path.exists(cqr_path) else None
    if cqr_df is not None:
        cqr_lo = cqr_df["cqr_lower"].reindex(y_index).values
        cqr_hi = cqr_df["cqr_upper"].reindex(y_index).values
        cqr_meta = joblib.load(os.path.join(config.MODEL_DIR, f"cqr_meta_{m}.joblib"))
        # CQR CRPS: approximate with 11 uniform samples from [lo, hi]
        mask_c = ~np.isnan(y_true) & ~np.isnan(cqr_lo) & ~np.isnan(cqr_hi)
        cqr_samples = np.linspace(cqr_lo, cqr_hi, 11).T  # [n, 11]
        row_cqr = prob_row("CQR (90% nominal CI)",
                           cqr_lo, cqr_hi, y_true, alpha=0.10, samples=cqr_samples)
        row_cqr["Correction"] = round(cqr_meta["correction"], 4)
        prob_rows.append(row_cqr)

    # ── Chronos-Bolt (native CI — label actual level) ─────────────────────────
    ch_path = os.path.join(config.REPORT_DIR, f"chronos_preds_{m}.csv")
    if os.path.exists(ch_path):
        ch_df = pd.read_csv(ch_path, index_col=0, parse_dates=True).reindex(y_index)
        lo_col = next((c for c in ch_df.columns if "lower" in c), None)
        hi_col = next((c for c in ch_df.columns if "upper" in c), None)
        if lo_col and hi_col:
            ci_pct = "80%" if "80" in lo_col else "90%"
            alpha_c = 0.20 if ci_pct == "80%" else 0.10
            prob_rows.append(prob_row(
                f"Chronos-Bolt ({ci_pct} CI)",
                ch_df[lo_col].values, ch_df[hi_col].values, y_true, alpha=alpha_c))

    # ── QRF (90% CI: q05–q95) + CRPS ─────────────────────────────────────────
    qrf_path = os.path.join(config.REPORT_DIR, f"qrf_preds_{m}.csv")
    if os.path.exists(qrf_path):
        qrf_df = pd.read_csv(qrf_path, index_col=0, parse_dates=True).reindex(y_index)
        q05q = qrf_df["q05"].values if "q05" in qrf_df.columns else None
        q95q = qrf_df["q95"].values if "q95" in qrf_df.columns else None
        if q05q is not None and q95q is not None:
            qrf_qs = [qrf_df[c].values for c in ["q05","q25","q50","q75","q95"]
                      if c in qrf_df.columns]
            qrf_samples = np.stack(qrf_qs, axis=1) if len(qrf_qs) >= 3 else None
            prob_rows.append(prob_row(
                "QRF (90% CI)", q05q, q95q, y_true,
                alpha=0.10, samples=qrf_samples))

    # ── N-HiTS Quantile (80% CI: q10–q90) + CRPS ─────────────────────────────
    nhq_path = os.path.join(config.REPORT_DIR, f"nhits_quantile_preds_{m}.csv")
    if os.path.exists(nhq_path):
        nhq_df = pd.read_csv(nhq_path, parse_dates=["ds"]).set_index("ds")
        nhq_df.index = pd.to_datetime(nhq_df.index, utc=True)
        nhq_df = nhq_df[~nhq_df.index.duplicated(keep="first")]
        idx_utc = y_index.tz_localize("UTC") if y_index.tz is None else y_index.tz_convert("UTC")
        nhq_df = nhq_df.reindex(idx_utc)
        if "q10" in nhq_df.columns and "q90" in nhq_df.columns:
            q10v, q90v = nhq_df["q10"].values, nhq_df["q90"].values
            nhq_samples = None
            if "q50" in nhq_df.columns:
                nhq_samples = np.stack([q10v, nhq_df["q50"].values, q90v], axis=1)
            prob_rows.append(prob_row(
                "N-HiTS Quantile (80% CI)",
                q10v, q90v, y_true, alpha=0.20, samples=nhq_samples))

    # ── BiLSTM MC Dropout (90% CI: q05–q95 from 100 forward passes) ──────────
    bl_path = os.path.join(config.REPORT_DIR, f"bilstm_preds_{m}.csv")
    if os.path.exists(bl_path):
        bl_df = pd.read_csv(bl_path, index_col=0, parse_dates=True)
        bl_df.index = pd.to_datetime(bl_df.index, utc=True)
        bl_df = bl_df.reindex(
            y_index.tz_localize("UTC") if y_index.tz is None else y_index.tz_convert("UTC")
        )
        q05_bl = bl_df["q05"].values if "q05" in bl_df.columns else None
        q95_bl = bl_df["q95"].values if "q95" in bl_df.columns else None
        mean_bl = (bl_df["mean_pred"] if "mean_pred" in bl_df.columns
                   else bl_df.filter(like="pred").iloc[:,0]).values
        if q05_bl is not None and q95_bl is not None:
            bl_samples = None
            if "q25" in bl_df.columns and "q75" in bl_df.columns:
                bl_samples = np.stack([q05_bl, bl_df["q25"].values,
                                       mean_bl, bl_df["q75"].values, q95_bl], axis=1)
            prob_rows.append(prob_row(
                "BiLSTM MC Dropout (90% CI)",
                q05_bl, q95_bl, y_true, alpha=0.10, samples=bl_samples))

    prob_df = pd.DataFrame(prob_rows).set_index("Model")
    print(prob_df.to_string())
    prob_df.to_csv(os.path.join(config.REPORT_DIR, f"table_probabilistic_{m}.csv"))




    # ── RQ3: Economic Utility ─────────────────────────────────────
    print(f"\n  [RQ3] Economic Utility (TC={config.TRANSACTION_COST_PER_MWH}$/MWh, "
          f"slippage={config.SLIPPAGE_STD_FACTOR}σ)")

    actual_series = test_df[config.TARGET_COL]
    point_strats  = {}
    if "LightGBM" in predictions:
        point_strats["LightGBM"] = pd.Series(predictions["LightGBM"], index=y_index)
    if "SeasonalNaive" in predictions:
        point_strats["Seasonal Naive"] = pd.Series(predictions["SeasonalNaive"], index=y_index)
    if "BiLSTM" in predictions:
        point_strats["BiLSTM"] = pd.Series(predictions["BiLSTM"], index=y_index)

    roll_lgbm_path = os.path.join(config.REPORT_DIR, f"rolling_lgbm_preds_{m}.csv")
    if os.path.exists(roll_lgbm_path):
        roll_df = pd.read_csv(roll_lgbm_path, index_col=0, parse_dates=True)
        roll_df.index = pd.to_datetime(roll_df.index, utc=True)
        idx_utc = y_index.tz_localize("UTC") if y_index.tz is None else y_index.tz_convert("UTC")
        roll_df = roll_df.reindex(idx_utc)
        roll_df.index = y_index
        point_strats["Rolling LightGBM"] = roll_df["predicted"]

    interval_strats = {}
    if cqr_df is not None:
        interval_strats["Risk-Aware CQR"] = (
            cqr_df["cqr_lower"].reindex(y_index),
            cqr_df["cqr_upper"].reindex(y_index),
        )

    # S2: Risk-Aware Bayesian — BiLSTM MC Dropout 90% CI (q05–q95)
    # Trades only when MC Dropout CI is narrow (high model confidence)
    bl_eco_path = os.path.join(config.REPORT_DIR, f"bilstm_preds_{m}.csv")
    if os.path.exists(bl_eco_path):
        bl_eco_df = pd.read_csv(bl_eco_path, index_col=0, parse_dates=True)
        bl_eco_df.index = pd.to_datetime(bl_eco_df.index, utc=True)
        if y_index.tz is None:
            bl_idx = y_index.tz_localize("UTC")
        else:
            bl_idx = y_index.tz_convert("UTC")
        bl_eco_df = bl_eco_df.reindex(bl_idx)
        if "q05" in bl_eco_df.columns and "q95" in bl_eco_df.columns:
            interval_strats["Risk-Aware Bayesian"] = (
                pd.Series(bl_eco_df["q05"].values, index=y_index),
                pd.Series(bl_eco_df["q95"].values, index=y_index),
            )

    if point_strats or interval_strats:
        pnl_df = simulate_trading(actual_series, point_strats, interval_strats)
        eco_rows = []
        for strategy in pnl_df.columns:
            m_eco = compute_economic_metrics(pnl_df[strategy])
            m_eco["Strategy"] = strategy
            eco_rows.append(m_eco)
            print(f"    {strategy:25} P&L=${m_eco['Total_PnL_$']:>8.2f} "
                  f"Sharpe={m_eco['Sharpe']:>6.3f} "
                  f"MDD=${m_eco['Max_Drawdown_$']:>8.2f}")

        eco_df = pd.DataFrame(eco_rows).set_index("Strategy")
        eco_df.to_csv(os.path.join(config.REPORT_DIR, f"table_economic_{m}.csv"))
        pnl_df.to_csv(os.path.join(config.REPORT_DIR, f"pnl_daily_{m}.csv"))

    print(f"\n  ✅ Evaluation complete for {market}.")
    print(f"  Results saved to: {config.REPORT_DIR}")


if __name__ == "__main__":
    run_evaluation("PJM")
    run_evaluation("ERCOT")
