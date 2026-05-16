"""
utils.py — Shared utilities for the V2 EPF Pipeline
=====================================================
Centralises metric functions, data loaders, and helpers that were
previously duplicated across step_09_evaluate.py, step_14_dm_tests.py,
step_16_stress_test.py, and step_19_rq4_crossmarket.py.

Usage:
    from utils import mae, rmse, smape, picp, mpiw, winkler_score, dm_test
"""

import numpy as np
import pandas as pd

try:
    import properscoring as ps
    HAS_PROPERSCORING = True
except ImportError:
    HAS_PROPERSCORING = False


# ─────────────────────────────────────────────────────────────────────────────
# POINT ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mae(y, yhat):
    """Mean Absolute Error — primary metric (robust to spikes)."""
    return np.nanmean(np.abs(y - yhat))


def rmse(y, yhat):
    """Root Mean Squared Error — penalises extreme errors."""
    return np.sqrt(np.nanmean((y - yhat) ** 2))


def smape(y, yhat):
    """Symmetric MAPE — handles near-zero prices."""
    denom = (np.abs(y) + np.abs(yhat)) / 2
    denom[denom == 0] = np.nan
    return np.nanmean(np.abs(y - yhat) / denom) * 100


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILISTIC METRICS  (all assume matched 1-D arrays)
# ─────────────────────────────────────────────────────────────────────────────

def picp(y, lower, upper):
    """Prediction Interval Coverage Probability (%)."""
    return np.nanmean((y >= lower) & (y <= upper)) * 100


def mpiw(lower, upper):
    """Mean Prediction Interval Width."""
    return np.nanmean(upper - lower)


def winkler_score(y, lower, upper, alpha=0.10):
    """
    Winkler interval score at significance level alpha.
    Lower is better — rewards narrow intervals + penalises violations.
    """
    width = upper - lower
    pen_l = (2 / alpha) * np.maximum(lower - y, 0)
    pen_u = (2 / alpha) * np.maximum(y - upper, 0)
    return np.nanmean(width + pen_l + pen_u)


def pinball_loss(y, q_pred, alpha):
    """Pinball (quantile) loss at quantile level alpha."""
    return np.nanmean(
        np.where(y >= q_pred,
                 alpha * (y - q_pred),
                 (1 - alpha) * (q_pred - y))
    )


def crps(y, samples):
    """
    Continuous Ranked Probability Score.
    samples: array of shape [n_points, n_samples].
    Uses properscoring if available, otherwise a manual approximation.
    """
    if HAS_PROPERSCORING:
        return ps.crps_ensemble(y, samples).mean()
    return _crps_fallback(y, samples)


def _crps_fallback(y, samples):
    """Simple CRPS approximation when properscoring is unavailable."""
    result = []
    for i in range(len(y)):
        s = np.sort(samples[i])
        expected = np.mean(np.abs(s - y[i]))
        spread = np.mean(np.abs(s[:, None] - s[None, :])) / 2
        result.append(expected - spread)
    return np.mean(result)


# ─────────────────────────────────────────────────────────────────────────────
# DIEBOLD-MARIANO TEST  (Harvey-Leybourne-Newbold correction)
# ─────────────────────────────────────────────────────────────────────────────

def dm_test(y, pred_a, pred_b, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy (MAE loss).
    H0: Models A and B have equal MAE.
    Returns: (dm_stat, p_value)
    Uses Harvey-Leybourne-Newbold (HLN) small-sample correction.
    Reference: Harvey, Leybourne, Newbold (1997).
    """
    try:
        e1 = np.abs(y - pred_a)
        e2 = np.abs(y - pred_b)
        d = e1 - e2
        n = len(d)
        d_mean = np.mean(d)

        # Newey-West HAC variance for autocorrelation
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
        var_d = np.var(d, ddof=1)
        for k in range(1, max_lag + 1):
            w = 1 - k / (max_lag + 1)
            cov = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
            var_d += 2 * w * cov
        var_d = max(var_d, 1e-10)

        # HLN correction
        dm_stat = d_mean / np.sqrt(var_d / n)
        dm_corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm_stat

        from scipy import stats
        p_val = 2 * stats.t.sf(np.abs(dm_corr), df=n - 1)
        return dm_corr, p_val
    except Exception:
        return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions_csv(path, col_name, ref_index=None):
    """
    Load a prediction column from a CSV, optionally aligned to ref_index.
    Returns numpy array or None if file/column doesn't exist.
    """
    import os
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")]
    if col_name not in df.columns:
        return None
    if ref_index is not None:
        idx_utc = ref_index.tz_localize("UTC") if ref_index.tz is None \
                  else ref_index.tz_convert("UTC")
        return df[col_name].reindex(idx_utc).values
    return df[col_name].values
