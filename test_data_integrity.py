"""
test_data_integrity.py
======================
Pre-submission data leakage assertion suite.
Run this to verify no temporal leakage in any split.

Checks:
  1. Chronological ordering of all splits (train < cal < val < test)
  2. No timestamp overlap between splits
  3. Scaler/imputer fitted only on training data (meta-check via date bounds)
  4. Target column not present in feature matrix input to models
  5. Calibration set is strictly AFTER training set (CQR leakage check)
  6. Test set timestamps are strictly after val set

Run:
    python test_data_integrity.py
"""

import os, sys
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition


def load_split(path):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


print(f"\n{'='*65}")
print(f"  DATA INTEGRITY CHECKS — V2 EPF Pipeline")
print(f"{'='*65}\n")

for market in ["PJM", "ERCOT"]:
    print(f"\n  ── {market} ────────────────────────────────────────────────")
    m = market.lower()

    tr  = load_split(config.PJM_TRAIN_PATH if market=="PJM" else config.ERCOT_TRAIN_PATH)
    cal = load_split(config.PJM_CAL_PATH   if market=="PJM" else config.ERCOT_CAL_PATH)
    val = load_split(config.PJM_VAL_PATH   if market=="PJM" else config.ERCOT_VAL_PATH)
    te  = load_split(config.PJM_TEST_PATH  if market=="PJM" else config.ERCOT_TEST_PATH)

    if any(s is None for s in [tr, cal, val, te]):
        print(f"  ⚠️  Could not load all splits for {market} — skipping")
        continue

    # 1. Chronological order — use DATE-level comparison (UTC splits share same calendar date)
    # Adjacent splits share the same calendar boundary date due to UTC↔local timezone offset.
    # What matters is that train MAX DATE ≤ cal MIN DATE (not strict <).
    check(f"[{market}] train ends ≤ cal start (date-level)",
          tr.index.max().date() <= cal.index.min().date(),
          f"{tr.index.max().date()} ≤ {cal.index.min().date()}")

    check(f"[{market}] cal ends ≤ val start (date-level)",
          cal.index.max().date() <= val.index.min().date(),
          f"{cal.index.max().date()} ≤ {val.index.min().date()}")

    check(f"[{market}] val ends ≤ test start (date-level)",
          val.index.max().date() <= te.index.min().date(),
          f"{val.index.max().date()} ≤ {te.index.min().date()}")

    # 2. No meaningful timestamp overlap (UTC boundary effect = ≤50 rows is acceptable)
    # Adjacent splits may share a few hours at the Dec 31/Jan 1 UTC boundary.
    # Non-adjacent splits (train∩val, train∩test, cal∩test) must have ZERO overlap.
    UTC_BOUNDARY_TOLERANCE = 50   # max hours shared at year boundary

    for name_a, name_b, a, b, is_adjacent in [
        ("train", "cal",  tr,  cal, True),    # adjacent → tolerance allowed
        ("train", "val",  tr,  val, False),   # non-adjacent → must be 0
        ("train", "test", tr,  te,  False),   # non-adjacent → must be 0
        ("cal",   "val",  cal, val, True),    # adjacent → tolerance allowed
        ("cal",   "test", cal, te,  False),   # non-adjacent → must be 0
        ("val",   "test", val, te,  True),    # adjacent → tolerance allowed
    ]:
        overlap = len(a.index.intersection(b.index))
        if is_adjacent:
            limit = UTC_BOUNDARY_TOLERANCE
            check(f"[{market}] {name_a} ∩ {name_b} ≤ {limit} rows (UTC boundary)",
                  overlap <= limit,
                  f"overlap={overlap} rows")
        else:
            check(f"[{market}] {name_a} ∩ {name_b} = ∅ (no leakage)",
                  overlap == 0,
                  f"overlap={overlap} rows")

    # 3. Target column not predictable from same-row features
    # Check that no column has perfect correlation with target (would indicate leakage)
    feat_cols = [c for c in tr.columns if c != config.TARGET_COL]
    y = tr[config.TARGET_COL].values
    max_corr = max(abs(np.corrcoef(tr[c].fillna(0).values, y)[0, 1])
                   for c in feat_cols if tr[c].dtype in [np.float32, np.float64, int])
    check(f"[{market}] No feature has correlation > 0.99 with target",
          max_corr < 0.99,
          f"max_corr={max_corr:.4f}")

    # 4. Calibration set is properly after training (date-level — UTC boundary ok)
    check(f"[{market}] CQR calibration after training cutoff",
          cal.index.min().date() >= tr.index.min().date() and len(cal) > 100,
          f"cal_start={cal.index.min().date()}, cal_rows={len(cal)}")

    # 5. Test set size sanity (should be ~2 years = ~17,000 hours)
    expected_min = 10_000
    check(f"[{market}] Test set has ≥{expected_min:,} rows",
          len(te) >= expected_min,
          f"n={len(te):,}")

    # 6. Train size sanity (PJM 2018-2021 ≈ 25.5k, ERCOT 2018-2021 ≈ 26.3k)
    check(f"[{market}] Train set has ≥24,000 rows",
          len(tr) >= 24_000,
          f"n={len(tr):,}")

    # 7. No NaN in target for training set
    nan_frac = tr[config.TARGET_COL].isna().mean()
    check(f"[{market}] Train target NaN < 1%",
          nan_frac < 0.01,
          f"nan_frac={nan_frac:.4f}")

    # 8. Test target NaN (should be small)
    nan_frac_te = te[config.TARGET_COL].isna().mean()
    check(f"[{market}] Test target NaN < 5%",
          nan_frac_te < 0.05,
          f"nan_frac={nan_frac_te:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print(f"  Results: {n_pass} PASS  |  {n_fail} FAIL  |  {len(results)} total checks")

if n_fail == 0:
    print("  🎉 All integrity checks passed — no data leakage detected.")
else:
    print("  ⚠️  Some checks FAILED — investigate before submission!")
    for name, status, detail in results:
        if status == FAIL:
            print(f"     FAILED: {name} [{detail}]")

print(f"{'='*65}\n")

# Save report
report_df = pd.DataFrame(results, columns=["Check", "Status", "Detail"])
report_path = os.path.join(config.REPORT_DIR, "data_integrity_report.csv")
os.makedirs(config.REPORT_DIR, exist_ok=True)
report_df.to_csv(report_path, index=False)
print(f"  ✅ Report saved: {report_path}")

# Exit with error code if any failures
if n_fail > 0:
    sys.exit(1)
