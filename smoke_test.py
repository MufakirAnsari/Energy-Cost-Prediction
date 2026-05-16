"""
smoke_test.py
=============
Quick smoke test for the full V2 pipeline.
Checks:
  1. Syntax validity (py_compile) for all step scripts
  2. Config constants referenced in each script actually exist in config.py
  3. Required third-party packages are importable
  4. Preprocessed data files exist (steps 03+ depend on them)

Does NOT run any training — completes in seconds.
"""

import os, sys, py_compile, importlib, traceback
sys.path.insert(0, os.path.dirname(__file__))

PASS = "✅"; FAIL = "❌"; WARN = "⚠️ "

results = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}" + (f"  — {detail}" if detail else ""))
    return ok

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG IMPORT
# ─────────────────────────────────────────────────────────────────────────────
section("1. Config")
try:
    import config
    check("config.py imports", True)
except Exception as e:
    check("config.py imports", False, str(e))
    print("\n  ❌ Cannot continue without config. Exiting.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTAX CHECK — ALL STEP SCRIPTS
# ─────────────────────────────────────────────────────────────────────────────
section("2. Syntax Check")

STEPS = [
    "utils.py",
    "step_02_train_baselines.py",
    "step_03_train_lgbm.py",
    "step_04_train_xgboost.py",
    "step_05_train_bilstm.py",
    "step_05b_retrain_bilstm_ercot.py",
    "step_06_train_patchtst.py",
    "step_06b_train_bitcn.py",
    "step_07_train_itransformer.py",
    "step_07b_train_tft.py",
    "step_08_train_nhits.py",
    "step_08b_nhits_quantile.py",
    "step_09_chronos_inference.py",
    "step_10_conformal.py",
    "step_10b_alpha_sweep.py",
    "step_11_qrf.py",
    "step_12_ensemble.py",
    "step_09_evaluate.py",    # step_13
    "step_14_dm_tests.py",
    "step_15_ablation.py",
    "step_16_stress_test.py",
    "step_17_figures.py",
    "step_18_paper_tables.py",
    "step_19_rq4_crossmarket.py",
]

base = os.path.dirname(os.path.abspath(__file__))
for fname in STEPS:
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        check(fname, False, "FILE NOT FOUND")
        continue
    try:
        py_compile.compile(path, doraise=True)
        check(fname, True)
    except py_compile.PyCompileError as e:
        check(fname, False, str(e).split("\n")[0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIG CONSTANTS CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("3. Config Constants")

required_attrs = [
    "DATA_START", "DATA_END",
    "PJM_TRAIN_PATH", "PJM_CAL_PATH", "PJM_VAL_PATH", "PJM_TEST_PATH",
    "ERCOT_TRAIN_PATH", "ERCOT_CAL_PATH", "ERCOT_VAL_PATH", "ERCOT_TEST_PATH",
    "MODEL_DIR", "REPORT_DIR",
    "TARGET_COL", "RANDOM_SEED",
    "LGBM_POINT_PARAMS", "LGBM_QUANTILE_LEVELS", "get_lgbm_quantile_params",
    "XGB_PARAMS",
    "SEQ_LEN_DEFAULT", "PRED_LEN", "BATCH_SIZE", "MAX_EPOCHS",
    "LEARNING_RATE", "PATIENCE", "L2_REG",
    "MC_SAMPLES", "NOMINAL_COVERAGE", "CONFORMAL_ALPHA",
    "TRANSACTION_COST_PER_MWH", "SLIPPAGE_STD_FACTOR", "TRADE_VOLUME_MWH",
    "ENSEMBLE_META_PARAMS",
    "REGIMES",
    "PLOT_DPI",
]

for attr in required_attrs:
    ok = hasattr(config, attr)
    check(f"config.{attr}", ok, "" if ok else "MISSING")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROCESSED DATA FILES
# ─────────────────────────────────────────────────────────────────────────────
section("4. Processed Data Files")

data_files = {
    "PJM train":   config.PJM_TRAIN_PATH,
    "PJM cal":     config.PJM_CAL_PATH,
    "PJM val":     config.PJM_VAL_PATH,
    "PJM test":    config.PJM_TEST_PATH,
    "ERCOT train": config.ERCOT_TRAIN_PATH,
    "ERCOT cal":   config.ERCOT_CAL_PATH,
    "ERCOT val":   config.ERCOT_VAL_PATH,
    "ERCOT test":  config.ERCOT_TEST_PATH,
}
for label, path in data_files.items():
    exists = os.path.exists(path)
    check(label, exists, path if not exists else f"{os.path.getsize(path)//1024} KB")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PYTHON PACKAGE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
section("5. Package Imports")

packages = [
    # name,            import_name,          required
    ("pandas",         "pandas",             True),
    ("numpy",          "numpy",              True),
    ("scikit-learn",   "sklearn",            True),
    ("lightgbm",       "lightgbm",           True),
    ("xgboost",        "xgboost",            True),
    ("joblib",         "joblib",             True),
    ("statsforecast",  "statsforecast",      True),
    ("scipy",          "scipy",              True),
    ("pyarrow",        "pyarrow",            True),
    ("matplotlib",     "matplotlib",         True),
    ("tensorflow",     "tensorflow",         False),  # Optional: BiLSTM
    ("tensorflow_prob","tensorflow_probability", False),
    ("torch",          "torch",              False),  # Optional: DL models
    ("neuralforecast", "neuralforecast",     False),
    ("chronos",        "chronos",            False),
    ("quantile_forest","quantile_forest",    False),
    ("properscoring",  "properscoring",      False),
    ("shap",           "shap",               False),
]

for pkg_name, import_name, required in packages:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        check(f"{pkg_name} ({ver})", True)
    except ImportError:
        tag = "REQUIRED" if required else "optional"
        check(f"{pkg_name}", not required, f"{tag} — pip install {pkg_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. GPU CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("6. GPU / CUDA")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
    check(f"CUDA available ({gpu_name})", cuda_ok,
          "GPU training will be used for DL models" if cuda_ok else
          "DL steps will run on CPU (slower — steps 05–09)")
except ImportError:
    # torch not installed yet — warn but don't fail (steps 03–04 don't need it)
    results.append((WARN, "PyTorch (optional for steps 05–09)", "pip install torch"))
    print(f"  {WARN}  PyTorch (optional for steps 05–09)  — pip install torch")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    check(f"TF GPU ({len(gpus)} device(s))", len(gpus) > 0,
          "GPU will be used for BiLSTM" if gpus else "BiLSTM (step 05) will run on CPU")
except ImportError:
    results.append((WARN, "TensorFlow (optional for step 05)", "pip install tensorflow tensorflow-probability"))
    print(f"  {WARN}  TensorFlow (optional for step 05)  — pip install tensorflow tensorflow-probability")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SMOKE TEST SUMMARY")
print(f"{'='*60}")
passed  = sum(1 for s, _, _ in results if s == PASS)
failed  = sum(1 for s, _, _ in results if s == FAIL)
total   = len(results)
print(f"  {PASS}  Passed: {passed}/{total}")
print(f"  {FAIL}  Failed: {failed}/{total}")

if failed > 0:
    print(f"\n  Failed checks:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"    {FAIL} {label}: {detail}")

if failed == 0:
    print("\n  🎉 All checks passed — pipeline is ready to run!")
elif failed <= 3:
    print("\n  ⚠️  Minor issues — check failed items above before running.")
else:
    print("\n  ❌ Multiple failures — fix before running pipeline.")

sys.exit(0 if failed == 0 else 1)
