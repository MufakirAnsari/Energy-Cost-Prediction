#!/usr/bin/env bash
# =============================================================================
#  run_all.sh — V2 EPF Pipeline (UPDATED v2 — Journal-Ready)
#  Runs all training, inference, evaluation, and reporting steps.
#  Each step logs to logs/stepXX_TIMESTAMP.log
#  Skips a step if its primary output file already exists.
#
#  NEW IN v2:
#    step 00  — Data integrity check (no leakage assertion)
#    step 01  — EDA (price stats, ACF, regime violin plots)
#    step 05b — BiLSTM ERCOT retrain (log1p+clip fix)
#    step 07b — TFT (Temporal Fusion Transformer)
#    step 08b — N-HiTS Quantile (MQLoss)
#    step 10b — Conformal alpha sweep (A5)
#    step 19  — RQ4 cross-market generalizability
#    step 18  — LaTeX table generation (6 tables)
#    Figures: F1–F11 (reliability diagram F10, fan chart F11)
#
#  Usage:
#    chmod +x run_all.sh
#    ./run_all.sh 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M).log
#
#  Prerequisites:
#    - step_02_train_baselines.py must have finished
#    - Virtual environment .venv must be activated OR at ./.venv
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()    { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn()   { echo -e "${YELLOW}[$(date '+%H:%M:%S')] SKIP:${NC} $*"; }
err()    { echo -e "${RED}[$(date '+%H:%M:%S')] FAILED:${NC} $*"; }
header() { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

FAILED_STEPS=()

run_step() {
    local step="$1"; local script="$2"; local desc="$3"; local skip_if="${4:-}"
    echo ""
    echo "================================================================="
    log "STEP ${step}: ${desc}"
    echo "================================================================="
    if [[ -n "$skip_if" && -e "$skip_if" ]]; then
        warn "Output exists: $(basename "$skip_if") — skipping. Delete to re-run."
        return 0
    fi
    local logfile="$LOG_DIR/step${step}_$(date +%Y%m%d_%H%M%S).log"
    log "Log: $logfile"
    if "$PYTHON" "$SCRIPT_DIR/$script" 2>&1 | tee "$logfile"; then
        log "✅ Step $step complete."
    else
        err "Step $step FAILED — see $logfile"
        FAILED_STEPS+=("Step $step: $desc")
    fi
}

START=$(date +%s)
log "V2 EPF Pipeline v2 — Journal-Ready Edition"
log "Python: $PYTHON"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 0: Pre-flight checks"
# ─────────────────────────────────────────────────────────────────────────────

run_step "00" "test_data_integrity.py" \
    "Data Integrity — No-leakage assertion suite (PJM + ERCOT)"
# Note: non-fatal — pipeline continues even if checks fail

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 1: EDA"
# ─────────────────────────────────────────────────────────────────────────────

run_step "01" "step_01_eda.py" \
    "EDA — Price stats, ACF/PACF, regime violin plots" \
    "$SCRIPT_DIR/reports/figures/EDA1_violin_regimes.png"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 2: Classical + Tree Models"
# ─────────────────────────────────────────────────────────────────────────────

run_step "02" "step_02_train_baselines.py" \
    "Baselines — SeasonalNaive, AutoARIMA, MSTL (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/baseline_preds_pjm_test.csv"

run_step "03" "step_03_train_lgbm.py" \
    "LightGBM — Point + 5-quantile (PJM + ERCOT)" \
    "$SCRIPT_DIR/models/lgbm_point_pjm.joblib"

run_step "04" "step_04_train_xgboost.py" \
    "XGBoost — Point forecast (PJM + ERCOT)" \
    "$SCRIPT_DIR/models/xgboost_point_pjm.joblib"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 3: Deep Learning Models (GPU)"
# ─────────────────────────────────────────────────────────────────────────────

run_step "05" "step_05_train_bilstm.py" \
    "Bayesian Bi-LSTM — MC Dropout, PJM only" \
    "$SCRIPT_DIR/models/bilstm_pjm.keras"

run_step "05b" "step_05b_retrain_bilstm_ercot.py" \
    "Bayesian Bi-LSTM — ERCOT retrain (log1p+clipnorm fix)" \
    "$SCRIPT_DIR/models/bilstm_ercot.keras"

run_step "06" "step_06_train_patchtst.py" \
    "PatchTST — Patch-based Transformer (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/patchtst_preds_pjm.csv"

run_step "07" "step_07_train_itransformer.py" \
    "iTransformer — Inverted Attention (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/itransformer_preds_pjm.csv"

run_step "07b" "step_07b_train_tft.py" \
    "TFT — Temporal Fusion Transformer (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/tft_preds_pjm.csv"

run_step "06b" "step_06b_train_bitcn.py" \
    "BiTCN — Bidirectional Temporal CNN (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/bitcn_preds_pjm.csv"

run_step "08" "step_08_train_nhits.py" \
    "N-HiTS — Point forecast (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/nhits_preds_pjm.csv"

run_step "08b" "step_08b_nhits_quantile.py" \
    "N-HiTS Quantile — MQLoss (80% CI, PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/nhits_quantile_preds_pjm.csv"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 4: Zero-Shot + Probabilistic Wrappers"
# ─────────────────────────────────────────────────────────────────────────────

run_step "09" "step_09_chronos_inference.py" \
    "Chronos-Bolt — Zero-Shot Foundation Model (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/chronos_preds_pjm.csv"

run_step "10" "step_10_conformal.py" \
    "CQR — Conformal Quantile Regression (90% nominal CI; coverage under distributional shift documented)" \
    "$SCRIPT_DIR/reports/cqr_preds_pjm.csv"

run_step "10b" "step_10b_alpha_sweep.py" \
    "Conformal Alpha Sweep — PICP vs MPIW at {80,90,95}% (A5)" \
    "$SCRIPT_DIR/reports/table_alpha_sweep_pjm.csv"

run_step "11" "step_11_qrf.py" \
    "QRF — Quantile Regression Forest (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/qrf_preds_pjm.csv"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 5: Ensemble"
# ─────────────────────────────────────────────────────────────────────────────

run_step "12" "step_12_ensemble.py" \
    "Ensemble — LGBM+XGB+BiLSTM stacked meta-learner (PJM + ERCOT)" \
    "$SCRIPT_DIR/reports/ensemble_preds_pjm.csv"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 6: Evaluation + Analysis"
# ─────────────────────────────────────────────────────────────────────────────

run_step "13" "step_09_evaluate.py" \
    "Evaluation — RQ1 point + RQ2 probabilistic + RQ3 economic (PJM + ERCOT)"

run_step "14" "step_14_dm_tests.py" \
    "DM Tests — Pairwise + BH correction (PJM + ERCOT)"

run_step "15" "step_15_ablation.py" \
    "Ablation — A1-A5: features, lag, dropout, ensemble, alpha sweep (PJM + ERCOT)"

run_step "16" "step_16_stress_test.py" \
    "Stress Test — Out-of-sample regime evaluation (PJM + ERCOT) [v2 fixed]"

run_step "19" "step_19_rq4_crossmarket.py" \
    "RQ4 — Cross-Market Transfer: PJM→ERCOT (zero-shot, 31 shared features)"

# ─────────────────────────────────────────────────────────────────────────────
header "  PHASE 7: Figures + Tables"
# ─────────────────────────────────────────────────────────────────────────────

run_step "17" "step_17_figures.py" \
    "Figures — F1–F11 at 300 DPI (PNG + PDF)"

run_step "18" "step_18_paper_tables.py" \
    "Tables — 6 LaTeX booktabs tables for paper submission"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "================================================================="
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    log "🎉 ALL STEPS COMPLETE — Pipeline is journal-ready!"
else
    err "Pipeline finished with ${#FAILED_STEPS[@]} failed step(s):"
    for s in "${FAILED_STEPS[@]}"; do
        echo -e "  ${RED}✗${NC} $s"
    done
fi
log "Runtime: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
log "Results:  $SCRIPT_DIR/reports/"
log "Models:   $SCRIPT_DIR/models/"
log "Figures:  $SCRIPT_DIR/reports/figures/"
log "Tables:   $SCRIPT_DIR/reports/tex/"
echo "================================================================="
