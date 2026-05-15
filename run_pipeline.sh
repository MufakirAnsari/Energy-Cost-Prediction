#!/bin/bash
# ============================================================
# run_pipeline.sh — V2 EPF Full Pipeline Runner
# ============================================================
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh         # Run everything
#   ./run_pipeline.sh --from step_05  # Resume from a step
# ============================================================

set -e  # Exit on any error

VENV_DIR="venv_v2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/reports/logs"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $1" | tee -a "$LOG_DIR/pipeline.log"; }

log "=========================================="
log "  V2 EPF Pipeline Starting"
log "=========================================="

# ── Python environment check ─────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if ! python -c "import gridstatus" 2>/dev/null; then
    log "Installing requirements..."
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$SCRIPT_DIR"

FROM_STEP="${1:-step_00}"
STEPS=(
    "step_00:step_00_download_pjm.py"
    "step_00b:step_00_download_ercot.py"
    "step_00c:step_00_download_eia.py"
    "step_01:step_01_preprocess.py"
    "step_02:step_02_train_baselines.py"
    "step_03:step_03_train_lgbm.py"
    "step_05:step_05_train_bilstm.py"
    "step_06:step_06_train_modern_dl.py"
    "step_07:step_07_chronos_inference.py"
    "step_08:step_08_conformal.py"
    "step_09:step_09_evaluate.py"
    "step_10:step_10_stress_test.py"
)

SKIP=true
for entry in "${STEPS[@]}"; do
    KEY="${entry%%:*}"
    SCRIPT="${entry##*:}"

    if [[ "$KEY" == *"$FROM_STEP"* ]]; then
        SKIP=false
    fi

    if $SKIP; then
        log "Skipping $SCRIPT (before $FROM_STEP)"
        continue
    fi

    log "Running: $SCRIPT"
    START_T=$(date +%s)
    python "$SCRIPT" 2>&1 | tee -a "$LOG_DIR/${KEY}.log"
    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))
    log "Completed: $SCRIPT in ${ELAPSED}s"
done

log "=========================================="
log "  Pipeline Complete!"
log "  Results in: $SCRIPT_DIR/reports/"
log "=========================================="
