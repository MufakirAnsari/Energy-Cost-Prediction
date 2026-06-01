#!/bin/bash
# push_to_github.sh — Run this from /home/ansari/Desktop/Energy/V2
# ================================================================

set -e

echo "═══════════════════════════════════════════════════"
echo "  PUSHING V2 PIPELINE TO GITHUB"
echo "═══════════════════════════════════════════════════"

# 1. Regenerate graphical abstract with corrected numbers
echo ""
echo "[1/5] Regenerating Graphical Abstract..."
python -c "import step_23_final_updates; step_23_final_updates.create_graphical_abstract()"

# 2. Check for files exceeding GitHub 100MB limit
echo ""
echo "[2/5] Checking for oversized files..."
find . -path ./.git -prune -o -path ./.venv -prune -o -type f -size +100M -print 2>/dev/null | while read f; do
    echo "  ⚠️  LARGE: $f ($(du -h "$f" | cut -f1))"
done

# 3. Stage everything (respecting .gitignore)
echo ""
echo "[3/5] Staging files..."
git add -A
echo "  Staged files:"
git diff --cached --stat | tail -5

# 4. Commit
echo ""
echo "[4/5] Committing..."
git commit -m "V2 complete: 18-model EPF benchmark with rolling window, Chronos-2, and CQR analysis

- 24 pipeline scripts (step_00 through step_23)
- 16 model families in 18 configurations (3 statistical, 3 tree, 6 DL, 2 TSFM, 2 ensemble, 2 rolling)
- Monthly expanding-window retraining: 24% MAE improvement
- Chronos-2 covariate-enhanced inference: 49% improvement over univariate
- 3 probabilistic methods: CQR, QRF, MC Dropout
- Economic evaluation with PnL/Sharpe/Sortino
- DM tests with HLN correction and BH FDR control
- 31 publication-quality figures, 22 result tables
- All data (raw + processed) and models included (except QRF >100MB)
"

# 5. Push
echo ""
echo "[5/5] Pushing to origin..."
git push origin main

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ DONE! Check: https://github.com/MufakirAnsari/Energy-Cost-Prediction"
echo "═══════════════════════════════════════════════════"
