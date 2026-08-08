#!/bin/bash
set -e
export TF_CPP_MIN_LOG_LEVEL="3"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

cd /home/ansari/Desktop/Energy/V2

echo "=== 1/7: Running Chronos-2 Inference (Covariates) ==="
.venv/bin/python step_09b_chronos2_inference.py

echo "=== 2/7: Running Ensemble Meta-Learner (BiLSTM Fix) ==="
.venv/bin/python step_12_ensemble.py

echo "=== 3/7: Running Rolling BiLSTM (Warm-Start) ==="
.venv/bin/python step_26_rolling_bilstm.py

echo "=== 4/7: Running Rolling PatchTST (Warm-Start) ==="
.venv/bin/python step_21b_rolling_patchtst.py

echo "=== 5/7: Generating All Tables ==="
.venv/bin/python step_30_generate_tables.py || true

echo "=== 6/7: Generating All Figures ==="
.venv/bin/python step_31_generate_figures.py || true

echo "=== 7/7: DONE ==="
