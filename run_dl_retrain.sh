#!/bin/bash
set -e

echo "============================================="
echo "   RETRAINING DEEP LEARNING MODELS (VST)     "
echo "============================================="

echo "[1/4] Training BiLSTM (MC Dropout)..."
.venv/bin/python step_05_train_bilstm.py

echo "\n[2/4] Training PatchTST..."
.venv/bin/python step_06_train_patchtst.py

echo "\n[3/4] Training iTransformer..."
.venv/bin/python step_07_train_itransformer.py

echo "\n[4/4] Training N-HiTS..."
.venv/bin/python step_08_train_nhits.py

echo "\n============================================="
echo "   ALL DEEP LEARNING MODELS RETRAINED!       "
echo "============================================="
