#!/usr/bin/env bash
# install_deps.sh — Install all V2 pipeline dependencies
# Run once before starting the pipeline.

PYTHON="$(dirname "$0")/.venv/bin/python"
PIP="$PYTHON -m pip"

echo "Installing V2 EPF Pipeline dependencies..."

# Core ML
$PIP install --upgrade \
    lightgbm \
    xgboost \
    scikit-learn \
    joblib \
    statsforecast \
    properscoring

# Deep Learning — TensorFlow + TFP (for BiLSTM)
$PIP install --upgrade \
    "tensorflow>=2.14" \
    tensorflow-probability

# Deep Learning — PyTorch + NeuralForecast (for PatchTST, iTransformer, N-HiTS)
$PIP install --upgrade \
    torch \
    neuralforecast

# Chronos (Amazon foundation model)
$PIP install --upgrade \
    "chronos-forecasting @ git+https://github.com/amazon-science/chronos-forecasting.git"

# Other utilities
$PIP install --upgrade \
    scipy \
    pandas \
    numpy \
    pyarrow \
    requests \
    gridstatus

echo "✅ All dependencies installed."
