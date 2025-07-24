# Deep Learning and Ensemble Methods for Electricity Price Forecasting

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.13+-D00000?style=for-the-badge&logo=keras&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1-006400?style=for-the-badge&logo=microsoft&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

A comprehensive repository for forecasting the Spanish electricity market price using a variety of advanced time series models. This project rigorously compares classical statistical methods, modern deep learning architectures, and a powerful stacked ensemble model to determine the most effective forecasting strategy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Models Implemented](#models-implemented)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Usage Workflow](#usage-workflow)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Forecasting electricity prices is a notoriously difficult task due to the complex interplay of factors like weather, demand, generation mix, and market dynamics. This project provides a robust framework to tackle this challenge. It includes a complete pipeline from data ingestion and feature engineering to training, evaluation, and ensembling of multiple state-of-the-art models.

The primary goal is to perform a fair and rigorous comparison between different model families to provide insights into their respective strengths and weaknesses on high-frequency energy data.

---

## Project Architecture

The project follows a clear, modular, and reproducible workflow, from data ingestion to ensemble model training. The complete architecture is visualized in the following diagram:

![Project Workflow](reports/Workflow%20for%20Energy%20Forecasting.png)

---

## Key Features

- **Comprehensive EDA:** Exploratory analysis to uncover trends, patterns, and seasonality.
- **Rich Feature Engineering:** Includes date/time cyclicals, holidays, lags, and rolling statistics.
- **Diverse Model Implementations:** Statistical, tree-based, and deep learning models.
- **Fair Deep Learning Benchmarking:** All models trained using the same input sequence (168 hours).
- **Advanced Architectures:** Includes **Transformer** and **Autoformer** for long-sequence modeling.
- **Probabilistic Forecasting:** With **Bayesian Bi-LSTM** and **LightGBM Quantile Regression**.
- **Stacked Ensemble:** Uses predictions from all models as input to a meta-learner for optimal performance.
- **Modular and Configurable:** Centralized configuration via `config.py`.

---

## Models Implemented

| Model Name        | Type         | Description |
|------------------|--------------|-------------|
| **SARIMAX**       | Statistical  | Classical model with seasonality and exogenous regressors (resampled to 3H). |
| **LightGBM**      | Tree-Based   | Strong baseline for tabular data, fast and scalable. |
| **LGBM Quantile** | Tree-Based   | Outputs prediction intervals (p10, p50, p90). |
| **Bayesian Bi-LSTM** | Recurrent DL | Bi-directional LSTM with uncertainty via dropout/Bayesian inference. |
| **Transformer**   | Attention DL | Standard Transformer with multi-head attention. |
| **Autoformer**    | Attention DL | Transformer variant optimized for long sequences using auto-correlation. |
| **Stacked Ensemble** | Meta-Learner | Combines all base model predictions using LightGBM. |

---

```bash
.
├── 01_eda.py                     # Perform Exploratory Data Analysis (EDA)
├── 02_preprocess.py              # Clean data and engineer features
├── 03_train_lightgbm.py          # Train LightGBM (point + quantile) models
├── 03_train_sarimax_resampled.py # Train SARIMAX model on resampled data
├── 03_train_transformer.py       # Train standard Transformer model
├── 03_train_bayesian_lstm.py     # Train Bayesian Bi-LSTM (with uncertainty)
├── 03_train_autoformer.py        # Train Autoformer for long-sequence forecasting
├── 04_ensemble.py                # Train Stacked Ensemble meta-model
├── config.py                     # Centralized configuration for all modules
├── utils.py                      # Utility functions and custom Keras layers
├── requirements.txt              # List of required Python packages
├── README.md                     # Project overview and usage guide

├── data/                         # Input datasets
│   ├── energy_dataset.csv        # Raw energy price data
│   └── weather_features.csv      # Raw weather data

├── models/                       # Saved model artifacts
│   ├── *.keras                   # Deep learning models
│   └── *.joblib                  # Tree-based and statistical models

└── reports/                      # Visual reports and diagrams
    ├── eda_.png                  # Plots from EDA
    └── Workflow for Energy Forecasting.png  # Project architecture diagram
```


---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MufakirAnsari/Energy-Cost-Prediction.git
cd Energy-Cost-Prediction
```
### 2. Create a Virtual Environment

```bash
conda create -n env python=3.11
conda activate env
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Add the Data


Create a folder called data/ and place the following files inside:
    `energy_dataset.csv`
    `weather_features.csv`


## Usage Workflow

### Step 1: Preprocess the Data
```bash
python 02_preprocess.py
```

### Step 2: Train Base Models
# Train LightGBM (important to run first)
```bash
python 03_train_lightgbm.py
```
# Train SARIMAX using LightGBM output
```bash
python 03_train_sarimax_resampled.py
```
# Train Deep Learning Models
```bash
python 03_train_transformer.py
python 03_train_bayesian_lstm.py
python 03_train_autoformer.py
```

### Step 3: Train the Stacked Ensemble

```bash
python 04_ensemble.py
```
The final ensemble model will be saved as:

```
models/enhanced_ensemble_model.joblib
```

## Configuration
All core settings (file paths, hyperparameters, etc.) are located in `config.py`.


```python
# --- Sequential Model Parameters ---
SEQ_LENGTH = 168
PRED_LENGTH = 24

# --- Transformer & Autoformer ---
D_MODEL = 64
NUM_HEADS = 4
D_FF = 128
DROPOUT = 0.2

# --- LightGBM Parameters ---
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 1500,
    # more params...
}
```

 <details> <summary><strong>📦 requirements.txt</strong></summary> 


```
pandas
numpy
tensorflow
tensorflow-probability
scikit-learn
pmdarima
lightgbm
holidays
joblib
matplotlib
seaborn
pyarrow
```
</details>


## Contributing

Contributions are welcome!

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/AmazingNewModel`
3. Commit your changes: `git commit -m 'Add new model'`
4. Push to the branch: `git push origin feature/AmazingNewModel`
5. Open a Pull Request ✔️




