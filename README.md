```markdown
# Advanced Time Series Forecasting for Electricity Prices
### A Thesis Project Comparing Deep Learning, Statistical, and Ensemble Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete codebase for a master's thesis focused on forecasting the day-ahead electricity price in the Spanish market. The project implements and rigorously compares a diverse set of modern forecasting models, from statistical baselines to state-of-the-art deep learning architectures and an ensemble meta-model.

The primary goal is to determine the most effective modeling techniques by evaluating their performance on real-world, high-frequency energy data, emphasizing fair comparisons and reproducible results.

---

## Key Features

-   **Diverse Model Architectures:** Implements and compares a wide range of models:
    -   **Transformer:** The standard attention-based model.
    -   **Autoformer:** A Transformer variant designed for long-sequence forecasting with an Auto-Correlation mechanism.
    -   **Bayesian Bi-LSTM:** A recurrent neural network capable of producing probabilistic forecasts to quantify uncertainty.
    -   **LightGBM:** A powerful gradient-boosting machine for a strong, tabular baseline (point and quantile forecasts).
    -   **SARIMAX:** A robust statistical model (Seasonal Auto-Regressive Integrated Moving Average with eXogenous variables) serving as a classical baseline.
-   **Stacked Ensemble Model:** Combines the predictions of the base models using a LightGBM meta-learner to leverage the strengths of each.
-   **Rigorous and Fair Comparison:** All deep learning models are trained on sequences of the same length for a direct architectural comparison, configured in a central `config.py` file.
-   **Comprehensive Feature Engineering:** The preprocessing pipeline creates dozens of features, including cyclical time-based features (hour, day, month), holiday indicators, and lagged/rolling window statistics.
-   **Probabilistic Forecasting:** Includes models that predict a distribution or quantiles (Bayesian LSTM, Quantile LightGBM) rather than just a single point, which is crucial for risk management.
-   **Modular and Reproducible Codebase:** The project is structured logically into data processing, training, and evaluation scripts, making it easy to understand and reproduce the entire experimental workflow.

---

## Project Structure

```

.
├── data/
│   ├── energy\_dataset.csv          # Raw energy generation and price data
│   ├── weather\_features.csv        # Raw weather data for Spanish cities
│   └── processed\_data.parquet      # Final, cleaned, and feature-engineered dataset
│
├── models/                         # Directory where all trained models are saved
│   ├── autoformer\_model.keras
│   ├── bayesian\_model.keras
│   ├── lightgbm\_point\_model.joblib
│   ├── sarimax\_resampled\_model.joblib
│   └── ...
│
├── reports/                        # Output directory for EDA plots and analysis
│   ├── eda\_01\_full\_timeseries.png
│   └── ...
│
├── 01\_eda.py                       # Script for Exploratory Data Analysis
├── 02\_preprocess.py                # Script for data cleaning and feature engineering
├── 03\_train\_autoformer.py          # Training script for the Autoformer model
├── 03\_train\_bayesian\_lstm.py       # Training script for the Bayesian Bi-LSTM
├── 03\_train\_lightgbm.py            # Training script for LightGBM (point & quantile)
├── 03\_train\_sarimax\_resampled.py   # Training script for the feasible SARIMAX model
├── 03\_train\_transformer.py         # Training script for the standard Transformer
├── 04\_ensemble.py                  # Script to train the final stacked ensemble model
├── config.py                       # Central configuration for all parameters
├── utils.py                        # Custom Keras layers and helper functions
├── requirements.txt                # All Python dependencies for the project
└── README.md                       # This file

````

---

## Setup and Installation

Follow these steps to set up the environment and run the project.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment. This project was developed using Conda.

```bash
# Create a conda environment
conda create -n thesis-env python=3.11

# Activate the environment
conda activate thesis-env
```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

*Note on GPU Support:* The `requirements.txt` file installs TensorFlow. To leverage your NVIDIA GPU, ensure you have the appropriate NVIDIA drivers and CUDA Toolkit installed on your system.

### 4. Download the Data

Place the raw `energy_dataset.csv` and `weather_features.csv` files into the `data/` directory. These can be sourced from Kaggle.

---

## How to Run: Reproducing the Experiment

The scripts are designed to be run in the following order:

### Step 1: Exploratory Data Analysis (Optional)

Run the EDA script to generate plots and analysis of the raw data.

```bash
python 01_eda.py
```

### Step 2: Preprocessing and Feature Engineering

This step is mandatory. It cleans the raw data, engineers features, and saves the final `processed_data.parquet` file that all models use.

```bash
python 02_preprocess.py
```

### Step 3: Train the Base Models

Train each of the individual models. Note that the LightGBM model should be trained before the SARIMAX model, as SARIMAX uses it for feature selection.

```bash
# 1. Train the powerful tabular baseline (creates point and quantile models)
python 03_train_lightgbm.py

# 2. Train the statistical baseline (uses the LGBM model for feature selection)
python 03_train_sarimax_resampled.py

# 3. Train the deep learning models
python 03_train_transformer.py
python 03_train_autoformer.py
python 03_train_bayesian_lstm.py
```

After this step, your `models/` directory will be populated with all the trained base models.

### Step 4: Train the Ensemble Model

This script loads all the base models trained in the previous step and uses their predictions to train a final meta-learner.

```bash
python 04_ensemble.py
```

### Step 5: Evaluation (Future Work)

An evaluation script (`05_evaluation.py`) can be created to load all trained models (including the ensemble) and evaluate their performance on the test set using metrics like MAE, RMSE, and MAPE.

---

## Methodology and Key Decisions

* **SARIMAX Feasibility Compromise:** Initial attempts to train a SARIMAX model on hourly data (`m=24`) with exogenous features led to Out-Of-Memory errors due to the massive computational resources required. A pragmatic decision was made to resample the data to a 3-hour frequency (`m=8`), which allows the model to train successfully while still capturing daily seasonality. This is documented in `03_train_sarimax_resampled.py`.
* **Fair Comparison of Deep Learning Models:** To ensure a fair architectural comparison, all sequential deep learning models (Transformer, Autoformer, LSTM) are configured in `config.py` to use the same input sequence length (`SEQ_LENGTH = 168` hours).
* **Hyperparameter Tuning:** The parameters in `config.py` provide a strong and consistent baseline. For a fully exhaustive study, these parameters should be tuned for each model using a library like Optuna or KerasTuner to optimize performance on the validation set.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

* The dataset used in this project is the "Energy consumption, generation, prices and weather" dataset, available on [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

```
```

