# ⚡️ Advanced Time Series Forecasting for Electricity Prices
### A Thesis Project Comparing Deep Learning, Statistical, and Ensemble Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-TensorFlow_2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

This repository contains the complete codebase for a master's thesis focused on forecasting the day-ahead electricity price in the Spanish market. The project implements and rigorously compares a diverse set of modern forecasting models, from statistical baselines to state-of-the-art deep learning architectures and an ensemble meta-model.

The primary goal is to determine the most effective modeling techniques by evaluating their performance on real-world, high-frequency energy data, emphasizing fair comparisons and reproducible results.

---

## 📋 Table of Contents
1. [Project Goals](#-project-goals)
2. [Visualizing the Workflow](#-visualizing-the-workflow)
3. [Models Implemented](#-models-implemented)
4. [Technology Stack](#-technology-stack)
5. [Project Structure](#-project-structure)
6. [Setup and Installation](#-setup-and-installation)
7. [Execution Workflow](#-execution-workflow)
8. [Methodological Notes](#-methodological-notes)
9. [Future Work](#-future-work)
10. [License](#-license)
11. [Acknowledgments](#-acknowledgments)

---

## 🚀 Project Goals

-   **Benchmark Modern Architectures:** To implement and evaluate a range of time series models, from classical statistical methods to cutting-edge deep learning architectures.
-   **Ensure Fair Comparison:** To create a controlled experimental setup where models are compared on a level playing field (e.g., using identical sequence lengths for all deep learning models).
-   **Quantify Uncertainty:** To go beyond simple point forecasts by implementing probabilistic models (Bayesian LSTM, Quantile LGBM) that can estimate prediction uncertainty, which is critical for real-world risk management.
-   **Leverage Ensemble Power:** To investigate whether a stacked ensemble model, trained on the predictions of diverse base models, can outperform any single model.
-   **Provide a Reproducible Framework:** To deliver a well-documented and modular codebase that allows for easy reproduction and extension of the experimental results.

---

## 📊 Visualizing the Workflow

The project follows a clear, sequential pipeline from raw data to a final trained ensemble model.


```
+------------------+      +------------------------+      +------------------------------+
|    Raw Data      |  --> |   02_preprocess.py     |  --> |    processed_data.parquet    |
|    (CSV files)   |      | (Clean & Feature Eng.) |      |     (Ready for Training)     |
+------------------+      +------------------------+      +------------------------------+
         |                          |                             |
         |                          |                             |
         |                          v                             |
         |              +----------------------------------+      |
         |              | Step 3: Train Base Models         |      |
         |              |          (in parallel)            |      |
         |              |                                  |      |
         +--------------| - 03_train_lightgbm.py           |      |
                        | - 03_train_sarimax_resampled.py   |      |
                        | - 03_train_transformer.py         |      |
                        | - 03_train_autoformer.py          |      |
                        | - 03_train_bayesian_lstm.py       |      |
                        +----------------------------------+      |
                                   |                                  |
                                   |                                  |
                                   v                                  v
                      +----------------------------+      +----------------------------+
                      | Saved Base Models           |      |                            |
                      | (in /models folder)         |      |                            |
                      +----------------------------+      +----------------------------+
                                   |
                                   |
                                   v
                      +----------------------------+
                      |       04_ensemble.py       |
                      | (Train Meta-Learner on     |
                      |  Base Model Predictions)   |
                      +----------------------------+
                                   |
                                   |
                                   v
                      +----------------------------+
                      |    Final Ensemble Model    |
                      | (enhanced_ensemble_model.joblib) |
                      +----------------------------+
```


## 🧠 Models Implemented

A diverse set of models was chosen to cover different forecasting paradigms.

| Model                   | Category                | Key Characteristic                                         |
| ----------------------- | ----------------------- | ---------------------------------------------------------- |
| **SARIMAX**             | Statistical             | Classical time series model with seasonality & external vars |
| **LightGBM**            | Gradient Boosting       | Powerful, fast tabular model for point forecasts           |
| **LightGBM Quantile**   | Gradient Boosting       | Predicts quantiles (p10, p50, p90) for uncertainty         |
| **Transformer**         | Deep Learning (Attention) | The standard architecture for sequence-to-sequence tasks   |
| **Autoformer**          | Deep Learning (Attention) | Advanced Transformer with Auto-Correlation for efficiency  |
| **Bayesian Bi-LSTM**    | Deep Learning (RNN)     | A recurrent model that predicts a full probability distribution |
| **Stacked Ensemble**    | Meta-Learning           | LightGBM model trained on the outputs of all other models  |

---

## ⚙️ Technology Stack

-   **Data Manipulation & Analysis:** Pandas, NumPy
-   **Deep Learning Framework:** TensorFlow (with Keras API)
-   **Probabilistic Modeling:** TensorFlow Probability
-   **Gradient Boosting:** LightGBM
-   **Statistical Modeling:** `pmdarima` (for Auto-ARIMA)
-   **Data Storage & I/O:** `pyarrow` (for Parquet files), `joblib`
-   **Data Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

.
├── data/
│   ├── energy_dataset.csv          # Raw energy generation and price data
│   ├── weather_features.csv        # Raw weather data for Spanish cities
│   └── processed_data.parquet      # Final, cleaned, and feature-engineered dataset
│
├── models/                        # Directory where all trained models are saved
│
├── reports/                       # Output directory for EDA plots and analysis
│
├── 01_eda.py                     # Script for Exploratory Data Analysis
├── 02_preprocess.py              # Script for data cleaning and feature engineering
├── 03_train_autoformer.py        # Training script for the Autoformer model
├── 03_train_bayesian_lstm.py     # Training script for the Bayesian Bi-LSTM
├── 03_train_lightgbm.py          # Training script for LightGBM (point & quantile)
├── 03_train_sarimax_resampled.py # Training script for the feasible SARIMAX model
├── 03_train_transformer.py       # Training script for the standard Transformer
├── 04_ensemble.py                # Script to train the final stacked ensemble model
├── config.py                    # Central configuration for all parameters
├── utils.py                     # Custom Keras layers and helper functions
├── requirements.txt             # All Python dependencies for the project
└── README.md                    # This file

````markdown
## 🛠️ Setup and Installation

Follow these steps to set up the environment and run the project.

#### 1. Clone the Repository
```bash
git clone https://github.com/MufakirAnsari/Energy-Cost-Prediction.git
cd Energy-Cost-Prediction
````

#### 2. Create and Activate a Virtual Environment

Using a dedicated environment is crucial for reproducibility.

```bash
# Create a conda environment
conda create -n env python=3.11

# Activate the environment
conda activate env
```

#### 3. Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

> **Note on GPU Support:** The `requirements.txt` installs the standard TensorFlow package. To leverage an NVIDIA GPU, ensure you have compatible NVIDIA drivers and CUDA Toolkit installed on your system.

#### 4. Download the Data

Place the raw `energy_dataset.csv` and `weather_features.csv` files into the `data/` directory. These can be sourced from the [Kaggle: Energy & Weather Dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

---

## ▶️ Execution Workflow

The scripts are designed to be run in a specific order to ensure dependencies are met.

#### Step 1: Preprocessing and Feature Engineering (Mandatory)

This script is the foundation for all models. It cleans the raw data, engineers features, and saves the final `processed_data.parquet` file.

```bash
python 02_preprocess.py
```

#### Step 2: Train the Base Models

Train each of the individual models. Note that LightGBM must be trained before SARIMAX, as SARIMAX uses the trained LightGBM model for its feature selection process.

```bash
# 1. Train the powerful tabular baseline (creates point and quantile models)
python 03_train_lightgbm.py

# 2. Train the statistical baseline (uses the LGBM model for feature selection)
python 03_train_sarimax_resampled.py

# 3. Train the deep learning models (can be run in any order)
python 03_train_transformer.py
python 03_train_autoformer.py
python 03_train_bayesian_lstm.py
```

After this step, your `models/` directory will be populated with all the trained base models.

#### Step 3: Train the Ensemble Model

This script loads all the base models trained in the previous step and uses their predictions on a validation set to train a final meta-learner.

```bash
python 04_ensemble.py
```

#### Step 4: Evaluate and Analyze

With all models trained, the next step is to write an evaluation script (`05_evaluation.py`) to load them and compare their performance on the held-out test set using metrics like MAE, RMSE, and MAPE.

---

## 📝 Methodological Notes

* **SARIMAX Feasibility Compromise:** Initial attempts to train SARIMAX on hourly data (`m=24`) with exogenous features led to out-of-memory errors. Resampling to a 3-hour frequency (`m=8`) enables feasible training while preserving daily seasonality (see `03_train_sarimax_resampled.py`).

* **Fair Comparison of Deep Learning Models:** All sequential deep learning models (Transformer, Autoformer, LSTM) use the same input sequence length (`SEQ_LENGTH = 168` hours) configured in `config.py`.

* **Hyperparameter Tuning:** Parameters in `config.py` provide a solid baseline. Full optimization should involve automated tuning (e.g., with Optuna or KerasTuner).

---

## 🔮 Future Work

* Implement automated hyperparameter search for all models to maximize performance.
* Develop comprehensive evaluation scripts including statistical tests (e.g., Diebold-Mariano) and rich visualizations.
* Simulate trading strategies using model forecasts, incorporating transaction costs to assess economic value.
* Containerize the best model using Docker and deploy as a REST API with Flask or FastAPI.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

This work uses the ["Energy consumption, generation, prices and weather"](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) dataset from Kaggle.

Special thanks to the open-source community for providing invaluable tools and support.

```
```
