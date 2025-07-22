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

```ascii
+--------------------------+         +--------------------------+         +----------------------------+
|        Raw Data          |    →    |    02_preprocess.py      |    →    |  processed_data.parquet    |
|       (CSV files)        |         |  (Clean & Feature Eng.)  |         |   (Ready for Training)     |
+--------------------------+         +--------------------------+         +----------------------------+
                                                                                       |
                                                                                       ↓
                                         +-------------------------------------------+
                                         |      Step 3: Train Base Models            |
                                         |  (LGBM must run before SARIMAX)           |
                                         |                                           |
                                         |   • 03_train_lightgbm.py                  |
                                         |   • 03_train_sarimax_resampled.py         |
                                         |   • 03_train_transformer.py               |
                                         |   • 03_train_autoformer.py                |
                                         |   • 03_train_bayesian_lstm.py             |
                                         +-------------------------------------------+
                                                                                       |
                                                                                       ↓
                                         +-------------------------------------------+
                                         |      Saved Base Models                    |
                                         |      (in /models folder)                  |
                                         +-------------------------------------------+
                                                                                       |
                                                                                       ↓
                                         +-------------------------------------------+
                                         |      04_ensemble.py                       |
                                         | (Train Meta-Learner on Predictions)       |
                                         +-------------------------------------------+
                                                                                       |
                                                                                       ↓
                                         +-------------------------------------------+
                                         |      🏆 Final Ensemble Model              |
                                         | (enhanced_ensemble_model.joblib)          |
                                         +-------------------------------------------+


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
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

.
├── data/
│ ├── energy_dataset.csv # Raw energy generation and price data
│ ├── weather_features.csv # Raw weather data for Spanish cities
│ └── processed_data.parquet # Final, cleaned, and feature-engineered dataset
│
├── models/ # Directory where all trained models are saved
│
├── reports/ # Output directory for EDA plots and analysis
│
├── 01_eda.py # Script for Exploratory Data Analysis
├── 02_preprocess.py # Script for data cleaning and feature engineering
├── 03_train_autoformer.py # Training script for the Autoformer model
├── 03_train_bayesian_lstm.py # Training script for the Bayesian Bi-LSTM
├── 03_train_lightgbm.py # Training script for LightGBM (point & quantile)
├── 03_train_sarimax_resampled.py # Training script for the feasible SARIMAX model
├── 03_train_transformer.py # Training script for the standard Transformer
├── 04_ensemble.py # Script to train the final stacked ensemble model
├── config.py # Central configuration for all parameters
├── utils.py # Custom Keras layers and helper functions
├── requirements.txt # All Python dependencies for the project
└── README.md # This file

Generated code
---

## 🛠️ Setup and Installation

Follow these steps to set up the environment and run the project.

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
2. Create and Activate a Virtual Environment

Using a dedicated environment is crucial for reproducibility.

Generated bash
# Create a conda environment
conda create -n thesis-env python=3.11

# Activate the environment
conda activate thesis-env
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
3. Install Dependencies

All required packages are listed in requirements.txt.```bash
pip install -r requirements.txt

Generated code
> **Note on GPU Support:** The `requirements.txt` file installs the standard TensorFlow package. To leverage an NVIDIA GPU, ensure you have the appropriate NVIDIA drivers and CUDA Toolkit installed on your system that are compatible with your TensorFlow version.

#### 4. Download the Data
Place the raw `energy_dataset.csv` and `weather_features.csv` files into the `data/` directory. These can be sourced from the [Kaggle: Energy & Weather Dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

---

## ▶️ Execution Workflow

The scripts are designed to be run in a specific order to ensure dependencies are met.

#### Step 1: Preprocessing and Feature Engineering (Mandatory)
This script is the foundation for all models. It cleans the raw data, engineers features, and saves the final `processed_data.parquet` file.
```bash
python 02_preprocess.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Step 2: Train the Base Models

Train each of the individual models. Note that LightGBM must be trained before SARIMAX, as SARIMAX uses the trained LightGBM model for its feature selection process.

Generated bash
# 1. Train the powerful tabular baseline (creates point and quantile models)
python 03_train_lightgbm.py

# 2. Train the statistical baseline (uses the LGBM model for feature selection)
python 03_train_sarimax_resampled.py

# 3. Train the deep learning models (can be run in any order)
python 03_train_transformer.py
python 03_train_autoformer.py
python 03_train_bayesian_lstm.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

After this step, your models/ directory will be populated with all the trained base models.

Step 3: Train the Ensemble Model

This script loads all the base models trained in the previous step and uses their predictions on a validation set to train a final meta-learner.

Generated bash
python 04_ensemble.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Step 4: Evaluate and Analyze

With all models trained, the next step is to write an evaluation script (05_evaluation.py) to load them and compare their performance on the held-out test set using metrics like MAE, RMSE, and MAPE.

📝 Methodological Notes

SARIMAX Feasibility Compromise: Initial attempts to train a SARIMAX model on hourly data (m=24) with exogenous features led to recurring Out-Of-Memory errors due to the massive computational resources required. A pragmatic decision was made to resample the data to a 3-hour frequency (m=8), which allows the model to train successfully on consumer-grade hardware while still capturing daily seasonality. This is documented in 03_train_sarimax_resampled.py.

Fair Comparison of Deep Learning Models: To ensure a fair and direct architectural comparison, all sequential deep learning models (Transformer, Autoformer, LSTM) are configured in config.py to use the same input sequence length (SEQ_LENGTH = 168 hours).

Hyperparameter Tuning: The parameters in config.py provide a strong and consistent baseline for comparison. For a fully exhaustive study, these parameters should be systematically tuned for each model (e.g., using Optuna or KerasTuner) to optimize performance on the validation set.

🔮 Future Work

Rigorous Hyperparameter Tuning: Implement an automated hyperparameter search for all models to ensure each is performing at its best.

Comprehensive Evaluation Module: Build out the 05_evaluation.py script to include statistical significance tests (e.g., Diebold-Mariano test) and detailed visualizations of prediction errors.

Economic Simulation: Develop a module to simulate a trading strategy based on the model forecasts, incorporating transaction costs to evaluate the real-world economic value of each model's accuracy and uncertainty estimates.

Deployment: Containerize the best-performing model using Docker and deploy it as a REST API using a framework like Flask or FastAPI.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments

This work relies on the "Energy consumption, generation, prices and weather" dataset, generously made available on Kaggle.

The development of this project was heavily influenced by the open-source community and the incredible tools they provide.
