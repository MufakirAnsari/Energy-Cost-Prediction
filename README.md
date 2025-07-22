```markdown
# ⚡️ Advanced Time Series Forecasting for Electricity Prices
### A Project Comparing Deep Learning, Statistical, and Ensemble Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-TensorFlow_2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

This repository contains the complete codebase for a master's thesis focused on forecasting day-ahead electricity prices in the Spanish market. It rigorously compares a diverse set of modern forecasting models — from classical statistical baselines to state-of-the-art deep learning architectures and an ensemble meta-model.

The primary goal is to identify the most effective modeling techniques through evaluation on real-world, high-frequency energy data, ensuring fair comparisons and reproducible results.

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

- **Benchmark Modern Architectures:** Implement and evaluate a broad spectrum of time series models, spanning classical statistical techniques to cutting-edge deep learning.  
- **Ensure Fair Comparison:** Create a controlled experimental setup ensuring equal footing (e.g., identical sequence lengths for deep learning models).  
- **Quantify Uncertainty:** Go beyond point forecasts by using probabilistic models (Bayesian LSTM, Quantile LGBM) to estimate prediction uncertainty — vital for risk management.  
- **Leverage Ensemble Power:** Investigate whether stacking diverse base models via a meta-learner improves forecast accuracy.  
- **Provide a Reproducible Framework:** Deliver a well-documented, modular codebase for easy reproduction and extensibility.

---

## 📊 Visualizing the Workflow

```

+------------------+    +------------------------+    +------------------------------+
\|  Raw Data (CSV)  | -> | 02\_preprocess.py       | -> | processed\_data.parquet       |
\| (energy & weather)|    | (Clean & Feature Eng.) |    | (Ready for Training)         |
+------------------+    +------------------------+    +------------------------------+
|
v
+---------------------------------------------------------------+
\| Step 3: Train Base Models (can run in parallel)               |
\|                                                               |
\|  - 03\_train\_lightgbm.py                                       |
\|  - 03\_train\_sarimax\_resampled.py                              |
\|  - 03\_train\_transformer.py                                    |
\|  - 03\_train\_autoformer.py                                     |
\|  - 03\_train\_bayesian\_lstm.py                                 |
+---------------------------------------------------------------+
|
v
+-------------------------+
\| Saved Base Models       |
\| (in /models folder)     |
+-------------------------+
|
v
+-------------------------+
\| 04\_ensemble.py          |
\| (Train Meta-Learner)    |
+-------------------------+
|
v
+-------------------------+
\| Final Ensemble Model    |
\| (enhanced\_ensemble\_model.joblib) |
+-------------------------+

```

---

## 🧠 Models Implemented

| Model                 | Category              | Key Feature                                            |
|-----------------------|-----------------------|--------------------------------------------------------|
| **SARIMAX**           | Statistical           | Classical seasonal ARIMA with exogenous variables       |
| **LightGBM**          | Gradient Boosting     | Fast tabular model for point forecasts                   |
| **LightGBM Quantile** | Gradient Boosting     | Predicts quantiles (p10, p50, p90) for uncertainty       |
| **Transformer**       | Deep Learning (Attention) | Standard attention-based sequence-to-sequence model    |
| **Autoformer**        | Deep Learning (Attention) | Transformer variant with Auto-Correlation for efficiency |
| **Bayesian Bi-LSTM**  | Deep Learning (RNN)   | RNN that outputs probabilistic forecasts                  |
| **Stacked Ensemble**  | Meta-Learning         | LightGBM meta-learner trained on base model predictions  |

---

## ⚙️ Technology Stack

- **Data Manipulation & Analysis:** Pandas, NumPy  
- **Deep Learning Framework:** TensorFlow (Keras API)  
- **Probabilistic Modeling:** TensorFlow Probability  
- **Gradient Boosting:** LightGBM  
- **Statistical Modeling:** pmdarima (Auto-ARIMA)  
- **Data Storage & I/O:** PyArrow (Parquet), Joblib  
- **Visualization:** Matplotlib, Seaborn  

---

## 📁 Project Structure

```

.
├── data/
│   ├── energy\_dataset.csv          # Raw energy generation and price data
│   ├── weather\_features.csv        # Raw weather data for Spanish cities
│   └── processed\_data.parquet      # Cleaned and feature-engineered dataset
│
├── models/                        # Trained model files
│
├── reports/                       # EDA plots and analyses
│
├── 01\_eda.py                     # Exploratory Data Analysis script
├── 02\_preprocess.py              # Data cleaning & feature engineering
├── 03\_train\_autoformer.py        # Train Autoformer model
├── 03\_train\_bayesian\_lstm.py     # Train Bayesian Bi-LSTM model
├── 03\_train\_lightgbm.py          # Train LightGBM (point & quantile)
├── 03\_train\_sarimax\_resampled.py # Train SARIMAX on resampled data
├── 03\_train\_transformer.py       # Train Transformer model
├── 04\_ensemble.py                # Train stacked ensemble model
├── config.py                    # Centralized configuration file
├── utils.py                     # Helper functions & custom layers
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview

````

---

## 🛠️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MufakirAnsari/Energy-Cost-Prediction.git
cd Energy-Cost-Prediction
````

### 2. Create and Activate a Virtual Environment

Using a dedicated environment ensures reproducibility.

```bash
conda create -n thesis-env python=3.11
conda activate thesis-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs standard TensorFlow. To use an NVIDIA GPU, ensure compatible NVIDIA drivers and CUDA Toolkit are installed.

### 4. Download the Data

Place `energy_dataset.csv` and `weather_features.csv` in the `data/` directory.
These datasets are available on [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

---

## ▶️ Execution Workflow

### Step 1: Preprocessing & Feature Engineering (mandatory)

```bash
python 02_preprocess.py
```

### Step 2: Train Base Models

Train models in this order (LightGBM must precede SARIMAX):

```bash
# Train tabular baseline (point & quantile models)
python 03_train_lightgbm.py

# Train statistical baseline (SARIMAX with feature selection)
python 03_train_sarimax_resampled.py

# Train deep learning models (order flexible)
python 03_train_transformer.py
python 03_train_autoformer.py
python 03_train_bayesian_lstm.py
```

### Step 3: Train Ensemble Model

```bash
python 04_ensemble.py
```

### Step 4: Evaluate & Analyze (Future Work)

Develop `05_evaluation.py` to load all models, assess performance on the test set using metrics like MAE, RMSE, MAPE, and visualize results.

---

## 📝 Methodological Notes

* **SARIMAX Resampling:** Original hourly SARIMAX (`m=24`) was infeasible on consumer hardware (OOM errors). Resampling to 3-hour intervals (`m=8`) enables feasible training while preserving daily seasonality.
* **Fair Deep Learning Comparison:** All sequential models (Transformer, Autoformer, LSTM) use a fixed sequence length (`SEQ_LENGTH=168` hours) from `config.py`.
* **Hyperparameter Tuning:** Current parameters provide a solid baseline. Future work should automate hyperparameter optimization (e.g., with Optuna or KerasTuner).

---

## 🔮 Future Work

* Automated hyperparameter tuning for all models.
* Comprehensive evaluation module with statistical significance tests (e.g., Diebold-Mariano).
* Economic simulations incorporating transaction costs to assess practical value of forecasts.
* Deployment: Containerize best models using Docker; serve via REST API (Flask/FastAPI).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

* Data sourced from the ["Energy consumption, generation, prices and weather"](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) Kaggle dataset.
* Thanks to the open-source community for the invaluable tools and resources.

---

```
```
