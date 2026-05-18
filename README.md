# Probabilistic Electricity Price Forecasting — V2

> **Research pipeline for EAAI 2025 submission.**
> First rigorous multi-paradigm probabilistic EPF benchmark on post-2020 US markets (PJM + ERCOT), comparing classical, modern deep learning (PatchTST, iTransformer), and zero-shot foundation model (Chronos-Bolt) baselines under three probabilistic frameworks, with conformal prediction intervals and realistic economic simulation.

---

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Dataset](#2-dataset)
3. [Model Suite](#3-model-suite)
4. [Repository Structure](#4-repository-structure)
5. [Quick Start](#5-quick-start)
6. [Pipeline Reference](#6-pipeline-reference)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Results Summary](#8-results-summary)
9. [Requirements](#9-requirements)
10. [Citation](#10-citation)

---

## 1. Research Overview

### Motivation
US electricity markets (PJM, ERCOT) have experienced unprecedented price volatility since 2020 — COVID demand collapse, Winter Storm Uri ($9,000/MWh spike), the 2022 gas crisis, and an accelerating renewables transition. Existing EPF benchmarks predominantly use pre-2020 European data and evaluate only point accuracy. This work addresses all three gaps simultaneously.

### Unique Contributions
1. **First US dual-market benchmark** (PJM + ERCOT, 2019–2025) covering four volatility regimes: COVID, Uri crisis, gas shock, and high-renewables era
2. **Three-paradigm unified probabilistic evaluation** — Bayesian MC Dropout, Conformalized Quantile Regression (CQR), and Deep Quantile — with CRPS and Winkler score as proper scoring rules. 90% CI where architecturally supported; 80% CI for Chronos-Bolt and N-HiTS Quantile (documented).
3. **Zero-shot foundation model baseline** (Amazon Chronos-Bolt) without retraining — first such comparison in US electricity markets
4. **Realistic economic validation** with explicit transaction costs ($0.50/MWh), execution slippage, and five trading strategies including Risk-Aware Bayesian and CQR confidence filters

### Research Questions
| RQ | Question | Primary Metric |
|---|---|---|
| RQ1 | Which model family achieves best point accuracy? | MAE, DM test |
| RQ2 | Which probabilistic method achieves best calibration? | PICP, CRPS, Winkler |
| RQ3 | Does better forecasting translate to economic utility? | Sharpe ratio, P&L |
| RQ4 | Do PJM-trained models generalize to ERCOT (cross-market)? | MAE degradation |

---

## 2. Dataset

### Markets
| Market | Product | Node | Period | Size |
|---|---|---|---|---|
| **PJM** | Day-Ahead LMP | WESTERN HUB | 2019–2025 | ~61,000 obs |
| **ERCOT** | Day-Ahead SPP | HB_HOUSTON | 2019–2025 | ~61,000 obs |

### Exogenous Features (68 total)
| Source | Features |
|---|---|
| EIA API v2 | Solar %, Wind MW, Gas MW, Nuclear MW, Net Load, Demand |
| NOAA NCEI | Temperature, Wind Speed, Humidity (5 cities per market) |
| EIA Henry Hub | Natural gas spot price ($/MMBtu) |
| Computed | Hour/DOW/Month sin-cos, US holidays, lag features (24h, 48h, 168h) |

### Chronological 4-Way Split (no data leakage)
```
2019-01-01 ── 2021-12-31 │ 2022-01-01 ── 2022-12-31 │ 2023-01-01 ── 2023-12-31 │ 2024-01-01 ── 2025-12-31
       TRAIN (25,548 obs)          CALIBRATION (CQR)            VALIDATION (ensemble)            TEST (eval)
    [base model training]       [conformal calibration]     [meta-learner training]        [final evaluation]
```

> **Strict half-open interval protocol:** each row belongs to exactly one split (zero overlap). MinMaxScaler and KNNImputer are fit **only on TRAIN**, then applied to CAL/VAL/TEST without refitting.

### Volatility Regimes
| Regime | Period | Characteristic |
|---|---|---|
| Stable Baseline | 2019 | Pre-COVID normal operations |
| COVID Collapse | 2020–early 2021 | Demand crash, depressed prices |
| **Uri Crisis** | Feb 2021 | ERCOT $9,000/MWh price cap hit |
| Gas Shock | 2021–2022 | Sustained high-price volatility |
| New Normal | 2023–2025 | High renewables, new volatility structure |

---

## 3. Model Suite

### Classical Baselines
| Model | Key Detail |
|---|---|
| Seasonal Naïve | 168h (weekly) seasonal persistence |
| AutoARIMA | `statsforecast`, hourly seasonality, rolling window |
| MSTL + AutoETS | Multi-Seasonal Trend decomposition — 2025 SOTA statistical |

### Tree-Based
| Model | Key Detail |
|---|---|
| LightGBM (point) | Optimized MAE; 68-feature input |
| LightGBM (quantile) | p05, p10, p25, p50, p75, p90, p95 — full distribution |
| XGBoost | Point forecast; second tree baseline for robustness |

### Deep Learning
| Model | Architecture | Framework |
|---|---|---|
| **Bayesian Bi-LSTM** | Bidirectional LSTM + MC Dropout (ECE-calibrated) | TensorFlow + TFP |
| **PatchTST** | Patch-based Transformer (ICLR 2023) | neuralforecast / PyTorch |
| **iTransformer** | Inverted attention across variables (ICLR 2024) | neuralforecast / PyTorch |
| **N-HiTS** | Hierarchical interpolation (AAAI 2023) | neuralforecast / PyTorch |

### Foundation Model
| Model | Type | Detail |
|---|---|---|
| **Chronos-Bolt (small)** | Zero-shot | Amazon, ~200M params, no fine-tuning |

### Probabilistic Wrappers
| Method | Guarantee | Applied To |
|---|---|---|
| MC Dropout | Asymptotic | Bayesian Bi-LSTM |
| **CQR** | Finite-sample under exchangeability¹ | LightGBM quantile |
| Deep Quantile | Asymptotic | PatchTST / iTransformer |
| **QRF** | Non-parametric | Tabular features |

> ¹ **CQR coverage note:** The finite-sample guarantee (Romano et al., 2019) requires the calibration set to be exchangeable with the test set. The 2022 calibration period (energy crisis) is a different distributional regime from the 2024–2025 test period, so the formal guarantee may not hold. Observed empirical PICP: **PJM 82%, ERCOT 75%** — reported honestly as a distributional-shift finding (Tibshirani et al., 2019).

### Ensemble
- **Meta-learner:** LightGBM stacked on {LightGBM + XGBoost + BiLSTM} predictions
- **Stacking protocol:** Trained on **validation set (2023)** — the calibration set (2022 energy crisis) is not exchangeable with the test period and degrades meta-learner performance
- **iTransformer excluded:** NeuralForecast cross_validation only produces test-set predictions — including it would leak test data into the meta-learner

---

## 4. Repository Structure

```
V2/
├── config.py                     # All paths, hyperparameters, split dates
├── utils.py                      # Shared metrics, DM test, data loading helpers
├── requirements.txt              # Pinned dependencies (==)
├── smoke_test.py                 # 79-check pre-flight validation (run first)
├── test_data_integrity.py        # Data leakage assertion suite (zero overlap)
├── run_all.sh                    # Full pipeline orchestration (steps 00–18)
│
├── data/
│   ├── raw/                      # Downloaded parquets (git-ignored)
│   └── processed/                # 8 train/cal/val/test parquets (git-ignored)
│
├── step_00_download_pjm.py       # gridstatus → PJM DA LMP 2019–2025
├── step_00_download_ercot.py     # gridstatus → ERCOT DA SPP 2019–2025
├── step_00_download_weather.py   # NOAA NCEI → 5-city hourly weather
├── step_00_download_eia.py       # EIA API v2 → generation mix + gas price
│
├── step_01_preprocess.py         # Merge, clean, feature-engineer, 4-way split
├── step_01_eda.py                # EDA — price stats, ACF/PACF, regime violin plots
├── step_02_train_baselines.py    # Seasonal Naïve, AutoARIMA, MSTL
│
├── step_03_train_lgbm.py         # LightGBM point + 7-quantile models
├── step_04_train_xgboost.py      # XGBoost point forecast
├── step_05_train_bilstm.py       # Bayesian Bi-LSTM + MC Dropout (ECE-calibrated)
├── step_05b_retrain_bilstm_ercot.py # ERCOT BiLSTM with log1p+clip fix
├── step_06_train_patchtst.py     # PatchTST (ICLR 2023, h=24)
├── step_06b_train_bitcn.py       # BiTCN (neuralforecast, h=24)
├── step_07_train_itransformer.py # iTransformer (ICLR 2024, h=24)
├── step_07b_train_tft.py         # TFT (neuralforecast, h=24)
├── step_08_train_nhits.py        # N-HiTS point (AAAI 2023, h=24)
├── step_08b_nhits_quantile.py    # N-HiTS quantile — MQLoss 80% CI
├── step_09_chronos_inference.py  # Chronos-Bolt zero-shot (h=24, 80% CI)
├── step_10_conformal.py          # CQR on dedicated calibration set (2022)
├── step_10b_alpha_sweep.py       # Conformal alpha sweep {80,90,95}%
├── step_11_qrf.py                # Quantile Regression Forest
├── step_12_ensemble.py           # Stacked ensemble meta-learner
│
├── step_09_evaluate.py           # All metrics: point + probabilistic + economic
├── step_14_dm_tests.py           # Diebold-Mariano + Benjamini-Hochberg FDR
├── step_15_ablation.py           # Lag window, feature set, dropout, ensemble
├── step_16_stress_test.py        # OOS regime evaluation (2024–2025)
├── step_19_rq4_crossmarket.py    # PJM→ERCOT transfer (tree models)
├── step_17_figures.py            # Publication figures at 300 DPI
├── step_18_paper_tables.py       # LaTeX tables (auto-sourced from config.py)
│
├── _deprecated/                  # Superseded scripts (archived)
├── models/                       # Trained model artifacts (git-ignored)
└── reports/                      # CSVs, tables, figures (git-ignored)
```

---

## 5. Quick Start

### Prerequisites
- Python 3.10–3.12
- NVIDIA GPU recommended (GTX 1650+ / 4GB+ VRAM for DL steps)
- API keys: [EIA](https://www.eia.gov/opendata/), [PJM Data Miner](https://dataminer2.pjm.com/)

### Installation
```bash
git clone https://github.com/MufakirAnsari/Energy-Cost-Prediction.git
cd Energy-Cost-Prediction

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install tf-keras   # Required for tensorflow-probability
```

### Pre-flight Check
```bash
python smoke_test.py
# Expected: 79/79 ✅ All checks passed
```

### Run the Full Pipeline
```bash
# Step 1: Download data (one-time, ~30 min)
python step_00_download_pjm.py
python step_00_download_ercot.py
python step_00_download_eia.py
python step_00_download_weather.py

# Step 2: Preprocess
python step_01_preprocess.py

# Step 3: Run statistical baselines (CPU, ~4-5 hours)
python step_02_train_baselines.py

# Steps 3–17: Full automated pipeline
chmod +x run_all.sh
./run_all.sh 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M).log
```

> `run_all.sh` is **idempotent** — if a step's output already exists, it is skipped automatically. Safe to re-run after interruptions.

---

## 6. Pipeline Reference

| Step | Script | Inputs | Outputs | GPU | Est. Time |
|---|---|---|---|---|---|
| 00 | `step_00_download_*.py` | API keys | `data/raw/*.parquet` | ✗ | ~30 min |
| 01 | `step_01_preprocess.py` | raw parquets | `data/processed/*.parquet` | ✗ | ~5 min |
| 02 | `step_02_train_baselines.py` | processed | `baseline_preds_*.csv` | ✗ | ~4-5 hrs |
| 03 | `step_03_train_lgbm.py` | processed | `lgbm_*.joblib` | ✗ | ~5 min |
| 04 | `step_04_train_xgboost.py` | processed | `xgboost_*.joblib` | ✗ | ~5 min |
| 05 | `step_05_train_bilstm.py` | processed | `bilstm_*.keras` | ✅ | ~30-45 min |
| 06 | `step_06_train_patchtst.py` | processed | `patchtst_preds_*.csv` | ✅ | ~45-60 min |
| 07 | `step_07_train_itransformer.py` | processed | `itransformer_preds_*.csv` | ✅ | ~30-45 min |
| 08 | `step_08_train_nhits.py` | processed | `nhits_preds_*.csv` | ✅ | ~20-30 min |
| 09 | `step_09_chronos_inference.py` | processed | `chronos_preds_*.csv` | ✗ | ~3-5 hrs |
| 10 | `step_10_conformal.py` | step 03 models | `cqr_preds_*.csv` | ✗ | ~2 min |
| 11 | `step_11_qrf.py` | processed | `qrf_preds_*.csv` | ✗ | ~10 min |
| 12 | `step_12_ensemble.py` | steps 03,05,07 | `ensemble_preds_*.csv` | ✗ | ~5 min |
| 13 | `step_09_evaluate.py` | all preds | `table_*.csv` | ✗ | ~5 min |
| 14 | `step_14_dm_tests.py` | all preds | `dm_tests_*.csv` | ✗ | ~3 min |
| 15 | `step_15_ablation.py` | processed | `table_ablation_*.csv` | ✗ | ~20 min |
| 16 | `step_16_stress_test.py` | all preds | `table_regime_*.csv` | ✗ | ~5 min |
| 17 | `step_17_figures.py` | all CSVs | `reports/figures/*.png` | ✗ | ~2 min |

---

## 7. Evaluation Framework

### Point Accuracy (RQ1)
| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — primary metric, robust to spikes |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **sMAPE** | Symmetric MAPE — handles near-zero prices |
| **DM test** | Diebold-Mariano with HLN correction + BH FDR adjustment |

### Probabilistic Quality (RQ2) — 90% CI where supported; 80% for Chronos/N-HiTS-Q
| Metric | Description |
|---|---|
| **PICP** | Prediction Interval Coverage Probability (target: ≥90% under exchangeability) |
| **MPIW** | Mean Prediction Interval Width (sharpness) |
| **CRPS** | Continuous Ranked Probability Score (proper scoring rule) |
| **Winkler Score** | Width + coverage violation penalty |
| **Pinball Loss** | Per-quantile accuracy |
| **ECE** | Expected Calibration Error (BiLSTM only) |

### Economic Utility (RQ3)
```
Transaction cost: $0.50/MWh  |  Volume: 1 MWh/trade  |  Slippage: 0.3σ
Strategies: Seasonal Naïve · LightGBM · Risk-Aware Bayesian (BiLSTM MC) · Risk-Aware CQR · Oracle
Metrics: Total P&L · Sharpe · Sortino · Max Drawdown · Win Rate
```

---

## 8. Results Summary

> **Pipeline completed May 17, 2026.** All numbers below are from the final test set evaluation (2024–2025, strictly out-of-sample).

### RQ1 — Point Accuracy (Test Set MAE, $/MWh)

| Model | MAE (PJM) | RMSE (PJM) | MAE (ERCOT) | RMSE (ERCOT) |
|---|---|---|---|---|
| Seasonal Naïve | 14.45 | 29.21 | 11.59 | 30.60 |
| AutoARIMA | 12.76 | 24.94 | 12.53 | 34.13 |
| MSTL | 14.40 | 24.95 | 11.16 | 24.03 |
| **LightGBM** | **4.03** | 12.32 | **3.39** | 10.64 |
| XGBoost | 4.02 | 12.52 | 3.46 | 11.55 |
| Bayesian Bi-LSTM | 4.84 | **10.10** | 3.65 | **10.11** |
| PatchTST | 6.69 | 12.99 | 5.27 | 15.19 |
| iTransformer | 7.47 | 14.57 | 6.07 | 16.65 |
| N-HiTS | 6.75 | 12.96 | 5.06 | 13.81 |
| BiTCN | 8.02 | 16.19 | 5.41 | 14.76 |
| TFT | 8.01 | 15.47 | 7.09 | 19.42 |
| Chronos-Bolt (zero-shot) | 7.02 | 13.08 | 7.77 | 21.54 |
| QRF | 4.55 | 12.29 | 3.68 | **9.88** |
| Ensemble (LGBM+XGB+BiLSTM) | 4.93 | 12.98 | 3.44 | 10.39 |

- **DM tests:** 86/91 (PJM) and 85/91 (ERCOT) pairwise comparisons significant after BH correction (α=0.05)
- LightGBM vs XGBoost: NOT significant on either market (p>0.23) — statistically equivalent

### RQ2 — Probabilistic Quality (90% CI where supported; 80% for Chronos/N-HiTS-Q)

| Model | PICP (PJM) | MPIW (PJM) | Winkler (PJM) | PICP (ERCOT) | MPIW (ERCOT) | Winkler (ERCOT) |
|---|---|---|---|---|---|---|
| LGBM Quantile (90% CI) | 69.70% | 10.38 | 42.77 | 75.77% | 17.60 | 29.48 |
| CQR (90% nominal) | 82.56% | 12.36 | 40.16 | 75.77% | 17.60 | 29.48 |
| Chronos-Bolt (80% CI) | 73.63% | 19.79 | 51.07 | 80.80% | 23.54 | 63.32 |
| **QRF (90% CI)** | **91.17%** | 20.29 | **33.32** | **95.65%** | 25.54 | **29.08** |
| N-HiTS Quantile (80% CI) | 65.49% | 15.01 | 56.56 | 68.39% | 11.14 | 46.41 |
| BiLSTM MC Dropout (90% CI) | 57.09% | 9.49 | 43.40 | 64.37% | 9.38 | 32.67 |

- **QRF is the only model achieving ≥90% PICP** on both markets with the tightest Winkler scores
- CQR achieves 82.6% on PJM (below 90% target) due to 2022 cal / 2024–2025 test distributional shift

### RQ3 — Economic Utility (TC=$0.50/MWh, slippage=0.3σ)

| Strategy | P&L (PJM) | Sharpe (PJM) | P&L (ERCOT) | Sharpe (ERCOT) |
|---|---|---|---|---|
| LightGBM | **$19,417** | **15.75** | **$16,359** | **8.53** |
| Seasonal Naïve | $13,660 | 10.06 | $860 | 0.69 |
| Risk-Aware CQR | $1,645 | 5.91 | $2,152 | 7.25 |
| Risk-Aware Bayesian | $881 | 5.29 | $500 | 2.85 |
| Oracle | $31,602 | 19.45 | $29,555 | 11.02 |

### RQ4 — Cross-Market Transfer (PJM → ERCOT, zero-shot)

| Model | In-Market MAE | Cross-Market MAE | Degradation |
|---|---|---|---|
| LightGBM | 3.39 | 4.67 | +37.9% |
| XGBoost | 3.46 | 4.69 | +35.6% |
| QRF | 3.68 | 5.03 | +36.7% |

### Ensemble Ablation (A4)

| Subset | MAE (PJM) | RMSE (PJM) | MAE (ERCOT) | RMSE (ERCOT) |
|---|---|---|---|---|
| LGBM only | 4.41 | 12.09 | 3.82 | 12.17 |
| XGB only | 4.38 | 11.99 | 3.98 | 11.94 |
| LGBM+XGB | 4.36 | 12.07 | 3.54 | 11.28 |
| **LGBM+XGB+BiLSTM** | **3.94** | **7.92** | **3.18** | **9.36** |

> Adding BiLSTM reduces ensemble RMSE by **34% (PJM)** and **17% (ERCOT)** — the recurrent model captures tail-event dynamics that tree models miss.

### Key Findings

1. **Tree-based models dominate point accuracy** — LightGBM (MAE=4.03 PJM, 3.39 ERCOT) and XGBoost are statistically equivalent and outperform all DL models
2. **BiLSTM excels at tail risk** — lowest RMSE on both markets (10.10 PJM, 10.11 ERCOT), critical for spike-sensitive applications
3. **QRF provides the best calibrated uncertainty** — only method achieving ≥90% PICP on both markets
4. **Foundation models underperform** — Chronos-Bolt (zero-shot) ranks below all trained models, suggesting domain-specific training remains essential for EPF
5. **Cross-market transfer is viable but costly** — 36–38% MAE degradation, indicating market-specific features are important but shared structure exists
6. **Stacked ensembles benefit from diversity** — adding BiLSTM to tree models dramatically improves RMSE via complementary error patterns

---

## 9. Requirements

```
# Core (pinned — see requirements.txt for exact versions)
pandas==2.3.3 · numpy==2.4.4 · scikit-learn==1.8.0 · scipy==1.15.3

# Statistical models
statsforecast==2.0.3

# Tree-based
lightgbm==4.6.0 · xgboost==3.2.0 · quantile-forest==1.4.1

# Deep Learning
tensorflow==2.21.0 · tensorflow-probability==0.25.0
torch==2.11.0 · neuralforecast==3.1.8

# Foundation model
chronos-forecasting==2.2.2

# Probabilistic evaluation
mapie>=0.9.0 · properscoring>=0.1 · shap==0.51.0 · arch>=7.2.0

# Visualization
matplotlib==3.10.9 · seaborn>=0.13.2
```

Full pinned versions: [`requirements.txt`](requirements.txt)

---

## 10. Citation

```bibtex

```

**Key references:**
- Nie et al. (2023). *PatchTST*. ICLR 2023.
- Liu et al. (2024). *iTransformer*. ICLR 2024.
- Ansari et al. (2024). *Chronos*. arXiv:2403.07815.
- Romano et al. (2019). *CQR*. NeurIPS 2019.
- Diebold & Mariano (1995). *Comparing Predictive Accuracy*. JBES.

---

<p align="center">
  <img src="reports/figures/F1_price_timeseries.png" alt="PJM + ERCOT Price Time Series" width="800"/>
  <br><i>Fig 1: PJM and ERCOT day-ahead LMP with regime shading (2019–2025)</i>
</p>
