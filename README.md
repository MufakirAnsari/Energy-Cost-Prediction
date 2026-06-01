# Trees Still Beat Transformers for Day-Ahead Electricity Price Forecasting

> **Research pipeline for *Applied Energy* submission.**
> A comprehensive 18-configuration benchmark of gradient-boosted trees, deep learning architectures, and time-series foundation models for day-ahead electricity price forecasting across PJM and ERCOT (2019–2025), with conformal prediction, expanding-window retraining, and realistic economic simulation.

<p align="center">
  <img src="reports/figures/Graphical_Abstract.png" alt="Graphical Abstract" width="900"/>
</p>

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

### Key Contributions
1. **Scale and recency.** 18 forecasting configurations spanning 16 model families — 3 statistical baselines, 3 tree-based methods, 6 deep learning architectures, 2 foundation model variants, and 2 ensemble strategies, plus rolling-retrained variants — on 2 structurally distinct US markets with 7 years of hourly data (2019–2025).
2. **Expanding-window protocol.** Monthly expanding-window retraining across 24 windows, demonstrating that periodic retraining improves LightGBM's MAE by **24.1% (PJM)** and **23.7% (ERCOT)**.
3. **Three probabilistic paradigms.** Conformalized Quantile Regression (CQR), Quantile Regression Forests (QRF), and Monte Carlo Dropout under real-world regime shift, documenting a critical failure of CQR's exchangeability assumption.
4. **Foundation model evaluation.** First benchmark of Chronos-Bolt v1, Chronos-Bolt-Base v2 (univariate), and a covariate-enhanced v2 variant on US electricity prices — demonstrating that covariates reduce MAE by **49%** but cannot close the gap with rolling LightGBM.
5. **Economic evaluation.** Statistical accuracy translated into trading PnL with realistic transaction costs and slippage, revealing that superior MAE does not automatically translate to superior economic value when risk-aware interval filters are applied.

### Research Questions
| RQ | Question | Primary Metric |
|---|---|---|
| RQ1 | Which model family achieves best point accuracy? | MAE, DM test |
| RQ2 | Which probabilistic method achieves best calibration? | PICP, CRPS |
| RQ3 | Does better forecasting translate to economic utility? | Sharpe ratio, PnL |
| RQ4 | Do PJM-trained models generalize to ERCOT (cross-market)? | MAE degradation |

---

## 2. Dataset

### Markets
| Market | Product | Node | Period | Hourly Obs |
|---|---|---|---|---|
| **PJM** | Day-Ahead LMP | WESTERN HUB | 2019–2025 | ~59,600 |
| **ERCOT** | Day-Ahead SPP | HB_HOUSTON | 2019–2025 | ~59,600 |

### Exogenous Features (~50 engineered)
| Source | Features |
|---|---|
| EIA API v2 | Solar %, Wind MW, Gas MW, Nuclear MW, Net Load, Demand |
| Open-Meteo | Temperature, Wind Speed, Humidity, Solar Radiation (5 cities per market) |
| EIA Henry Hub | Natural gas spot price ($/MMBtu) |
| Computed | Hour/DOW/Month sin-cos, US holidays, lag features (1h, 2h, 3h, 24h, 48h, 168h) |

### Chronological 4-Way Split (no data leakage)
```
2019-01-01 ── 2021-12-31 │ 2022-01-01 ── 2022-12-31 │ 2023-01-01 ── 2023-12-31 │ 2024-01-01 ── 2025-12-31
       TRAIN (25,548h)           CALIBRATION (CQR)            VALIDATION              TEST (17,040h)
    [base model training]     [conformal calibration]     [hyperparameter tuning]     [final evaluation]
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

## 3. Model Suite (18 Configurations)

### Statistical Baselines (3)
| Model | Key Detail |
|---|---|
| Seasonal Naïve | 168h (weekly) seasonal persistence |
| AutoARIMA | `statsforecast`, hourly seasonality |
| MSTL + AutoETS | Multi-Seasonal Trend decomposition |

### Tree-Based (3)
| Model | Key Detail |
|---|---|
| LightGBM (point + quantile) | MAE-optimized; 7 quantile models for full distribution |
| XGBoost | Point forecast; second tree baseline |
| Quantile Regression Forest | Non-parametric probabilistic intervals |

### Deep Learning (6)
| Model | Architecture | Source |
|---|---|---|
| Bayesian Bi-LSTM | Bidirectional LSTM + MC Dropout (ECE-calibrated) | TensorFlow |
| PatchTST | Patch-based Transformer | ICLR 2023 |
| iTransformer | Inverted attention across variables | ICLR 2024 |
| N-HiTS | Hierarchical interpolation | AAAI 2023 |
| BiTCN | Bidirectional Temporal CNN | neuralforecast |
| TFT | Temporal Fusion Transformer | neuralforecast |

### Foundation Models (2 + 1 covariate-enhanced)
| Model | Type | Detail |
|---|---|---|
| Chronos-Bolt (v1) | Zero-shot | Amazon, univariate, no fine-tuning |
| Chronos-Bolt-Base (v2) | Zero-shot | Updated architecture, univariate |
| Chronos-Base + Covariates | Enhanced | v2 with top-5 SHAP features via Ridge residuals |

### Ensemble (2)
| Model | Detail |
|---|---|
| Stacked Ensemble | LightGBM meta-learner on {LightGBM + XGBoost + BiLSTM} |
| Rolling variants | Monthly expanding-window retrained LightGBM and XGBoost |

---

## 4. Repository Structure

```
├── config.py                         # All paths, hyperparameters, split dates
├── utils.py                          # Shared metrics, DM test, data loading
├── requirements.txt                  # Pinned dependencies
├── smoke_test.py                     # Pre-flight validation suite
├── test_data_integrity.py            # Data leakage assertion tests
├── run_all.sh                        # Full pipeline orchestration
│
├── data/
│   ├── raw/                          # Downloaded parquets (PJM, ERCOT, EIA, weather)
│   └── processed/                    # 8 train/cal/val/test parquets
│
├── step_00_download_pjm.py           # gridstatus → PJM DA LMP 2019–2025
├── step_00_download_ercot.py         # gridstatus → ERCOT DA SPP 2019–2025
├── step_00_download_weather.py       # Open-Meteo → hourly weather
├── step_00_download_eia.py           # EIA API v2 → generation mix + gas price
│
├── step_01_preprocess.py             # Merge, clean, feature-engineer, 4-way split
├── step_01_eda.py                    # EDA — price stats, ACF/PACF, regime plots
├── step_02_train_baselines.py        # Seasonal Naïve, AutoARIMA, MSTL
├── step_03_train_lgbm.py             # LightGBM point + 7-quantile models
├── step_04_train_xgboost.py          # XGBoost point forecast
├── step_05_train_bilstm.py           # Bayesian Bi-LSTM + MC Dropout
├── step_05b_retrain_bilstm_ercot.py  # ERCOT BiLSTM with log1p+clip fix
├── step_06_train_patchtst.py         # PatchTST (h=24)
├── step_06b_train_bitcn.py           # BiTCN (h=24)
├── step_07_train_itransformer.py     # iTransformer (h=24)
├── step_07b_train_tft.py             # TFT (h=24)
├── step_08_train_nhits.py            # N-HiTS point (h=24)
├── step_08b_nhits_quantile.py        # N-HiTS quantile — MQLoss 80% CI
├── step_09_chronos_inference.py      # Chronos-Bolt v1 zero-shot
├── step_09b_chronos2_inference.py    # Chronos-Bolt-Base v2 + covariate-enhanced
├── step_09_evaluate.py               # All metrics: point + probabilistic + economic
├── step_10_conformal.py              # CQR on dedicated calibration set (2022)
├── step_10b_alpha_sweep.py           # Conformal alpha sweep
├── step_11_qrf.py                    # Quantile Regression Forest
├── step_12_ensemble.py               # Stacked ensemble meta-learner
├── step_14_dm_tests.py               # Diebold-Mariano + Benjamini-Hochberg FDR
├── step_15_ablation.py               # Lag window, feature set, dropout, ensemble
├── step_16_stress_test.py            # OOS regime evaluation
├── step_17_figures.py                # Publication figures (300 DPI)
├── step_17b_novelty_figures.py       # Chronos evolution, SHAP, CQR vs QRF figures
├── step_18_paper_tables.py           # LaTeX tables
├── step_19_rq4_crossmarket.py        # PJM→ERCOT transfer
├── step_20_fix_weaknesses.py         # Reviewer mitigation: BiTCN, TFT, coverage
├── step_21_rolling_window.py         # 24-month expanding-window retraining
├── step_22_integrate_new_results.py  # Consolidate all new results into tables
├── step_23_final_updates.py          # DM tests update + graphical abstract
│
├── models/                           # Trained model artifacts
│   ├── lgbm_*.joblib                 # LightGBM point + quantile models
│   ├── xgboost_*.joblib              # XGBoost models
│   ├── bilstm_*.keras                # BiLSTM models
│   ├── patchtst_*/                   # PatchTST checkpoints
│   ├── itransformer_*/               # iTransformer checkpoints
│   ├── nhits_*/                      # N-HiTS checkpoints
│   ├── bitcn_*/                      # BiTCN checkpoints
│   └── tft_*/                        # TFT checkpoints
│
└── reports/
    ├── figures/                      # 31 publication-quality figures
    │   ├── Graphical_Abstract.png    # Journal graphical abstract
    │   ├── F1–F15_*.png              # Main body figures
    │   ├── Fig7–Fig8_*.png           # Chronos comparison figures
    │   ├── FigA–FigE_*.png           # Novelty figures
    │   ├── FigW2–FigW6_*.png         # Supplementary figures
    │   └── EDA1–EDA4_*.png           # Exploratory analysis
    ├── table_*.csv                   # All result tables
    └── *_preds_*.csv                 # Model predictions
```

> **Note:** QRF models (`qrf_pjm.joblib`, `qrf_ercot.joblib`, ~200MB each) exceed GitHub's 100MB file limit and are excluded. Regenerate with `python step_11_qrf.py`.

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
```

### Run the Full Pipeline
```bash
# Step 1: Download data (one-time, ~30 min)
python step_00_download_pjm.py
python step_00_download_ercot.py
python step_00_download_eia.py
python step_00_download_weather.py

# Step 2: Preprocess + EDA
python step_01_preprocess.py
python step_01_eda.py

# Step 3: Full automated pipeline
chmod +x run_all.sh
./run_all.sh 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M).log
```

> `run_all.sh` is **idempotent** — if a step's output already exists, it is skipped automatically. Safe to re-run after interruptions.

---

## 6. Pipeline Reference

| Step | Script | GPU | Est. Time |
|---|---|---|---|
| 00 | `step_00_download_*.py` (×4) | ✗ | ~30 min |
| 01 | `step_01_preprocess.py` | ✗ | ~5 min |
| 01b | `step_01_eda.py` | ✗ | ~2 min |
| 02 | `step_02_train_baselines.py` | ✗ | ~4–5 hrs |
| 03 | `step_03_train_lgbm.py` | ✗ | ~5 min |
| 04 | `step_04_train_xgboost.py` | ✗ | ~5 min |
| 05 | `step_05_train_bilstm.py` | ✅ | ~30–45 min |
| 06 | `step_06_train_patchtst.py` | ✅ | ~45–60 min |
| 06b | `step_06b_train_bitcn.py` | ✅ | ~30 min |
| 07 | `step_07_train_itransformer.py` | ✅ | ~30–45 min |
| 07b | `step_07b_train_tft.py` | ✅ | ~45 min |
| 08 | `step_08_train_nhits.py` | ✅ | ~20–30 min |
| 08b | `step_08b_nhits_quantile.py` | ✅ | ~20 min |
| 09 | `step_09_chronos_inference.py` | ✗ | ~3–5 hrs |
| 09b | `step_09b_chronos2_inference.py` | ✅ | ~1–2 hrs |
| 10 | `step_10_conformal.py` | ✗ | ~2 min |
| 11 | `step_11_qrf.py` | ✗ | ~10 min |
| 12 | `step_12_ensemble.py` | ✗ | ~5 min |
| 13 | `step_09_evaluate.py` | ✗ | ~5 min |
| 14 | `step_14_dm_tests.py` | ✗ | ~3 min |
| 15 | `step_15_ablation.py` | ✗ | ~20 min |
| 16 | `step_16_stress_test.py` | ✗ | ~5 min |
| 17 | `step_17_figures.py` + `step_17b_novelty_figures.py` | ✗ | ~3 min |
| 18 | `step_18_paper_tables.py` | ✗ | ~2 min |
| 19 | `step_19_rq4_crossmarket.py` | ✗ | ~5 min |
| 20 | `step_20_fix_weaknesses.py` | ✅ | ~1 hr |
| 21 | `step_21_rolling_window.py` | ✗ | ~40 min |
| 22 | `step_22_integrate_new_results.py` | ✗ | ~2 min |
| 23 | `step_23_final_updates.py` | ✗ | ~1 min |

**Total estimated time:** ~15–18 hours (GPU), ~24+ hours (CPU only)

---

## 7. Evaluation Framework

### Point Accuracy (RQ1)
| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — primary metric, robust to spikes |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **DM test** | Diebold-Mariano with HLN small-sample correction + BH FDR adjustment |

### Probabilistic Quality (RQ2)
| Metric | Description |
|---|---|
| **PICP** | Prediction Interval Coverage Probability (target: ≥90%) |
| **MPIW** | Mean Prediction Interval Width (sharpness) |
| **CRPS** | Continuous Ranked Probability Score (proper scoring rule) |

### Economic Utility (RQ3)
```
Transaction cost: $0.50/MWh  |  Volume: 1 MWh/trade  |  Slippage: 0.3σ
Strategies: Seasonal Naïve · LightGBM · Risk-Aware CQR · Risk-Aware Bayesian · Oracle
Metrics: Total PnL · Sharpe · Sortino · Max Drawdown · Win Rate
```

---

## 8. Results Summary

> **All numbers below are from the final test set (2024–2025), strictly out-of-sample.**

### RQ1 — Point Accuracy (Test Set MAE, $/MWh)

| Model | MAE (PJM) | RMSE (PJM) | MAE (ERCOT) | RMSE (ERCOT) |
|---|---|---|---|---|
| **LightGBM (rolling)** | **3.05** | **6.40** | **2.58** | 6.50 |
| XGBoost (rolling) | 3.09 | 6.40 | 2.87 | 7.41 |
| Chronos-Base + Covariates | 3.48 | 6.65 | 5.58 | 14.92 |
| XGBoost (static) | 4.02 | 12.52 | 3.46 | 11.55 |
| LightGBM (static) | 4.03 | 12.32 | 3.39 | 10.64 |
| QRF | 4.55 | 12.29 | 3.68 | **9.88** |
| BiLSTM | 4.84 | **10.10** | 3.65 | **10.11** |
| Ensemble | 4.93 | 12.98 | 3.44 | 10.39 |
| PatchTST | 6.69 | 12.99 | 5.27 | 15.19 |
| N-HiTS | 6.75 | 12.96 | 5.06 | 13.81 |
| Chronos-Bolt-Base (v2) | 6.85 | 12.77 | 7.96 | 22.01 |
| Chronos-Bolt (v1) | 7.02 | 13.08 | 7.77 | 21.54 |
| iTransformer | 7.47 | 14.57 | 6.07 | 16.65 |
| TFT | 8.01 | 15.47 | 7.09 | 19.42 |
| BiTCN | 8.02 | 16.19 | 5.41 | 14.76 |
| AutoARIMA | 12.76 | 24.94 | 12.53 | 34.13 |
| MSTL | 14.40 | 24.95 | 11.16 | 24.03 |
| Seasonal Naïve | 14.45 | 29.21 | 11.59 | 30.60 |

### Expanding-Window Retraining

| Market | Model | Static MAE | Rolling MAE | Improvement |
|---|---|---|---|---|
| PJM | LightGBM | 4.02 | 3.05 | **−24.1%** |
| PJM | XGBoost | 4.01 | 3.09 | **−22.9%** |
| ERCOT | LightGBM | 3.39 | 2.58 | **−23.7%** |
| ERCOT | XGBoost | 3.45 | 2.87 | **−17.0%** |

### RQ2 — Probabilistic Quality

| Method | PICP (PJM) | CRPS (PJM) | PICP (ERCOT) | CRPS (ERCOT) |
|---|---|---|---|---|
| **QRF (90% CI)** | **91.17%** | 3.660 | **95.65%** | 3.238 |
| CQR (90% nominal) | 82.56% | 3.804 | 75.77% | 4.713 |
| LGBM Quantile (90% CI) | 69.70% | **3.349** | 75.77% | **2.776** |
| Chronos-Bolt (80% CI) | 73.63% | — | 80.80% | — |
| N-HiTS Quantile (80% CI) | 65.49% | 5.337 | 68.39% | 4.134 |
| BiLSTM MC Dropout (90% CI) | 57.09% | — | 64.37% | 2.904 |

> **QRF is the only model achieving ≥90% PICP** on both markets.
> CQR falls short at 82.6% (PJM) and 75.8% (ERCOT) due to exchangeability violation between the 2022 calibration set and 2024–2025 test set.

### RQ3 — Economic Utility

| Strategy | PnL (PJM) | Sharpe (PJM) | Win% (PJM) | PnL (ERCOT) | Sharpe (ERCOT) | Win% (ERCOT) |
|---|---|---|---|---|---|---|
| Oracle | $31,602 | 19.45 | 100.0% | $29,555 | 11.02 | 100.0% |
| **LightGBM** | **$19,417** | **15.75** | **97.5%** | **$16,359** | **8.53** | **92.1%** |
| Seasonal Naïve | $13,660 | 10.06 | 81.0% | $860 | 0.69 | 49.9% |
| Risk-Aware CQR | $1,645 | 5.91 | 18.7% | $2,152 | 7.25 | 30.9% |
| Risk-Aware Bayesian | $881 | 5.29 | 14.3% | $500 | 2.85 | 14.6% |

### RQ4 — Cross-Market Transfer (PJM → ERCOT, zero-shot)

| Model | In-Market MAE | Cross-Market MAE | Degradation |
|---|---|---|---|
| LightGBM | 3.39 | 4.67 | +37.9% |
| XGBoost | 3.46 | 4.69 | +35.6% |
| QRF | 3.68 | 5.03 | +36.7% |

### Key Findings

1. **Trees dominate.** LightGBM with monthly expanding-window retraining achieves MAE 3.05 (PJM) and 2.58 (ERCOT), outperforming all DL models by 20–130% and all foundation models by 12–130%.
2. **Covariates > Architecture.** Adding covariates to Chronos-2 reduces MAE by 49%, but the architectural upgrade from v1 to v2 yields only 2.3% — feature engineering matters more.
3. **Monthly retraining is essential.** Expanding-window retraining provides a consistent 24% improvement across all 24 test months.
4. **CQR fails under regime shift.** Conformal prediction's exchangeability assumption is violated between 2022 calibration and 2024–2025 test, causing PICP to fall 7–14 percentage points below target. QRF is robust without this assumption.
5. **Statistical accuracy ≠ economic value.** The risk-aware CQR strategy captures only 8.5% of LightGBM's PnL despite its theoretical coverage guarantee, due to overly wide intervals causing trade abstention.
6. **Cross-market transfer is viable but costly.** PJM→ERCOT zero-shot transfer incurs 36–38% MAE degradation, indicating market-specific features matter but shared price dynamics exist.

---

## 9. Requirements

```
# Core
pandas · numpy · scikit-learn · scipy

# Statistical
statsforecast

# Tree-based
lightgbm · xgboost · quantile-forest

# Deep Learning
tensorflow · tensorflow-probability · torch · neuralforecast

# Foundation model
chronos-forecasting

# Probabilistic
mapie · properscoring · shap · arch

# Visualization
matplotlib · seaborn
```

Full pinned versions: [`requirements.txt`](requirements.txt)

---

## 10. Citation

```bibtex
@article{ansari2026trees,
  title   = {Trees Still Beat Transformers for Day-Ahead Electricity Price Forecasting},
  author  = {Ansari, Mufakir Qamar},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under review}
}
```

**Key references:**
- Lago et al. (2021). *Forecasting day-ahead electricity prices*. Applied Energy.
- Nie et al. (2023). *PatchTST*. ICLR 2023.
- Liu et al. (2024). *iTransformer*. ICLR 2024.
- Ansari et al. (2024). *Chronos: Learning the language of time series*. TMLR.
- Romano et al. (2019). *Conformalized Quantile Regression*. NeurIPS 2019.
- Meinshausen (2006). *Quantile Regression Forests*. JMLR.
- Diebold & Mariano (1995). *Comparing Predictive Accuracy*. JBES.

---

<p align="center">
  <img src="reports/figures/F1_price_timeseries.png" alt="PJM + ERCOT Price Time Series" width="800"/>
  <br><i>PJM and ERCOT day-ahead LMP with regime shading (2019–2025)</i>
</p>
