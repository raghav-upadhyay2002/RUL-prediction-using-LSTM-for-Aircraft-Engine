# RUL prediction using LSTM for Aircraft Engine
RUL prediction using LSTM for Aircraft Engine
# Turbofan Engine RUL Prediction — NASA C-MAPSS

A deep learning pipeline for **Remaining Useful Life (RUL) prediction** of turbofan engines using the NASA C-MAPSS dataset. The project covers the full ML lifecycle: exploratory data analysis, feature engineering, model training, uncertainty quantification, explainability (XAI), domain adaptation, and an interactive maintenance decision dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Models](#models)
- [Results](#results)
- [Explainability (XAI)](#explainability-xai)
- [Uncertainty Estimation](#uncertainty-estimation)
- [Domain Adaptation](#domain-adaptation)
- [Maintenance Dashboard](#maintenance-dashboard)
- [Deployment](#deployment)
- [Ethical Considerations](#ethical-considerations)

---

## Overview

This project tackles the predictive maintenance problem of estimating how many operational cycles remain before a turbofan engine requires maintenance. Using multi-sensor time-series data, four LSTM-based deep learning architectures are trained and benchmarked across all four NASA C-MAPSS sub-datasets. The pipeline goes beyond standard regression by incorporating uncertainty quantification, gradient-based explainability, and an interactive fleet monitoring dashboard.

**Key features:**
- Four LSTM-based architectures benchmarked head-to-head
- Sliding-window sequence generation for temporal modeling
- Piecewise-linear RUL capping (cap = 125 cycles)
- Monte Carlo Dropout and Deep Ensembles for uncertainty estimation
- Temporal attention visualization and gradient-based feature attribution
- Condition-aware normalization and cross-dataset zero-shot transfer
- Interactive ipywidgets maintenance dashboard with MC Dropout inference
- Full model saving with deployment metadata

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**  
Source: [NASA Ames Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

The dataset contains run-to-failure sensor readings from simulated turbofan engines across four sub-datasets:

| Sub-dataset | Training Rows | Test Engines | Fault Modes | Operating Conditions |
|-------------|--------------|--------------|-------------|----------------------|
| FD001       | 20,631       | 100          | 1           | 1                    |
| FD002       | 53,759       | 259          | 1           | 6                    |
| FD003       | 24,720       | 100          | 2           | 1                    |
| FD004       | 61,249       | 248          | 2           | 6                    |

Each record includes 3 operational settings (`op_1–3`) and 21 sensor readings (`s1–s21`). Seven near-zero-variance sensors (`s1, s5, s6, s10, s16, s18, s19`) are dropped, leaving **17 input features**.

**Setup:** Upload `CMAPSSData.zip` to your Colab environment when prompted. The notebook will auto-extract and locate the data files.

---

## Project Structure









**Saved outputs** (to `/content/rul_models/`):
- `{arch}_{fd}.keras` — trained model weights for all 16 combinations
- `scalers.pkl` — fitted StandardScaler objects per sub-dataset
- `metadata.json` — deployment configuration
- `maintenance_schedule.csv` — per-engine RUL predictions and decisions

---

## Installation

This project runs entirely in **Google Colab**. No local installation is required.

**Dependencies (pre-installed in Colab):**
