# RUL prediction using LSTM for Aircraft Engine
## Turbofan Engine RUL Prediction — NASA C-MAPSS

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


Open the notebook in Colab and run cells sequentially. When Cell 1 prompts for a file upload, provide `CMAPSSData.zip`.

---

## Pipeline Walkthrough

### 1. EDA (Cell 2)
- Engine lifetime distribution and RUL statistics
- Sensor variance analysis to identify and drop uninformative sensors
- Pearson correlation of sensors with the cycle index as a degradation proxy
- Sensor-to-sensor correlation heatmap
- Visual degradation trends for sensors `s4` (Bypass Ratio) and `s11` (HPC Outlet Temperature)

### 2. Preprocessing & Feature Engineering (Cell 3)
- **RUL Labeling:** Piecewise-linear capping at 125 cycles (flattens early-life RUL, then decreases linearly)
- **Sensor Selection:** 7 low-variance sensors dropped; 17 features retained
- **Normalization:** Z-score (`StandardScaler`) fitted on training data, applied to test
- **Sliding Window:** 30-cycle windows with stride 1 over each engine's lifecycle
  - Training: all windows → `(N, 30, 17)` arrays
  - Test: last 30 cycles per engine → one sample per engine

### 3. Model Training (Cell 5)
- Up to 50 epochs with early stopping (patience = 7)
- `ReduceLROnPlateau` for adaptive learning rate
- Best weights restored via `ModelCheckpoint`
- Evaluated using MAE, RMSE, and the **NASA asymmetric score** (penalizes late predictions more than early)

---

## Models

All four architectures share the same input shape `(30, 17)` and linear output for regression.

| Architecture | Parameters | Key Design Choice |
|---|---|---|
| **VanillaLSTM** | 127,041 | Two stacked LSTM layers — sequential degradation baseline |
| **StackedBiLSTM** | 319,553 | Bidirectional LSTM captures both forward and backward temporal context |
| **LSTM_Attention** | 127,105 | Custom Bahdanau-style temporal attention over LSTM output |
| **CNN_LSTM** | 123,329 | CNN extracts local patterns; LSTM models long-range dependencies |

All models use `BatchNormalization`, `Dropout (0.3)`, and `Adam (lr=1e-3)` with MSE loss.

---

## Results

Test set performance across all architectures and sub-datasets:

| Model           | Dataset | MAE   | RMSE  | NASA Score |
|-----------------|---------|-------|-------|------------|
| VanillaLSTM     | FD001   | 12.04 | 15.47 | 498        |
| VanillaLSTM     | FD002   | 13.04 | 17.34 | 1,749      |
| VanillaLSTM     | FD003   | 10.42 | 15.26 | 835        |
| VanillaLSTM     | FD004   | 13.16 | 17.78 | 1,776      |
| StackedBiLSTM   | FD001   | 11.92 | 15.78 | 507        |
| StackedBiLSTM   | FD002   | 11.23 | 15.83 | 1,450      |
| StackedBiLSTM   | FD003   | 10.91 | 15.56 | 578        |
| StackedBiLSTM   | FD004   | 11.83 | 17.07 | 1,431      |
| LSTM_Attention  | FD001   | 11.64 | 15.66 | 585        |
| LSTM_Attention  | FD002   | 12.50 | 16.58 | 1,506      |
| LSTM_Attention  | FD003   | 11.57 | 15.65 | 744        |
| LSTM_Attention  | FD004   | 15.06 | 20.41 | 3,069      |
| CNN_LSTM        | FD001   | 12.93 | 15.87 | 469        |
| CNN_LSTM        | FD002   | 12.72 | 17.48 | 2,382      |
| CNN_LSTM        | FD003   | 14.09 | 20.37 | 2,614      |
| CNN_LSTM        | FD004   | 15.54 | 21.50 | 3,876      |

**StackedBiLSTM** achieves the most consistent performance across all four datasets. The NASA score penalizes under-predictions (late alarms) exponentially, which is why simpler models sometimes outscore attention-based ones on this metric despite higher MAE.

---

## Explainability (XAI)

Three complementary XAI methods are implemented in Cell 6:

**1. Temporal Attention Heatmap**  
Visualizes which timesteps within a 30-cycle window the `LSTM_Attention` model attends to most. Near-failure engines show attention concentrated on the most recent timesteps, while healthy engines distribute attention more broadly.

**2. Gradient-Based Feature Attribution**  
Computes `d(RUL prediction) / d(input features)` via `tf.GradientTape`. Features with higher mean absolute gradient are more influential. The top-3 most influential features are reported per run.

**3. Per-Engine Explanation**  
For any individual engine, a combined plot shows its temporal attention bar chart alongside a gradient sensitivity heatmap over the time × feature grid — providing an auditable, human-readable explanation for each maintenance decision.

Outputs saved: `eda_overview.png`, `eda_sensors.png`, `training_curves.png`, `pred_vs_actual.png`, `error_dist.png`, `attention_weights.png`, `xai_attention_heatmap.png`, `xai_feature_attribution.png`, `xai_single_engine.png`

---

## Uncertainty Estimation

Cell 7 implements two uncertainty quantification techniques:

**Monte Carlo Dropout**  
Keeps dropout active at inference time. 100 forward passes through `LSTM_Attention` produce a distribution of RUL predictions. Epistemic uncertainty (model uncertainty) and aleatoric uncertainty (data uncertainty) are decomposed from the variance across passes.

**Deep Ensembles**  
5 independently-trained `VanillaLSTM` models (each with a different random seed) are trained for 15 epochs. Disagreement between ensemble members captures epistemic uncertainty in a more robust way than MC Dropout, especially for out-of-distribution inputs.

Both methods produce `mean ± 2σ` prediction intervals, which are used as the basis for conservative maintenance decisions.

---

## Domain Adaptation

Cell 8 addresses two generalization challenges:

**Condition-Aware Normalization (FD002/FD004)**  
FD002 and FD004 contain 6 distinct operating conditions (altitude, Mach, throttle settings). Global Z-score normalization conflates these conditions. A k-means clustering step (k=6) groups rows by operational setting, and a separate `StandardScaler` is fitted per cluster — improving MAE on FD002 compared to global normalization.

**Zero-Shot Cross-Dataset Transfer**  
Models trained on FD001 (1 fault mode, 1 operating condition) are evaluated directly on FD003 (2 fault modes, 1 operating condition) without retraining. The generalization gap across all four architectures is visualized to quantify robustness to unseen fault modes.

---

## Maintenance Dashboard

Cell 9 provides two dashboard interfaces:

**Static Dashboard**  
A full-fleet visualization showing predicted RUL ± uncertainty bands for all test engines, with color-coded decision zones (URGENT / SCHEDULE / OK) and a priority-sorted maintenance queue.

**Interactive ipywidgets Dashboard**  
Adjust the following parameters live in Colab:
- Engine ID (slider)
- Model architecture (dropdown)
- Sub-dataset (dropdown)
- Number of MC Dropout passes (slider)
- URGENT and SCHEDULE RUL thresholds (sliders)

Click **▶ Predict Engine** to see the MC dropout distribution, sensor input heatmap, and maintenance recommendation for the selected engine. Click **📊 Fleet Overview** for a summary bar chart of all engines at the current threshold settings.

**Decision logic:**  
`conservative_RUL = mean_RUL - 2σ`  
- `conservative_RUL < URGENT_THR (10)` → 🔴 **URGENT**  
- `conservative_RUL < SCHEDULE_THR (30)` → 🟠 **SCHEDULE**  
- Otherwise → 🟢 **OK**

---

## Deployment

Saved artifacts in `/content/rul_models/`:





**Inference pipeline:**
1. Buffer the last 30 sensor cycles from the engine's ECU stream
2. Apply the condition-aware `StandardScaler` for the detected operating condition
3. Run forward passes through `LSTM_Attention` with `training=True` (MC Dropout, 100 passes)
4. Compute `mean ± 2σ` and apply decision thresholds
5. Emit URGENT / SCHEDULE / OK alert to the maintenance system

**Serving options:** TF SavedModel → TF Serving (Docker) → REST API (FastAPI / Flask)  
**Scaling:** Kubernetes with horizontal pod autoscaling for fleet-wide monitoring

---

## Ethical Considerations

1. **Safety-first uncertainty:** Conservative lower-bound predictions (mean − 2σ) reduce the risk of missing a genuine failure.
2. **Transparency:** Uncertainty bounds are always reported alongside point estimates — never just a single number.
3. **Human-in-the-loop:** The model is a decision support tool; all maintenance actions require engineer sign-off.
4. **Distribution shift:** C-MAPSS is simulated data. Real-engine deployment requires continuous monitoring for model drift and periodic retraining.
5. **Data privacy:** Sensor streams may contain proprietary operational data; on-premise inference is recommended.
6. **Fleet equity:** Decision thresholds should be calibrated for engine age, type, and usage history.
7. **Auditability:** All predictions should be logged with timestamps for EASA/FAA regulatory compliance.

---

## License

This project uses the NASA C-MAPSS dataset, which is publicly available for research and educational purposes. Please cite the original dataset when publishing results:

> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. International Conference on Prognostics and Health Management.
