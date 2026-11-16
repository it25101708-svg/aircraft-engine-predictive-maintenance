# ✈️ AI-Powered Aircraft Engine Predictive Maintenance System

An Artificial Intelligence and Machine Learning-based predictive maintenance prototype designed to estimate the Remaining Useful Life (RUL) of aircraft turbofan engines using sensor data.

The system uses NASA's C-MAPSS turbofan engine degradation dataset and a Random Forest Regression model to predict engine remaining useful life, identify potential maintenance risks, and visualize fleet health through an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Traditional aircraft maintenance often relies on scheduled inspections and maintenance intervals. Predictive maintenance introduces a data-driven approach by analyzing equipment condition and estimating when maintenance may be required.

This project demonstrates how Machine Learning can be applied to aircraft engine health monitoring.

The system can:

- Predict the Remaining Useful Life (RUL) of aircraft engines.
- Analyze engine sensor measurements.
- Estimate engine health based on historical operating data.
- Generate automated maintenance alerts.
- Compare actual and predicted engine RUL.
- Display fleet-level insights through an interactive dashboard.

> **Project type:** Educational AI/ML prototype
> **Domain:** Aviation Technology + Artificial Intelligence
> **Dataset:** NASA C-MAPSS FD001

---

## 🎯 Project Objectives

1. Explore aircraft engine sensor data.
2. Understand engine degradation patterns.
3. Develop a Machine Learning model for RUL prediction.
4. Evaluate prediction accuracy using regression metrics.
5. Build an automated maintenance alert system.
6. Develop an interactive dashboard for engine monitoring.

---

## 🧠 Technologies Used

| Technology              | Purpose                          |
| ----------------------- | -------------------------------- |
| Python                  | Main programming language        |
| Pandas                  | Data processing and manipulation |
| NumPy                   | Numerical operations             |
| Matplotlib              | Data visualization               |
| Seaborn                 | Statistical visualization        |
| Scikit-learn            | Machine Learning                 |
| Random Forest Regressor | RUL prediction                   |
| Joblib                  | Model serialization              |
| Streamlit               | Interactive dashboard            |
| Plotly                  | Interactive charts               |

---

## 📊 Dataset

This project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset.

The FD001 dataset contains simulated turbofan engine operational data, including:

- Engine operational cycles
- Operational settings
- 21 sensor measurements
- Engine degradation information

### Dataset files

| File              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `train_FD001.txt` | Training engine data                                 |
| `test_FD001.txt`  | Testing engine data                                  |
| `RUL_FD001.txt`   | Actual remaining useful life values for test engines |

Dataset source:

[NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## ⚙️ Machine Learning Model

The project uses a **Random Forest Regressor** to estimate engine Remaining Useful Life.

### Model configuration

| Parameter             |                   Value |
| --------------------- | ----------------------: |
| Algorithm             | Random Forest Regressor |
| Number of estimators  |                     150 |
| Maximum depth         |                      15 |
| Minimum samples split |                       5 |
| Minimum samples leaf  |                       2 |
| Random state          |                      42 |

### Remaining Useful Life Calculation

For training engines:

`RUL = Maximum Engine Cycle - Current Engine Cycle`

The RUL target is capped at 125 cycles to limit the influence of early-life operating periods.

For testing engines, the provided NASA RUL values are combined with the last observed engine cycle to calculate the target RUL.

---

## 📈 Model Performance

The model was evaluated using the independent FD001 test dataset.

\<details> \<summary>\<strong>View evaluation results\</strong>\</summary>

| Evaluation Metric              |       Result |
| ------------------------------ | -----------: |
| Mean Absolute Error (MAE)      | 12.45 cycles |
| Root Mean Squared Error (RMSE) | 16.83 cycles |
| R² Score                       |       0.8237 |

**Interpretation:**

- **MAE – 12.45:** Predictions differ from actual RUL by approximately 12.45 cycles on average.
- **RMSE – 16.83:** Measures prediction error while penalizing larger errors more heavily.
- **R² – 0.8237:** The model explains approximately 82.37% of the variation in the evaluated RUL targets.

\</details>

These results are based on the project's capped RUL targets and are not real-world aircraft maintenance performance measurements.

---

## 🛠️ System Features

### 1. Engine Sensor Analysis

Analyzes engine sensor measurements and visualizes operating patterns across engine cycles.

### 2. RUL Prediction

Predicts the remaining useful life of an individual engine using the trained Random Forest model.

### 3. Automated Maintenance Alerts

Categorizes engines based on predicted RUL using illustrative demonstration thresholds.

| Status      | Predicted RUL       |
| ----------- | ------------------- |
| 🔴 Critical | 20 cycles or less   |
| 🟠 Warning  | 21–50 cycles        |
| 🟢 Normal   | More than 50 cycles |

The alert thresholds are demonstration settings, not approved aviation maintenance limits.

### 4. Prediction Visualization

Provides visual comparisons between actual and predicted RUL values.

### 5. Interactive Dashboard

Built with Streamlit to provide:

- Fleet overview
- Engine-level prediction insights
- Maintenance alert monitoring
- Model performance analysis
- Interactive charts and data tables

---

## 🖥️ Dashboard Preview

Add your actual dashboard screenshots here after capturing them.

**Fleet Overview**



**Engine Intelligence**



**Maintenance Control**



**AI Performance**



---

## 📁 Project Structure

```text
AI-Powered Aircraft Engine Predictive Maintenance System/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   │
│   └── processed/
│       ├── train_processed.csv
│       ├── test_processed.csv
│       ├── prediction_results.csv
│       └── maintenance_alert_report.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── data_processing.py
│   ├── data_analysis.py
│   ├── sensor_analysis.py
│   ├── rul_analysis.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── prediction.py
│   ├── prediction_visualization.py
│   └── maintenance_alert.py
│
├── visualizations/
│   ├── actual_vs_predicted.png
│   └── engine_comparison.png
│
└── README.md
```

---

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git
- Visual Studio Code or another Python IDE

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/aircraft-engine-predictive-maintenance.git
```

### 2. Navigate to the project directory

```bash
cd aircraft-engine-predictive-maintenance
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit plotly
```

### 6. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard should open in your default web browser.

---

## 🔍 Example Prediction

Example output for Engine ID 5:

```text
Engine ID: 5

Current Operating Cycle: 98

Predicted Remaining Useful Life: 92.98 cycles

Maintenance Status: NORMAL
```

The example is based on the project's test dataset and model output.

---

## 🔮 Future Improvements

Potential future development includes:

- Explainable AI using SHAP.
- Advanced sensor anomaly detection.
- Maintenance recommendation generation.
- Comparison of multiple Machine Learning algorithms.
- Digital twin simulation.
- Model optimization and hyperparameter tuning.
- Cloud deployment of the dashboard.

---

## ⚠️ Disclaimer

This project is developed for educational, research, and portfolio purposes.

The dataset represents simulated turbofan engine degradation. Predictions and maintenance alerts are experimental and must not be used to make actual aircraft maintenance, airworthiness, or flight-safety decisions.

This system is not certified or approved for operational aviation use.

---

## 👨‍💻 Author

**Amintha Lakshan**

BSc (Hons) in Information Technology – Specializing in Artificial Intelligence

Interests:

- Artificial Intelligence & Machine Learning
- Aviation Technology
- Predictive Maintenance
- Data Science
- Automotive and Aerospace Technologies

---

## ⭐ Support

If you find this project interesting, consider giving the repository a star on GitHub!

**Bringing Artificial Intelligence closer to Aviation Engineering.** ✈️
