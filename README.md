# PredictOps AI
### Intelligent Predictive Maintenance Using Machine Learning

PredictOps AI is an end-to-end predictive maintenance platform that estimates the Remaining Useful Life (RUL) of turbofan engines using historical sensor measurements. It combines machine learning, temporal feature engineering, explainable AI, REST APIs and an interactive monitoring dashboard.

## Project Overview

Unexpected engine failures can cause operational downtime and expensive maintenance. PredictOps AI uses historical engine telemetry to identify degradation patterns and estimate how many operating cycles remain before failure.

**Dataset:** NASA C-MAPSS — FD001

## Key Features

- **Predictive Modelling:** Estimates turbofan engine Remaining Useful Life.
- **Temporal Feature Engineering:** Rolling means, standard deviations and sensor trends.
- **Model Benchmarking:** Random Forest, XGBoost and LightGBM.
- **Hyperparameter Optimization:** Optuna-based XGBoost tuning.
- **Explainable AI:** SHAP-based feature contribution analysis.
- **REST API:** FastAPI prediction endpoint.
- **Interactive Dashboard:** Streamlit-based engine monitoring and sensor editing.
- **Containerization:** Docker and Docker Compose deployment.

## Machine Learning Pipeline

1. Load and inspect NASA C-MAPSS FD001 data.
2. Perform exploratory data analysis and sensor analysis.
3. Generate RUL targets with a training cap of 125 cycles.
4. Engineer temporal features from eight selected sensors.
5. Split data by engine using GroupShuffleSplit to prevent engine-level leakage.
6. Train and benchmark Random Forest, XGBoost and LightGBM.
7. Tune XGBoost hyperparameters using Optuna.
8. Interpret model behaviour using SHAP.
9. Evaluate against NASA's official test dataset.

## Model Performance

### Validation Results

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 10.23 | 15.43 |
| XGBoost | 10.16 | 15.11 |
| LightGBM | 10.18 | 15.29 |
| Optimized XGBoost | 10.33 | 14.83 |

*Validation metrics are measured in operating cycles. Optuna tuning reduced RMSE by approximately 1.88%, while MAE increased slightly.*

## Explainable AI

SHAP TreeExplainer is used to investigate how individual features influence model predictions.

Analysis includes:
- SHAP summary (beeswarm) plot.
- Global feature importance.
- Sensor degradation feature contributions.

## Application Architecture

```text
NASA C-MAPSS Dataset
        |
        v
Data Preprocessing & EDA
        |
        v
Temporal Feature Engineering
        |
        v
Model Training & Benchmarking
        |
        v
Optuna Hyperparameter Tuning
        |
        v
SHAP Explainability
        |
        v
Model Serialization (Joblib)
        |
        v
      FastAPI
        |
        v
   Streamlit Dashboard
        |
        v
 Docker Compose Deployment
```

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Optimization | Optuna |
| Explainable AI | SHAP |
| Backend | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Serialization | Joblib |
| Deployment | Docker, Docker Compose |

## Running Locally


### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PredictOps-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
python -m uvicorn api.main:app --reload
```

API documentation: `http://localhost:8000/docs`

### 4. Start the dashboard

Open another terminal:

```bash
streamlit run dashboard/app.py
```

Dashboard: `http://localhost:8501`

## Running with Docker

Build and start both services:

```bash
docker compose up -d --build
```

Check container status:

```bash
docker compose ps
```

Access the application:

- Dashboard: `http://localhost:8502`
- API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

Stop the application:

```bash
docker compose down
```

## Project Structure

```text
PredictOps-AI/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── artifacts/
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   ├── demo_engines.json
│   ├── demo_engine_history.json
│   └── sensor_ranges.json
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md
```

## Future Improvements

- Centralize feature engineering into a reusable training and inference pipeline.
- Add individual prediction explanations using SHAP.
- Implement sensor anomaly detection.
- Add automated testing and CI/CD.
- Improve model monitoring and prediction reliability.

## Disclaimer

Predicted RUL is a machine learning estimate based on historical data. It is not a certified airworthiness assessment or a substitute for approved maintenance procedures.

## Dataset Reference

NASA — Prognostics Center of Excellence  
C-MAPSS Aircraft Engine Simulator Dataset.

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/