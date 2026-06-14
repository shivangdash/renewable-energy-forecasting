# Renewable Energy Forecasting (MLOps)

MLOps project for regional renewable grid output forecasting (wind/solar) using synthetic weather data.

## Features
- Synthetic data generation for weather + grid output
- XGBoost training pipeline with MLflow tracking and model registration
- FastAPI inference service with SHAP explainability
- Docker + Docker Compose setup (API + MLflow)
- EU AI Act transparency/fairness notes

## Project structure
- `renewable_energy_forecasting/data.py` - synthetic data generation
- `renewable_energy_forecasting/train.py` - model training + MLflow logging
- `renewable_energy_forecasting/model.py` - model loading, prediction, SHAP explanation logging
- `api/main.py` - FastAPI endpoints
- `tests/` - focused unit tests
- `docs/ai-act-compliance.md` - transparency/fairness documentation

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model
```bash
python -m renewable_energy_forecasting.train --n-samples 5000 --seed 42
```

Environment variables:
- `MLFLOW_TRACKING_URI` (default: `file:./mlruns`)
- `MLFLOW_EXPERIMENT_NAME` (default: `renewable-energy-forecasting`)
- `MODEL_REGISTRY_NAME` (default: `RenewableEnergyXGBoost`)
- `MODEL_OUTPUT_DIR` (default: `models`)
- `MODEL_FILE_NAME` (default: `latest_model.json`)
- `PREDICTION_LOG_PATH` (default: `logs/predictions.log`)

## Run API locally
```bash
uvicorn api.main:app --reload
```

Endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict`

Example request:
```json
{
  "region_id": 2,
  "is_solar": 1,
  "temperature": 25.0,
  "humidity": 40.0,
  "wind_speed": 5.0,
  "solar_radiation": 700.0,
  "pressure": 1012.0,
  "cloud_cover": 20.0,
  "precipitation": 0.5
}
```

## Docker Compose
```bash
docker compose up --build
```

- API: `http://localhost:8000`
- MLflow: `http://localhost:5000`
