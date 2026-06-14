from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from renewable_energy_forecasting.config import SETTINGS
from renewable_energy_forecasting.data import FEATURE_COLUMNS, generate_synthetic_weather_energy_data


def train_and_log_model(n_samples: int = 5000, seed: int = 42) -> dict:
    os.makedirs(SETTINGS.model_output_dir, exist_ok=True)

    data = generate_synthetic_weather_energy_data(n_samples=n_samples, seed=seed)
    X = data[FEATURE_COLUMNS]
    y = data["grid_output_mw"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
    }
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    versioned_model_path = os.path.join(SETTINGS.model_output_dir, f"xgb_model_{timestamp}.json")
    latest_model_path = os.path.join(SETTINGS.model_output_dir, SETTINGS.model_file_name)
    model.save_model(versioned_model_path)
    model.save_model(latest_model_path)

    mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
    mlflow.set_experiment(SETTINGS.experiment_name)

    with mlflow.start_run():
        mlflow.log_params({"n_samples": n_samples, "seed": seed, **model_params})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.log_artifact(versioned_model_path)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=SETTINGS.model_registry_name,
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": latest_model_path,
        "versioned_model_path": versioned_model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model for renewable energy forecasting")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_and_log_model(n_samples=args.n_samples, seed=args.seed)
    print(metrics)


if __name__ == "__main__":
    main()
