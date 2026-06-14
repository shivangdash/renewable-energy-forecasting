import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "renewable-energy-forecasting")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    model_registry_name: str = os.getenv("MODEL_REGISTRY_NAME", "RenewableEnergyXGBoost")
    model_output_dir: str = os.getenv("MODEL_OUTPUT_DIR", "models")
    model_file_name: str = os.getenv("MODEL_FILE_NAME", "latest_model.json")
    prediction_log_path: str = os.getenv("PREDICTION_LOG_PATH", "logs/predictions.log")


SETTINGS = Settings()
