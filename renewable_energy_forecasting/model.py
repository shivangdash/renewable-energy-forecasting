from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from renewable_energy_forecasting.config import SETTINGS
from renewable_energy_forecasting.data import FEATURE_COLUMNS


logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    prediction: float
    shap_values: Dict[str, float]


class ModelService:
    def __init__(self, model_path: str | None = None, prediction_log_path: str | None = None) -> None:
        self.model_path = model_path or os.path.join(SETTINGS.model_output_dir, SETTINGS.model_file_name)
        self.prediction_log_path = prediction_log_path or SETTINGS.prediction_log_path
        self.model = self._load_model()
        self.explainer = shap.TreeExplainer(self.model)

    def _load_model(self) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor()
        model.load_model(self.model_path)
        return model

    def predict_with_explanations(self, features: Dict[str, float]) -> PredictionResult:
        frame = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        prediction = float(self.model.predict(frame)[0])
        shap_raw = self.explainer.shap_values(frame)
        shap_vector = np.array(shap_raw).reshape(-1)
        shap_values = {col: float(shap_vector[idx]) for idx, col in enumerate(FEATURE_COLUMNS)}
        self._log_prediction(features, prediction, shap_values)
        return PredictionResult(prediction=prediction, shap_values=shap_values)

    def _log_prediction(self, features: Dict[str, float], prediction: float, shap_values: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(self.prediction_log_path), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": features,
            "prediction": prediction,
            "shap_values": shap_values,
        }
        with open(self.prediction_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("prediction_logged")

    def model_info(self) -> Dict[str, List[str] | str]:
        return {
            "model_type": "xgboost-regressor",
            "model_path": self.model_path,
            "features": FEATURE_COLUMNS,
        }
