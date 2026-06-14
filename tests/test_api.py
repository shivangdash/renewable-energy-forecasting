import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app, get_model_service
from renewable_energy_forecasting.data import FEATURE_COLUMNS, generate_synthetic_weather_energy_data
from renewable_energy_forecasting.model import ModelService
import xgboost as xgb


def _train_test_model(path: str) -> None:
    data = generate_synthetic_weather_energy_data(n_samples=120, seed=4)
    X = data[FEATURE_COLUMNS]
    y = data["grid_output_mw"]
    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=4)
    model.fit(X, y)
    model.save_model(path)


def test_predict_endpoint_returns_prediction_and_shap(tmp_path: Path) -> None:
    model_path = tmp_path / "test_model.json"
    log_path = tmp_path / "predictions.log"
    _train_test_model(str(model_path))

    app.dependency_overrides[get_model_service] = lambda: ModelService(
        model_path=str(model_path), prediction_log_path=str(log_path)
    )
    client = TestClient(app)

    payload = {
        "region_id": 2,
        "is_solar": 1,
        "temperature": 25.0,
        "humidity": 40.0,
        "wind_speed": 5.0,
        "solar_radiation": 700.0,
        "pressure": 1012.0,
        "cloud_cover": 20.0,
        "precipitation": 0.5,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert set(body["shap_values"].keys()) == set(FEATURE_COLUMNS)
    assert os.path.exists(log_path)

    app.dependency_overrides.clear()
