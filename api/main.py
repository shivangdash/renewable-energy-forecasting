from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from renewable_energy_forecasting.model import ModelService


class WeatherFeatures(BaseModel):
    region_id: int = Field(ge=0, le=20)
    is_solar: int = Field(ge=0, le=1)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    solar_radiation: float = Field(ge=0)
    pressure: float
    cloud_cover: float = Field(ge=0, le=100)
    precipitation: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: float
    shap_values: dict[str, float]


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()


app = FastAPI(title="Renewable Energy Forecasting API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info(model_service: ModelService = Depends(get_model_service)) -> dict:
    return model_service.model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: WeatherFeatures, model_service: ModelService = Depends(get_model_service)
) -> PredictionResponse:
    try:
        result = model_service.predict_with_explanations(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train model first.") from exc
    return PredictionResponse(prediction=result.prediction, shap_values=result.shap_values)
