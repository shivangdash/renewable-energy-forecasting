# EU AI Act Transparency and Fairness Notes

## Explainability and traceability
- The API returns per-feature SHAP values for every prediction.
- Prediction outputs and SHAP values are stored in `logs/predictions.log` for traceability.
- Model training runs are tracked in MLflow with parameters, metrics, and model artifacts.

## Model transparency
- Input features are documented in the API schema and `renewable_energy_forecasting/data.py`.
- The model is an `XGBRegressor` trained on synthetic weather and regional indicators.
- Model versions are persisted as timestamped files under `models/` and in MLflow Model Registry.

## Fairness considerations
- Inputs include a coarse numeric region identifier and weather factors only.
- Synthetic data does not represent all real-world regional effects; fairness should be validated on representative production datasets.
- Monitor prediction errors by region and energy type (`is_solar`) before production rollout.
