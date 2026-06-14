from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "region_id",
    "is_solar",
    "temperature",
    "humidity",
    "wind_speed",
    "solar_radiation",
    "pressure",
    "cloud_cover",
    "precipitation",
]


def generate_synthetic_weather_energy_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region_id = rng.integers(0, 8, n_samples)
    is_solar = rng.integers(0, 2, n_samples)
    temperature = rng.normal(18, 8, n_samples)
    humidity = rng.uniform(20, 95, n_samples)
    wind_speed = rng.gamma(shape=2.0, scale=3.0, size=n_samples)
    solar_radiation = np.clip(rng.normal(450, 220, n_samples), 0, None)
    pressure = rng.normal(1013, 9, n_samples)
    cloud_cover = rng.uniform(0, 100, n_samples)
    precipitation = np.clip(rng.normal(2, 4, n_samples), 0, None)

    wind_generation = (
        30
        + 9 * wind_speed
        - 0.15 * humidity
        + 0.01 * pressure
        - 0.5 * precipitation
        + 0.7 * region_id
    )

    solar_generation = (
        20
        + 0.06 * solar_radiation
        + 0.7 * temperature
        - 0.25 * cloud_cover
        - 0.08 * humidity
        + 0.5 * region_id
    )

    base = np.where(is_solar == 1, solar_generation, wind_generation)
    noise = rng.normal(0, 5, n_samples)
    grid_output_mw = np.clip(base + noise, 0, None)

    return pd.DataFrame(
        {
            "region_id": region_id,
            "is_solar": is_solar,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "solar_radiation": solar_radiation,
            "pressure": pressure,
            "cloud_cover": cloud_cover,
            "precipitation": precipitation,
            "grid_output_mw": grid_output_mw,
        }
    )
