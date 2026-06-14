from renewable_energy_forecasting.data import FEATURE_COLUMNS, generate_synthetic_weather_energy_data


def test_generate_synthetic_data_schema_and_target() -> None:
    data = generate_synthetic_weather_energy_data(n_samples=50, seed=1)
    expected_columns = FEATURE_COLUMNS + ["grid_output_mw"]
    assert list(data.columns) == expected_columns
    assert len(data) == 50
    assert (data["grid_output_mw"] >= 0).all()
