import pandas as pd

from predictive_maintenance.data import normalise_columns
from predictive_maintenance.modeling import candidate_models


def test_normalise_columns() -> None:
    df = pd.DataFrame(columns=["Air temperature [K]", "Machine failure"])
    out = normalise_columns(df)
    assert list(out.columns) == ["air_temperature_k", "machine_failure"]


def test_candidate_models_contains_interpretable_baseline() -> None:
    models = candidate_models()
    assert "dummy" in models
    assert "logistic_regression" in models
