import pandas as pd

from predictive_maintenance.analysis import summarise_dataset


def test_summarise_dataset_reports_class_balance() -> None:
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "machine_failure": [0, 0, 0, 1],
        }
    )

    summary = summarise_dataset(df, "machine_failure")

    assert summary.rows == 4
    assert summary.positives == 1
    assert summary.negatives == 3
    assert summary.positive_rate == 0.25
