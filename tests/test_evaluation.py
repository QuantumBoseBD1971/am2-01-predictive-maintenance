import numpy as np
import pandas as pd

from predictive_maintenance.evaluation import (
    CostConfig,
    select_cost_optimal_threshold,
    threshold_table,
)


def test_threshold_table_counts_outcomes() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.4, 0.9])

    table = threshold_table(
        y_true,
        probabilities,
        thresholds=np.array([0.5]),
        cost_config=CostConfig(
            false_negative_cost=10.0,
            false_positive_cost=1.0,
        ),
    )

    row = table.iloc[0]
    assert row["true_negative"] == 1
    assert row["false_positive"] == 1
    assert row["false_negative"] == 1
    assert row["true_positive"] == 1
    assert row["expected_cost"] == 11.0


def test_select_cost_optimal_threshold() -> None:
    table = pd.DataFrame(
        {
            "threshold": [0.2, 0.5, 0.8],
            "expected_cost": [7.0, 3.0, 5.0],
        }
    )

    assert select_cost_optimal_threshold(table) == 0.5
