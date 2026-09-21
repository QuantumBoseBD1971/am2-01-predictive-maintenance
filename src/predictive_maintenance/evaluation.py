"""Advanced evaluation utilities: thresholds, calibration and robustness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class CostConfig:
    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0


def threshold_table(
    y_true,
    probabilities,
    thresholds: np.ndarray | None = None,
    cost_config: CostConfig | None = None,
) -> pd.DataFrame:
    """Evaluate classification outcomes across probability thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    if cost_config is None:
        cost_config = CostConfig()

    y_true_array = np.asarray(y_true, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        prediction = (probabilities_array >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true_array,
            prediction,
            labels=[0, 1],
        ).ravel()

        total_cost = (
            fn * cost_config.false_negative_cost
            + fp * cost_config.false_positive_cost
        )

        rows.append(
            {
                "threshold": float(threshold),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "expected_cost": float(total_cost),
            }
        )

    return pd.DataFrame(rows)


def select_cost_optimal_threshold(table: pd.DataFrame) -> float:
    """Return the threshold with the minimum expected cost."""
    required = {"threshold", "expected_cost"}
    if not required.issubset(table.columns):
        missing = sorted(required.difference(table.columns))
        raise KeyError(f"Missing required columns: {missing}")

    best_row = table.sort_values(
        ["expected_cost", "threshold"],
        ascending=[True, True],
    ).iloc[0]
    return float(best_row["threshold"])


def probability_summary(probabilities) -> dict[str, float]:
    """Return simple probability diagnostics for reporting."""
    values = np.asarray(probabilities, dtype=float)
    return {
        "min_probability": float(values.min()),
        "mean_probability": float(values.mean()),
        "max_probability": float(values.max()),
        "std_probability": float(values.std()),
    }
