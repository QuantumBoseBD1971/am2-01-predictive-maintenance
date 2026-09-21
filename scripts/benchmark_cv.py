"""Cross-validated model benchmark for imbalanced classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from predictive_maintenance.config import RANDOM_STATE, TARGET_COLUMN
from predictive_maintenance.data import load_dataset
from predictive_maintenance.modeling import candidate_models, make_pipeline

RESULTS_PATH = Path("results/tables/cv_benchmark.csv")


def main() -> None:
    df = load_dataset()
    drop_columns = [c for c in ["uid", "product_id"] if c in df.columns]

    X = df.drop(columns=[TARGET_COLUMN, *drop_columns])
    y = df[TARGET_COLUMN].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "neg_brier_score": "neg_brier_score",
    }

    rows: list[dict[str, float | str]] = []

    for name, estimator in candidate_models().items():
        pipeline = make_pipeline(X, estimator)
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            metric_name = "brier_score" if metric == "neg_brier_score" else metric
            mean_value = float(values.mean())
            if metric == "neg_brier_score":
                mean_value *= -1
            row[f"{metric_name}_mean"] = mean_value
            row[f"{metric_name}_std"] = float(values.std())
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("average_precision_mean", ascending=False)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
