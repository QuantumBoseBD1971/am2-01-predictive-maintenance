"""Train and compare the initial baseline models."""

from dataclasses import asdict

import pandas as pd
from sklearn.model_selection import train_test_split

from predictive_maintenance.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from predictive_maintenance.data import load_dataset
from predictive_maintenance.modeling import (
    candidate_models,
    evaluate_binary_classifier,
    make_pipeline,
)


def main() -> None:
    df = load_dataset()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Expected target '{TARGET_COLUMN}' not found. "
            f"Available columns: {sorted(df.columns)}"
        )

    # Identifiers are intentionally excluded from the first benchmark.
    drop_columns = [c for c in ["uid", "product_id"] if c in df.columns]
    X = df.drop(columns=[TARGET_COLUMN, *drop_columns])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    for name, estimator in candidate_models().items():
        pipeline = make_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        result = evaluate_binary_classifier(name, pipeline, X_test, y_test)
        results.append(asdict(result))

    table = pd.DataFrame(results).sort_values(
        ["average_precision", "recall"],
        ascending=False,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
