"""Run calibration, threshold-cost, explainability and robustness analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

from predictive_maintenance.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import (
    CostConfig,
    select_cost_optimal_threshold,
    threshold_table,
)
from predictive_maintenance.modeling import candidate_models, make_pipeline

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def split_data():
    df = load_dataset()
    drop_columns = [c for c in ["uid", "product_id"] if c in df.columns]

    X = df.drop(columns=[TARGET_COLUMN, *drop_columns])
    y = df[TARGET_COLUMN].astype(int)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def choose_reference_model(X_train, X_test, y_train, y_test):
    """Select a reference model using hold-out average precision."""
    candidates = {
        name: estimator
        for name, estimator in candidate_models().items()
        if name != "dummy"
    }

    scored = []
    fitted = {}

    for name, estimator in candidates.items():
        pipeline = make_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        score = average_precision_score(y_test, probabilities)
        scored.append({"model": name, "average_precision": float(score)})
        fitted[name] = pipeline

    ranking = pd.DataFrame(scored).sort_values(
        "average_precision",
        ascending=False,
    )
    best_name = str(ranking.iloc[0]["model"])
    return best_name, fitted[best_name], ranking


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data()

    model_name, base_model, ranking = choose_reference_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    ranking.to_csv(TABLES_DIR / "phase3_reference_model_ranking.csv", index=False)

    base_probabilities = base_model.predict_proba(X_test)[:, 1]

    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    calibrated_probabilities = calibrated.predict_proba(X_test)[:, 1]

    calibration_metrics = pd.DataFrame(
        [
            {
                "model": f"{model_name}_uncalibrated",
                "brier_score": brier_score_loss(y_test, base_probabilities),
                "average_precision": average_precision_score(
                    y_test,
                    base_probabilities,
                ),
            },
            {
                "model": f"{model_name}_calibrated",
                "brier_score": brier_score_loss(
                    y_test,
                    calibrated_probabilities,
                ),
                "average_precision": average_precision_score(
                    y_test,
                    calibrated_probabilities,
                ),
            },
        ]
    )
    calibration_metrics.to_csv(
        TABLES_DIR / "calibration_metrics.csv",
        index=False,
    )

    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(
        y_test,
        base_probabilities,
        n_bins=10,
        name="Uncalibrated",
        ax=ax,
    )
    CalibrationDisplay.from_predictions(
        y_test,
        calibrated_probabilities,
        n_bins=10,
        name="Calibrated",
        ax=ax,
    )
    ax.set_title("Probability calibration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=160)
    plt.close(fig)

    costs = threshold_table(
        y_test,
        calibrated_probabilities,
        cost_config=CostConfig(
            false_negative_cost=10.0,
            false_positive_cost=1.0,
        ),
    )
    costs.to_csv(TABLES_DIR / "threshold_cost_analysis.csv", index=False)

    best_threshold = select_cost_optimal_threshold(costs)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "selected_threshold": best_threshold,
                "false_negative_cost": 10.0,
                "false_positive_cost": 1.0,
            }
        ]
    ).to_csv(TABLES_DIR / "selected_threshold.csv", index=False)

    importance = permutation_importance(
        base_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(
        TABLES_DIR / "permutation_importance.csv",
        index=False,
    )

    robustness_rows = []
    for seed in [7, 21, 42, 84, 168]:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            pd.concat([X_train, X_test]),
            pd.concat([y_train, y_test]),
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=pd.concat([y_train, y_test]),
        )
        model = make_pipeline(
            X_train_r,
            candidate_models()[model_name],
        )
        model.fit(X_train_r, y_train_r)
        probabilities = model.predict_proba(X_test_r)[:, 1]
        robustness_rows.append(
            {
                "seed": seed,
                "average_precision": average_precision_score(
                    y_test_r,
                    probabilities,
                ),
                "brier_score": brier_score_loss(
                    y_test_r,
                    probabilities,
                ),
            }
        )

    pd.DataFrame(robustness_rows).to_csv(
        TABLES_DIR / "robustness_repeated_splits.csv",
        index=False,
    )

    print(f"Reference model: {model_name}")
    print(f"Cost-optimal threshold: {best_threshold:.2f}")
    print(calibration_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
