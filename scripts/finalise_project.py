"""Create a compact final project evidence summary from generated artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_PATH = RESULTS_DIR / "final_project_summary.json"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = TABLES_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    summary: dict[str, object] = {
        "project": "am2-01-predictive-maintenance",
        "generated_from": "local reproducible result artefacts",
    }

    cv = _read_csv("cv_benchmark.csv")
    if cv is not None and not cv.empty:
        best = cv.sort_values("average_precision_mean", ascending=False).iloc[0]
        summary["cross_validation"] = {
            "reference_model": str(best["model"]),
            "average_precision_mean": float(best["average_precision_mean"]),
            "average_precision_std": float(best["average_precision_std"]),
        }

    calibration = _read_csv("calibration_metrics.csv")
    if calibration is not None and not calibration.empty:
        summary["calibration"] = calibration.to_dict(orient="records")

    threshold = _read_csv("selected_threshold.csv")
    if threshold is not None and not threshold.empty:
        summary["threshold_policy"] = threshold.iloc[0].to_dict()

    robustness = _read_csv("robustness_repeated_splits.csv")
    if robustness is not None and not robustness.empty:
        summary["robustness"] = {
            "average_precision_mean": float(robustness["average_precision"].mean()),
            "average_precision_std": float(robustness["average_precision"].std()),
            "brier_score_mean": float(robustness["brier_score"].mean()),
        }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
