"""Build a small assessor-facing evidence pack from real experiment outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
EVIDENCE_DIR = Path("evidence")
EVIDENCE_TABLES = EVIDENCE_DIR / "tables"
EVIDENCE_FIGURES = EVIDENCE_DIR / "figures"


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def top_feature_lines(limit: int = 8) -> list[str]:
    path = TABLES_DIR / "permutation_importance.csv"
    if not path.exists():
        return []

    frame = pd.read_csv(path).head(limit)
    return [
        f"- {row.feature}: mean importance {row.importance_mean:.4f} "
        f"(std {row.importance_std:.4f})"
        for row in frame.itertuples(index=False)
    ]


def main() -> None:
    EVIDENCE_TABLES.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FIGURES.mkdir(parents=True, exist_ok=True)

    selected_tables = [
        "dataset_summary.csv",
        "numeric_feature_summary.csv",
        "cv_benchmark.csv",
        "phase3_reference_model_ranking.csv",
        "calibration_metrics.csv",
        "selected_threshold.csv",
        "permutation_importance.csv",
        "robustness_repeated_splits.csv",
    ]

    for name in selected_tables:
        copy_if_exists(
            TABLES_DIR / name,
            EVIDENCE_TABLES / name,
        )

    selected_figures = [
        "class_distribution.png",
        "calibration_curve.png",
    ]
    for name in selected_figures:
        copy_if_exists(
            FIGURES_DIR / name,
            EVIDENCE_FIGURES / name,
        )

    summary_path = RESULTS_DIR / "final_project_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "results/final_project_summary.json is missing. "
            "Run the full experiment pipeline before building evidence."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    copy_if_exists(
        summary_path,
        EVIDENCE_DIR / "final_project_summary.json",
    )

    cv = summary.get("cross_validation", {})
    threshold = summary.get("threshold_policy", {})
    robustness = summary.get("robustness", {})
    calibration = summary.get("calibration", [])

    lines = [
        "# Real Experiment Results",
        "",
        "This evidence pack was generated from an executed end-to-end run of the project.",
        "It is intentionally small enough to keep in Git while raw data and transient outputs remain excluded.",
        "",
        "## Cross-validation",
        "",
        f"- Reference model: **{cv.get('reference_model', 'n/a')}**",
        f"- Mean average precision: **{cv.get('average_precision_mean', float('nan')):.4f}**",
        f"- Average precision std: **{cv.get('average_precision_std', float('nan')):.4f}**",
        "",
        "## Threshold policy",
        "",
        f"- Selected threshold: **{threshold.get('selected_threshold', float('nan')):.4f}**",
        f"- False-negative cost: **{threshold.get('false_negative_cost', 'n/a')}**",
        f"- False-positive cost: **{threshold.get('false_positive_cost', 'n/a')}**",
        "",
        "## Robustness",
        "",
        f"- Mean repeated-split average precision: **{robustness.get('average_precision_mean', float('nan')):.4f}**",
        f"- Repeated-split AP std: **{robustness.get('average_precision_std', float('nan')):.4f}**",
        f"- Mean Brier score: **{robustness.get('brier_score_mean', float('nan')):.4f}**",
        "",
        "## Calibration",
        "",
    ]

    for row in calibration:
        lines.append(
            f"- {row.get('model', 'model')}: "
            f"Brier={float(row.get('brier_score', float('nan'))):.4f}, "
            f"average precision={float(row.get('average_precision', float('nan'))):.4f}"
        )

    lines.extend(
        [
            "",
            "## Leading permutation-importance features",
            "",
            *top_feature_lines(),
            "",
            "## Evidence files",
            "",
            "- `tables/cv_benchmark.csv`",
            "- `tables/calibration_metrics.csv`",
            "- `tables/selected_threshold.csv`",
            "- `tables/permutation_importance.csv`",
            "- `tables/robustness_repeated_splits.csv`",
            "- `figures/class_distribution.png`",
            "- `figures/calibration_curve.png`",
            "- `final_project_summary.json`",
            "",
            "Generated automatically by `scripts/build_evidence_pack.py`.",
        ]
    )

    (EVIDENCE_DIR / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Evidence pack written to {EVIDENCE_DIR.resolve()}")


if __name__ == "__main__":
    main()
