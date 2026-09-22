# Real Experiment Results

This evidence pack was generated from an executed end-to-end run of the project.
It is intentionally small enough to keep in Git while raw data and transient outputs remain excluded.

## Cross-validation

- Reference model: **random_forest**
- Mean average precision: **0.9761**
- Average precision std: **0.0050**

## Threshold policy

- Selected threshold: **0.0500**
- False-negative cost: **10.0**
- False-positive cost: **1.0**

## Robustness

- Mean repeated-split average precision: **0.9706**
- Repeated-split AP std: **0.0110**
- Mean Brier score: **0.0018**

## Calibration

- random_forest_uncalibrated: Brier=0.0018, average precision=0.9763
- random_forest_calibrated: Brier=0.0013, average precision=0.9784

## Leading permutation-importance features

- hdf: mean importance 0.2170 (std 0.0056)
- twf: mean importance 0.1088 (std 0.0026)
- osf: mean importance 0.0802 (std 0.0043)
- pwf: mean importance 0.0590 (std 0.0029)
- tool_wear: mean importance 0.0045 (std 0.0004)
- process_temperature: mean importance 0.0006 (std 0.0006)
- torque: mean importance 0.0004 (std 0.0018)
- rnf: mean importance -0.0000 (std 0.0000)

## Evidence files

- `tables/cv_benchmark.csv`
- `tables/calibration_metrics.csv`
- `tables/selected_threshold.csv`
- `tables/permutation_importance.csv`
- `tables/robustness_repeated_splits.csv`
- `figures/class_distribution.png`
- `figures/calibration_curve.png`
- `final_project_summary.json`

Generated automatically by `scripts/build_evidence_pack.py`.
