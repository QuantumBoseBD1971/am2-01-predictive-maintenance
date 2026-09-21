# Final Project Summary

## Research question

How reliably can supervised machine-learning models identify machine-failure risk from operational measurements, and what is gained or lost as model complexity increases?

## Technical scope

The repository compares classical and ensemble models under a shared preprocessing and evaluation framework.

The evidence chain covers:

- reproducible UCI data acquisition
- class-imbalance analysis
- baseline and nonlinear classifiers
- stratified cross-validation
- calibration
- threshold/cost analysis
- permutation importance
- repeated-split robustness
- CI and tests
- experiment tracking
- deployment/MLOps design

## Interpretation principle

The repository does **not** claim that a single model is universally best.

The final choice should be based on generated benchmark results plus the operational objective. In a real maintenance setting that would include safety constraints, alert capacity, maintenance cost, calibration quality and robustness.

## Reproducing the full analysis

Run:

```bash
python scripts/download_data.py
python scripts/run_eda.py
python scripts/train_baseline.py
python scripts/benchmark_cv.py
python scripts/run_phase3_analysis.py
python scripts/finalise_project.py
pytest
```

The generated result files under `results/` remain local and are intentionally excluded from Git to keep the repository lightweight and reproducible.
