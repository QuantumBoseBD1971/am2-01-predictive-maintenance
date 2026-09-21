# AM2-01 — Predictive Maintenance Benchmark

A comparative machine-learning project for predictive maintenance using the **UCI AI4I 2020 Predictive Maintenance Dataset**.

This repository demonstrates an end-to-end supervised AI-engineering workflow: reproducible data acquisition, EDA, comparative modelling, cross-validation, probability calibration, threshold/cost analysis, explainability, robustness, experiment tracking, CI and deployment design.

## Project question

> How reliably can machine-learning models identify machine-failure risk from operational measurements, and how do simple interpretable models compare with more flexible ensemble models?

## Dataset

Source: UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset.

- 10,000 observations
- synthetic predictive-maintenance data
- target: `Machine failure`
- predictors include temperatures, rotational speed, torque, tool wear and product type

Dataset DOI: `10.24432/C5HS5C`

The raw dataset is downloaded reproducibly and is not committed.

## Model benchmark

- Dummy classifier
- Logistic regression
- Random forest
- Gradient boosting

## Evaluation strategy

The project includes:

- precision, recall and F1
- ROC-AUC
- average precision
- Brier score
- stratified 5-fold cross-validation
- probability calibration
- threshold/cost analysis
- permutation importance
- repeated-split robustness checks

Accuracy is deliberately not used as the primary model-selection metric because the target is imbalanced.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/download_data.py
python scripts/run_eda.py
python scripts/train_baseline.py
python scripts/benchmark_cv.py
python scripts/run_phase3_analysis.py
python scripts/finalise_project.py
pytest
```

Generated outputs are written under `results/` and excluded from version control.

## Documentation

- `docs/dataset.md` — data source and leakage considerations
- `docs/methodology.md` — baseline methodology
- `docs/phase2_eda_benchmarking.md` — EDA and CV design
- `docs/phase3_calibration_explainability.md` — calibration, thresholds and robustness
- `docs/model_card.md` — intended use, limitations and risks
- `docs/deployment_mlops.md` — productionisation, monitoring and rollback
- `docs/final_summary.md` — concise end-to-end project summary
- `docs/final_reflection.md` — learning and AM2 reflection
- `docs/am2_evidence.md` — consolidated evidence map

## Development status

- **Phase 1 — complete:** package scaffold, reproducible data acquisition, baseline pipeline, CI and tests.
- **Phase 2 — complete:** EDA, class imbalance and stratified cross-validation.
- **Phase 3 — complete:** calibration, threshold/cost analysis, explainability and robustness.
- **Phase 4 — complete:** experiment tracking, model card, MLOps design and AM2 evidence synthesis.

## Responsible use

This project is educational and experimental. The dataset is synthetic, so results must not be interpreted as evidence of deployment performance on real machinery.

## Licence

Code: MIT License. Dataset: see the UCI dataset page and its stated licence/citation requirements.
