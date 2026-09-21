# AM2-01 — Predictive Maintenance Benchmark

A comparative machine-learning project for predictive maintenance using the **UCI AI4I 2020 Predictive Maintenance Dataset**.

The project is designed as AM2 portfolio evidence and demonstrates a complete supervised-learning workflow: data acquisition, validation, preprocessing, exploratory analysis, comparative modelling, evaluation, calibration, explainability, testing, and reproducibility.

## Project question

> How reliably can machine-learning models identify machine-failure risk from operational measurements, and how do simple interpretable models compare with more flexible ensemble models?

## Dataset

Source: UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset.

- 10,000 observations
- synthetic but designed to reflect industrial predictive-maintenance data
- target: `Machine failure`
- predictors include air temperature, process temperature, rotational speed, torque, tool wear, and product type

Dataset DOI: `10.24432/C5HS5C`

The raw dataset is **not committed**. It is downloaded reproducibly by the project code.

## Model benchmark

1. Dummy classifier
2. Logistic regression
3. Random forest
4. Gradient boosting
5. Optional later extension: XGBoost / LightGBM
6. Optional later extension: compact neural-network classifier

## Evaluation strategy

Because the target is imbalanced, accuracy is not used as the main selection criterion. Current evaluation includes:

- precision
- recall
- F1
- ROC-AUC
- average precision
- Brier score
- stratified 5-fold cross-validation

Later phases add threshold/cost analysis, calibration plots, explainability and robustness tests.

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
pytest
```

Generated outputs are written under `results/` and intentionally excluded from version control.

## Development status

- **Phase 1 — complete:** package scaffold, reproducible data acquisition, baseline pipeline, CI and tests.
- **Phase 2 — in progress:** EDA, class-imbalance diagnostics and stratified cross-validated benchmarking.
- **Phase 3 — planned:** calibration, threshold/cost analysis, explainability and robustness.
- **Phase 4 — planned:** experiment tracking, final model comparison and AM2 evidence synthesis.

## Responsible use

This project is educational and experimental. The dataset is synthetic, so results must not be interpreted as evidence of deployment performance on real machinery.

## Licence

Code: MIT License. Dataset: see the UCI dataset page and its stated licence/citation requirements.
