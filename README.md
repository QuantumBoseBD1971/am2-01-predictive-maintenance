# AM2-01 — Predictive Maintenance Benchmark

A comparative machine-learning project for predictive maintenance using the **UCI AI4I 2020 Predictive Maintenance Dataset**.

The project is designed as AM2 portfolio evidence and demonstrates a complete supervised-learning workflow: data acquisition, validation, preprocessing, baseline modelling, comparative evaluation, probability calibration, explainability, testing, and reproducibility.

## Project question

> How reliably can machine-learning models identify machine-failure risk from operational measurements, and how do simple interpretable models compare with more flexible ensemble models?

## Dataset

Source: UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset.

- 10,000 observations
- synthetic but designed to reflect industrial predictive-maintenance data
- target: `Machine failure`
- key predictors include air temperature, process temperature, rotational speed, torque, tool wear, and product type
- no missing values reported by the source

Dataset DOI: `10.24432/C5HS5C`

The raw dataset is **not committed** to the repository. It is downloaded reproducibly with the project code.

## Planned model benchmark

1. Dummy classifier — establishes the minimum baseline
2. Logistic regression — interpretable linear probabilistic baseline
3. Random forest — nonlinear ensemble benchmark
4. Gradient boosting — sequential tree ensemble benchmark
5. Optional extension: XGBoost / LightGBM
6. Optional extension: compact neural-network classifier

## Evaluation

Because machine failure is an imbalanced classification problem, accuracy alone is insufficient. The benchmark will emphasise:

- precision
- recall
- F1
- ROC-AUC
- PR-AUC / average precision
- confusion matrix
- calibration curve / Brier score

Threshold selection will be treated as a business decision rather than assuming 0.5 is always optimal.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── configs/
├── data/
│   └── README.md
├── docs/
│   ├── am2_evidence.md
│   ├── dataset.md
│   └── methodology.md
├── scripts/
│   ├── download_data.py
│   └── train_baseline.py
├── src/
│   └── predictive_maintenance/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       └── modeling.py
├── tests/
│   └── test_smoke.py
├── pyproject.toml
└── README.md
```

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
python scripts/train_baseline.py
pytest
```

## Current status

**Phase 1 — repository scaffold and reproducible baseline pipeline.**

Next phases will add deeper EDA, imbalance analysis, expanded benchmarking, calibration, threshold analysis, explainability, experiment tracking, and final AM2 evidence mapping.

## Responsible use

This project is educational and experimental. It does not provide real industrial maintenance decisions. The source dataset is synthetic, so results must not be interpreted as evidence of deployment performance on real machinery.

## Licence

Code: MIT License. Dataset: see the UCI dataset page and its stated licence/citation requirements.
