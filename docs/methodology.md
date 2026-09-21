# Methodology

## Research question

How reliably can supervised machine-learning models identify machine-failure risk from machine operating measurements, and what is gained or lost as model complexity increases?

## Experimental design

The first benchmark uses a stratified hold-out split so the rare positive class is represented in both train and test partitions.

The initial comparison contains:

- dummy prior classifier
- logistic regression with balanced class weighting
- random forest with balanced class weighting
- gradient boosting

All estimators are wrapped in a common preprocessing pipeline.

## Why accuracy is not the primary metric

Predictive-maintenance failures are rare. A model that predicts "no failure" almost all the time may obtain high accuracy while being operationally useless.

Therefore the project emphasises:

- **Recall** — fraction of true failures detected
- **Precision** — fraction of failure alerts that are correct
- **F1** — harmonic balance of precision and recall
- **ROC-AUC** — ranking quality across thresholds
- **Average precision / PR-AUC proxy** — performance focused on the positive class
- **Brier score** — probability-calibration quality

## Next methodological layers

1. exploratory data analysis and imbalance quantification
2. cross-validated benchmarking
3. threshold optimisation under explicit cost assumptions
4. calibration comparison
5. feature importance and model explainability
6. robustness checks
7. optional external model families
8. experiment tracking and reproducible artefacts
