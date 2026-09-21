# AM2 Evidence Notes

This document is a working evidence map and is updated as the project develops.

## Problem framing

Predictive maintenance is formulated as an imbalanced supervised binary-classification problem.

## Data engineering

The repository demonstrates reproducible source acquisition, column normalisation, local persistence, integrity checks and deterministic loading.

## Exploratory analysis

The automated EDA quantifies class prevalence, missingness, duplicates and numeric-feature distributions before model interpretation.

## Model development

A common preprocessing pipeline benchmarks a dummy baseline, logistic regression, random forest and gradient boosting.

## Evaluation

The project uses:

- stratified cross-validation
- precision
- recall
- F1
- ROC-AUC
- average precision
- Brier score

This avoids using accuracy as the primary decision criterion in an imbalanced problem.

## Calibration and decision thresholds

The project compares uncalibrated and sigmoid-calibrated probabilities.

Threshold selection is explicitly separated from model training and is driven by an illustrative false-negative vs false-positive cost ratio.

## Explainability

Permutation importance provides model-agnostic feature influence estimates on held-out data.

## Robustness

Repeated stratified hold-out splits test whether model performance is overly dependent on one random partition.

## Reproducibility

The project uses:

- packaged Python source
- declared dependencies
- fixed seeds
- tests
- GitHub Actions CI
- reproducible scripts
- generated result artefacts

## Responsible AI and limitations

The dataset is synthetic and cannot establish real industrial safety performance.

A real deployment would additionally require:

- representative machinery data
- domain-expert review
- explicit maintenance/safety cost modelling
- drift monitoring
- uncertainty monitoring
- human oversight
- retraining and rollback procedures

## Evidence still to add

- experiment tracking
- final comparative result summary
- model card
- final reflection
- deployment/MLOps discussion
