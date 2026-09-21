# Phase 3 — Calibration, Thresholds, Explainability and Robustness

## Purpose

Phase 3 moves beyond ranking models and asks whether predicted probabilities are trustworthy, how decision thresholds should be selected, which features influence predictions, and whether performance is stable across alternative data splits.

## Calibration

A classifier can rank failures correctly while still producing poorly calibrated probabilities.

For the selected reference model, the project compares:

- uncalibrated probabilities
- sigmoid-calibrated probabilities

The main calibration metric is **Brier score**, where lower values indicate probabilities closer to observed outcomes.

A calibration curve is also generated to compare predicted probability bins with observed failure frequencies.

## Threshold and cost analysis

The default 0.5 threshold is not assumed to be optimal.

The project evaluates thresholds from 0.05 to 0.95 and applies a transparent illustrative cost model:

- false negative cost = 10
- false positive cost = 1

This encodes the assumption that missing a real machine failure is more costly than creating a false maintenance alert.

The selected operational threshold is the threshold with minimum expected cost under those assumptions.

These costs are illustrative only. A real deployment would require maintenance engineers and business stakeholders to define credible financial and safety costs.

## Explainability

Permutation importance is calculated on the held-out test set using average precision as the scoring function.

This measures how much predictive performance deteriorates when each feature is randomly permuted.

Permutation importance is preferred here because it can be applied consistently to the complete fitted pipeline without requiring model-specific interpretation logic.

## Robustness

The selected model family is retrained across several stratified train/test splits using different random seeds.

For each split the project records:

- average precision
- Brier score

Large variation would indicate that conclusions are sensitive to the particular hold-out sample.

## Important limitation

The reference model is selected using hold-out performance in this educational phase. A stricter production study would use nested cross-validation or a separate validation set for model selection and preserve an untouched final test set.
