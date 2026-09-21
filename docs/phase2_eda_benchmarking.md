# Phase 2 — EDA and Benchmarking

## Purpose

Phase 2 adds reproducible exploratory analysis and cross-validated benchmarking before any model-selection decision is made.

## EDA checks

The automated EDA records:

- number of rows and columns
- positive and negative target counts
- positive-class prevalence
- total missing values
- duplicate rows
- descriptive statistics for numeric features
- class-distribution figure

This ensures that the class imbalance is quantified before interpreting model metrics.

## Cross-validation

The benchmark uses **5-fold stratified cross-validation** with shuffling and a fixed random seed.

Stratification is important because machine failures are rare. Each fold should preserve approximately the same positive-class proportion as the full dataset.

## Metrics

For every candidate model, the benchmark records mean and standard deviation for:

- precision
- recall
- F1
- ROC-AUC
- average precision
- Brier score

Average precision is the primary ranking metric in this phase because it focuses on positive-class ranking performance in an imbalanced problem.

No model will be declared the final champion from this benchmark alone. Calibration, threshold/cost analysis, explainability and robustness checks are still required.
