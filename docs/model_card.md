# Model Card

## Model family

This repository benchmarks several supervised binary classifiers rather than presenting one fixed production model:

- Dummy classifier
- Logistic regression
- Random forest
- Gradient boosting

A reference model may be selected from the benchmark outputs, but the repository deliberately avoids hard-coding a permanent "champion" without evidence from the generated experiments.

## Intended use

Educational predictive-maintenance benchmarking and AM2 portfolio evidence.

The model output represents estimated machine-failure risk from operational variables in the AI4I 2020 dataset.

## Out-of-scope use

The models must not be used to make real industrial maintenance or safety decisions.

## Training data

UCI AI4I 2020 Predictive Maintenance Dataset.

The source dataset is synthetic, which is a major limitation for external validity.

## Evaluation

The project evaluates:

- precision
- recall
- F1
- ROC-AUC
- average precision
- Brier score
- stratified cross-validation
- calibration
- cost-sensitive threshold selection
- repeated-split robustness
- permutation importance

## Decision policy

The classification threshold is not assumed to be 0.5.

Phase 3 includes an illustrative cost ratio in which a false negative is treated as ten times more costly than a false positive. This is a teaching assumption only and must be replaced by domain-approved costs for any real deployment.

## Risks and limitations

- synthetic rather than live industrial data
- possible distribution shift between machines/sites
- rare-event instability
- maintenance outcomes may have delayed or censored labels
- model calibration can degrade over time
- feature importance does not establish causality
- operational intervention may itself change the future data distribution

## Human oversight

A real deployment should support, not replace, qualified maintenance staff. Alerts should be reviewable and accompanied by relevant measurements, uncertainty and model/version metadata.
