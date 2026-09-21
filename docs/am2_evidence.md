# AM2 Evidence Notes

This repository is a self-contained AM2 portfolio example demonstrating the engineering lifecycle of an imbalanced supervised-learning system.

## Problem framing

Predictive maintenance is formulated as a binary classification problem where the positive class is rare and false negatives may be operationally expensive.

## Data engineering

Evidence includes:

- reproducible public-data acquisition
- column normalisation
- local persistence
- integrity checks
- deterministic loading
- identifier/leakage considerations

## Exploratory analysis

The automated EDA quantifies:

- class prevalence
- missingness
- duplicates
- numeric feature distributions

## Model development

The common pipeline benchmarks:

- dummy prior classifier
- logistic regression
- random forest
- gradient boosting

This supports comparison between simple/interpretable and more flexible nonlinear models.

## Evaluation

The project uses:

- stratified 5-fold cross-validation
- precision
- recall
- F1
- ROC-AUC
- average precision
- Brier score

Accuracy is not the primary criterion because of target imbalance.

## Calibration and decision thresholds

The project compares uncalibrated and sigmoid-calibrated probabilities.

It separates probability estimation from the operational threshold and demonstrates threshold selection under an explicit illustrative cost model.

## Explainability

Permutation importance is calculated on held-out data using average precision as the scoring function.

## Robustness

Repeated stratified hold-out splits test whether conclusions are sensitive to one random partition.

## Experiment tracking

A lightweight immutable JSONL experiment registry records:

- run id
- timestamp
- experiment
- model
- parameters
- metrics
- notes

The design can later be migrated to a full model registry such as MLflow.

## MLOps and deployment

The documented production design covers:

- packaging
- model/version metadata
- immutable artefacts
- champion aliasing rather than overwriting models
- monitoring
- drift
- promotion
- rollback

## Responsible AI and limitations

The dataset is synthetic and therefore does not establish safety or effectiveness on real equipment.

A real deployment would require representative operational data, domain-expert review, time-aware validation, approved maintenance costs, drift monitoring and human oversight.

## Reflection

The project demonstrates that model development is only one component of AI engineering. The decision system also depends on data quality, calibration, thresholds, reproducibility, explainability, operational costs, monitoring and governance.
