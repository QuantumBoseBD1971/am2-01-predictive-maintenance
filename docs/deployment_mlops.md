# Deployment and MLOps Considerations

## Proposed production architecture

A realistic deployment could separate the workflow into:

1. sensor or operational data ingestion
2. feature validation
3. model inference service or scheduled batch scoring
4. alert/maintenance decision layer
5. model and prediction logging
6. drift and performance monitoring
7. retraining and controlled promotion

## Model packaging

The repository uses a Python package structure so preprocessing and inference logic can be imported consistently by:

- batch jobs
- APIs
- notebooks
- scheduled workflows

The preprocessing pipeline is fitted together with the estimator to reduce training-serving skew.

## Versioning

A production system should version:

- source data snapshot or data contract
- code commit
- dependency lockfile
- preprocessing configuration
- trained model artefact
- threshold policy
- calibration method
- evaluation report

## Monitoring

Monitoring should include both data and model signals:

- feature distribution drift
- missing-value rates
- failure prevalence
- alert volume
- precision/recall when labels arrive
- probability calibration
- threshold-related maintenance cost
- latency and inference failures

## Promotion and rollback

A new candidate model should not overwrite the previous champion artefact.

Instead, every trained model should receive a unique version and immutable metadata. Promotion should update a separate alias such as `champion` or `production`.

This means rollback only changes the alias back to an earlier validated model; historical artefacts and hyperparameters remain preserved.

## Future tooling

The lightweight JSONL tracker in this repository is sufficient for transparent local evidence. A future production version could migrate run metadata and artefacts to tools such as MLflow, an object store and a model registry without changing the conceptual experiment lifecycle.
