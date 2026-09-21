# AM2 Evidence Notes

This document is a working evidence map. It will be updated as the project develops.

## Evidence currently demonstrated

### Problem framing
Predictive maintenance is formulated as a supervised binary-classification problem with a rare positive class.

### Data engineering
The repository contains reproducible source acquisition, column normalisation, local persistence, and deterministic loading.

### Model development
A common preprocessing pipeline is used to benchmark both simple and nonlinear classifiers.

### Evaluation
The baseline evaluates precision, recall, F1, ROC-AUC, average precision and Brier score rather than relying on accuracy alone.

### Reproducibility
Python package structure, declared dependencies, tests, fixed random seed and CI are included from the start.

### Responsible AI / limitations
The selected dataset is synthetic. Model results therefore cannot be treated as evidence of safe performance on real machinery. Deployment would require representative operational data, drift monitoring, failure-cost analysis and human maintenance oversight.

## Evidence to add

- EDA outputs and interpretation
- class-imbalance analysis
- cross-validation benchmark table
- calibration plots
- threshold/cost analysis
- explainability artefacts
- error analysis
- model-card style limitations
- final reflection on model choice and trade-offs
