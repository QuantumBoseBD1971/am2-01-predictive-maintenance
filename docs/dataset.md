# Dataset

## Selected source

**AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository, dataset id 601.

DOI: **10.24432/C5HS5C**

The source describes the dataset as synthetic data intended to reflect predictive-maintenance data encountered in industry.

## Why this dataset

This dataset is suitable for the first AM2 project because it supports a clear binary-classification problem while still exposing realistic engineering issues:

- mixed numeric and categorical features
- a rare positive class
- operational measurements with physical units
- a need to compare discrimination, calibration and threshold trade-offs
- a clear distinction between model development and real-world deployment evidence

## Target

`machine_failure`

The first benchmark predicts whether a machine-failure event is present.

## Leakage and identifier policy

`uid` and `product_id` are excluded from the first benchmark because they primarily identify observations/products rather than represent operational measurements.

Failure-mode indicator columns, if present in a local representation of the dataset, must **not** be used as predictors of `machine_failure` because they directly encode the failure outcome and would create target leakage.

## Reproducibility

Run:

```bash
python scripts/download_data.py
```

The dataset is written to `data/ai4i2020.csv`, which should remain untracked.
