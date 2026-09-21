# Final Reflection

## What the project demonstrates

This project began as a conventional binary-classification benchmark and was deliberately extended into a fuller AI-engineering workflow.

The main technical progression was:

1. reproducible public-data acquisition
2. EDA and data-quality checks
3. baseline and nonlinear model comparison
4. stratified cross-validation
5. probability calibration
6. threshold and cost analysis
7. model-agnostic explainability
8. robustness testing
9. experiment tracking and MLOps design

## Key learning

The strongest model is not automatically the model with the highest raw accuracy.

For an imbalanced predictive-maintenance problem, the engineering decision depends on several distinct questions:

- can the model rank failures effectively?
- are its probabilities calibrated?
- what recall/precision trade-off is acceptable?
- what are the relative costs of false negatives and false positives?
- are conclusions robust to different data partitions?
- can the result be reproduced and monitored after deployment?

## What I would change with real data

With live industrial data I would add:

- time-aware validation rather than only random stratified splits
- machine/site grouping to prevent leakage between related assets
- explicit maintenance-event windows
- delayed-label handling
- domain-approved failure costs
- drift baselines
- prospective shadow deployment before any operational use

## Why this is useful AM2 evidence

The repository provides concrete evidence that I can move beyond training a model and reason about the wider AI-engineering lifecycle: data quality, evaluation, decision thresholds, explainability, reproducibility, deployment, monitoring and responsible use.
