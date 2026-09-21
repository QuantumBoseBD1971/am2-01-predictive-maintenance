"""Exploratory-analysis helpers for the predictive-maintenance dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    positives: int
    negatives: int
    positive_rate: float
    missing_values: int
    duplicate_rows: int


def summarise_dataset(df: pd.DataFrame, target: str) -> DatasetSummary:
    """Return core integrity and class-balance diagnostics."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing.")

    y = df[target].astype(int)
    positives = int(y.sum())
    rows = len(df)

    return DatasetSummary(
        rows=rows,
        columns=int(df.shape[1]),
        positives=positives,
        negatives=rows - positives,
        positive_rate=float(positives / rows) if rows else 0.0,
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
    )


def numeric_feature_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return descriptive statistics for numeric predictors."""
    numeric = df.drop(columns=[target]).select_dtypes(include="number")
    return numeric.describe().T.reset_index(names="feature")
