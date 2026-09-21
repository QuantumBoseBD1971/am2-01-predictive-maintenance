"""Baseline model construction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    brier_score: float


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing from inferred numeric/categorical columns."""
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def candidate_models() -> dict[str, object]:
    """Return the initial model benchmark."""
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def make_pipeline(X: pd.DataFrame, estimator: object) -> Pipeline:
    """Create one end-to-end preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", estimator),
        ]
    )


def evaluate_binary_classifier(name: str, model: Pipeline, X_test, y_test) -> EvaluationResult:
    """Evaluate a fitted binary classifier at the default decision threshold."""
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    return EvaluationResult(
        model=name,
        precision=precision_score(y_test, prediction, zero_division=0),
        recall=recall_score(y_test, prediction, zero_division=0),
        f1=f1_score(y_test, prediction, zero_division=0),
        roc_auc=roc_auc_score(y_test, probability),
        average_precision=average_precision_score(y_test, probability),
        brier_score=brier_score_loss(y_test, probability),
    )
