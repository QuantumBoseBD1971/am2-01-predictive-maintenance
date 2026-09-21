"""Data loading and preprocessing helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from predictive_maintenance.config import DATA_DIR, RAW_DATA_PATH


def _snake_case(name: str) -> str:
    """Convert source column names to stable snake_case names."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return name.strip("_").lower()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case column names."""
    out = df.copy()
    out.columns = [_snake_case(c) for c in out.columns]
    return out


def download_dataset(destination: Path = RAW_DATA_PATH) -> Path:
    """Download AI4I 2020 from UCI and persist a local CSV.

    The UCI dataset id is 601.
    """
    dataset = fetch_ucirepo(id=601)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = pd.concat([features, targets], axis=1)
    df = normalise_columns(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the local dataset, downloading it first if necessary."""
    if not path.exists():
        download_dataset(path)
    return normalise_columns(pd.read_csv(path))
