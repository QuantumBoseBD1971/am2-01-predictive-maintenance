"""Generate reproducible EDA tables and figures."""

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from predictive_maintenance.analysis import numeric_feature_summary, summarise_dataset
from predictive_maintenance.config import TARGET_COLUMN
from predictive_maintenance.data import load_dataset

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def main() -> None:
    df = load_dataset()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = summarise_dataset(df, TARGET_COLUMN)
    pd.DataFrame([asdict(summary)]).to_csv(
        TABLES_DIR / "dataset_summary.csv",
        index=False,
    )

    numeric_feature_summary(df, TARGET_COLUMN).to_csv(
        TABLES_DIR / "numeric_feature_summary.csv",
        index=False,
    )

    counts = df[TARGET_COLUMN].astype(int).value_counts().sort_index()
    ax = counts.plot(kind="bar", title="Machine failure class distribution")
    ax.set_xlabel("Machine failure")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()

    print(pd.DataFrame([asdict(summary)]).to_string(index=False))


if __name__ == "__main__":
    main()
