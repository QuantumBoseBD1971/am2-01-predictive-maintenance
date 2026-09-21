"""Lightweight local experiment-tracking utilities.

The repository deliberately keeps the core tracker dependency-free. Each run is
stored as JSON Lines so experiment history can be inspected with standard tools
and migrated to MLflow or another registry later if required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    created_at_utc: str
    experiment: str
    model: str
    params: dict[str, Any]
    metrics: dict[str, float]
    notes: str = ""


def create_run(
    experiment: str,
    model: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    notes: str = "",
) -> ExperimentRun:
    """Create an immutable experiment-run record."""
    return ExperimentRun(
        run_id=str(uuid4()),
        created_at_utc=datetime.now(UTC).isoformat(),
        experiment=experiment,
        model=model,
        params=params,
        metrics=metrics,
        notes=notes,
    )


def append_run(run: ExperimentRun, path: Path) -> Path:
    """Append one run to a JSONL registry and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(run), sort_keys=True) + "\n")
    return path


def read_runs(path: Path) -> list[ExperimentRun]:
    """Load all runs from a JSONL registry."""
    if not path.exists():
        return []

    runs: list[ExperimentRun] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                runs.append(ExperimentRun(**json.loads(line)))
    return runs
