from pathlib import Path

from predictive_maintenance.tracking import append_run, create_run, read_runs


def test_experiment_registry_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    run = create_run(
        experiment="unit-test",
        model="logistic_regression",
        params={"c": 1.0},
        metrics={"average_precision": 0.8},
        notes="test run",
    )

    append_run(run, path)
    loaded = read_runs(path)

    assert len(loaded) == 1
    assert loaded[0].run_id == run.run_id
    assert loaded[0].metrics["average_precision"] == 0.8
