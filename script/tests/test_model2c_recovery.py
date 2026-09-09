from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from script import run_model2c_recovery as recovery_runner

from pertussis_global_dynamics.bayes.model2c_recovery import (
    RECOVERY_PARAMETERS,
    apply_recovery_scenario,
    assert_isolated_recovery_dir,
    negative_binomial_2_rng,
    recovery_aggregate,
    recovery_scenarios,
    summarise_recovery_run,
    synthetic_stan_data,
)


ROOT = Path(__file__).resolve().parents[2]


def test_scenarios_neutralize_reporting_and_transmission_aliases() -> None:
    draw = pd.Series({name: 0.2 for name in RECOVERY_PARAMETERS})
    draw["post_reporting_peak_multiplier"] = 1.2
    draw["post_2023_transmission_shift_multiplier"] = 1.2
    draw["excess_epidemic_pressure_scale"] = 0.2
    draw["outbreak_ignition_pulse_gain"] = 0.2
    null_scenario = next(item for item in recovery_scenarios() if item.scenario_id == "null_both")
    truth = apply_recovery_scenario(draw, null_scenario)
    assert truth["log_post_reporting_multiplier"] == 0.0
    assert truth["post_reporting_peak_multiplier"] == 1.0
    assert truth["log_post_transmission_multiplier"] == 0.0
    assert truth["post_2023_transmission_shift_multiplier"] == 1.0
    assert truth["excess_epidemic_pressure_scale"] == 0.0
    assert truth["outbreak_ignition_pulse_gain"] == 0.0


def test_nb2_rng_is_reproducible_and_nonnegative() -> None:
    mean = np.array([0.0, 2.0, 20.0])
    first = negative_binomial_2_rng(mean, 10.0, np.random.default_rng(7))
    second = negative_binomial_2_rng(mean, 10.0, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0)
    assert first[0] == 0


def test_synthetic_data_preserves_dimensions_with_real_prepared_input() -> None:
    data = json.loads((ROOT / "outputs/model2c_stan_data_AUS_full.json").read_text(encoding="utf-8"))
    draw = pd.read_csv(ROOT / "outputs/model2c_posterior_draws.csv", nrows=1).iloc[0]
    synthetic, expected = synthetic_stan_data(data, draw, seed=77)
    assert len(synthetic["y_month"]) == data["N_month_obs"]
    assert len(synthetic["y_month_ppc"]) == data["N_month_ppc_obs"]
    assert len(synthetic["y_age"]) == data["N_age_obs"]
    assert len(expected["age_mean"]) == data["N_age_obs"]
    assert all(isinstance(value, int) and value >= 0 for value in synthetic["y_age"])


def test_recovery_summary_calculates_coverage_and_discrimination() -> None:
    truth = pd.Series({name: 0.2 for name in RECOVERY_PARAMETERS})
    posterior = pd.DataFrame({name: np.linspace(0.1, 0.3, 200) for name in RECOVERY_PARAMETERS})
    parameter, mechanism = summarise_recovery_run(
        truth,
        posterior,
        run_id="AUS_multi_mechanism_r001",
        iso3="AUS",
        scenario_id="multi_mechanism",
        replicate=1,
    )
    assert parameter["covered_95"].all()
    assert mechanism["truth_active"].all()
    assert mechanism["discrimination_correct"].all()
    aggregate = recovery_aggregate(parameter, mechanism)
    assert set(aggregate["metric_family"]) == {"parameter_recovery", "mechanism_discrimination"}


def test_recovery_directory_must_be_tagged_child(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    assert_isolated_recovery_dir(outputs / "model2c_recovery" / "tagged", outputs)
    with pytest.raises(ValueError):
        assert_isolated_recovery_dir(outputs / "model2c_fit_summary.csv", outputs)


def test_run_worker_default_is_backward_compatible() -> None:
    args = recovery_runner.parse_args(["--countries", "AUS", "--replicates", "1", "--prepare-only"])
    assert args.run_workers == 1


def test_markdown_table_does_not_require_optional_tabulate() -> None:
    frame = pd.DataFrame([{"metric": "coverage|95", "value": 0.875, "missing": np.nan}])
    rendered = recovery_runner._markdown_table(frame)
    assert "coverage\\|95" in rendered
    assert "0.875" in rendered
    assert "NA" in rendered


def test_parallel_scheduler_uses_requested_workers_without_stan(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, int] = {}

    class FakeExecutor:
        def __init__(self, max_workers: int):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, payload):
            future: Future = Future()
            try:
                future.set_result(function(payload))
            except Exception as exc:  # pragma: no cover - mirrors concurrent future behaviour
                future.set_exception(exc)
            return future

    monkeypatch.setattr(recovery_runner, "ProcessPoolExecutor", FakeExecutor)

    def fake_worker(payload: dict) -> dict[str, str]:
        return {"run_id": payload["row"]["run_id"], "status": "fit_complete", "error": ""}

    payloads = [{"row": {"run_id": f"run_{index}"}} for index in range(3)]
    outcomes = list(recovery_runner._execute_fit_tasks(payloads, run_workers=3, worker=fake_worker))
    assert observed["max_workers"] == 3
    assert {result["run_id"] for _, result, error in outcomes if result is not None and error is None} == {
        "run_0",
        "run_1",
        "run_2",
    }


def test_atomic_manifest_update_and_sequential_failure_recovery(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    frame = pd.DataFrame([{"run_id": "ok", "status": "fit_running"}])
    recovery_runner._write_manifest_atomic(frame, manifest_path)
    assert pd.read_csv(manifest_path).to_dict("records") == frame.to_dict("records")
    assert not manifest_path.with_suffix(".csv.tmp").exists()

    def fake_worker(payload: dict) -> dict[str, str]:
        if payload["row"]["run_id"] == "failed":
            raise RuntimeError("synthetic failure")
        return {"run_id": "ok", "status": "fit_complete", "error": ""}

    payloads = [{"row": {"run_id": "ok"}}, {"row": {"run_id": "failed"}}]
    outcomes = list(recovery_runner._execute_fit_tasks(payloads, run_workers=1, worker=fake_worker))
    assert outcomes[0][1]["status"] == "fit_complete"
    assert isinstance(outcomes[1][2], RuntimeError)
