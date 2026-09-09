from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from script import summarize_model2c_recovery_diagnostics as diagnostics
from pertussis_global_dynamics.bayes.model2c_recovery import RECOVERY_PARAMETERS


def _write_summary(path: Path, *, rhat: float = 1.005, ess: float = 500.0) -> None:
    frame = pd.DataFrame(
        {"R_hat": [rhat] * len(RECOVERY_PARAMETERS), "ESS_bulk": [ess] * len(RECOVERY_PARAMETERS)},
        index=RECOVERY_PARAMETERS,
    )
    frame.to_csv(path)


def _write_marker(path: Path, run_id: str) -> None:
    path.write_text(json.dumps({"run_id": run_id, "status": "fit_complete"}), encoding="utf-8")


def test_completion_is_separate_from_strict_convergence_gate(tmp_path: Path) -> None:
    experiment = tmp_path / "tag"
    summaries = experiment / "fit_summaries"
    summaries.mkdir(parents=True)
    rows = [
        {"run_id": "AUS_reporting_only_r001", "iso3": "AUS", "scenario_id": "reporting_only", "replicate": 1},
        {"run_id": "AUS_reporting_only_r002", "iso3": "AUS", "scenario_id": "reporting_only", "replicate": 2},
        {"run_id": "BRA_null_both_r001", "iso3": "BRA", "scenario_id": "null_both", "replicate": 1},
    ]
    pd.DataFrame(rows).to_csv(experiment / "manifest.csv", index=False)

    passing = rows[0]["run_id"]
    _write_summary(summaries / f"{passing}.csv")
    _write_marker(summaries / f"{passing}.complete.json", passing)

    # R-hat equal to 1.01 fails because the registered gate is strict (< 1.01).
    failing = rows[1]["run_id"]
    _write_summary(summaries / f"{failing}.csv", rhat=1.01)
    _write_marker(summaries / f"{failing}.complete.json", failing)

    # A summary without an atomic completion marker is not read as a finished fit.
    unmarked = rows[2]["run_id"]
    _write_summary(summaries / f"{unmarked}.csv", rhat=1.0, ess=900.0)

    by_fit, grouped = diagnostics.summarize_experiment(experiment)
    indexed = by_fit.set_index("run_id")
    assert indexed.loc[passing, "diagnostic_gate_pass"]
    assert indexed.loc[failing, "fit_complete"]
    assert not indexed.loc[failing, "diagnostic_gate_pass"]
    assert not indexed.loc[unmarked, "fit_complete"]
    assert not indexed.loc[unmarked, "diagnostics_evaluable"]

    overall = grouped.loc[grouped["aggregation_level"].eq("overall")].iloc[0]
    assert overall["n_expected_fits"] == 3
    assert overall["n_fit_complete"] == 2
    assert overall["n_diagnostic_gate_pass"] == 1


def test_missing_core_parameter_makes_fit_unevaluable_and_report_is_written(tmp_path: Path) -> None:
    experiment = tmp_path / "tag"
    summaries = experiment / "fit_summaries"
    summaries.mkdir(parents=True)
    run_id = "JPN_multi_mechanism_r001"
    pd.DataFrame(
        [{"run_id": run_id, "iso3": "JPN", "scenario_id": "multi_mechanism", "replicate": 1}]
    ).to_csv(experiment / "manifest.csv", index=False)
    frame = pd.DataFrame(
        {"R_hat": [1.0] * (len(RECOVERY_PARAMETERS) - 1), "ESS_bulk": [800.0] * (len(RECOVERY_PARAMETERS) - 1)},
        index=RECOVERY_PARAMETERS[:-1],
    )
    frame.to_csv(summaries / f"{run_id}.csv")
    _write_marker(summaries / f"{run_id}.complete.json", run_id)

    by_fit, grouped = diagnostics.summarize_experiment(experiment)
    assert by_fit.loc[0, "n_core_parameters_present"] == 26
    assert not by_fit.loc[0, "diagnostics_evaluable"]
    assert "init_natural_recent_fraction" in by_fit.loc[0, "artifact_error"]

    diagnostics.write_outputs(experiment, by_fit, grouped)
    report = (experiment / diagnostics.REPORT_OUTPUT).read_text(encoding="utf-8")
    assert "not evidence of convergence" in report
    assert "all 27 core parameters" in report
    assert (experiment / diagnostics.FIT_OUTPUT).exists()
    assert (experiment / diagnostics.GROUP_OUTPUT).exists()
