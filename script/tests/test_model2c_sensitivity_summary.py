from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "script"))

from run_model2c_sensitivity import SensitivitySpec
from summarize_model2c_sensitivity import summarise, write_outputs


def _spec(tag: str = "contact_upper_7", countries: tuple[str, ...] = ("AUS", "BRA")) -> SensitivitySpec:
    return SensitivitySpec(tag, "contact", countries, {}, "synthetic sensitivity")


def _write_primary(outputs: Path) -> None:
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "max_rhat": [1.005, 1.006], "min_ess_bulk": [600, 500], "posterior_draws": [2000, 2000],
        }
    ).to_csv(outputs / "model2c_fit_summary_full.csv", index=False)
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "scenario_id": ["no_waning", "no_waning"], "period_start": [2023, 2023],
            "period_end": [2025, 2025], "outcome_target": ["reported", "reported"],
            "scenario_to_full_ratio_median": [0.5, 0.8], "full_minus_scenario_median": [100.0, 50.0],
        }
    ).to_csv(outputs / "model2c_counterfactual_scenario_summary_full.csv", index=False)
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "post_reporting_multiplier_median": [1.0, 2.0], "post_reporting_peak_multiplier_median": [1.1, 2.1],
            "post_transmission_multiplier_median": [1.0, 1.0], "excess_epidemic_pressure_scale_median": [1.0, 1.0],
            "outbreak_ignition_pulse_gain_median": [0.1, 0.1], "post_reporting_trend_per_year_median": [0.0, 0.0],
        }
    ).to_csv(outputs / "model2c_observation_process_audit_full.csv", index=False)


def test_summariser_compares_complete_tag_and_classifies_change(tmp_path: Path) -> None:
    _write_primary(tmp_path)
    tag = "contact_upper_7"
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "max_rhat": [1.004, 1.008], "min_ess_bulk": [700, 450], "posterior_draws": [2000, 2000],
        }
    ).to_csv(tmp_path / f"model2c_fit_summary_{tag}.csv", index=False)
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "scenario_id": ["no_waning", "no_waning"], "period_start": [2023, 2023],
            "period_end": [2025, 2025], "outcome_target": ["reported", "reported"],
            "scenario_to_full_ratio_median": [0.75, 0.808], "full_minus_scenario_median": [50.0, 49.5],
        }
    ).to_csv(tmp_path / f"model2c_counterfactual_scenario_summary_{tag}.csv", index=False)
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "post_reporting_multiplier_median": [1.1, 2.0], "post_reporting_peak_multiplier_median": [1.1, 2.1],
            "post_transmission_multiplier_median": [1.0, 1.0], "excess_epidemic_pressure_scale_median": [1.0, 1.0],
            "outbreak_ignition_pulse_gain_median": [0.1, 0.1], "post_reporting_trend_per_year_median": [0.0, 0.0],
        }
    ).to_csv(tmp_path / f"model2c_observation_process_audit_{tag}.csv", index=False)

    status, detail, summary = summarise(tmp_path, [_spec()])
    assert status.loc[0, "run_status"] == "complete"
    assert status.loc[0, "diagnostic_status"] == "pass"
    changed = detail.loc[
        (detail["iso3"] == "AUS") & (detail["metric"] == "scenario_to_full_ratio_median")
    ].iloc[0]
    assert changed["relative_change_pct"] == 50.0
    assert changed["direction"] == "increased"
    assert changed["magnitude_class"] == "large_ge_50pct"
    assert changed["prior_dependence_class"] == "strong"
    aggregate = summary.loc[summary["metric"] == "scenario_to_full_ratio_median"].iloc[0]
    assert aggregate["max_absolute_relative_change_pct"] == 50.0
    assert aggregate["prior_dependence_class"] == "strong"

    write_outputs(tmp_path, status, detail, summary)
    assert (tmp_path / "model2c_sensitivity_comparison.csv").exists()
    assert "Missing or partial runs" in (tmp_path / "model2c_sensitivity_comparison.md").read_text()


def test_summariser_marks_partial_outputs_without_claiming_completion(tmp_path: Path) -> None:
    _write_primary(tmp_path)
    tag = "contact_upper_7"
    pd.DataFrame(
        {"iso3": ["AUS"], "country": ["Australia"], "max_rhat": [1.02], "min_ess_bulk": [100], "posterior_draws": [20]}
    ).to_csv(tmp_path / f"model2c_fit_summary_{tag}.csv", index=False)
    (tmp_path / f"model2c_counterfactual_scenario_summary_{tag}.csv").write_text('iso3,country\n"AUS', encoding="utf-8")

    status, detail, summary = summarise(tmp_path, [_spec()])
    assert status.loc[0, "run_status"] == "partial_or_running"
    assert status.loc[0, "diagnostic_status"] == "fail_or_unavailable"
    assert detail.empty
    assert summary.empty


def test_summariser_distinguishes_complete_fit_from_pending_replay(tmp_path: Path) -> None:
    _write_primary(tmp_path)
    tag = "contact_upper_7"
    pd.DataFrame(
        {
            "iso3": ["AUS", "BRA"], "country": ["Australia", "Brazil"],
            "max_rhat": [1.005, 1.006], "min_ess_bulk": [600, 500], "posterior_draws": [2000, 2000],
        }
    ).to_csv(tmp_path / f"model2c_fit_summary_{tag}.csv", index=False)

    status, detail, summary = summarise(tmp_path, [_spec()])
    assert status.loc[0, "run_status"] == "fit_complete_replay_pending"
    assert detail.empty
    assert summary.empty


def test_summariser_tolerates_fit_schema_without_diagnostic_columns(tmp_path: Path) -> None:
    _write_primary(tmp_path)
    tag = "contact_upper_7"
    pd.DataFrame({"iso3": ["AUS"], "country": ["Australia"]}).to_csv(
        tmp_path / f"model2c_fit_summary_{tag}.csv", index=False
    )

    status, _, _ = summarise(tmp_path, [_spec(countries=("AUS",))])
    assert status.loc[0, "run_status"] == "fit_complete_replay_pending"
    assert status.loc[0, "diagnostic_status"] == "fail_or_unavailable"
