from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "script"))

import build_model2c_country_sampling_diagnostics as diagnostics


def _scenario(iso3: str, scenario_id: str, full: float, counterfactual: float) -> dict[str, object]:
    return {
        "iso3": iso3,
        "country": iso3,
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "outcome_target": "reported_cases",
        "full_total_median": full,
        "scenario_total_median": counterfactual,
        "scenario_to_full_ratio_median": counterfactual / full,
    }


def test_nine_vs_ten_panel_comparison_is_descriptive_sum(monkeypatch) -> None:
    primary = pd.DataFrame([
        _scenario("AUS", "low_vaccine_waning", 100.0, 60.0),
        _scenario("JPN", "low_vaccine_waning", 300.0, 240.0),
    ])
    singapore = pd.DataFrame([_scenario("SGP", "low_vaccine_waning", 100.0, 50.0)])
    mapping = {
        "model2c_counterfactual_scenario_summary.csv": primary,
        f"model2c_counterfactual_scenario_summary_{diagnostics.SGP_TAG}.csv": singapore,
    }
    monkeypatch.setattr(diagnostics, "_read", lambda name: mapping[name])
    result = diagnostics.build_panel_comparison().iloc[0]
    assert result["primary_9_descriptive_reduction_percent"] == pytest.approx(25.0)
    assert result["expanded_10_descriptive_reduction_percent"] == pytest.approx(30.0)
    assert result["change_after_singapore_percentage_points"] == pytest.approx(5.0)


def test_japan_long_refit_comparison_reports_percentage_point_change(monkeypatch) -> None:
    primary = pd.DataFrame([_scenario("JPN", "low_vaccine_waning", 100.0, 70.0)])
    long = pd.DataFrame([_scenario("JPN", "low_vaccine_waning", 110.0, 66.0)])
    mapping = {
        "model2c_counterfactual_scenario_summary.csv": primary,
        f"model2c_counterfactual_scenario_summary_{diagnostics.JPN_TAG}.csv": long,
    }
    monkeypatch.setattr(diagnostics, "_read", lambda name: mapping[name])
    result = diagnostics.build_japan_scenario_comparison().iloc[0]
    assert round(result["reduction_percent_primary"], 8) == 30.0
    assert round(result["reduction_percent_long_refit"], 8) == 40.0
    assert round(result["reduction_change_percentage_points"], 8) == 10.0
    assert round(result["full_total_change_percent"], 8) == 10.0


def test_low_fit_exclusion_uses_prespecified_rhat_and_ess_gate(monkeypatch) -> None:
    fit = pd.DataFrame([
        {"iso3": "AUS", "max_rhat": 1.005, "min_ess_bulk": 600.0},
        {"iso3": "BRA", "max_rhat": 1.012, "min_ess_bulk": 500.0},
        {"iso3": "JPN", "max_rhat": 1.005, "min_ess_bulk": 350.0},
    ])
    scenarios = pd.DataFrame([
        _scenario("AUS", "low_vaccine_waning", 100.0, 50.0),
        _scenario("BRA", "low_vaccine_waning", 100.0, 90.0),
        _scenario("JPN", "low_vaccine_waning", 100.0, 90.0),
    ])
    mapping = {
        "model2c_fit_summary.csv": fit,
        "model2c_counterfactual_scenario_summary.csv": scenarios,
    }
    monkeypatch.setattr(diagnostics, "_read", lambda name: mapping[name])
    result = diagnostics.build_low_fit_exclusion_comparison().iloc[0]
    assert result["excluded_iso3"] == "BRA;JPN"
    assert result["n_retained_countries"] == 1
    assert result["all_9_descriptive_reduction_percent"] == pytest.approx(23.3333333333)
    assert result["low_fit_excluded_descriptive_reduction_percent"] == pytest.approx(50.0)


def test_ebfmi_uses_squared_energy_transitions_over_variance() -> None:
    energy = np.array([1.0, 2.0, 4.0, 3.0])
    expected = np.mean(np.diff(energy) ** 2) / np.var(energy, ddof=1)
    assert diagnostics._ebfmi(energy) == pytest.approx(expected)
