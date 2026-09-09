from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "script"))

from pertussis_global_dynamics.bayes.model2b_data import build_model2b_stan_data
from pertussis_global_dynamics.model2c import _output_suffix

from run_model2c_sensitivity import sensitivity_specs
from test_model2b_bayes import _inputs, _synthetic_context


def test_model2c_output_tag_isolated_from_main_outputs(monkeypatch) -> None:
    monkeypatch.setenv("MODEL2C_CONTACT_MODE", "full")
    monkeypatch.delenv("MODEL2C_OUTPUT_TAG", raising=False)
    assert _output_suffix() == "full"

    monkeypatch.setenv("MODEL2C_OUTPUT_TAG", "contact-upper-7")
    assert _output_suffix() == "contact_upper_7"


def test_model2c_contact_upper_reporting_and_waning_modes_reach_stan_data(monkeypatch) -> None:
    context = _synthetic_context()
    inputs = _inputs()

    monkeypatch.setenv("MODEL2C_CONTACT_SENSITIVITY_UPPER", "7.0")
    monkeypatch.setenv("MODEL2C_REPORTING_PRIOR_MODE", "weak")
    monkeypatch.setenv("MODEL2C_WANING_PRIOR_MODE", "low")
    data_low, metadata_low = build_model2b_stan_data(context, inputs)
    assert data_low["contact_sensitivity_upper"] == 7.0
    assert metadata_low["model2c_contact_sensitivity_upper"] == 7.0
    assert metadata_low["model2c_reporting_prior_sensitivity_mode"] == "weak"
    assert metadata_low["model2c_waning_prior_sensitivity_mode"] == "low"
    assert metadata_low["reporting_prior_method"] == "weak_reporting_prior_sensitivity"

    monkeypatch.setenv("MODEL2C_CONTACT_SENSITIVITY_UPPER", "10.0")
    monkeypatch.setenv("MODEL2C_REPORTING_PRIOR_MODE", "null")
    monkeypatch.setenv("MODEL2C_WANING_PRIOR_MODE", "high")
    data_high, metadata_high = build_model2b_stan_data(context, inputs)
    assert data_high["contact_sensitivity_upper"] == 10.0
    assert metadata_high["model2c_contact_sensitivity_upper"] == 10.0
    assert metadata_high["model2c_reporting_prior_sensitivity_mode"] == "null"
    assert metadata_high["model2c_waning_prior_sensitivity_mode"] == "high"
    assert metadata_high["reporting_prior_method"] == "null_reporting_prior_sensitivity"
    assert data_high["waning_rate_prior_mean"] > data_low["waning_rate_prior_mean"]
    assert np.isfinite(data_high["waning_rate_prior_mean"])


def test_model2c_sensitivity_specs_cover_required_suites_and_tags() -> None:
    specs = sensitivity_specs()
    tags = {spec.tag for spec in specs}
    assert {
        "contact_upper_7",
        "contact_upper_10",
        "reporting_weak",
        "reporting_null",
        "reporting_direct",
        "waning_low",
        "waning_high",
        "objective_unit_weights",
        "age_exclude_split_bins",
        "temporal_fit_2022",
    }.issubset(tags)
    by_tag = {spec.tag: spec for spec in specs}
    assert set(by_tag["contact_upper_7"].countries) == {"AUS", "BRA", "NZL"}
    assert set(by_tag["reporting_weak"].countries) == {"AUS", "BRA", "CHN", "THA"}
    assert len(by_tag["waning_low"].countries) == 9
    assert len(by_tag["objective_unit_weights"].countries) == 9
    assert by_tag["objective_unit_weights"].env["MODEL2C_MONTH_WEIGHT_MODE"] == "unit"
    assert by_tag["age_exclude_split_bins"].env["MODEL2C_AGE_OBSERVATION_MODE"] == "exclude_split_bins"
    assert by_tag["temporal_fit_2022"].env["MODEL2C_FIT_END_YEAR"] == "2022"


def test_model2c_objective_age_and_temporal_modes_reach_stan_data(monkeypatch) -> None:
    context = _synthetic_context()
    inputs = _inputs()

    default_data, default_metadata = build_model2b_stan_data(context, inputs)

    monkeypatch.setenv("MODEL2C_MONTH_WEIGHT_MODE", "unit")
    monkeypatch.setenv("MODEL2C_AGE_OBSERVATION_MODE", "unit_weight")
    unit_data, unit_metadata = build_model2b_stan_data(context, inputs)
    assert set(unit_data["month_obs_weight"]) == {1.0}
    assert all(weight == 1.0 for weight in unit_data["age_obs_weight"])
    assert unit_metadata["model2c_month_weight_mode"] == "unit"
    assert unit_metadata["model2c_age_observation_mode"] == "unit_weight"

    monkeypatch.setenv("MODEL2C_MONTH_WEIGHT_MODE", "peak_balanced")
    monkeypatch.setenv("MODEL2C_AGE_OBSERVATION_MODE", "exclude_split_bins")
    exclude_data, exclude_metadata = build_model2b_stan_data(context, inputs)
    assert exclude_data["N_age_obs"] <= default_data["N_age_obs"]
    assert exclude_metadata["model2c_age_observation_mode"] == "exclude_split_bins"

    monkeypatch.setenv("MODEL2C_AGE_OBSERVATION_MODE", "allocation_weighted")
    monkeypatch.setenv("MODEL2C_FIT_END_YEAR", "2022")
    temporal_data, temporal_metadata = build_model2b_stan_data(context, inputs)
    assert temporal_metadata["fit_end_year"] == 2022
    assert temporal_data["N_month_obs"] < default_data["N_month_obs"]
    assert temporal_data["N_month_ppc_obs"] > default_data["N_month_ppc_obs"]
    assert default_metadata["fit_end_year"] == 2025


def test_model2c_sensitivity_runner_does_not_write_main_suffixes() -> None:
    text = (ROOT / "script/run_model2c_sensitivity.py").read_text(encoding="utf-8")
    assert "MODEL2C_OUTPUT_TAG" in text
    assert "--manifest-only" in text
    assert "model2c_fit_summary_{tag}.csv" in text
    assert "model2c_posterior_draws_{tag}.csv" in text
    assert "if not suffix:" in text
    assert "model2c_fit_summary_full.csv" not in text
