from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pertussis_global_dynamics.bayes.model2b_data import build_model2b_stan_data
from pertussis_global_dynamics.bayes.model2c_scenarios import (
    MODEL2C_OUTPUT_VERSION,
    MODEL2C_SCENARIOS,
    MODEL2C_SUPPLEMENT_CONTEXT_AUDIT_SCENARIOS,
    _predict_monthly,
    _scenario_prediction_frame,
    model2c_counterfactual_outputs,
    model2c_posterior_predictive,
)

from test_model2b_bayes import _draws, _inputs, _synthetic_context


ROOT = Path(__file__).resolve().parents[2]


def _scenario_data() -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    data, metadata = build_model2b_stan_data(_synthetic_context(), _inputs())
    data = dict(data)
    year = np.asarray(data["year"], dtype=int)
    contact = np.asarray(data["contact_intensity"], dtype=float)
    contact[(year >= 2020) & (year <= 2023)] = 70.0
    data["contact_intensity"] = contact.tolist()
    data["booster_reach_modifier"] = np.repeat(0.55, int(data["N_age"])).tolist()
    data["vaccine_waning_age_modifier"] = np.linspace(1.0, 1.8, int(data["N_age"])).tolist()
    data["monthly_births_ref"] = (0.55 * np.asarray(data["monthly_births"], dtype=float)).tolist()
    data["age_population_ref"] = (0.82 * np.asarray(data["age_population"], dtype=float)).tolist()
    pathogen = np.asarray(data["pathogen_pressure_prior"], dtype=float)
    pathogen[year >= 2023] = 1.45
    data["pathogen_pressure_prior"] = pathogen.tolist()

    draws = _draws().copy()
    draws["log_beta"] = math.log(0.000035)
    draws["contact_sensitivity"] = [0.80, 0.90]
    draws["waning_rate"] = [0.16, 0.18]
    draws["log_post_transmission_multiplier"] = math.log(1.30)
    draws["log_post_reporting_multiplier"] = math.log(1.35)
    draws["natural_boosting_efficiency"] = [0.70, 0.75]
    draws["pathogen_pressure_scale"] = [1.30, 1.40]
    draws["outbreak_seeding_gain"] = [0.060, 0.070]
    draws["rebound_release_gain"] = [1.60, 1.70]
    draws["rebound_asymmetry"] = [0.15, 0.20]
    draws["secondary_rebound_delay"] = [8.0, 9.0]
    draws["secondary_rebound_width"] = [5.0, 6.0]
    draws["secondary_rebound_release_gain"] = [0.60, 0.70]
    draws["post_reporting_peak_gain"] = [0.45, 0.55]
    draws["post_reporting_trend"] = [0.08, 0.10]
    metadata = {**metadata, "model2c_output_version": MODEL2C_OUTPUT_VERSION}
    return data, metadata, draws


def _total(prediction: pd.DataFrame, outcome: str = "reported_cases_predicted") -> float:
    return float(prediction.loc[prediction["year"].between(2023, 2025), outcome].sum())


def test_model2c_generates_required_scenarios_and_outputs() -> None:
    data, metadata, draws = _scenario_data()

    monthly, summary = model2c_counterfactual_outputs(
        data,
        metadata,
        draws,
        max_draws=2,
        start_year=2023,
        end_year=2025,
    )

    assert set(monthly["scenario_id"]) == set(MODEL2C_SCENARIOS)
    assert set(summary["scenario_id"]) == set(MODEL2C_SCENARIOS)
    assert {
        "true_infections",
        "symptomatic_infections",
        "reported_cases",
        "true_infections_under5",
        "symptomatic_infections_under5",
        "reported_cases_under5",
        "true_infections_school_age",
        "symptomatic_infections_school_age",
        "reported_cases_school_age",
        "true_infections_adult",
        "symptomatic_infections_adult",
        "reported_cases_adult",
        "effective_susceptible_start_2023",
        "effective_susceptible_start_2025",
        "effective_susceptible_under5_start_2025",
        "effective_susceptible_school_age_start_2025",
        "effective_susceptible_adult_start_2025",
        "natural_immunity_under5_start_2025",
        "natural_immunity_start_2025",
        "natural_immunity_school_age_start_2025",
        "natural_immunity_adult_start_2025",
        "routine_delay_debt_start_2025",
    }.issubset(set(summary["outcome_target"]))
    assert monthly["year"].min() == 2023
    assert monthly["year"].max() == 2025
    case_summary = summary.loc[
        summary["outcome_target"].isin({"true_infections", "symptomatic_infections", "reported_cases"})
    ]
    state_2023 = summary.loc[summary["outcome_target"].eq("effective_susceptible_start_2023")]
    state_2025 = summary.loc[summary["outcome_target"].eq("effective_susceptible_start_2025")]
    assert case_summary["period_start"].eq(2023).all()
    assert case_summary["period_end"].eq(2025).all()
    assert state_2023["period_start"].eq(2023).all()
    assert state_2023["period_end"].eq(2023).all()
    assert state_2025["period_start"].eq(2025).all()
    assert state_2025["period_end"].eq(2025).all()
    assert monthly["n_posterior_draws"].min() == 2
    assert summary["n_posterior_draws"].min() == 2
    assert monthly["model2c_output_version"].eq(MODEL2C_OUTPUT_VERSION).all()
    assert summary["model2c_output_version"].eq(MODEL2C_OUTPUT_VERSION).all()
    assert not set(MODEL2C_SUPPLEMENT_CONTEXT_AUDIT_SCENARIOS).intersection(monthly["scenario_id"])

    required_monthly = {
        "iso3",
        "country",
        "scenario_id",
        "scenario_label",
        "scenario_family",
        "neutralized_mechanisms",
        "year",
        "month",
        "date",
        "fit_role",
        "observed_cases",
        "true_infections_median",
        "symptomatic_infections_median",
        "reported_cases_median",
        "true_infections_under5_median",
        "symptomatic_infections_under5_median",
        "reported_cases_under5_median",
        "true_infections_school_age_median",
        "symptomatic_infections_school_age_median",
        "reported_cases_school_age_median",
        "true_infections_adult_median",
        "symptomatic_infections_adult_median",
        "reported_cases_adult_median",
        "effective_susceptible_median",
        "effective_susceptible_under5_median",
        "effective_susceptible_school_age_median",
        "effective_susceptible_adult_median",
        "natural_immunity_median",
        "waned_natural_immunity_median",
        "natural_immunity_under5_median",
        "waned_natural_immunity_under5_median",
        "natural_immunity_school_age_median",
        "waned_natural_immunity_school_age_median",
        "natural_immunity_adult_median",
        "waned_natural_immunity_adult_median",
        "exposure_boosting_events_median",
        "partial_vaccine_immunity_median",
        "natural_exposure_recharge_median",
        "routine_immunization_shortfall_median",
        "routine_delay_debt_median",
    }
    required_summary = {
        "period_start",
        "period_end",
        "outcome_target",
        "full_total_median",
        "scenario_total_median",
        "full_minus_scenario_median",
        "scenario_to_full_ratio_median",
        "full_peak_median",
        "scenario_peak_median",
        "peak_ratio_median",
        "full_peak_month",
        "scenario_peak_month",
        "scenario_interpretation_note",
    }
    assert required_monthly.issubset(monthly.columns)
    assert required_summary.issubset(summary.columns)


def test_model2c_posterior_predictive_fit_role_uses_observed_month_indices() -> None:
    data, metadata, draws = _scenario_data()
    data = dict(data)
    data["month_obs_idx"] = [61, 62]
    data["month_ppc_obs_idx"] = [133]
    data["y_month"] = [10, 20]
    data["y_month_ppc"] = [5]
    data["N_month_obs"] = 2
    data["N_month_ppc_obs"] = 1

    ppc = model2c_posterior_predictive(data, metadata, draws.head(1))
    by_index = ppc.reset_index(drop=True)

    assert by_index.loc[60, "fit_role"] == "fit_likelihood"
    assert by_index.loc[62, "fit_role"] == "state_history_no_monthly_observation"
    assert by_index.loc[132, "fit_role"] == "posterior_predictive_check"
    assert by_index.loc[133, "fit_role"] == "posterior_predictive_unobserved_month"


def test_model2c_contact_sensitivity_upper_is_data_driven() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0].copy()
    draw["contact_sensitivity"] = 7.0

    data_main = {**data, "contact_sensitivity_upper": 5.0}
    data_relaxed = {**data, "contact_sensitivity_upper": 10.0}
    main = _predict_monthly(data_main, draw, scenario_id="full_model", totals_only=True)
    relaxed = _predict_monthly(data_relaxed, draw, scenario_id="full_model", totals_only=True)

    assert not math.isclose(main["true_infections"], relaxed["true_infections"])


def test_model2c_contact_and_missed_boosting_are_separate_lambda_switches() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0]
    full = _predict_monthly(data, draw, scenario_id="full_model")
    no_contact = _predict_monthly(data, draw, scenario_id="no_contact_suppression")
    no_missed_boosting = _predict_monthly(data, draw, scenario_id="no_missed_exposure_boosting")
    restored_joint = _predict_monthly(data, draw, scenario_id="restored_circulation_boosting_path")

    assert not math.isclose(_total(full, "true_infections_predicted"), _total(no_contact, "true_infections_predicted"))
    assert not math.isclose(
        _total(full, "true_infections_predicted"),
        _total(no_missed_boosting, "true_infections_predicted"),
    )
    assert not np.allclose(
        no_contact["reported_cases_predicted"].to_numpy(),
        no_missed_boosting["reported_cases_predicted"].to_numpy(),
    )
    assert not np.allclose(
        restored_joint["reported_cases_predicted"].to_numpy(),
        no_missed_boosting["reported_cases_predicted"].to_numpy(),
    )
    assert set(MODEL2C_SCENARIOS["restored_circulation_boosting_path"]["neutralized_mechanisms"]) == {
        "pre_2023_contact_suppression",
        "pre_2023_missed_exposure_boosting",
    }
    assert set(MODEL2C_SCENARIOS["no_missed_exposure_boosting"]["neutralized_mechanisms"]) == {
        "pre_2023_missed_exposure_boosting",
    }

    source = inspect.getsource(_predict_monthly)
    assert "NATURAL_BOOST_REPAIR_START_YEAR <= year_m <= NATURAL_BOOST_REPAIR_END_YEAR" in source
    assert '"pre_2023_contact_suppression" in neutralized and natural_boosting_repair_window' in source
    assert '"pre_2023_missed_exposure_boosting" in neutralized and natural_boosting_repair_window' in source
    assert "pre_pandemic_boosting_reference" in source
    assert "normal_boosting_lam" in source
    assert "silent_exposure_floor_monthly" in source
    assert "natural_exposure_contact_sensitivity" in source
    assert "observed_boosting_contact_multiplier" in source
    assert "reported_exposure_floor" in source
    assert "background_exposure_floor" in source
    assert "natural_exposure_recharge_predicted" in source
    assert "total_natural_immunity_under5" in source
    assert "total_natural_immunity_school_age" in source
    assert "total_natural_immunity_adult" in source
    assert "reported_cases_school_age_predicted" in source
    assert "reported_cases_adult_predicted" in source
    assert "recharge_vaccine_recent" in source
    assert "recharge_partial_recent" in source


def test_model2c_routine_replay_uses_dtp1_and_partial_vaccine_state() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0]
    full = _predict_monthly(data, draw, scenario_id="full_model")
    no_routine = _predict_monthly(data, draw, scenario_id="no_routine_immunization_disruption")

    full_partial = float(full.loc[(full["year"].eq(2023)) & (full["month"].eq(1)), "partial_vaccine_immunity_predicted"].iloc[0])
    restored_partial = float(
        no_routine.loc[
            (no_routine["year"].eq(2023)) & (no_routine["month"].eq(1)),
            "partial_vaccine_immunity_predicted",
        ].iloc[0]
    )
    assert not math.isclose(full_partial, restored_partial)
    assert full.loc[full["year"].between(2020, 2024), "routine_immunization_shortfall_predicted"].sum() > 0
    assert no_routine.loc[no_routine["year"].between(2020, 2024), "routine_immunization_shortfall_predicted"].sum() == 0

    source = inspect.getsource(_predict_monthly)
    assert "dtp1_eff" in source
    assert "routine_dtp1_eligibility" in source
    assert "routine_dtp3_eligibility" in source
    assert "routine_delay_debt" in source
    assert "routine_delay_multiplier" in source
    assert "routine_delay_risk_base" in source
    assert "age_effective_susceptible += routine_delay_risk_base" in source
    assert "vaccine_partial_recent" in source
    assert "completion_flow" in source


def test_model2c_waning_booster_and_demography_switches_are_distinct() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0]
    full = _predict_monthly(data, draw, scenario_id="full_model", totals_only=True)
    low_waning = _predict_monthly(data, draw, scenario_id="low_vaccine_waning", totals_only=True)
    restored_booster = _predict_monthly(data, draw, scenario_id="restored_booster_reach", totals_only=True)
    fixed_demography = _predict_monthly(data, draw, scenario_id="fixed_birth_cohort_demography", totals_only=True)

    assert not math.isclose(low_waning["true_infections"], full["true_infections"])
    assert not math.isclose(restored_booster["true_infections"], full["true_infections"])
    assert not math.isclose(low_waning["true_infections"], restored_booster["true_infections"])
    assert not math.isclose(fixed_demography["reported_cases"], full["reported_cases"])

    equal_reference_data = {
        **data,
        "monthly_births_ref": data["monthly_births"],
        "age_population_ref": data["age_population"],
    }
    fixed_equal = _predict_monthly(equal_reference_data, draw, scenario_id="fixed_birth_cohort_demography")
    full_equal = _predict_monthly(equal_reference_data, draw, scenario_id="full_model")
    np.testing.assert_allclose(
        fixed_equal["reported_cases_predicted"].to_numpy(),
        full_equal["reported_cases_predicted"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    source = inspect.getsource(_predict_monthly)
    assert "vaccine_waning_neutralized" in source
    assert "month_waning_rate = 0.0 if vaccine_waning_neutralized else waning_rate" in source
    assert "susc_vaccine_waned = susc_vaccine_recent" in source
    assert "symptom_vaccine_waned = symptom_vaccine_recent" in source


def test_model2c_reporting_and_transmission_context_audits_have_expected_scope() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0]
    full = _predict_monthly(data, draw, scenario_id="full_model", totals_only=True)
    no_context = _predict_monthly(data, draw, scenario_id="no_post_2023_transmission_context", totals_only=True)
    no_reporting = _predict_monthly(data, draw, scenario_id="no_reporting_testing_shift", totals_only=True)

    assert not math.isclose(no_context["true_infections"], full["true_infections"])
    assert math.isclose(no_reporting["true_infections"], full["true_infections"])
    assert math.isclose(no_reporting["symptomatic_infections"], full["symptomatic_infections"])
    assert not math.isclose(no_reporting["reported_cases"], full["reported_cases"])


def test_model2c_reporting_peak_gain_is_observation_process_only() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0].copy()
    no_peak = draw.copy()
    no_peak["post_reporting_peak_gain"] = 0.0
    strong_peak = draw.copy()
    strong_peak["post_reporting_peak_gain"] = 1.2

    base = _predict_monthly(data, no_peak, scenario_id="full_model")
    flexible = _predict_monthly(data, strong_peak, scenario_id="full_model")
    neutral = _predict_monthly(data, strong_peak, scenario_id="no_reporting_testing_shift")
    post_peak = base["year"].between(2024, 2025)

    assert flexible.loc[post_peak, "reported_cases_predicted"].sum() > base.loc[post_peak, "reported_cases_predicted"].sum()
    assert np.allclose(
        neutral["true_infections_predicted"].to_numpy(),
        flexible["true_infections_predicted"].to_numpy(),
    )
    assert neutral.loc[post_peak, "reported_cases_predicted"].sum() < flexible.loc[
        post_peak, "reported_cases_predicted"
    ].sum()


def test_model2c_secondary_rebound_pulse_changes_post_peak_shape() -> None:
    data, _metadata, draws = _scenario_data()
    draw = draws.iloc[0].copy()
    no_secondary = draw.copy()
    no_secondary["secondary_rebound_release_gain"] = 0.0
    strong_secondary = draw.copy()
    strong_secondary["secondary_rebound_release_gain"] = 2.0

    base = _predict_monthly(data, no_secondary, scenario_id="full_model")
    flexible = _predict_monthly(data, strong_secondary, scenario_id="full_model")
    post_peak = base["year"].between(2024, 2025)

    assert flexible.loc[post_peak, "reported_cases_predicted"].sum() > base.loc[post_peak, "reported_cases_predicted"].sum()
    assert not np.allclose(
        flexible.loc[post_peak, "reported_cases_predicted"].to_numpy(),
        base.loc[post_peak, "reported_cases_predicted"].to_numpy(),
    )


def test_model2c_summary_matches_draw_level_monthly_totals_and_peaks() -> None:
    data, metadata, draws = _scenario_data()
    frame = _scenario_prediction_frame(
        data,
        metadata,
        draws,
        max_draws=2,
        start_year=2023,
        end_year=2025,
    )
    _monthly, summary = model2c_counterfactual_outputs(
        data,
        metadata,
        draws,
        max_draws=2,
        start_year=2023,
        end_year=2025,
    )
    scenario_id = "no_contact_suppression"
    outcome_column = "reported_cases_predicted"
    scenario_frame = frame.loc[frame["scenario_id"].eq(scenario_id)].copy()
    full_frame = frame.loc[frame["scenario_id"].eq("full_model")].copy()
    scenario_totals = scenario_frame.groupby("draw")[outcome_column].sum()
    full_totals = full_frame.groupby("draw")[outcome_column].sum()
    differences = full_totals - scenario_totals
    ratios = scenario_totals / full_totals
    scenario_peaks = scenario_frame.groupby("draw")[outcome_column].max()
    full_peaks = full_frame.groupby("draw")[outcome_column].max()
    peak_ratios = scenario_peaks / full_peaks

    row = summary.loc[
        summary["scenario_id"].eq(scenario_id) & summary["outcome_target"].eq("reported_cases")
    ].iloc[0]
    assert math.isclose(row["scenario_total_median"], float(scenario_totals.median()))
    assert math.isclose(row["full_total_median"], float(full_totals.median()))
    assert math.isclose(row["full_minus_scenario_median"], float(differences.median()))
    assert math.isclose(row["scenario_to_full_ratio_median"], float(ratios.median()))
    assert math.isclose(row["scenario_peak_median"], float(scenario_peaks.median()))
    assert math.isclose(row["full_peak_median"], float(full_peaks.median()))
    assert math.isclose(row["peak_ratio_median"], float(peak_ratios.median()))


def test_model2c_draw_parallel_counterfactual_matches_serial() -> None:
    data, metadata, draws = _scenario_data()

    monthly_serial, summary_serial = model2c_counterfactual_outputs(
        data,
        metadata,
        draws,
        max_draws=2,
        start_year=2023,
        end_year=2025,
        draw_workers=1,
    )
    monthly_parallel, summary_parallel = model2c_counterfactual_outputs(
        data,
        metadata,
        draws,
        max_draws=2,
        start_year=2023,
        end_year=2025,
        draw_workers=2,
    )

    pd.testing.assert_frame_equal(
        monthly_serial.reset_index(drop=True),
        monthly_parallel.reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        summary_serial.reset_index(drop=True),
        summary_parallel.reset_index(drop=True),
        check_exact=True,
    )


def test_model2c_active_pipeline_no_longer_requires_shapley_outputs() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    if "PUBLIC_RELEASE := 1" in makefile:
        pytest.skip("The public release uses a lightweight Makefile without fit targets.")
    analysis_block = makefile.split("analysis: data", maxsplit=1)[1].split("model2 model2-nodata", maxsplit=1)[0]
    figure_deps_block = makefile.split("FIGURE_DEPS :=", maxsplit=1)[1].split(".DELETE_ON_ERROR", maxsplit=1)[0]
    model2c_block = makefile.split("MODEL2C_OUTPUTS :=", maxsplit=1)[1].split("MODEL2B_DEPS :=", maxsplit=1)[0]

    assert "$(MODEL2C_STAMP)" in analysis_block
    assert "$(MODEL2B_STAMP)" not in analysis_block
    assert "figure_4_bayesian_primary_factor_shares.csv" not in figure_deps_block
    assert "figure_4_model2c_scenario_replay_rendered.csv" in makefile
    assert "figure_4_bayesian_transmission_attribution_rendered.csv" not in figure_deps_block
    assert "model2b_bayes_shapley_draws" not in model2c_block
    assert "model2b_bayes_factor_weight_summary" not in model2c_block
    assert "layered_attribution" not in model2c_block
    assert "pooled_9country" not in model2c_block

    active_scripts = [
        ROOT / "script/build_model2c_counterfactual_scenarios.py",
        ROOT / "script/build_model2c_manuscript_tables.py",
        ROOT / "script/src/pertussis_global_dynamics/model2c.py",
        ROOT / "script/figures/figure_4_model2c_scenario_replay.R",
    ]
    for path in active_scripts:
        text = path.read_text(encoding="utf-8")
        assert "model2b_bayes_shapley_draws" not in text
        assert "summarize_model2b_pooled_attribution" not in text
        assert "figure_4_bayesian_primary_factor_shares" not in text
        if path.name == "figure_4_model2c_scenario_replay.R":
            assert "save_main_figure(" in text
            assert '"figure_4_model2c_scenario_replay"' in text
            assert "figure_4_bayesian_transmission_attribution.png" not in text
