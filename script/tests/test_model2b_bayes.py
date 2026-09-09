from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pertussis_global_dynamics.bayes.model2b_attribution import (
    MODEL2B_COUNTERFACTUAL_SCENARIOS,
    MODEL2B_CONTEXT_AUDIT_FACTORS,
    MODEL2B_OUTPUT_VERSION,
    MODEL2B_PRIMARY_MECHANISMS,
    MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS,
    MODEL2B_REPORTED_FACTORS,
    MODEL2B_TRANSMISSION_FACTORS,
    MODEL2B_TRANSMISSION_CONTEXT_FACTORS,
    _predict_monthly,
    model2b_transmission_context_audit,
    model2b_counterfactual_scenarios,
    model2b_country_outputs,
    model2b_observation_audit,
    summarize_model2b_factor_weights,
    summarize_model2b_pooled_attribution,
)
from pertussis_global_dynamics.bayes.model2b_data import (
    GBD_TO_SILENT_EXPOSURE_MULTIPLIER,
    NATURAL_EXPOSURE_CONTACT_SENSITIVITY_DEFAULT,
    SILENT_EXPOSURE_FLOOR_MIN_MONTHLY,
    build_model2b_stan_data,
)
from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.data.model2_inputs import (
    choose_dtp_coverage,
    pre_mean_dtp1_from_coverage,
    pre_mean_dtp3_from_coverage,
)


ROOT = Path(__file__).resolve().parents[2]


def test_choose_dtp_coverage_uses_wuenic_only() -> None:
    coverage = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "year": 2024,
                "antigen": "DTPCV3",
                "dose": "3",
                "coverage": 0.91,
                "coverage_type": "WHO/UNICEF Estimates of National Immunization Coverage",
            },
            {
                "iso3": "TST",
                "year": 2025,
                "antigen": "DTPCV3",
                "dose": "3",
                "coverage": 0.73,
                "coverage_type": "Official coverage",
            },
        ]
    )

    chosen = choose_dtp_coverage(coverage, "3")

    assert chosen.to_dict("records") == [{"iso3": "TST", "year": 2024, "dtp3_coverage": 0.91}]


def test_pre_mean_dtp3_uses_full_coverage_window_not_case_window() -> None:
    dtp3 = pd.DataFrame(
        [
            {"iso3": "TST", "year": year, "dtp3_coverage": coverage}
            for year, coverage in [
                (2010, 0.99),
                (2011, 0.99),
                (2012, 0.95),
                (2013, 0.97),
                (2014, 0.93),
                (2015, 0.96),
                (2016, 0.89),
                (2017, 0.83),
                (2018, 0.87),
                (2019, 0.70),
            ]
        ]
    )

    assert math.isclose(pre_mean_dtp3_from_coverage(dtp3, "TST"), 0.908)


def test_pre_mean_dtp1_uses_full_coverage_window_not_case_window() -> None:
    dtp1 = pd.DataFrame(
        [
            {"iso3": "TST", "year": year, "dtp1_coverage": coverage}
            for year, coverage in [
                (2010, 0.96),
                (2011, 0.96),
                (2012, 0.95),
                (2013, 0.95),
                (2014, 0.94),
                (2015, 0.93),
                (2016, 0.92),
                (2017, 0.91),
                (2018, 0.90),
                (2019, 0.89),
            ]
        ]
    )

    assert math.isclose(pre_mean_dtp1_from_coverage(dtp1, "TST"), 0.931)


def _synthetic_context(
    *,
    surveillance_outcomes: pd.DataFrame | None = None,
    pathogen_timeline: pd.DataFrame | None = None,
) -> CountryContext:
    annual = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "year": year,
                "cases_total": 100 + 5 * (year - 2015),
                "population": 1_000_000 + 1_000 * (year - 2015),
                "dtp1_coverage": 0.96 if year < 2020 else 0.89,
                "dtp3_coverage": 0.92 if year < 2020 else 0.86,
                "periodic_expected_incidence_per_100k": 10.0,
            }
            for year in range(2015, 2027)
        ]
    )
    time_resolved = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "year": year,
                "month": month,
                "date_start": f"{year}-{month:02d}-01",
                "cases_total": 5 + (year >= 2023) * 20 + month % 3,
            }
            for year in range(2015, 2027)
            for month in range(1, 13)
        ]
    )
    population_age = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "year": year,
                "age_group": age_group,
                "population": population,
            }
            for year in (2019, 2023, 2026)
            for age_group, population in [
                ("infant_0_2m", 15_000),
                ("infant_3_11m", 45_000),
                ("child_1_4y", 120_000),
                ("child_5_9y", 130_000),
                ("adolescent_10_17y", 150_000),
                ("adult_18_39y", 260_000),
                ("adult_40_64y", 210_000),
                ("adult_65plus", 70_000),
            ]
        ]
    )
    monthly_covid = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "year": year,
                "month": month,
                "monthly_stringency_index": 55.0 if 2020 <= year <= 2022 else 0.0,
            }
            for year in range(2015, 2027)
            for month in range(1, 13)
        ]
    )
    return CountryContext(
        iso3="TST",
        country="Testland",
        annual=annual,
        time_resolved=time_resolved,
        population_age=population_age,
        metadata={"routine_scheduler_code": "aP"},
        pre_mean_dtp3=0.92,
        pre_mean_incidence=10.0,
        baseline_note="synthetic",
        monthly_covid=monthly_covid,
        pathogen_timeline=pathogen_timeline if pathogen_timeline is not None else pd.DataFrame(),
        surveillance_outcomes=surveillance_outcomes if surveillance_outcomes is not None else pd.DataFrame(),
        pre_mean_dtp1=0.96,
    )


def _inputs(
    *,
    contact_mobility: pd.DataFrame | None = None,
    gbd_burden: pd.DataFrame | None = None,
    serology_evidence: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "age_cases": pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "year",
                "model_age_group",
                "cases_total",
                "allocation_fraction",
                "source_age_groups",
            ]
        ),
        "contact_mobility": contact_mobility if contact_mobility is not None else pd.DataFrame(),
        "gbd_burden": gbd_burden if gbd_burden is not None else pd.DataFrame(),
        "serology_evidence": serology_evidence if serology_evidence is not None else pd.DataFrame(),
    }


def _contact_mobility() -> pd.DataFrame:
    bins = [
        "[00,05)",
        "[05,10)",
        "[10,15)",
        "[15,20)",
        "[20,25)",
        "[25,30)",
        "[30,35)",
        "[35,40)",
        "[40,45)",
        "[45,50)",
        "[50,55)",
        "[55,60)",
        "[60,65)",
        "[65,70)",
        "[70,75)",
        "[75,80)",
    ]
    return pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "source_age_group": source,
                "target_age_group": target,
                "contact_location": "all",
                "contacts_per_day": 0.10 + (2.0 if source == target else 0.0) + 0.01 * source_index,
            }
            for source_index, source in enumerate(bins)
            for target in bins
        ]
    )


def _draws() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "draw": 1,
                "log_beta": math.log(0.000012),
                "log_reporting_rate": math.log(0.08),
                "contact_sensitivity": 0.50,
                "waning_rate": 0.10,
                "log_post_transmission_multiplier": math.log(1.18),
                "log_post_reporting_multiplier": math.log(1.02),
                "rebound_peak_month": 20.0,
                "rebound_width": 8.0,
                "rebound_release_gain": 1.10,
                "outbreak_seeding_gain": 0.015,
                "seasonal_strength": 1.0,
                "natural_boosting_efficiency": 0.25,
                "pathogen_pressure_scale": 1.0,
                "epidemic_feedback_gain": 4.0,
                "endemic_reseed_pressure": 0.01,
                "epidemic_pressure_retention": 0.45,
                "post_reporting_peak_gain": 0.20,
                "nb_phi_month": 20.0,
                "nb_phi_age": 10.0,
                "missed_natural_boosting_2020_2022": 1200.0,
                "natural_boosting_flow_2020_2022": 4000.0,
                "routine_vaccination_shortfall_2020_2022": 2500.0,
                "natural_waned_stock_2023_start": 10000.0,
                "init_natural_share": 0.08,
                "init_vaccine_recent_fraction": 0.35,
                "init_natural_recent_fraction": 0.40,
                "nb_phi": 20.0,
            },
            {
                "draw": 2,
                "log_beta": math.log(0.000013),
                "log_reporting_rate": math.log(0.07),
                "contact_sensitivity": 0.65,
                "waning_rate": 0.12,
                "log_post_transmission_multiplier": math.log(1.20),
                "log_post_reporting_multiplier": math.log(0.98),
                "rebound_peak_month": 24.0,
                "rebound_width": 9.0,
                "rebound_release_gain": 1.20,
                "outbreak_seeding_gain": 0.018,
                "seasonal_strength": 1.2,
                "natural_boosting_efficiency": 0.30,
                "pathogen_pressure_scale": 1.1,
                "epidemic_feedback_gain": 5.0,
                "endemic_reseed_pressure": 0.012,
                "epidemic_pressure_retention": 0.50,
                "post_reporting_peak_gain": 0.25,
                "nb_phi_month": 25.0,
                "nb_phi_age": 12.0,
                "missed_natural_boosting_2020_2022": 1400.0,
                "natural_boosting_flow_2020_2022": 4200.0,
                "routine_vaccination_shortfall_2020_2022": 2600.0,
                "natural_waned_stock_2023_start": 11000.0,
                "init_natural_share": 0.09,
                "init_vaccine_recent_fraction": 0.38,
                "init_natural_recent_fraction": 0.45,
                "nb_phi": 25.0,
            },
        ]
    )


def test_model2b_reporting_prior_is_tight_null_without_direct_testing_evidence() -> None:
    data, metadata = build_model2b_stan_data(_synthetic_context(), _inputs())

    assert data["N_months"] == 144
    assert data["N_month_obs"] == 132
    assert data["N_month_ppc_obs"] == 12
    assert len(data["month_obs_weight"]) == data["N_month_obs"]
    month_obs_weight = np.asarray(data["month_obs_weight"], dtype=float)
    observed_months = np.asarray(data["y_month"], dtype=float)
    high_months = observed_months >= np.quantile(observed_months[observed_months > 0], 0.90)
    assert month_obs_weight.min() >= 0.55
    assert month_obs_weight.max() <= 5.50
    assert not np.allclose(month_obs_weight, np.ones(data["N_month_obs"]))
    assert float(np.median(month_obs_weight[high_months])) > float(np.median(month_obs_weight[~high_months]))
    assert metadata["model2b_month_obs_weight_min"] == float(month_obs_weight.min())
    assert metadata["model2b_month_obs_weight_mean"] == float(month_obs_weight.mean())
    assert metadata["model2b_month_obs_weight_max"] == float(month_obs_weight.max())
    assert "peak-balanced likelihood weights" in metadata["model2b_note"]
    assert "seasonal_multiplier" in data
    assert max(data["seasonal_multiplier"]) > min(data["seasonal_multiplier"])
    assert "pathogen_pressure_prior" in data
    assert min(data["pathogen_pressure_prior"]) >= 0.30
    assert data["pre_mean_incidence_per_100k"] >= 0
    assert "silent_exposure_floor_monthly" in data
    assert data["silent_exposure_floor_monthly"] == SILENT_EXPOSURE_FLOOR_MIN_MONTHLY
    assert data["natural_exposure_contact_sensitivity"] == NATURAL_EXPOSURE_CONTACT_SENSITIVITY_DEFAULT
    assert metadata["model2c_silent_exposure_floor_monthly"] == SILENT_EXPOSURE_FLOOR_MIN_MONTHLY
    assert metadata["model2c_silent_exposure_floor_method"] == "minimum_silent_exposure_floor_without_gbd_anchor"
    assert metadata["model2c_natural_exposure_contact_sensitivity"] == NATURAL_EXPOSURE_CONTACT_SENSITIVITY_DEFAULT
    assert 0 < metadata["model2b_initial_natural_boosting_efficiency"] < 1
    assert metadata["model2b_initial_pathogen_pressure_scale"] > 0
    assert metadata["model2b_initial_outbreak_seeding_gain"] > 0
    assert metadata["model2b_initial_epidemic_feedback_gain"] > 0
    assert metadata["model2b_initial_endemic_reseed_pressure"] > 0
    assert metadata["model2b_initial_nb_phi_month"] > 0.2
    assert metadata["model2b_initial_nb_phi_age"] > 0.2
    assert 0.15 < metadata["model2b_initial_epidemic_pressure_retention"] < 0.85
    assert np.asarray(data["age_contact_matrix"]).shape == (data["N_age"], data["N_age"])
    np.testing.assert_allclose(
        np.asarray(data["age_contact_matrix"], dtype=float),
        np.outer(np.asarray(data["age_rel_contact"], dtype=float), np.asarray(data["age_rel_contact"], dtype=float)),
    )
    assert len(data["booster_reach_modifier"]) == data["N_age"]
    assert len(data["vaccine_waning_age_modifier"]) == data["N_age"]
    assert len(data["routine_dtp1_eligibility"]) == data["N_age"]
    assert len(data["routine_dtp3_eligibility"]) == data["N_age"]
    assert len(data["routine_delay_age_weight"]) == data["N_age"]
    assert data["routine_delay_sensitivity"] == 8.0
    assert math.isclose(sum(data["routine_dtp1_eligibility"]), 1.0)
    assert math.isclose(sum(data["routine_dtp3_eligibility"]), 1.0)
    assert metadata["model2c_routine_flow_method"] == (
        "monthly_birth_cohort_dtp1_dtp3_eligibility_with_limited_1_4y_catchup_and_delay_debt"
    )
    assert metadata["model2c_routine_delay_sensitivity"] == 8.0
    assert len(data["monthly_births"]) == data["N_months"]
    assert len(data["monthly_births_ref"]) == data["N_months"]
    assert len(data["age_aging_rate"]) == data["N_age"]
    assert max(data["age_aging_rate"]) > 0
    assert data["age_aging_rate"][-1] == 0
    assert data["init_natural_share_prior_mean"] >= SILENT_EXPOSURE_FLOOR_MIN_MONTHLY * 24.0
    assert 0 <= data["init_natural_share_prior_mean"] <= 0.50
    assert metadata["model2c_initial_natural_share_prior_mean"] == data["init_natural_share_prior_mean"]
    assert "boosting_debt_weight_prior_mean" not in data
    assert "coverage_disruption_weight_prior_mean" not in data
    assert metadata["model2b_age_state_transition_method"] == (
        "monthly_birth_entry_age_group_aging_and_wpp_net_demographic_adjustment_cohort"
    )
    assert metadata["model2b_initial_immunity_method"].startswith("posterior_initial_immunity")
    assert metadata["model2b_contact_intensity_method"] == (
        "oxcgrt_policy_stringency_2020_2022_plus_decaying_low_circulation_memory_through_2024"
    )
    contact_by_year = pd.DataFrame({"year": data["year"], "contact_intensity": data["contact_intensity"]})
    assert contact_by_year.loc[contact_by_year["year"].eq(2023), "contact_intensity"].max() > 0
    assert contact_by_year.loc[contact_by_year["year"].eq(2024), "contact_intensity"].max() > 0
    assert contact_by_year.loc[contact_by_year["year"].eq(2025), "contact_intensity"].max() == 0
    assert metadata["contact_mode_requested"] == "full"
    assert metadata["model2b_contact_matrix_source"].startswith("rank_one_fallback")
    assert metadata["model2b_contact_matrix_fallback"] is True
    assert metadata["vaccine_type"] == "aP"
    assert data["waning_rate_prior_mean"] == 0.14
    assert data["waning_rate_prior_sd"] == 0.05
    assert metadata["model2b_waning_rate_prior_sd"] == 0.05
    assert metadata["maternal_program_context"] is False
    assert metadata["direct_testing_support"] is False
    assert metadata["direct_testing_denominator_evidence"] is False
    assert metadata["case_definition_reporting_practice_evidence"] is False
    assert metadata["severe_outcome_evidence"] is False
    assert metadata["serology_evidence"] is False
    assert data["reporting_prior_log_mean"] == 0.0
    assert data["reporting_prior_log_sd"] <= 0.25
    assert metadata["reporting_prior_method"] == "strong_null_reporting_prior_without_direct_testing_denominator"
    assert metadata["model2b_output_version"] == MODEL2B_OUTPUT_VERSION
    assert data["log_post_transmission_prior_sd"] == 0.20
    assert data["pathogen_pressure_scale_prior_mean"] == 0.25
    assert data["pathogen_pressure_scale_prior_sd"] == 0.25
    assert data["outbreak_seeding_gain_prior_mean"] == 0.010
    assert data["outbreak_seeding_gain_prior_log_sd"] == 0.45
    assert metadata["model2b_pathogen_pressure_scale_prior_mean"] == 0.25
    assert metadata["model2b_outbreak_seeding_gain_prior_mean"] == 0.010


def test_model2c_silent_exposure_floor_uses_gbd_burden_independent_of_reporting_prior() -> None:
    context = _synthetic_context()
    gbd_burden = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "year": year,
                "measure_name": "Incidence",
                "metric_name": "Number",
                "age_name": "All ages",
                "val": 1200.0,
            }
            for year in range(2015, 2022)
        ]
    )

    data, metadata = build_model2b_stan_data(context, _inputs(gbd_burden=gbd_burden))

    mean_population = float(context.annual.loc[context.annual["year"].between(2010, 2021), "population"].mean())
    expected_gbd_per_100k = 100000.0 * 1200.0 / mean_population
    expected_floor = expected_gbd_per_100k / 100000.0 / 12.0 * GBD_TO_SILENT_EXPOSURE_MULTIPLIER
    assert data["silent_exposure_floor_monthly"] > SILENT_EXPOSURE_FLOOR_MIN_MONTHLY
    assert math.isclose(data["silent_exposure_floor_monthly"], expected_floor)
    assert math.isclose(metadata["model2c_silent_exposure_gbd_incidence_per_100k"], expected_gbd_per_100k)
    assert metadata["model2c_silent_exposure_floor_method"] == "gbd_incidence_scaled_silent_exposure_floor"


def test_model2c_silent_exposure_floor_uses_recent_infection_serology_when_available() -> None:
    serology = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "year": 2018,
                "serology_metric": "recent_infection_prevalence",
                "serology_value": 12.0,
                "serology_unit": "percent",
            }
        ]
    )

    data, metadata = build_model2b_stan_data(_synthetic_context(), _inputs(serology_evidence=serology))

    assert math.isclose(data["silent_exposure_floor_monthly"], 0.12 / 12.0)
    assert math.isclose(data["init_natural_share_prior_mean"], 0.12 / 12.0 * 24.0)
    assert metadata["model2c_silent_exposure_serology_recent_infection_percent"] == 12.0
    assert math.isclose(metadata["model2c_silent_exposure_serology_floor_monthly"], 0.12 / 12.0)
    assert metadata["model2c_silent_exposure_floor_method"] == (
        "serology_recent_infection_prevalence_with_optional_gbd_floor"
    )


def test_model2b_reporting_prior_uses_direct_testing_denominator_evidence() -> None:
    surveillance = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "year": 2023,
                "testing_denominator": 1000,
                "source_type": "annual report",
                "source_ids": "testing-source",
            },
            {
                "iso3": "TST",
                "country": "Testland",
                "year": 2024,
                "testing_denominator": 1600,
                "source_type": "annual report",
                "source_ids": "testing-source",
            },
        ]
    )

    data, metadata = build_model2b_stan_data(
        _synthetic_context(surveillance_outcomes=surveillance),
        _inputs(),
    )

    assert metadata["direct_testing_support"] is True
    assert metadata["direct_testing_denominator_evidence"] is True
    assert "direct_testing_denominator" in metadata["reporting_prior_method"]
    assert data["reporting_prior_log_sd"] >= 0.10
    assert data["reporting_prior_log_sd"] <= 0.75


def test_model2b_pathogen_evidence_uses_informed_pressure_priors() -> None:
    pathogen_timeline = pd.DataFrame(
        [
            {
                "iso3": "TST",
                "country": "Testland",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
                "transmission_prior_mean": 1.5,
                "transmission_prior_lower": 1.1,
                "transmission_prior_upper": 2.0,
            },
        ]
    )

    data, metadata = build_model2b_stan_data(
        _synthetic_context(pathogen_timeline=pathogen_timeline),
        _inputs(),
    )

    assert metadata["pathogen_pressure_evidence"] is True
    assert data["pathogen_pressure_scale_prior_mean"] == 1.0
    assert data["pathogen_pressure_scale_prior_sd"] == 0.45
    assert data["outbreak_seeding_gain_prior_mean"] == 0.025
    assert data["outbreak_seeding_gain_prior_log_sd"] == 0.70


def test_model2b_full_contact_matrix_shape_and_source(monkeypatch) -> None:
    monkeypatch.setenv("MODEL2B_CONTACT_MODE", "full")

    data, metadata = build_model2b_stan_data(
        _synthetic_context(),
        _inputs(contact_mobility=_contact_mobility()),
    )

    contact = np.asarray(data["age_contact_matrix"], dtype=float)
    rank_one = np.outer(np.asarray(data["age_rel_contact"], dtype=float), np.asarray(data["age_rel_contact"], dtype=float))
    assert contact.shape == (data["N_age"], data["N_age"])
    assert metadata["contact_mode_requested"] == "full"
    assert metadata["model2b_contact_matrix_source"] == "full_contact_matrix"
    assert metadata["model2b_contact_matrix_fallback"] is False
    assert not np.allclose(contact, rank_one)


def test_model2b_full_contact_matrix_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MODEL2B_CONTACT_MODE", "full")

    data, metadata = build_model2b_stan_data(_synthetic_context(), _inputs())

    contact = np.asarray(data["age_contact_matrix"], dtype=float)
    rank_one = np.outer(np.asarray(data["age_rel_contact"], dtype=float), np.asarray(data["age_rel_contact"], dtype=float))
    assert contact.shape == (data["N_age"], data["N_age"])
    assert metadata["contact_mode_requested"] == "full"
    assert metadata["model2b_contact_matrix_source"].startswith("rank_one_fallback")
    assert metadata["model2b_contact_matrix_fallback"] is True
    np.testing.assert_allclose(contact, rank_one)


def test_model2b_stan_removes_external_debt_lambda_multipliers() -> None:
    stan = (ROOT / "script/stan/model2b_transmission_attribution.stan").read_text(encoding="utf-8")

    assert "coverage_penalty" not in stan
    assert "coverage_disruption_weight" not in stan
    assert "boosting_debt_weight" not in stan
    assert "natural_immunity_debt" not in stan
    assert "routine_immunization_debt" not in stan
    assert "missed_natural_boosting_2020_2022" in stan
    assert "routine_vaccination_shortfall_2020_2022" in stan
    assert "contact_sensitivity_upper" in stan
    assert "upper=contact_sensitivity_upper> contact_sensitivity" in stan
    assert "0.03 + 0.97 * exp(-contact_sensitivity * contact_intensity[m] / 100.0)" in stan
    assert "if (contact_intensity[m] > 0.0)" in stan
    assert "contact_sensitivity ~ normal(contact_sensitivity_prior_mean, 0.75)" in stan
    assert "lower=-0.69314718, upper=0.91629073> log_post_transmission_multiplier" in stan
    assert "lower=0.2, upper=500.0> nb_phi_month" in stan
    assert "upper=4.0> rebound_release_gain" in stan


def test_model2c_stan_uses_silent_exposure_floor_for_natural_recharge() -> None:
    stan = (ROOT / "script/stan/model2c_transmission_scenario_replay.stan").read_text(encoding="utf-8")

    assert "silent_exposure_floor_monthly" in stan
    assert "natural_exposure_contact_sensitivity" in stan
    assert "exposure_boosting_contact_multiplier" in stan
    assert "boosting_lambda_age" in stan
    assert "reported_exposure_floor = reported_baseline_monthly_risk * silent_exposure_multiplier * 10.0" in stan
    assert "background_exposure_floor = fmin(0.05, fmax(silent_exposure_floor_monthly, reported_exposure_floor))" in stan
    assert "natural_boosting_efficiency * boosting_lambda_age * available_natural_waned" in stan
    assert "natural_boosting_efficiency ~ normal(0.60, 0.12)" in stan
    assert "routine_dtp1_eligibility" in stan
    assert "routine_dtp3_eligibility" in stan
    assert "routine_delay_debt" in stan
    assert "routine_delay_sensitivity" in stan
    assert "routine_delay_multiplier" in stan
    assert "recharge_vaccine_recent" in stan
    assert "recharge_partial_recent" in stan
    assert "waning_rate_prior_sd" in stan
    assert "waning_rate ~ normal(waning_rate_prior_mean, waning_rate_prior_sd)" in stan
    assert "upper=0.30> outbreak_seeding_gain" in stan
    assert "log_post_transmission_prior_sd" in stan
    assert "pathogen_pressure_scale_prior_mean" in stan
    assert "pathogen_pressure_scale_prior_sd" in stan
    assert "outbreak_seeding_gain_prior_mean" in stan
    assert "outbreak_seeding_gain_prior_log_sd" in stan
    assert "endemic_background_force = endemic_reseed_pressure" in stan
    assert "excess_epidemic_pressure_multiplier" in stan
    assert "outbreak_ignition_pressure" in stan
    assert "outbreak_ignition_pulse_gain = outbreak_seeding_gain" in stan
    assert "post_reporting_peak_gain ~ normal(0.0, 0.45)" in stan
    assert "post_reporting_peak_multiplier = 1.0 + post_reporting_peak_gain" in stan
    assert "log_post_transmission_multiplier ~ normal(log_post_transmission_prior_mean, log_post_transmission_prior_sd)" in stan
    assert "pathogen_pressure_scale ~ normal(pathogen_pressure_scale_prior_mean, pathogen_pressure_scale_prior_sd)" in stan
    assert "outbreak_seeding_gain ~ lognormal(log(outbreak_seeding_gain_prior_mean), outbreak_seeding_gain_prior_log_sd)" in stan
    assert "0.35 * rebound_release_gain" in stan
    assert "monthly_births" in stan
    assert "age_aging_rate" in stan
    assert "init_natural_share" in stan
    assert "net_demographic_adjustment_2015_2026" in stan


def test_context_factors_are_not_primary_susceptibility_factors() -> None:
    data, _ = build_model2b_stan_data(_synthetic_context(), _inputs())
    draw = _draws().iloc[0]
    full_prediction = _predict_monthly(data, draw, active_factors=set(MODEL2B_REPORTED_FACTORS))
    no_shift_prediction = _predict_monthly(
        data,
        draw,
        active_factors=set(MODEL2B_REPORTED_FACTORS) - {"post_2023_transmission_shift"},
    )
    no_excess_pressure_prediction = _predict_monthly(
        data,
        draw,
        active_factors=set(MODEL2B_REPORTED_FACTORS) - {"excess_epidemic_pressure", "outbreak_ignition_pulse"},
    )

    full_true = full_prediction.loc[full_prediction["year"].between(2023, 2025), "true_infections_predicted"].sum()
    no_shift_true = no_shift_prediction.loc[
        no_shift_prediction["year"].between(2023, 2025), "true_infections_predicted"
    ].sum()
    no_excess_pressure_true = no_excess_pressure_prediction.loc[
        no_excess_pressure_prediction["year"].between(2023, 2025), "true_infections_predicted"
    ].sum()

    assert set(MODEL2B_PRIMARY_MECHANISMS) == set(MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS)
    assert "post_2023_transmission_shift" not in MODEL2B_PRIMARY_MECHANISMS
    assert "excess_epidemic_pressure" not in MODEL2B_PRIMARY_MECHANISMS
    assert "outbreak_ignition_pulse" not in MODEL2B_PRIMARY_MECHANISMS
    assert set(MODEL2B_CONTEXT_AUDIT_FACTORS) == set(MODEL2B_TRANSMISSION_CONTEXT_FACTORS)
    assert "reporting_testing_shift" not in MODEL2B_TRANSMISSION_FACTORS
    assert no_shift_true > 0
    assert no_excess_pressure_true > 0
    assert not math.isclose(no_shift_true, full_true)
    assert not math.isclose(no_excess_pressure_true, full_true)


def test_no_low_circulation_missed_boosting_removes_pandemic_contact_drop() -> None:
    data, _ = build_model2b_stan_data(_synthetic_context(), _inputs())
    draw = _draws().iloc[0]
    full_prediction = _predict_monthly(data, draw, active_factors=set(MODEL2B_REPORTED_FACTORS))
    restored_factors = set(MODEL2B_REPORTED_FACTORS) - {"low_circulation_missed_boosting"}
    restored_prediction = _predict_monthly(data, draw, active_factors=restored_factors)

    full_true = full_prediction.loc[full_prediction["year"].between(2023, 2025), "true_infections_predicted"].sum()
    restored_true = restored_prediction.loc[
        restored_prediction["year"].between(2023, 2025), "true_infections_predicted"
    ].sum()

    note = MODEL2B_COUNTERFACTUAL_SCENARIOS["no_low_circulation_missed_boosting"]["scenario_note"]
    assert "2020-2023 contact and low-circulation suppression" in note
    assert "boosting" in note
    assert restored_true > 0
    assert not math.isclose(restored_true, full_true)


def test_factor_neutralizations_have_expected_state_or_observation_effects() -> None:
    data, _ = build_model2b_stan_data(_synthetic_context(), _inputs())
    data = {
        **data,
        "monthly_births_ref": (0.75 * np.asarray(data["monthly_births"], dtype=float)).tolist(),
        "age_population_ref": (0.90 * np.asarray(data["age_population"], dtype=float)).tolist(),
    }
    pathogen_prior = np.asarray(data["pathogen_pressure_prior"], dtype=float)
    pathogen_prior[np.asarray(data["year"], dtype=int) >= 2023] = 1.35
    data["pathogen_pressure_prior"] = pathogen_prior.tolist()
    draw = _draws().iloc[0]
    full = _predict_monthly(data, draw, active_factors=set(MODEL2B_REPORTED_FACTORS), totals_only=True)

    for factor_id in [
        "routine_immunization_disruption",
        "vaccine_waning_booster_gap",
        "birth_cohort_demography",
        "post_2023_transmission_shift",
        "excess_epidemic_pressure",
        "outbreak_ignition_pulse",
    ]:
        neutral = _predict_monthly(
            data,
            draw,
            active_factors=set(MODEL2B_REPORTED_FACTORS) - {factor_id},
            totals_only=True,
        )
        assert neutral["true_infections"] > 0
        assert not math.isclose(neutral["true_infections"], full["true_infections"])

    no_reporting = _predict_monthly(
        data,
        draw,
        active_factors=set(MODEL2B_REPORTED_FACTORS) - {"reporting_testing_shift"},
        totals_only=True,
    )
    assert math.isclose(no_reporting["true_infections"], full["true_infections"])
    assert not math.isclose(no_reporting["reported_cases"], full["reported_cases"])


def test_observation_audit_parses_string_boolean_metadata() -> None:
    data, metadata = build_model2b_stan_data(_synthetic_context(), _inputs())
    metadata = {
        **metadata,
        "direct_testing_support": "False",
        "direct_testing_denominator_evidence": "False",
        "case_definition_reporting_practice_evidence": "False",
        "severe_outcome_evidence": "False",
        "serology_evidence": "False",
        "pathogen_pressure_evidence": "False",
        "model2b_contact_matrix_fallback": "False",
    }

    audit = model2b_observation_audit(data, metadata, _draws())

    assert not bool(audit["direct_testing_support"].iloc[0])
    assert not bool(audit["direct_testing_denominator_evidence"].iloc[0])
    assert not bool(audit["case_definition_reporting_practice_evidence"].iloc[0])
    assert not bool(audit["severe_outcome_evidence"].iloc[0])
    assert not bool(audit["serology_evidence"].iloc[0])
    assert not bool(audit["pathogen_pressure_evidence"].iloc[0])
    assert not bool(audit["model2b_contact_matrix_fallback"].iloc[0])


def test_recompute_script_uses_contact_mode_specific_candidates() -> None:
    script_names = [
        "recompute_model2b_attribution_from_draws",
        "build_model2b_counterfactual_scenarios",
    ]
    sys.path.insert(0, str(ROOT / "script"))
    for script_name in script_names:
        script_path = ROOT / f"script/{script_name}.py"
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rank_candidates = [path.name for path in module._mode_candidates("model2b_bayes_posterior_draws.csv", "rank_one")]
        full_candidates = [path.name for path in module._mode_candidates("model2b_bayes_posterior_draws.csv", "full")]

        assert rank_candidates == ["model2b_bayes_posterior_draws_rank_one.csv"]
        assert full_candidates == ["model2b_bayes_posterior_draws_full.csv"]


def test_model2b_counterfactual_scenarios_are_dynamic_predictions() -> None:
    data, metadata = build_model2b_stan_data(_synthetic_context(), _inputs())

    scenarios = model2b_counterfactual_scenarios(
        data,
        metadata,
        _draws(),
        max_draws=2,
        start_year=2023,
        end_year=2023,
    )

    assert set(scenarios["scenario_id"]) == set(MODEL2B_COUNTERFACTUAL_SCENARIOS)
    assert "full_model" in set(scenarios["scenario_id"])
    assert "primary_susceptibility_neutral_with_fitted_background" in set(scenarios["scenario_id"])
    assert "transmission_context_neutral_with_susceptibility_fitted" in set(scenarios["scenario_id"])
    assert "observation_neutral_with_transmission_fitted" in set(scenarios["scenario_id"])
    assert "no_low_circulation_missed_boosting" in set(scenarios["scenario_id"])
    assert "no_routine_immunization_disruption" in set(scenarios["scenario_id"])
    assert "no_post_2023_transmission_shift" in set(scenarios["scenario_id"])
    assert "no_excess_epidemic_pressure" in set(scenarios["scenario_id"])
    assert "no_outbreak_ignition_pulse" in set(scenarios["scenario_id"])
    assert "no_reporting_testing_shift" in set(scenarios["scenario_id"])
    assert scenarios["year"].min() == 2023
    assert scenarios["year"].max() == 2023
    assert scenarios["n_posterior_draws"].min() == 2
    assert scenarios["model2b_output_version"].eq(MODEL2B_OUTPUT_VERSION).all()
    assert scenarios["reported_cases_median"].notna().all()
    assert scenarios["true_infections_median"].notna().all()
    assert scenarios.loc[
        scenarios["scenario_id"].eq("no_low_circulation_missed_boosting"),
        "scenario_removed_mechanisms",
    ].eq("low_circulation_missed_boosting").all()
    assert scenarios.loc[
        scenarios["scenario_id"].eq("primary_susceptibility_neutral_with_fitted_background"),
        "scenario_removed_mechanisms",
    ].str.contains("low_circulation_missed_boosting", regex=False).all()
    assert scenarios.loc[
        scenarios["scenario_id"].eq("transmission_context_neutral_with_susceptibility_fitted"),
        "scenario_removed_mechanisms",
    ].str.contains("excess_epidemic_pressure", regex=False).all()
