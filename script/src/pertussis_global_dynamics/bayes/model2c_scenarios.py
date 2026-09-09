from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd


MODEL2C_OUTPUT_VERSION = "model2c_v10_20260705_cohort_routine_natural_exposure"
NATURAL_BOOST_REPAIR_START_YEAR = 2021
NATURAL_BOOST_REPAIR_END_YEAR = 2024
NATURAL_BOOST_STOCK_READOUT_YEAR = 2025
VACCINE_WANING_NEUTRALIZATION_START_YEAR = 2023
UNDER5_INDEX_MAX = 2
SCHOOL_AGE_INDEX_MIN = 3
SCHOOL_AGE_INDEX_MAX = 4
ADULT_INDEX_MIN = 5
DEFAULT_ROUTINE_DTP1_ELIGIBILITY = np.asarray([0.30, 0.60, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_ROUTINE_DTP3_ELIGIBILITY = np.asarray([0.00, 0.82, 0.18, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_ROUTINE_DELAY_AGE_WEIGHT = np.asarray([2.5, 2.0, 1.4, 0.6, 0.2, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_ROUTINE_DELAY_SENSITIVITY = 8.0

MODEL2C_OUTCOMES = {
    "true_infections": "true_infections_2023_2025",
    "symptomatic_infections": "symptomatic_infections_2023_2025",
    "reported_cases": "reported_cases_2023_2025",
}

MODEL2C_SCENARIOS: dict[str, dict[str, Any]] = {
    "full_model": {
        "scenario_label": "Full model",
        "scenario_family": "fitted_model",
        "scenario_order": 0,
        "neutralized_mechanisms": (),
        "scenario_interpretation_note": (
            "All fitted susceptibility, transmission-context, endemic-background and reporting terms are active."
        ),
    },
    "no_contact_suppression": {
        "scenario_label": "No contact suppression",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 1,
        "neutralized_mechanisms": ("contact_suppression",),
        "scenario_interpretation_note": (
            "The 2021-2024 low-circulation contact multiplier is restored for infection pressure while "
            "exposure-driven boosting remains on the observed low-circulation path."
        ),
    },
    "no_missed_exposure_boosting": {
        "scenario_label": "Normal 2021-2024 exposure boosting",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 2,
        "neutralized_mechanisms": ("pre_2023_missed_exposure_boosting",),
        "scenario_interpretation_note": (
            "Infection pressure remains on the fitted path, but exposure-driven immune boosting and background "
            "exposure recharge receive the no-low-circulation reference during 2021-2024. This is the single-switch "
            "counterfactual for pandemic-era natural or exposure-boosting interruption."
        ),
    },
    "restored_circulation_boosting_path": {
        "scenario_label": "Restored 2021-2024 circulation/boosting history",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 3,
        "neutralized_mechanisms": (
            "pre_2023_contact_suppression",
            "pre_2023_missed_exposure_boosting",
        ),
        "scenario_interpretation_note": (
            "The 2021-2024 low-circulation contact multiplier is restored together for infection pressure "
            "and exposure-driven immune boosting, while 2023-2025 transmission and reporting context remain "
            "on the fitted path. This historical joint switch is retained as a diagnostic only; the primary "
            "factor split uses one-switch contact and exposure-boosting scenarios."
        ),
    },
    "no_routine_immunization_disruption": {
        "scenario_label": "No routine disruption",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 4,
        "neutralized_mechanisms": ("routine_immunization_disruption",),
        "scenario_interpretation_note": (
            "DTP1 initiation and DTP3 completion flow are restored to at least the pre-pandemic baseline from 2020 onward."
        ),
    },
    "low_vaccine_waning": {
        "scenario_label": "No vaccine waning",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 5,
        "neutralized_mechanisms": ("vaccine_waning",),
        "scenario_interpretation_note": (
            "The fitted pre-2023 immune history is retained, then vaccine-derived protection is restored to "
            "recent-vaccine protection from 2023 onward while booster reach remains on the observed programme "
            "path. This is the single-switch counterfactual for post-pandemic vaccine waning."
        ),
    },
    "restored_booster_reach": {
        "scenario_label": "Restored booster reach",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 6,
        "neutralized_mechanisms": ("booster_reach",),
        "scenario_interpretation_note": (
            "Booster-related reach modifiers are restored to at least neutral programme reach while the posterior "
            "vaccine waning rate is retained."
        ),
    },
    "fixed_birth_cohort_demography": {
        "scenario_label": "Fixed birth-cohort demography",
        "scenario_family": "susceptibility_scenario",
        "scenario_order": 7,
        "neutralized_mechanisms": ("birth_cohort_demography",),
        "scenario_interpretation_note": (
            "Monthly births, aging targets and age-population denominators use the 2019 demographic reference path."
        ),
    },
    "susceptibility_neutral_combined": {
        "scenario_label": "Combined susceptibility neutral",
        "scenario_family": "combined_susceptibility_scenario",
        "scenario_order": 8,
        "neutralized_mechanisms": (
            "contact_suppression",
            "pre_2023_missed_exposure_boosting",
            "routine_immunization_disruption",
            "vaccine_waning",
            "booster_reach",
            "birth_cohort_demography",
        ),
        "scenario_interpretation_note": (
            "All susceptibility-side scenario switches are neutralized together under the same fitted posterior draw."
        ),
    },
    "no_post_2023_transmission_context": {
        "scenario_label": "No post-2023 transmission context",
        "scenario_family": "transmission_context_audit",
        "scenario_order": 9,
        "neutralized_mechanisms": ("post_2023_transmission_context",),
        "scenario_interpretation_note": (
            "Post-2023 transmission multiplier, excess epidemic pressure and outbreak ignition pulse are neutralized; "
            "endemic background circulation remains active."
        ),
    },
    "no_reporting_testing_shift": {
        "scenario_label": "No reporting/testing shift",
        "scenario_family": "observation_process_audit",
        "scenario_order": 10,
        "neutralized_mechanisms": ("reporting_testing_shift",),
        "scenario_interpretation_note": (
            "The post-2023 reporting multiplier is neutralized. This scenario describes reported-case visibility, "
            "not a transmission mechanism."
        ),
    },
}

MODEL2C_SUPPLEMENT_CONTEXT_AUDIT_SCENARIOS: dict[str, dict[str, Any]] = {
    "no_post_transmission_multiplier_only": {
        "scenario_label": "No post-transmission multiplier only",
        "scenario_family": "supplement_transmission_context_audit",
        "scenario_order": 101,
        "neutralized_mechanisms": ("post_transmission_multiplier_only",),
        "scenario_interpretation_note": (
            "Supplement-only audit neutralizing the post-2023 transmission multiplier while retaining excess "
            "epidemic pressure and outbreak ignition pulse terms."
        ),
    },
    "no_excess_epidemic_pressure_only": {
        "scenario_label": "No excess epidemic pressure only",
        "scenario_family": "supplement_transmission_context_audit",
        "scenario_order": 102,
        "neutralized_mechanisms": ("excess_epidemic_pressure_only",),
        "scenario_interpretation_note": (
            "Supplement-only audit neutralizing excess epidemic pressure while retaining the post-2023 "
            "transmission multiplier and outbreak ignition pulse terms."
        ),
    },
    "no_outbreak_ignition_pulse_only": {
        "scenario_label": "No outbreak ignition pulse only",
        "scenario_family": "supplement_transmission_context_audit",
        "scenario_order": 103,
        "neutralized_mechanisms": ("outbreak_ignition_pulse_only",),
        "scenario_interpretation_note": (
            "Supplement-only audit neutralizing outbreak ignition pulse structure while retaining the "
            "post-2023 transmission multiplier and excess epidemic pressure terms."
        ),
    },
}


def _scenario_definitions(*, include_supplement_context_audit: bool = False) -> dict[str, dict[str, Any]]:
    if not include_supplement_context_audit:
        return MODEL2C_SCENARIOS
    return {**MODEL2C_SCENARIOS, **MODEL2C_SUPPLEMENT_CONTEXT_AUDIT_SCENARIOS}


def _array(data: dict[str, Any], key: str, dtype=float) -> np.ndarray:
    return np.asarray(data[key], dtype=dtype)


def _draw_value(draw: pd.Series | dict[str, Any], key: str, default: float) -> float:
    value = draw.get(key, default) if isinstance(draw, (pd.Series, dict)) else default
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return value if np.isfinite(value) else default


def _metadata_bool(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            return max(minimum, int(raw))
        except ValueError:
            return default
    return default


def _scenario_draw_workers(draw_count: int, requested: int | None = None) -> int:
    if draw_count <= 1:
        return 1
    workers = requested if requested is not None else _env_int("MODEL2C_SCENARIO_DRAW_WORKERS", 1)
    return max(1, min(int(workers), draw_count))


def _scenario_neutralizers(scenario_id: str) -> set[str]:
    definitions = _scenario_definitions(include_supplement_context_audit=True)
    if scenario_id not in definitions:
        raise KeyError(f"Unknown Model 2C scenario_id: {scenario_id}")
    return set(definitions[scenario_id]["neutralized_mechanisms"])


def _predict_monthly(
    data: dict[str, Any],
    draw: pd.Series | dict[str, Any],
    *,
    scenario_id: str = "full_model",
    include_predictive_replicate: bool = False,
    totals_only: bool = False,
    include_age_predictions: bool = False,
) -> pd.DataFrame | dict[str, float]:
    neutralized = _scenario_neutralizers(scenario_id)
    year = _array(data, "year", int)
    month = _array(data, "month", int)
    dtp3 = _array(data, "dtp3")
    dtp1 = _array(data, "dtp1") if "dtp1" in data else np.asarray(dtp3, dtype=float)
    dtp1 = np.maximum(dtp1, dtp3)
    pre_mean_dtp3 = float(data["pre_mean_dtp3"])
    pre_mean_dtp1 = float(data.get("pre_mean_dtp1", max(pre_mean_dtp3, float(np.nanmedian(dtp1)))))
    pre_mean_dtp1 = float(np.clip(max(pre_mean_dtp1, pre_mean_dtp3), 0.01, 0.99))
    use_reference_demography = "birth_cohort_demography" in neutralized
    age_population = _array(data, "age_population_ref" if use_reference_demography else "age_population")
    monthly_births = _array(data, "monthly_births_ref" if use_reference_demography else "monthly_births")
    age_aging_rate = _array(data, "age_aging_rate")
    contact_intensity = _array(data, "contact_intensity")
    periodic_multiplier = _array(data, "periodic_multiplier")
    seasonal_multiplier = _array(data, "seasonal_multiplier") if "seasonal_multiplier" in data else np.ones(len(year))
    pathogen_pressure_prior = _array(data, "pathogen_pressure_prior") if "pathogen_pressure_prior" in data else np.ones(len(year))
    age_rel_contact = _array(data, "age_rel_contact")
    age_symptom_prob = _array(data, "age_symptom_prob")
    age_report_weight = _array(data, "age_report_weight")
    age_program_reach = _array(data, "age_program_reach")
    routine_dtp1_eligibility = np.asarray(
        data.get("routine_dtp1_eligibility", DEFAULT_ROUTINE_DTP1_ELIGIBILITY[: len(age_program_reach)]),
        dtype=float,
    )
    routine_dtp3_eligibility = np.asarray(
        data.get("routine_dtp3_eligibility", DEFAULT_ROUTINE_DTP3_ELIGIBILITY[: len(age_program_reach)]),
        dtype=float,
    )
    routine_delay_age_weight = np.asarray(
        data.get("routine_delay_age_weight", DEFAULT_ROUTINE_DELAY_AGE_WEIGHT[: len(age_program_reach)]),
        dtype=float,
    )
    routine_delay_sensitivity = float(data.get("routine_delay_sensitivity", DEFAULT_ROUTINE_DELAY_SENSITIVITY))
    age_waning_years = _array(data, "age_waning_years")
    age_contact_matrix = np.asarray(
        data.get("age_contact_matrix", np.outer(age_rel_contact, age_rel_contact)),
        dtype=float,
    )
    booster_reach_modifier = np.asarray(data.get("booster_reach_modifier", np.ones(len(age_rel_contact))), dtype=float)
    vaccine_waning_age_modifier = np.asarray(
        data.get("vaccine_waning_age_modifier", np.ones(len(age_rel_contact))),
        dtype=float,
    )

    beta = float(np.exp(_draw_value(draw, "log_beta", np.log(0.01))))
    reporting_rate = float(np.exp(_draw_value(draw, "log_reporting_rate", np.log(0.06))))
    contact_sensitivity_upper = float(np.clip(float(data.get("contact_sensitivity_upper", 5.0)), 5.0, 10.0))
    contact_sensitivity = float(np.clip(_draw_value(draw, "contact_sensitivity", 1.0), 0.0, contact_sensitivity_upper))
    waning_rate = _draw_value(draw, "waning_rate", 0.10)
    post_transmission_multiplier = float(np.exp(_draw_value(draw, "log_post_transmission_multiplier", 0.0)))
    post_reporting_multiplier = float(np.exp(_draw_value(draw, "log_post_reporting_multiplier", 0.0)))
    rebound_peak_month = _draw_value(draw, "rebound_peak_month", float(data.get("rebound_peak_month_prior_mean", 24.0)))
    rebound_width = max(_draw_value(draw, "rebound_width", 8.0), 1e-6)
    rebound_asymmetry = _draw_value(draw, "rebound_asymmetry", 0.0)
    rebound_release_gain = _draw_value(draw, "rebound_release_gain", 1.25)
    secondary_rebound_delay = _draw_value(draw, "secondary_rebound_delay", 9.0)
    secondary_rebound_width = max(_draw_value(draw, "secondary_rebound_width", 5.0), 1e-6)
    secondary_rebound_release_gain = _draw_value(draw, "secondary_rebound_release_gain", 0.35)
    outbreak_seeding_gain = _draw_value(
        draw,
        "outbreak_ignition_pulse_gain",
        _draw_value(draw, "outbreak_seeding_gain", 0.03),
    )
    outbreak_seed_basis_gain = np.asarray(
        [_draw_value(draw, f"outbreak_seed_basis_gain[{index}]", 0.0) for index in range(1, 5)],
        dtype=float,
    )
    use_exposure_interruption = bool(int(data.get("use_exposure_interruption", 1)))
    use_post_2023_waning = bool(int(data.get("use_post_2023_waning", 1)))
    use_reporting_shift = bool(int(data.get("use_reporting_shift", 1)))
    use_flexible_outbreak_seeding = bool(int(data.get("use_flexible_outbreak_seeding", 0)))
    post_reporting_peak_gain = _draw_value(
        draw,
        "post_reporting_peak_gain",
        max(_draw_value(draw, "post_reporting_peak_multiplier", 1.0) - 1.0, 0.0),
    )
    post_reporting_trend = _draw_value(
        draw,
        "post_reporting_trend_per_year",
        _draw_value(draw, "post_reporting_trend", 0.0),
    )
    seasonal_strength = _draw_value(draw, "seasonal_strength", 1.0)
    natural_boosting_efficiency = _draw_value(draw, "natural_boosting_efficiency", 0.60)
    pathogen_pressure_scale = _draw_value(
        draw,
        "excess_epidemic_pressure_scale",
        _draw_value(draw, "pathogen_pressure_scale", 1.0),
    )
    epidemic_feedback_gain = _draw_value(draw, "epidemic_feedback_gain", 4.0)
    endemic_reseed_pressure = _draw_value(
        draw,
        "endemic_background_force",
        _draw_value(draw, "endemic_reseed_pressure", 0.01),
    )
    epidemic_pressure_retention = _draw_value(draw, "epidemic_pressure_retention", 0.45)
    nb_phi_month = max(_draw_value(draw, "nb_phi_month", _draw_value(draw, "nb_phi", 20.0)), 1e-6)

    if "booster_reach" in neutralized:
        booster_reach_modifier = np.maximum(booster_reach_modifier, 1.0)
    if "post_2023_transmission_context" in neutralized:
        post_transmission_multiplier = 1.0
        rebound_release_gain = 0.0
        secondary_rebound_release_gain = 0.0
        pathogen_pressure_scale = 0.0
        outbreak_seeding_gain = 0.0
    if "post_transmission_multiplier_only" in neutralized:
        post_transmission_multiplier = 1.0
    if "excess_epidemic_pressure_only" in neutralized:
        pathogen_pressure_scale = 0.0
    if "outbreak_ignition_pulse_only" in neutralized:
        rebound_release_gain = 0.0
        secondary_rebound_release_gain = 0.0
        outbreak_seeding_gain = 0.0
    if "reporting_testing_shift" in neutralized:
        post_reporting_multiplier = 1.0
        post_reporting_peak_gain = 0.0
        post_reporting_trend = 0.0

    base_ve_infection = float(data["base_ve_infection"])
    base_ve_disease = float(data["base_ve_disease"])
    init_natural_share = float(
        np.clip(
            _draw_value(draw, "init_natural_share", float(data.get("init_natural_share_prior_mean", 0.08))),
            0.0,
            0.50,
        )
    )
    init_vaccine_recent_fraction = float(
        np.clip(
            _draw_value(
                draw,
                "init_vaccine_recent_fraction",
                float(data.get("init_vaccine_recent_fraction_prior_mean", 0.35)),
            ),
            0.05,
            0.95,
        )
    )
    init_natural_recent_fraction = float(
        np.clip(
            _draw_value(
                draw,
                "init_natural_recent_fraction",
                float(data.get("init_natural_recent_fraction_prior_mean", 0.40)),
            ),
            0.05,
            0.95,
        )
    )

    n_age = int(data["N_age"])
    total_true = np.zeros(len(year), dtype=float)
    total_symptomatic = np.zeros(len(year), dtype=float)
    total_reported = np.zeros(len(year), dtype=float)
    total_symptomatic_infant = np.zeros(len(year), dtype=float)
    reported_by_age = np.zeros((len(year), int(data["N_age"])), dtype=float) if include_age_predictions else None
    total_true_under5 = np.zeros(len(year), dtype=float)
    total_symptomatic_under5 = np.zeros(len(year), dtype=float)
    total_reported_under5 = np.zeros(len(year), dtype=float)
    total_true_school_age = np.zeros(len(year), dtype=float)
    total_symptomatic_school_age = np.zeros(len(year), dtype=float)
    total_reported_school_age = np.zeros(len(year), dtype=float)
    total_true_adult = np.zeros(len(year), dtype=float)
    total_symptomatic_adult = np.zeros(len(year), dtype=float)
    total_reported_adult = np.zeros(len(year), dtype=float)
    total_effective_susceptible = np.zeros(len(year), dtype=float)
    total_effective_susceptible_under5 = np.zeros(len(year), dtype=float)
    total_effective_susceptible_school_age = np.zeros(len(year), dtype=float)
    total_effective_susceptible_adult = np.zeros(len(year), dtype=float)
    total_natural_immunity = np.zeros(len(year), dtype=float)
    total_waned_natural_immunity = np.zeros(len(year), dtype=float)
    total_natural_immunity_under5 = np.zeros(len(year), dtype=float)
    total_waned_natural_immunity_under5 = np.zeros(len(year), dtype=float)
    total_natural_immunity_school_age = np.zeros(len(year), dtype=float)
    total_waned_natural_immunity_school_age = np.zeros(len(year), dtype=float)
    total_natural_immunity_adult = np.zeros(len(year), dtype=float)
    total_waned_natural_immunity_adult = np.zeros(len(year), dtype=float)
    total_partial_vaccine_immunity = np.zeros(len(year), dtype=float)
    total_exposure_boosting = np.zeros(len(year), dtype=float)
    total_natural_recharge = np.zeros(len(year), dtype=float)
    total_routine_immunization_shortfall = np.zeros(len(year), dtype=float)
    total_routine_delay_debt = np.zeros(len(year), dtype=float)
    natural_waning_month = 0.010
    deep_natural_waning_month = 0.003
    natural_exposure_contact_sensitivity = float(
        np.clip(float(data.get("natural_exposure_contact_sensitivity", 3.0)), 0.0, 10.0)
    )
    reported_baseline_monthly_risk = max(float(data.get("pre_mean_incidence_per_100k", 0.0)), 0.0) / 100000.0 / 12.0
    silent_exposure_multiplier = float(np.clip(1.0 / max(reporting_rate, 0.01), 4.0, 40.0))
    reported_exposure_floor = reported_baseline_monthly_risk * silent_exposure_multiplier * 10.0
    silent_exposure_floor = max(float(data.get("silent_exposure_floor_monthly", 0.0)), 0.0)
    background_exposure_floor = float(np.clip(max(silent_exposure_floor, reported_exposure_floor), 0.0, 0.05))
    exposed_progression = min(0.95, max(0.55, 1.0 - 0.50 * epidemic_pressure_retention))
    infectious_clearance = min(0.95, max(0.35, 1.0 - 0.65 * epidemic_pressure_retention))
    pop0 = np.maximum(age_population[0, :], 1.0)
    dtp1_share = np.clip(dtp1[0] * age_program_reach * booster_reach_modifier, 0.0, 0.97)
    dtp3_share = np.clip(dtp3[0] * age_program_reach * booster_reach_modifier, 0.0, 0.92)
    partial_vaccine_share = np.maximum(0.0, dtp1_share - dtp3_share)
    natural_share = np.clip(init_natural_share, 0.0, 0.50)
    vaccine_recent = pop0 * dtp3_share * init_vaccine_recent_fraction
    vaccine_waned = pop0 * dtp3_share * (1.0 - init_vaccine_recent_fraction)
    vaccine_partial_recent = pop0 * partial_vaccine_share * init_vaccine_recent_fraction
    vaccine_partial_waned = pop0 * partial_vaccine_share * (1.0 - init_vaccine_recent_fraction)
    natural_recent = pop0 * natural_share * init_natural_recent_fraction
    natural_waned = pop0 * natural_share * (1.0 - init_natural_recent_fraction)
    exposed = pop0 * 1e-8
    infectious = pop0 * 1e-8
    routine_delay_debt = np.zeros(n_age, dtype=float)
    pre_pandemic_boosting_reference: dict[tuple[int, int], list[float]] = {}
    susceptible_naive = np.maximum(
        0.0,
        pop0
        - vaccine_recent
        - vaccine_waned
        - vaccine_partial_recent
        - vaccine_partial_waned
        - natural_recent
        - natural_waned
        - exposed
        - infectious,
    )
    contact_weight_sum = max(float(np.sum(age_rel_contact)), 1e-9)

    for m in range(len(year)):
        year_m = int(year[m])
        month_m = int(month[m])
        observed_contact_multiplier = 1.0
        if use_exposure_interruption and float(contact_intensity[m]) > 0.0:
            observed_contact_multiplier = 0.03 + 0.97 * np.exp(
                -contact_sensitivity * float(contact_intensity[m]) / 100.0
            )
        observed_boosting_contact_multiplier = 1.0
        if use_exposure_interruption and float(contact_intensity[m]) > 0.0:
            observed_boosting_contact_multiplier = 0.03 + 0.97 * np.exp(
                -natural_exposure_contact_sensitivity * float(contact_intensity[m]) / 100.0
            )
        natural_boosting_repair_window = (
            NATURAL_BOOST_REPAIR_START_YEAR <= year_m <= NATURAL_BOOST_REPAIR_END_YEAR
        )
        vaccine_waning_neutralized = year_m >= VACCINE_WANING_NEUTRALIZATION_START_YEAR
        vaccine_waning_neutralized = vaccine_waning_neutralized and "vaccine_waning" in neutralized
        month_waning_rate = 0.0 if vaccine_waning_neutralized else waning_rate
        restore_infection_contact = (
            "contact_suppression" in neutralized and natural_boosting_repair_window
        ) or (
            "pre_2023_contact_suppression" in neutralized and natural_boosting_repair_window
        )
        restore_boosting_contact = (
            "missed_exposure_boosting" in neutralized and natural_boosting_repair_window
        ) or (
            "pre_2023_missed_exposure_boosting" in neutralized and natural_boosting_repair_window
        )
        infection_contact_multiplier = 1.0 if restore_infection_contact else observed_contact_multiplier
        boosting_contact_multiplier = 1.0 if restore_boosting_contact else observed_boosting_contact_multiplier
        dtp1_eff = float(dtp1[m])
        dtp3_eff = float(dtp3[m])
        if "routine_immunization_disruption" in neutralized and year_m >= 2020:
            dtp1_eff = max(dtp1_eff, pre_mean_dtp1)
            dtp3_eff = max(dtp3_eff, pre_mean_dtp3)
        dtp1_eff = max(dtp1_eff, dtp3_eff)
        primary_release_shape = 0.0
        secondary_release_shape = 0.0
        reporting_peak_shape = 0.0
        reporting_trend_multiplier = 1.0
        if year_m >= 2023:
            post_month = (year_m - 2023) * 12 + month_m
            primary_width = (
                rebound_width * np.exp(-0.5 * rebound_asymmetry)
                if post_month < rebound_peak_month
                else rebound_width * np.exp(rebound_asymmetry)
            )
            secondary_peak_month = min(48.0, rebound_peak_month + secondary_rebound_delay)
            primary_release_shape = float(np.exp(-0.5 * ((post_month - rebound_peak_month) / primary_width) ** 2))
            secondary_release_shape = float(
                np.exp(-0.5 * ((post_month - secondary_peak_month) / secondary_rebound_width) ** 2)
            )
            reporting_peak_width = max(0.75, 0.55 * rebound_width)
            reporting_peak_shape = float(np.exp(-0.5 * ((post_month - rebound_peak_month) / reporting_peak_width) ** 2))
            reporting_trend_multiplier = float(np.exp(post_reporting_trend * (post_month - 1.0) / 12.0))
        seasonal_force = float(np.power(max(float(seasonal_multiplier[m]), 0.01), seasonal_strength))
        pathogen_multiplier = float(
            np.clip(1.0 + pathogen_pressure_scale * (float(pathogen_pressure_prior[m]) - 1.0), 0.40, 6.0)
        )

        next_susceptible_naive = np.zeros(n_age, dtype=float)
        next_vaccine_recent = np.zeros(n_age, dtype=float)
        next_vaccine_waned = np.zeros(n_age, dtype=float)
        next_vaccine_partial_recent = np.zeros(n_age, dtype=float)
        next_vaccine_partial_waned = np.zeros(n_age, dtype=float)
        next_natural_recent = np.zeros(n_age, dtype=float)
        next_natural_waned = np.zeros(n_age, dtype=float)
        next_exposed = np.zeros(n_age, dtype=float)
        next_infectious = np.zeros(n_age, dtype=float)
        next_routine_delay_debt = np.zeros(n_age, dtype=float)
        for a in range(n_age):
            aging_out = float(np.clip(age_aging_rate[a], 0.0, 0.95))
            stay = 1.0 - aging_out
            next_susceptible_naive[a] += susceptible_naive[a] * stay
            next_vaccine_recent[a] += vaccine_recent[a] * stay
            next_vaccine_waned[a] += vaccine_waned[a] * stay
            next_vaccine_partial_recent[a] += vaccine_partial_recent[a] * stay
            next_vaccine_partial_waned[a] += vaccine_partial_waned[a] * stay
            next_natural_recent[a] += natural_recent[a] * stay
            next_natural_waned[a] += natural_waned[a] * stay
            next_exposed[a] += exposed[a] * stay
            next_infectious[a] += infectious[a] * stay
            next_routine_delay_debt[a] += routine_delay_debt[a] * stay
            if a < n_age - 1:
                next_susceptible_naive[a + 1] += susceptible_naive[a] * aging_out
                next_vaccine_recent[a + 1] += vaccine_recent[a] * aging_out
                next_vaccine_waned[a + 1] += vaccine_waned[a] * aging_out
                next_vaccine_partial_recent[a + 1] += vaccine_partial_recent[a] * aging_out
                next_vaccine_partial_waned[a + 1] += vaccine_partial_waned[a] * aging_out
                next_natural_recent[a + 1] += natural_recent[a] * aging_out
                next_natural_waned[a + 1] += natural_waned[a] * aging_out
                next_exposed[a + 1] += exposed[a] * aging_out
                next_infectious[a + 1] += infectious[a] * aging_out
                next_routine_delay_debt[a + 1] += routine_delay_debt[a] * aging_out
        next_susceptible_naive[0] += float(monthly_births[m])
        susceptible_naive = next_susceptible_naive
        vaccine_recent = next_vaccine_recent
        vaccine_waned = next_vaccine_waned
        vaccine_partial_recent = next_vaccine_partial_recent
        vaccine_partial_waned = next_vaccine_partial_waned
        natural_recent = next_natural_recent
        natural_waned = next_natural_waned
        exposed = next_exposed
        infectious = next_infectious
        routine_delay_debt = next_routine_delay_debt

        for a in range(n_age):
            target_population = max(float(age_population[m, a]), 1.0)
            state_total = max(
                float(
                    susceptible_naive[a]
                    + vaccine_recent[a]
                    + vaccine_waned[a]
                    + vaccine_partial_recent[a]
                    + vaccine_partial_waned[a]
                    + natural_recent[a]
                    + natural_waned[a]
                    + exposed[a]
                    + infectious[a]
                ),
                1.0,
            )
            demographic_adjustment = target_population / state_total
            susceptible_naive[a] *= demographic_adjustment
            vaccine_recent[a] *= demographic_adjustment
            vaccine_waned[a] *= demographic_adjustment
            vaccine_partial_recent[a] *= demographic_adjustment
            vaccine_partial_waned[a] *= demographic_adjustment
            natural_recent[a] *= demographic_adjustment
            natural_waned[a] *= demographic_adjustment
            exposed[a] *= demographic_adjustment
            infectious[a] *= demographic_adjustment
            routine_delay_debt[a] = min(0.50 * target_population, routine_delay_debt[a] * demographic_adjustment)

        state_population = np.maximum(
            susceptible_naive
            + vaccine_recent
            + vaccine_waned
            + vaccine_partial_recent
            + vaccine_partial_waned
            + natural_recent
            + natural_waned
            + exposed
            + infectious,
            1.0,
        )
        infectious_fraction = (infectious + 0.35 * exposed) / state_population
        age_infectious_pressure = np.asarray(age_contact_matrix @ infectious_fraction, dtype=float) / contact_weight_sum
        transmission_multiplier = post_transmission_multiplier if year_m >= 2023 else 1.0
        reporting_multiplier = (
            post_reporting_multiplier * (1.0 + post_reporting_peak_gain * reporting_peak_shape) * reporting_trend_multiplier
            if use_reporting_shift and year_m >= 2023
            else 1.0
        )

        for a in range(n_age):
            target_population = max(float(age_population[m, a]), 1.0)
            target_started_share = float(np.clip(dtp1_eff * age_program_reach[a] * booster_reach_modifier[a], 0.0, 0.97))
            target_completed_share = float(np.clip(dtp3_eff * age_program_reach[a] * booster_reach_modifier[a], 0.0, 0.95))
            dtp1_eligible_count = max(float(monthly_births[m]), 0.0) * float(routine_dtp1_eligibility[a])
            dtp3_eligible_count = max(float(monthly_births[m]), 0.0) * float(routine_dtp3_eligibility[a])
            target_started_count = dtp1_eff * dtp1_eligible_count
            target_completed_count = dtp3_eff * dtp3_eligible_count
            full_vaccine_count = vaccine_recent[a] + vaccine_waned[a]
            partial_vaccine_count = vaccine_partial_recent[a] + vaccine_partial_waned[a]
            initiation_flow = min(
                susceptible_naive[a],
                target_started_count,
            )
            susceptible_naive[a] -= initiation_flow
            vaccine_partial_recent[a] += initiation_flow
            partial_vaccine_count += initiation_flow
            completion_need = target_completed_count
            completion_flow = min(partial_vaccine_count, completion_need)
            if completion_flow > 0.0 and partial_vaccine_count > 0.0:
                completion_from_recent = completion_flow * vaccine_partial_recent[a] / partial_vaccine_count
                completion_from_waned = completion_flow * vaccine_partial_waned[a] / partial_vaccine_count
                vaccine_partial_recent[a] = max(0.0, vaccine_partial_recent[a] - completion_from_recent)
                vaccine_partial_waned[a] = max(0.0, vaccine_partial_waned[a] - completion_from_waned)
                vaccine_recent[a] += completion_flow
                full_vaccine_count += completion_flow
            catchup_completion_flow = min(
                susceptible_naive[a],
                max(0.0, target_completed_count - completion_flow) * 0.50,
            )
            susceptible_naive[a] -= catchup_completion_flow
            vaccine_recent[a] += catchup_completion_flow
            if 2020 <= year_m <= 2024:
                reference_started_count = max(pre_mean_dtp1, pre_mean_dtp3) * dtp1_eligible_count
                reference_completed_count = pre_mean_dtp3 * dtp3_eligible_count
                dtp1_shortfall = max(0.0, reference_started_count - target_started_count)
                dtp3_shortfall = max(0.0, reference_completed_count - target_completed_count)
                total_routine_immunization_shortfall[m] += (
                    dtp1_shortfall
                    + dtp3_shortfall
                )
                routine_delay_debt[a] = min(
                    0.50 * target_population,
                    routine_delay_debt[a] + 0.50 * dtp1_shortfall + dtp3_shortfall,
                )

            age_waning_rate = month_waning_rate * vaccine_waning_age_modifier[a]
            stock_waning_active = use_post_2023_waning or year_m < 2023
            vaccine_wane = (
                min(vaccine_recent[a], vaccine_recent[a] * age_waning_rate / 12.0)
                if stock_waning_active else 0.0
            )
            vaccine_recent[a] -= vaccine_wane
            vaccine_waned[a] += vaccine_wane
            partial_vaccine_wane = (
                min(vaccine_partial_recent[a], vaccine_partial_recent[a] * age_waning_rate * 1.20 / 12.0)
                if stock_waning_active else 0.0
            )
            vaccine_partial_recent[a] -= partial_vaccine_wane
            vaccine_partial_waned[a] += partial_vaccine_wane
            natural_wane = (
                min(natural_recent[a], natural_recent[a] * natural_waning_month)
                if stock_waning_active else 0.0
            )
            natural_recent[a] -= natural_wane
            natural_waned[a] += natural_wane
            deep_natural_wane = (
                min(natural_waned[a], natural_waned[a] * deep_natural_waning_month)
                if stock_waning_active else 0.0
            )
            natural_waned[a] -= deep_natural_wane
            susceptible_naive[a] += deep_natural_wane

            waning_multiplier = np.exp(-age_waning_rate * age_waning_years[a] / 8.0)
            recent_ve_infection = min(0.95, base_ve_infection * waning_multiplier)
            recent_ve_disease = min(0.98, base_ve_disease * waning_multiplier)
            susc_vaccine_recent = float(np.clip(1.0 - recent_ve_infection, 0.05, 1.0))
            susc_vaccine_waned = float(np.clip(1.0 - 0.45 * recent_ve_infection, 0.10, 1.0))
            susc_partial_recent = float(np.clip(1.0 - 0.45 * recent_ve_infection, 0.20, 1.0))
            susc_partial_waned = float(np.clip(1.0 - 0.20 * recent_ve_infection, 0.35, 1.0))
            symptom_naive = age_symptom_prob[a]
            symptom_vaccine_recent = age_symptom_prob[a] * float(np.clip(1.0 - 0.80 * recent_ve_disease, 0.12, 1.0))
            symptom_vaccine_waned = age_symptom_prob[a] * float(np.clip(1.0 - 0.40 * recent_ve_disease, 0.25, 1.0))
            symptom_partial_recent = age_symptom_prob[a] * float(np.clip(1.0 - 0.45 * recent_ve_disease, 0.40, 1.0))
            symptom_partial_waned = age_symptom_prob[a] * float(np.clip(1.0 - 0.20 * recent_ve_disease, 0.55, 1.0))
            if vaccine_waning_neutralized:
                susc_vaccine_waned = susc_vaccine_recent
                susc_partial_waned = susc_partial_recent
                symptom_vaccine_waned = symptom_vaccine_recent
                symptom_partial_waned = symptom_partial_recent
            symptom_natural_recent = age_symptom_prob[a] * 0.15
            symptom_natural_waned = age_symptom_prob[a] * 0.45
            routine_delay_fraction = min(0.40, routine_delay_debt[a] / target_population)
            routine_delay_multiplier = min(
                3.0,
                1.0 + routine_delay_sensitivity * routine_delay_age_weight[a] * routine_delay_fraction,
            )
            age_effective_susceptible = (
                susceptible_naive[a]
                + vaccine_recent[a] * susc_vaccine_recent
                + vaccine_waned[a] * susc_vaccine_waned
                + vaccine_partial_recent[a] * susc_partial_recent
                + vaccine_partial_waned[a] * susc_partial_waned
                + natural_recent[a] * 0.08
                + natural_waned[a] * 0.35
            )
            routine_delay_risk_base = (
                susceptible_naive[a]
                + vaccine_partial_recent[a] * susc_partial_recent
                + vaccine_partial_waned[a] * susc_partial_waned
            )
            age_effective_susceptible += routine_delay_risk_base * max(0.0, routine_delay_multiplier - 1.0)
            total_effective_susceptible[m] += age_effective_susceptible
            if a <= UNDER5_INDEX_MAX:
                total_effective_susceptible_under5[m] += age_effective_susceptible
            if SCHOOL_AGE_INDEX_MIN <= a <= SCHOOL_AGE_INDEX_MAX:
                total_effective_susceptible_school_age[m] += age_effective_susceptible
            if a >= ADULT_INDEX_MIN:
                total_effective_susceptible_adult[m] += age_effective_susceptible
            total_routine_delay_debt[m] += routine_delay_debt[a]
            total_partial_vaccine_immunity[m] += vaccine_partial_recent[a] + vaccine_partial_waned[a]
            total_natural_immunity[m] += natural_recent[a] + natural_waned[a]
            total_waned_natural_immunity[m] += natural_waned[a]
            if a <= UNDER5_INDEX_MAX:
                total_natural_immunity_under5[m] += natural_recent[a] + natural_waned[a]
                total_waned_natural_immunity_under5[m] += natural_waned[a]
            if SCHOOL_AGE_INDEX_MIN <= a <= SCHOOL_AGE_INDEX_MAX:
                total_natural_immunity_school_age[m] += natural_recent[a] + natural_waned[a]
                total_waned_natural_immunity_school_age[m] += natural_waned[a]
            if a >= ADULT_INDEX_MIN:
                total_natural_immunity_adult[m] += natural_recent[a] + natural_waned[a]
                total_waned_natural_immunity_adult[m] += natural_waned[a]
            if year_m < 2023:
                outbreak_seeding_pressure = 0.0
            elif use_flexible_outbreak_seeding:
                basis_centres = np.asarray([6.0, 18.0, 30.0, 42.0])
                basis = np.exp(-0.5 * np.square((post_month - basis_centres) / 4.0))
                outbreak_seeding_pressure = float(np.sum(outbreak_seed_basis_gain * basis))
            else:
                outbreak_seeding_pressure = outbreak_seeding_gain * (
                    (1.0 + 0.35 * rebound_release_gain) * primary_release_shape
                    + secondary_rebound_release_gain * secondary_release_shape
                )
            transmission_pressure = (
                endemic_reseed_pressure + outbreak_seeding_pressure + epidemic_feedback_gain * age_infectious_pressure[a]
            )
            raw_lam = (
                beta
                * periodic_multiplier[m]
                * seasonal_force
                * pathogen_multiplier
                * transmission_multiplier
                * transmission_pressure
            )
            infection_lam = min(1.25, max(0.0, raw_lam * infection_contact_multiplier))
            boosting_lam = min(1.25, max(0.0, raw_lam * boosting_contact_multiplier))
            if year_m <= 2019:
                pre_pandemic_boosting_reference.setdefault((month_m, a), []).append(boosting_lam)
            if "pre_2023_missed_exposure_boosting" in neutralized and natural_boosting_repair_window:
                normal_boosting_values = pre_pandemic_boosting_reference.get((month_m, a), [])
                if normal_boosting_values:
                    normal_boosting_lam = float(np.median(normal_boosting_values))
                    boosting_lam = max(boosting_lam, normal_boosting_lam)
            new_naive = susceptible_naive[a] * (1.0 - np.exp(-infection_lam * routine_delay_multiplier))
            new_vaccine_recent = vaccine_recent[a] * (1.0 - np.exp(-infection_lam * susc_vaccine_recent))
            new_vaccine_waned = vaccine_waned[a] * (1.0 - np.exp(-infection_lam * susc_vaccine_waned))
            new_partial_recent = vaccine_partial_recent[a] * (
                1.0 - np.exp(-infection_lam * susc_partial_recent * routine_delay_multiplier)
            )
            new_partial_waned = vaccine_partial_waned[a] * (
                1.0 - np.exp(-infection_lam * susc_partial_waned * routine_delay_multiplier)
            )
            new_natural_recent = natural_recent[a] * (1.0 - np.exp(-infection_lam * 0.08))
            new_natural_waned = natural_waned[a] * (1.0 - np.exp(-infection_lam * 0.35))
            available_vaccine_waned = max(0.0, vaccine_waned[a] - new_vaccine_waned)
            available_partial_waned = max(0.0, vaccine_partial_waned[a] - new_partial_waned)
            available_natural_waned = max(0.0, natural_waned[a] - new_natural_waned)
            boost_vaccine_waned = min(
                available_vaccine_waned,
                natural_boosting_efficiency * boosting_lam * available_vaccine_waned,
            )
            boost_partial_waned = min(
                available_partial_waned,
                natural_boosting_efficiency * boosting_lam * available_partial_waned,
            )
            boost_natural_waned = min(
                available_natural_waned,
                natural_boosting_efficiency * boosting_lam * available_natural_waned,
            )
            recharge_driver_lam = max(
                boosting_lam * silent_exposure_multiplier,
                background_exposure_floor * boosting_contact_multiplier,
            )
            recharge_lam = min(0.35, natural_boosting_efficiency * recharge_driver_lam)
            remaining_naive = max(0.0, susceptible_naive[a] - new_naive)
            remaining_vaccine_recent = max(0.0, vaccine_recent[a] - new_vaccine_recent)
            remaining_vaccine_waned = max(0.0, available_vaccine_waned - boost_vaccine_waned)
            remaining_partial_recent = max(0.0, vaccine_partial_recent[a] - new_partial_recent)
            remaining_partial_waned = max(0.0, available_partial_waned - boost_partial_waned)
            remaining_natural_waned = max(0.0, available_natural_waned - boost_natural_waned)
            recharge_naive = min(remaining_naive, 0.35 * recharge_lam * remaining_naive)
            recharge_vaccine_recent = min(remaining_vaccine_recent, 0.15 * recharge_lam * remaining_vaccine_recent)
            recharge_vaccine_waned = min(remaining_vaccine_waned, 0.70 * recharge_lam * remaining_vaccine_waned)
            recharge_partial_recent = min(remaining_partial_recent, 0.25 * recharge_lam * remaining_partial_recent)
            recharge_partial_waned = min(remaining_partial_waned, 0.80 * recharge_lam * remaining_partial_waned)
            recharge_natural_waned = min(remaining_natural_waned, recharge_lam * remaining_natural_waned)
            natural_recharge = (
                recharge_naive
                + recharge_vaccine_recent
                + recharge_vaccine_waned
                + recharge_partial_recent
                + recharge_partial_waned
                + recharge_natural_waned
            )
            true_cases = (
                new_naive
                + new_vaccine_recent
                + new_vaccine_waned
                + new_partial_recent
                + new_partial_waned
                + new_natural_recent
                + new_natural_waned
            )
            symptomatic = (
                new_naive * symptom_naive
                + new_vaccine_recent * symptom_vaccine_recent
                + new_vaccine_waned * symptom_vaccine_waned
                + new_partial_recent * symptom_partial_recent
                + new_partial_waned * symptom_partial_waned
                + new_natural_recent * symptom_natural_recent
                + new_natural_waned * symptom_natural_waned
            )
            reported = symptomatic * reporting_rate * reporting_multiplier * age_report_weight[a]
            total_true[m] += true_cases
            total_symptomatic[m] += symptomatic
            total_reported[m] += reported
            if a < 2:  # 0–2 and 3–11 months; do not include the 1–4 year group.
                total_symptomatic_infant[m] += symptomatic
            if reported_by_age is not None:
                reported_by_age[m, a] = reported
            if a <= UNDER5_INDEX_MAX:
                total_true_under5[m] += true_cases
                total_symptomatic_under5[m] += symptomatic
                total_reported_under5[m] += reported
            if SCHOOL_AGE_INDEX_MIN <= a <= SCHOOL_AGE_INDEX_MAX:
                total_true_school_age[m] += true_cases
                total_symptomatic_school_age[m] += symptomatic
                total_reported_school_age[m] += reported
            if a >= ADULT_INDEX_MIN:
                total_true_adult[m] += true_cases
                total_symptomatic_adult[m] += symptomatic
                total_reported_adult[m] += reported
            total_exposure_boosting[m] += boost_vaccine_waned + boost_partial_waned + boost_natural_waned
            total_natural_recharge[m] += natural_recharge

            exposed_old = exposed[a]
            infectious_old = infectious[a]
            progressed_exposed = exposed_progression * exposed_old
            cleared_infectious = infectious_clearance * infectious_old
            same_month_progressed = 0.35 * true_cases
            susceptible_naive[a] = max(0.0, susceptible_naive[a] - new_naive - recharge_naive)
            vaccine_recent[a] = max(0.0, vaccine_recent[a] - new_vaccine_recent - recharge_vaccine_recent)
            vaccine_waned[a] = max(0.0, vaccine_waned[a] - new_vaccine_waned - boost_vaccine_waned - recharge_vaccine_waned)
            vaccine_partial_recent[a] = max(0.0, vaccine_partial_recent[a] - new_partial_recent - recharge_partial_recent)
            vaccine_partial_waned[a] = max(
                0.0,
                vaccine_partial_waned[a] - new_partial_waned - boost_partial_waned - recharge_partial_waned,
            )
            natural_recent[a] = max(
                0.0,
                natural_recent[a]
                - new_natural_recent
                + cleared_infectious
                + boost_vaccine_waned
                + boost_partial_waned
                + boost_natural_waned
                + natural_recharge,
            )
            natural_waned[a] = max(0.0, natural_waned[a] - new_natural_waned - boost_natural_waned - recharge_natural_waned)
            exposed[a] = max(0.0, exposed_old - progressed_exposed + true_cases - same_month_progressed)
            infectious[a] = max(0.0, infectious_old - cleared_infectious + progressed_exposed + same_month_progressed)

    if totals_only:
        period = (year >= 2023) & (year <= 2025)
        return {
            "true_infections": float(total_true[period].sum()),
            "symptomatic_infections": float(total_symptomatic[period].sum()),
            "reported_cases": float(total_reported[period].sum()),
        }
    observed = {int(idx): int(value) for idx, value in zip(data.get("month_obs_idx", []), data.get("y_month", []))}
    observed.update({int(idx): int(value) for idx, value in zip(data.get("month_ppc_obs_idx", []), data.get("y_month_ppc", []))})
    out = pd.DataFrame(
        {
            "year": year,
            "month": month,
            "observed_cases": [observed.get(index + 1, np.nan) for index in range(len(year))],
            "true_infections_predicted": total_true,
            "symptomatic_infections_predicted": total_symptomatic,
            "symptomatic_infections_infant_predicted": total_symptomatic_infant,
            "reported_cases_predicted": total_reported,
            "true_infections_under5_predicted": total_true_under5,
            "symptomatic_infections_under5_predicted": total_symptomatic_under5,
            "reported_cases_under5_predicted": total_reported_under5,
            "true_infections_school_age_predicted": total_true_school_age,
            "symptomatic_infections_school_age_predicted": total_symptomatic_school_age,
            "reported_cases_school_age_predicted": total_reported_school_age,
            "true_infections_adult_predicted": total_true_adult,
            "symptomatic_infections_adult_predicted": total_symptomatic_adult,
            "reported_cases_adult_predicted": total_reported_adult,
            "effective_susceptible_predicted": total_effective_susceptible,
            "effective_susceptible_under5_predicted": total_effective_susceptible_under5,
            "effective_susceptible_school_age_predicted": total_effective_susceptible_school_age,
            "effective_susceptible_adult_predicted": total_effective_susceptible_adult,
            "natural_immunity_predicted": total_natural_immunity,
            "waned_natural_immunity_predicted": total_waned_natural_immunity,
            "natural_immunity_under5_predicted": total_natural_immunity_under5,
            "waned_natural_immunity_under5_predicted": total_waned_natural_immunity_under5,
            "natural_immunity_school_age_predicted": total_natural_immunity_school_age,
            "waned_natural_immunity_school_age_predicted": total_waned_natural_immunity_school_age,
            "natural_immunity_adult_predicted": total_natural_immunity_adult,
            "waned_natural_immunity_adult_predicted": total_waned_natural_immunity_adult,
            "partial_vaccine_immunity_predicted": total_partial_vaccine_immunity,
            "exposure_boosting_events_predicted": total_exposure_boosting,
            "natural_exposure_recharge_predicted": total_natural_recharge,
            "routine_immunization_shortfall_predicted": total_routine_immunization_shortfall,
            "routine_delay_debt_predicted": total_routine_delay_debt,
        }
    )
    if reported_by_age is not None:
        for age_index in range(reported_by_age.shape[1]):
            out[f"reported_cases_age_{age_index + 1}_predicted"] = reported_by_age[:, age_index]
    if include_predictive_replicate:
        draw_id = int(_draw_value(draw, "draw", 1.0))
        seed = int((20260617 + 1009 * draw_id + int(np.round(total_reported.sum())) % 1_000_000) % (2**32 - 1))
        rng = np.random.default_rng(seed)
        mean_reported = np.maximum(total_reported, 0.0)
        poisson_rate = rng.gamma(shape=nb_phi_month, scale=mean_reported / nb_phi_month)
        out["reported_cases_predictive"] = rng.poisson(poisson_rate).astype(float)
    return out


def _select_scenario_draws(draws: pd.DataFrame, max_draws: int | None) -> pd.DataFrame:
    if draws.empty:
        return draws.copy()
    if max_draws is None:
        max_draws = _env_int("MODEL2C_SCENARIO_MAX_DRAWS", 160)
    max_draws = max(1, int(max_draws))
    if len(draws) <= max_draws:
        return draws.copy()
    positions = np.unique(np.linspace(0, len(draws) - 1, max_draws, dtype=int))
    return draws.iloc[positions].copy()


def _first_non_missing(values: pd.Series) -> float:
    observed = pd.to_numeric(values, errors="coerce").dropna()
    return float(observed.iloc[0]) if not observed.empty else np.nan


def _quantile(probability: float):
    def summarise(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return float(numeric.quantile(probability)) if not numeric.empty else np.nan

    return summarise


def _mode_month(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    return clean.value_counts().sort_values(ascending=False).index[0]


def _prepare_scenario_draws(draws: pd.DataFrame, max_draws: int | None) -> pd.DataFrame:
    selected = _select_scenario_draws(draws, max_draws).copy()
    if selected.empty:
        return selected
    fallback_draw_ids = pd.Series(np.arange(1, len(selected) + 1), index=selected.index)
    if "draw" not in selected.columns:
        selected.insert(0, "draw", fallback_draw_ids.to_numpy(dtype=int))
    else:
        selected["draw"] = pd.to_numeric(selected["draw"], errors="coerce")
        missing_draw = selected["draw"].isna()
        selected.loc[missing_draw, "draw"] = fallback_draw_ids.loc[missing_draw]
        selected["draw"] = selected["draw"].astype(int)
    return selected


def _chunk_frame(frame: pd.DataFrame, workers: int) -> list[pd.DataFrame]:
    chunk_size = max(1, (len(frame) + workers - 1) // workers)
    return [frame.iloc[start : start + chunk_size].copy() for start in range(0, len(frame), chunk_size)]


def _scenario_prediction_frame_from_selected(
    data: dict[str, Any],
    metadata: dict[str, Any],
    selected: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    include_supplement_context_audit: bool,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    predictions: list[pd.DataFrame] = []
    scenarios = _scenario_definitions(include_supplement_context_audit=include_supplement_context_audit)
    for _, draw in selected.iterrows():
        draw_series = draw.copy()
        draw_id = int(draw_series["draw"])
        for scenario_id, scenario in scenarios.items():
            pred = _predict_monthly(data, draw_series, scenario_id=scenario_id)
            pred = pred.loc[pred["year"].between(start_year, end_year)].copy()
            pred.insert(0, "draw", draw_id)
            pred.insert(0, "scenario_id", scenario_id)
            pred["scenario_label"] = scenario["scenario_label"]
            pred["scenario_family"] = scenario["scenario_family"]
            pred["scenario_order"] = int(scenario["scenario_order"])
            pred["neutralized_mechanisms"] = ";".join(scenario["neutralized_mechanisms"]) or "none"
            pred["scenario_interpretation_note"] = scenario["scenario_interpretation_note"]
            predictions.append(pred)
    frame = pd.concat(predictions, ignore_index=True, sort=False)
    frame.insert(0, "country", metadata["country"])
    frame.insert(0, "iso3", metadata["iso3"])
    frame["date"] = pd.to_datetime(
        {"year": frame["year"].astype(int), "month": frame["month"].astype(int), "day": 15}
    ).dt.strftime("%Y-%m-%d")
    frame["fit_role"] = np.where(frame["year"].le(2025), "fit_likelihood", "posterior_predictive_check")
    return frame


def _scenario_prediction_frame_worker(
    payload: tuple[dict[str, Any], dict[str, Any], pd.DataFrame, int, int, bool],
) -> pd.DataFrame:
    data, metadata, selected, start_year, end_year, include_supplement_context_audit = payload
    return _scenario_prediction_frame_from_selected(
        data,
        metadata,
        selected,
        start_year=start_year,
        end_year=end_year,
        include_supplement_context_audit=include_supplement_context_audit,
    )


def _scenario_prediction_frame(
    data: dict[str, Any],
    metadata: dict[str, Any],
    draws: pd.DataFrame,
    *,
    max_draws: int | None = None,
    start_year: int = 2020,
    end_year: int = 2026,
    include_supplement_context_audit: bool = False,
    draw_workers: int | None = None,
) -> pd.DataFrame:
    selected = _prepare_scenario_draws(draws, max_draws)
    if selected.empty:
        return pd.DataFrame()
    workers = _scenario_draw_workers(len(selected), requested=draw_workers)
    if workers <= 1:
        return _scenario_prediction_frame_from_selected(
            data,
            metadata,
            selected,
            start_year=start_year,
            end_year=end_year,
            include_supplement_context_audit=include_supplement_context_audit,
        )
    chunks = _chunk_frame(selected, workers)
    payloads = [
        (data, metadata, chunk, start_year, end_year, include_supplement_context_audit)
        for chunk in chunks
        if not chunk.empty
    ]
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        parts = list(executor.map(_scenario_prediction_frame_worker, payloads))
    return pd.concat(parts, ignore_index=True, sort=False)


def _monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "true_infections_lower_95",
        "true_infections_upper_95",
        "symptomatic_infections_median",
        "symptomatic_infections_lower_95",
        "symptomatic_infections_upper_95",
        "reported_cases_median",
        "reported_cases_lower_95",
        "reported_cases_upper_95",
        "true_infections_under5_median",
        "true_infections_under5_lower_95",
        "true_infections_under5_upper_95",
        "symptomatic_infections_under5_median",
        "symptomatic_infections_under5_lower_95",
        "symptomatic_infections_under5_upper_95",
        "reported_cases_under5_median",
        "reported_cases_under5_lower_95",
        "reported_cases_under5_upper_95",
        "true_infections_school_age_median",
        "true_infections_school_age_lower_95",
        "true_infections_school_age_upper_95",
        "symptomatic_infections_school_age_median",
        "symptomatic_infections_school_age_lower_95",
        "symptomatic_infections_school_age_upper_95",
        "reported_cases_school_age_median",
        "reported_cases_school_age_lower_95",
        "reported_cases_school_age_upper_95",
        "true_infections_adult_median",
        "true_infections_adult_lower_95",
        "true_infections_adult_upper_95",
        "symptomatic_infections_adult_median",
        "symptomatic_infections_adult_lower_95",
        "symptomatic_infections_adult_upper_95",
        "reported_cases_adult_median",
        "reported_cases_adult_lower_95",
        "reported_cases_adult_upper_95",
        "effective_susceptible_median",
        "effective_susceptible_lower_95",
        "effective_susceptible_upper_95",
        "effective_susceptible_under5_median",
        "effective_susceptible_under5_lower_95",
        "effective_susceptible_under5_upper_95",
        "effective_susceptible_school_age_median",
        "effective_susceptible_school_age_lower_95",
        "effective_susceptible_school_age_upper_95",
        "effective_susceptible_adult_median",
        "effective_susceptible_adult_lower_95",
        "effective_susceptible_adult_upper_95",
        "natural_immunity_median",
        "natural_immunity_lower_95",
        "natural_immunity_upper_95",
        "waned_natural_immunity_median",
        "waned_natural_immunity_lower_95",
        "waned_natural_immunity_upper_95",
        "natural_immunity_under5_median",
        "natural_immunity_under5_lower_95",
        "natural_immunity_under5_upper_95",
        "waned_natural_immunity_under5_median",
        "waned_natural_immunity_under5_lower_95",
        "waned_natural_immunity_under5_upper_95",
        "natural_immunity_school_age_median",
        "natural_immunity_school_age_lower_95",
        "natural_immunity_school_age_upper_95",
        "waned_natural_immunity_school_age_median",
        "waned_natural_immunity_school_age_lower_95",
        "waned_natural_immunity_school_age_upper_95",
        "natural_immunity_adult_median",
        "natural_immunity_adult_lower_95",
        "natural_immunity_adult_upper_95",
        "waned_natural_immunity_adult_median",
        "waned_natural_immunity_adult_lower_95",
        "waned_natural_immunity_adult_upper_95",
        "partial_vaccine_immunity_median",
        "partial_vaccine_immunity_lower_95",
        "partial_vaccine_immunity_upper_95",
        "exposure_boosting_events_median",
        "exposure_boosting_events_lower_95",
        "exposure_boosting_events_upper_95",
        "natural_exposure_recharge_median",
        "natural_exposure_recharge_lower_95",
        "natural_exposure_recharge_upper_95",
        "routine_immunization_shortfall_median",
        "routine_immunization_shortfall_lower_95",
        "routine_immunization_shortfall_upper_95",
        "routine_delay_debt_median",
        "routine_delay_debt_lower_95",
        "routine_delay_debt_upper_95",
        "n_posterior_draws",
        "model2c_output_version",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    group_cols = [
        "iso3",
        "country",
        "scenario_id",
        "scenario_label",
        "scenario_family",
        "scenario_order",
        "neutralized_mechanisms",
        "year",
        "month",
        "date",
        "fit_role",
    ]
    summary = (
        frame.groupby(group_cols, dropna=False, sort=False)
        .agg(
            observed_cases=("observed_cases", _first_non_missing),
            true_infections_median=("true_infections_predicted", "median"),
            true_infections_lower_95=("true_infections_predicted", _quantile(0.025)),
            true_infections_upper_95=("true_infections_predicted", _quantile(0.975)),
            symptomatic_infections_median=("symptomatic_infections_predicted", "median"),
            symptomatic_infections_lower_95=("symptomatic_infections_predicted", _quantile(0.025)),
            symptomatic_infections_upper_95=("symptomatic_infections_predicted", _quantile(0.975)),
            reported_cases_median=("reported_cases_predicted", "median"),
            reported_cases_lower_95=("reported_cases_predicted", _quantile(0.025)),
            reported_cases_upper_95=("reported_cases_predicted", _quantile(0.975)),
            true_infections_under5_median=("true_infections_under5_predicted", "median"),
            true_infections_under5_lower_95=("true_infections_under5_predicted", _quantile(0.025)),
            true_infections_under5_upper_95=("true_infections_under5_predicted", _quantile(0.975)),
            symptomatic_infections_under5_median=("symptomatic_infections_under5_predicted", "median"),
            symptomatic_infections_under5_lower_95=("symptomatic_infections_under5_predicted", _quantile(0.025)),
            symptomatic_infections_under5_upper_95=("symptomatic_infections_under5_predicted", _quantile(0.975)),
            reported_cases_under5_median=("reported_cases_under5_predicted", "median"),
            reported_cases_under5_lower_95=("reported_cases_under5_predicted", _quantile(0.025)),
            reported_cases_under5_upper_95=("reported_cases_under5_predicted", _quantile(0.975)),
            true_infections_school_age_median=("true_infections_school_age_predicted", "median"),
            true_infections_school_age_lower_95=("true_infections_school_age_predicted", _quantile(0.025)),
            true_infections_school_age_upper_95=("true_infections_school_age_predicted", _quantile(0.975)),
            symptomatic_infections_school_age_median=("symptomatic_infections_school_age_predicted", "median"),
            symptomatic_infections_school_age_lower_95=("symptomatic_infections_school_age_predicted", _quantile(0.025)),
            symptomatic_infections_school_age_upper_95=("symptomatic_infections_school_age_predicted", _quantile(0.975)),
            reported_cases_school_age_median=("reported_cases_school_age_predicted", "median"),
            reported_cases_school_age_lower_95=("reported_cases_school_age_predicted", _quantile(0.025)),
            reported_cases_school_age_upper_95=("reported_cases_school_age_predicted", _quantile(0.975)),
            true_infections_adult_median=("true_infections_adult_predicted", "median"),
            true_infections_adult_lower_95=("true_infections_adult_predicted", _quantile(0.025)),
            true_infections_adult_upper_95=("true_infections_adult_predicted", _quantile(0.975)),
            symptomatic_infections_adult_median=("symptomatic_infections_adult_predicted", "median"),
            symptomatic_infections_adult_lower_95=("symptomatic_infections_adult_predicted", _quantile(0.025)),
            symptomatic_infections_adult_upper_95=("symptomatic_infections_adult_predicted", _quantile(0.975)),
            reported_cases_adult_median=("reported_cases_adult_predicted", "median"),
            reported_cases_adult_lower_95=("reported_cases_adult_predicted", _quantile(0.025)),
            reported_cases_adult_upper_95=("reported_cases_adult_predicted", _quantile(0.975)),
            effective_susceptible_median=("effective_susceptible_predicted", "median"),
            effective_susceptible_lower_95=("effective_susceptible_predicted", _quantile(0.025)),
            effective_susceptible_upper_95=("effective_susceptible_predicted", _quantile(0.975)),
            effective_susceptible_under5_median=("effective_susceptible_under5_predicted", "median"),
            effective_susceptible_under5_lower_95=("effective_susceptible_under5_predicted", _quantile(0.025)),
            effective_susceptible_under5_upper_95=("effective_susceptible_under5_predicted", _quantile(0.975)),
            effective_susceptible_school_age_median=("effective_susceptible_school_age_predicted", "median"),
            effective_susceptible_school_age_lower_95=("effective_susceptible_school_age_predicted", _quantile(0.025)),
            effective_susceptible_school_age_upper_95=("effective_susceptible_school_age_predicted", _quantile(0.975)),
            effective_susceptible_adult_median=("effective_susceptible_adult_predicted", "median"),
            effective_susceptible_adult_lower_95=("effective_susceptible_adult_predicted", _quantile(0.025)),
            effective_susceptible_adult_upper_95=("effective_susceptible_adult_predicted", _quantile(0.975)),
            natural_immunity_median=("natural_immunity_predicted", "median"),
            natural_immunity_lower_95=("natural_immunity_predicted", _quantile(0.025)),
            natural_immunity_upper_95=("natural_immunity_predicted", _quantile(0.975)),
            waned_natural_immunity_median=("waned_natural_immunity_predicted", "median"),
            waned_natural_immunity_lower_95=("waned_natural_immunity_predicted", _quantile(0.025)),
            waned_natural_immunity_upper_95=("waned_natural_immunity_predicted", _quantile(0.975)),
            natural_immunity_under5_median=("natural_immunity_under5_predicted", "median"),
            natural_immunity_under5_lower_95=("natural_immunity_under5_predicted", _quantile(0.025)),
            natural_immunity_under5_upper_95=("natural_immunity_under5_predicted", _quantile(0.975)),
            waned_natural_immunity_under5_median=("waned_natural_immunity_under5_predicted", "median"),
            waned_natural_immunity_under5_lower_95=("waned_natural_immunity_under5_predicted", _quantile(0.025)),
            waned_natural_immunity_under5_upper_95=("waned_natural_immunity_under5_predicted", _quantile(0.975)),
            natural_immunity_school_age_median=("natural_immunity_school_age_predicted", "median"),
            natural_immunity_school_age_lower_95=("natural_immunity_school_age_predicted", _quantile(0.025)),
            natural_immunity_school_age_upper_95=("natural_immunity_school_age_predicted", _quantile(0.975)),
            waned_natural_immunity_school_age_median=("waned_natural_immunity_school_age_predicted", "median"),
            waned_natural_immunity_school_age_lower_95=("waned_natural_immunity_school_age_predicted", _quantile(0.025)),
            waned_natural_immunity_school_age_upper_95=("waned_natural_immunity_school_age_predicted", _quantile(0.975)),
            natural_immunity_adult_median=("natural_immunity_adult_predicted", "median"),
            natural_immunity_adult_lower_95=("natural_immunity_adult_predicted", _quantile(0.025)),
            natural_immunity_adult_upper_95=("natural_immunity_adult_predicted", _quantile(0.975)),
            waned_natural_immunity_adult_median=("waned_natural_immunity_adult_predicted", "median"),
            waned_natural_immunity_adult_lower_95=("waned_natural_immunity_adult_predicted", _quantile(0.025)),
            waned_natural_immunity_adult_upper_95=("waned_natural_immunity_adult_predicted", _quantile(0.975)),
            partial_vaccine_immunity_median=("partial_vaccine_immunity_predicted", "median"),
            partial_vaccine_immunity_lower_95=("partial_vaccine_immunity_predicted", _quantile(0.025)),
            partial_vaccine_immunity_upper_95=("partial_vaccine_immunity_predicted", _quantile(0.975)),
            exposure_boosting_events_median=("exposure_boosting_events_predicted", "median"),
            exposure_boosting_events_lower_95=("exposure_boosting_events_predicted", _quantile(0.025)),
            exposure_boosting_events_upper_95=("exposure_boosting_events_predicted", _quantile(0.975)),
            natural_exposure_recharge_median=("natural_exposure_recharge_predicted", "median"),
            natural_exposure_recharge_lower_95=("natural_exposure_recharge_predicted", _quantile(0.025)),
            natural_exposure_recharge_upper_95=("natural_exposure_recharge_predicted", _quantile(0.975)),
            routine_immunization_shortfall_median=("routine_immunization_shortfall_predicted", "median"),
            routine_immunization_shortfall_lower_95=("routine_immunization_shortfall_predicted", _quantile(0.025)),
            routine_immunization_shortfall_upper_95=("routine_immunization_shortfall_predicted", _quantile(0.975)),
            routine_delay_debt_median=("routine_delay_debt_predicted", "median"),
            routine_delay_debt_lower_95=("routine_delay_debt_predicted", _quantile(0.025)),
            routine_delay_debt_upper_95=("routine_delay_debt_predicted", _quantile(0.975)),
            n_posterior_draws=("reported_cases_predicted", "size"),
        )
        .reset_index()
        .sort_values(["iso3", "scenario_order", "year", "month"])
    )
    summary["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
    return summary.loc[:, columns]


def _period_summary(frame: pd.DataFrame, period_start: int = 2023, period_end: int = 2025) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "scenario_id",
        "scenario_label",
        "scenario_family",
        "neutralized_mechanisms",
        "period_start",
        "period_end",
        "outcome_target",
        "full_total_median",
        "full_total_lower_95",
        "full_total_upper_95",
        "scenario_total_median",
        "scenario_total_lower_95",
        "scenario_total_upper_95",
        "full_minus_scenario_median",
        "full_minus_scenario_lower_95",
        "full_minus_scenario_upper_95",
        "scenario_to_full_ratio_median",
        "scenario_to_full_ratio_lower_95",
        "scenario_to_full_ratio_upper_95",
        "full_peak_median",
        "scenario_peak_median",
        "peak_ratio_median",
        "full_peak_month",
        "scenario_peak_month",
        "n_posterior_draws",
        "scenario_interpretation_note",
        "model2c_output_version",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    period = frame.loc[frame["year"].between(period_start, period_end)].copy()
    outcome_map = {
        "true_infections": "true_infections_predicted",
        "symptomatic_infections": "symptomatic_infections_predicted",
        "reported_cases": "reported_cases_predicted",
        "true_infections_under5": "true_infections_under5_predicted",
        "symptomatic_infections_under5": "symptomatic_infections_under5_predicted",
        "reported_cases_under5": "reported_cases_under5_predicted",
        "true_infections_school_age": "true_infections_school_age_predicted",
        "symptomatic_infections_school_age": "symptomatic_infections_school_age_predicted",
        "reported_cases_school_age": "reported_cases_school_age_predicted",
        "true_infections_adult": "true_infections_adult_predicted",
        "symptomatic_infections_adult": "symptomatic_infections_adult_predicted",
        "reported_cases_adult": "reported_cases_adult_predicted",
    }
    rows: list[dict[str, Any]] = []
    for outcome, column in outcome_map.items():
        totals = (
            period.groupby(
                [
                    "iso3",
                    "country",
                    "scenario_id",
                    "scenario_label",
                    "scenario_family",
                    "scenario_order",
                    "neutralized_mechanisms",
                    "scenario_interpretation_note",
                    "draw",
                ],
                dropna=False,
                sort=False,
            )[column]
            .sum()
            .rename("scenario_total")
            .reset_index()
        )
        peaks = (
            period.sort_values(["iso3", "scenario_id", "draw", column, "date"], ascending=[True, True, True, False, True])
            .groupby(["iso3", "scenario_id", "draw"], dropna=False, sort=False)
            .head(1)
            .loc[:, ["iso3", "scenario_id", "draw", "date", column]]
            .rename(columns={"date": "scenario_peak_month", column: "scenario_peak"})
        )
        totals = totals.merge(peaks, on=["iso3", "scenario_id", "draw"], how="left")
        full = totals.loc[totals["scenario_id"].eq("full_model"), ["iso3", "draw", "scenario_total", "scenario_peak", "scenario_peak_month"]]
        full = full.rename(
            columns={
                "scenario_total": "full_total",
                "scenario_peak": "full_peak",
                "scenario_peak_month": "full_peak_month",
            }
        )
        merged = totals.merge(full, on=["iso3", "draw"], how="left")
        merged["full_minus_scenario"] = merged["full_total"] - merged["scenario_total"]
        merged["scenario_to_full_ratio"] = np.where(
            merged["full_total"].gt(0),
            merged["scenario_total"] / merged["full_total"],
            np.nan,
        )
        merged["peak_ratio"] = np.where(
            merged["full_peak"].gt(0),
            merged["scenario_peak"] / merged["full_peak"],
            np.nan,
        )
        for keys, group in merged.groupby(
            [
                "iso3",
                "country",
                "scenario_id",
                "scenario_label",
                "scenario_family",
                "scenario_order",
                "neutralized_mechanisms",
                "scenario_interpretation_note",
            ],
            sort=False,
            dropna=False,
        ):
            (
                iso3,
                country,
                scenario_id,
                scenario_label,
                scenario_family,
                scenario_order,
                neutralized_mechanisms,
                scenario_note,
            ) = keys
            rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "scenario_id": scenario_id,
                    "scenario_label": scenario_label,
                    "scenario_family": scenario_family,
                    "scenario_order": int(scenario_order),
                    "neutralized_mechanisms": neutralized_mechanisms,
                    "period_start": int(period_start),
                    "period_end": int(period_end),
                    "outcome_target": outcome,
                    "full_total_median": float(group["full_total"].median()),
                    "full_total_lower_95": float(group["full_total"].quantile(0.025)),
                    "full_total_upper_95": float(group["full_total"].quantile(0.975)),
                    "scenario_total_median": float(group["scenario_total"].median()),
                    "scenario_total_lower_95": float(group["scenario_total"].quantile(0.025)),
                    "scenario_total_upper_95": float(group["scenario_total"].quantile(0.975)),
                    "full_minus_scenario_median": float(group["full_minus_scenario"].median()),
                    "full_minus_scenario_lower_95": float(group["full_minus_scenario"].quantile(0.025)),
                    "full_minus_scenario_upper_95": float(group["full_minus_scenario"].quantile(0.975)),
                    "scenario_to_full_ratio_median": float(group["scenario_to_full_ratio"].median()),
                    "scenario_to_full_ratio_lower_95": float(group["scenario_to_full_ratio"].quantile(0.025)),
                    "scenario_to_full_ratio_upper_95": float(group["scenario_to_full_ratio"].quantile(0.975)),
                    "full_peak_median": float(group["full_peak"].median()),
                    "scenario_peak_median": float(group["scenario_peak"].median()),
                    "peak_ratio_median": float(group["peak_ratio"].median()),
                    "full_peak_month": _mode_month(group["full_peak_month"]),
                    "scenario_peak_month": _mode_month(group["scenario_peak_month"]),
                    "n_posterior_draws": int(group["draw"].nunique()),
                    "scenario_interpretation_note": scenario_note,
                    "model2c_output_version": MODEL2C_OUTPUT_VERSION,
                }
            )
    state_snapshot_years = sorted({period_start, NATURAL_BOOST_STOCK_READOUT_YEAR})
    for snapshot_year in state_snapshot_years:
        snapshot = frame.loc[(frame["year"].eq(snapshot_year)) & (frame["month"].eq(1))].copy()
        state_outcomes = {
            f"effective_susceptible_start_{snapshot_year}": "effective_susceptible_predicted",
            f"effective_susceptible_under5_start_{snapshot_year}": "effective_susceptible_under5_predicted",
            f"effective_susceptible_school_age_start_{snapshot_year}": "effective_susceptible_school_age_predicted",
            f"effective_susceptible_adult_start_{snapshot_year}": "effective_susceptible_adult_predicted",
            f"natural_immunity_start_{snapshot_year}": "natural_immunity_predicted",
            f"waned_natural_immunity_start_{snapshot_year}": "waned_natural_immunity_predicted",
            f"natural_immunity_under5_start_{snapshot_year}": "natural_immunity_under5_predicted",
            f"waned_natural_immunity_under5_start_{snapshot_year}": "waned_natural_immunity_under5_predicted",
            f"natural_immunity_school_age_start_{snapshot_year}": "natural_immunity_school_age_predicted",
            f"waned_natural_immunity_school_age_start_{snapshot_year}": "waned_natural_immunity_school_age_predicted",
            f"natural_immunity_adult_start_{snapshot_year}": "natural_immunity_adult_predicted",
            f"waned_natural_immunity_adult_start_{snapshot_year}": "waned_natural_immunity_adult_predicted",
            f"routine_delay_debt_start_{snapshot_year}": "routine_delay_debt_predicted",
        }
        for outcome, column in state_outcomes.items():
            if snapshot.empty or column not in snapshot.columns:
                continue
            values = snapshot.loc[
                :,
                [
                    "iso3",
                    "country",
                    "scenario_id",
                    "scenario_label",
                    "scenario_family",
                    "scenario_order",
                    "neutralized_mechanisms",
                    "scenario_interpretation_note",
                    "draw",
                    column,
                ],
            ].rename(columns={column: "scenario_total"})
            full = values.loc[values["scenario_id"].eq("full_model"), ["iso3", "draw", "scenario_total"]].rename(
                columns={"scenario_total": "full_total"}
            )
            merged = values.merge(full, on=["iso3", "draw"], how="left")
            merged["full_minus_scenario"] = merged["full_total"] - merged["scenario_total"]
            merged["scenario_to_full_ratio"] = np.where(
                merged["full_total"].gt(0),
                merged["scenario_total"] / merged["full_total"],
                np.nan,
            )
            for keys, group in merged.groupby(
                [
                    "iso3",
                    "country",
                    "scenario_id",
                    "scenario_label",
                    "scenario_family",
                    "scenario_order",
                    "neutralized_mechanisms",
                    "scenario_interpretation_note",
                ],
                sort=False,
                dropna=False,
            ):
                (
                    iso3,
                    country,
                    scenario_id,
                    scenario_label,
                    scenario_family,
                    scenario_order,
                    neutralized_mechanisms,
                    scenario_note,
                ) = keys
                rows.append(
                    {
                        "iso3": iso3,
                        "country": country,
                        "scenario_id": scenario_id,
                        "scenario_label": scenario_label,
                        "scenario_family": scenario_family,
                        "scenario_order": int(scenario_order),
                        "neutralized_mechanisms": neutralized_mechanisms,
                        "period_start": int(snapshot_year),
                        "period_end": int(snapshot_year),
                        "outcome_target": outcome,
                        "full_total_median": float(group["full_total"].median()),
                        "full_total_lower_95": float(group["full_total"].quantile(0.025)),
                        "full_total_upper_95": float(group["full_total"].quantile(0.975)),
                        "scenario_total_median": float(group["scenario_total"].median()),
                        "scenario_total_lower_95": float(group["scenario_total"].quantile(0.025)),
                        "scenario_total_upper_95": float(group["scenario_total"].quantile(0.975)),
                        "full_minus_scenario_median": float(group["full_minus_scenario"].median()),
                        "full_minus_scenario_lower_95": float(group["full_minus_scenario"].quantile(0.025)),
                        "full_minus_scenario_upper_95": float(group["full_minus_scenario"].quantile(0.975)),
                        "scenario_to_full_ratio_median": float(group["scenario_to_full_ratio"].median()),
                        "scenario_to_full_ratio_lower_95": float(group["scenario_to_full_ratio"].quantile(0.025)),
                        "scenario_to_full_ratio_upper_95": float(group["scenario_to_full_ratio"].quantile(0.975)),
                        "full_peak_median": float("nan"),
                        "scenario_peak_median": float("nan"),
                        "peak_ratio_median": float("nan"),
                        "full_peak_month": f"{snapshot_year}-01-15",
                        "scenario_peak_month": f"{snapshot_year}-01-15",
                        "n_posterior_draws": int(group["draw"].nunique()),
                        "scenario_interpretation_note": scenario_note,
                        "model2c_output_version": MODEL2C_OUTPUT_VERSION,
                    }
                )
    return pd.DataFrame(rows).sort_values(["iso3", "scenario_order", "outcome_target"]).loc[:, columns]


def model2c_counterfactual_outputs(
    data: dict[str, Any],
    metadata: dict[str, Any],
    draws: pd.DataFrame,
    *,
    max_draws: int | None = None,
    start_year: int = 2020,
    end_year: int = 2026,
    period_start: int = 2023,
    period_end: int = 2025,
    include_supplement_context_audit: bool = False,
    draw_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _scenario_prediction_frame(
        data,
        metadata,
        draws,
        max_draws=max_draws,
        start_year=start_year,
        end_year=end_year,
        include_supplement_context_audit=include_supplement_context_audit,
        draw_workers=draw_workers,
    )
    return _monthly_summary(frame), _period_summary(frame, period_start=period_start, period_end=period_end)


def model2c_posterior_predictive(data: dict[str, Any], metadata: dict[str, Any], draws: pd.DataFrame) -> pd.DataFrame:
    month_obs_idx = {int(index) for index in data.get("month_obs_idx", [])}
    month_ppc_obs_idx = {int(index) for index in data.get("month_ppc_obs_idx", [])}
    rows: list[pd.DataFrame] = []
    for draw in draws.itertuples(index=False):
        draw_series = pd.Series(draw._asdict())
        pred = _predict_monthly(data, draw_series, scenario_id="full_model", include_predictive_replicate=True)
        pred.insert(0, "draw", int(draw_series["draw"]))
        pred.insert(0, "country", metadata["country"])
        pred.insert(0, "iso3", metadata["iso3"])
        month_index = np.arange(1, len(pred) + 1)
        pred["fit_role"] = np.select(
            [
                np.isin(month_index, list(month_obs_idx)),
                np.isin(month_index, list(month_ppc_obs_idx)),
                pred["year"].le(2025),
            ],
            [
                "fit_likelihood",
                "posterior_predictive_check",
                "state_history_no_monthly_observation",
            ],
            default="posterior_predictive_unobserved_month",
        )
        pred["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
        rows.append(pred)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def model2c_observation_audit(data: dict[str, Any], metadata: dict[str, Any], draws: pd.DataFrame) -> pd.DataFrame:
    def posterior_numeric(name: str, default: float) -> pd.Series:
        if name in draws:
            values = pd.to_numeric(draws[name], errors="coerce")
            return values.fillna(default)
        return pd.Series(np.repeat(default, max(len(draws), 1)))

    reporting = np.exp(posterior_numeric("log_post_reporting_multiplier", 0.0))
    post_reporting_peak_gain = posterior_numeric("post_reporting_peak_gain", 0.0)
    post_reporting_peak_multiplier = posterior_numeric("post_reporting_peak_multiplier", np.nan)
    if post_reporting_peak_multiplier.isna().all():
        post_reporting_peak_multiplier = 1.0 + post_reporting_peak_gain
    transmission = np.exp(posterior_numeric("log_post_transmission_multiplier", 0.0))
    pathogen_pressure_scale = posterior_numeric("excess_epidemic_pressure_scale", np.nan)
    if pathogen_pressure_scale.isna().all():
        pathogen_pressure_scale = posterior_numeric("pathogen_pressure_scale", 0.0)
    outbreak_seeding_gain = posterior_numeric("outbreak_ignition_pulse_gain", np.nan)
    if outbreak_seeding_gain.isna().all():
        outbreak_seeding_gain = posterior_numeric("outbreak_seeding_gain", 0.0)
    secondary_outbreak_gain = posterior_numeric("secondary_outbreak_ignition_pulse_gain", np.nan)
    if secondary_outbreak_gain.isna().all():
        secondary_outbreak_gain = outbreak_seeding_gain * posterior_numeric("secondary_rebound_release_gain", 0.0)
    rebound_asymmetry = posterior_numeric("rebound_asymmetry", 0.0)
    post_reporting_trend = posterior_numeric("post_reporting_trend_per_year", np.nan)
    if post_reporting_trend.isna().all():
        post_reporting_trend = posterior_numeric("post_reporting_trend", 0.0)
    if not bool(int(data.get("use_reporting_shift", 1))):
        reporting = pd.Series(np.ones(max(len(draws), 1)))
        post_reporting_peak_gain = pd.Series(np.zeros(max(len(draws), 1)))
        post_reporting_peak_multiplier = pd.Series(np.ones(max(len(draws), 1)))
        post_reporting_trend = pd.Series(np.zeros(max(len(draws), 1)))
    if bool(int(data.get("use_flexible_outbreak_seeding", 0))):
        outbreak_seeding_gain = posterior_numeric("flexible_outbreak_seeding_total_gain", np.nan)
        if outbreak_seeding_gain.isna().all():
            outbreak_seeding_gain = sum(
                (posterior_numeric(f"outbreak_seed_basis_gain[{index}]", 0.0) for index in range(1, 5)),
                start=pd.Series(np.zeros(max(len(draws), 1))),
            )
        secondary_outbreak_gain = pd.Series(np.zeros(max(len(draws), 1)))
    nb_phi_month = posterior_numeric("nb_phi_month", np.nan)
    if nb_phi_month.isna().all():
        nb_phi_month = posterior_numeric("nb_phi", 20.0)
    direct_support = _metadata_bool(metadata.get("direct_testing_support", False))
    return pd.DataFrame(
        [
            {
                "iso3": metadata["iso3"],
                "country": metadata["country"],
                "reporting_prior_mean": metadata.get("reporting_prior_mean", np.nan),
                "reporting_prior_lower": metadata.get("reporting_prior_lower", np.nan),
                "reporting_prior_upper": metadata.get("reporting_prior_upper", np.nan),
                "reporting_prior_log_sd": metadata.get("reporting_prior_log_sd", np.nan),
                "direct_testing_support": direct_support,
                "reporting_prior_support_level": metadata.get("reporting_prior_support_level", "not_available"),
                "contact_mode_requested": metadata.get("contact_mode_requested", "rank_one"),
                "model2c_contact_matrix_source": metadata.get("model2b_contact_matrix_source", "rank_one_contact_vector_outer_product"),
                "model2c_contact_matrix_fallback": _metadata_bool(metadata.get("model2b_contact_matrix_fallback", False)),
                "direct_testing_denominator_evidence": _metadata_bool(metadata.get("direct_testing_denominator_evidence", False)),
                "case_definition_reporting_practice_evidence": _metadata_bool(
                    metadata.get("case_definition_reporting_practice_evidence", False)
                ),
                "severe_outcome_evidence": _metadata_bool(metadata.get("severe_outcome_evidence", False)),
                "serology_evidence": _metadata_bool(metadata.get("serology_evidence", False)),
                "pathogen_pressure_evidence": _metadata_bool(metadata.get("pathogen_pressure_evidence", False)),
                "post_reporting_multiplier_median": float(reporting.median()),
                "post_reporting_multiplier_lower_95": float(reporting.quantile(0.025)),
                "post_reporting_multiplier_upper_95": float(reporting.quantile(0.975)),
                "post_reporting_peak_multiplier_median": float(post_reporting_peak_multiplier.median()),
                "post_reporting_peak_gain_median": float(post_reporting_peak_gain.median()),
                "post_transmission_multiplier_median": float(transmission.median()),
                "post_transmission_multiplier_lower_95": float(transmission.quantile(0.025)),
                "post_transmission_multiplier_upper_95": float(transmission.quantile(0.975)),
                "excess_epidemic_pressure_scale_median": float(pathogen_pressure_scale.median()),
                "outbreak_ignition_pulse_gain_median": float(outbreak_seeding_gain.median()),
                "secondary_outbreak_ignition_pulse_gain_median": float(secondary_outbreak_gain.median()),
                "rebound_asymmetry_median": float(rebound_asymmetry.median()),
                "post_reporting_trend_per_year_median": float(post_reporting_trend.median()),
                "nb_phi_month_median": float(nb_phi_month.median()) if nb_phi_month.notna().any() else np.nan,
                "posterior_reporting_shift_probability": float((np.abs(np.log(reporting.clip(lower=1e-9))) > np.log(1.25)).mean()),
                "posterior_reporting_peak_probability": float((post_reporting_peak_gain > 0.25).mean()),
                "posterior_transmission_context_shift_probability": float(
                    (np.abs(np.log(transmission.clip(lower=1e-9))) > np.log(1.25)).mean()
                ),
                "posterior_excess_epidemic_pressure_active_probability": float((pathogen_pressure_scale > 0.5).mean()),
                "posterior_outbreak_ignition_pulse_active_probability": float((outbreak_seeding_gain > 0.02).mean()),
                "posterior_secondary_outbreak_ignition_active_probability": float((secondary_outbreak_gain > 0.01).mean()),
                "posterior_reporting_trend_probability": float((np.abs(post_reporting_trend) > np.log(1.25)).mean()),
                "month_obs_weight_min": metadata.get("model2b_month_obs_weight_min", np.nan),
                "month_obs_weight_mean": metadata.get("model2b_month_obs_weight_mean", np.nan),
                "month_obs_weight_max": metadata.get("model2b_month_obs_weight_max", np.nan),
                "observation_process_supported": direct_support,
                "observation_process_role": (
                    "supported_observation_process_sensitivity"
                    if direct_support
                    else "constrained_nuisance_not_transmission_scenario"
                ),
                "model2c_output_version": MODEL2C_OUTPUT_VERSION,
                "interpretation_note": (
                    "Reporting/testing is retained as an observation-process scenario. Model 2C scenario contrasts "
                    "are model-implied differences from calibrated posterior replay, not additive mechanism fractions."
                ),
            }
        ]
    )
