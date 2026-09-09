from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import factorial
from typing import Any

import numpy as np
import pandas as pd


MODEL2B_OUTPUT_VERSION = "model2b_v2_20260616_layered_endemic_background_attribution"

MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS = {
    "low_circulation_missed_boosting": {
        "label": "Low circulation / missed natural boosting",
        "family": "contact_circulation",
        "definition": (
            "2020-2023 contact/circulation suppression and missed natural immune boosting are active; "
            "fitted endemic background circulation remains active in both the full and neutral counterfactual."
        ),
    },
    "routine_immunization_disruption": {
        "label": "Routine immunization disruption",
        "family": "immunization_flow",
        "definition": (
            "Observed 2020+ DTP3 flow disruptions are active; the primary counterfactual restores "
            "DTP3 flow to the pre-pandemic baseline from 2020 onward."
        ),
    },
    "vaccine_waning_booster_gap": {
        "label": "Waning/booster gap",
        "family": "vaccine_immunity",
        "definition": (
            "Fitted vaccine waning, booster reach and age-specific waning modifiers are active; "
            "the neutral counterfactual constrains waning downward and restores booster-related modifiers."
        ),
    },
    "birth_cohort_demography": {
        "label": "Birth-cohort demography",
        "family": "demography",
        "definition": (
            "Observed monthly births, aging targets and age-population denominators are active; "
            "the neutral counterfactual uses the 2019 demographic reference path."
        ),
    },
}

MODEL2B_TRANSMISSION_CONTEXT_FACTORS = {
    "post_2023_transmission_shift": {
        "label": "Post-2023 transmission shift",
        "family": "post_2023_transmission",
        "definition": (
            "Fitted post-2023 transmission multiplier and rebound release gain are active. "
            "This is a transmission-context audit factor, not a primary susceptibility driver."
        ),
    },
    "excess_epidemic_pressure": {
        "label": "Excess epidemic pressure",
        "family": "epidemic_pressure",
        "definition": (
            "Fitted pathogen/evidence-informed excess pressure scaling is active above the always-on "
            "endemic background force. Neutralization sets only the excess scaling to zero."
        ),
    },
    "outbreak_ignition_pulse": {
        "label": "Outbreak ignition pulse",
        "family": "epidemic_pressure",
        "definition": (
            "Fitted post-2023 outbreak ignition pulse is active above the always-on endemic background force. "
            "Neutralization sets only the pulse gain to zero."
        ),
    },
}

MODEL2B_TRANSMISSION_FACTORS = {
    **MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS,
    **MODEL2B_TRANSMISSION_CONTEXT_FACTORS,
}
MODEL2B_REPORTING_FACTOR = {
    "reporting_testing_shift": {
        "label": "Reporting/testing shift",
        "family": "observation_process",
        "definition": "Fitted post-2023 reporting multiplier is active.",
    }
}
MODEL2B_REPORTED_FACTORS = {**MODEL2B_TRANSMISSION_FACTORS, **MODEL2B_REPORTING_FACTOR}

MODEL2B_FACTOR_ALIASES: dict[str, str | tuple[str, ...]] = {}

# Backward-compatible export name used by runner/tests; it now means the four
# primary susceptibility factors in the layered attribution.
MODEL2B_PRIMARY_MECHANISMS = {
    factor_id: spec["label"] for factor_id, spec in MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS.items()
}
MODEL2B_CONTEXT_AUDIT_FACTORS = {
    factor_id: MODEL2B_TRANSMISSION_CONTEXT_FACTORS[factor_id]
    for factor_id in MODEL2B_TRANSMISSION_CONTEXT_FACTORS
}

MODEL2B_COUNTERFACTUAL_SCENARIOS = {
    "full_model": {
        "scenario_label": "Fitted model",
        "removed_mechanisms": (),
        "scenario_order": 0,
        "scenario_note": (
            "All fitted susceptibility, transmission-context, endemic-background and reporting terms active."
        ),
    },
    "primary_susceptibility_neutral_with_fitted_background": {
        "scenario_label": "Primary susceptibility neutral",
        "removed_mechanisms": tuple(MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS),
        "scenario_order": 1,
        "scenario_note": (
            "The four primary susceptibility factors are neutralized while fitted endemic background "
            "circulation and transmission-context terms are retained."
        ),
    },
    "transmission_context_neutral_with_susceptibility_fitted": {
        "scenario_label": "Transmission context neutral",
        "removed_mechanisms": tuple(MODEL2B_TRANSMISSION_CONTEXT_FACTORS),
        "scenario_order": 2,
        "scenario_note": (
            "Post-2023 transmission shift, excess epidemic pressure and outbreak ignition pulse are "
            "neutralized while the primary susceptibility state remains fitted."
        ),
    },
    "observation_neutral_with_transmission_fitted": {
        "scenario_label": "Observation neutral",
        "removed_mechanisms": ("reporting_testing_shift",),
        "scenario_order": 3,
        "scenario_note": "The post-2023 reporting/testing multiplier is neutralized while transmission states are unchanged.",
    },
    "no_low_circulation_missed_boosting": {
        "scenario_label": "No low circulation / missed natural boosting",
        "removed_mechanisms": ("low_circulation_missed_boosting",),
        "scenario_order": 4,
        "scenario_note": (
            "The 2020-2023 contact and low-circulation suppression is removed for infection, "
            "natural-immunity replenishment and exposure-driven boosting, so circulation and "
            "boosting follow the no-pandemic reference path."
        ),
    },
    "no_routine_immunization_disruption": {
        "scenario_label": "No routine disruption",
        "removed_mechanisms": ("routine_immunization_disruption",),
        "scenario_order": 5,
        "scenario_note": "DTP3 vaccination flow is restored to the pre-pandemic baseline from 2020 onward.",
    },
    "no_waning_booster_gap": {
        "scenario_label": "No waning/booster gap",
        "removed_mechanisms": ("vaccine_waning_booster_gap",),
        "scenario_order": 6,
        "scenario_note": "Vaccine waning is constrained downward and booster-related age modifiers are restored.",
    },
    "fixed_birth_cohort_demography": {
        "scenario_label": "Fixed birth-cohort demography",
        "removed_mechanisms": ("birth_cohort_demography",),
        "scenario_order": 7,
        "scenario_note": "Monthly births, aging targets and age-population denominators are held at the 2019 demographic reference.",
    },
    "no_post_2023_transmission_shift": {
        "scenario_label": "No post-2023 transmission shift",
        "removed_mechanisms": ("post_2023_transmission_shift",),
        "scenario_order": 8,
        "scenario_note": (
            "The post-2023 transmission multiplier and rebound release gain are neutralized as a "
            "transmission-context audit scenario."
        ),
    },
    "no_excess_epidemic_pressure": {
        "scenario_label": "No excess epidemic pressure",
        "removed_mechanisms": ("excess_epidemic_pressure",),
        "scenario_order": 9,
        "scenario_note": (
            "Evidence-informed excess pressure scaling is neutralized; endemic background force remains active."
        ),
    },
    "no_outbreak_ignition_pulse": {
        "scenario_label": "No outbreak ignition pulse",
        "removed_mechanisms": ("outbreak_ignition_pulse",),
        "scenario_order": 10,
        "scenario_note": "The post-2023 outbreak ignition pulse is neutralized; endemic background force remains active.",
    },
    "no_reporting_testing_shift": {
        "scenario_label": "No reporting/testing shift",
        "removed_mechanisms": ("reporting_testing_shift",),
        "scenario_order": 11,
        "scenario_note": "The post-2023 reporting multiplier is neutralized; transmission states are unchanged.",
    },
}

MODEL2B_OUTCOMES = {
    "true_infections": "true_infections_2023_2025",
    "symptomatic_infections": "symptomatic_infections_2023_2025",
    "reported_cases": "reported_cases_2023_2025",
}


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
    return str(value).strip().casefold() in {"1", "true", "t", "yes", "y"}


def _array(data: dict[str, Any], key: str, dtype=float) -> np.ndarray:
    return np.asarray(data[key], dtype=dtype)


def _draw_value(draw: pd.Series | dict[str, Any], key: str, default: float) -> float:
    if isinstance(draw, pd.Series):
        value = draw.get(key, default)
    else:
        value = draw.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return value if np.isfinite(value) else default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            return max(minimum, int(raw))
        except ValueError:
            return default
    return default


def _normalise_factors(factors: set[str] | list[str] | tuple[str, ...] | None, *, default_all: bool = True) -> set[str]:
    if factors is None:
        return set(MODEL2B_REPORTED_FACTORS) if default_all else set()
    normalised: set[str] = set()
    for factor in factors:
        mapped = MODEL2B_FACTOR_ALIASES.get(str(factor), str(factor))
        if isinstance(mapped, tuple):
            normalised.update(str(item) for item in mapped)
        else:
            normalised.add(str(mapped))
    return normalised.intersection(MODEL2B_REPORTED_FACTORS)


def _predict_monthly(
    data: dict[str, Any],
    draw: pd.Series | dict[str, Any],
    *,
    active_factors: set[str] | None = None,
    active_mechanisms: set[str] | None = None,
    include_boundary: bool | None = None,
    include_predictive_replicate: bool = False,
    totals_only: bool = False,
) -> pd.DataFrame | dict[str, float]:
    if active_factors is None:
        active = _normalise_factors(active_mechanisms, default_all=True)
    else:
        active = _normalise_factors(active_factors, default_all=True)
    if include_boundary is False:
        active.difference_update(MODEL2B_TRANSMISSION_CONTEXT_FACTORS)
    year = _array(data, "year", int)
    month = _array(data, "month", int)
    dtp3 = _array(data, "dtp3")
    age_population = _array(
        data,
        "age_population" if "birth_cohort_demography" in active else "age_population_ref",
    )
    monthly_births = _array(
        data,
        "monthly_births" if "birth_cohort_demography" in active else "monthly_births_ref",
    )
    age_aging_rate = _array(data, "age_aging_rate")
    contact_intensity = _array(data, "contact_intensity")
    periodic_multiplier = _array(data, "periodic_multiplier")
    seasonal_multiplier = _array(data, "seasonal_multiplier") if "seasonal_multiplier" in data else np.ones(len(year))
    pathogen_pressure_prior = _array(data, "pathogen_pressure_prior") if "pathogen_pressure_prior" in data else np.ones(len(year))
    age_rel_contact = _array(data, "age_rel_contact")
    age_symptom_prob = _array(data, "age_symptom_prob")
    age_report_weight = _array(data, "age_report_weight")
    age_program_reach = _array(data, "age_program_reach")
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
    contact_sensitivity = _draw_value(draw, "contact_sensitivity", 0.55)
    waning_rate = _draw_value(draw, "waning_rate", 0.10)
    post_transmission_multiplier = float(np.exp(_draw_value(draw, "log_post_transmission_multiplier", 0.0)))
    post_reporting_multiplier = float(np.exp(_draw_value(draw, "log_post_reporting_multiplier", 0.0)))
    rebound_peak_month = _draw_value(draw, "rebound_peak_month", float(data.get("rebound_peak_month_prior_mean", 24.0)))
    rebound_width = max(_draw_value(draw, "rebound_width", 8.0), 1e-6)
    rebound_release_gain = _draw_value(draw, "rebound_release_gain", 1.25)
    outbreak_seeding_gain = _draw_value(
        draw,
        "outbreak_ignition_pulse_gain",
        _draw_value(draw, "outbreak_seeding_gain", 0.03),
    )
    post_reporting_peak_gain = _draw_value(
        draw,
        "post_reporting_peak_gain",
        max(_draw_value(draw, "post_reporting_peak_multiplier", 1.0) - 1.0, 0.0),
    )
    seasonal_strength = _draw_value(draw, "seasonal_strength", 1.0)
    natural_boosting_efficiency = _draw_value(draw, "natural_boosting_efficiency", 0.25)
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
    contact_circulation_active = "low_circulation_missed_boosting" in active
    if "post_2023_transmission_shift" not in active:
        post_transmission_multiplier = 1.0
        rebound_release_gain = 0.0
    if "excess_epidemic_pressure" not in active:
        pathogen_pressure_scale = 0.0
    if "outbreak_ignition_pulse" not in active:
        outbreak_seeding_gain = 0.0
    if "reporting_testing_shift" not in active:
        post_reporting_multiplier = 1.0
        post_reporting_peak_gain = 0.0
    if "vaccine_waning_booster_gap" not in active:
        waning_rate = min(waning_rate, 0.035)
        booster_reach_modifier = np.maximum(booster_reach_modifier, 1.0)
        vaccine_waning_age_modifier = np.minimum(vaccine_waning_age_modifier, 1.0)

    pre_mean_dtp3 = float(data["pre_mean_dtp3"])
    pre_mean_incidence = float(data.get("pre_mean_incidence_per_100k", 0.0))
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
    natural_waning_month = 0.010
    deep_natural_waning_month = 0.003
    exposed_progression = min(0.95, max(0.55, 1.0 - 0.50 * epidemic_pressure_retention))
    infectious_clearance = min(0.95, max(0.35, 1.0 - 0.65 * epidemic_pressure_retention))
    pop0 = np.maximum(age_population[0, :], 1.0)
    vaccine_share = np.clip(dtp3[0] * age_program_reach * booster_reach_modifier, 0.0, 0.92)
    natural_share = np.clip(init_natural_share, 0.0, 0.50)
    susceptible_naive = np.zeros(n_age, dtype=float)
    vaccine_recent = pop0 * vaccine_share * init_vaccine_recent_fraction
    vaccine_waned = pop0 * vaccine_share * (1.0 - init_vaccine_recent_fraction)
    natural_recent = pop0 * natural_share * init_natural_recent_fraction
    natural_waned = pop0 * natural_share * (1.0 - init_natural_recent_fraction)
    exposed = pop0 * 1e-8
    infectious = pop0 * 1e-8
    susceptible_naive[:] = np.maximum(
        0.0,
        pop0 - vaccine_recent - vaccine_waned - natural_recent - natural_waned - exposed - infectious,
    )
    contact_weight_sum = max(float(np.sum(age_rel_contact)), 1e-9)

    for m in range(len(year)):
        year_m = int(year[m])
        month_m = int(month[m])
        if contact_circulation_active and float(contact_intensity[m]) > 0.0:
            contact_multiplier = max(0.03, 1.0 - contact_sensitivity * float(contact_intensity[m]) / 100.0)
        else:
            contact_multiplier = 1.0
        dtp3_eff = float(dtp3[m])
        if "routine_immunization_disruption" not in active and year_m >= 2020:
            dtp3_eff = max(dtp3_eff, pre_mean_dtp3)
        release_shape = 0.0
        reporting_peak_shape = 0.0
        if year_m >= 2023:
            post_month = (year_m - 2023) * 12 + month_m
            release_shape = float(np.exp(-0.5 * ((post_month - rebound_peak_month) / rebound_width) ** 2))
            reporting_peak_width = max(0.75, 0.55 * rebound_width)
            reporting_peak_shape = float(np.exp(-0.5 * ((post_month - rebound_peak_month) / reporting_peak_width) ** 2))
        seasonal_force = float(np.power(max(float(seasonal_multiplier[m]), 0.01), seasonal_strength))
        pathogen_multiplier = float(np.clip(1.0 + pathogen_pressure_scale * (float(pathogen_pressure_prior[m]) - 1.0), 0.40, 6.0))
        next_susceptible_naive = np.zeros(n_age, dtype=float)
        next_vaccine_recent = np.zeros(n_age, dtype=float)
        next_vaccine_waned = np.zeros(n_age, dtype=float)
        next_natural_recent = np.zeros(n_age, dtype=float)
        next_natural_waned = np.zeros(n_age, dtype=float)
        next_exposed = np.zeros(n_age, dtype=float)
        next_infectious = np.zeros(n_age, dtype=float)
        for a in range(n_age):
            aging_out = float(np.clip(age_aging_rate[a], 0.0, 0.95))
            stay = 1.0 - aging_out
            next_susceptible_naive[a] += susceptible_naive[a] * stay
            next_vaccine_recent[a] += vaccine_recent[a] * stay
            next_vaccine_waned[a] += vaccine_waned[a] * stay
            next_natural_recent[a] += natural_recent[a] * stay
            next_natural_waned[a] += natural_waned[a] * stay
            next_exposed[a] += exposed[a] * stay
            next_infectious[a] += infectious[a] * stay
            if a < n_age - 1:
                next_susceptible_naive[a + 1] += susceptible_naive[a] * aging_out
                next_vaccine_recent[a + 1] += vaccine_recent[a] * aging_out
                next_vaccine_waned[a + 1] += vaccine_waned[a] * aging_out
                next_natural_recent[a + 1] += natural_recent[a] * aging_out
                next_natural_waned[a + 1] += natural_waned[a] * aging_out
                next_exposed[a + 1] += exposed[a] * aging_out
                next_infectious[a + 1] += infectious[a] * aging_out
        next_susceptible_naive[0] += float(monthly_births[m])
        susceptible_naive = next_susceptible_naive
        vaccine_recent = next_vaccine_recent
        vaccine_waned = next_vaccine_waned
        natural_recent = next_natural_recent
        natural_waned = next_natural_waned
        exposed = next_exposed
        infectious = next_infectious
        for a in range(n_age):
            target_population = max(float(age_population[m, a]), 1.0)
            state_total = max(
                float(
                    susceptible_naive[a]
                    + vaccine_recent[a]
                    + vaccine_waned[a]
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
            natural_recent[a] *= demographic_adjustment
            natural_waned[a] *= demographic_adjustment
            exposed[a] *= demographic_adjustment
            infectious[a] *= demographic_adjustment
        state_population = np.maximum(
            susceptible_naive + vaccine_recent + vaccine_waned + natural_recent + natural_waned + exposed + infectious,
            1.0,
        )
        infectious_fraction = (infectious + 0.35 * exposed) / state_population
        age_infectious_pressure = np.asarray(age_contact_matrix @ infectious_fraction, dtype=float) / contact_weight_sum
        transmission_multiplier = post_transmission_multiplier if year_m >= 2023 else 1.0
        reporting_multiplier = (
            post_reporting_multiplier * (1.0 + post_reporting_peak_gain * reporting_peak_shape)
            if year_m >= 2023
            else 1.0
        )
        for a in range(n_age):
            target_population = max(float(age_population[m, a]), 1.0)
            target_vaccine_share = float(np.clip(dtp3_eff * age_program_reach[a] * booster_reach_modifier[a], 0.0, 0.95))
            target_vaccine_count = target_population * target_vaccine_share
            vaccination_flow = min(
                susceptible_naive[a],
                max(0.0, target_vaccine_count - vaccine_recent[a] - vaccine_waned[a]) * 0.25,
            )
            susceptible_naive[a] -= vaccination_flow
            vaccine_recent[a] += vaccination_flow

            age_waning_rate = waning_rate * vaccine_waning_age_modifier[a]
            vaccine_wane = min(vaccine_recent[a], vaccine_recent[a] * age_waning_rate / 12.0)
            vaccine_recent[a] -= vaccine_wane
            vaccine_waned[a] += vaccine_wane
            natural_wane = min(natural_recent[a], natural_recent[a] * natural_waning_month)
            natural_recent[a] -= natural_wane
            natural_waned[a] += natural_wane
            deep_natural_wane = min(natural_waned[a], natural_waned[a] * deep_natural_waning_month)
            natural_waned[a] -= deep_natural_wane
            susceptible_naive[a] += deep_natural_wane

            waning_multiplier = np.exp(-age_waning_rate * age_waning_years[a] / 8.0)
            recent_ve_infection = min(0.95, base_ve_infection * waning_multiplier)
            recent_ve_disease = min(0.98, base_ve_disease * waning_multiplier)
            susc_vaccine_recent = float(np.clip(1.0 - recent_ve_infection, 0.05, 1.0))
            susc_vaccine_waned = float(np.clip(1.0 - 0.45 * recent_ve_infection, 0.10, 1.0))
            symptom_naive = age_symptom_prob[a]
            symptom_vaccine_recent = age_symptom_prob[a] * float(np.clip(1.0 - 0.80 * recent_ve_disease, 0.12, 1.0))
            symptom_vaccine_waned = age_symptom_prob[a] * float(np.clip(1.0 - 0.40 * recent_ve_disease, 0.25, 1.0))
            symptom_natural_recent = age_symptom_prob[a] * 0.15
            symptom_natural_waned = age_symptom_prob[a] * 0.45
            outbreak_seeding_pressure = (
                outbreak_seeding_gain * (1.0 + 0.35 * rebound_release_gain) * release_shape
                if year_m >= 2023
                else 0.0
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
            lam = min(
                1.25,
                max(
                    0.0,
                    raw_lam * contact_multiplier,
                ),
            )
            boost_lam = lam
            boost_vaccine_lam = boost_lam
            boost_natural_lam = boost_lam
            new_naive = susceptible_naive[a] * (1.0 - np.exp(-lam))
            new_vaccine_recent = vaccine_recent[a] * (1.0 - np.exp(-lam * susc_vaccine_recent))
            new_vaccine_waned = vaccine_waned[a] * (1.0 - np.exp(-lam * susc_vaccine_waned))
            new_natural_recent = natural_recent[a] * (1.0 - np.exp(-lam * 0.08))
            new_natural_waned = natural_waned[a] * (1.0 - np.exp(-lam * 0.35))
            available_vaccine_waned = max(0.0, vaccine_waned[a] - new_vaccine_waned)
            available_natural_waned = max(0.0, natural_waned[a] - new_natural_waned)
            boost_vaccine_waned = min(
                available_vaccine_waned,
                natural_boosting_efficiency * boost_vaccine_lam * available_vaccine_waned,
            )
            boost_natural_waned = min(
                available_natural_waned,
                natural_boosting_efficiency * boost_natural_lam * available_natural_waned,
            )
            true_cases = new_naive + new_vaccine_recent + new_vaccine_waned + new_natural_recent + new_natural_waned
            symptomatic = (
                new_naive * symptom_naive
                + new_vaccine_recent * symptom_vaccine_recent
                + new_vaccine_waned * symptom_vaccine_waned
                + new_natural_recent * symptom_natural_recent
                + new_natural_waned * symptom_natural_waned
            )
            reported = symptomatic * reporting_rate * reporting_multiplier * age_report_weight[a]
            total_true[m] += true_cases
            total_symptomatic[m] += symptomatic
            total_reported[m] += reported
            exposed_old = exposed[a]
            infectious_old = infectious[a]
            progressed_exposed = exposed_progression * exposed_old
            cleared_infectious = infectious_clearance * infectious_old
            same_month_progressed = 0.35 * true_cases
            susceptible_naive[a] = max(0.0, susceptible_naive[a] - new_naive)
            vaccine_recent[a] = max(0.0, vaccine_recent[a] - new_vaccine_recent)
            vaccine_waned[a] = max(0.0, vaccine_waned[a] - new_vaccine_waned - boost_vaccine_waned)
            natural_recent[a] = max(
                0.0,
                natural_recent[a] - new_natural_recent + cleared_infectious + boost_vaccine_waned + boost_natural_waned,
            )
            natural_waned[a] = max(0.0, natural_waned[a] - new_natural_waned - boost_natural_waned)
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
            "reported_cases_predicted": total_reported,
        }
    )
    if include_predictive_replicate:
        draw_id = int(_draw_value(draw, "draw", 1.0))
        seed = int((20260612 + 1009 * draw_id + int(np.round(total_reported.sum())) % 1_000_000) % (2**32 - 1))
        rng = np.random.default_rng(seed)
        mean_reported = np.maximum(total_reported, 0.0)
        poisson_rate = rng.gamma(shape=nb_phi_month, scale=mean_reported / nb_phi_month)
        out["reported_cases_predictive"] = rng.poisson(poisson_rate).astype(float)
    return out


def _period_totals(predicted: pd.DataFrame, start_year: int = 2023, end_year: int = 2025) -> dict[str, float]:
    period = predicted.loc[predicted["year"].between(start_year, end_year)]
    return {
        "true_infections": float(period["true_infections_predicted"].sum()),
        "symptomatic_infections": float(period["symptomatic_infections_predicted"].sum()),
        "reported_cases": float(period["reported_cases_predicted"].sum()),
    }


MODEL2B_ATTRIBUTION_LAYERS = {
    "primary_susceptibility": {
        "label": "Primary susceptibility attribution",
        "scope": "primary_susceptibility",
        "factors": tuple(MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS),
        "base_factors": tuple(MODEL2B_TRANSMISSION_CONTEXT_FACTORS),
        "outcomes": ("true_infections", "symptomatic_infections"),
        "denominator_label_by_outcome": {
            "true_infections": "full_minus_susceptibility_neutral_with_fitted_background_true_infections_2023_2025",
            "symptomatic_infections": (
                "full_minus_susceptibility_neutral_with_fitted_background_symptomatic_infections_2023_2025"
            ),
        },
    },
    "transmission_context_audit": {
        "label": "Transmission context audit",
        "scope": "transmission_context_audit",
        "factors": tuple(MODEL2B_TRANSMISSION_CONTEXT_FACTORS),
        "base_factors": tuple(MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS),
        "outcomes": ("true_infections", "symptomatic_infections"),
        "denominator_label_by_outcome": {
            "true_infections": "full_minus_context_neutral_with_susceptibility_fitted_true_infections_2023_2025",
            "symptomatic_infections": (
                "full_minus_context_neutral_with_susceptibility_fitted_symptomatic_infections_2023_2025"
            ),
        },
    },
    "observation_secondary": {
        "label": "Observation secondary attribution",
        "scope": "observation_secondary",
        "factors": tuple(MODEL2B_REPORTING_FACTOR),
        "base_factors": tuple(MODEL2B_TRANSMISSION_FACTORS),
        "outcomes": ("reported_cases",),
        "denominator_label_by_outcome": {
            "reported_cases": "full_reported_minus_observation_neutral_with_transmission_fitted_reported_cases_2023_2025",
        },
    },
}


def _factor_spec(factor_id: str) -> dict[str, str]:
    return MODEL2B_REPORTED_FACTORS[factor_id]


def _predict_period_totals(data: dict[str, Any], draw: pd.Series, active_factors: set[str]) -> dict[str, float]:
    totals = _predict_monthly(data, draw, active_factors=active_factors, totals_only=True)
    return dict(totals)


def _exact_shapley_for_draw(
    data: dict[str, Any],
    draw: pd.Series,
    _unused_factors: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_id, layer in MODEL2B_ATTRIBUTION_LAYERS.items():
        factors = list(layer["factors"])
        base_factors = set(layer["base_factors"])
        n = len(factors)
        values: dict[frozenset[str], dict[str, float]] = {}
        for size in range(n + 1):
            for combo in combinations(factors, size):
                subset = frozenset(combo)
                active = base_factors | set(subset)
                values[subset] = _predict_period_totals(data, draw, active_factors=active)
        full_subset = frozenset(factors)
        for outcome in layer["outcomes"]:
            label = MODEL2B_OUTCOMES[outcome]
            full_model = values[full_subset][outcome]
            neutral_baseline = values[frozenset()][outcome]
            denominator = full_model - neutral_baseline
            denominator_label = layer["denominator_label_by_outcome"][outcome]
            for factor_id in factors:
                remaining = [item for item in factors if item != factor_id]
                contribution = 0.0
                for size in range(n):
                    for combo in combinations(remaining, size):
                        subset = frozenset(combo)
                        weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                        contribution += weight * (
                            values[subset | {factor_id}][outcome] - values[subset][outcome]
                        )
                factor = _factor_spec(factor_id)
                rows.append(
                    {
                        "draw": int(draw["draw"]),
                        "factor_id": factor_id,
                        "factor_label": factor["label"],
                        "factor_family": factor["family"],
                        "mechanism": factor_id,
                        "mechanism_label": factor["label"],
                        "attribution_scope": layer["scope"],
                        "outcome_target": label,
                        "contribution_cases_2023_2025": contribution,
                        "contribution_fraction": contribution / denominator if denominator > 0 else np.nan,
                        "full_model_cases_2023_2025": full_model,
                        "no_factors_cases_2023_2025": neutral_baseline,
                        "no_mechanisms_cases_2023_2025": neutral_baseline,
                        "neutral_baseline_cases_2023_2025": neutral_baseline,
                        "full_minus_no_factors_cases_2023_2025": denominator,
                        "full_minus_no_mechanisms_cases_2023_2025": denominator,
                        "total_excess_cases_2023_2025": denominator,
                        "counterfactual_definition": factor["definition"],
                        "denominator_label": denominator_label,
                        "attribution_layer": layer_id,
                        "attribution_denominator": denominator_label,
                        "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                    }
                )
    return rows


def model2b_shapley_draws(data: dict[str, Any], draws: pd.DataFrame) -> pd.DataFrame:
    workers = min(
        len(draws),
        _env_int("MODEL2B_ATTRIBUTION_WORKERS_PER_COUNTRY", 1),
    )
    if workers > 1 and len(draws) >= workers:
        records = draws.to_dict("records")
        chunk_size = int(np.ceil(len(records) / workers))
        chunks = [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parts = executor.map(_model2b_shapley_chunk, [(data, chunk) for chunk in chunks])
        rows = [row for part in parts for row in part]
        return pd.DataFrame(rows)
    rows: list[dict[str, Any]] = []
    for draw in draws.itertuples(index=False):
        draw_series = pd.Series(draw._asdict())
        rows.extend(_exact_shapley_for_draw(data, draw_series))
    return pd.DataFrame(rows)


def _model2b_shapley_chunk(payload: tuple[dict[str, Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    data, records = payload
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(_exact_shapley_for_draw(data, pd.Series(record)))
    return rows


def summarize_model2b_factor_weights(
    shapley_draws: pd.DataFrame,
    iso3: str,
    country: str,
    *,
    attribution_layers: set[str] | str | None = None,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "factor_id",
        "factor_label",
        "factor_family",
        "attribution_scope",
        "outcome_target",
        "counterfactual_definition",
        "denominator_label",
        "share_percent_basis",
        "contribution_median",
        "contribution_lower_95",
        "contribution_upper_95",
        "share_percent_median",
        "share_percent_lower_95",
        "share_percent_upper_95",
        "signed_share_percent_median",
        "signed_share_percent_lower_95",
        "signed_share_percent_upper_95",
        "positive_normalized_share_percent_median",
        "positive_normalized_share_percent_lower_95",
        "positive_normalized_share_percent_upper_95",
        "full_model_cases_median",
        "neutral_baseline_cases_median",
        "full_minus_no_factors_cases_median",
        "full_minus_no_mechanisms_cases_median",
        "total_excess_cases_median",
        "positive_contribution_sum_median",
        "negative_contribution_sum_median",
        "dominant_probability",
        "n_posterior_draws",
        "attribution_layer",
        "model2b_output_version",
        "interpretation_note",
    ]
    if shapley_draws.empty:
        return pd.DataFrame(columns=columns)
    frame = shapley_draws.copy()
    if "attribution_layer" in frame:
        if attribution_layers is None:
            allowed_layers: set[str] | None = {"primary_susceptibility"}
        elif attribution_layers == "all":
            allowed_layers = None
        elif isinstance(attribution_layers, str):
            allowed_layers = {attribution_layers}
        else:
            allowed_layers = set(attribution_layers)
        if allowed_layers is not None:
            frame = frame.loc[frame["attribution_layer"].astype(str).isin(allowed_layers)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if "factor_id" not in frame and "mechanism" in frame:
        frame["factor_id"] = frame["mechanism"].map(lambda value: MODEL2B_FACTOR_ALIASES.get(str(value), str(value)))
    elif "factor_id" in frame:
        frame["factor_id"] = frame["factor_id"].map(lambda value: MODEL2B_FACTOR_ALIASES.get(str(value), str(value)))
    frame = frame.loc[frame["factor_id"].map(lambda value: isinstance(value, str))].copy()
    frame["contribution_cases_2023_2025"] = pd.to_numeric(
        frame["contribution_cases_2023_2025"],
        errors="coerce",
    )
    frame["full_model_cases_2023_2025"] = pd.to_numeric(
        frame["full_model_cases_2023_2025"],
        errors="coerce",
    )
    frame["full_minus_no_mechanisms_cases_2023_2025"] = pd.to_numeric(
        frame["full_minus_no_mechanisms_cases_2023_2025"],
        errors="coerce",
    )
    for column in ["no_factors_cases_2023_2025", "no_mechanisms_cases_2023_2025", "total_excess_cases_2023_2025"]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "full_minus_no_factors_cases_2023_2025" not in frame:
        frame["full_minus_no_factors_cases_2023_2025"] = frame["full_minus_no_mechanisms_cases_2023_2025"]
    frame["full_minus_no_factors_cases_2023_2025"] = pd.to_numeric(
        frame["full_minus_no_factors_cases_2023_2025"],
        errors="coerce",
    )
    for column in [
        "factor_family",
        "attribution_scope",
        "counterfactual_definition",
        "denominator_label",
        "attribution_layer",
    ]:
        if column not in frame:
            frame[column] = ""
    frame["positive_contribution"] = frame["contribution_cases_2023_2025"].clip(lower=0.0)
    frame["negative_contribution"] = frame["contribution_cases_2023_2025"].clip(upper=0.0)
    contribution_mass = (
        frame.groupby(["draw", "outcome_target", "attribution_layer"], dropna=False)
        .agg(
            positive_sum=("positive_contribution", "sum"),
            negative_sum=("negative_contribution", "sum"),
        )
        .reset_index()
    )
    frame = frame.merge(contribution_mass, on=["draw", "outcome_target", "attribution_layer"], how="left")
    excess_denominator = frame["total_excess_cases_2023_2025"].where(frame["total_excess_cases_2023_2025"].gt(0))
    if excess_denominator.isna().all():
        excess_denominator = frame["full_minus_no_mechanisms_cases_2023_2025"].where(
            frame["full_minus_no_mechanisms_cases_2023_2025"].gt(0)
        )
    frame["signed_share_percent"] = 100.0 * frame["contribution_cases_2023_2025"] / excess_denominator
    frame["positive_normalized_share_percent"] = np.where(
        frame["positive_sum"].gt(0),
        100.0 * frame["positive_contribution"] / frame["positive_sum"],
        np.nan,
    )
    frame["share_percent_basis"] = np.where(
        frame["attribution_layer"].astype(str).eq("primary_susceptibility"),
        "positive_normalized_primary_susceptibility_share",
        "signed_layer_denominator_share",
    )
    frame["share_percent"] = np.where(
        frame["attribution_layer"].astype(str).eq("primary_susceptibility"),
        frame["positive_normalized_share_percent"],
        frame["signed_share_percent"],
    )
    dominant = (
        frame.loc[
            frame["outcome_target"].eq(MODEL2B_OUTCOMES["true_infections"])
            & frame["contribution_cases_2023_2025"].gt(0)
        ]
        .sort_values(["draw", "attribution_layer", "positive_contribution"], ascending=[True, True, False], kind="stable")
        .groupby(["draw", "attribution_layer"], dropna=False)
        .head(1)
        .loc[:, ["draw", "attribution_layer", "factor_id"]]
        .rename(columns={"factor_id": "dominant_factor_id"})
    )
    frame = frame.merge(dominant, on=["draw", "attribution_layer"], how="left")
    frame["is_dominant"] = frame["factor_id"].eq(frame["dominant_factor_id"]) & frame["outcome_target"].eq(
        MODEL2B_OUTCOMES["true_infections"]
    )
    rows: list[dict[str, Any]] = []
    for (factor_id, outcome, attribution_layer), group in frame.groupby(
        ["factor_id", "outcome_target", "attribution_layer"],
        sort=False,
    ):
        contribution = pd.to_numeric(group["contribution_cases_2023_2025"], errors="coerce")
        share = pd.to_numeric(group["share_percent"], errors="coerce")
        signed_share = pd.to_numeric(group["signed_share_percent"], errors="coerce")
        positive_normalized_share = pd.to_numeric(group["positive_normalized_share_percent"], errors="coerce")
        full_model_cases = pd.to_numeric(group["full_model_cases_2023_2025"], errors="coerce")
        full_minus_no_factors = pd.to_numeric(
            group["full_minus_no_factors_cases_2023_2025"],
            errors="coerce",
        )
        full_minus_no_mechanisms = pd.to_numeric(
            group["full_minus_no_mechanisms_cases_2023_2025"],
            errors="coerce",
        )
        neutral_baseline_cases = pd.to_numeric(group["no_mechanisms_cases_2023_2025"], errors="coerce")
        total_excess = pd.to_numeric(group["total_excess_cases_2023_2025"], errors="coerce")
        positive_sum = pd.to_numeric(group["positive_sum"], errors="coerce")
        negative_sum = pd.to_numeric(group["negative_sum"], errors="coerce")
        dominant_probability = float(group["is_dominant"].mean()) if outcome == MODEL2B_OUTCOMES["true_infections"] else np.nan
        factor = _factor_spec(factor_id)
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "factor_id": factor_id,
                "factor_label": factor["label"],
                "factor_family": factor["family"],
                "attribution_scope": str(group["attribution_scope"].dropna().iloc[0]),
                "outcome_target": outcome,
                "counterfactual_definition": factor["definition"],
                "denominator_label": str(group["denominator_label"].dropna().iloc[0]),
                "share_percent_basis": str(group["share_percent_basis"].dropna().iloc[0]),
                "contribution_median": float(contribution.median()),
                "contribution_lower_95": float(contribution.quantile(0.025)),
                "contribution_upper_95": float(contribution.quantile(0.975)),
                "share_percent_median": float(share.median()) if share.notna().any() else np.nan,
                "share_percent_lower_95": float(share.quantile(0.025)) if share.notna().any() else np.nan,
                "share_percent_upper_95": float(share.quantile(0.975)) if share.notna().any() else np.nan,
                "signed_share_percent_median": (
                    float(signed_share.median()) if signed_share.notna().any() else np.nan
                ),
                "signed_share_percent_lower_95": (
                    float(signed_share.quantile(0.025)) if signed_share.notna().any() else np.nan
                ),
                "signed_share_percent_upper_95": (
                    float(signed_share.quantile(0.975)) if signed_share.notna().any() else np.nan
                ),
                "positive_normalized_share_percent_median": (
                    float(positive_normalized_share.median()) if positive_normalized_share.notna().any() else np.nan
                ),
                "positive_normalized_share_percent_lower_95": (
                    float(positive_normalized_share.quantile(0.025))
                    if positive_normalized_share.notna().any()
                    else np.nan
                ),
                "positive_normalized_share_percent_upper_95": (
                    float(positive_normalized_share.quantile(0.975))
                    if positive_normalized_share.notna().any()
                    else np.nan
                ),
                "full_model_cases_median": (
                    float(full_model_cases.median()) if full_model_cases.notna().any() else np.nan
                ),
                "neutral_baseline_cases_median": (
                    float(neutral_baseline_cases.median()) if neutral_baseline_cases.notna().any() else np.nan
                ),
                "full_minus_no_factors_cases_median": (
                    float(full_minus_no_factors.median()) if full_minus_no_factors.notna().any() else np.nan
                ),
                "full_minus_no_mechanisms_cases_median": (
                    float(full_minus_no_mechanisms.median()) if full_minus_no_mechanisms.notna().any() else np.nan
                ),
                "total_excess_cases_median": (
                    float(total_excess.median()) if total_excess.notna().any() else np.nan
                ),
                "positive_contribution_sum_median": (
                    float(positive_sum.median()) if positive_sum.notna().any() else np.nan
                ),
                "negative_contribution_sum_median": (
                    float(negative_sum.median()) if negative_sum.notna().any() else np.nan
                ),
                "dominant_probability": dominant_probability,
                "n_posterior_draws": int(group["draw"].nunique()),
                "attribution_layer": str(attribution_layer),
                "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                "interpretation_note": (
                    "Model-implied layered interventional Shapley attribution under fitted transmission-dynamic "
                    "assumptions. Primary true/symptomatic infection share_percent columns are bounded positive-"
                    "normalized shares among factors with positive contributions; signed denominator shares are "
                    "retained in signed_share_percent columns. Transmission-context and observation factors are "
                    "interpreted in their own audit layers."
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_model2b_layered_attribution_summary(
    shapley_draws: pd.DataFrame,
    iso3: str,
    country: str,
) -> pd.DataFrame:
    return summarize_model2b_factor_weights(shapley_draws, iso3, country, attribution_layers="all")


def summarize_model2b_pooled_attribution(shapley_draws: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor_id",
        "factor_label",
        "factor_family",
        "attribution_layer",
        "attribution_scope",
        "outcome_target",
        "pooled_contribution_cases_median",
        "pooled_contribution_cases_lower_95",
        "pooled_contribution_cases_upper_95",
        "pooled_share_percent_median",
        "pooled_share_percent_lower_95",
        "pooled_share_percent_upper_95",
        "signed_pooled_share_percent_median",
        "signed_pooled_share_percent_lower_95",
        "signed_pooled_share_percent_upper_95",
        "positive_normalized_pooled_share_percent_median",
        "positive_normalized_pooled_share_percent_lower_95",
        "positive_normalized_pooled_share_percent_upper_95",
        "pooled_share_percent_basis",
        "n_countries",
        "n_posterior_draws",
        "pooling_method",
        "model2b_output_version",
        "interpretation_note",
    ]
    if shapley_draws.empty:
        return pd.DataFrame(columns=columns)
    frame = shapley_draws.copy()
    required = {
        "iso3",
        "draw",
        "factor_id",
        "factor_label",
        "factor_family",
        "attribution_layer",
        "attribution_scope",
        "outcome_target",
        "contribution_cases_2023_2025",
        "full_minus_no_factors_cases_2023_2025",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Pooled Model 2B attribution is missing columns: " + ", ".join(sorted(missing)))
    frame["contribution_cases_2023_2025"] = pd.to_numeric(
        frame["contribution_cases_2023_2025"],
        errors="coerce",
    )
    frame["full_minus_no_factors_cases_2023_2025"] = pd.to_numeric(
        frame["full_minus_no_factors_cases_2023_2025"],
        errors="coerce",
    )
    denominators = (
        frame.loc[:, ["iso3", "draw", "outcome_target", "attribution_layer", "full_minus_no_factors_cases_2023_2025"]]
        .drop_duplicates()
        .groupby(["draw", "outcome_target", "attribution_layer"], dropna=False, sort=False)
        .agg(pooled_denominator=("full_minus_no_factors_cases_2023_2025", "sum"), n_countries=("iso3", "nunique"))
        .reset_index()
    )
    pooled = (
        frame.groupby(
            [
                "draw",
                "factor_id",
                "factor_label",
                "factor_family",
                "attribution_layer",
                "attribution_scope",
                "outcome_target",
            ],
            dropna=False,
            sort=False,
        )
        .agg(pooled_contribution=("contribution_cases_2023_2025", "sum"))
        .reset_index()
        .merge(denominators, on=["draw", "outcome_target", "attribution_layer"], how="left")
    )
    pooled["signed_pooled_share_percent"] = np.where(
        pooled["pooled_denominator"].gt(0),
        100.0 * pooled["pooled_contribution"] / pooled["pooled_denominator"],
        np.nan,
    )
    pooled["pooled_positive_contribution"] = pooled["pooled_contribution"].clip(lower=0.0)
    pooled_positive_mass = (
        pooled.groupby(["draw", "outcome_target", "attribution_layer"], dropna=False, sort=False)[
            "pooled_positive_contribution"
        ]
        .sum()
        .rename("pooled_positive_sum")
        .reset_index()
    )
    pooled = pooled.merge(pooled_positive_mass, on=["draw", "outcome_target", "attribution_layer"], how="left")
    pooled["positive_normalized_pooled_share_percent"] = np.where(
        pooled["pooled_positive_sum"].gt(0),
        100.0 * pooled["pooled_positive_contribution"] / pooled["pooled_positive_sum"],
        np.nan,
    )
    pooled["pooled_share_percent_basis"] = np.where(
        pooled["attribution_layer"].astype(str).eq("primary_susceptibility"),
        "positive_normalized_primary_susceptibility_share",
        "signed_layer_denominator_share",
    )
    pooled["pooled_share_percent"] = np.where(
        pooled["attribution_layer"].astype(str).eq("primary_susceptibility"),
        pooled["positive_normalized_pooled_share_percent"],
        pooled["signed_pooled_share_percent"],
    )
    rows: list[dict[str, Any]] = []
    for keys, group in pooled.groupby(
        ["factor_id", "factor_label", "factor_family", "attribution_layer", "attribution_scope", "outcome_target"],
        dropna=False,
        sort=False,
    ):
        factor_id, factor_label, factor_family, attribution_layer, attribution_scope, outcome = keys
        contribution = pd.to_numeric(group["pooled_contribution"], errors="coerce")
        share = pd.to_numeric(group["pooled_share_percent"], errors="coerce")
        signed_share = pd.to_numeric(group["signed_pooled_share_percent"], errors="coerce")
        positive_normalized_share = pd.to_numeric(
            group["positive_normalized_pooled_share_percent"],
            errors="coerce",
        )
        rows.append(
            {
                "factor_id": factor_id,
                "factor_label": factor_label,
                "factor_family": factor_family,
                "attribution_layer": attribution_layer,
                "attribution_scope": attribution_scope,
                "outcome_target": outcome,
                "pooled_contribution_cases_median": float(contribution.median()),
                "pooled_contribution_cases_lower_95": float(contribution.quantile(0.025)),
                "pooled_contribution_cases_upper_95": float(contribution.quantile(0.975)),
                "pooled_share_percent_median": float(share.median()) if share.notna().any() else np.nan,
                "pooled_share_percent_lower_95": float(share.quantile(0.025)) if share.notna().any() else np.nan,
                "pooled_share_percent_upper_95": float(share.quantile(0.975)) if share.notna().any() else np.nan,
                "signed_pooled_share_percent_median": (
                    float(signed_share.median()) if signed_share.notna().any() else np.nan
                ),
                "signed_pooled_share_percent_lower_95": (
                    float(signed_share.quantile(0.025)) if signed_share.notna().any() else np.nan
                ),
                "signed_pooled_share_percent_upper_95": (
                    float(signed_share.quantile(0.975)) if signed_share.notna().any() else np.nan
                ),
                "positive_normalized_pooled_share_percent_median": (
                    float(positive_normalized_share.median()) if positive_normalized_share.notna().any() else np.nan
                ),
                "positive_normalized_pooled_share_percent_lower_95": (
                    float(positive_normalized_share.quantile(0.025))
                    if positive_normalized_share.notna().any()
                    else np.nan
                ),
                "positive_normalized_pooled_share_percent_upper_95": (
                    float(positive_normalized_share.quantile(0.975))
                    if positive_normalized_share.notna().any()
                    else np.nan
                ),
                "pooled_share_percent_basis": str(group["pooled_share_percent_basis"].dropna().iloc[0]),
                "n_countries": int(group["n_countries"].max()),
                "n_posterior_draws": int(group["draw"].nunique()),
                "pooling_method": "sum_country_contributions_over_sum_country_complete_denominators",
                "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                "interpretation_note": (
                    "Nine-country pooled model-implied theoretical attribution. Primary pooled share_percent "
                    "columns are bounded positive-normalized shares; signed denominator shares are retained in "
                    "signed_pooled_share_percent columns. This is not a globally representative causal percentage."
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def model2b_observation_audit(data: dict[str, Any], metadata: dict[str, Any], draws: pd.DataFrame) -> pd.DataFrame:
    def posterior_numeric(name: str, default: float) -> pd.Series:
        if name in draws:
            values = pd.to_numeric(draws[name], errors="coerce")
            return values.fillna(default)
        return pd.Series(np.repeat(default, max(len(draws), 1)))

    reporting = np.exp(posterior_numeric("log_post_reporting_multiplier", 0.0))
    transmission = np.exp(posterior_numeric("log_post_transmission_multiplier", 0.0))
    pathogen_pressure_scale = posterior_numeric(
        "excess_epidemic_pressure_scale",
        np.nan,
    )
    if pathogen_pressure_scale.isna().all():
        pathogen_pressure_scale = posterior_numeric("pathogen_pressure_scale", 0.0)
    outbreak_seeding_gain = posterior_numeric(
        "outbreak_ignition_pulse_gain",
        np.nan,
    )
    if outbreak_seeding_gain.isna().all():
        outbreak_seeding_gain = posterior_numeric("outbreak_seeding_gain", 0.0)
    nb_phi_month = posterior_numeric("nb_phi_month", np.nan)
    if nb_phi_month.isna().all():
        nb_phi_month = posterior_numeric("nb_phi", 20.0)
    nb_phi_age = posterior_numeric("nb_phi_age", np.nan)
    missed_natural_boosting = posterior_numeric("missed_natural_boosting_2020_2022", np.nan)
    natural_boosting_flow = posterior_numeric("natural_boosting_flow_2020_2022", np.nan)
    routine_vaccination_shortfall = posterior_numeric("routine_vaccination_shortfall_2020_2022", np.nan)
    natural_waned_stock = posterior_numeric("natural_waned_stock_2023_start", np.nan)
    direct_support = _metadata_bool(metadata.get("direct_testing_support", False))
    reporting_shift_probability = float((np.abs(np.log(reporting.clip(lower=1e-9))) > np.log(1.25)).mean())
    transmission_context_shift_probability = float((np.abs(np.log(transmission.clip(lower=1e-9))) > np.log(1.25)).mean())
    pathogen_pressure_active_probability = float((pathogen_pressure_scale > 0.5).mean())
    outbreak_seeding_active_probability = float((outbreak_seeding_gain > 0.02).mean())
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
                "model2b_contact_matrix_source": metadata.get("model2b_contact_matrix_source", "rank_one_contact_vector_outer_product"),
                "model2b_contact_matrix_fallback": _metadata_bool(metadata.get("model2b_contact_matrix_fallback", False)),
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
                "post_transmission_multiplier_median": float(transmission.median()),
                "post_transmission_multiplier_lower_95": float(transmission.quantile(0.025)),
                "post_transmission_multiplier_upper_95": float(transmission.quantile(0.975)),
                "pathogen_pressure_scale_median": float(pathogen_pressure_scale.median()),
                "pathogen_pressure_scale_lower_95": float(pathogen_pressure_scale.quantile(0.025)),
                "pathogen_pressure_scale_upper_95": float(pathogen_pressure_scale.quantile(0.975)),
                "outbreak_seeding_gain_median": float(outbreak_seeding_gain.median()),
                "outbreak_seeding_gain_lower_95": float(outbreak_seeding_gain.quantile(0.025)),
                "outbreak_seeding_gain_upper_95": float(outbreak_seeding_gain.quantile(0.975)),
                "nb_phi_month_median": float(nb_phi_month.median()),
                "nb_phi_month_lower_95": float(nb_phi_month.quantile(0.025)),
                "nb_phi_month_upper_95": float(nb_phi_month.quantile(0.975)),
                "nb_phi_age_median": float(nb_phi_age.median()) if nb_phi_age.notna().any() else np.nan,
                "nb_phi_age_lower_95": float(nb_phi_age.quantile(0.025)) if nb_phi_age.notna().any() else np.nan,
                "nb_phi_age_upper_95": float(nb_phi_age.quantile(0.975)) if nb_phi_age.notna().any() else np.nan,
                "missed_natural_boosting_2020_2022_median": (
                    float(missed_natural_boosting.median()) if missed_natural_boosting.notna().any() else np.nan
                ),
                "natural_boosting_flow_2020_2022_median": (
                    float(natural_boosting_flow.median()) if natural_boosting_flow.notna().any() else np.nan
                ),
                "routine_vaccination_shortfall_2020_2022_median": (
                    float(routine_vaccination_shortfall.median()) if routine_vaccination_shortfall.notna().any() else np.nan
                ),
                "natural_waned_stock_2023_start_median": (
                    float(natural_waned_stock.median()) if natural_waned_stock.notna().any() else np.nan
                ),
                "posterior_reporting_shift_probability": reporting_shift_probability,
                "posterior_transmission_context_shift_probability": transmission_context_shift_probability,
                "posterior_transmission_recovery_probability": transmission_context_shift_probability,
                "posterior_residual_transmission_probability": transmission_context_shift_probability,
                "posterior_pathogen_pressure_active_probability": pathogen_pressure_active_probability,
                "excess_epidemic_pressure_scale_median": float(pathogen_pressure_scale.median()),
                "excess_epidemic_pressure_scale_lower_95": float(pathogen_pressure_scale.quantile(0.025)),
                "excess_epidemic_pressure_scale_upper_95": float(pathogen_pressure_scale.quantile(0.975)),
                "posterior_outbreak_seeding_active_probability": outbreak_seeding_active_probability,
                "posterior_excess_epidemic_pressure_active_probability": pathogen_pressure_active_probability,
                "outbreak_ignition_pulse_gain_median": float(outbreak_seeding_gain.median()),
                "outbreak_ignition_pulse_gain_lower_95": float(outbreak_seeding_gain.quantile(0.025)),
                "outbreak_ignition_pulse_gain_upper_95": float(outbreak_seeding_gain.quantile(0.975)),
                "posterior_outbreak_ignition_pulse_active_probability": outbreak_seeding_active_probability,
                "month_obs_weight_min": metadata.get("model2b_month_obs_weight_min", np.nan),
                "month_obs_weight_mean": metadata.get("model2b_month_obs_weight_mean", np.nan),
                "month_obs_weight_max": metadata.get("model2b_month_obs_weight_max", np.nan),
                "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                "observation_process_supported": direct_support,
                "observation_process_role": (
                    "supported_observation_process_sensitivity"
                    if direct_support
                    else "constrained_nuisance_not_primary_driver"
                ),
                "interpretation_note": (
                    "Reporting/testing is a posterior observation-process quantity. It is not included in "
                    "primary transmission-driver shares. Missed exposure boosting and routine vaccination "
                    "shortfall are posterior state-flow diagnostics. The post-pandemic transmission multiplier, "
                    "excess epidemic pressure scale and outbreak ignition pulse are fitted transmission-context "
                    "audit quantities. Endemic background circulation remains active in all primary susceptibility "
                    "counterfactuals."
                ),
            }
        ]
    )


def model2b_transmission_context_audit(data: dict[str, Any], metadata: dict[str, Any], draws: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "context_factor_id",
        "context_factor_label",
        "factor_family",
        "outcome_target",
        "full_model_cases_median",
        "factor_neutral_cases_median",
        "context_neutral_cases_median",
        "context_denominator_cases_median",
        "contribution_cases_median",
        "contribution_cases_lower_95",
        "contribution_cases_upper_95",
        "share_percent_median",
        "share_percent_lower_95",
        "share_percent_upper_95",
        "n_posterior_draws",
        "audit_layer",
        "model2b_output_version",
        "interpretation_note",
    ]
    if draws.empty:
        return pd.DataFrame(columns=columns)
    primary_factors = set(MODEL2B_PRIMARY_SUSCEPTIBILITY_FACTORS)
    context_factors = set(MODEL2B_TRANSMISSION_CONTEXT_FACTORS)
    full_factors = primary_factors | context_factors
    rows: list[dict[str, Any]] = []
    for draw in draws.itertuples(index=False):
        draw_series = pd.Series(draw._asdict())
        full = _predict_period_totals(data, draw_series, full_factors)
        context_neutral = _predict_period_totals(data, draw_series, primary_factors)
        for outcome in ("true_infections", "symptomatic_infections"):
            label = MODEL2B_OUTCOMES[outcome]
            denominator = full[outcome] - context_neutral[outcome]
            for factor_id, factor in MODEL2B_CONTEXT_AUDIT_FACTORS.items():
                factor_neutral = _predict_period_totals(data, draw_series, full_factors - {factor_id})
                contribution = full[outcome] - factor_neutral[outcome]
                rows.append(
                    {
                        "draw": int(draw_series["draw"]),
                        "context_factor_id": factor_id,
                        "context_factor_label": factor["label"],
                        "factor_family": factor["family"],
                        "outcome_target": label,
                        "full_model_cases": full[outcome],
                        "factor_neutral_cases": factor_neutral[outcome],
                        "context_neutral_cases": context_neutral[outcome],
                        "context_denominator_cases": denominator,
                        "contribution_cases": contribution,
                        "share_percent": 100.0 * contribution / denominator if denominator > 0 else np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (factor_id, outcome), group in frame.groupby(["context_factor_id", "outcome_target"], sort=False):
        contribution = pd.to_numeric(group["contribution_cases"], errors="coerce")
        share = pd.to_numeric(group["share_percent"], errors="coerce")
        factor = MODEL2B_CONTEXT_AUDIT_FACTORS[factor_id]
        summaries.append(
            {
                "iso3": metadata["iso3"],
                "country": metadata["country"],
                "context_factor_id": factor_id,
                "context_factor_label": factor["label"],
                "factor_family": factor["family"],
                "outcome_target": outcome,
                "full_model_cases_median": float(group["full_model_cases"].median()),
                "factor_neutral_cases_median": float(group["factor_neutral_cases"].median()),
                "context_neutral_cases_median": float(group["context_neutral_cases"].median()),
                "context_denominator_cases_median": float(group["context_denominator_cases"].median()),
                "contribution_cases_median": float(contribution.median()),
                "contribution_cases_lower_95": float(contribution.quantile(0.025)),
                "contribution_cases_upper_95": float(contribution.quantile(0.975)),
                "share_percent_median": float(share.median()) if share.notna().any() else np.nan,
                "share_percent_lower_95": (
                    float(share.quantile(0.025)) if share.notna().any() else np.nan
                ),
                "share_percent_upper_95": (
                    float(share.quantile(0.975)) if share.notna().any() else np.nan
                ),
                "n_posterior_draws": int(group["draw"].nunique()),
                "audit_layer": "transmission_context_audit_one_factor_neutralized",
                "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                "interpretation_note": (
                    "Transmission-context audit under the fitted Model 2B v2 assumptions. These contrasts report "
                    "dependence on post-2023 transmission shift, excess epidemic pressure or outbreak ignition "
                    "pulse while endemic background circulation remains active; they are not primary susceptibility "
                    "driver shares."
                ),
            }
        )
    return pd.DataFrame(summaries, columns=columns)


def model2b_posterior_predictive(data: dict[str, Any], metadata: dict[str, Any], draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for draw in draws.itertuples(index=False):
        draw_series = pd.Series(draw._asdict())
        pred = _predict_monthly(data, draw_series, include_predictive_replicate=True)
        pred.insert(0, "draw", int(draw_series["draw"]))
        pred.insert(0, "country", metadata["country"])
        pred.insert(0, "iso3", metadata["iso3"])
        pred["fit_role"] = np.where(pred["year"].le(2025), "fit_likelihood", "posterior_predictive_check")
        pred["model2b_output_version"] = MODEL2B_OUTPUT_VERSION
        rows.append(pred)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _select_scenario_draws(draws: pd.DataFrame, max_draws: int | None) -> pd.DataFrame:
    if draws.empty:
        return draws.copy()
    if max_draws is None:
        max_draws = _env_int("MODEL2B_SCENARIO_MAX_DRAWS", 160)
    max_draws = max(1, int(max_draws))
    if len(draws) <= max_draws:
        return draws.copy()
    positions = np.unique(np.linspace(0, len(draws) - 1, max_draws, dtype=int))
    return draws.iloc[positions].copy()


def _first_non_missing(values: pd.Series) -> float:
    observed = pd.to_numeric(values, errors="coerce").dropna()
    return float(observed.iloc[0]) if not observed.empty else np.nan


def _scenario_quantile(probability: float):
    def summarise(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return float(numeric.quantile(probability)) if not numeric.empty else np.nan

    return summarise


def model2b_counterfactual_scenarios(
    data: dict[str, Any],
    metadata: dict[str, Any],
    draws: pd.DataFrame,
    *,
    max_draws: int | None = None,
    start_year: int = 2020,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Summarise fitted-posterior dynamic predictions under one-mechanism-removed scenarios."""

    columns = [
        "iso3",
        "country",
        "scenario_id",
        "scenario_label",
        "scenario_order",
        "scenario_removed_mechanisms",
        "scenario_note",
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
        "n_posterior_draws",
        "prediction_layer",
        "model2b_output_version",
    ]
    selected = _select_scenario_draws(draws, max_draws)
    if selected.empty:
        return pd.DataFrame(columns=columns)

    all_factors = set(MODEL2B_REPORTED_FACTORS)
    predictions: list[pd.DataFrame] = []
    for fallback_draw_id, (_, draw) in enumerate(selected.iterrows(), start=1):
        draw_series = draw.copy()
        if "draw" not in draw_series or pd.isna(draw_series.get("draw")):
            draw_series["draw"] = fallback_draw_id
        draw_id = int(draw_series["draw"])
        for scenario_id, scenario in MODEL2B_COUNTERFACTUAL_SCENARIOS.items():
            removed = set(scenario["removed_mechanisms"])
            active = all_factors - removed
            pred = _predict_monthly(
                data,
                draw_series,
                active_factors=active,
            )
            pred = pred.loc[pred["year"].between(start_year, end_year)].copy()
            pred.insert(0, "draw", draw_id)
            pred.insert(0, "scenario_id", scenario_id)
            pred["scenario_label"] = scenario["scenario_label"]
            pred["scenario_order"] = int(scenario["scenario_order"])
            removed_labels = list(scenario["removed_mechanisms"])
            pred["scenario_removed_mechanisms"] = ";".join(removed_labels) or "none"
            pred["scenario_note"] = scenario["scenario_note"]
            predictions.append(pred)

    if not predictions:
        return pd.DataFrame(columns=columns)

    frame = pd.concat(predictions, ignore_index=True, sort=False)
    frame.insert(0, "country", metadata["country"])
    frame.insert(0, "iso3", metadata["iso3"])
    group_cols = [
        "iso3",
        "country",
        "scenario_id",
        "scenario_label",
        "scenario_order",
        "scenario_removed_mechanisms",
        "scenario_note",
        "year",
        "month",
    ]
    summary = (
        frame.groupby(group_cols, dropna=False, sort=False)
        .agg(
            observed_cases=("observed_cases", _first_non_missing),
            true_infections_median=("true_infections_predicted", "median"),
            true_infections_lower_95=("true_infections_predicted", _scenario_quantile(0.025)),
            true_infections_upper_95=("true_infections_predicted", _scenario_quantile(0.975)),
            symptomatic_infections_median=("symptomatic_infections_predicted", "median"),
            symptomatic_infections_lower_95=("symptomatic_infections_predicted", _scenario_quantile(0.025)),
            symptomatic_infections_upper_95=("symptomatic_infections_predicted", _scenario_quantile(0.975)),
            reported_cases_median=("reported_cases_predicted", "median"),
            reported_cases_lower_95=("reported_cases_predicted", _scenario_quantile(0.025)),
            reported_cases_upper_95=("reported_cases_predicted", _scenario_quantile(0.975)),
            n_posterior_draws=("reported_cases_predicted", "size"),
        )
        .reset_index()
    )
    summary["date"] = pd.to_datetime(
        {
            "year": summary["year"].astype(int),
            "month": summary["month"].astype(int),
            "day": 15,
        }
    ).dt.strftime("%Y-%m-%d")
    summary["fit_role"] = np.where(summary["year"].le(2025), "fit_likelihood", "posterior_predictive_check")
    summary["prediction_layer"] = "model2b_fitted_posterior_counterfactual_scenario"
    summary["model2b_output_version"] = MODEL2B_OUTPUT_VERSION
    return summary.loc[:, columns]


def model2b_country_outputs(
    data: dict[str, Any],
    metadata: dict[str, Any],
    draws: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shapley = model2b_shapley_draws(data, draws)
    if not shapley.empty:
        shapley.insert(0, "country", metadata["country"])
        shapley.insert(0, "iso3", metadata["iso3"])
    summary = summarize_model2b_factor_weights(shapley, metadata["iso3"], metadata["country"])
    layered_summary = summarize_model2b_layered_attribution_summary(shapley, metadata["iso3"], metadata["country"])
    audit = model2b_observation_audit(data, metadata, draws)
    residual_audit = model2b_transmission_context_audit(data, metadata, draws)
    ppc = model2b_posterior_predictive(data, metadata, draws)
    return shapley, summary, layered_summary, audit, residual_audit, ppc
