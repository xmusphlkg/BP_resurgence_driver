from __future__ import annotations

import os
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.data.model2_observations import monthly_detection_prior_table
from pertussis_global_dynamics.data.model2_inputs import baseline_maternal_coverage_from_metadata, periodic_baseline
from pertussis_global_dynamics.data.model2_inputs import model2_country_limit, worker_count
from pertussis_global_dynamics.model2_constants import (
    AGE_LABELS,
    AGE_ORDER,
    AGE_REPORT_WEIGHT,
    AGE_REL_CONTACT,
    AGE_SYMPTOM_PROB,
    AGE_VACCINE_PROGRAM_REACH,
    AGE_WANING_YEARS,
    BOOTSTRAP_OUTCOME_METRICS,
    BOOTSTRAP_SEED,
    MODEL2_PARAMETER_NAMES,
    PARAMETER_BOUNDS,
)
from pertussis_global_dynamics.model2c_windows import (
    MODEL2C_BASELINE_END_YEAR,
    MODEL2C_BASELINE_START_YEAR,
    MODEL2C_OBSERVATION_END_YEAR,
    MODEL2C_OBSERVATION_START_YEAR,
    MODEL2C_SCENARIO_END_YEAR,
    MODEL2C_SCENARIO_START_YEAR,
)
from pertussis_global_dynamics.simulation.likelihood import nb_log_likelihood, negative_binomial_sample


def default_params_for_context(context: CountryContext) -> np.ndarray:
    observed_mean = context.annual.loc[
        context.annual["year"].between(MODEL2C_BASELINE_START_YEAR, MODEL2C_BASELINE_END_YEAR),
        "cases_total",
    ].mean()
    population_mean = context.annual.loc[
        context.annual["year"].between(MODEL2C_BASELINE_START_YEAR, MODEL2C_BASELINE_END_YEAR),
        "population",
    ].mean()
    rough_incidence = (observed_mean / population_mean) if population_mean and np.isfinite(population_mean) else 1e-5
    beta = float(np.clip(rough_incidence * 15.0, 1e-5, 0.08))
    return np.array(
        [
            beta,
            0.06,
            0.55,
            1.0,
            0.8,
            1.15,
            1.0,
            0.12,
        ],
        dtype=float,
    )


def vector_to_params(values: np.ndarray) -> dict[str, float]:
    return dict(zip(MODEL2_PARAMETER_NAMES, [float(value) for value in values]))


def _post_pandemic_reporting_prior(context: CountryContext) -> float:
    prior = monthly_detection_prior_table(
        context,
        MODEL2C_OBSERVATION_START_YEAR,
        MODEL2C_OBSERVATION_END_YEAR,
        include_testing_denominator=True,
    )
    if prior.empty:
        return np.nan
    post = prior.loc[prior["year"].between(MODEL2C_SCENARIO_START_YEAR, MODEL2C_SCENARIO_END_YEAR)].copy()
    if post.empty:
        return np.nan
    candidate_values = []
    direct = pd.to_numeric(
        post.get("ode_testing_denominator_series_relative_detection_prior_factor", np.nan),
        errors="coerce",
    )
    if direct.notna().any():
        candidate_values.append(float(np.nanmedian(direct)))
    relative = pd.to_numeric(post.get("ode_relative_detection_prior_mean", np.nan), errors="coerce")
    if relative.notna().any():
        candidate_values.append(float(np.nanmedian(relative)))
    if not candidate_values:
        return np.nan
    return float(np.nanmedian(candidate_values))


def population_template(context: CountryContext, year: int, structure_year: int | None = None) -> pd.DataFrame:
    pop = context.population_age.copy()
    target_structure_year = int(structure_year) if structure_year is not None else int(year)
    if pop.empty:
        shares = pd.DataFrame({"age_group": AGE_ORDER, "share": np.repeat(1.0 / len(AGE_ORDER), len(AGE_ORDER))})
        total = float(context.annual.loc[context.annual["year"].eq(year), "population"].mean())
        shares["population"] = total * shares["share"]
        return shares
    pop["year_distance"] = (pd.to_numeric(pop["year"], errors="coerce") - target_structure_year).abs()
    chosen_year = pop.sort_values("year_distance")["year"].iloc[0]
    chosen = pop.loc[pop["year"].eq(chosen_year), ["age_group", "population"]].copy()
    chosen = chosen.loc[chosen["age_group"].isin(AGE_ORDER)]
    total = chosen["population"].sum()
    if total <= 0:
        return population_template(
            CountryContext(
                context.iso3,
                context.country,
                context.annual,
                pd.DataFrame(),
                pd.DataFrame(),
                context.metadata,
                context.pre_mean_dtp3,
                context.pre_mean_incidence,
                context.baseline_note,
            ),
            year,
            structure_year=structure_year,
        )
    chosen["share"] = chosen["population"] / total
    observed_total = float(context.annual.loc[context.annual["year"].eq(year), "population"].mean())
    if np.isfinite(observed_total) and observed_total > 0:
        chosen["population"] = chosen["share"] * observed_total
    chosen["age_structure_reference_year"] = target_structure_year
    chosen["age_structure_source_year"] = int(chosen_year)
    return chosen


def apply_counterfactual_overrides(annual: pd.DataFrame, context: CountryContext, params: dict[str, float], overrides: dict[str, Any] | None) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    data = annual.copy()
    model_params = params.copy()
    options = {
        "include_contact_shock": True,
        "include_immunization_disruption": True,
        "include_reporting_change": True,
        "include_post_pandemic_transmission_drift": True,
        "policy_maternal_coverage": None,
        "preschool_booster_gain": 0.0,
        "adolescent_booster_gain": 0.0,
        "adult_booster_gain": 0.0,
        "next_generation_vaccine": False,
        "catch_up_gain": 0.0,
        "future_years": None,
        "post_transmission_multiplier": 1.0,
        "post_transmission_start_year": 2023,
        "vaccine_infection_multiplier": 1.0,
        "vaccine_disease_multiplier": 1.0,
        "asymptomatic_relative_infectiousness": 0.5,
        "contact_proxy_profile": "stringency_index",
        "include_population_age_structure_change": True,
        "population_age_structure_reference_year": 2019,
    }
    if overrides:
        options.update(overrides)

    if not options["include_immunization_disruption"]:
        affected = data["year"].between(2020, 2025)
        data.loc[affected, "dtp3_coverage"] = np.maximum(data.loc[affected, "dtp3_coverage"], context.pre_mean_dtp3)
        data.loc[affected, "dtp1_coverage"] = np.maximum(data.loc[affected, "dtp1_coverage"], context.pre_mean_dtp3)
    if not options["include_contact_shock"]:
        data.loc[:, ["stringency_index", "school_closure", "workplace_closure", "stay_home_requirement"]] = 0.0
    if not options["include_reporting_change"]:
        model_params["post_reporting_multiplier"] = 1.0
    if not options["include_post_pandemic_transmission_drift"]:
        model_params["post_transmission_multiplier"] = 1.0
    if options.get("long_lasting_vaccine_immunity"):
        model_params["waning_rate"] = min(model_params["waning_rate"], 0.035)
    if options.get("next_generation_vaccine"):
        model_params["waning_rate"] = min(model_params["waning_rate"], 0.025)
        model_params["beta"] *= 0.82
    return data, model_params, options


def contact_proxy_intensity(row: pd.Series, profile: str) -> float:
    def scaled_ordinal(column: str, max_value: float = 3.0) -> float:
        value = float(row.get(column, 0.0) or 0.0)
        return float(np.clip(value / max_value * 100.0, 0.0, 100.0))

    if profile == "school_closure":
        return scaled_ordinal("school_closure")
    if profile == "workplace_stay_home":
        return float(np.nanmean([scaled_ordinal("workplace_closure"), scaled_ordinal("stay_home_requirement")]))
    if profile == "closure_composite":
        return float(
            np.nanmean(
                [
                    scaled_ordinal("school_closure"),
                    scaled_ordinal("workplace_closure"),
                    scaled_ordinal("stay_home_requirement"),
                ]
            )
        )
    value = float(row.get("stringency_index", 0.0) or 0.0)
    return float(np.clip(value, 0.0, 100.0))


def simulate_country(
    context: CountryContext,
    params: dict[str, float],
    overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual, model_params, options = apply_counterfactual_overrides(context.annual, context, params, overrides)
    annual = annual.sort_values("year").copy()
    if options["future_years"]:
        last = annual.sort_values("year").tail(1).iloc[0].copy()
        future_rows = []
        for year in options["future_years"]:
            row = last.copy()
            row["year"] = int(year)
            row["date_start"] = f"{int(year)}-01-01"
            row["date_end"] = f"{int(year)}-12-31"
            row["cases_total"] = np.nan
            row["incidence_per_100k"] = np.nan
            row["population"] = float(last["population"]) * (1.003 ** (int(year) - int(last["year"])))
            row["dtp1_coverage"] = min(float(row["dtp1_coverage"]) + float(options["catch_up_gain"]), 0.99)
            row["dtp3_coverage"] = min(float(row["dtp3_coverage"]) + float(options["catch_up_gain"]), 0.99)
            row["stringency_index"] = 0.0
            row["school_closure"] = 0.0
            row["workplace_closure"] = 0.0
            row["stay_home_requirement"] = 0.0
            row["periodic_expected_incidence_per_100k"] = max(context.pre_mean_incidence, 0.001)
            future_rows.append(row)
        annual = pd.concat([annual, pd.DataFrame(future_rows)], ignore_index=True, sort=False)

    beta = model_params["beta"]
    reporting_rate = model_params["reporting_rate"]
    contact_sensitivity = model_params["contact_sensitivity"]
    boosting_debt_weight = model_params["boosting_debt_weight"]
    coverage_disruption_weight = model_params["coverage_disruption_weight"]
    post_reporting_multiplier = model_params["post_reporting_multiplier"]
    post_transmission_multiplier = model_params["post_transmission_multiplier"]
    waning_rate = model_params["waning_rate"]
    asymptomatic_relative_infectiousness = float(options.get("asymptomatic_relative_infectiousness", 0.5))
    base_asymptomatic_relative_infectiousness = 0.5
    contact_proxy_profile = str(options.get("contact_proxy_profile", "stringency_index"))

    vaccine_type = str(context.metadata.get("routine_scheduler_code", context.metadata.get("vaccine_type_primary", "unknown")))
    is_ap = vaccine_type == "aP"
    base_ve_infection = 0.50 if is_ap else 0.42
    base_ve_disease = 0.88 if is_ap else 0.82
    base_ve_infection = float(np.clip(base_ve_infection * float(options["vaccine_infection_multiplier"]), 0.0, 0.95))
    base_ve_disease = float(np.clip(base_ve_disease * float(options["vaccine_disease_multiplier"]), 0.0, 0.98))
    if options["next_generation_vaccine"]:
        base_ve_infection = 0.72
        base_ve_disease = 0.94

    maternal_coverage = options["policy_maternal_coverage"]
    if maternal_coverage is None:
        maternal_coverage = baseline_maternal_coverage_from_metadata(context.metadata)

    pre_expected_values = annual.loc[annual["year"].between(2000, 2019), "periodic_expected_incidence_per_100k"].to_numpy(dtype=float)
    pre_expected_values = pre_expected_values[np.isfinite(pre_expected_values)]
    if pre_expected_values.size:
        pre_expected_mean = float(np.mean(pre_expected_values))
        pre_expected_mean = max(pre_expected_mean, 0.001)
    else:
        pre_expected_mean = max(context.pre_mean_incidence, 0.001)
    age_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    boosting_debt = 0.0

    for _, row in annual.iterrows():
        year = int(row["year"])
        stringency = float(row.get("stringency_index", 0.0) or 0.0)
        contact_intensity = contact_proxy_intensity(row, contact_proxy_profile)
        contact_multiplier = max(0.18, 1.0 - contact_sensitivity * contact_intensity / 100.0)
        if year < 2020 or year > 2022:
            contact_intensity = 0.0
            contact_multiplier = 1.0
        if year in (2020, 2021, 2022):
            boosting_debt += max(0.0, 1.0 - contact_multiplier) * waning_rate * 2.5
        else:
            boosting_debt *= 0.72

        expected_incidence = float(row.get("periodic_expected_incidence_per_100k", np.nan))
        periodic_multiplier = max(expected_incidence, 0.001) / pre_expected_mean if np.isfinite(expected_incidence) else 1.0
        periodic_multiplier = float(np.clip(periodic_multiplier, 0.15, 6.0))

        dtp3 = float(row.get("dtp3_coverage", context.pre_mean_dtp3) or context.pre_mean_dtp3)
        if year in (2026, 2027):
            dtp3 = min(dtp3 + float(options["catch_up_gain"]), 0.99)
        disruption = max(context.pre_mean_dtp3 - dtp3, 0.0)
        coverage_penalty = 1.0 + coverage_disruption_weight * disruption
        report_multiplier = post_reporting_multiplier if year >= 2023 else 1.0
        post_transmission_start_year = int(options.get("post_transmission_start_year", 2023))
        transmission_multiplier = float(options.get("post_transmission_multiplier", 1.0)) if year >= post_transmission_start_year else 1.0

        age_structure_reference_year = None
        if not options["include_population_age_structure_change"]:
            age_structure_reference_year = int(options.get("population_age_structure_reference_year", 2019))
        pop = population_template(context, year, structure_year=age_structure_reference_year)
        age_structure_source_year = (
            int(pop["age_structure_source_year"].iloc[0])
            if "age_structure_source_year" in pop.columns and pop["age_structure_source_year"].notna().any()
            else year
        )
        age_structure_reference_year_out = (
            int(pop["age_structure_reference_year"].iloc[0])
            if "age_structure_reference_year" in pop.columns and pop["age_structure_reference_year"].notna().any()
            else year
        )
        total_infections = 0.0
        total_symptomatic = 0.0
        total_reported = 0.0
        total_infant_under_6m = 0.0
        total_under_1y = 0.0
        under1_population = float(pop.loc[pop["age_group"].isin(["infant_0_2m", "infant_3_11m"]), "population"].sum())
        total_age_population = float(pop["population"].sum())
        under1_population_share = under1_population / total_age_population if total_age_population > 0 else np.nan

        for _, pop_row in pop.iterrows():
            age_group = str(pop_row["age_group"])
            age_population = float(pop_row["population"])
            waning_multiplier = float(np.exp(-waning_rate * AGE_WANING_YEARS[age_group] / 8.0))
            program_reach = AGE_VACCINE_PROGRAM_REACH[age_group]
            if age_group == "child_1_4y":
                program_reach = min(program_reach + float(options["preschool_booster_gain"]), 1.0)
            if age_group == "child_5_9y":
                program_reach = min(program_reach + float(options["preschool_booster_gain"]) * 0.7, 1.0)
            if age_group == "adolescent_10_17y":
                program_reach = min(program_reach + float(options["adolescent_booster_gain"]), 1.0)
            if age_group in ("young_adult_18_39y", "middle_adult_40_64y", "elderly_65plus"):
                program_reach = min(program_reach + float(options["adult_booster_gain"]), 1.0)

            maternal_protection = 0.0
            if age_group == "infant_0_2m":
                maternal_protection = min(float(maternal_coverage) * 0.70, 0.70)
            elif age_group == "infant_3_11m":
                maternal_protection = min(float(maternal_coverage) * 0.28, 0.35)

            vaccine_protection = dtp3 * program_reach * base_ve_infection * waning_multiplier
            disease_protection = dtp3 * program_reach * base_ve_disease * waning_multiplier
            susceptibility = np.clip(1.0 - maternal_protection - vaccine_protection, 0.05, 1.0)
            symptom_probability = AGE_SYMPTOM_PROB[age_group] * np.clip(1.0 - 0.45 * disease_protection, 0.20, 1.0)
            base_effective_infectiousness = symptom_probability + base_asymptomatic_relative_infectiousness * (1.0 - symptom_probability)
            profile_effective_infectiousness = symptom_probability + asymptomatic_relative_infectiousness * (1.0 - symptom_probability)
            infectiousness_multiplier = profile_effective_infectiousness / max(base_effective_infectiousness, 1e-9)
            lambda_age = (
                beta
                * AGE_REL_CONTACT[age_group]
                * contact_multiplier
                * periodic_multiplier
                * (1.0 + boosting_debt_weight * boosting_debt)
                * coverage_penalty
                * transmission_multiplier
                * infectiousness_multiplier
                * susceptibility
            )
            infection_probability = 1.0 - np.exp(-max(lambda_age, 0.0))
            infections = age_population * infection_probability
            symptomatic = infections * symptom_probability
            reported = symptomatic * reporting_rate * report_multiplier * AGE_REPORT_WEIGHT[age_group]

            total_infections += infections
            total_symptomatic += symptomatic
            total_reported += reported
            if age_group == "infant_0_2m":
                total_infant_under_6m += reported
            if age_group in ("infant_0_2m", "infant_3_11m"):
                total_under_1y += reported

            age_rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "year": year,
                    "age_group": age_group,
                    "age_label": AGE_LABELS[age_group],
                    "population": age_population,
                    "age_structure_reference_year": age_structure_reference_year_out,
                    "age_structure_source_year": age_structure_source_year,
                    "under1_population_share": under1_population_share,
                    "contact_proxy_profile": contact_proxy_profile,
                    "contact_intensity": contact_intensity,
                    "contact_multiplier": contact_multiplier,
                    "post_transmission_multiplier": transmission_multiplier,
                    "boosting_debt": boosting_debt,
                    "susceptibility": susceptibility,
                    "asymptomatic_relative_infectiousness": asymptomatic_relative_infectiousness,
                    "effective_infectiousness_multiplier": infectiousness_multiplier,
                    "true_infections": infections,
                    "symptomatic_infections": symptomatic,
                    "reported_cases_predicted": reported,
                    "maternal_coverage_assumed": maternal_coverage,
                    "dtp3_coverage": dtp3,
                }
            )

        annual_rows.append(
            {
                "iso3": context.iso3,
                "country": context.country,
                "year": year,
                "observed_cases": row.get("cases_total", np.nan),
                "population": row.get("population", np.nan),
                "age_structure_reference_year": age_structure_reference_year_out,
                "age_structure_source_year": age_structure_source_year,
                "under1_population": under1_population,
                "under1_population_share": under1_population_share,
                "observed_incidence_per_100k": row.get("incidence_per_100k", np.nan),
                "periodic_expected_incidence_per_100k": expected_incidence,
                "dtp3_coverage": dtp3,
                "stringency_index": stringency,
                "contact_proxy_profile": contact_proxy_profile,
                "contact_intensity": contact_intensity,
                "contact_multiplier": contact_multiplier,
                "boosting_debt": boosting_debt,
                "true_infections_predicted": total_infections,
                "symptomatic_infections_predicted": total_symptomatic,
                "reported_cases_predicted": total_reported,
                "reported_infant_under_6m_predicted": total_infant_under_6m,
                "reported_under_1y_predicted": total_under_1y,
                "reporting_rate": reporting_rate,
                "post_reporting_multiplier": report_multiplier,
                "post_transmission_multiplier": transmission_multiplier,
            }
        )
    return pd.DataFrame(annual_rows), pd.DataFrame(age_rows)


def fit_country(context: CountryContext) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    params = vector_to_params(default_params_for_context(context))
    vaccine_type = str(context.metadata.get("routine_scheduler_code", "unknown"))
    params["waning_rate"] = 0.16 if vaccine_type == "aP" else 0.10

    pre = context.annual.loc[context.annual["year"].between(2000, 2019)].copy()
    pandemic = context.annual.loc[context.annual["year"].between(2020, 2022)].copy()
    post = context.annual.loc[context.annual["year"].between(2023, 2025)].copy()

    initial_pred, _ = simulate_country(context, params)
    pre_pred = initial_pred.loc[initial_pred["year"].between(2000, 2019), "reported_cases_predicted"].mean()
    pre_obs = pre["cases_total"].mean()
    if np.isfinite(pre_pred) and pre_pred > 0 and np.isfinite(pre_obs):
        params["beta"] = float(np.clip(params["beta"] * pre_obs / pre_pred, 1e-6, 0.20))

    params["contact_sensitivity"] = 0.55
    if not pandemic.empty:
        pandemic_stringency = pandemic["stringency_index"].mean()
        params["contact_sensitivity"] = float(np.clip(0.30 + pandemic_stringency / 140.0, 0.20, 0.85))
        pandemic_pred, _ = simulate_country(context, params)
        pred_low = pandemic_pred.loc[pandemic_pred["year"].between(2020, 2022), "reported_cases_predicted"].mean()
        obs_low = pandemic["cases_total"].mean()
        if np.isfinite(pred_low) and pred_low > 0 and np.isfinite(obs_low):
            ratio = obs_low / pred_low
            if ratio < 0.8:
                params["contact_sensitivity"] = float(np.clip(params["contact_sensitivity"] + (0.8 - ratio) * 0.25, 0.20, 0.90))

    post_obs_inc = post["incidence_per_100k"].mean() if not post.empty else np.nan
    expected_post = post["periodic_expected_incidence_per_100k"].mean() if not post.empty else np.nan
    post_rr = post_obs_inc / expected_post if expected_post and np.isfinite(expected_post) else 1.0
    params["boosting_debt_weight"] = float(np.clip((post_rr - 1.0) / 6.0, 0.0, 4.0))
    pandemic_dtp3 = context.annual.loc[context.annual["year"].between(2020, 2022), "dtp3_coverage"].mean()
    coverage_drop = max(context.pre_mean_dtp3 - pandemic_dtp3, 0.0) if np.isfinite(pandemic_dtp3) else 0.0
    params["coverage_disruption_weight"] = float(np.clip(0.75 + coverage_drop * 4.0, 0.0, 3.0))

    reporting_prior = _post_pandemic_reporting_prior(context)
    params["post_reporting_multiplier"] = float(
        np.clip(
            reporting_prior,
            *PARAMETER_BOUNDS["post_reporting_multiplier"],
        )
        if np.isfinite(reporting_prior)
        else 1.0,
    )
    params["post_transmission_multiplier"] = 1.0
    post_pred, _ = simulate_country(context, params)
    pred_post_cases = post_pred.loc[post_pred["year"].between(2023, 2025), "reported_cases_predicted"].mean()
    obs_post_cases = post["cases_total"].mean() if not post.empty else np.nan
    if np.isfinite(pred_post_cases) and pred_post_cases > 0 and np.isfinite(obs_post_cases):
        params["post_transmission_multiplier"] = float(
            np.clip(
                obs_post_cases / (pred_post_cases * params["post_reporting_multiplier"]),
                *PARAMETER_BOUNDS["post_transmission_multiplier"],
            )
        )

    annual_pred, age_pred = simulate_country(context, params)
    residual = annual_pred.dropna(subset=["observed_cases"]).copy()
    residual_value = float(
        np.nanmean(
            (
                np.log1p(residual["reported_cases_predicted"].clip(lower=0.0))
                - np.log1p(residual["observed_cases"].clip(lower=0.0))
            )
            ** 2
        )
    )
    result = {
        "success": True,
        "fun": residual_value,
        "method": "deterministic staged calibration",
    }
    return params, annual_pred, age_pred, result


def period_rmse(predicted: pd.DataFrame, start: int, end: int) -> float:
    subset = predicted.loc[predicted["year"].between(start, end)].dropna(subset=["observed_cases", "reported_cases_predicted"])
    if subset.empty:
        return np.nan
    residual = np.log1p(subset["reported_cases_predicted"].clip(lower=0.0)) - np.log1p(subset["observed_cases"].clip(lower=0.0))
    return float(np.sqrt(np.mean(residual.to_numpy(dtype=float) ** 2)))


def deterministic_country_seed(iso3: str) -> int:
    return BOOTSTRAP_SEED + sum((index + 1) * ord(char) for index, char in enumerate(str(iso3)))


def refresh_bootstrap_baseline(context: CountryContext, annual: pd.DataFrame) -> CountryContext:
    annual = annual.copy()
    annual["incidence_per_100k"] = annual["cases_total"] / annual["population"].replace({0: np.nan}) * 100000.0
    expected, note = periodic_baseline(
        annual["year"].to_numpy(dtype=float),
        annual["incidence_per_100k"].to_numpy(dtype=float),
        annual["year"].to_numpy(dtype=float),
    )
    annual["periodic_expected_incidence_per_100k"] = expected
    pre = annual.loc[annual["year"].between(2010, 2019)]
    pre_mean_dtp3 = float(pre["dtp3_coverage"].mean()) if pre["dtp3_coverage"].notna().any() else context.pre_mean_dtp3
    pre_mean_incidence = float(pre["incidence_per_100k"].mean()) if pre["incidence_per_100k"].notna().any() else context.pre_mean_incidence
    return replace(
        context,
        annual=annual,
        pre_mean_dtp3=max(min(pre_mean_dtp3, 0.99), 0.01) if np.isfinite(pre_mean_dtp3) else context.pre_mean_dtp3,
        pre_mean_incidence=max(pre_mean_incidence, 0.001) if np.isfinite(pre_mean_incidence) else context.pre_mean_incidence,
        baseline_note=f"parametric bootstrap refit from {note}",
    )


def summarize_post_period(predicted: pd.DataFrame, observed_reference: pd.DataFrame | None = None) -> dict[str, float]:
    post = predicted.loc[predicted["year"].between(2023, 2025)]
    if observed_reference is not None and not observed_reference.empty:
        observed = observed_reference.loc[:, ["year", "cases_total"]].rename(columns={"cases_total": "reference_observed_cases"})
        rmse_data = predicted.drop(columns=["observed_cases"], errors="ignore").merge(observed, on="year", how="left")
        rmse_data = rmse_data.rename(columns={"reference_observed_cases": "observed_cases"})
    else:
        rmse_data = predicted
    return {
        "reported_cases_2023_2025": float(post["reported_cases_predicted"].sum()),
        "true_infections_2023_2025": float(post["true_infections_predicted"].sum()),
        "reported_under_1y_2023_2025": float(post["reported_under_1y_predicted"].sum()),
        "post_pandemic_rmse_log_cases": period_rmse(rmse_data, 2023, 2025),
    }


_BOOTSTRAP_WORKER_STATE: dict[str, Any] | None = None


def _initialize_parametric_bootstrap_worker(state: dict[str, Any]) -> None:
    global _BOOTSTRAP_WORKER_STATE
    _BOOTSTRAP_WORKER_STATE = state


def _default_parametric_bootstrap_workers(replicates: int) -> int:
    raw = os.environ.get("MODEL2_BOOTSTRAP_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    available = worker_count()
    outer_workers = min(available, model2_country_limit(default=9))
    if os.environ.get("MODEL2_OUTER_WORKER", "").strip() == "1":
        if outer_workers <= 1:
            return 1
        return max(1, min(available // outer_workers, replicates, 16))
    return max(1, min(available, replicates, 16))


def _parametric_bootstrap_country_worker(task: tuple[int, np.ndarray]) -> tuple[dict[str, Any], pd.DataFrame]:
    if _BOOTSTRAP_WORKER_STATE is None:
        raise RuntimeError("parametric bootstrap worker state was not initialized")
    replicate_id, sampled_cases = task
    context = _BOOTSTRAP_WORKER_STATE["context"]
    annual_template = _BOOTSTRAP_WORKER_STATE["annual_template"]
    observed_mask = _BOOTSTRAP_WORKER_STATE["observed_mask"]

    bootstrap_annual = annual_template.copy()
    bootstrap_annual.loc[observed_mask, "cases_total"] = sampled_cases
    bootstrap_context = refresh_bootstrap_baseline(context, bootstrap_annual)
    bootstrap_params, bootstrap_pred, _, bootstrap_result = fit_country(bootstrap_context)
    post_metrics = summarize_post_period(bootstrap_pred, observed_reference=context.annual)
    draw_row = {
        "iso3": context.iso3,
        "country": context.country,
        "replicate": replicate_id,
        "optimizer_success": bool(bootstrap_result["success"]),
        **bootstrap_params,
        **post_metrics,
        "method_note": "Parametric bootstrap from fitted annual reduced SIRWS mean with fixed negative-binomial dispersion; staged calibration is refit in each replicate.",
    }
    pred = bootstrap_pred.loc[
        :,
        ["iso3", "country", "year", "reported_cases_predicted", "true_infections_predicted"],
    ].copy()
    pred.insert(2, "replicate", replicate_id)
    return draw_row, pred


def parametric_bootstrap_country(
    context: CountryContext,
    fitted_predictions: pd.DataFrame,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()

    annual_template = context.annual.copy()
    observed_mask = annual_template["cases_total"].notna() & annual_template["population"].gt(0)
    if not observed_mask.any():
        return pd.DataFrame(), pd.DataFrame()

    fitted_means = fitted_predictions.set_index("year")["reported_cases_predicted"]
    bootstrap_years = annual_template.loc[observed_mask, "year"].astype(int)
    mean = bootstrap_years.map(fitted_means).fillna(annual_template.loc[observed_mask, "cases_total"]).to_numpy(dtype=float)
    rng = np.random.default_rng(deterministic_country_seed(context.iso3))

    sampled_cases = [negative_binomial_sample(rng, mean) for _ in range(replicates)]
    parallel_workers = _default_parametric_bootstrap_workers(replicates)
    if parallel_workers > 1:
        state = {
            "context": context,
            "annual_template": annual_template,
            "observed_mask": observed_mask,
        }
        tasks = [(replicate_id, sample) for replicate_id, sample in enumerate(sampled_cases, start=1)]
        with ProcessPoolExecutor(
            max_workers=min(parallel_workers, replicates),
            initializer=_initialize_parametric_bootstrap_worker,
            initargs=(state,),
        ) as executor:
            results = list(executor.map(_parametric_bootstrap_country_worker, tasks))
        draw_rows = [item[0] for item in results]
        prediction_rows = [item[1] for item in results if not item[1].empty]
    else:
        draw_rows = []
        prediction_rows = []
        for replicate_id, sampled_case in enumerate(sampled_cases, start=1):
            bootstrap_annual = annual_template.copy()
            bootstrap_annual.loc[observed_mask, "cases_total"] = sampled_case
            bootstrap_context = refresh_bootstrap_baseline(context, bootstrap_annual)
            bootstrap_params, bootstrap_pred, _, bootstrap_result = fit_country(bootstrap_context)
            post_metrics = summarize_post_period(bootstrap_pred, observed_reference=context.annual)
            draw_rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "replicate": replicate_id,
                    "optimizer_success": bool(bootstrap_result["success"]),
                    **bootstrap_params,
                    **post_metrics,
                    "method_note": "Parametric bootstrap from fitted annual reduced SIRWS mean with fixed negative-binomial dispersion; staged calibration is refit in each replicate.",
                }
            )
            pred = bootstrap_pred.loc[
                :,
                ["iso3", "country", "year", "reported_cases_predicted", "true_infections_predicted"],
            ].copy()
            pred.insert(2, "replicate", replicate_id)
            prediction_rows.append(pred)

    bootstrap_draws = pd.DataFrame(draw_rows)
    bootstrap_predictions = pd.concat(prediction_rows, ignore_index=True, sort=False) if prediction_rows else pd.DataFrame()
    return bootstrap_draws, bootstrap_predictions


def summarize_bootstrap_draws(bootstrap_draws: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if bootstrap_draws.empty:
        return pd.DataFrame()

    metrics = MODEL2_PARAMETER_NAMES + BOOTSTRAP_OUTCOME_METRICS
    long = bootstrap_draws.melt(
        id_vars=["iso3", "country", "replicate"],
        value_vars=[metric for metric in metrics if metric in bootstrap_draws.columns],
        var_name="metric",
        value_name="bootstrap_value",
    ).dropna(subset=["bootstrap_value"])
    grouped = long.groupby(["iso3", "country", "metric"], as_index=False)
    intervals = grouped.agg(
        bootstrap_median=("bootstrap_value", "median"),
        bootstrap_lower_95=("bootstrap_value", lambda values: float(np.nanquantile(values, 0.025))),
        bootstrap_upper_95=("bootstrap_value", lambda values: float(np.nanquantile(values, 0.975))),
        n_bootstrap_replicates=("replicate", "nunique"),
    )

    point_rows: list[dict[str, Any]] = []
    summary_by_iso = summary.set_index("iso3")
    for (iso3, country), group in predictions.groupby(["iso3", "country"]):
        if iso3 in summary_by_iso.index:
            summary_row = summary_by_iso.loc[iso3]
            for metric in MODEL2_PARAMETER_NAMES:
                if metric in summary_row.index:
                    point_rows.append({"iso3": iso3, "country": country, "metric": metric, "point_estimate": float(summary_row[metric])})
        for metric, value in summarize_post_period(group).items():
            point_rows.append({"iso3": iso3, "country": country, "metric": metric, "point_estimate": float(value)})
    point = pd.DataFrame(point_rows)
    intervals = intervals.merge(point, on=["iso3", "country", "metric"], how="left")
    intervals["method_note"] = "Country-level parametric bootstrap; 95% interval is empirical 2.5th-97.5th percentile across refits."
    return intervals.loc[
        :,
        [
            "iso3",
            "country",
            "metric",
            "point_estimate",
            "bootstrap_median",
            "bootstrap_lower_95",
            "bootstrap_upper_95",
            "n_bootstrap_replicates",
            "method_note",
        ],
    ].sort_values(["iso3", "metric"]).reset_index(drop=True)
