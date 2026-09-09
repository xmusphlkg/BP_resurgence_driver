from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from scipy.optimize import minimize
import numpy as np
import pandas as pd
from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.data.model2_observations import (
    allocated_age_observations,
    default_detection_prior_row,
    default_pathogen_prior_row,
    monthly_detection_prior_table,
    monthly_observed_cases,
    monthly_pathogen_prior_table,
)
from pertussis_global_dynamics.data.model2_inputs import model2_country_limit, worker_count
from pertussis_global_dynamics.model2_constants import (
    AGE_LABELS,
    AGE_ORDER,
    AGE_REPORT_WEIGHT,
    AGE_REL_CONTACT,
    AGE_SYMPTOM_PROB,
    AGE_VACCINE_PROGRAM_REACH,
    AGE_WANING_YEARS,
    MONTHLY_ODE_AGE_OBJECTIVE_WEIGHT,
    MONTHLY_ODE_INITIAL_OBSERVATION_BLEND,
    MONTHLY_ODE_OPTIMIZER_MAXITER,
    MONTHLY_ODE_PARAMETER_BOUNDS,
    MONTHLY_ODE_PARAMETER_NAMES,
    MONTHLY_ODE_PROFILE_FIXED_FACTORS,
    MONTHLY_ODE_PROFILE_NUISANCE_FACTORS,
    MONTHLY_ODE_PROFILE_SUPPORT_95_1DF,
    MONTHLY_ODE_JOINT_AGE_MONTH_OBJECTIVE_WEIGHT,
    MONTHLY_ODE_RESIDUAL_ADJUSTMENT_FREE_LOG,
    MONTHLY_ODE_RESIDUAL_ADJUSTMENT_PENALTY_WEIGHT,
    MONTHLY_ODE_SEARCH_PARAMETERS,
    MONTHLY_ODE_SIGNED_PARAMETERS,
    MONTHLY_ODE_TRANSITION_FREE_LOG,
    MONTHLY_ODE_TRANSITION_PENALTY_WEIGHT,
    NB_DISPERSION,
)
from pertussis_global_dynamics.model2c_windows import (
    MODEL2C_OBSERVATION_END_YEAR,
    MODEL2C_OBSERVATION_START_YEAR,
)
from pertussis_global_dynamics.simulation.likelihood import nb_log_likelihood, weighted_median
from pertussis_global_dynamics.simulation.observation_likelihood import (
    joint_age_month_composite_observations,
    joint_age_month_ode_likelihood_components,
)
from pertussis_global_dynamics.simulation.monthly_state_bridge import initialize_monthly_state, month_end, monthly_seasonal_fractions


AGE_REL_CONTACT_VECTOR = np.array([AGE_REL_CONTACT[age] for age in AGE_ORDER], dtype=float)
AGE_REL_CONTACT_VECTOR_SUM = float(AGE_REL_CONTACT_VECTOR.sum())
AGE_SYMPTOM_PROB_VECTOR = np.array([AGE_SYMPTOM_PROB[age] for age in AGE_ORDER], dtype=float)


def collapse_unique_text(values: pd.Series, default: str = "not_available") -> str:
    cleaned = [
        str(value).strip()
        for value in values.fillna("").astype(str)
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]
    return "; ".join(dict.fromkeys(cleaned)) if cleaned else default


_MONTHLY_ODE_SCORE_WORKER_STATE: dict[str, Any] | None = None


def _initialize_monthly_ode_score_worker(state: dict[str, Any]) -> None:
    global _MONTHLY_ODE_SCORE_WORKER_STATE
    _MONTHLY_ODE_SCORE_WORKER_STATE = state


def _build_monthly_ode_score_state(
    context: CountryContext,
    params: dict[str, float],
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    *,
    observed_month: pd.DataFrame | None = None,
    observed_age: pd.DataFrame | None = None,
    age_objective_weight: float,
    include_joint_observation: bool,
) -> dict[str, Any]:
    cache = _build_monthly_ode_cache(
        context,
        age_pred,
        inputs,
        observed_month=observed_month,
        observed_age=observed_age,
        include_testing_denominator=True,
        include_joint_observation=include_joint_observation,
    )
    pathogen_prior_monthly = cache["pathogen_prior_table"]
    pathogen_pressure_values = (
        pd.to_numeric(pathogen_prior_monthly["ode_pathogen_pressure_prior_mean"], errors="coerce").dropna()
        if not pathogen_prior_monthly.empty
        else pd.Series(dtype=float)
    )
    static_moderate_pathogen_prior = bool(
        not pathogen_pressure_values.empty
        and pathogen_pressure_values.max() <= 1.25
        and (pathogen_pressure_values.max() - pathogen_pressure_values.min()) <= 0.05
    )
    return {
        "context": context,
        "params": params,
        "age_pred": age_pred,
        "inputs": inputs if inputs is not None else {},
        "observed": cache.get("observed_month", pd.DataFrame()),
        "observed_age": cache.get("observed_age", pd.DataFrame()),
        "cache": cache,
        "age_objective_weight": age_objective_weight,
        "static_moderate_pathogen_prior": static_moderate_pathogen_prior,
    }


def _build_monthly_ode_cache(
    context: CountryContext,
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    *,
    observed_month: pd.DataFrame | None = None,
    observed_age: pd.DataFrame | None = None,
    include_testing_denominator: bool = True,
    include_joint_observation: bool = False,
) -> dict[str, Any]:
    cache: dict[str, Any] = {}

    monthly_observed = observed_month.copy() if observed_month is not None else monthly_observed_cases(context)
    if not monthly_observed.empty:
        monthly_observed = monthly_observed.loc[
            monthly_observed["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        ].copy()
        monthly_observed["year"] = pd.to_numeric(monthly_observed["year"], errors="coerce").astype(int)
        monthly_observed["month"] = pd.to_numeric(monthly_observed["month"], errors="coerce").astype(int)
    cache["observed_month"] = monthly_observed
    if not monthly_observed.empty:
        cache["observed_month_lookup"] = (
            monthly_observed.groupby(["year", "month"])["observed_cases"].sum().to_dict()
        )
    else:
        cache["observed_month_lookup"] = {}

    allocated_age = observed_age.copy() if observed_age is not None else pd.DataFrame()
    if observed_age is None and inputs is not None:
        allocated_age = allocated_age_observations(inputs, context.iso3)
    if not allocated_age.empty:
        allocated_age = allocated_age.loc[
            allocated_age["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        ].copy()
        allocated_age["year"] = pd.to_numeric(allocated_age["year"], errors="coerce").astype(int)
    cache["observed_age"] = allocated_age

    detection_prior = monthly_detection_prior_table(
        context,
        include_testing_denominator=include_testing_denominator,
    )
    cache["detection_prior_table"] = detection_prior
    cache["detection_prior_lookup"] = (
        detection_prior.set_index(["year", "month"]).to_dict("index") if not detection_prior.empty else {}
    )

    pathogen_prior = monthly_pathogen_prior_table(context)
    cache["pathogen_prior_table"] = pathogen_prior
    cache["pathogen_prior_lookup"] = (
        pathogen_prior.set_index(["year", "month"]).to_dict("index") if not pathogen_prior.empty else {}
    )

    cache["seasonal_lookup"] = monthly_seasonal_fractions(context).set_index("month")["seasonal_fraction"].to_dict()

    monthly_covid = context.monthly_covid.copy()
    if not monthly_covid.empty:
        monthly_covid["year"] = pd.to_numeric(monthly_covid["year"], errors="coerce").astype("Int64")
        monthly_covid["month"] = pd.to_numeric(monthly_covid["month"], errors="coerce").astype("Int64")
        monthly_covid["monthly_stringency_index"] = pd.to_numeric(
            monthly_covid["monthly_stringency_index"], errors="coerce"
        )
        monthly_covid = monthly_covid.dropna(subset=["year", "month"])
        monthly_covid["year"] = monthly_covid["year"].astype(int)
        monthly_covid["month"] = monthly_covid["month"].astype(int)
        cache["monthly_covid_lookup"] = monthly_covid.set_index(["year", "month"]).to_dict("index")
    else:
        cache["monthly_covid_lookup"] = {}

    annual_age = age_pred.loc[
        age_pred["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
    ].copy()
    if not annual_age.empty:
        annual_age["year"] = pd.to_numeric(annual_age["year"], errors="coerce").astype(int)
        annual_age = annual_age.sort_values(["year", "age_group"])
    cache["annual_age"] = annual_age

    cache["detection_prior_use_testing_denominator"] = bool(include_testing_denominator)
    cache["joint_observed"] = (
        joint_age_month_composite_observations(
            context,
            inputs,
            observed_month=monthly_observed,
            observed_age=allocated_age,
        )
        if include_joint_observation
        else pd.DataFrame()
    )
    return cache


def _score_monthly_ode_options_worker(task: tuple[float, dict[str, float]]) -> dict[str, float] | None:
    if _MONTHLY_ODE_SCORE_WORKER_STATE is None:
        raise RuntimeError("monthly ODE score worker state was not initialized")
    joint_age_month_objective_weight, options = task
    return _score_monthly_ode_options(_MONTHLY_ODE_SCORE_WORKER_STATE, joint_age_month_objective_weight, options)


def _default_monthly_ode_parallel_workers() -> int:
    raw = os.environ.get("MODEL2_ODE_PARALLEL_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    available = worker_count()
    if os.environ.get("MODEL2_OUTER_WORKER", "").strip() == "1":
        outer_workers = min(worker_count(), model2_country_limit(default=9))
        if outer_workers <= 1:
            return 1
        return max(1, min(available // outer_workers, 16))
    return max(1, min(available, 16))


def _score_monthly_ode_options(
    state: dict[str, Any],
    joint_age_month_objective_weight: float,
    options: dict[str, float],
) -> dict[str, float] | None:
    context = state["context"]
    params = state["params"]
    age_pred = state["age_pred"]
    inputs = state["inputs"]
    cache = state.get("cache", {})
    observed = cache.get("observed_month", state.get("observed"))
    observed_age = cache.get("observed_age", state.get("observed_age"))
    age_objective_weight = float(state["age_objective_weight"])
    static_moderate_pathogen_prior = bool(state["static_moderate_pathogen_prior"])

    predicted = simulate_monthly_ode(
        context,
        params,
        age_pred,
        beta_scale=options["ode_beta_scale"],
        reporting_scale=options["ode_reporting_scale"],
        cache=cache,
        **{key: value for key, value in options.items() if key not in {"ode_beta_scale", "ode_reporting_scale"}},
    )
    if predicted.empty:
        return None
    predicted = apply_ode_observation_adjustments(
        context,
        predicted,
        inputs,
        observed_month=observed,
        observed_age=observed_age,
    )
    residual_adjustment_penalty = 0.0
    transition_penalty = 0.0
    if "ode_smoothed_year_reporting_adjustment" in predicted:
        year_adjustment = (
            predicted.groupby("year")["ode_smoothed_year_reporting_adjustment"]
            .mean()
            .pipe(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .clip(lower=0.02, upper=80.0)
        )
        if not year_adjustment.empty:
            excess = np.maximum(
                np.abs(np.log(year_adjustment.to_numpy(dtype=float))) - MONTHLY_ODE_RESIDUAL_ADJUSTMENT_FREE_LOG,
                0.0,
            )
            residual_adjustment_penalty = float(MONTHLY_ODE_RESIDUAL_ADJUSTMENT_PENALTY_WEIGHT * np.nanmean(excess**2))
    if "reported_cases_before_year_adjustment" in predicted:
        pre_adjusted_month = (
            predicted.groupby(["year", "month"], as_index=False)
            .agg(predicted_cases_before_year_adjustment=("reported_cases_before_year_adjustment", "sum"))
            .sort_values(["year", "month"])
        )
        pre_adjusted_month["month_index"] = pre_adjusted_month["year"] * 12 + pre_adjusted_month["month"]
        pre_adjusted_month = pre_adjusted_month.loc[pre_adjusted_month["month_index"].diff().fillna(1).eq(1)].copy()
        if len(pre_adjusted_month) >= 2:
            current_values = pre_adjusted_month["predicted_cases_before_year_adjustment"].to_numpy(dtype=float)[1:] + 1.0
            previous_values = pre_adjusted_month["predicted_cases_before_year_adjustment"].to_numpy(dtype=float)[:-1] + 1.0
            ratios = np.maximum(current_values / previous_values, previous_values / current_values)
            excess = np.maximum(np.log(np.clip(ratios, 1.0, 1e6)) - MONTHLY_ODE_TRANSITION_FREE_LOG, 0.0)
            transition_penalty = float(MONTHLY_ODE_TRANSITION_PENALTY_WEIGHT * np.nanmean(excess**2))
    predicted_month = predicted.groupby(["iso3", "country", "year", "month"], as_index=False).agg(
        predicted_cases=("reported_cases_predicted", "sum")
    )
    monthly_score = np.nan
    monthly_n = 0
    monthly_ll = np.nan
    if not observed.empty:
        month_merged = observed.merge(predicted_month, on=["iso3", "country", "year", "month"], how="left")
        month_merged["predicted_cases"] = month_merged["predicted_cases"].fillna(0.0)
        valid_month = month_merged["observed_cases"].ge(0) & month_merged["predicted_cases"].ge(0)
        if valid_month.any():
            observed_month = month_merged.loc[valid_month, "observed_cases"].to_numpy(dtype=float)
            predicted_cases = month_merged.loc[valid_month, "predicted_cases"].to_numpy(dtype=float)
            residual = np.log1p(predicted_cases) - np.log1p(observed_month)
            monthly_score = float(np.nanmean(np.clip(residual, -7.0, 7.0) ** 2))
            monthly_ll = float(np.nansum([nb_log_likelihood(obs, mu) for obs, mu in zip(observed_month, predicted_cases)]))
            monthly_n = int(valid_month.sum())

    age_score = np.nan
    age_effective_n = 0.0
    age_ll = np.nan
    if not observed_age.empty:
        predicted_age = predicted.groupby(["iso3", "country", "year", "age_group"], as_index=False).agg(
            predicted_cases=("reported_cases_predicted", "sum")
        ).rename(columns={"age_group": "model_age_group"})
        age_merged = observed_age.merge(predicted_age, on=["iso3", "country", "year", "model_age_group"], how="inner")
        if not age_merged.empty:
            age_merged["age_allocation_weight"] = pd.to_numeric(age_merged["age_allocation_weight"], errors="coerce").fillna(1.0)
            valid_age = age_merged["observed_cases"].ge(0) & age_merged["predicted_cases"].ge(0)
            if valid_age.any():
                observed_cases = age_merged.loc[valid_age, "observed_cases"].to_numpy(dtype=float)
                predicted_cases = age_merged.loc[valid_age, "predicted_cases"].to_numpy(dtype=float)
                weights = age_merged.loc[valid_age, "age_allocation_weight"].to_numpy(dtype=float)
                weights = np.clip(weights, 0.05, 1.0)
                residual = np.log1p(predicted_cases) - np.log1p(observed_cases)
                age_effective_n = float(np.nansum(weights))
                if age_effective_n > 0:
                    age_score = float(np.nansum(weights * np.clip(residual, -7.0, 7.0) ** 2) / age_effective_n)
                    age_ll = float(
                        np.nansum(
                            [
                                nb_log_likelihood(obs, mu) * weight
                                for obs, mu, weight in zip(observed_cases, predicted_cases, weights)
                            ]
                        )
                    )

    joint_age_month_score = np.nan
    joint_age_month_effective_n = 0.0
    joint_age_month_ll = np.nan
    joint_age_month_weighted_ll = np.nan
    joint_age_month_rows = 0
    if joint_age_month_objective_weight > 0.0:
        joint_likelihood = joint_age_month_ode_likelihood_components(
            context,
            predicted,
            inputs,
            observed=cache.get("joint_observed"),
        )
        joint_age_month_rows = int(len(joint_likelihood))
        if not joint_likelihood.empty:
            joint_weights = pd.to_numeric(joint_likelihood["age_allocation_weight"], errors="coerce").fillna(1.0)
            joint_weights = joint_weights.clip(lower=0.05, upper=1.0)
            joint_weight_values = joint_weights.to_numpy(dtype=float)
            joint_weight_sum = float(np.nansum(joint_weight_values))
            if joint_weight_sum > 0:
                joint_age_month_effective_n = joint_weight_sum
                joint_residuals = pd.to_numeric(joint_likelihood["log_residual"], errors="coerce").to_numpy(dtype=float)
                joint_age_month_score = float(
                    np.nansum(np.clip(joint_residuals, -7.0, 7.0) ** 2 * joint_weight_values) / joint_weight_sum
                )
                joint_age_month_ll = float(
                    pd.to_numeric(joint_likelihood["nb_log_likelihood"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
                joint_age_month_weighted_ll = float(
                    pd.to_numeric(joint_likelihood["weighted_nb_log_likelihood"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )

    component_scores: list[float] = []
    component_weights: list[float] = []
    if np.isfinite(monthly_score):
        component_scores.append(monthly_score)
        component_weights.append(max(1.0 - age_objective_weight - joint_age_month_objective_weight, 0.0))
    if np.isfinite(age_score):
        component_scores.append(age_score)
        component_weights.append(age_objective_weight if component_scores else 1.0)
    if np.isfinite(joint_age_month_score):
        component_scores.append(joint_age_month_score)
        component_weights.append(joint_age_month_objective_weight if component_scores else 1.0)
    if not component_scores:
        return None
    score = float(np.average(component_scores, weights=np.asarray(component_weights, dtype=float)))
    defaults = monthly_ode_default_options()
    penalty_terms = []
    for key in MONTHLY_ODE_PARAMETER_NAMES:
        if key in {"ode_reporting_scale", "ode_post_reporting_scale", "ode_pandemic_reporting_scale"}:
            weight = 0.002
        elif key in MONTHLY_ODE_SIGNED_PARAMETERS:
            weight = 0.004
        elif key == "ode_pathogen_drift_scale" and static_moderate_pathogen_prior:
            weight = 0.035
        else:
            weight = 0.006
        reference = max(float(defaults[key]), 1e-9)
        if key in MONTHLY_ODE_SIGNED_PARAMETERS:
            lower, upper = MONTHLY_ODE_PARAMETER_BOUNDS[key]
            span = max(upper - lower, 1e-9)
            penalty_terms.append(weight * ((float(options[key]) - float(defaults[key])) / span) ** 2)
        else:
            penalty_terms.append(weight * (np.log(max(options[key], 1e-9) / reference) ** 2))
    penalty = float(np.nansum(penalty_terms))
    score_with_penalty = score + penalty
    score_with_penalty += residual_adjustment_penalty + transition_penalty
    return {
        **options,
        "score": score_with_penalty,
        "unpenalized_score": score,
        "penalty": penalty,
        "residual_adjustment_penalty": residual_adjustment_penalty,
        "transition_penalty": transition_penalty,
        "monthly_score": monthly_score,
        "age_score": age_score,
        "monthly_weighted_nb_log_likelihood": monthly_ll,
        "age_weighted_nb_log_likelihood": age_ll,
        "monthly_n": float(monthly_n),
        "age_effective_n": age_effective_n,
        "joint_age_month_score": joint_age_month_score,
        "joint_age_month_nb_log_likelihood": joint_age_month_ll,
        "joint_age_month_weighted_nb_log_likelihood": joint_age_month_weighted_ll,
        "joint_age_month_effective_n": joint_age_month_effective_n,
        "joint_age_month_likelihood_rows": joint_age_month_rows,
    }


def monthly_ode_default_options(overrides: dict[str, float] | None = None) -> dict[str, float]:
    options = {
        "ode_beta_scale": 1.0,
        "ode_reporting_scale": 1.0,
        "ode_pandemic_beta_scale": 0.55,
        "ode_post_beta_scale": 1.25,
        "ode_pandemic_reporting_scale": 1.0,
        "ode_post_reporting_scale": 1.0,
        "ode_importation_scale": 1.0,
        "ode_seasonal_strength": 1.0,
        "ode_pandemic_seasonal_shift_months": 0.0,
        "ode_post_seasonal_shift_months": 0.0,
        "ode_initial_infectious_scale": 1.0,
        "ode_pathogen_drift_scale": 1.0,
    }
    if overrides:
        for key, value in overrides.items():
            if key in options and np.isfinite(value):
                lower, upper = MONTHLY_ODE_PARAMETER_BOUNDS[key]
                options[key] = float(np.clip(value, lower, upper))
    return options


def monthly_ode_calibration_start_options(starting_options: dict[str, float] | None = None) -> list[dict[str, float]]:
    starts = [
        monthly_ode_default_options(),
        monthly_ode_default_options({"ode_beta_scale": 1.6, "ode_importation_scale": 4.0, "ode_post_beta_scale": 3.5}),
        monthly_ode_default_options({"ode_beta_scale": 0.75, "ode_pandemic_beta_scale": 0.25, "ode_pandemic_reporting_scale": 0.35, "ode_pathogen_drift_scale": 0.70}),
        monthly_ode_default_options({"ode_beta_scale": 1.25, "ode_post_beta_scale": 2.8, "ode_pathogen_drift_scale": 1.60, "ode_seasonal_strength": 1.35}),
        monthly_ode_default_options({"ode_pandemic_seasonal_shift_months": -3.0, "ode_seasonal_strength": 0.85}),
        monthly_ode_default_options({"ode_post_seasonal_shift_months": -3.0, "ode_post_beta_scale": 1.8, "ode_seasonal_strength": 0.85}),
        monthly_ode_default_options({"ode_post_seasonal_shift_months": 2.0, "ode_post_beta_scale": 1.2, "ode_seasonal_strength": 0.75}),
    ]
    if starting_options is None:
        return starts
    # Warm-start refits only need a smaller basin-search set because the
    # previous fitted solution is already a high-quality local seed.
    warm_seed = monthly_ode_default_options(starting_options)
    return [
        warm_seed,
        monthly_ode_default_options({"ode_beta_scale": 1.6, "ode_importation_scale": 4.0, "ode_post_beta_scale": 3.5}),
        monthly_ode_default_options({"ode_beta_scale": 0.75, "ode_pandemic_beta_scale": 0.25, "ode_pandemic_reporting_scale": 0.35, "ode_pathogen_drift_scale": 0.70}),
        monthly_ode_default_options({"ode_post_seasonal_shift_months": 2.0, "ode_post_beta_scale": 1.2, "ode_seasonal_strength": 0.75}),
    ]


def monthly_ode_options_from_kwargs(
    beta_scale: float = 1.0,
    reporting_scale: float = 1.0,
    **kwargs: float,
) -> dict[str, float]:
    options = monthly_ode_default_options(kwargs)
    options["ode_beta_scale"] = float(np.clip(beta_scale, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_beta_scale"]))
    options["ode_reporting_scale"] = float(np.clip(reporting_scale, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_reporting_scale"]))
    return options


def monthly_ode_vector_to_options(values: np.ndarray) -> dict[str, float]:
    options: dict[str, float] = {}
    for idx, name in enumerate(MONTHLY_ODE_PARAMETER_NAMES):
        lower, upper = MONTHLY_ODE_PARAMETER_BOUNDS[name]
        raw = float(values[idx])
        if name in MONTHLY_ODE_SIGNED_PARAMETERS:
            options[name] = float(np.clip(raw, lower, upper))
        else:
            options[name] = float(np.clip(np.exp(raw), lower, upper))
    return options


def monthly_ode_options_to_vector(options: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            np.log(float(np.clip(options.get(name, 1.0), *MONTHLY_ODE_PARAMETER_BOUNDS[name])))
            if name not in MONTHLY_ODE_SIGNED_PARAMETERS
            else float(np.clip(options.get(name, 0.0), *MONTHLY_ODE_PARAMETER_BOUNDS[name]))
            for name in MONTHLY_ODE_PARAMETER_NAMES
        ],
        dtype=float,
    )


def monthly_pandemic_phase_weight(year: int, month: int) -> float:
    year = int(year)
    month = int(month)
    if year < 2020 or year > 2022:
        return 0.0
    if year > 2020:
        return 1.0
    if month <= 1:
        return 0.0
    if month == 2:
        return 0.35
    if month == 3:
        return 0.75
    return 1.0


def monthly_ode_phase_multipliers(year: int, month: int, options: dict[str, float]) -> tuple[float, float, float]:
    beta_multiplier = float(options["ode_beta_scale"])
    reporting_multiplier = float(options["ode_reporting_scale"])
    pandemic_weight = monthly_pandemic_phase_weight(year, month)
    if pandemic_weight > 0:
        beta_multiplier *= float(options["ode_pandemic_beta_scale"]) ** pandemic_weight
        reporting_multiplier *= float(options["ode_pandemic_reporting_scale"]) ** pandemic_weight
    elif int(year) >= 2023:
        beta_multiplier *= float(options["ode_post_beta_scale"])
        reporting_multiplier *= float(options["ode_post_reporting_scale"])
    return beta_multiplier, reporting_multiplier, pandemic_weight


def monthly_ode_seasonal_source_month(month: int, year: int, options: dict[str, float]) -> tuple[int, float]:
    shift = 0.0
    if 2020 <= int(year) <= 2022:
        shift = float(options.get("ode_pandemic_seasonal_shift_months", 0.0))
    elif int(year) >= 2024:
        shift = float(options.get("ode_post_seasonal_shift_months", 0.0))
    rounded_shift = int(np.rint(np.clip(shift, -4.0, 4.0)))
    source_month = ((int(month) - 1 - rounded_shift) % 12) + 1
    return source_month, float(rounded_shift)


def apply_ode_year_reporting_adjustment(
    context: CountryContext,
    ode_state: pd.DataFrame,
    observed_month: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ode_state.empty:
        return ode_state
    adjusted = ode_state.copy()
    if "reported_cases_before_year_adjustment" not in adjusted:
        adjusted["reported_cases_before_year_adjustment"] = adjusted["reported_cases_predicted"]
    adjusted["reported_cases_before_year_adjustment"] = pd.to_numeric(
        adjusted["reported_cases_before_year_adjustment"], errors="coerce"
    ).fillna(0.0)
    predicted_year = adjusted.groupby("year", as_index=False).agg(
        unadjusted_reported_cases_year=("reported_cases_before_year_adjustment", "sum")
    )
    if observed_month is None:
        observed_month = monthly_observed_cases(context)
    else:
        observed_month = observed_month.copy()
    if not observed_month.empty:
        observed_month["year"] = pd.to_numeric(observed_month["year"], errors="coerce").astype("Int64")
        observed_month["month"] = pd.to_numeric(observed_month["month"], errors="coerce").astype("Int64")
        observed_month["observed_cases"] = pd.to_numeric(observed_month["observed_cases"], errors="coerce")
        observed_month = observed_month.loc[
            observed_month["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
            & observed_month["observed_cases"].notna()
        ].copy()
    if observed_month.empty:
        adjusted["reported_cases_unadjusted"] = adjusted["reported_cases_predicted"]
        adjusted["ode_year_reporting_adjustment_raw"] = 1.0
        adjusted["ode_smoothed_year_reporting_adjustment"] = 1.0
        adjusted["ode_month_reporting_adjustment_raw"] = 1.0
        adjusted["ode_smoothed_month_reporting_adjustment"] = 1.0
        adjusted["ode_year_total_rescale_after_smoothing"] = 1.0
        adjusted["ode_year_reporting_adjustment"] = 1.0
        adjusted["ode_reporting_adjustment_method"] = "No monthly observations available for year-level observation adjustment."
        return adjusted
    observed_year = observed_month.groupby("year", as_index=False).agg(observed_reported_cases_year=("observed_cases", "sum"))
    scale = observed_year.merge(predicted_year, on="year", how="right")
    scale["observed_reported_cases_year"] = pd.to_numeric(scale["observed_reported_cases_year"], errors="coerce")
    scale["unadjusted_reported_cases_year"] = pd.to_numeric(scale["unadjusted_reported_cases_year"], errors="coerce")
    scale["ode_year_reporting_adjustment_raw"] = (
        (scale["observed_reported_cases_year"] + 1.0) / (scale["unadjusted_reported_cases_year"] + 1.0)
    ).clip(lower=0.02, upper=80.0)
    scale_points = scale.loc[
        np.isfinite(scale["ode_year_reporting_adjustment_raw"]),
        ["year", "ode_year_reporting_adjustment_raw"],
    ].drop_duplicates("year")
    scale_points = scale_points.sort_values("year")
    adjusted["month_index"] = pd.to_numeric(adjusted["year"], errors="coerce") * 12.0 + pd.to_numeric(
        adjusted["month"], errors="coerce"
    )
    if len(scale_points) >= 2:
        yearly_midpoints = scale_points["year"].to_numpy(dtype=float) * 12.0 + 6.5
        yearly_scales = scale_points["ode_year_reporting_adjustment_raw"].to_numpy(dtype=float)
        adjusted["ode_smoothed_year_reporting_adjustment"] = np.interp(
            adjusted["month_index"].to_numpy(dtype=float),
            yearly_midpoints,
            yearly_scales,
            left=yearly_scales[0],
            right=yearly_scales[-1],
        )
    elif len(scale_points) == 1:
        adjusted["ode_smoothed_year_reporting_adjustment"] = float(
            scale_points["ode_year_reporting_adjustment_raw"].iloc[0]
        )
    else:
        adjusted["ode_smoothed_year_reporting_adjustment"] = 1.0
    adjusted["ode_smoothed_year_reporting_adjustment"] = pd.to_numeric(
        adjusted["ode_smoothed_year_reporting_adjustment"], errors="coerce"
    ).fillna(1.0).clip(lower=0.02, upper=80.0)
    adjusted = adjusted.merge(scale.loc[:, ["year", "ode_year_reporting_adjustment_raw"]], on="year", how="left")
    adjusted["ode_year_reporting_adjustment_raw"] = adjusted["ode_year_reporting_adjustment_raw"].fillna(1.0)
    if "reported_cases_unadjusted" not in adjusted:
        adjusted["reported_cases_unadjusted"] = adjusted["reported_cases_predicted"]
    predicted_month = adjusted.groupby(["year", "month"], as_index=False).agg(
        predicted_cases_before_year_adjustment_month=("reported_cases_before_year_adjustment", "sum"),
        ode_smoothed_year_reporting_adjustment=("ode_smoothed_year_reporting_adjustment", "mean"),
    )
    month_scale = predicted_month.merge(observed_month.loc[:, ["year", "month", "observed_cases"]], on=["year", "month"], how="left")
    month_scale["observed_cases"] = pd.to_numeric(month_scale["observed_cases"], errors="coerce")
    month_scale["predicted_cases_before_year_adjustment_month"] = pd.to_numeric(
        month_scale["predicted_cases_before_year_adjustment_month"], errors="coerce"
    )
    month_scale["ode_month_reporting_adjustment_raw"] = (
        (month_scale["observed_cases"] + 1.0)
        / (month_scale["predicted_cases_before_year_adjustment_month"] + 1.0)
    ).clip(lower=0.02, upper=80.0)
    month_scale["month_index"] = month_scale["year"] * 12 + month_scale["month"]
    month_scale = month_scale.sort_values("month_index").reset_index(drop=True)
    month_scale["ode_month_reporting_adjustment_raw"] = month_scale[
        "ode_month_reporting_adjustment_raw"
    ].fillna(month_scale["ode_smoothed_year_reporting_adjustment"])
    log_month = np.log(month_scale["ode_month_reporting_adjustment_raw"].clip(lower=0.02, upper=80.0))
    log_year_bridge = np.log(month_scale["ode_smoothed_year_reporting_adjustment"].clip(lower=0.02, upper=80.0))
    smoothed_log_month = log_month.rolling(window=3, min_periods=1, center=True).mean()
    month_scale["ode_smoothed_month_reporting_adjustment"] = np.exp(
        0.65 * smoothed_log_month + 0.35 * log_year_bridge
    ).clip(lower=0.02, upper=80.0)
    adjusted = adjusted.merge(
        month_scale.loc[
            :,
            [
                "year",
                "month",
                "ode_month_reporting_adjustment_raw",
                "ode_smoothed_month_reporting_adjustment",
            ],
        ],
        on=["year", "month"],
        how="left",
    )
    adjusted["ode_month_reporting_adjustment_raw"] = adjusted["ode_month_reporting_adjustment_raw"].fillna(
        adjusted["ode_smoothed_year_reporting_adjustment"]
    )
    adjusted["ode_smoothed_month_reporting_adjustment"] = adjusted["ode_smoothed_month_reporting_adjustment"].fillna(
        adjusted["ode_smoothed_year_reporting_adjustment"]
    )
    adjusted["ode_year_total_rescale_after_smoothing"] = 1.0
    adjusted["reported_cases_predicted"] = (
        adjusted["reported_cases_before_year_adjustment"] * adjusted["ode_smoothed_month_reporting_adjustment"]
    )
    adjusted["ode_year_reporting_adjustment"] = adjusted["ode_smoothed_month_reporting_adjustment"].clip(lower=0.02, upper=80.0)
    adjusted["ode_reporting_adjustment_method"] = (
        "Smoothed residual observation-process calibration: fitted ODE monthly age pattern is aligned with log-smoothed monthly reporting ratios where monthly observations exist, with a smoothed annual bridge used as shrinkage and fallback rather than a hard year-boundary step."
    )
    return adjusted.drop(columns=["month_index"], errors="ignore")


def apply_ode_age_reporting_adjustment(
    context: CountryContext,
    ode_state: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    observed_age: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ode_state.empty:
        return ode_state
    adjusted = ode_state.copy()
    if "reported_cases_unadjusted" not in adjusted:
        adjusted["reported_cases_unadjusted"] = adjusted["reported_cases_predicted"]
    if observed_age is None:
        observed = inputs.get("age_cases", pd.DataFrame())
        if observed.empty or "model_age_group" not in observed:
            adjusted["ode_age_reporting_adjustment"] = 1.0
            return adjusted
        value_column = "cases_total" if "cases_total" in observed.columns else "observed_cases" if "observed_cases" in observed.columns else None
        if value_column is None:
            adjusted["ode_age_reporting_adjustment"] = 1.0
            return adjusted
        observed = observed.loc[
            observed["iso3"].astype(str).eq(context.iso3)
            & pd.to_numeric(observed["year"], errors="coerce").between(
                MODEL2C_OBSERVATION_START_YEAR,
                MODEL2C_OBSERVATION_END_YEAR,
            )
            & pd.to_numeric(observed[value_column], errors="coerce").notna()
        ].copy()
    else:
        observed = observed_age.copy()
        if observed.empty or "model_age_group" not in observed:
            adjusted["ode_age_reporting_adjustment"] = 1.0
            return adjusted
        observed["year"] = pd.to_numeric(observed["year"], errors="coerce").astype("Int64")
        value_column = "observed_cases" if "observed_cases" in observed.columns else "cases_total" if "cases_total" in observed.columns else None
        if value_column is None:
            adjusted["ode_age_reporting_adjustment"] = 1.0
            return adjusted
        observed[value_column] = pd.to_numeric(observed[value_column], errors="coerce")
        observed = observed.loc[
            observed["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
            & observed[value_column].notna()
        ].copy()
        if "iso3" in observed.columns:
            observed = observed.loc[observed["iso3"].astype(str).eq(context.iso3)].copy()
    if observed.empty:
        adjusted["ode_age_reporting_adjustment"] = 1.0
        return adjusted
    observed = observed.rename(columns={value_column: "observation_cases"}) if value_column is not None else observed
    observed_age = observed.groupby("model_age_group", as_index=False).agg(observed_age_cases=("observation_cases", "sum"))
    predicted_age = adjusted.groupby("age_group", as_index=False).agg(predicted_age_cases=("reported_cases_predicted", "sum"))
    scale = observed_age.merge(predicted_age, left_on="model_age_group", right_on="age_group", how="right")
    scale["observed_age_cases"] = pd.to_numeric(scale["observed_age_cases"], errors="coerce").fillna(0.0)
    scale["predicted_age_cases"] = pd.to_numeric(scale["predicted_age_cases"], errors="coerce").fillna(0.0)
    scale["ode_age_reporting_adjustment"] = (
        (scale["observed_age_cases"] + 1.0) / (scale["predicted_age_cases"] + 1.0)
    ).clip(lower=0.05, upper=20.0)
    adjusted = adjusted.merge(
        scale.loc[:, ["age_group", "ode_age_reporting_adjustment"]],
        on="age_group",
        how="left",
    )
    adjusted["ode_age_reporting_adjustment"] = adjusted["ode_age_reporting_adjustment"].fillna(1.0)
    adjusted["reported_cases_predicted"] = adjusted["reported_cases_predicted"] * adjusted["ode_age_reporting_adjustment"]
    return adjusted


def apply_ode_observation_adjustments(
    context: CountryContext,
    ode_state: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    observed_month: pd.DataFrame | None = None,
    observed_age: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ode_state.empty:
        return ode_state
    adjusted = ode_state.copy()
    adjusted["reported_cases_unadjusted"] = adjusted["reported_cases_predicted"]
    adjusted = apply_ode_age_reporting_adjustment(context, adjusted, inputs, observed_age=observed_age)
    adjusted["reported_cases_before_year_adjustment"] = adjusted["reported_cases_predicted"]
    adjusted = apply_ode_year_reporting_adjustment(context, adjusted, observed_month=observed_month)
    if "ode_age_reporting_adjustment" not in adjusted:
        adjusted["ode_age_reporting_adjustment"] = 1.0
    adjusted["ode_reporting_adjustment_method"] = (
        "Observation-process calibration: an external diagnostic/reporting prior is applied in the ODE observation mapping, age-group reporting ratios are estimated from allocated annual age-bin totals, then residual monthly all-age reporting ratios are log-smoothed with annual bridge shrinkage."
    )
    return adjusted


def ode_initial_state(
    age_row: pd.Series,
    context: CountryContext,
    initial_infectious_scale: float = 1.0,
    initial_reported_case_target: float | None = None,
    initial_reporting_multiplier: float = 1.0,
    initial_symptom_fraction: float | None = None,
) -> np.ndarray:
    population = max(float(age_row["population"]), 1.0)
    base = initialize_monthly_state(age_row, context)
    susceptible = base["susceptible_naive"]
    vaccine_recent = base["vaccine_recent"]
    immunity_waned = base["vaccine_waned"] + base["infection_waned"]
    seeding_scale = float(np.clip(initial_infectious_scale, 0.05, 35.0))
    expected_monthly = max(float(age_row.get("true_infections", 0.0)), 0.0) / 12.0
    base_exposed = expected_monthly * 0.20
    base_infectious = max(float(age_row.get("symptomatic_infections", 0.0)), 0.0) / 12.0 * 0.16
    exposed = base_exposed
    infectious = base_infectious
    target = float(initial_reported_case_target) if initial_reported_case_target is not None else np.nan
    reporting_multiplier = float(initial_reporting_multiplier)
    symptom_fraction = (
        float(initial_symptom_fraction)
        if initial_symptom_fraction is not None and np.isfinite(initial_symptom_fraction)
        else np.nan
    )
    if target > 0 and reporting_multiplier > 0 and symptom_fraction > 0:
        target_symptomatic = target / reporting_multiplier
        target_exposed = target_symptomatic / max(0.85 * symptom_fraction, 1e-9)
        target_infectious = target_exposed * 0.08
        blend = MONTHLY_ODE_INITIAL_OBSERVATION_BLEND
        exposed = (1.0 - blend) * base_exposed + blend * target_exposed
        infectious = (1.0 - blend) * base_infectious + blend * target_infectious
    exposed *= seeding_scale
    infectious *= seeding_scale
    exposed = min(max(exposed, 0.0), population * 0.020)
    infectious = min(max(infectious, 0.0), population * 0.010)
    recovered = base["infection_recent"] + base["recovered"] + base["maternal_protected"]
    state = np.array([susceptible, vaccine_recent, immunity_waned, exposed, infectious, recovered, 0.0], dtype=float)
    compartment_total = state[:6].sum()
    if compartment_total <= 0:
        state[0] = population
        return state
    state[:6] *= population / compartment_total
    return state


def ode_rhs(
    _t: float,
    y: np.ndarray,
    beta: float,
    contact_multiplier: float,
    seasonal_multiplier: float,
    importation_force: float,
    waning_rate: float,
    susceptibility_scale: np.ndarray,
    disease_scale: np.ndarray,
) -> np.ndarray:
    state = np.maximum(y.reshape(len(AGE_ORDER), 7), 0.0)
    compartments = state[:, :6]
    population = np.maximum(compartments.sum(axis=1), 1.0)
    infectious = compartments[:, 4]
    infection_pressure = float(np.sum(AGE_REL_CONTACT_VECTOR * infectious / population) / max(AGE_REL_CONTACT_VECTOR_SUM, 1e-9))
    importation_force = float(np.clip(importation_force, 0.0, 0.08))
    lam = contact_multiplier * seasonal_multiplier * (
        beta * AGE_REL_CONTACT_VECTOR * infection_pressure + importation_force * AGE_REL_CONTACT_VECTOR
    )
    lam = np.clip(lam, 0.0, 0.85)
    s = state[:, 0]
    v = state[:, 1]
    w = state[:, 2]
    e = state[:, 3]
    i = state[:, 4]
    r = state[:, 5]
    n = np.maximum(s + v + w + e + i + r, 1.0)
    new_s = lam * s
    new_v = lam * 0.35 * susceptibility_scale * v
    new_w = lam * 0.75 * susceptibility_scale * w
    new_infections = np.minimum(new_s + new_v + new_w, 0.35 * n)
    vaccine_wane = v * min(waning_rate / 12.0, 0.10)
    immune_wane = r * 0.010
    boost = np.minimum(w * lam * 0.20, w)
    sigma = 2.0
    gamma = 1.3
    symptom_fraction = AGE_SYMPTOM_PROB_VECTOR * disease_scale
    symptomatic_incidence = sigma * e * symptom_fraction
    dydt = np.zeros_like(state)
    dydt[:, 0] = -new_s + immune_wane
    dydt[:, 1] = -new_v - vaccine_wane + boost
    dydt[:, 2] = vaccine_wane - new_w - boost
    dydt[:, 3] = new_infections - sigma * e
    dydt[:, 4] = sigma * e - gamma * i
    dydt[:, 5] = gamma * i - immune_wane
    dydt[:, 6] = symptomatic_incidence
    return dydt.reshape(-1)


def integrate_monthly_ode_state(y0: np.ndarray, args: tuple[Any, ...], steps: int = 4) -> np.ndarray:
    y = np.asarray(y0, dtype=float).copy()
    dt = 1.0 / max(int(steps), 1)
    for _ in range(max(int(steps), 1)):
        k1 = ode_rhs(0.0, y, *args)
        k2 = ode_rhs(0.0, np.maximum(y + 0.5 * dt * k1, 0.0), *args)
        k3 = ode_rhs(0.0, np.maximum(y + 0.5 * dt * k2, 0.0), *args)
        k4 = ode_rhs(0.0, np.maximum(y + dt * k3, 0.0), *args)
        y = np.maximum(y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)
    return y


def simulate_monthly_ode(
    context: CountryContext,
    params: dict[str, float],
    age_pred: pd.DataFrame,
    beta_scale: float = 1.0,
    reporting_scale: float = 1.0,
    detection_prior_use_testing_denominator: bool = True,
    cache: dict[str, Any] | None = None,
    **ode_option_overrides: float,
) -> pd.DataFrame:
    if age_pred.empty:
        return pd.DataFrame()
    cache = cache or {}
    annual_age = cache.get("annual_age")
    if annual_age is None:
        annual_age = age_pred.loc[
            age_pred["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        ].copy()
        if annual_age.empty:
            return pd.DataFrame()
        annual_age["year"] = pd.to_numeric(annual_age["year"], errors="coerce").astype(int)
        annual_age = annual_age.sort_values(["year", "age_group"])
    if annual_age.empty:
        return pd.DataFrame()
    ode_options = monthly_ode_options_from_kwargs(beta_scale, reporting_scale, **ode_option_overrides)
    seasonal = cache.get("seasonal_lookup")
    if seasonal is None:
        seasonal = monthly_seasonal_fractions(context).set_index("month")["seasonal_fraction"].to_dict()
    if cache.get("detection_prior_use_testing_denominator") == bool(detection_prior_use_testing_denominator):
        detection_prior = cache.get("detection_prior_lookup", {})
    else:
        detection_prior = monthly_detection_prior_table(
            context,
            include_testing_denominator=detection_prior_use_testing_denominator,
        ).set_index(["year", "month"]).to_dict("index")
    pathogen_prior = cache.get("pathogen_prior_lookup")
    if pathogen_prior is None:
        pathogen_prior = monthly_pathogen_prior_table(context).set_index(["year", "month"]).to_dict("index")
    observed_month_lookup = cache.get("observed_month_lookup")
    if observed_month_lookup is None:
        observed_month = monthly_observed_cases(context)
        if not observed_month.empty:
            observed_month_lookup = (
                observed_month.loc[
                    observed_month["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
                ]
                .groupby(["year", "month"])["observed_cases"]
                .sum()
                .to_dict()
            )
        else:
            observed_month_lookup = {}
    monthly_covid_lookup = cache.get("monthly_covid_lookup")
    if monthly_covid_lookup is None:
        monthly_covid = context.monthly_covid.copy()
        if not monthly_covid.empty:
            monthly_covid["year"] = pd.to_numeric(monthly_covid["year"], errors="coerce").astype("Int64")
            monthly_covid["month"] = pd.to_numeric(monthly_covid["month"], errors="coerce").astype("Int64")
            monthly_covid["monthly_stringency_index"] = pd.to_numeric(
                monthly_covid["monthly_stringency_index"], errors="coerce"
            )
            monthly_covid = monthly_covid.dropna(subset=["year", "month"])
            monthly_covid["year"] = monthly_covid["year"].astype(int)
            monthly_covid["month"] = monthly_covid["month"].astype(int)
            monthly_covid_lookup = monthly_covid.set_index(["year", "month"]).to_dict("index")
        else:
            monthly_covid_lookup = {}
    else:
        monthly_covid_lookup = dict(monthly_covid_lookup)
    states: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    vaccine_type = str(context.metadata.get("routine_scheduler_code", context.metadata.get("vaccine_type_primary", "unknown")))
    is_ap = vaccine_type == "aP"
    base_ve_infection = 0.50 if is_ap else 0.42
    base_ve_disease = 0.88 if is_ap else 0.82

    for year in sorted(annual_age["year"].unique()):
        year_age = annual_age.loc[annual_age["year"].eq(year)].set_index("age_group")
        year_age_records = {age: {**record, "age_group": age} for age, record in year_age.to_dict("index").items()}
        year_age_groups = [age for age in AGE_ORDER if age in year_age_records]
        annual_row = context.annual.loc[context.annual["year"].eq(year)]
        annual_stringency = float(annual_row["stringency_index"].mean()) if not annual_row.empty else 0.0
        expected_incidence = (
            float(annual_row["periodic_expected_incidence_per_100k"].mean())
            if not annual_row.empty and "periodic_expected_incidence_per_100k" in annual_row
            else context.pre_mean_incidence
        )
        expected_incidence = max(expected_incidence, context.pre_mean_incidence * 0.25, 0.001)
        base_importation = expected_incidence / 100000.0 / 12.0
        for month in range(1, 13):
            phase_beta_scale, phase_reporting_scale, pandemic_phase_weight = monthly_ode_phase_multipliers(
                year, month, ode_options
            )
            detection = {**default_detection_prior_row(), **detection_prior.get((year, month), {})}
            detection_multiplier = float(np.clip(detection["ode_relative_detection_prior_mean"], 0.05, 12.0))
            report_multiplier = params["post_reporting_multiplier"] if year >= 2023 else 1.0
            observed_initial_month_cases = float(observed_month_lookup.get((year, month), np.nan))
            use_observed_initial_conditions = bool(not states and np.isfinite(observed_initial_month_cases) and observed_initial_month_cases > 0)
            age_initial_weights: dict[str, float] = {}
            if use_observed_initial_conditions:
                age_weight_values: dict[str, float] = {}
                for candidate_age in AGE_ORDER:
                    if candidate_age not in year_age_records:
                        continue
                    candidate_row = year_age_records[candidate_age]
                    age_weight_values[candidate_age] = max(float(candidate_row.get("reported_cases_predicted", 0.0)), 0.0)
                total_weight = float(sum(age_weight_values.values()))
                if total_weight <= 0:
                    age_weight_values = {
                        age: max(float(year_age_records[age].get("population", 0.0)), 0.0)
                        for age in year_age_groups
                    }
                    total_weight = float(sum(age_weight_values.values()))
                if total_weight > 0:
                    age_initial_weights = {age: value / total_weight for age, value in age_weight_values.items()}
            y0_parts = []
            population_by_age = []
            susceptibility_scale = []
            disease_scale = []
            initial_condition_method_by_age: dict[str, str] = {}
            initial_condition_target_by_age: dict[str, float] = {}
            for age_group in year_age_groups:
                age_row = year_age_records[age_group]
                population = max(float(age_row["population"]), 1.0)
                dtp3 = float(age_row.get("dtp3_coverage", context.pre_mean_dtp3) or context.pre_mean_dtp3)
                reach = AGE_VACCINE_PROGRAM_REACH[age_group]
                waning_multiplier = float(np.exp(-params["waning_rate"] * AGE_WANING_YEARS[age_group] / 8.0))
                vaccine_protection = np.clip(dtp3 * reach * base_ve_infection * waning_multiplier, 0.0, 0.95)
                disease_protection = np.clip(dtp3 * reach * base_ve_disease * waning_multiplier, 0.0, 0.98)
                age_susceptibility_scale = 1.0 - 0.55 * vaccine_protection
                age_disease_scale = float(np.clip(1.0 - 0.45 * disease_protection, 0.20, 1.0))
                symptom_fraction = AGE_SYMPTOM_PROB[age_group] * age_disease_scale
                initial_target = (
                    observed_initial_month_cases * age_initial_weights.get(age_group, 0.0)
                    if use_observed_initial_conditions
                    else np.nan
                )
                initial_reporting_multiplier = (
                    params["reporting_rate"]
                    * report_multiplier
                    * phase_reporting_scale
                    * detection_multiplier
                    * AGE_REPORT_WEIGHT[age_group]
                )
                previous = states.get(age_group)
                if previous is None:
                    previous = ode_initial_state(
                        age_row,
                        context,
                        ode_options["ode_initial_infectious_scale"],
                        initial_reported_case_target=initial_target,
                        initial_reporting_multiplier=initial_reporting_multiplier,
                        initial_symptom_fraction=symptom_fraction,
                    )
                    initial_condition_method_by_age[age_group] = (
                        "first_month_observation_blended_initialization"
                        if use_observed_initial_conditions and initial_target > 0
                        else "annual_model_seed_initialization"
                    )
                    initial_condition_target_by_age[age_group] = float(initial_target) if np.isfinite(initial_target) else np.nan
                else:
                    initial_condition_method_by_age[age_group] = "carried_forward_monthly_state"
                    initial_condition_target_by_age[age_group] = np.nan
                previous = np.maximum(previous, 0.0)
                previous[:6] *= population / max(previous[:6].sum(), 1e-9)
                previous[6] = 0.0
                y0_parts.append(previous)
                population_by_age.append(population)
                susceptibility_scale.append(age_susceptibility_scale)
                disease_scale.append(age_disease_scale)
            if not y0_parts:
                continue

            seasonal_source_month, effective_seasonal_shift = monthly_ode_seasonal_source_month(month, year, ode_options)
            seasonal_fraction = float(seasonal.get(seasonal_source_month, 1.0 / 12.0))
            seasonal_base = max(seasonal_fraction * 12.0, 1e-6)
            seasonal_multiplier = float(np.clip(np.exp(ode_options["ode_seasonal_strength"] * np.log(seasonal_base)), 0.08, 6.0))
            importation_force = float(
                np.clip(base_importation * ode_options["ode_importation_scale"] * seasonal_multiplier, 0.0, 0.08)
            )
            pathogen = {**default_pathogen_prior_row(), **pathogen_prior.get((year, month), {})}
            pathogen_pressure = float(np.clip(pathogen["ode_pathogen_pressure_prior_mean"], 0.50, 4.00))
            pathogen_multiplier = float(
                np.clip(1.0 + ode_options["ode_pathogen_drift_scale"] * (pathogen_pressure - 1.0), 0.40, 6.00)
            )
            covid_policy = monthly_covid_lookup.get((year, month), {})
            monthly_stringency = float(covid_policy.get("monthly_stringency_index", np.nan))
            if 2020 <= year <= 2022:
                if np.isfinite(monthly_stringency):
                    contact_policy_source = "monthly_oxcgrt"
                    stringency_for_contact = monthly_stringency
                else:
                    contact_policy_source = "annual_oxcgrt_fallback"
                    stringency_for_contact = annual_stringency
                contact_multiplier = max(
                    0.18,
                    1.0 - params["contact_sensitivity"] * float(np.clip(stringency_for_contact, 0.0, 100.0)) / 100.0,
                )
            else:
                contact_policy_source = "outside_covid_contact_window"
                stringency_for_contact = 0.0
                contact_multiplier = 1.0
            y0 = np.vstack(y0_parts).reshape(-1)
            ode_args = (
                max(params["beta"] * phase_beta_scale * pathogen_multiplier * 35.0, 1e-9),
                contact_multiplier,
                seasonal_multiplier,
                importation_force,
                params["waning_rate"],
                np.asarray(susceptibility_scale, dtype=float),
                np.asarray(disease_scale, dtype=float),
            )
            end_state = np.maximum(integrate_monthly_ode_state(y0, ode_args).reshape(len(y0_parts), 7), 0.0)
            for idx, age_group in enumerate(year_age_groups):
                population = population_by_age[idx]
                state = end_state[idx]
                state[:6] *= population / max(state[:6].sum(), 1e-9)
                states[age_group] = state.copy()
                symptomatic = max(float(state[6]), 0.0)
                reported_before_detection_prior = (
                    symptomatic * params["reporting_rate"] * report_multiplier * phase_reporting_scale * AGE_REPORT_WEIGHT[age_group]
                )
                reported = reported_before_detection_prior * detection_multiplier
                true_infections = symptomatic / max(AGE_SYMPTOM_PROB[age_group] * disease_scale[idx], 1e-6)
                rows.append(
                    {
                        "iso3": context.iso3,
                        "country": context.country,
                        "date_start": f"{year}-{month:02d}-01",
                        "date_end": month_end(year, month),
                        "year": year,
                        "month": month,
                        "age_group": age_group,
                        "age_label": AGE_LABELS[age_group],
                        "population": population,
                        "ode_beta_scale": ode_options["ode_beta_scale"],
                        "ode_reporting_scale": ode_options["ode_reporting_scale"],
                        "ode_pandemic_beta_scale": ode_options["ode_pandemic_beta_scale"],
                        "ode_post_beta_scale": ode_options["ode_post_beta_scale"],
                        "ode_pandemic_reporting_scale": ode_options["ode_pandemic_reporting_scale"],
                        "ode_post_reporting_scale": ode_options["ode_post_reporting_scale"],
                        "ode_importation_scale": ode_options["ode_importation_scale"],
                        "ode_seasonal_strength": ode_options["ode_seasonal_strength"],
                        "ode_pandemic_seasonal_shift_months": ode_options["ode_pandemic_seasonal_shift_months"],
                        "ode_post_seasonal_shift_months": ode_options["ode_post_seasonal_shift_months"],
                        "ode_initial_infectious_scale": ode_options["ode_initial_infectious_scale"],
                        "ode_initial_condition_method": initial_condition_method_by_age.get(age_group, "not_available"),
                        "ode_initial_observed_total_cases": observed_initial_month_cases if use_observed_initial_conditions else np.nan,
                        "ode_initial_age_target_reported_cases": initial_condition_target_by_age.get(age_group, np.nan),
                        "ode_initial_observation_blend": (
                            MONTHLY_ODE_INITIAL_OBSERVATION_BLEND if use_observed_initial_conditions else 0.0
                        ),
                        "ode_pathogen_drift_scale": ode_options["ode_pathogen_drift_scale"],
                        "ode_phase_beta_multiplier": phase_beta_scale,
                        "ode_phase_reporting_multiplier": phase_reporting_scale,
                        "ode_pandemic_phase_weight": pandemic_phase_weight,
                        "ode_pathogen_transmission_multiplier": pathogen_multiplier,
                        **pathogen,
                        "contact_multiplier": contact_multiplier,
                        "monthly_stringency_index": stringency_for_contact,
                        "annual_stringency_index": annual_stringency,
                        "covid_contact_policy_source": contact_policy_source,
                        "covid_contact_shock_resolution": (
                            "month-specific OxCGRT contact shock in the monthly ODE; annual policy averages are used only when monthly OxCGRT rows are unavailable."
                        ),
                        "seasonal_source_month": seasonal_source_month,
                        "effective_seasonal_shift_months": effective_seasonal_shift,
                        "seasonal_fraction": seasonal_fraction,
                        "seasonal_multiplier": seasonal_multiplier,
                        "importation_force": importation_force,
                        "true_infections": true_infections,
                        "symptomatic_infections": symptomatic,
                        "reported_cases_before_detection_prior": reported_before_detection_prior,
                        "reported_cases_predicted": reported,
                        **detection,
                        "susceptible_naive": state[0],
                        "vaccine_recent": state[1],
                        "immunity_waned": state[2],
                        "exposed": state[3],
                        "infectious": state[4],
                        "recovered": state[5],
                        "state_total": state[:6].sum(),
                        "model_note": "Fitted continuous-time monthly age-stratified ODE integrated by fixed-step RK4 with phase-specific transmission/reporting scales, seasonal forcing, endemic re-seeding, pathogen strain/AMR pressure prior in the transmission mapping, and an external diagnostic/reporting prior in the observation mapping.",
                    }
                )
    return pd.DataFrame(rows)


def fit_monthly_ode_parameters(
    context: CountryContext,
    params: dict[str, float],
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame] | None = None,
    *,
    age_objective_weight: float | None = None,
    joint_age_month_objective_weight: float | None = None,
    parallel_workers: int | None = None,
    starting_options: dict[str, float] | None = None,
    score_state: dict[str, Any] | None = None,
    score_pool: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    if score_state is not None:
        worker_state = score_state
        context = worker_state["context"]
        params = worker_state["params"]
        age_pred = worker_state["age_pred"]
        inputs = worker_state["inputs"]
        observed = worker_state.get("observed", pd.DataFrame())
        observed_age = worker_state.get("observed_age", pd.DataFrame())
        age_objective_weight = float(worker_state["age_objective_weight"])
    else:
        observed = monthly_observed_cases(context)
        observed = observed.loc[
            observed["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        ].copy()
        observed_age = pd.DataFrame()
        if inputs is not None:
            observed_age = allocated_age_observations(inputs, context.iso3)
            if not observed_age.empty:
                observed_age = observed_age.loc[
                    observed_age["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
                ].copy()

    def empty_fit(message: str, nfev: int = 0) -> tuple[dict[str, float], dict[str, Any]]:
        defaults = monthly_ode_default_options()
        return (
            {
                **defaults,
                "ode_grid_beta_scale": np.nan,
                "ode_grid_reporting_scale": np.nan,
                "ode_grid_objective_value": np.nan,
                "ode_local_refinement_objective_value": np.nan,
                "ode_objective_improvement_vs_grid": np.nan,
                "ode_local_refinement_success": False,
                "ode_local_refinement_status": pd.NA,
                "ode_local_refinement_message": message,
                "ode_local_refinement_nfev": 0,
                "ode_joint_age_weight": np.nan,
                "ode_joint_age_month_weight": np.nan,
                "ode_monthly_calibration_score": np.nan,
                "ode_age_calibration_score": np.nan,
                "ode_joint_age_month_calibration_score": np.nan,
                "ode_monthly_calibration_n": 0,
                "ode_age_calibration_effective_n": 0.0,
                "ode_joint_age_month_effective_n": 0.0,
                "ode_penalty_score": np.nan,
                "ode_joint_age_month_nb_log_likelihood": np.nan,
                "ode_joint_age_month_weighted_nb_log_likelihood": np.nan,
                "ode_joint_age_month_likelihood_rows": 0,
            },
            {"success": False, "fun": np.nan, "nfev": nfev, "method": message},
        )

    if observed.empty and observed_age.empty:
        return empty_fit("no monthly or annual age observations")

    best: dict[str, float] | None = None
    evaluations = 0
    age_objective_weight = (
        float(np.clip(age_objective_weight, 0.0, 1.0))
        if age_objective_weight is not None
        else MONTHLY_ODE_AGE_OBJECTIVE_WEIGHT
    )
    age_objective_weight = age_objective_weight if not observed_age.empty else 0.0
    joint_age_month_objective_weight = (
        float(np.clip(joint_age_month_objective_weight, 0.0, 1.0))
        if joint_age_month_objective_weight is not None
        else MONTHLY_ODE_JOINT_AGE_MONTH_OBJECTIVE_WEIGHT
    )
    joint_age_month_objective_weight = (
        joint_age_month_objective_weight if not observed.empty and not observed_age.empty else 0.0
    )
    if parallel_workers is None:
        parallel_workers = _default_monthly_ode_parallel_workers()
    parallel_workers = max(1, int(parallel_workers))
    if score_state is None:
        worker_state = _build_monthly_ode_score_state(
            context,
            params,
            age_pred,
            inputs if inputs is not None else {},
            observed_month=observed,
            observed_age=observed_age,
            age_objective_weight=age_objective_weight,
            include_joint_observation=joint_age_month_objective_weight > 0.0,
        )
    cache = worker_state["cache"]

    def score_options(options: dict[str, float]) -> dict[str, float] | None:
        return _score_monthly_ode_options(worker_state, joint_age_month_objective_weight, options)

    pool: ProcessPoolExecutor | None = score_pool
    owns_pool = False
    if pool is None and parallel_workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=min(parallel_workers, 16),
            initializer=_initialize_monthly_ode_score_worker,
            initargs=(worker_state,),
        )
        owns_pool = True

    def evaluate_options_batch(options_batch: list[dict[str, float]]) -> list[dict[str, float] | None]:
        if not options_batch:
            return []
        if pool is None or parallel_workers <= 1:
            return [_score_monthly_ode_options(worker_state, joint_age_month_objective_weight, options) for options in options_batch]
        chunksize = max(1, len(options_batch) // max(1, parallel_workers * 2))
        payloads = [(joint_age_month_objective_weight, options) for options in options_batch]
        return list(pool.map(_score_monthly_ode_options_worker, payloads, chunksize=chunksize))

    def calibrate_reporting_scales(options: dict[str, float]) -> dict[str, float]:
        base = monthly_ode_default_options(options)
        base["ode_reporting_scale"] = 1.0
        base["ode_pandemic_reporting_scale"] = 1.0
        base["ode_post_reporting_scale"] = 1.0
        predicted = simulate_monthly_ode(
            context,
            params,
            age_pred,
            beta_scale=base["ode_beta_scale"],
            reporting_scale=1.0,
            cache=cache,
            **{key: value for key, value in base.items() if key not in {"ode_beta_scale", "ode_reporting_scale"}},
        )
        if predicted.empty:
            return monthly_ode_default_options(options)
        predicted_month = predicted.groupby(["iso3", "country", "year", "month"], as_index=False).agg(
            predicted_cases=("reported_cases_predicted", "sum")
        )
        merged = observed.merge(predicted_month, on=["iso3", "country", "year", "month"], how="left") if not observed.empty else pd.DataFrame()
        if merged.empty:
            return monthly_ode_default_options(options)
        merged["predicted_cases"] = merged["predicted_cases"].fillna(0.0)

        def period_ratio(mask: pd.Series) -> float:
            valid_month = mask & merged["predicted_cases"].gt(0) & merged["observed_cases"].ge(0)
            if valid_month.any():
                ratios = (
                    (merged.loc[valid_month, "observed_cases"].to_numpy(dtype=float) + 1.0)
                    / (merged.loc[valid_month, "predicted_cases"].to_numpy(dtype=float) + 1.0)
                )
                return weighted_median(ratios, np.ones(len(ratios), dtype=float))
            return np.nan

        pre_ratio = period_ratio(merged["year"].lt(2020))
        if not np.isfinite(pre_ratio):
            pre_ratio = period_ratio(
                merged["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
            )
        pandemic_ratio = period_ratio(merged["year"].between(2020, 2022))
        post_ratio = period_ratio(merged["year"].ge(2023))

        calibrated = monthly_ode_default_options(options)
        calibrated["ode_reporting_scale"] = float(np.clip(pre_ratio if np.isfinite(pre_ratio) else 1.0, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_reporting_scale"]))
        if np.isfinite(pandemic_ratio) and calibrated["ode_reporting_scale"] > 0:
            calibrated["ode_pandemic_reporting_scale"] = float(
                np.clip(pandemic_ratio / calibrated["ode_reporting_scale"], *MONTHLY_ODE_PARAMETER_BOUNDS["ode_pandemic_reporting_scale"])
            )
        if np.isfinite(post_ratio) and calibrated["ode_reporting_scale"] > 0:
            calibrated["ode_post_reporting_scale"] = float(
                np.clip(post_ratio / calibrated["ode_reporting_scale"], *MONTHLY_ODE_PARAMETER_BOUNDS["ode_post_reporting_scale"])
            )
        return calibrated

    raw_starts = monthly_ode_calibration_start_options(starting_options=starting_options)
    starts = [calibrate_reporting_scales(start) for start in raw_starts]
    for metrics in evaluate_options_batch(starts):
        evaluations += 1
        if metrics is not None and (best is None or metrics["score"] < best["score"]):
            best = metrics

    if best is None:
        if owns_pool and pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
        return empty_fit("multi-parameter monthly ODE starts failed", nfev=evaluations)

    grid_best = best.copy()
    current = best.copy()
    local_refinement_nfev = 0
    try:
        for step in (0.45, 0.25):
            improved = True
            coordinate_passes = 0
            while improved and coordinate_passes < 2:
                coordinate_passes += 1
                improved = False
                trial_options: list[dict[str, float]] = []
                for name in MONTHLY_ODE_SEARCH_PARAMETERS:
                    for factor in (1.0 - step, 1.0 + step):
                        trial = monthly_ode_default_options(current)
                        lower, upper = MONTHLY_ODE_PARAMETER_BOUNDS[name]
                        if name in MONTHLY_ODE_SIGNED_PARAMETERS:
                            delta = 4.0 * step * (-1.0 if factor < 1.0 else 1.0)
                            trial[name] = float(np.clip(trial[name] + delta, lower, upper))
                        else:
                            trial[name] = float(np.clip(trial[name] * factor, lower, upper))
                        trial_options.append(trial)
                best_candidate: dict[str, float] | None = None
                for metrics in evaluate_options_batch(trial_options):
                    local_refinement_nfev += 1
                    evaluations += 1
                    if metrics is not None and np.isfinite(metrics["score"]) and metrics["score"] < current["score"]:
                        if best_candidate is None or metrics["score"] < best_candidate["score"]:
                            best_candidate = metrics
                if best_candidate is not None:
                    current = best_candidate
                    improved = True
    except Exception as exc:  # pragma: no cover - defensive guard for ODE optimizer edge cases
        current["coordinate_descent_exception"] = str(exc)

    powell_success = False
    powell_message = "Powell refinement skipped by default; set MODEL2_ODE_POWELL=1 to enable."
    powell_enabled = os.environ.get("MODEL2_ODE_POWELL", "").strip() == "1"
    if powell_enabled:
        bounds = [
            tuple(MONTHLY_ODE_PARAMETER_BOUNDS[name])
            if name in MONTHLY_ODE_SIGNED_PARAMETERS
            else tuple(np.log(MONTHLY_ODE_PARAMETER_BOUNDS[name]))
            for name in MONTHLY_ODE_PARAMETER_NAMES
        ]

        def objective(vector: np.ndarray) -> float:
            nonlocal evaluations, local_refinement_nfev
            options = monthly_ode_vector_to_options(vector)
            metrics = score_options(options)
            evaluations += 1
            local_refinement_nfev += 1
            if metrics is None or not np.isfinite(metrics["score"]):
                return 1e6
            return float(metrics["score"])

        result = minimize(
            objective,
            monthly_ode_options_to_vector(current),
            method="Powell",
            bounds=bounds,
            options={"maxiter": MONTHLY_ODE_OPTIMIZER_MAXITER, "maxfev": MONTHLY_ODE_OPTIMIZER_MAXITER * 2, "xtol": 0.025, "ftol": 0.002},
        )
        powell_success = bool(result.success)
        powell_message = str(result.message)
        powell_options = monthly_ode_vector_to_options(result.x)
        powell_metrics = score_options(powell_options)
        evaluations += 1
        if powell_metrics is not None and np.isfinite(powell_metrics["score"]) and powell_metrics["score"] < current["score"]:
            current = powell_metrics
        if not powell_success:
            powell_message = str(result.message)

    local_refinement_success = True
    local_refinement_status: Any = "completed" if powell_success else "coordinate_descent_completed"
    local_refinement_message = (
        "Multi-parameter ODE coordinate refinement completed; Powell refinement: " + powell_message
    )
    local_refinement_objective = float(current["score"])
    best = current

    fitted = {
        **{name: best[name] for name in MONTHLY_ODE_PARAMETER_NAMES},
        "ode_grid_beta_scale": grid_best["ode_beta_scale"],
        "ode_grid_reporting_scale": grid_best["ode_reporting_scale"],
        "ode_grid_objective_value": grid_best["score"],
        "ode_local_refinement_objective_value": local_refinement_objective,
        "ode_objective_improvement_vs_grid": (
            float(grid_best["score"] - best["score"]) if np.isfinite(grid_best["score"]) and np.isfinite(best["score"]) else np.nan
        ),
        "ode_local_refinement_success": local_refinement_success,
        "ode_local_refinement_status": local_refinement_status,
        "ode_local_refinement_message": local_refinement_message,
        "ode_local_refinement_nfev": local_refinement_nfev,
        "ode_joint_age_weight": age_objective_weight,
        "ode_joint_age_month_weight": joint_age_month_objective_weight,
        "ode_monthly_calibration_score": best["monthly_score"],
        "ode_age_calibration_score": best["age_score"],
        "ode_joint_age_month_calibration_score": best["joint_age_month_score"],
        "ode_monthly_calibration_n": int(best["monthly_n"]),
        "ode_age_calibration_effective_n": best["age_effective_n"],
        "ode_joint_age_month_effective_n": best["joint_age_month_effective_n"],
        "ode_penalty_score": best["penalty"],
        "ode_residual_adjustment_penalty_score": best["residual_adjustment_penalty"],
        "ode_transition_penalty_score": best["transition_penalty"],
        "ode_joint_age_month_nb_log_likelihood": best["joint_age_month_nb_log_likelihood"],
        "ode_joint_age_month_weighted_nb_log_likelihood": best["joint_age_month_weighted_nb_log_likelihood"],
        "ode_joint_age_month_likelihood_rows": best["joint_age_month_likelihood_rows"],
    }
    info = {
        "success": True,
        "fun": best["score"],
        "nfev": evaluations,
        "method": (
            "multi-parameter monthly ODE calibration with country-specific phase transmission/reporting scales, "
            "pathogen strain/AMR pressure, bounded pandemic/post-pandemic seasonal phase shifts, endemic re-seeding, "
            "composite monthly all-age plus annual age-structure plus joint age-month bridge objectives, coordinate refinement"
            + (" and bounded Powell refinement" if powell_enabled else "; Powell refinement is opt-in via MODEL2_ODE_POWELL=1")
        ),
    }
    if owns_pool and pool is not None:
        pool.shutdown(wait=True, cancel_futures=False)
    return fitted, info


def monthly_ode_likelihood_components(
    context: CountryContext,
    ode_state: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    observed_month: pd.DataFrame | None = None,
    observed_age: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ode_state.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    predicted_month = ode_state.groupby(["iso3", "country", "year", "month"], as_index=False).agg(predicted_cases=("reported_cases_predicted", "sum"))
    if observed_month is None:
        observed_month = monthly_observed_cases(context)
    else:
        observed_month = observed_month.copy()
    if not observed_month.empty:
        observed_month = observed_month.loc[
            observed_month["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        ].copy()
        merged = observed_month.merge(predicted_month, on=["iso3", "country", "year", "month"], how="inner")
        for item in merged.itertuples(index=False):
            observed_cases = float(item.observed_cases)
            predicted_cases = float(item.predicted_cases)
            log_residual = np.log1p(predicted_cases) - np.log1p(observed_cases)
            nb_ll = nb_log_likelihood(observed_cases, predicted_cases)
            rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "component": "monthly_ode_all_age",
                    "year": int(item.year),
                    "month": int(item.month),
                    "model_age_group": "all_ages",
                    "observed_cases": observed_cases,
                    "predicted_cases": predicted_cases,
                    "log_residual": log_residual,
                    "nb_log_likelihood": nb_ll,
                    "age_allocation_weight": 1.0,
                    "weighted_nb_log_likelihood": nb_ll,
                    "weighted_log_residual_square": log_residual**2,
                    "age_allocation_note": "All-age monthly observation; no age-bin allocation down-weighting.",
                    "nb_dispersion": NB_DISPERSION,
                    "likelihood_note": "Monthly all-age observations compared with fitted continuous-time monthly age-stratified ODE pilot predictions.",
                }
            )

    if observed_age is None:
        observed_age = allocated_age_observations(inputs, context.iso3)
    else:
        observed_age = observed_age.copy()
    if not observed_age.empty:
        predicted_age = ode_state.groupby(["iso3", "country", "year", "age_group"], as_index=False).agg(
            predicted_cases=("reported_cases_predicted", "sum")
        ).rename(columns={"age_group": "model_age_group"})
        merged_age = observed_age.merge(predicted_age, on=["iso3", "country", "year", "model_age_group"], how="inner")
        for item in merged_age.itertuples(index=False):
            observed_cases = float(item.observed_cases)
            predicted_cases = float(item.predicted_cases)
            log_residual = np.log1p(predicted_cases) - np.log1p(observed_cases)
            nb_ll = nb_log_likelihood(observed_cases, predicted_cases)
            allocation_weight = float(item.age_allocation_weight)
            rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "component": "monthly_ode_annual_age_group",
                    "year": int(item.year),
                    "month": np.nan,
                    "model_age_group": item.model_age_group,
                    "observed_cases": observed_cases,
                    "predicted_cases": predicted_cases,
                    "log_residual": log_residual,
                    "nb_log_likelihood": nb_ll,
                    "age_allocation_weight": allocation_weight,
                    "weighted_nb_log_likelihood": nb_ll * allocation_weight,
                    "weighted_log_residual_square": log_residual**2 * allocation_weight,
                    "age_allocation_note": item.age_allocation_note,
                    "nb_dispersion": NB_DISPERSION,
                    "likelihood_note": "Annual age-group observations compared with annual sums of fitted continuous-time monthly ODE predictions; broad source-bin allocations are down-weighted in weighted diagnostics.",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "iso3",
            "country",
            "component",
            "year",
            "month",
            "model_age_group",
            "observed_cases",
            "predicted_cases",
            "log_residual",
            "nb_log_likelihood",
            "age_allocation_weight",
            "weighted_nb_log_likelihood",
            "weighted_log_residual_square",
            "age_allocation_note",
            "nb_dispersion",
            "likelihood_note",
        ],
    )


def annual_predictions_from_monthly_ode(
    context: CountryContext,
    ode_state: pd.DataFrame,
    annual_pred: pd.DataFrame,
) -> pd.DataFrame:
    if ode_state.empty:
        return pd.DataFrame()
    state = ode_state.copy()
    state["year"] = pd.to_numeric(state["year"], errors="coerce").astype("Int64")
    state["reported_under_1y_component"] = np.where(
        state["age_group"].isin(["infant_0_2m", "infant_3_11m"]),
        state["reported_cases_predicted"],
        0.0,
    )
    state["reported_infant_under_6m_component"] = np.select(
        [
            state["age_group"].eq("infant_0_2m"),
            state["age_group"].eq("infant_3_11m"),
        ],
        [
            state["reported_cases_predicted"],
            state["reported_cases_predicted"] / 3.0,
        ],
        default=0.0,
    )
    if "reported_cases_unadjusted" not in state:
        state["reported_cases_unadjusted"] = state["reported_cases_predicted"]
    if "reported_cases_before_detection_prior" not in state:
        state["reported_cases_before_detection_prior"] = state["reported_cases_predicted"]
    if "reported_cases_before_year_adjustment" not in state:
        state["reported_cases_before_year_adjustment"] = state["reported_cases_predicted"]
    if "ode_year_reporting_adjustment" not in state:
        state["ode_year_reporting_adjustment"] = 1.0
    if "ode_age_reporting_adjustment" not in state:
        state["ode_age_reporting_adjustment"] = 1.0
    for column, default in {
        "ode_relative_detection_prior_mean": 1.0,
        "ode_relative_detection_prior_lower": 0.75,
        "ode_relative_detection_prior_upper": 1.35,
        "ode_pathogen_pressure_prior_mean": 1.0,
        "ode_pathogen_pressure_prior_lower": 0.85,
        "ode_pathogen_pressure_prior_upper": 1.20,
        "ode_pathogen_transmission_multiplier": 1.0,
        "ode_year_reporting_adjustment_raw": 1.0,
        "ode_smoothed_year_reporting_adjustment": 1.0,
        "ode_month_reporting_adjustment_raw": 1.0,
        "ode_smoothed_month_reporting_adjustment": 1.0,
        "ode_year_total_rescale_after_smoothing": 1.0,
        "monthly_stringency_index": 0.0,
        "annual_stringency_index": 0.0,
        "ode_pandemic_phase_weight": 0.0,
        "ode_initial_observed_total_cases": np.nan,
        "ode_initial_age_target_reported_cases": np.nan,
        "ode_initial_observation_blend": 0.0,
    }.items():
        if column not in state:
            state[column] = default
        state[column] = pd.to_numeric(state[column], errors="coerce").fillna(default)
    for column, default in {
        "ode_detection_effect_direction": "not_available",
        "ode_detection_evidence_strength": "none",
        "ode_surveillance_regime": "not_available",
        "ode_primary_diagnostic_methods": "not_available",
        "ode_case_definition_or_reporting_change": "not_available",
        "ode_detection_source_ids": "not_available",
        "ode_detection_prior_source_type": "not_available",
        "ode_pathogen_regime": "not_available",
        "ode_pathogen_features": "not_available",
        "ode_macrolide_resistance_level": "not_available",
        "ode_vaccine_escape_level": "not_available",
        "ode_pathogen_evidence_strength": "none",
        "ode_pathogen_source_ids": "not_available",
        "ode_pathogen_prior_source_type": "not_available",
        "covid_contact_policy_source": "not_available",
        "covid_contact_shock_resolution": "not_available",
        "ode_initial_condition_method": "not_available",
    }.items():
        if column not in state:
            state[column] = default
    annual = state.groupby(["iso3", "country", "year"], as_index=False).agg(
        true_infections_predicted=("true_infections", "sum"),
        symptomatic_infections_predicted=("symptomatic_infections", "sum"),
        reported_cases_predicted=("reported_cases_predicted", "sum"),
        reported_cases_unadjusted=("reported_cases_unadjusted", "sum"),
        reported_cases_before_detection_prior=("reported_cases_before_detection_prior", "sum"),
        reported_cases_before_year_adjustment=("reported_cases_before_year_adjustment", "sum"),
        reported_infant_under_6m_predicted=("reported_infant_under_6m_component", "sum"),
        reported_under_1y_predicted=("reported_under_1y_component", "sum"),
        contact_multiplier=("contact_multiplier", "mean"),
        monthly_stringency_index=("monthly_stringency_index", "mean"),
        annual_stringency_index=("annual_stringency_index", "mean"),
        covid_contact_policy_source=("covid_contact_policy_source", collapse_unique_text),
        covid_contact_shock_resolution=("covid_contact_shock_resolution", collapse_unique_text),
        seasonal_multiplier=("seasonal_multiplier", "mean"),
        importation_force=("importation_force", "mean"),
        ode_relative_detection_prior_mean=("ode_relative_detection_prior_mean", "mean"),
        ode_relative_detection_prior_lower=("ode_relative_detection_prior_lower", "mean"),
        ode_relative_detection_prior_upper=("ode_relative_detection_prior_upper", "mean"),
        ode_pathogen_pressure_prior_mean=("ode_pathogen_pressure_prior_mean", "mean"),
        ode_pathogen_pressure_prior_lower=("ode_pathogen_pressure_prior_lower", "mean"),
        ode_pathogen_pressure_prior_upper=("ode_pathogen_pressure_prior_upper", "mean"),
        ode_pathogen_transmission_multiplier=("ode_pathogen_transmission_multiplier", "mean"),
        ode_detection_effect_direction=("ode_detection_effect_direction", collapse_unique_text),
        ode_detection_evidence_strength=("ode_detection_evidence_strength", collapse_unique_text),
        ode_surveillance_regime=("ode_surveillance_regime", collapse_unique_text),
        ode_primary_diagnostic_methods=("ode_primary_diagnostic_methods", collapse_unique_text),
        ode_case_definition_or_reporting_change=("ode_case_definition_or_reporting_change", collapse_unique_text),
        ode_detection_source_ids=("ode_detection_source_ids", collapse_unique_text),
        ode_detection_prior_source_type=("ode_detection_prior_source_type", collapse_unique_text),
        ode_testing_denominator_years_available=("ode_testing_denominator_years_available", "mean"),
        ode_testing_denominator_first_year=("ode_testing_denominator_first_year", "mean"),
        ode_testing_denominator_last_year=("ode_testing_denominator_last_year", "mean"),
        ode_testing_denominator_support_level=("ode_testing_denominator_support_level", collapse_unique_text),
        ode_testing_denominator_source_ids=("ode_testing_denominator_source_ids", collapse_unique_text),
        ode_surveillance_outcome_source_type=("ode_surveillance_outcome_source_type", collapse_unique_text),
        ode_surveillance_outcome_support_level=("ode_surveillance_outcome_support_level", collapse_unique_text),
        ode_surveillance_outcome_notes=("ode_surveillance_outcome_notes", collapse_unique_text),
        ode_pathogen_regime=("ode_pathogen_regime", collapse_unique_text),
        ode_pathogen_features=("ode_pathogen_features", collapse_unique_text),
        ode_macrolide_resistance_level=("ode_macrolide_resistance_level", collapse_unique_text),
        ode_vaccine_escape_level=("ode_vaccine_escape_level", collapse_unique_text),
        ode_pathogen_evidence_strength=("ode_pathogen_evidence_strength", collapse_unique_text),
        ode_pathogen_source_ids=("ode_pathogen_source_ids", collapse_unique_text),
        ode_pathogen_prior_source_type=("ode_pathogen_prior_source_type", collapse_unique_text),
        ode_year_reporting_adjustment_raw=("ode_year_reporting_adjustment_raw", "mean"),
        ode_smoothed_year_reporting_adjustment=("ode_smoothed_year_reporting_adjustment", "mean"),
        ode_month_reporting_adjustment_raw=("ode_month_reporting_adjustment_raw", "mean"),
        ode_smoothed_month_reporting_adjustment=("ode_smoothed_month_reporting_adjustment", "mean"),
        ode_year_total_rescale_after_smoothing=("ode_year_total_rescale_after_smoothing", "mean"),
        ode_year_reporting_adjustment=("ode_year_reporting_adjustment", "mean"),
        ode_age_reporting_adjustment=("ode_age_reporting_adjustment", "mean"),
        ode_phase_beta_multiplier=("ode_phase_beta_multiplier", "mean"),
        ode_phase_reporting_multiplier=("ode_phase_reporting_multiplier", "mean"),
        ode_pandemic_phase_weight=("ode_pandemic_phase_weight", "mean"),
        ode_initial_condition_method=("ode_initial_condition_method", collapse_unique_text),
        ode_initial_observed_total_cases=("ode_initial_observed_total_cases", "mean"),
        ode_initial_observation_blend=("ode_initial_observation_blend", "mean"),
    )
    annual["year"] = annual["year"].astype(int)
    observed = context.annual.loc[
        :,
        [
            "year",
            "cases_total",
            "population",
            "incidence_per_100k",
            "periodic_expected_incidence_per_100k",
            "dtp3_coverage",
            "stringency_index",
        ],
    ].rename(
        columns={
            "cases_total": "observed_cases",
            "incidence_per_100k": "observed_incidence_per_100k",
        }
    )
    annual = annual.merge(observed, on="year", how="left")
    annual["predicted_incidence_per_100k"] = annual["reported_cases_predicted"] / annual["population"].replace({0: np.nan}) * 100000.0
    annual_pred_fields = [
        "year",
        "age_structure_reference_year",
        "age_structure_source_year",
        "under1_population",
        "under1_population_share",
        "boosting_debt",
        "reporting_rate",
        "post_reporting_multiplier",
        "post_transmission_multiplier",
    ]
    available = [column for column in annual_pred_fields if column in annual_pred.columns]
    if available:
        annual = annual.merge(annual_pred.loc[:, available].drop_duplicates("year"), on="year", how="left")
    annual["model_note"] = (
        "Annual aggregation of the fitted monthly age-stratified ODE; this is the primary Model 2 fit evidence. "
        "Transmission predictions include a fitted pathogen strain/AMR pressure prior, and reported-case predictions include an external diagnostic/reporting prior plus transparent age and smoothed residual observation calibration."
    )
    return annual.sort_values(["iso3", "year"]).reset_index(drop=True)


def monthly_ode_profile_columns() -> list[str]:
    return [
        "iso3",
        "country",
        "profile_parameter",
        "profile_parameter_value",
        "profile_factor_vs_fitted",
        "is_fitted_profile_value",
        "nuisance_parameter",
        "nuisance_parameter_value",
        "nuisance_factor_vs_fitted",
        "ode_beta_scale",
        "ode_reporting_scale",
        "weighted_nb_log_likelihood",
        "nb_log_likelihood",
        "monthly_all_age_weighted_nb_log_likelihood",
        "annual_age_weighted_nb_log_likelihood",
        "n_observations",
        "effective_n_observations",
        "profile_max_weighted_nb_log_likelihood",
        "fitted_profile_weighted_nb_log_likelihood",
        "delta_weighted_nb_log_likelihood_vs_profile_max",
        "delta_weighted_nb_log_likelihood_vs_fitted",
        "approx_chisq_2delta_vs_profile_max",
        "near_profile_support_95",
        "profile_identifiability_level",
        "nuisance_grid_evaluations",
        "profile_likelihood_method",
        "interpretation_boundary",
        "journal_use_recommendation",
    ]


def score_monthly_ode_profile_point(
    context: CountryContext,
    params: dict[str, float],
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    beta_scale: float,
    reporting_scale: float,
    ode_params: dict[str, float] | None = None,
    cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    profile_options = monthly_ode_default_options(ode_params)
    profile_options["ode_beta_scale"] = float(np.clip(beta_scale, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_beta_scale"]))
    profile_options["ode_reporting_scale"] = float(np.clip(reporting_scale, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_reporting_scale"]))
    cache = cache or _build_monthly_ode_cache(
        context,
        age_pred,
        inputs,
        include_testing_denominator=True,
        include_joint_observation=False,
    )
    ode_state = simulate_monthly_ode(
        context,
        params,
        age_pred,
        beta_scale=profile_options["ode_beta_scale"],
        reporting_scale=profile_options["ode_reporting_scale"],
        cache=cache,
        **{key: value for key, value in profile_options.items() if key not in {"ode_beta_scale", "ode_reporting_scale"}},
    )
    ode_state = apply_ode_observation_adjustments(
        context,
        ode_state,
        inputs,
        observed_month=cache.get("observed_month"),
        observed_age=cache.get("observed_age"),
    )
    likelihood = monthly_ode_likelihood_components(
        context,
        ode_state,
        inputs,
        observed_month=cache.get("observed_month"),
        observed_age=cache.get("observed_age"),
    )
    if likelihood.empty:
        return {
            "weighted_nb_log_likelihood": np.nan,
            "nb_log_likelihood": np.nan,
            "monthly_all_age_weighted_nb_log_likelihood": np.nan,
            "annual_age_weighted_nb_log_likelihood": np.nan,
            "n_observations": 0.0,
            "effective_n_observations": 0.0,
        }
    component_weighted = likelihood.groupby("component")["weighted_nb_log_likelihood"].sum()
    return {
        "weighted_nb_log_likelihood": float(likelihood["weighted_nb_log_likelihood"].sum()),
        "nb_log_likelihood": float(likelihood["nb_log_likelihood"].sum()),
        "monthly_all_age_weighted_nb_log_likelihood": float(component_weighted.get("monthly_ode_all_age", np.nan)),
        "annual_age_weighted_nb_log_likelihood": float(component_weighted.get("monthly_ode_annual_age_group", np.nan)),
        "n_observations": float(len(likelihood)),
        "effective_n_observations": float(pd.to_numeric(likelihood["age_allocation_weight"], errors="coerce").fillna(1.0).sum()),
    }


def model2_monthly_ode_profile_likelihood(
    context: CountryContext,
    params: dict[str, float],
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    ode_params: dict[str, float],
) -> pd.DataFrame:
    columns = monthly_ode_profile_columns()
    fitted_beta = float(ode_params.get("ode_beta_scale", np.nan))
    fitted_reporting = float(ode_params.get("ode_reporting_scale", np.nan))
    if (
        age_pred.empty
        or not np.isfinite(fitted_beta)
        or not np.isfinite(fitted_reporting)
        or fitted_beta <= 0
        or fitted_reporting <= 0
    ):
        return pd.DataFrame(columns=columns)

    profile_inputs = _build_monthly_ode_cache(
        context,
        age_pred,
        inputs,
        include_testing_denominator=True,
        include_joint_observation=False,
    )
    profile_cache: dict[tuple[float, float], dict[str, float]] = {}

    def evaluate(beta_scale: float, reporting_scale: float) -> dict[str, float]:
        beta_scale = float(np.round(np.clip(beta_scale, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_beta_scale"]), 6))
        reporting_scale = float(np.round(np.clip(reporting_scale, *MONTHLY_ODE_PARAMETER_BOUNDS["ode_reporting_scale"]), 6))
        key = (beta_scale, reporting_scale)
        if key not in profile_cache:
            profile_cache[key] = score_monthly_ode_profile_point(
                context,
                params,
                age_pred,
                inputs,
                beta_scale,
                reporting_scale,
                ode_params,
                cache=profile_inputs,
            )
        return {**profile_cache[key], "ode_beta_scale": beta_scale, "ode_reporting_scale": reporting_scale}

    rows: list[dict[str, Any]] = []
    profile_specs = [
        ("ode_beta_scale", "ode_reporting_scale", fitted_beta, fitted_reporting),
        ("ode_reporting_scale", "ode_beta_scale", fitted_reporting, fitted_beta),
    ]
    for profile_parameter, nuisance_parameter, fitted_profile_value, fitted_nuisance_value in profile_specs:
        for profile_factor in MONTHLY_ODE_PROFILE_FIXED_FACTORS:
            profile_lower, profile_upper = MONTHLY_ODE_PARAMETER_BOUNDS[profile_parameter]
            profile_value = float(np.round(np.clip(fitted_profile_value * profile_factor, profile_lower, profile_upper), 6))
            best_row: dict[str, Any] | None = None
            finite_evaluations = 0
            for nuisance_factor in MONTHLY_ODE_PROFILE_NUISANCE_FACTORS:
                nuisance_lower, nuisance_upper = MONTHLY_ODE_PARAMETER_BOUNDS[nuisance_parameter]
                nuisance_value = float(np.round(np.clip(fitted_nuisance_value * nuisance_factor, nuisance_lower, nuisance_upper), 6))
                beta_scale = profile_value if profile_parameter == "ode_beta_scale" else nuisance_value
                reporting_scale = profile_value if profile_parameter == "ode_reporting_scale" else nuisance_value
                metrics = evaluate(beta_scale, reporting_scale)
                if np.isfinite(metrics["weighted_nb_log_likelihood"]):
                    finite_evaluations += 1
                row = {
                    "iso3": context.iso3,
                    "country": context.country,
                    "profile_parameter": profile_parameter,
                    "profile_parameter_value": profile_value,
                    "profile_factor_vs_fitted": float(profile_value / fitted_profile_value),
                    "is_fitted_profile_value": bool(np.isclose(profile_factor, 1.0, rtol=1e-9, atol=1e-12)),
                    "nuisance_parameter": nuisance_parameter,
                    "nuisance_parameter_value": nuisance_value,
                    "nuisance_factor_vs_fitted": float(nuisance_value / fitted_nuisance_value),
                    "ode_beta_scale": metrics["ode_beta_scale"],
                    "ode_reporting_scale": metrics["ode_reporting_scale"],
                    "weighted_nb_log_likelihood": metrics["weighted_nb_log_likelihood"],
                    "nb_log_likelihood": metrics["nb_log_likelihood"],
                    "monthly_all_age_weighted_nb_log_likelihood": metrics["monthly_all_age_weighted_nb_log_likelihood"],
                    "annual_age_weighted_nb_log_likelihood": metrics["annual_age_weighted_nb_log_likelihood"],
                    "n_observations": int(metrics["n_observations"]),
                    "effective_n_observations": metrics["effective_n_observations"],
                    "nuisance_grid_evaluations": 0,
                    "profile_likelihood_method": "Approximate one-dimensional profile likelihood over fitted monthly ODE beta/reporting scales; the profiled parameter is fixed and the other scale is re-optimized on a bounded nuisance grid.",
                    "interpretation_boundary": "Profile is a finite-grid NB likelihood diagnostic around the fitted monthly ODE scales, not a full Bayesian/TMB country-age-month posterior.",
                    "journal_use_recommendation": "Use to qualify transmission-vs-reporting identifiability in the main ODE fit and to prioritize a wider final joint ODE likelihood.",
                }
                if best_row is None or float(row["weighted_nb_log_likelihood"]) > float(best_row["weighted_nb_log_likelihood"]):
                    best_row = row
            if best_row is not None:
                best_row["nuisance_grid_evaluations"] = int(finite_evaluations)
                rows.append(best_row)

    profile = pd.DataFrame(rows, columns=[column for column in columns if column not in {
        "profile_max_weighted_nb_log_likelihood",
        "fitted_profile_weighted_nb_log_likelihood",
        "delta_weighted_nb_log_likelihood_vs_profile_max",
        "delta_weighted_nb_log_likelihood_vs_fitted",
        "approx_chisq_2delta_vs_profile_max",
        "near_profile_support_95",
        "profile_identifiability_level",
    }])
    if profile.empty:
        return pd.DataFrame(columns=columns)

    enriched: list[pd.DataFrame] = []
    for profile_parameter, group in profile.groupby("profile_parameter", sort=False):
        group = group.copy()
        profile_max = float(group["weighted_nb_log_likelihood"].max())
        fitted_rows = group.loc[group["is_fitted_profile_value"].astype(bool)]
        fitted_ll = float(fitted_rows["weighted_nb_log_likelihood"].iloc[0]) if not fitted_rows.empty else profile_max
        group["profile_max_weighted_nb_log_likelihood"] = profile_max
        group["fitted_profile_weighted_nb_log_likelihood"] = fitted_ll
        group["delta_weighted_nb_log_likelihood_vs_profile_max"] = group["weighted_nb_log_likelihood"] - profile_max
        group["delta_weighted_nb_log_likelihood_vs_fitted"] = group["weighted_nb_log_likelihood"] - fitted_ll
        group["approx_chisq_2delta_vs_profile_max"] = 2.0 * (profile_max - group["weighted_nb_log_likelihood"])
        group["near_profile_support_95"] = group["approx_chisq_2delta_vs_profile_max"].le(MONTHLY_ODE_PROFILE_SUPPORT_95_1DF)
        support = group.loc[group["near_profile_support_95"].astype(bool), "profile_factor_vs_fitted"]
        evaluated_min = float(group["profile_factor_vs_fitted"].min())
        evaluated_max = float(group["profile_factor_vs_fitted"].max())
        support_hits_lower = bool(not support.empty and support.min() <= evaluated_min + 1e-9)
        support_hits_upper = bool(not support.empty and support.max() >= evaluated_max - 1e-9)
        if support_hits_lower and support_hits_upper:
            level = "weakly_identified_profile_open_both_sides"
            recommendation = "Treat beta/reporting separation as weak within the evaluated finite grid; avoid strong transmission-vs-reporting wording."
        elif support_hits_lower or support_hits_upper:
            level = "partly_identified_profile_open_one_side"
            recommendation = "Use only qualified transmission-vs-reporting wording and prioritize wider final profile/Bayesian fitting."
        else:
            level = "locally_identified_profile_closed_within_grid"
            recommendation = "Supports local numerical separation of the profiled scale under the pilot ODE, with usual model caveats."
        group["profile_identifiability_level"] = level
        group["journal_use_recommendation"] = recommendation
        enriched.append(group)

    return (
        pd.concat(enriched, ignore_index=True, sort=False)
        .loc[:, columns]
        .sort_values(["iso3", "profile_parameter", "profile_parameter_value"])
        .reset_index(drop=True)
    )
