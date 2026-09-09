from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.data.model2_observations import allocated_age_observations
from pertussis_global_dynamics.model2_constants import AGE_LABELS, AGE_ORDER, AGE_SYMPTOM_PROB, AGE_VACCINE_PROGRAM_REACH, AGE_WANING_YEARS, NB_DISPERSION
from pertussis_global_dynamics.model2c_windows import (
    MODEL2C_OBSERVATION_END_YEAR,
    MODEL2C_OBSERVATION_START_YEAR,
)
from pertussis_global_dynamics.simulation.likelihood import nb_log_likelihood


def month_end(year: int, month: int) -> str:
    return pd.Timestamp(year=year, month=month, day=1).to_period("M").end_time.strftime("%Y-%m-%d")


def initialize_monthly_state(age_row: pd.Series | Mapping[str, Any], context: CountryContext) -> dict[str, float]:
    age_group = str(age_row.get("age_group", getattr(age_row, "name", "unknown")))
    population = max(float(age_row["population"]), 1.0)
    dtp3 = float(age_row.get("dtp3_coverage", context.pre_mean_dtp3) or context.pre_mean_dtp3)
    program_reach = AGE_VACCINE_PROGRAM_REACH[age_group]
    susceptibility = float(np.clip(age_row.get("susceptibility", 0.5), 0.03, 0.98))
    maternal_coverage = float(age_row.get("maternal_coverage_assumed", 0.0) or 0.0)
    maternal_share = 0.0
    if age_group == "infant_0_2m":
        maternal_share = min(maternal_coverage * 0.70, 0.70)
    elif age_group == "infant_3_11m":
        maternal_share = min(maternal_coverage * 0.28, 0.35)
    maternal_protected = population * maternal_share
    vaccine_recent = population * min(dtp3 * program_reach * 0.18, 0.35)
    vaccine_waned = population * min(dtp3 * program_reach * 0.22, 0.40)
    infection_recent = population * min(max(1.0 - susceptibility - maternal_share, 0.0) * 0.10, 0.12)
    infection_waned = population * min(max(1.0 - susceptibility - maternal_share, 0.0) * 0.16, 0.18)
    recovered = population * 0.02
    susceptible = population - maternal_protected - vaccine_recent - vaccine_waned - infection_recent - infection_waned - recovered
    return {
        "maternal_protected": max(maternal_protected, 0.0),
        "susceptible_naive": max(susceptible, population * 0.01),
        "vaccine_recent": max(vaccine_recent, 0.0),
        "vaccine_waned": max(vaccine_waned, 0.0),
        "infection_recent": max(infection_recent, 0.0),
        "infection_waned": max(infection_waned, 0.0),
        "exposed": 0.0,
        "infectious_symptomatic": 0.0,
        "infectious_asymptomatic": 0.0,
        "recovered": max(recovered, 0.0),
    }


def rescale_state_to_population(state: dict[str, float], population: float) -> dict[str, float]:
    compartments = [
        "maternal_protected",
        "susceptible_naive",
        "vaccine_recent",
        "vaccine_waned",
        "infection_recent",
        "infection_waned",
        "exposed",
        "infectious_symptomatic",
        "infectious_asymptomatic",
        "recovered",
    ]
    for key in compartments:
        state[key] = max(float(state.get(key, 0.0)), 0.0)
    total = sum(state[key] for key in compartments)
    if total <= 0:
        state["susceptible_naive"] = max(population, 0.0)
        return state
    scale = max(population, 0.0) / total
    for key in compartments:
        state[key] *= scale
    return state


def simulate_monthly_age_state(context: CountryContext, params: dict[str, float], age_pred: pd.DataFrame) -> pd.DataFrame:
    if age_pred.empty:
        return pd.DataFrame()
    seasonal = monthly_seasonal_fractions(context).set_index("month")["seasonal_fraction"].to_dict()
    monthly_rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, float]] = {}
    annual_age = age_pred.loc[
        age_pred["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
    ].copy()
    annual_age["year"] = pd.to_numeric(annual_age["year"], errors="coerce").astype(int)
    annual_age = annual_age.sort_values(["year", "age_group"])
    waning_rate = float(params["waning_rate"])

    for year in sorted(annual_age["year"].unique()):
        year_age = annual_age.loc[annual_age["year"].eq(year)].set_index("age_group")
        for month in range(1, 13):
            seasonal_fraction = float(seasonal.get(month, 1.0 / 12.0))
            for age_group in AGE_ORDER:
                if age_group not in year_age.index:
                    continue
                age_row = year_age.loc[age_group]
                if isinstance(age_row, pd.DataFrame):
                    age_row = age_row.iloc[0]
                population = max(float(age_row["population"]), 1.0)
                state = states.get(age_group)
                if state is None:
                    state = initialize_monthly_state(age_row, context)
                state = rescale_state_to_population(state, population)

                maternal_wane = state["maternal_protected"] * (0.45 if age_group == "infant_0_2m" else 0.18 if age_group == "infant_3_11m" else 0.02)
                vaccine_wane = state["vaccine_recent"] * min(waning_rate / 12.0, 0.08)
                infection_wane = state["infection_recent"] * 0.018
                recovered_wane = state["recovered"] * 0.010
                old_immunity_loss = state["infection_waned"] * 0.012
                state["maternal_protected"] -= maternal_wane
                state["susceptible_naive"] += maternal_wane + recovered_wane + old_immunity_loss
                state["vaccine_recent"] -= vaccine_wane
                state["vaccine_waned"] += vaccine_wane
                state["infection_recent"] -= infection_wane
                state["infection_waned"] += infection_wane - old_immunity_loss
                state["recovered"] -= recovered_wane

                true_infections = max(float(age_row["true_infections"]) * seasonal_fraction, 0.0)
                symptomatic = max(float(age_row["symptomatic_infections"]) * seasonal_fraction, 0.0)
                reported = max(float(age_row["reported_cases_predicted"]) * seasonal_fraction, 0.0)
                asymptomatic = max(true_infections - symptomatic, 0.0)
                infectable = max(state["susceptible_naive"] + 0.60 * state["vaccine_waned"] + 0.80 * state["infection_waned"], 1e-9)
                force_of_infection = min(true_infections / infectable, 0.95)
                from_s = min(state["susceptible_naive"], true_infections * 0.78)
                remaining = max(true_infections - from_s, 0.0)
                from_vw = min(state["vaccine_waned"], remaining * 0.45)
                remaining = max(remaining - from_vw, 0.0)
                from_iw = min(state["infection_waned"], remaining)
                state["susceptible_naive"] -= from_s
                state["vaccine_waned"] -= from_vw
                state["infection_waned"] -= from_iw

                exposure_state = true_infections * 0.25
                symptomatic_state = symptomatic * 0.30
                asymptomatic_state = asymptomatic * 0.30
                infection_recent_increment = max(true_infections - exposure_state - symptomatic_state - asymptomatic_state, 0.0)
                state["exposed"] = exposure_state
                state["infectious_symptomatic"] = symptomatic_state
                state["infectious_asymptomatic"] = asymptomatic_state
                state["infection_recent"] += infection_recent_increment

                exposure_pressure = true_infections / population
                boost_pool = min((state["vaccine_waned"] + state["infection_waned"]) * exposure_pressure * 0.18, state["vaccine_waned"] + state["infection_waned"])
                boost_vaccine = min(state["vaccine_waned"], boost_pool * 0.60)
                boost_infection = min(state["infection_waned"], boost_pool - boost_vaccine)
                state["vaccine_waned"] -= boost_vaccine
                state["vaccine_recent"] += boost_vaccine
                state["infection_waned"] -= boost_infection
                state["infection_recent"] += boost_infection
                state = rescale_state_to_population(state, population)
                states[age_group] = state

                monthly_rows.append(
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
                        "seasonal_fraction": seasonal_fraction,
                        "force_of_infection": force_of_infection,
                        "true_infections": true_infections,
                        "symptomatic_infections": symptomatic,
                        "reported_cases_predicted": reported,
                        "maternal_protected": state["maternal_protected"],
                        "susceptible_naive": state["susceptible_naive"],
                        "vaccine_recent": state["vaccine_recent"],
                        "vaccine_waned": state["vaccine_waned"],
                        "infection_recent": state["infection_recent"],
                        "infection_waned": state["infection_waned"],
                        "exposed": state["exposed"],
                        "infectious_symptomatic": state["infectious_symptomatic"],
                        "infectious_asymptomatic": state["infectious_asymptomatic"],
                        "recovered": state["recovered"],
                        "state_total": sum(state.values()),
                        "model_note": "Monthly age-state bridge scaled to annual Model 2 age-specific predictions; not yet a fitted continuous-time ODE.",
                    }
                )
    return pd.DataFrame(monthly_rows)


def monthly_seasonal_fractions(context: CountryContext) -> pd.DataFrame:
    intervals = context.time_resolved.copy()
    if intervals.empty:
        return pd.DataFrame({"month": range(1, 13), "seasonal_fraction": np.repeat(1.0 / 12.0, 12)})
    intervals["date_start"] = pd.to_datetime(intervals["date_start"], errors="coerce")
    intervals["year"] = pd.to_numeric(intervals["year"], errors="coerce").astype("Int64")
    intervals["month"] = pd.to_numeric(intervals["month"], errors="coerce")
    intervals["month"] = intervals["month"].fillna(intervals["date_start"].dt.month)
    intervals["cases_total"] = pd.to_numeric(intervals["cases_total"], errors="coerce")
    seasonal_base = intervals.loc[intervals["year"].between(2015, 2019)].dropna(subset=["month", "cases_total"]).copy()
    if seasonal_base.empty:
        seasonal_base = intervals.dropna(subset=["month", "cases_total"]).copy()
    if seasonal_base.empty:
        return pd.DataFrame({"month": range(1, 13), "seasonal_fraction": np.repeat(1.0 / 12.0, 12)})
    seasonal = seasonal_base.groupby("month", as_index=False).agg(month_cases=("cases_total", "sum"))
    all_months = pd.DataFrame({"month": range(1, 13)})
    seasonal = all_months.merge(seasonal, on="month", how="left").fillna({"month_cases": 0.0})
    total = seasonal["month_cases"].sum()
    if total <= 0:
        seasonal["seasonal_fraction"] = 1.0 / 12.0
    else:
        seasonal["seasonal_fraction"] = seasonal["month_cases"] / total
        seasonal["seasonal_fraction"] = seasonal["seasonal_fraction"].replace(0.0, 0.002)
        seasonal["seasonal_fraction"] = seasonal["seasonal_fraction"] / seasonal["seasonal_fraction"].sum()
    return seasonal.loc[:, ["month", "seasonal_fraction"]]


def monthly_state_likelihood_components(context: CountryContext, monthly_state: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if monthly_state.empty:
        return pd.DataFrame()
    predicted_month = monthly_state.groupby(["iso3", "country", "year", "month"], as_index=False).agg(
        predicted_cases=("reported_cases_predicted", "sum")
    )
    observed = context.time_resolved.copy()
    if not observed.empty:
        observed["date_start"] = pd.to_datetime(observed["date_start"], errors="coerce")
        observed["year"] = pd.to_numeric(observed["year"], errors="coerce").astype("Int64")
        observed["month"] = pd.to_numeric(observed["month"], errors="coerce").fillna(observed["date_start"].dt.month)
        observed["cases_total"] = pd.to_numeric(observed["cases_total"], errors="coerce")
        observed_month = observed.dropna(subset=["year", "month", "cases_total"]).groupby(["iso3", "country", "year", "month"], as_index=False).agg(
            observed_cases=("cases_total", "sum")
        )
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
                    "component": "monthly_state_all_age",
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
                    "likelihood_note": "Monthly all-age observations compared with monthly age-state bridge predictions.",
                }
            )

    observed_age = allocated_age_observations(inputs, context.iso3)
    if not observed_age.empty:
        predicted_age = monthly_state.groupby(["iso3", "country", "year", "age_group"], as_index=False).agg(
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
                    "component": "monthly_state_annual_age_group",
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
                    "likelihood_note": "Annual age-group observations compared with the annual sum of monthly age-state bridge predictions; broad source-bin allocations are down-weighted in weighted diagnostics.",
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
