from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.data.model2_observations import allocated_age_observations, monthly_observed_cases
from pertussis_global_dynamics.model2_constants import AGE_BOUNDS, AGE_ORDER, NB_DISPERSION
from pertussis_global_dynamics.model2c_windows import (
    MODEL2C_OBSERVATION_END_YEAR,
    MODEL2C_OBSERVATION_START_YEAR,
)
from pertussis_global_dynamics.simulation.likelihood import nb_log_likelihood

def source_age_bin_likelihood_components(
    context: CountryContext,
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "component",
        "year",
        "source_age_group",
        "source_age_lower",
        "source_age_upper",
        "source_age_width",
        "observed_cases",
        "predicted_cases",
        "log_residual",
        "nb_log_likelihood",
        "age_allocation_weight",
        "source_age_projection_weight",
        "source_bin_covered_fraction",
        "weighted_nb_log_likelihood",
        "weighted_log_residual_square",
        "model_age_groups_contributing",
        "n_model_age_groups_contributing",
        "nb_dispersion",
        "likelihood_note",
    ]
    source = inputs.get("age_source_cases", pd.DataFrame()).copy()
    if source.empty or age_pred.empty:
        return pd.DataFrame(columns=columns)
    source = source.loc[source["iso3"].astype(str).eq(context.iso3)].copy()
    if source.empty:
        return pd.DataFrame(columns=columns)
    source["year"] = pd.to_numeric(source["year"], errors="coerce").astype("Int64")
    source["cases_total"] = pd.to_numeric(source["cases_total"], errors="coerce")
    source["age_lower"] = pd.to_numeric(source["age_lower"], errors="coerce").fillna(0.0)
    source["age_upper"] = pd.to_numeric(source["age_upper"], errors="coerce").fillna(100.0)
    source = source.loc[
        source["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        & source["cases_total"].notna()
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=columns)

    predicted = age_pred.loc[
        age_pred["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
    ].copy()
    predicted = predicted.loc[predicted["age_group"].isin(AGE_ORDER)].copy()
    predicted["year"] = pd.to_numeric(predicted["year"], errors="coerce").astype("Int64")
    predicted["reported_cases_predicted"] = pd.to_numeric(predicted["reported_cases_predicted"], errors="coerce").fillna(0.0)
    predicted_by_year = {
        int(year): group.set_index("age_group")["reported_cases_predicted"].to_dict()
        for year, group in predicted.groupby("year")
    }
    rows: list[dict[str, Any]] = []
    for item in source.itertuples(index=False):
        year = int(item.year)
        year_pred = predicted_by_year.get(year, {})
        if not year_pred:
            continue
        source_lower = float(item.age_lower)
        source_upper = float(item.age_upper)
        if not np.isfinite(source_lower):
            source_lower = 0.0
        if not np.isfinite(source_upper) or source_upper <= source_lower:
            source_upper = 100.0
        source_lower = float(np.clip(source_lower, 0.0, 100.0))
        source_upper = float(np.clip(source_upper, source_lower + 1e-6, 100.0))
        source_width = max(source_upper - source_lower, 1e-6)
        predicted_cases = 0.0
        overlap_total = 0.0
        projection_weight = 0.0
        contributing: list[str] = []
        for age_group in AGE_ORDER:
            model_lower, model_upper = AGE_BOUNDS[age_group]
            overlap = max(0.0, min(source_upper, model_upper) - max(source_lower, model_lower))
            if overlap <= 0:
                continue
            model_width = max(model_upper - model_lower, 1e-6)
            source_fraction = overlap / source_width
            model_fraction = overlap / model_width
            predicted_cases += float(year_pred.get(age_group, 0.0)) * model_fraction
            overlap_total += overlap
            projection_weight += source_fraction * model_fraction
            contributing.append(age_group)
        if not contributing:
            continue
        observed_cases = float(item.cases_total)
        predicted_cases = max(predicted_cases, 0.0)
        source_bin_covered_fraction = float(np.clip(overlap_total / source_width, 0.0, 1.0))
        projection_weight = float(np.clip(projection_weight * source_bin_covered_fraction, 0.05, 1.0))
        log_residual = np.log1p(predicted_cases) - np.log1p(max(observed_cases, 0.0))
        nb_ll = nb_log_likelihood(observed_cases, predicted_cases)
        rows.append(
            {
                "iso3": context.iso3,
                "country": context.country,
                "component": "source_age_bin_original_scale",
                "year": year,
                "source_age_group": str(item.age_group),
                "source_age_lower": source_lower,
                "source_age_upper": source_upper,
                "source_age_width": source_width,
                "observed_cases": observed_cases,
                "predicted_cases": predicted_cases,
                "log_residual": log_residual,
                "nb_log_likelihood": nb_ll,
                "age_allocation_weight": projection_weight,
                "source_age_projection_weight": projection_weight,
                "source_bin_covered_fraction": source_bin_covered_fraction,
                "weighted_nb_log_likelihood": nb_ll * projection_weight,
                "weighted_log_residual_square": log_residual**2 * projection_weight,
                "model_age_groups_contributing": ";".join(contributing),
                "n_model_age_groups_contributing": int(len(contributing)),
                "nb_dispersion": NB_DISPERSION,
                "likelihood_note": "Original source-age-bin likelihood: model age-group predictions are projected back to each country-specific source age bin by age-overlap fractions. This avoids treating allocated narrow model-age observations as direct source counts and down-weights bins that cut through broad model age groups.",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["iso3", "year", "source_age_lower", "source_age_upper"]).reset_index(drop=True)


def observation_likelihood_components(
    context: CountryContext,
    interval_pred: pd.DataFrame,
    age_pred: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not interval_pred.empty:
        for item in interval_pred.itertuples(index=False):
            observed = float(item.observed_cases)
            predicted = float(item.predicted_cases)
            if not np.isfinite(observed) or not np.isfinite(predicted):
                continue
            log_residual = np.log1p(max(predicted, 0.0)) - np.log1p(max(observed, 0.0))
            nb_ll = nb_log_likelihood(observed, predicted)
            rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "component": "time_resolved_all_age",
                    "year": int(item.year),
                    "date_start": item.date_start,
                    "date_end": item.date_end,
                    "time_resolution": item.time_resolution,
                    "interval_index": item.interval_index,
                    "model_age_group": "all_ages",
                    "observed_cases": observed,
                    "predicted_cases": predicted,
                    "log_residual": log_residual,
                    "nb_log_likelihood": nb_ll,
                    "age_allocation_weight": 1.0,
                    "weighted_nb_log_likelihood": nb_ll,
                    "weighted_log_residual_square": log_residual**2,
                    "age_allocation_note": "All-age time-resolved observation; no age-bin allocation down-weighting.",
                    "nb_dispersion": NB_DISPERSION,
                    "likelihood_note": "All-age weekly/monthly PertussisIncidence observation with annual model predictions distributed by observed pre-pandemic seasonal fractions.",
                }
            )

    observed_age = allocated_age_observations(inputs, context.iso3)
    if not observed_age.empty:
        predicted_age = age_pred.loc[
            age_pred["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
        ].groupby(
            ["iso3", "country", "year", "age_group"],
            as_index=False,
        ).agg(predicted_cases=("reported_cases_predicted", "sum"))
        predicted_age = predicted_age.rename(columns={"age_group": "model_age_group"})
        merged = observed_age.merge(
            predicted_age.loc[:, ["iso3", "year", "model_age_group", "predicted_cases"]],
            on=["iso3", "year", "model_age_group"],
            how="left",
        )
        for item in merged.dropna(subset=["observed_cases", "predicted_cases"]).itertuples(index=False):
            observed = float(item.observed_cases)
            predicted = float(item.predicted_cases)
            if not np.isfinite(observed) or not np.isfinite(predicted):
                continue
            log_residual = np.log1p(max(predicted, 0.0)) - np.log1p(max(observed, 0.0))
            nb_ll = nb_log_likelihood(observed, predicted)
            allocation_weight = float(item.age_allocation_weight)
            rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "component": "annual_age_group",
                    "year": int(item.year),
                    "date_start": f"{int(item.year)}-01-01",
                    "date_end": f"{int(item.year)}-12-31",
                    "time_resolution": "annual_age",
                    "interval_index": np.nan,
                    "model_age_group": item.model_age_group,
                    "observed_cases": observed,
                    "predicted_cases": predicted,
                    "log_residual": log_residual,
                    "nb_log_likelihood": nb_ll,
                    "age_allocation_weight": allocation_weight,
                    "weighted_nb_log_likelihood": nb_ll * allocation_weight,
                    "weighted_log_residual_square": log_residual**2 * allocation_weight,
                    "age_allocation_note": item.age_allocation_note,
                    "nb_dispersion": NB_DISPERSION,
                    "likelihood_note": "Annual model-age-group PertussisIncidence observation; source age bins are allocated to model groups by age-bin overlap and broad-bin allocations are down-weighted in weighted diagnostics.",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "iso3",
            "country",
            "component",
            "year",
            "date_start",
            "date_end",
            "time_resolution",
            "interval_index",
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


def joint_age_month_composite_observations(
    context: CountryContext,
    inputs: dict[str, pd.DataFrame],
    observed_month: pd.DataFrame | None = None,
    observed_age: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "month",
        "model_age_group",
        "observed_month_all_age_cases",
        "observed_annual_age_cases",
        "observed_annual_age_fraction",
        "observed_cases",
        "age_allocation_weight",
        "source_age_group_count",
        "max_source_split_count",
        "min_allocation_fraction",
        "age_allocation_note",
        "annual_age_total_cases",
        "monthly_all_age_total_cases",
        "annual_age_to_monthly_all_age_ratio",
        "composite_observation_note",
    ]
    if observed_month is None:
        observed_month = monthly_observed_cases(context)
    else:
        observed_month = observed_month.copy()
    if observed_age is None:
        observed_age = allocated_age_observations(inputs, context.iso3)
    else:
        observed_age = observed_age.copy()
    if observed_month.empty or observed_age.empty:
        return pd.DataFrame(columns=columns)

    observed_month = observed_month.loc[
        observed_month["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
    ].copy()
    observed_age = observed_age.loc[
        observed_age["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
    ].copy()
    if observed_month.empty or observed_age.empty:
        return pd.DataFrame(columns=columns)

    month_totals = observed_month.groupby(["iso3", "country", "year"], as_index=False).agg(
        monthly_all_age_total_cases=("observed_cases", "sum")
    )
    age_totals = observed_age.groupby(["iso3", "country", "year"], as_index=False).agg(
        annual_age_total_cases=("observed_cases", "sum")
    )
    age = observed_age.merge(age_totals, on=["iso3", "country", "year"], how="left")
    age = age.loc[age["annual_age_total_cases"].gt(0)].copy()
    if age.empty:
        return pd.DataFrame(columns=columns)
    age["observed_annual_age_fraction"] = age["observed_cases"] / age["annual_age_total_cases"]
    age = age.rename(columns={"observed_cases": "observed_annual_age_cases"})

    merged = observed_month.merge(month_totals, on=["iso3", "country", "year"], how="left").merge(
        age,
        on=["iso3", "country", "year"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged = merged.rename(columns={"observed_cases": "observed_month_all_age_cases"})
    merged["observed_cases"] = merged["observed_month_all_age_cases"] * merged["observed_annual_age_fraction"]
    merged["annual_age_to_monthly_all_age_ratio"] = (
        merged["annual_age_total_cases"] / merged["monthly_all_age_total_cases"].replace({0: np.nan})
    )
    merged["composite_observation_note"] = (
        "Composite country-age-month target: monthly all-age observed cases distribute timing, annual allocated age-group observations distribute age structure. It is not a direct source age-month surveillance count."
    )
    return merged.loc[:, columns].sort_values(["iso3", "year", "month", "model_age_group"]).reset_index(drop=True)


def joint_age_month_ode_likelihood_components(
    context: CountryContext,
    ode_state: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    observed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "component",
        "year",
        "month",
        "model_age_group",
        "observed_month_all_age_cases",
        "observed_annual_age_cases",
        "observed_annual_age_fraction",
        "observed_cases",
        "predicted_cases",
        "log_residual",
        "nb_log_likelihood",
        "age_allocation_weight",
        "weighted_nb_log_likelihood",
        "weighted_log_residual_square",
        "source_age_group_count",
        "max_source_split_count",
        "min_allocation_fraction",
        "annual_age_total_cases",
        "monthly_all_age_total_cases",
        "annual_age_to_monthly_all_age_ratio",
        "age_allocation_note",
        "nb_dispersion",
        "likelihood_note",
    ]
    if ode_state.empty:
        return pd.DataFrame(columns=columns)
    if observed is None:
        observed = joint_age_month_composite_observations(context, inputs)
    if observed.empty:
        return pd.DataFrame(columns=columns)

    predicted = ode_state.groupby(["iso3", "country", "year", "month", "age_group"], as_index=False).agg(
        predicted_cases=("reported_cases_predicted", "sum")
    )
    predicted = predicted.rename(columns={"age_group": "model_age_group"})
    merged = observed.merge(predicted, on=["iso3", "country", "year", "month", "model_age_group"], how="inner")
    rows: list[dict[str, Any]] = []
    for item in merged.itertuples(index=False):
        observed_cases = float(item.observed_cases)
        predicted_cases = float(item.predicted_cases)
        if not np.isfinite(observed_cases) or not np.isfinite(predicted_cases):
            continue
        log_residual = np.log1p(max(predicted_cases, 0.0)) - np.log1p(max(observed_cases, 0.0))
        nb_ll = nb_log_likelihood(observed_cases, predicted_cases)
        allocation_weight = float(np.clip(item.age_allocation_weight, 0.05, 1.0))
        rows.append(
            {
                "iso3": context.iso3,
                "country": context.country,
                "component": "joint_age_month_composite",
                "year": int(item.year),
                "month": int(item.month),
                "model_age_group": item.model_age_group,
                "observed_month_all_age_cases": float(item.observed_month_all_age_cases),
                "observed_annual_age_cases": float(item.observed_annual_age_cases),
                "observed_annual_age_fraction": float(item.observed_annual_age_fraction),
                "observed_cases": observed_cases,
                "predicted_cases": predicted_cases,
                "log_residual": log_residual,
                "nb_log_likelihood": nb_ll,
                "age_allocation_weight": allocation_weight,
                "weighted_nb_log_likelihood": nb_ll * allocation_weight,
                "weighted_log_residual_square": log_residual**2 * allocation_weight,
                "source_age_group_count": int(item.source_age_group_count),
                "max_source_split_count": int(item.max_source_split_count),
                "min_allocation_fraction": float(item.min_allocation_fraction),
                "annual_age_total_cases": float(item.annual_age_total_cases),
                "monthly_all_age_total_cases": float(item.monthly_all_age_total_cases),
                "annual_age_to_monthly_all_age_ratio": float(item.annual_age_to_monthly_all_age_ratio),
                "age_allocation_note": item.age_allocation_note,
                "nb_dispersion": NB_DISPERSION,
                "likelihood_note": item.composite_observation_note
                + " Compared with fitted continuous-time monthly age-stratified ODE reported-case predictions using a weighted negative-binomial composite likelihood.",
            }
        )
    return pd.DataFrame(rows, columns=columns)
