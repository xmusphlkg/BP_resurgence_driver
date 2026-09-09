from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.bayes.model2b_attribution import MODEL2B_OUTPUT_VERSION
from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.data.model2_observations import (
    allocated_age_observations,
    monthly_detection_prior_table,
    monthly_observed_cases,
    monthly_pathogen_prior_table,
)
from pertussis_global_dynamics.model2_constants import (
    AGE_BOUNDS,
    AGE_ORDER,
    AGE_REL_CONTACT,
    AGE_REPORT_WEIGHT,
    AGE_SYMPTOM_PROB,
    AGE_VACCINE_PROGRAM_REACH,
    AGE_WANING_YEARS,
    NB_DISPERSION,
    PARAMETER_BOUNDS,
)
from pertussis_global_dynamics.paths import OUTPUTS
from pertussis_global_dynamics.simulation.monthly_state_bridge import monthly_seasonal_fractions

MODEL2B_START_YEAR = 2015
MODEL2B_END_YEAR = 2026
MODEL2B_FIT_END_YEAR = 2025
SILENT_EXPOSURE_FLOOR_MIN_MONTHLY = 0.0030
SILENT_EXPOSURE_FLOOR_MAX_MONTHLY = 0.0350
GBD_TO_SILENT_EXPOSURE_MULTIPLIER = 60.0
NATURAL_EXPOSURE_CONTACT_SENSITIVITY_DEFAULT = 4.5
NATURAL_EXPOSURE_INITIAL_STOCK_MONTHS = 24.0
ROUTINE_DTP1_ELIGIBILITY_BY_AGE = {
    "infant_0_2m": 0.30,
    "infant_3_11m": 0.60,
    "child_1_4y": 0.10,
}
ROUTINE_DTP3_ELIGIBILITY_BY_AGE = {
    "infant_0_2m": 0.00,
    "infant_3_11m": 0.82,
    "child_1_4y": 0.18,
}
ROUTINE_DELAY_AGE_WEIGHT_BY_AGE = {
    "infant_0_2m": 2.5,
    "infant_3_11m": 2.0,
    "child_1_4y": 1.4,
    "child_5_9y": 0.6,
    "adolescent_10_17y": 0.2,
}
ROUTINE_DELAY_SENSITIVITY_DEFAULT = 8.0

MODEL2B_DEFAULT_INITIAL_PARAMS = {
    "beta": 0.01,
    "reporting_rate": 0.06,
    "contact_sensitivity": 1.0,
    "post_reporting_multiplier": 1.0,
    "post_transmission_multiplier": 1.0,
    "rebound_peak_month": 24.0,
    "rebound_width": 6.0,
    "rebound_asymmetry": 0.0,
    "rebound_release_gain": 1.25,
    "secondary_rebound_delay": 9.0,
    "secondary_rebound_width": 5.0,
    "secondary_rebound_release_gain": 0.35,
    "outbreak_seeding_gain": 0.025,
    "seasonal_strength": 1.0,
    "natural_boosting_efficiency": 0.60,
    "pathogen_pressure_scale": 1.0,
    "epidemic_feedback_gain": 4.0,
    "endemic_reseed_pressure": 0.01,
    "epidemic_pressure_retention": 0.45,
    "post_reporting_peak_gain": 0.15,
    "post_reporting_trend": 0.0,
    "waning_rate": 0.10,
    "nb_phi": NB_DISPERSION,
    "nb_phi_month": NB_DISPERSION,
    "nb_phi_age": 10.0,
    "init_vaccine_recent_fraction": 0.35,
    "init_natural_recent_fraction": 0.40,
}

CONTACT_BIN_RE = re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)")


def _contact_sensitivity_upper() -> float:
    raw = os.environ.get("MODEL2C_CONTACT_SENSITIVITY_UPPER", os.environ.get("MODEL2B_CONTACT_SENSITIVITY_UPPER", "5.0"))
    return float(np.clip(_finite(raw, 5.0), 5.0, 10.0))


def _natural_exposure_contact_sensitivity() -> float:
    raw = os.environ.get(
        "MODEL2C_NATURAL_EXPOSURE_CONTACT_SENSITIVITY",
        str(NATURAL_EXPOSURE_CONTACT_SENSITIVITY_DEFAULT),
    )
    return float(np.clip(_finite(raw, NATURAL_EXPOSURE_CONTACT_SENSITIVITY_DEFAULT), 0.50, 7.00))


def _routine_delay_sensitivity() -> float:
    raw = os.environ.get("MODEL2C_ROUTINE_DELAY_SENSITIVITY", str(ROUTINE_DELAY_SENSITIVITY_DEFAULT))
    return float(np.clip(_finite(raw, ROUTINE_DELAY_SENSITIVITY_DEFAULT), 0.0, 15.0))


def _env_mode(name: str, allowed: set[str], default: str) -> str:
    value = os.environ.get(name, default).strip().casefold().replace("-", "_")
    return value if value in allowed else default


def _reporting_prior_mode() -> str:
    return _env_mode("MODEL2C_REPORTING_PRIOR_MODE", {"default", "weak", "null", "direct"}, "default")


def _month_observation_weight_mode() -> str:
    return _env_mode("MODEL2C_MONTH_WEIGHT_MODE", {"peak_balanced", "unit"}, "peak_balanced")


def _age_observation_mode() -> str:
    return _env_mode(
        "MODEL2C_AGE_OBSERVATION_MODE",
        {"allocation_weighted", "unit_weight", "exclude_split_bins"},
        "allocation_weighted",
    )


def _fit_end_year(start_year: int, end_year: int) -> int:
    raw = os.environ.get("MODEL2C_FIT_END_YEAR", str(MODEL2B_FIT_END_YEAR))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = MODEL2B_FIT_END_YEAR
    return int(np.clip(value, start_year, end_year))


def _waning_prior_mode() -> str:
    return _env_mode("MODEL2C_WANING_PRIOR_MODE", {"default", "low", "high"}, "default")


def _waning_prior_sd() -> float:
    raw = os.environ.get("MODEL2C_WANING_PRIOR_SD", "0.05")
    return float(np.clip(_finite(raw, 0.05), 0.03, 0.12))


def _finite(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _clip_log(value: float, lower: float = 1e-9) -> float:
    return float(math.log(max(float(value), lower)))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _month_grid(start_year: int, end_year: int) -> pd.DataFrame:
    rows = [
        {"year": year, "month": month}
        for year in range(int(start_year), int(end_year) + 1)
        for month in range(1, 13)
    ]
    grid = pd.DataFrame(rows)
    grid["month_index"] = np.arange(1, len(grid) + 1, dtype=int)
    grid["year_index"] = grid["year"].rank(method="dense").astype(int)
    return grid


def _annual_lookup(context: CountryContext, year: int, column: str, default: float) -> float:
    annual = context.annual.copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    annual[column] = pd.to_numeric(annual.get(column), errors="coerce")
    exact = annual.loc[annual["year"].eq(year), column]
    if exact.notna().any():
        return _finite(exact.dropna().iloc[0], default)
    valid = annual.dropna(subset=["year", column]).sort_values("year")
    if valid.empty:
        return default
    before = valid.loc[valid["year"].le(year)]
    after = valid.loc[valid["year"].ge(year)]
    if not before.empty and not after.empty:
        lo = before.iloc[-1]
        hi = after.iloc[0]
        if int(lo["year"]) == int(hi["year"]):
            return _finite(lo[column], default)
        span = float(hi["year"] - lo["year"])
        weight = float((year - lo["year"]) / span) if span else 0.0
        return _finite((1.0 - weight) * float(lo[column]) + weight * float(hi[column]), default)
    edge = before.iloc[-1] if not before.empty else after.iloc[0]
    value = _finite(edge[column], default)
    if column == "population" and int(edge["year"]) != int(year):
        value *= 1.003 ** (int(year) - int(edge["year"]))
    return value


def _age_population_matrix(context: CountryContext, years: list[int], *, reference_year: int | None = None) -> list[list[float]]:
    pop = context.population_age.copy()
    if pop.empty:
        return [
            [
                _annual_lookup(context, year, "population", 1.0) / len(AGE_ORDER)
                for _age in AGE_ORDER
            ]
            for year in years
        ]
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    out: list[list[float]] = []
    for year in years:
        target_year = int(reference_year) if reference_year is not None else int(year)
        available = pop.dropna(subset=["year", "population"]).copy()
        available["year_distance"] = (available["year"] - target_year).abs()
        chosen_year = int(available.sort_values("year_distance")["year"].iloc[0])
        chosen = available.loc[available["year"].eq(chosen_year)].copy()
        by_age = chosen.groupby("age_group")["population"].sum()
        values = np.array([_finite(by_age.get(age), 0.0) for age in AGE_ORDER], dtype=float)
        total = values.sum()
        if total <= 0:
            shares = np.repeat(1.0 / len(AGE_ORDER), len(AGE_ORDER))
        else:
            shares = values / total
        annual_total = _annual_lookup(context, year, "population", float(total if total > 0 else 1.0))
        out.append((shares * max(annual_total, 1.0)).astype(float).tolist())
    return out


def _age_width_years() -> list[float]:
    return [
        float(max(AGE_BOUNDS[age][1] - AGE_BOUNDS[age][0], 1.0 / 12.0))
        for age in AGE_ORDER
    ]


def _age_aging_rate() -> list[float]:
    widths = _age_width_years()
    rates = []
    for index, width in enumerate(widths):
        if index == len(widths) - 1:
            rates.append(0.0)
        else:
            rates.append(float(np.clip(1.0 / (12.0 * max(width, 1e-6)), 0.0, 0.95)))
    return rates


def _annual_births_lookup(context: CountryContext, year: int, *, reference_year: int | None = None) -> float:
    pop = context.population_age.copy()
    target_year = int(reference_year) if reference_year is not None else int(year)
    if not pop.empty and {"year", "births"}.issubset(pop.columns):
        pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
        pop["births"] = pd.to_numeric(pop["births"], errors="coerce")
        available = pop.dropna(subset=["year", "births"]).copy()
        if not available.empty:
            available["year_distance"] = (available["year"] - target_year).abs()
            chosen_year = int(available.sort_values(["year_distance", "year"])["year"].iloc[0])
            chosen = available.loc[available["year"].eq(chosen_year), "births"].dropna()
            if not chosen.empty:
                return max(float(chosen.iloc[0]), 0.0)

    age_pop = np.asarray(_age_population_matrix(context, [target_year])[0], dtype=float)
    infant_birth_proxy = age_pop[0] * 12.0 / max(_age_width_years()[0] * 12.0, 1.0)
    population = _annual_lookup(context, target_year, "population", float(np.nansum(age_pop)))
    fallback_births = max(infant_birth_proxy, 0.012 * max(population, 1.0))
    return float(fallback_births)


def _monthly_births(context: CountryContext, years: list[int], *, reference_year: int | None = None) -> list[float]:
    return [
        _annual_births_lookup(context, int(year), reference_year=reference_year) / 12.0
        for year in years
    ]


def _monthly_contact_intensity(context: CountryContext, grid: pd.DataFrame) -> list[float]:
    def with_low_circulation_memory(raw_values: np.ndarray, years: pd.Series) -> list[float]:
        retention = 0.5 ** (1.0 / 18.0)
        memory = 0.0
        adjusted: list[float] = []
        for raw, year in zip(raw_values, years.astype(int), strict=False):
            value = float(np.clip(raw if np.isfinite(raw) else 0.0, 0.0, 100.0))
            if 2020 <= int(year) <= 2022:
                memory = max(value, memory * retention)
                adjusted.append(max(value, memory))
            elif 2023 <= int(year) <= 2024:
                memory *= retention
                adjusted.append(memory)
            else:
                if int(year) < 2020 or int(year) > 2024:
                    memory = 0.0
                adjusted.append(0.0)
        return np.clip(np.asarray(adjusted, dtype=float), 0.0, 100.0).astype(float).tolist()

    monthly = context.monthly_covid.copy()
    if not monthly.empty:
        monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce").astype("Int64")
        monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype("Int64")
        monthly["monthly_stringency_index"] = pd.to_numeric(
            monthly.get("monthly_stringency_index"),
            errors="coerce",
        )
        merged = grid.merge(
            monthly.loc[:, ["year", "month", "monthly_stringency_index"]],
            on=["year", "month"],
            how="left",
        )
        values = merged["monthly_stringency_index"].fillna(0.0).clip(lower=0.0, upper=100.0)
        values = np.where(merged["year"].between(2020, 2022), values, 0.0)
        return with_low_circulation_memory(np.asarray(values, dtype=float), merged["year"])
    annual = []
    for row in grid.itertuples(index=False):
        value = _annual_lookup(context, int(row.year), "stringency_index", 0.0)
        annual.append(float(np.clip(value if 2020 <= int(row.year) <= 2022 else 0.0, 0.0, 100.0)))
    return with_low_circulation_memory(np.asarray(annual, dtype=float), grid["year"])


def _monthly_seasonal_multiplier(context: CountryContext, grid: pd.DataFrame) -> list[float]:
    seasonal = monthly_seasonal_fractions(context).copy()
    if seasonal.empty:
        lookup = {month: 1.0 for month in range(1, 13)}
    else:
        seasonal["month"] = pd.to_numeric(seasonal["month"], errors="coerce").astype("Int64")
        seasonal["seasonal_fraction"] = pd.to_numeric(seasonal["seasonal_fraction"], errors="coerce")
        seasonal = seasonal.dropna(subset=["month", "seasonal_fraction"])
        lookup = {
            int(row.month): float(np.clip(float(row.seasonal_fraction) * 12.0, 0.08, 6.0))
            for row in seasonal.itertuples(index=False)
        }
    values = np.array([lookup.get(int(month), 1.0) for month in grid["month"]], dtype=float)
    if not np.isfinite(values).all() or values.mean() <= 0:
        return [1.0 for _ in range(len(grid))]
    values = values / values.mean()
    return values.clip(0.08, 6.0).astype(float).tolist()


def _historical_periodic_multiplier(context: CountryContext, grid: pd.DataFrame) -> list[float]:
    values = []
    for year in grid["year"]:
        year_int = int(year)
        if year_int <= 2019:
            expected = _annual_lookup(
                context,
                year_int,
                "periodic_expected_incidence_per_100k",
                context.pre_mean_incidence,
            )
            value = max(expected, 0.001) / max(context.pre_mean_incidence, 0.001)
        else:
            value = 1.0
        values.append(float(value))
    return np.clip(np.asarray(values, dtype=float), 0.15, 4.0).astype(float).tolist()


def _reporting_prior(context: CountryContext) -> dict[str, Any]:
    mode = _reporting_prior_mode()
    prior = monthly_detection_prior_table(context, MODEL2B_START_YEAR, MODEL2B_FIT_END_YEAR, include_testing_denominator=True)
    post = prior.loc[prior["year"].between(2023, 2025)].copy() if not prior.empty else pd.DataFrame()
    direct_support = False
    support_level = "not_available"
    source_ids = "not_available"
    if not post.empty:
        support_fields = [
            "ode_testing_denominator_support_level",
            "ode_testing_denominator_series_support_level",
        ]
        support_values = [
            str(value).strip()
            for field in support_fields
            for value in post.get(field, pd.Series(dtype=str)).dropna().unique()
            if str(value).strip()
        ]
        support_level = ";".join(dict.fromkeys(support_values)) if support_values else "not_available"
        direct_support = any(value != "not_available" for value in support_values)
        source_values = post.get("ode_testing_denominator_series_source_ids", pd.Series(dtype=str)).dropna().astype(str)
        source_ids = ";".join(value for value in source_values.unique() if value != "not_available") or "not_available"

    if mode == "null":
        mean = 1.0
        lower = math.exp(-0.20)
        upper = math.exp(0.20)
        sigma = 0.10
        method = "null_reporting_prior_sensitivity"
    elif direct_support and not post.empty:
        mean = float(pd.to_numeric(post.get("ode_relative_detection_prior_mean"), errors="coerce").median())
        lower = float(pd.to_numeric(post.get("ode_relative_detection_prior_lower"), errors="coerce").median())
        upper = float(pd.to_numeric(post.get("ode_relative_detection_prior_upper"), errors="coerce").median())
        mean = float(np.clip(mean if np.isfinite(mean) else 1.0, 0.25, 4.0))
        lower = float(np.clip(lower if np.isfinite(lower) else mean * 0.85, 0.10, 4.0))
        upper = float(np.clip(upper if np.isfinite(upper) else mean * 1.15, 0.10, 5.0))
        sigma = max((math.log(max(upper, lower + 1e-6)) - math.log(max(lower, 1e-6))) / (2.0 * 1.96), 0.10)
        method = "direct_testing_denominator_or_strong_detection_prior"
    else:
        mean = 1.0
        lower = math.exp(-0.40)
        upper = math.exp(0.40)
        sigma = 0.20
        method = "strong_null_reporting_prior_without_direct_testing_denominator"
    if mode == "weak":
        lower = float(np.clip(mean * math.exp(-1.96 * 0.75), 0.10, 4.0))
        upper = float(np.clip(mean * math.exp(1.96 * 0.75), 0.10, 5.0))
        sigma = 0.75
        method = "weak_reporting_prior_sensitivity"
    elif mode == "direct" and not direct_support:
        mean = 1.0
        lower = math.exp(-0.40)
        upper = math.exp(0.40)
        sigma = 0.20
        method = "direct_reporting_prior_requested_without_direct_denominator"
    elif mode == "direct":
        method = "direct_reporting_prior_sensitivity"
    return {
        "reporting_prior_mean": mean,
        "reporting_prior_lower": lower,
        "reporting_prior_upper": upper,
        "reporting_prior_log_mean": _clip_log(mean),
        "reporting_prior_log_sd": float(np.clip(sigma, 0.10, 0.75)),
        "direct_testing_support": direct_support,
        "reporting_prior_support_level": support_level,
        "reporting_prior_source_ids": source_ids,
        "reporting_prior_method": method,
        "reporting_prior_sensitivity_mode": mode,
    }


def _absolute_reporting_rate_prior(
    context: CountryContext,
    inputs: dict[str, pd.DataFrame],
    *,
    direct_testing_support: bool,
) -> dict[str, Any]:
    gbd_incidence_per_100k = _gbd_incidence_per_100k(context, inputs)
    if direct_testing_support:
        mean = 0.03
        sigma = 0.75
        return {
            "absolute_reporting_rate_prior_mean": mean,
            "absolute_reporting_rate_prior_lower": float(np.clip(mean * math.exp(-1.96 * sigma), 0.002, 0.35)),
            "absolute_reporting_rate_prior_upper": float(np.clip(mean * math.exp(1.96 * sigma), 0.002, 0.35)),
            "absolute_reporting_rate_prior_log_mean": _clip_log(mean, lower=0.002),
            "absolute_reporting_rate_prior_log_sd": sigma,
            "absolute_reporting_rate_prior_method": "weak_absolute_reporting_prior_with_direct_relative_detection_evidence",
            "gbd_incidence_per_100k_for_reporting_anchor": np.nan,
            "reported_to_gbd_incidence_ratio_for_reporting_anchor": np.nan,
        }

    if np.isfinite(gbd_incidence_per_100k) and gbd_incidence_per_100k > 0:
        reported_to_modeled_ratio = float(context.pre_mean_incidence) / max(gbd_incidence_per_100k, 1e-6)
        mean = float(np.clip(reported_to_modeled_ratio, 0.002, 0.35))
        sigma = 0.45
        method = "gbd_incidence_to_reported_baseline_empirical_bayes"
    else:
        reported_to_modeled_ratio = np.nan
        mean = 0.03
        sigma = 0.75
        method = "weak_default_without_gbd_incidence_anchor"

    return {
        "absolute_reporting_rate_prior_mean": mean,
        "absolute_reporting_rate_prior_lower": float(np.clip(mean * math.exp(-1.96 * sigma), 0.002, 0.35)),
        "absolute_reporting_rate_prior_upper": float(np.clip(mean * math.exp(1.96 * sigma), 0.002, 0.35)),
        "absolute_reporting_rate_prior_log_mean": _clip_log(mean, lower=0.002),
        "absolute_reporting_rate_prior_log_sd": float(np.clip(sigma, 0.10, 1.50)),
        "absolute_reporting_rate_prior_method": method,
        "gbd_incidence_per_100k_for_reporting_anchor": float(gbd_incidence_per_100k) if np.isfinite(gbd_incidence_per_100k) else np.nan,
        "reported_to_gbd_incidence_ratio_for_reporting_anchor": (
            float(reported_to_modeled_ratio) if np.isfinite(reported_to_modeled_ratio) else np.nan
        ),
    }


def _gbd_incidence_per_100k(context: CountryContext, inputs: dict[str, pd.DataFrame]) -> float:
    annual = context.annual.copy()
    annual["year"] = pd.to_numeric(annual.get("year"), errors="coerce")
    annual["population"] = pd.to_numeric(annual.get("population"), errors="coerce")
    population = float(annual.loc[annual["year"].between(2010, 2021), "population"].mean())
    gbd = inputs.get("gbd_burden", pd.DataFrame()).copy()
    if not gbd.empty and "iso3" in gbd:
        gbd = gbd.loc[gbd["iso3"].astype(str).eq(context.iso3)].copy()
    if not gbd.empty:
        gbd["year"] = pd.to_numeric(gbd.get("year"), errors="coerce")
        gbd["val"] = pd.to_numeric(gbd.get("val"), errors="coerce")
        measure = gbd.get("measure_name", pd.Series(dtype=str)).astype(str).str.contains("Incidence", case=False, na=False)
        metric = gbd.get("metric_name", pd.Series(dtype=str)).astype(str).eq("Number")
        age = gbd.get("age_name", pd.Series(dtype=str)).astype(str).eq("All ages")
        selected = gbd.loc[measure & metric & age & gbd["year"].between(2010, 2021), "val"].dropna()
        gbd_incidence_number = float(selected.mean()) if not selected.empty else np.nan
    else:
        gbd_incidence_number = np.nan

    if np.isfinite(gbd_incidence_number) and gbd_incidence_number > 0 and np.isfinite(population) and population > 0:
        return float(100000.0 * gbd_incidence_number / population)
    return float("nan")


def _silent_exposure_floor_prior(context: CountryContext, inputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    gbd_incidence_per_100k = _gbd_incidence_per_100k(context, inputs)
    if np.isfinite(gbd_incidence_per_100k) and gbd_incidence_per_100k > 0:
        gbd_floor = gbd_incidence_per_100k / 100000.0 / 12.0 * GBD_TO_SILENT_EXPOSURE_MULTIPLIER
    else:
        gbd_floor = np.nan
    serology_recent_percent = _serology_recent_infection_percent(context, inputs)
    serology_floor = (
        serology_recent_percent / 100.0 / 12.0
        if np.isfinite(serology_recent_percent) and serology_recent_percent > 0
        else np.nan
    )
    anchors = [SILENT_EXPOSURE_FLOOR_MIN_MONTHLY]
    if np.isfinite(gbd_floor):
        anchors.append(float(gbd_floor))
    if np.isfinite(serology_floor):
        anchors.append(float(serology_floor))
    floor = float(np.clip(max(anchors), 0.0, SILENT_EXPOSURE_FLOOR_MAX_MONTHLY))
    if np.isfinite(serology_floor):
        method = "serology_recent_infection_prevalence_with_optional_gbd_floor"
    elif np.isfinite(gbd_floor):
        method = "gbd_incidence_scaled_silent_exposure_floor"
    else:
        method = "minimum_silent_exposure_floor_without_gbd_anchor"
    return {
        "silent_exposure_floor_monthly": floor,
        "silent_exposure_floor_min_monthly": float(SILENT_EXPOSURE_FLOOR_MIN_MONTHLY),
        "silent_exposure_floor_max_monthly": float(SILENT_EXPOSURE_FLOOR_MAX_MONTHLY),
        "silent_exposure_gbd_multiplier": float(GBD_TO_SILENT_EXPOSURE_MULTIPLIER),
        "silent_exposure_gbd_incidence_per_100k": (
            float(gbd_incidence_per_100k) if np.isfinite(gbd_incidence_per_100k) else np.nan
        ),
        "silent_exposure_gbd_floor_monthly": float(gbd_floor) if np.isfinite(gbd_floor) else np.nan,
        "silent_exposure_serology_recent_infection_percent": (
            float(serology_recent_percent) if np.isfinite(serology_recent_percent) else np.nan
        ),
        "silent_exposure_serology_floor_monthly": float(serology_floor) if np.isfinite(serology_floor) else np.nan,
        "silent_exposure_floor_method": method,
    }


def _initial_natural_share_prior_mean(context: CountryContext, silent_exposure_floor: dict[str, Any]) -> float:
    reported_anchor = max(_finite(context.pre_mean_incidence, 0.0), 0.0) / 500.0
    exposure_anchor = (
        _finite(silent_exposure_floor.get("silent_exposure_floor_monthly"), SILENT_EXPOSURE_FLOOR_MIN_MONTHLY)
        * NATURAL_EXPOSURE_INITIAL_STOCK_MONTHS
    )
    return float(np.clip(max(0.02, reported_anchor, exposure_anchor), 0.02, 0.45))


def _routine_weight_vector(weights: dict[str, float], *, upper: float = 1.0) -> list[float]:
    return [float(np.clip(weights.get(age, 0.0), 0.0, upper)) for age in AGE_ORDER]


def _serology_recent_infection_percent(context: CountryContext, inputs: dict[str, pd.DataFrame]) -> float:
    serology = inputs.get("serology_evidence", pd.DataFrame()).copy()
    if serology.empty or "iso3" not in serology:
        return float("nan")
    serology = serology.loc[serology["iso3"].astype(str).eq(context.iso3)].copy()
    if serology.empty:
        return float("nan")
    metric = serology.get("serology_metric", pd.Series(dtype=str)).astype(str).str.casefold()
    unit = serology.get("serology_unit", pd.Series(dtype=str)).astype(str).str.casefold()
    serology["serology_value"] = pd.to_numeric(serology.get("serology_value"), errors="coerce")
    selected = serology.loc[
        metric.eq("recent_infection_prevalence")
        & unit.eq("percent")
        & serology["serology_value"].gt(0),
        "serology_value",
    ].dropna()
    if selected.empty:
        return float("nan")
    return float(selected.min())


def _rebound_peak_month_prior(fit_monthly: pd.DataFrame) -> float:
    if fit_monthly.empty:
        return 24.0
    post = fit_monthly.loc[fit_monthly["year"].between(2023, MODEL2B_FIT_END_YEAR)].copy()
    if post.empty:
        return 24.0
    post["observed_cases"] = pd.to_numeric(post.get("observed_cases"), errors="coerce")
    post = post.dropna(subset=["observed_cases", "year", "month"])
    if post.empty or post["observed_cases"].max() <= 0:
        return 24.0
    peak = post.sort_values(["observed_cases", "year", "month"], ascending=[False, True, True]).iloc[0]
    post_month = (int(peak["year"]) - 2023) * 12 + int(peak["month"])
    return float(np.clip(post_month, 1.0, 36.0))


def _monthly_observation_weights(fit_monthly: pd.DataFrame) -> list[float]:
    if fit_monthly.empty:
        return []
    if _month_observation_weight_mode() == "unit":
        return [1.0 for _ in range(len(fit_monthly))]
    monthly = fit_monthly.copy()
    monthly["observed_cases"] = pd.to_numeric(monthly.get("observed_cases"), errors="coerce")
    cases = monthly["observed_cases"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    positive = cases[cases > 0.0]
    if positive.size == 0:
        return [1.0 for _ in range(len(monthly))]

    high_reference = float(np.nanquantile(positive, 0.90))
    high_reference = max(high_reference, float(np.nanmedian(positive)), 1.0)
    peak_reference = max(float(np.nanmax(positive)), high_reference * 1.01)
    log_cases = np.log1p(cases)
    log_high = np.log1p(high_reference)
    log_peak = np.log1p(peak_reference)
    base_score = np.clip(log_cases / max(log_high, 1e-6), 0.0, 1.35)
    weights = 0.60 + 0.95 * base_score**1.25

    year = pd.to_numeric(monthly.get("year"), errors="coerce").fillna(0).astype(int).to_numpy()
    post = year >= 2023
    weights[post] += 0.45 * np.clip(base_score[post], 0.0, 1.20) ** 1.10
    post_score = np.zeros(len(cases), dtype=float)
    post_score[post] = np.clip((log_cases[post] - log_high) / max(log_peak - log_high, 1e-6), 0.0, 1.0)
    post_high = post & (cases >= high_reference)
    weights[post_high] += 1.25 + 1.85 * post_score[post_high] ** 0.80
    post_peak = post & (cases >= 0.85 * peak_reference)
    weights[post_peak] += 0.50

    zero_or_near_zero = cases <= 0.0
    weights[zero_or_near_zero] = np.minimum(weights[zero_or_near_zero], 0.60)
    return np.clip(weights, 0.55, 5.50).astype(float).tolist()


def _contact_mode() -> str:
    raw = os.environ.get("MODEL2B_CONTACT_MODE", "full").strip().lower().replace("-", "_")
    return "full" if raw == "full" else "rank_one"


def _rank_one_contact_matrix() -> np.ndarray:
    rel = np.array([float(AGE_REL_CONTACT[age]) for age in AGE_ORDER], dtype=float)
    return np.outer(rel, rel)


def _parse_contact_bin(value: object) -> tuple[float, float] | None:
    match = CONTACT_BIN_RE.match(str(value).strip())
    if not match:
        return None
    lower = float(match.group(1))
    upper = float(match.group(2))
    return (lower, upper) if upper > lower else None


def _overlap_fraction(source: tuple[float, float], target: tuple[float, float]) -> float:
    lower = max(float(source[0]), float(target[0]))
    upper = min(float(source[1]), float(target[1]))
    if upper <= lower:
        return 0.0
    return float((upper - lower) / max(source[1] - source[0], 1e-9))


def _source_aggregation_fraction(source: tuple[float, float], target: tuple[float, float]) -> float:
    lower = max(float(source[0]), float(target[0]))
    upper = min(float(source[1]), float(target[1]))
    if upper <= lower:
        return 0.0
    return float((upper - lower) / max(target[1] - target[0], 1e-9))


def _normalise_contact_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.where(np.isfinite(matrix) & (matrix > 0), matrix, 0.0)
    target_total = float(_rank_one_contact_matrix().sum())
    current_total = float(matrix.sum())
    if current_total <= 0 or target_total <= 0:
        return _rank_one_contact_matrix()
    return matrix * (target_total / current_total)


def _full_contact_matrix(context: CountryContext, inputs: dict[str, pd.DataFrame]) -> tuple[np.ndarray | None, str]:
    contact = inputs.get("contact_mobility", pd.DataFrame()).copy()
    if contact.empty or "iso3" not in contact.columns:
        return None, "rank_one_fallback_no_contact_mobility_table"
    contact = contact.loc[contact["iso3"].astype(str).str.upper().eq(context.iso3.upper())].copy()
    if contact.empty:
        return None, "rank_one_fallback_no_country_contact_matrix"
    if "contact_location" in contact and contact["contact_location"].astype(str).eq("all").any():
        contact = contact.loc[contact["contact_location"].astype(str).eq("all")].copy()
    required = {"source_age_group", "target_age_group", "contacts_per_day"}
    if not required.issubset(contact.columns):
        return None, "rank_one_fallback_incomplete_contact_matrix_columns"
    contact["contacts_per_day"] = pd.to_numeric(contact["contacts_per_day"], errors="coerce")
    contact = contact.dropna(subset=["contacts_per_day"])
    if contact.empty:
        return None, "rank_one_fallback_empty_contact_matrix_values"

    age_bounds = [AGE_BOUNDS[age] for age in AGE_ORDER]
    matrix = np.zeros((len(AGE_ORDER), len(AGE_ORDER)), dtype=float)
    for row in contact.itertuples(index=False):
        source_bin = _parse_contact_bin(getattr(row, "source_age_group"))
        target_bin = _parse_contact_bin(getattr(row, "target_age_group"))
        if source_bin is None or target_bin is None:
            continue
        contacts = _finite(getattr(row, "contacts_per_day"), 0.0)
        if contacts <= 0:
            continue
        source_weights = [_source_aggregation_fraction(source_bin, bounds) for bounds in age_bounds]
        target_weights = [_overlap_fraction(target_bin, bounds) for bounds in age_bounds]
        for source_index, source_weight in enumerate(source_weights):
            if source_weight <= 0:
                continue
            for target_index, target_weight in enumerate(target_weights):
                if target_weight > 0:
                    matrix[source_index, target_index] += contacts * source_weight * target_weight

    if matrix.sum() <= 0:
        return None, "rank_one_fallback_unparseable_contact_matrix"
    return _normalise_contact_matrix(matrix), "full_contact_matrix"


def _age_contact_matrix(context: CountryContext, inputs: dict[str, pd.DataFrame]) -> tuple[np.ndarray, str, str]:
    mode = _contact_mode()
    if mode == "full":
        matrix, source = _full_contact_matrix(context, inputs)
        if matrix is not None:
            return matrix, source, mode
        return _rank_one_contact_matrix(), source, mode
    return _rank_one_contact_matrix(), "rank_one_contact_vector_outer_product", mode


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
    text = str(value).strip().casefold()
    return text in {"1", "true", "t", "yes", "y"}


def _vaccine_program_priors(context: CountryContext, init: dict[str, float]) -> dict[str, Any]:
    waning_mode = _waning_prior_mode()
    waning_rate_prior_sd = _waning_prior_sd()
    raw_type = str(context.metadata.get("routine_scheduler_code", context.metadata.get("vaccine_type_primary", "unknown")))
    type_key = raw_type.strip().casefold()
    if type_key == "ap" or "acellular" in type_key:
        vaccine_type = "aP"
        base_ve_infection = 0.50
        base_ve_disease = 0.88
        waning_rate_prior_mean = 0.14
        platform_waning_modifier = 1.12
    elif type_key == "wp" or "whole" in type_key:
        vaccine_type = "wP"
        base_ve_infection = 0.42
        base_ve_disease = 0.82
        waning_rate_prior_mean = 0.075
        platform_waning_modifier = 0.88
    else:
        vaccine_type = raw_type if raw_type and raw_type != "nan" else "unknown"
        base_ve_infection = 0.46
        base_ve_disease = 0.85
        waning_rate_prior_mean = init["waning_rate"]
        platform_waning_modifier = 1.0
    if waning_mode == "low":
        waning_rate_prior_mean = min(waning_rate_prior_mean, 0.045)
        platform_waning_modifier *= 0.85
    elif waning_mode == "high":
        waning_rate_prior_mean = max(waning_rate_prior_mean * 1.55, 0.16 if vaccine_type == "aP" else 0.11)
        platform_waning_modifier *= 1.15

    adolescent_booster = _metadata_bool(context.metadata.get("adolescent_booster", False))
    maternal_program = _metadata_bool(context.metadata.get("maternal_program", False))
    reach = np.ones(len(AGE_ORDER), dtype=float)
    waning_modifier = np.ones(len(AGE_ORDER), dtype=float) * platform_waning_modifier
    adolescent_index = AGE_ORDER.index("adolescent_10_17y")
    young_adult_index = AGE_ORDER.index("young_adult_18_39y")
    if adolescent_booster:
        reach[adolescent_index] = 1.20
        reach[young_adult_index] = 1.08
        waning_modifier[adolescent_index] *= 0.82
        waning_modifier[young_adult_index] *= 0.92
    else:
        reach[adolescent_index] = 0.82
        reach[young_adult_index] = 0.90
        waning_modifier[adolescent_index] *= 1.18
        waning_modifier[young_adult_index] *= 1.08

    return {
        "vaccine_type": vaccine_type,
        "base_ve_infection": float(np.clip(base_ve_infection, 0.0, 1.0)),
        "base_ve_disease": float(np.clip(base_ve_disease, 0.0, 1.0)),
        "waning_rate_prior_mean": float(np.clip(waning_rate_prior_mean, 0.005, 0.40)),
        "waning_rate_prior_sd": float(waning_rate_prior_sd),
        "waning_prior_sensitivity_mode": waning_mode,
        "booster_reach_modifier": np.clip(reach, 0.40, 1.40).astype(float).tolist(),
        "vaccine_waning_age_modifier": np.clip(waning_modifier, 0.35, 2.50).astype(float).tolist(),
        "adolescent_booster": adolescent_booster,
        "maternal_program_context": maternal_program,
    }


def _country_table_rows(table: pd.DataFrame, iso3: str) -> pd.DataFrame:
    if table.empty or "iso3" not in table.columns:
        return pd.DataFrame()
    return table.loc[table["iso3"].astype(str).str.upper().eq(iso3.upper())].copy()


def _evidence_metadata(context: CountryContext, inputs: dict[str, pd.DataFrame]) -> dict[str, bool]:
    surveillance = context.surveillance_outcomes.copy()
    testing_denominator = False
    if not surveillance.empty and "testing_denominator" in surveillance:
        testing_values = pd.to_numeric(surveillance["testing_denominator"], errors="coerce")
        testing_denominator = bool(testing_values.gt(0).any())

    severe_columns = (
        [
            column
            for column in surveillance.columns
            if any(token in column.lower() for token in ("hospital", "death", "fatal", "severe"))
        ]
        if not surveillance.empty
        else []
    )
    severe_outcome = False
    if severe_columns:
        severe_outcome = bool(surveillance[severe_columns].notna().to_numpy().any())

    diagnostic = context.diagnostic_timeline.copy()
    case_definition = bool(not diagnostic.empty)
    pathogen_pressure = bool(not context.pathogen_timeline.empty)
    if not pathogen_pressure:
        pathogen_pressure = not _country_table_rows(inputs.get("strain_amr_timeline", pd.DataFrame()), context.iso3).empty
    serology = not _country_table_rows(inputs.get("serology_evidence", pd.DataFrame()), context.iso3).empty
    return {
        "direct_testing_denominator_evidence": testing_denominator,
        "case_definition_reporting_practice_evidence": case_definition,
        "severe_outcome_evidence": severe_outcome,
        "serology_evidence": serology,
        "pathogen_pressure_evidence": pathogen_pressure,
    }


def model2b_default_initial_params() -> dict[str, float]:
    return dict(MODEL2B_DEFAULT_INITIAL_PARAMS)


def model2b_initial_values(params: dict[str, float] | None = None) -> dict[str, float]:
    params = model2b_default_initial_params() if params is None else params
    return {
        "log_beta": _clip_log(_finite(params.get("beta"), 0.01)),
        "log_reporting_rate": _clip_log(_finite(params.get("reporting_rate"), 0.06), lower=0.002),
        "contact_sensitivity": float(np.clip(_finite(params.get("contact_sensitivity"), 1.0), 0.02, 4.98)),
        "waning_rate": float(np.clip(_finite(params.get("waning_rate"), 0.10), 0.006, 0.399)),
        "log_post_transmission_multiplier": float(
            math.log(np.clip(_finite(params.get("post_transmission_multiplier"), 1.0), 0.50, 2.50))
        ),
        "log_post_reporting_multiplier": _clip_log(_finite(params.get("post_reporting_multiplier"), 1.0)),
        "rebound_peak_month": float(np.clip(_finite(params.get("rebound_peak_month"), 24.0), 1.01, 47.99)),
        "rebound_width": float(np.clip(_finite(params.get("rebound_width"), 6.0), 0.76, 13.99)),
        "rebound_asymmetry": float(np.clip(_finite(params.get("rebound_asymmetry"), 0.0), -0.74, 0.74)),
        "rebound_release_gain": float(np.clip(_finite(params.get("rebound_release_gain"), 1.25), 0.05, 3.99)),
        "secondary_rebound_delay": float(np.clip(_finite(params.get("secondary_rebound_delay"), 9.0), 2.01, 23.99)),
        "secondary_rebound_width": float(np.clip(_finite(params.get("secondary_rebound_width"), 5.0), 1.01, 15.99)),
        "secondary_rebound_release_gain": float(
            np.clip(_finite(params.get("secondary_rebound_release_gain"), 0.35), 0.001, 2.99)
        ),
        "outbreak_seeding_gain": float(np.clip(_finite(params.get("outbreak_seeding_gain"), 0.025), 0.0001, 0.299)),
        "seasonal_strength": float(np.clip(_finite(params.get("seasonal_strength"), 1.0), 0.01, 2.49)),
        "natural_boosting_efficiency": float(np.clip(_finite(params.get("natural_boosting_efficiency"), 0.60), 0.001, 0.999)),
        "pathogen_pressure_scale": float(np.clip(_finite(params.get("pathogen_pressure_scale"), 1.0), 0.001, 2.499)),
        "epidemic_feedback_gain": float(np.clip(_finite(params.get("epidemic_feedback_gain"), 4.0), 0.01, 24.99)),
        "endemic_reseed_pressure": float(np.clip(_finite(params.get("endemic_reseed_pressure"), 0.01), 0.0001, 0.079)),
        "epidemic_pressure_retention": float(np.clip(_finite(params.get("epidemic_pressure_retention"), 0.45), 0.151, 0.849)),
        "post_reporting_peak_gain": float(
            np.clip(_finite(params.get("post_reporting_peak_gain"), 0.15), 0.001, 2.99)
        ),
        "post_reporting_trend": float(np.clip(_finite(params.get("post_reporting_trend"), 0.0), -0.79, 0.79)),
        "nb_phi_month": float(
            np.clip(
                _finite(params.get("nb_phi_month", params.get("nb_phi")), NB_DISPERSION),
                0.21,
                200.0,
            )
        ),
        "nb_phi_age": float(np.clip(_finite(params.get("nb_phi_age"), 10.0), 0.21, 200.0)),
        "init_natural_share": float(np.clip(_finite(params.get("init_natural_share"), 0.08), 0.011, 0.499)),
        "init_vaccine_recent_fraction": float(
            np.clip(_finite(params.get("init_vaccine_recent_fraction"), 0.35), 0.051, 0.949)
        ),
        "init_natural_recent_fraction": float(
            np.clip(_finite(params.get("init_natural_recent_fraction"), 0.40), 0.051, 0.949)
        ),
    }


def build_model2b_stan_data(
    context: CountryContext,
    inputs: dict[str, pd.DataFrame],
    initial_params: dict[str, float] | None = None,
    *,
    start_year: int = MODEL2B_START_YEAR,
    end_year: int = MODEL2B_END_YEAR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    init = model2b_initial_values(initial_params)
    fit_end_year = _fit_end_year(start_year, end_year)
    month_weight_mode = _month_observation_weight_mode()
    age_observation_mode = _age_observation_mode()
    contact_sensitivity_upper = _contact_sensitivity_upper()
    grid = _month_grid(start_year, end_year)
    years = sorted(grid["year"].unique().astype(int).tolist())
    year_to_index = {year: index + 1 for index, year in enumerate(years)}
    grid["year_index"] = grid["year"].map(year_to_index).astype(int)
    pre_mean_dtp1 = float(np.clip(getattr(context, "pre_mean_dtp1", context.pre_mean_dtp3), 0.01, 0.99))
    grid["dtp3"] = [
        _annual_lookup(context, int(year), "dtp3_coverage", context.pre_mean_dtp3)
        for year in grid["year"]
    ]
    grid["dtp1"] = [
        _annual_lookup(context, int(year), "dtp1_coverage", pre_mean_dtp1)
        for year in grid["year"]
    ]
    grid["dtp1"] = np.maximum(grid["dtp1"].astype(float), grid["dtp3"].astype(float))
    grid["periodic_multiplier"] = _historical_periodic_multiplier(context, grid)
    grid["seasonal_multiplier"] = _monthly_seasonal_multiplier(context, grid)
    grid["contact_intensity"] = _monthly_contact_intensity(context, grid)
    pathogen_prior = monthly_pathogen_prior_table(context, start_year=start_year, end_year=end_year)
    if pathogen_prior.empty:
        grid["pathogen_pressure_prior"] = 1.0
    else:
        pathogen_prior = pathogen_prior.copy()
        pathogen_prior["year"] = pd.to_numeric(pathogen_prior["year"], errors="coerce").astype("Int64")
        pathogen_prior["month"] = pd.to_numeric(pathogen_prior["month"], errors="coerce").astype("Int64")
        pathogen_prior["ode_pathogen_pressure_prior_mean"] = pd.to_numeric(
            pathogen_prior.get("ode_pathogen_pressure_prior_mean"),
            errors="coerce",
        )
        grid = grid.merge(
            pathogen_prior.loc[:, ["year", "month", "ode_pathogen_pressure_prior_mean"]],
            on=["year", "month"],
            how="left",
        )
        grid["pathogen_pressure_prior"] = grid["ode_pathogen_pressure_prior_mean"].fillna(1.0)
        grid = grid.drop(columns=["ode_pathogen_pressure_prior_mean"])

    month_key = {(int(row.year), int(row.month)): int(row.month_index) for row in grid.itertuples(index=False)}
    monthly = monthly_observed_cases(context)
    if not monthly.empty:
        monthly = monthly.loc[monthly["year"].between(start_year, end_year)].copy()
        monthly["month_index"] = [month_key.get((int(row.year), int(row.month)), 0) for row in monthly.itertuples(index=False)]
        monthly = monthly.loc[monthly["month_index"].gt(0)].copy()
        fit_monthly = monthly.loc[monthly["year"].le(fit_end_year)].copy()
        ppc_monthly = monthly.loc[monthly["year"].gt(fit_end_year)].copy()
    else:
        fit_monthly = pd.DataFrame()
        ppc_monthly = pd.DataFrame()
    rebound_peak_month_prior_mean = _rebound_peak_month_prior(fit_monthly)
    month_obs_weight = _monthly_observation_weights(fit_monthly)

    age_obs = allocated_age_observations(inputs, context.iso3)
    if not age_obs.empty:
        age_obs = age_obs.loc[age_obs["year"].between(start_year, fit_end_year)].copy()
        age_obs["year_index"] = age_obs["year"].astype(int).map(year_to_index).fillna(0).astype(int)
        age_obs["age_index"] = age_obs["model_age_group"].map({age: index + 1 for index, age in enumerate(AGE_ORDER)}).fillna(0).astype(int)
        age_obs = age_obs.loc[age_obs["year_index"].gt(0) & age_obs["age_index"].gt(0)].copy()
        if age_observation_mode == "exclude_split_bins":
            age_obs = age_obs.loc[
                pd.to_numeric(age_obs["age_allocation_weight"], errors="coerce").fillna(0.0).ge(0.999)
            ].copy()
    if age_obs.empty:
        age_obs_weights: list[float] = []
    elif age_observation_mode == "unit_weight":
        age_obs_weights = [1.0 for _ in range(len(age_obs))]
    else:
        age_obs_weights = age_obs["age_allocation_weight"].astype(float).clip(lower=0.05, upper=1.0).tolist()
    reporting_prior = _reporting_prior(context)
    absolute_reporting_prior = _absolute_reporting_rate_prior(
        context,
        inputs,
        direct_testing_support=bool(reporting_prior["direct_testing_support"]),
    )
    silent_exposure_floor = _silent_exposure_floor_prior(context, inputs)
    init_natural_share_prior_mean = _initial_natural_share_prior_mean(context, silent_exposure_floor)
    natural_exposure_contact_sensitivity = _natural_exposure_contact_sensitivity()
    routine_delay_sensitivity = _routine_delay_sensitivity()
    contact_matrix, contact_matrix_source, contact_mode_requested = _age_contact_matrix(context, inputs)
    vaccine_program = _vaccine_program_priors(context, init)
    evidence = _evidence_metadata(context, inputs)
    pathogen_evidence = bool(evidence["pathogen_pressure_evidence"])
    pathogen_pressure_scale_prior_mean = 1.0 if pathogen_evidence else 0.25
    pathogen_pressure_scale_prior_sd = 0.45 if pathogen_evidence else 0.25
    outbreak_seeding_gain_prior_mean = 0.025 if pathogen_evidence else 0.010
    outbreak_seeding_gain_prior_log_sd = 0.70 if pathogen_evidence else 0.45
    log_post_transmission_prior_sd = 0.20

    data = {
        "N_months": int(len(grid)),
        "N_years": int(len(years)),
        "N_age": int(len(AGE_ORDER)),
        "year": grid["year"].astype(int).tolist(),
        "month": grid["month"].astype(int).tolist(),
        "year_index": grid["year_index"].astype(int).tolist(),
        "dtp1": grid["dtp1"].astype(float).clip(lower=0.01, upper=0.99).tolist(),
        "dtp3": grid["dtp3"].astype(float).clip(lower=0.01, upper=0.99).tolist(),
        "pre_mean_dtp1": float(np.clip(max(pre_mean_dtp1, context.pre_mean_dtp3), 0.01, 0.99)),
        "pre_mean_dtp3": float(np.clip(context.pre_mean_dtp3, 0.01, 0.99)),
        "pre_mean_incidence_per_100k": float(max(_finite(context.pre_mean_incidence, 0.0), 0.0)),
        "silent_exposure_floor_monthly": silent_exposure_floor["silent_exposure_floor_monthly"],
        "natural_exposure_contact_sensitivity": float(natural_exposure_contact_sensitivity),
        "periodic_multiplier": grid["periodic_multiplier"].astype(float).tolist(),
        "seasonal_multiplier": grid["seasonal_multiplier"].astype(float).clip(lower=0.08, upper=6.0).tolist(),
        "pathogen_pressure_prior": grid["pathogen_pressure_prior"].astype(float).clip(lower=0.30, upper=5.0).tolist(),
        "contact_intensity": grid["contact_intensity"].astype(float).clip(lower=0.0, upper=100.0).tolist(),
        "age_population": _age_population_matrix(context, grid["year"].astype(int).tolist()),
        "age_population_ref": _age_population_matrix(context, grid["year"].astype(int).tolist(), reference_year=2019),
        "monthly_births": _monthly_births(context, grid["year"].astype(int).tolist()),
        "monthly_births_ref": _monthly_births(context, grid["year"].astype(int).tolist(), reference_year=2019),
        "age_width_years": _age_width_years(),
        "age_aging_rate": _age_aging_rate(),
        "age_rel_contact": [float(AGE_REL_CONTACT[age]) for age in AGE_ORDER],
        "age_contact_matrix": contact_matrix.astype(float).tolist(),
        "age_symptom_prob": [float(AGE_SYMPTOM_PROB[age]) for age in AGE_ORDER],
        "age_report_weight": [float(AGE_REPORT_WEIGHT[age]) for age in AGE_ORDER],
        "age_program_reach": [float(AGE_VACCINE_PROGRAM_REACH[age]) for age in AGE_ORDER],
        "routine_dtp1_eligibility": _routine_weight_vector(ROUTINE_DTP1_ELIGIBILITY_BY_AGE),
        "routine_dtp3_eligibility": _routine_weight_vector(ROUTINE_DTP3_ELIGIBILITY_BY_AGE),
        "routine_delay_age_weight": _routine_weight_vector(ROUTINE_DELAY_AGE_WEIGHT_BY_AGE, upper=3.0),
        "routine_delay_sensitivity": float(routine_delay_sensitivity),
        "age_waning_years": [float(AGE_WANING_YEARS[age]) for age in AGE_ORDER],
        "booster_reach_modifier": vaccine_program["booster_reach_modifier"],
        "vaccine_waning_age_modifier": vaccine_program["vaccine_waning_age_modifier"],
        "base_ve_infection": float(vaccine_program["base_ve_infection"]),
        "base_ve_disease": float(vaccine_program["base_ve_disease"]),
        "N_month_obs": int(len(fit_monthly)),
        "month_obs_idx": fit_monthly["month_index"].astype(int).tolist() if not fit_monthly.empty else [],
        "y_month": fit_monthly["observed_cases"].round().clip(lower=0).astype(int).tolist() if not fit_monthly.empty else [],
        "month_obs_weight": month_obs_weight,
        "N_month_ppc_obs": int(len(ppc_monthly)),
        "month_ppc_obs_idx": ppc_monthly["month_index"].astype(int).tolist() if not ppc_monthly.empty else [],
        "y_month_ppc": ppc_monthly["observed_cases"].round().clip(lower=0).astype(int).tolist() if not ppc_monthly.empty else [],
        "N_age_obs": int(len(age_obs)),
        "age_obs_year_idx": age_obs["year_index"].astype(int).tolist() if not age_obs.empty else [],
        "age_obs_age_idx": age_obs["age_index"].astype(int).tolist() if not age_obs.empty else [],
        "y_age": age_obs["observed_cases"].round().clip(lower=0).astype(int).tolist() if not age_obs.empty else [],
        "age_obs_weight": age_obs_weights,
        "log_beta_prior_mean": init["log_beta"],
        "log_reporting_rate_prior_mean": absolute_reporting_prior["absolute_reporting_rate_prior_log_mean"],
        "log_reporting_rate_prior_sd": absolute_reporting_prior["absolute_reporting_rate_prior_log_sd"],
        "contact_sensitivity_upper": contact_sensitivity_upper,
        "contact_sensitivity_prior_mean": init["contact_sensitivity"],
        "waning_rate_prior_mean": vaccine_program["waning_rate_prior_mean"],
        "waning_rate_prior_sd": vaccine_program["waning_rate_prior_sd"],
        "init_natural_share_prior_mean": init_natural_share_prior_mean,
        "init_vaccine_recent_fraction_prior_mean": init["init_vaccine_recent_fraction"],
        "init_natural_recent_fraction_prior_mean": init["init_natural_recent_fraction"],
        "log_post_transmission_prior_mean": init["log_post_transmission_multiplier"],
        "log_post_transmission_prior_sd": float(log_post_transmission_prior_sd),
        "rebound_peak_month_prior_mean": float(rebound_peak_month_prior_mean),
        "pathogen_pressure_scale_prior_mean": float(pathogen_pressure_scale_prior_mean),
        "pathogen_pressure_scale_prior_sd": float(pathogen_pressure_scale_prior_sd),
        "outbreak_seeding_gain_prior_mean": float(outbreak_seeding_gain_prior_mean),
        "outbreak_seeding_gain_prior_log_sd": float(outbreak_seeding_gain_prior_log_sd),
        "reporting_prior_log_mean": reporting_prior["reporting_prior_log_mean"],
        "reporting_prior_log_sd": reporting_prior["reporting_prior_log_sd"],
        "nb_phi_prior_mean": float(NB_DISPERSION),
        "nb_phi_month_prior_mean": float(NB_DISPERSION),
        "nb_phi_age_prior_mean": 10.0,
    }
    metadata = {
        "iso3": context.iso3,
        "country": context.country,
        "model2b_output_version": MODEL2B_OUTPUT_VERSION,
        "model2b_attribution_layers": (
            "primary_susceptibility:low_circulation_missed_boosting,routine_immunization_disruption,"
            "vaccine_waning_booster_gap,birth_cohort_demography;"
            "transmission_context_audit:post_2023_transmission_shift,excess_epidemic_pressure,"
            "outbreak_ignition_pulse;observation_secondary:reporting_testing_shift"
        ),
        "model2b_endemic_background_assumption": (
            "endemic_background_circulation_is_always_on_and_not_a_primary_shapley_factor"
        ),
        "model2b_primary_denominator_label": (
            "full_minus_susceptibility_neutral_with_fitted_background_true_infections_2023_2025"
        ),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "fit_end_year": int(fit_end_year),
        "n_months": int(len(grid)),
        "n_month_observations": int(data["N_month_obs"]),
        "n_month_ppc_observations": int(data["N_month_ppc_obs"]),
        "n_age_observations": int(data["N_age_obs"]),
        "age_groups": ";".join(AGE_ORDER),
        "model2b_age_state_transition_method": (
            "monthly_birth_entry_age_group_aging_and_wpp_net_demographic_adjustment_cohort"
        ),
        "model2b_initial_immunity_method": (
            "posterior_initial_immunity_shares_with_silent_exposure_dtp1_initiation_and_dtp3_completion_priors"
        ),
        "pre_mean_dtp1": float(np.clip(max(pre_mean_dtp1, context.pre_mean_dtp3), 0.01, 0.99)),
        "pre_mean_dtp3": float(np.clip(context.pre_mean_dtp3, 0.01, 0.99)),
        "pre_mean_dtp1_to_dtp3_dropout": float(
            max(0.0, np.clip(max(pre_mean_dtp1, context.pre_mean_dtp3), 0.01, 0.99) - np.clip(context.pre_mean_dtp3, 0.01, 0.99))
        ),
        "model2b_contact_intensity_method": (
            "oxcgrt_policy_stringency_2020_2022_plus_decaying_low_circulation_memory_through_2024"
        ),
        "model2c_contact_sensitivity_upper": float(contact_sensitivity_upper),
        "model2c_reporting_prior_sensitivity_mode": reporting_prior["reporting_prior_sensitivity_mode"],
        "model2c_waning_prior_sensitivity_mode": vaccine_program["waning_prior_sensitivity_mode"],
        "model2c_month_weight_mode": month_weight_mode,
        "model2c_age_observation_mode": age_observation_mode,
        "vaccine_type": vaccine_program["vaccine_type"],
        "adolescent_booster": bool(vaccine_program["adolescent_booster"]),
        "maternal_program_context": bool(vaccine_program["maternal_program_context"]),
        "contact_mode_requested": contact_mode_requested,
        "model2b_contact_matrix_source": contact_matrix_source,
        "model2b_contact_matrix_fallback": bool(contact_mode_requested == "full" and contact_matrix_source.startswith("rank_one_fallback")),
        "model2b_waning_rate_prior_mean": float(vaccine_program["waning_rate_prior_mean"]),
        "model2b_waning_rate_prior_sd": float(vaccine_program["waning_rate_prior_sd"]),
        "direct_testing_denominator_evidence": bool(evidence["direct_testing_denominator_evidence"]),
        "case_definition_reporting_practice_evidence": bool(evidence["case_definition_reporting_practice_evidence"]),
        "severe_outcome_evidence": bool(evidence["severe_outcome_evidence"]),
        "serology_evidence": bool(evidence["serology_evidence"]),
        "pathogen_pressure_evidence": bool(evidence["pathogen_pressure_evidence"]),
        "reporting_prior_mean": reporting_prior["reporting_prior_mean"],
        "reporting_prior_lower": reporting_prior["reporting_prior_lower"],
        "reporting_prior_upper": reporting_prior["reporting_prior_upper"],
        "reporting_prior_log_sd": reporting_prior["reporting_prior_log_sd"],
        "direct_testing_support": bool(reporting_prior["direct_testing_support"]),
        "reporting_prior_support_level": reporting_prior["reporting_prior_support_level"],
        "reporting_prior_source_ids": reporting_prior["reporting_prior_source_ids"],
        "reporting_prior_method": reporting_prior["reporting_prior_method"],
        "absolute_reporting_rate_prior_mean": absolute_reporting_prior["absolute_reporting_rate_prior_mean"],
        "absolute_reporting_rate_prior_lower": absolute_reporting_prior["absolute_reporting_rate_prior_lower"],
        "absolute_reporting_rate_prior_upper": absolute_reporting_prior["absolute_reporting_rate_prior_upper"],
        "absolute_reporting_rate_prior_log_sd": absolute_reporting_prior["absolute_reporting_rate_prior_log_sd"],
        "absolute_reporting_rate_prior_method": absolute_reporting_prior["absolute_reporting_rate_prior_method"],
        "gbd_incidence_per_100k_for_reporting_anchor": absolute_reporting_prior[
            "gbd_incidence_per_100k_for_reporting_anchor"
        ],
        "reported_to_gbd_incidence_ratio_for_reporting_anchor": absolute_reporting_prior[
            "reported_to_gbd_incidence_ratio_for_reporting_anchor"
        ],
        "model2c_silent_exposure_floor_monthly": silent_exposure_floor["silent_exposure_floor_monthly"],
        "model2c_silent_exposure_floor_min_monthly": silent_exposure_floor["silent_exposure_floor_min_monthly"],
        "model2c_silent_exposure_floor_max_monthly": silent_exposure_floor["silent_exposure_floor_max_monthly"],
        "model2c_silent_exposure_gbd_multiplier": silent_exposure_floor["silent_exposure_gbd_multiplier"],
        "model2c_silent_exposure_gbd_incidence_per_100k": silent_exposure_floor[
            "silent_exposure_gbd_incidence_per_100k"
        ],
        "model2c_silent_exposure_gbd_floor_monthly": silent_exposure_floor[
            "silent_exposure_gbd_floor_monthly"
        ],
        "model2c_silent_exposure_serology_recent_infection_percent": silent_exposure_floor[
            "silent_exposure_serology_recent_infection_percent"
        ],
        "model2c_silent_exposure_serology_floor_monthly": silent_exposure_floor[
            "silent_exposure_serology_floor_monthly"
        ],
        "model2c_silent_exposure_floor_method": silent_exposure_floor["silent_exposure_floor_method"],
        "model2c_natural_exposure_contact_sensitivity": float(natural_exposure_contact_sensitivity),
        "model2c_natural_exposure_contact_sensitivity_method": (
            "fixed_low_circulation_natural_exposure_path_decoupled_from_fitted_infection_contact_sensitivity"
        ),
        "model2c_initial_natural_share_prior_mean": float(init_natural_share_prior_mean),
        "model2c_initial_natural_share_prior_method": (
            "max_of_reported_pre_incidence_and_two_year_silent_exposure_stock_anchor"
        ),
        "model2c_routine_dtp1_eligibility": ";".join(
            f"{age}:{weight:g}" for age, weight in zip(AGE_ORDER, data["routine_dtp1_eligibility"])
        ),
        "model2c_routine_dtp3_eligibility": ";".join(
            f"{age}:{weight:g}" for age, weight in zip(AGE_ORDER, data["routine_dtp3_eligibility"])
        ),
        "model2c_routine_flow_method": (
            "monthly_birth_cohort_dtp1_dtp3_eligibility_with_limited_1_4y_catchup_and_delay_debt"
        ),
        "model2c_routine_delay_sensitivity": float(routine_delay_sensitivity),
        "model2c_routine_delay_age_weight": ";".join(
            f"{age}:{weight:g}" for age, weight in zip(AGE_ORDER, data["routine_delay_age_weight"])
        ),
        "rebound_peak_month_prior_mean": float(rebound_peak_month_prior_mean),
        "model2b_initial_beta": float(math.exp(init["log_beta"])),
        "model2b_initial_reporting_rate": float(math.exp(init["log_reporting_rate"])),
        "model2b_initial_post_transmission_multiplier": float(math.exp(init["log_post_transmission_multiplier"])),
        "model2b_initial_post_reporting_multiplier": float(math.exp(init["log_post_reporting_multiplier"])),
        "model2b_initial_rebound_peak_month": float(init["rebound_peak_month"]),
        "model2b_initial_rebound_width": float(init["rebound_width"]),
        "model2b_initial_rebound_asymmetry": float(init["rebound_asymmetry"]),
        "model2b_initial_rebound_release_gain": float(init["rebound_release_gain"]),
        "model2b_initial_secondary_rebound_delay": float(init["secondary_rebound_delay"]),
        "model2b_initial_secondary_rebound_width": float(init["secondary_rebound_width"]),
        "model2b_initial_secondary_rebound_release_gain": float(init["secondary_rebound_release_gain"]),
        "model2b_initial_outbreak_seeding_gain": float(init["outbreak_seeding_gain"]),
        "model2b_initial_outbreak_ignition_pulse_gain": float(init["outbreak_seeding_gain"]),
        "model2b_initial_seasonal_strength": float(init["seasonal_strength"]),
        "model2b_initial_natural_boosting_efficiency": float(init["natural_boosting_efficiency"]),
        "model2b_initial_pathogen_pressure_scale": float(init["pathogen_pressure_scale"]),
        "model2b_initial_excess_epidemic_pressure_scale": float(init["pathogen_pressure_scale"]),
        "model2b_log_post_transmission_prior_sd": float(log_post_transmission_prior_sd),
        "model2b_pathogen_pressure_scale_prior_mean": float(pathogen_pressure_scale_prior_mean),
        "model2b_pathogen_pressure_scale_prior_sd": float(pathogen_pressure_scale_prior_sd),
        "model2b_outbreak_seeding_gain_prior_mean": float(outbreak_seeding_gain_prior_mean),
        "model2b_outbreak_seeding_gain_prior_log_sd": float(outbreak_seeding_gain_prior_log_sd),
        "model2b_initial_epidemic_feedback_gain": float(init["epidemic_feedback_gain"]),
        "model2b_initial_endemic_reseed_pressure": float(init["endemic_reseed_pressure"]),
        "model2b_initial_endemic_background_force": float(init["endemic_reseed_pressure"]),
        "model2b_initial_epidemic_pressure_retention": float(init["epidemic_pressure_retention"]),
        "model2b_initial_post_reporting_peak_gain": float(init["post_reporting_peak_gain"]),
        "model2b_initial_post_reporting_trend": float(init["post_reporting_trend"]),
        "model2b_initial_nb_phi_month": float(init["nb_phi_month"]),
        "model2b_initial_nb_phi_age": float(init["nb_phi_age"]),
        "model2b_initial_natural_share": float(init["init_natural_share"]),
        "model2b_initial_vaccine_recent_fraction": float(init["init_vaccine_recent_fraction"]),
        "model2b_initial_natural_recent_fraction": float(init["init_natural_recent_fraction"]),
        "model2b_age_aging_rate_min": float(np.nanmin(data["age_aging_rate"])),
        "model2b_age_aging_rate_max": float(np.nanmax(data["age_aging_rate"])),
        "model2b_monthly_births_min": float(np.nanmin(data["monthly_births"])),
        "model2b_monthly_births_max": float(np.nanmax(data["monthly_births"])),
        "model2b_seasonal_multiplier_min": float(np.nanmin(grid["seasonal_multiplier"])),
        "model2b_seasonal_multiplier_max": float(np.nanmax(grid["seasonal_multiplier"])),
        "model2b_periodic_multiplier_min": float(np.nanmin(grid["periodic_multiplier"])),
        "model2b_periodic_multiplier_max": float(np.nanmax(grid["periodic_multiplier"])),
        "model2b_pathogen_pressure_prior_min": float(np.nanmin(grid["pathogen_pressure_prior"])),
        "model2b_pathogen_pressure_prior_max": float(np.nanmax(grid["pathogen_pressure_prior"])),
        "model2b_month_obs_weight_min": float(np.nanmin(month_obs_weight)) if month_obs_weight else 1.0,
        "model2b_month_obs_weight_mean": float(np.nanmean(month_obs_weight)) if month_obs_weight else 1.0,
        "model2b_month_obs_weight_max": float(np.nanmax(month_obs_weight)) if month_obs_weight else 1.0,
        "model2b_age_obs_weight_min": float(np.nanmin(age_obs_weights)) if age_obs_weights else 1.0,
        "model2b_age_obs_weight_mean": float(np.nanmean(age_obs_weights)) if age_obs_weights else 1.0,
        "model2b_age_obs_weight_max": float(np.nanmax(age_obs_weights)) if age_obs_weights else 1.0,
        "model2b_note": (
            "Bayesian transmission scenario-replay data built directly from source inputs; the "
            "dynamic bridge includes pre-pandemic periodic calibration, monthly seasonality, "
            "age-stratified susceptible, vaccine-protected, vaccine-waned, natural-immunity, "
            "exposed and infectious states. It now uses monthly birth entry, age-group aging flows and "
            "a recorded WPP net-demographic adjustment to keep cohort state totals aligned with annual "
            "population denominators. The initial model-window immune state is estimated with bounded posterior "
            "shares anchored by DTP1 initiation, DTP3 completion, reported pre-mean incidence and silent-exposure evidence. "
            "Pandemic contact reductions create missed natural exposure through reduced immune reinforcement and recent-natural "
            "immunity recharge rather than through an external transmission multiplier, and routine immunization disruption acts through "
            "birth-cohort DTP1 initiation, DTP1-to-DTP3 completion flow and the susceptible pool rather than a coverage multiplier on "
            "lambda. Contact/circulation suppression uses OxCGRT policy intensity during "
            "2020-2022 plus a decaying low-circulation memory through 2024 so the fitted "
            "state trajectory can represent delayed pertussis re-circulation after policy "
            "relaxation. The contact structure uses the requested Model 2C contact mode: "
            f"{contact_mode_requested} with source {contact_matrix_source}. Monthly cases use robust "
            "peak-balanced likelihood weights: zero or near-zero reporting months are not allowed to dominate "
            "the dynamic state, while post-2023 resurgence peak months receive extra objective weight so the fitted "
            "mean is judged against outbreak peak shape rather than only against average endemic months. A narrow "
            "post-2023 reporting peak gain is retained in the observation-process layer so abrupt surveillance or "
            "notification concentration at outbreak maxima is not forced into annual transmission intensity. Monthly "
            "reported cases and age-allocated cases use separate negative-binomial observation "
            "dispersion parameters with a permissive over-dispersion range for heterogeneous "
            "pertussis surveillance data. "
            "The absolute reporting-rate scale is anchored with a country-level GBD-incidence-to-"
            "reported-baseline empirical-Bayes prior only when direct relative detection evidence "
            "is unavailable; countries with direct testing or reporting-process evidence use that "
            "evidence for the relative post-pandemic reporting layer and retain a weak absolute "
            "reporting-rate prior. This prevents latent true-infection burden from being identified "
            "only by the product of transmission and reporting parameters while avoiding conflicts "
            "between GBD burden anchors and direct observation-process evidence. The natural exposure "
            "recharge floor is separately anchored to country-level GBD incidence and recent-infection "
            "serology where available, with a conservative minimum monthly silent-exposure floor, so naturally "
            "acquired immunity is not mechanically erased in countries with sparse reported pertussis notifications. Natural exposure "
            "boosting and recharge use a fixed low-circulation natural-exposure sensitivity path that is audited in the Stan "
            "data, while infection pressure retains the posterior-fitted contact sensitivity. "
            "aP/wP schedule metadata shifts the vaccine waning prior and booster metadata modifies "
            "age-specific vaccine reach/waning modifiers; maternal programme metadata is retained as "
            "audit context only. Post-peak 2026 rows can be used as posterior predictive checks while being excluded from "
            "the 2023-2025 scenario-summary period."
        ),
    }
    return data, metadata


def write_model2b_stan_data(
    context: CountryContext,
    inputs: dict[str, pd.DataFrame],
    initial_params: dict[str, float] | None = None,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = build_model2b_stan_data(context, inputs, initial_params)
    contact_mode = str(metadata.get("contact_mode_requested", "rank_one"))
    if contact_mode == "rank_one":
        suffix = "_rank_one"
    elif contact_mode == "full":
        suffix = ""
    else:
        suffix = f"_{contact_mode}"
    path = output_dir / f"model2b_stan_data_{context.iso3}{suffix}.json"
    payload = json.dumps(_json_ready(data), indent=2)
    path.write_text(payload, encoding="utf-8")
    if contact_mode == "full":
        (output_dir / f"model2b_stan_data_{context.iso3}_full.json").write_text(payload, encoding="utf-8")
    return path, data, metadata


def parameter_bounds_for_model2b() -> dict[str, tuple[float, float]]:
    contact_sensitivity_upper = _contact_sensitivity_upper()
    return {
        "contact_sensitivity": (0.0, contact_sensitivity_upper),
        "waning_rate": PARAMETER_BOUNDS["waning_rate"],
        "post_transmission_multiplier": (0.50, 2.50),
        "rebound_release_gain": (0.0, 4.0),
        "outbreak_seeding_gain": (0.0, 0.30),
        "natural_boosting_efficiency": (0.0, 1.0),
        "pathogen_pressure_scale": (0.0, 2.5),
        "epidemic_feedback_gain": (0.0, 25.0),
        "endemic_reseed_pressure": (0.0, 0.08),
        "epidemic_pressure_retention": (0.15, 0.85),
    }
