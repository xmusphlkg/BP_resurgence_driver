from __future__ import annotations

from typing import Any

MODEL2C_STATE_START_YEAR = 2010
MODEL2C_STATE_END_YEAR = 2026
MODEL2C_MONTHLY_OBSERVATION_DEFAULT_START_YEAR = 2010
MODEL2C_MONTHLY_OBSERVATION_END_YEAR = 2025
MODEL2C_ANNUAL_AGE_OBSERVATION_START_YEAR = 2015
MODEL2C_ANNUAL_AGE_OBSERVATION_END_YEAR = 2025
# Backward-compatible aliases for older deterministic diagnostics. New Model 2C
# metadata should use the monthly/annual-age-specific fields above.
MODEL2C_OBSERVATION_START_YEAR = MODEL2C_ANNUAL_AGE_OBSERVATION_START_YEAR
MODEL2C_OBSERVATION_END_YEAR = MODEL2C_MONTHLY_OBSERVATION_END_YEAR
MODEL2C_FIT_END_YEAR = 2025
MODEL2C_SCENARIO_START_YEAR = 2023
MODEL2C_SCENARIO_END_YEAR = 2025
MODEL2C_BASELINE_START_YEAR = 2010
MODEL2C_BASELINE_END_YEAR = 2019

MODEL2C_WINDOW_SEMANTICS_NOTE = (
    "Model 2C carries the dynamic state and historical baseline from 2010. Monthly all-age "
    "reported-case likelihood rows use the source-specific observed monthly window, which starts "
    "as early as 2010 in countries with archived monthly observations and later where monthly "
    "observations begin later. Annual age-bin likelihood rows start in 2015 in the current source "
    "snapshot. Scenario replay summaries use 2023-2025, and observed 2026 months are retained as "
    "posterior predictive check context rather than as scenario-summary targets."
)


def _year_window_from_month_indices(data: dict[str, Any], key: str) -> tuple[int | None, int | None, int]:
    years = list(data.get("year", []))
    indices = list(data.get(key, []))
    observed_years: list[int] = []
    for raw_index in indices:
        try:
            index = int(raw_index) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(years):
            observed_years.append(int(years[index]))
    if not observed_years:
        return None, None, 0
    return min(observed_years), max(observed_years), len(indices)


def _year_window_from_year_indices(data: dict[str, Any], key: str) -> tuple[int | None, int | None, int]:
    years = sorted({int(year) for year in data.get("year", [])})
    indices = list(data.get(key, []))
    observed_years: list[int] = []
    for raw_index in indices:
        try:
            index = int(raw_index) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(years):
            observed_years.append(years[index])
    if not observed_years:
        return None, None, 0
    return min(observed_years), max(observed_years), len(indices)


def _month_index_count_for_year(data: dict[str, Any], key: str, target_year: int) -> int:
    years = list(data.get("year", []))
    count = 0
    for raw_index in data.get(key, []):
        try:
            index = int(raw_index) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(years) and int(years[index]) == target_year:
            count += 1
    return count


def model2c_stan_data_window_metadata(data: dict[str, Any]) -> dict[str, int | str | None]:
    monthly_start, monthly_end, monthly_n = _year_window_from_month_indices(data, "month_obs_idx")
    monthly_ppc_start, monthly_ppc_end, monthly_ppc_n = _year_window_from_month_indices(data, "month_ppc_obs_idx")
    ppc_2026_n = _month_index_count_for_year(data, "month_ppc_obs_idx", 2026)
    age_start, age_end, age_n = _year_window_from_year_indices(data, "age_obs_year_idx")
    return {
        "model2c_monthly_observation_start_year": monthly_start,
        "model2c_monthly_observation_end_year": monthly_end,
        "model2c_monthly_observation_n_months": monthly_n,
        "model2c_annual_age_observation_start_year": age_start,
        "model2c_annual_age_observation_end_year": age_end,
        "model2c_annual_age_observation_n_rows": age_n,
        "model2c_temporal_check_start_year": monthly_ppc_start,
        "model2c_temporal_check_end_year": monthly_ppc_end,
        "model2c_temporal_check_n_months": monthly_ppc_n,
        "model2c_2026_ppc_observed_start_year": 2026 if ppc_2026_n else None,
        "model2c_2026_ppc_observed_end_year": 2026 if ppc_2026_n else None,
        "model2c_2026_ppc_observed_n_months": ppc_2026_n,
    }


def model2c_window_metadata() -> dict[str, int | str]:
    return {
        "model2c_state_start_year": MODEL2C_STATE_START_YEAR,
        "model2c_state_end_year": MODEL2C_STATE_END_YEAR,
        "model2c_monthly_observation_default_start_year": MODEL2C_MONTHLY_OBSERVATION_DEFAULT_START_YEAR,
        "model2c_monthly_observation_end_year": MODEL2C_MONTHLY_OBSERVATION_END_YEAR,
        "model2c_annual_age_observation_start_year": MODEL2C_ANNUAL_AGE_OBSERVATION_START_YEAR,
        "model2c_annual_age_observation_end_year": MODEL2C_ANNUAL_AGE_OBSERVATION_END_YEAR,
        "model2c_observation_start_year": MODEL2C_MONTHLY_OBSERVATION_DEFAULT_START_YEAR,
        "model2c_observation_end_year": MODEL2C_MONTHLY_OBSERVATION_END_YEAR,
        "model2c_fit_end_year": MODEL2C_FIT_END_YEAR,
        "model2c_scenario_start_year": MODEL2C_SCENARIO_START_YEAR,
        "model2c_scenario_end_year": MODEL2C_SCENARIO_END_YEAR,
        "model2c_baseline_start_year": MODEL2C_BASELINE_START_YEAR,
        "model2c_baseline_end_year": MODEL2C_BASELINE_END_YEAR,
        "model2c_window_semantics": MODEL2C_WINDOW_SEMANTICS_NOTE,
    }
