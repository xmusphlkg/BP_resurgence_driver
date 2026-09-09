from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def peak_value_and_year(subset: pd.DataFrame, value_column: str) -> tuple[float, float]:
    if subset.empty or value_column not in subset.columns:
        return np.nan, np.nan
    values = pd.to_numeric(subset[value_column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return np.nan, np.nan
    idx = valid.idxmax()
    peak_year = pd.to_numeric(pd.Series([subset.loc[idx, "year"]]), errors="coerce").iloc[0]
    return float(values.loc[idx]), float(peak_year) if pd.notna(peak_year) else np.nan


def summarize_period(annual: pd.DataFrame, start: int, end: int) -> dict[str, float]:
    subset = annual.loc[annual["year"].between(start, end)].copy()
    subset = subset.sort_values("year")
    summary = {
        "reported_cases": float(subset["reported_cases_predicted"].sum()),
        "true_infections": float(subset["true_infections_predicted"].sum()),
        "symptomatic_infections": float(subset["symptomatic_infections_predicted"].sum()),
        "reported_infant_under_6m": float(subset["reported_infant_under_6m_predicted"].sum()),
        "reported_under_1y": float(subset["reported_under_1y_predicted"].sum()),
    }
    peak_sources = {
        "reported_cases": "reported_cases_predicted",
        "true_infections": "true_infections_predicted",
        "symptomatic_infections": "symptomatic_infections_predicted",
        "reported_infant_under_6m": "reported_infant_under_6m_predicted",
        "reported_under_1y": "reported_under_1y_predicted",
    }
    for metric, column in peak_sources.items():
        peak_value, peak_year = peak_value_and_year(subset, column)
        summary[f"peak_{metric}"] = peak_value
        summary[f"peak_{metric}_year"] = peak_year
    return summary


def observed_cases_period(context: Any, start: int, end: int) -> float:
    subset = context.annual.loc[context.annual["year"].between(start, end)].copy()
    subset["cases_total"] = pd.to_numeric(subset["cases_total"], errors="coerce")
    return float(subset["cases_total"].sum())


def log_rmse_against_observed(annual: pd.DataFrame, start: int, end: int) -> float:
    subset = annual.loc[annual["year"].between(start, end)].dropna(subset=["observed_cases", "reported_cases_predicted"]).copy()
    if subset.empty:
        return np.nan
    residual = np.log1p(subset["reported_cases_predicted"].clip(lower=0.0)) - np.log1p(subset["observed_cases"].clip(lower=0.0))
    return float(np.sqrt(np.nanmean(residual.to_numpy(dtype=float) ** 2)))


def summarize_observation_likelihood(likelihood: pd.DataFrame) -> pd.DataFrame:
    if likelihood.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "component",
                "n_observations",
                "effective_n_observations",
                "nb_log_likelihood",
                "weighted_nb_log_likelihood",
                "mean_nb_log_likelihood",
                "mean_weighted_nb_log_likelihood",
                "rmse_log_cases",
                "weighted_rmse_log_cases",
                "mean_age_allocation_weight",
            ]
        )
    likelihood = likelihood.copy()
    if "age_allocation_weight" not in likelihood:
        likelihood["age_allocation_weight"] = 1.0
    if "weighted_nb_log_likelihood" not in likelihood:
        likelihood["weighted_nb_log_likelihood"] = likelihood["nb_log_likelihood"] * likelihood["age_allocation_weight"]
    if "weighted_log_residual_square" not in likelihood:
        likelihood["weighted_log_residual_square"] = likelihood["log_residual"].astype(float) ** 2 * likelihood["age_allocation_weight"]
    summary = likelihood.groupby(["iso3", "country", "component"], as_index=False).agg(
        n_observations=("nb_log_likelihood", "count"),
        effective_n_observations=("age_allocation_weight", "sum"),
        nb_log_likelihood=("nb_log_likelihood", "sum"),
        weighted_nb_log_likelihood=("weighted_nb_log_likelihood", "sum"),
        mean_nb_log_likelihood=("nb_log_likelihood", "mean"),
        mean_age_allocation_weight=("age_allocation_weight", "mean"),
        weighted_log_residual_square=("weighted_log_residual_square", "sum"),
        rmse_log_cases=("log_residual", lambda values: float(np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2)))),
    )
    summary["mean_weighted_nb_log_likelihood"] = summary["weighted_nb_log_likelihood"] / summary["effective_n_observations"].replace({0: np.nan})
    summary["weighted_rmse_log_cases"] = np.sqrt(
        summary["weighted_log_residual_square"] / summary["effective_n_observations"].replace({0: np.nan})
    )
    return summary.loc[
        :,
        [
            "iso3",
            "country",
            "component",
            "n_observations",
            "effective_n_observations",
            "nb_log_likelihood",
            "weighted_nb_log_likelihood",
            "mean_nb_log_likelihood",
            "mean_age_allocation_weight",
            "rmse_log_cases",
            "mean_weighted_nb_log_likelihood",
            "weighted_rmse_log_cases",
        ],
    ]
