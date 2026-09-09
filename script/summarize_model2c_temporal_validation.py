#!/usr/bin/env python3
"""Summarise the tagged fit-through-2022 Model 2C temporal validation.

The posterior predictive file contains one row per draw and month.  This
script collapses those draws to month-level predictive intervals, then reports
proper interval scores, empirical coverage by forecast horizon, and annual
peak-timing error.  It never reads from or writes to the frozen primary fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "model2c_posterior_predictive_temporal_fit_2022.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "model2c_temporal_validation"
INTERVALS = ((0.50, 0.25, 0.75), (0.80, 0.10, 0.90), (0.95, 0.025, 0.975))


def interval_score(y: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    """Gneiting-Raftery central interval score (lower is better)."""
    width = upper - lower
    return width + (2.0 / alpha) * (lower - y).clip(lower=0) + (2.0 / alpha) * (y - upper).clip(lower=0)


def collapse_draws(draws: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    required = {
        "iso3", "country", "draw", "year", "month", "observed_cases",
        "reported_cases_predictive", "fit_role",
    }
    missing = sorted(required.difference(draws.columns))
    if missing:
        raise ValueError(f"Missing temporal-validation columns: {', '.join(missing)}")
    held = draws.loc[
        draws["year"].between(start_year, end_year)
        & draws["fit_role"].astype(str).ne("fit_likelihood")
    ].copy()
    if held.empty:
        raise ValueError(f"No held-out rows found for {start_year}-{end_year}.")
    keys = ["iso3", "country", "year", "month"]
    observed_counts = held.groupby(keys, observed=True)["observed_cases"].nunique(dropna=True)
    if observed_counts.gt(1).any():
        raise ValueError("Observed cases vary across posterior draws for at least one month.")
    quantiles = sorted({0.5, *(q for _, lo, hi in INTERVALS for q in (lo, hi))})
    grouped = held.groupby(keys, observed=True)
    result = grouped["reported_cases_predictive"].quantile(quantiles).unstack(-1).reset_index()
    result = result.rename(columns={q: f"pred_q{str(q).replace('.', '_')}" for q in quantiles})
    observed = grouped["observed_cases"].first().reset_index(name="observed_cases")
    result = result.merge(observed, on=keys, validate="one_to_one")
    result["horizon_month"] = 12 * (result["year"] - start_year) + result["month"]
    result["horizon_band"] = pd.cut(
        result["horizon_month"], bins=[0, 12, 24, 36, np.inf],
        labels=["months_01_12", "months_13_24", "months_25_36", "months_37_plus"],
    ).astype(str)
    median = result["pred_q0_5"]
    result["absolute_error"] = (result["observed_cases"] - median).abs()
    result["absolute_log1p_error"] = (
        np.log1p(result["observed_cases"]) - np.log1p(median.clip(lower=0))
    ).abs()
    weighted = 0.5 * result["absolute_error"]
    for coverage, lo_q, hi_q in INTERVALS:
        alpha = 1.0 - coverage
        lower = result[f"pred_q{str(lo_q).replace('.', '_')}"]
        upper = result[f"pred_q{str(hi_q).replace('.', '_')}"]
        result[f"covered_{int(100 * coverage)}"] = (
            result["observed_cases"].between(lower, upper).astype("boolean")
            .where(result["observed_cases"].notna())
        )
        score = interval_score(result["observed_cases"], lower, upper, alpha)
        result[f"interval_score_{int(100 * coverage)}"] = score
        weighted = weighted + (alpha / 2.0) * score
    result["weighted_interval_score"] = weighted / (len(INTERVALS) + 0.5)
    return result.sort_values(keys).reset_index(drop=True)


def validation_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    groupings = [
        ("country_all_horizons", ["iso3", "country"]),
        ("country_horizon_band", ["iso3", "country", "horizon_band"]),
        ("pooled_horizon_band", ["horizon_band"]),
        ("pooled_all_horizons", []),
    ]
    for scope, keys in groupings:
        iterator = [((), monthly)] if not keys else monthly.groupby(keys, observed=True, dropna=False)
        for raw_key, group in iterator:
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            row: dict[str, object] = {"scope": scope, "iso3": "ALL", "country": "All countries", "horizon_band": "all"}
            row.update(dict(zip(keys, values)))
            observed = group["observed_cases"].notna()
            row.update(
                {
                    "n_months": len(group),
                    "n_observed_months": int(observed.sum()),
                    "n_missing_months": int((~observed).sum()),
                    "mean_absolute_error": group["absolute_error"].mean(),
                    "mean_absolute_log1p_error": group["absolute_log1p_error"].mean(),
                    "mean_weighted_interval_score": group["weighted_interval_score"].mean(),
                    "coverage_50": group.loc[observed, "covered_50"].mean(),
                    "coverage_80": group.loc[observed, "covered_80"].mean(),
                    "coverage_95": group.loc[observed, "covered_95"].mean(),
                }
            )
            records.append(row)
    return pd.DataFrame.from_records(records)


def peak_timing_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (iso3, country, year), group in monthly.groupby(["iso3", "country", "year"], observed=True):
        # Compare peaks on the same observed window; an unobserved month cannot
        # supply a verifiable annual peak or count as an interval miss.
        group = group.loc[group["observed_cases"].notna()]
        if group.empty:
            continue
        observed_peak = int(group.loc[group["observed_cases"].idxmax(), "month"])
        predicted_peak = int(group.loc[group["pred_q0_5"].idxmax(), "month"])
        records.append(
            {
                "iso3": iso3,
                "country": country,
                "year": int(year),
                "n_observed_months": len(group),
                "complete_observed_year": len(group) == 12,
                "observed_peak_month": observed_peak,
                "posterior_median_peak_month": predicted_peak,
                "signed_peak_timing_error_months": predicted_peak - observed_peak,
                "absolute_peak_timing_error_months": abs(predicted_peak - observed_peak),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--monthly-input", type=Path,
                        help="Resummarise archived month-level predictions without reading or regenerating draws.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be at least --start-year")
    usecols = [
        "iso3", "country", "draw", "year", "month", "observed_cases",
        "reported_cases_predictive", "fit_role",
    ]
    if args.monthly_input:
        monthly = pd.read_csv(args.monthly_input)
        monthly = monthly.loc[monthly.year.between(args.start_year, args.end_year)].copy()
        for coverage, lo, hi in INTERVALS:
            lower = monthly[f"pred_q{str(lo).replace('.', '_')}"]
            upper = monthly[f"pred_q{str(hi).replace('.', '_')}"]
            monthly[f"covered_{int(100 * coverage)}"] = (
                monthly.observed_cases.between(lower, upper).astype("boolean")
                .where(monthly.observed_cases.notna())
            )
    else:
        draws = pd.read_csv(args.input, usecols=usecols)
        monthly = collapse_draws(draws, args.start_year, args.end_year)
    summary = validation_summary(monthly)
    peaks = peak_timing_summary(monthly)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.output_dir / "monthly_temporal_validation.csv", index=False)
    summary.to_csv(args.output_dir / "temporal_validation_summary.csv", index=False)
    peaks.to_csv(args.output_dir / "annual_peak_timing_validation.csv", index=False)
    print(f"Wrote temporal validation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
