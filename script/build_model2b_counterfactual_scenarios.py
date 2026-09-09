#!/usr/bin/env python3
"""Build fitted-posterior Model 2B counterfactual scenario trajectories.

This script does not refit Stan and does not recompute exact Shapley values.
It replays the fitted transmission-state equations under scenario switches
that remove one primary mechanism at a time.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics.bayes.model2b_attribution import (  # noqa: E402
    MODEL2B_OUTPUT_VERSION,
    model2b_counterfactual_scenarios,
)
from pertussis_global_dynamics.paths import FIGURE_DATA, OUTPUTS, ROOT  # noqa: E402


MODEL2B_DEFAULT_PILOT_ISO3 = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            return max(minimum, int(raw))
        except ValueError:
            return default
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _manuscript_min_draws() -> int:
    return _env_int("MODEL2B_MIN_MANUSCRIPT_DRAWS", 2000)


def _complete_draw_input(draws: pd.DataFrame) -> tuple[bool, str]:
    if draws.empty or not {"iso3", "draw"}.issubset(draws.columns):
        return False, "missing_posterior_draws"
    expected_iso = set(MODEL2B_DEFAULT_PILOT_ISO3)
    observed_iso = set(draws["iso3"].dropna().astype(str))
    missing = sorted(expected_iso.difference(observed_iso))
    if missing:
        return False, "missing_countries_" + ",".join(missing)
    draw_counts = draws.loc[draws["iso3"].isin(expected_iso)].groupby("iso3")["draw"].nunique()
    min_draws = _manuscript_min_draws()
    too_few = sorted(draw_counts.loc[draw_counts.lt(min_draws)].index.astype(str).tolist())
    if too_few:
        return False, f"posterior_draws_below_{min_draws}_" + ",".join(too_few)
    return True, "complete_nine_country_manuscript_output"


def _env_countries(name: str = "MODEL2B_COUNTRIES") -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()]


def _contact_mode_suffix() -> str:
    raw = os.environ.get("MODEL2B_CONTACT_MODE", "full").strip().lower().replace("-", "_")
    return "full" if raw == "full" else "rank_one"


def _first_existing(paths: list[Path], message: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = ", ".join(str(path.relative_to(ROOT)) for path in paths)
    raise SystemExit(f"{message}. Checked: {checked}")


def _mode_candidates(filename: str, contact_mode: str) -> list[Path]:
    path = OUTPUTS / filename
    stem = path.stem
    suffix = path.suffix
    suffixed = OUTPUTS / f"{stem}_{contact_mode}{suffix}"
    if contact_mode == "rank_one":
        return [suffixed]
    return [suffixed]


def _quantile(probability: float):
    def summarise(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return float(numeric.quantile(probability)) if not numeric.empty else float("nan")

    return summarise


def _full_model_predictive_intervals(contact_mode: str, countries: list[str]) -> pd.DataFrame:
    ppc_path = _first_existing(
        _mode_candidates("model2b_bayes_posterior_predictive.csv", contact_mode),
        "Missing Model 2B posterior predictive draws",
    )
    ppc = pd.read_csv(
        ppc_path,
        usecols=[
            "iso3",
            "country",
            "year",
            "month",
            "reported_cases_predictive",
            "model2b_output_version",
        ],
    )
    versions = set(ppc["model2b_output_version"].dropna().astype(str))
    if versions != {MODEL2B_OUTPUT_VERSION}:
        raise SystemExit(
            f"{ppc_path.relative_to(ROOT)} has Model 2B output version(s) "
            f"{sorted(versions)}; expected {MODEL2B_OUTPUT_VERSION}."
        )
    ppc = ppc.loc[ppc["iso3"].astype(str).isin(countries)].copy()
    ppc["year"] = pd.to_numeric(ppc["year"], errors="coerce")
    ppc["month"] = pd.to_numeric(ppc["month"], errors="coerce")
    ppc["reported_cases_predictive"] = pd.to_numeric(ppc["reported_cases_predictive"], errors="coerce")
    summary = (
        ppc.dropna(subset=["year", "month"])
        .groupby(["iso3", "country", "year", "month"], dropna=False, sort=False)
        .agg(
            reported_cases_predictive_median=("reported_cases_predictive", "median"),
            reported_cases_predictive_lower_95=("reported_cases_predictive", _quantile(0.025)),
            reported_cases_predictive_upper_95=("reported_cases_predictive", _quantile(0.975)),
        )
        .reset_index()
    )
    summary["year"] = summary["year"].astype(int)
    summary["month"] = summary["month"].astype(int)
    return summary


def _metadata_record(row: pd.Series) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}


def _stan_data_path(iso3: str, contact_mode: str) -> Path:
    if contact_mode == "rank_one":
        candidates = [OUTPUTS / f"model2b_stan_data_{iso3}_rank_one.json"]
    else:
        candidates = [OUTPUTS / f"model2b_stan_data_{iso3}_{contact_mode}.json"]
    return _first_existing(candidates, f"Missing Model 2B Stan data JSON for {iso3}")


def _run_country(payload: tuple[str, Path, pd.DataFrame, dict[str, Any], int]) -> pd.DataFrame:
    iso3, data_path, country_draws, metadata, max_draws = payload
    with data_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    scenarios = model2b_counterfactual_scenarios(data, metadata, country_draws, max_draws=max_draws)
    if scenarios.empty:
        raise SystemExit(f"Counterfactual scenario output is empty for {iso3}.")
    return scenarios


def main() -> None:
    contact_mode = _contact_mode_suffix()
    draws_path = _first_existing(
        _mode_candidates("model2b_bayes_posterior_draws.csv", contact_mode),
        "Missing Model 2B posterior draws",
    )
    metadata_path = _first_existing(
        _mode_candidates("model2b_stan_data_metadata.csv", contact_mode),
        "Missing Model 2B Stan metadata",
    )
    draws = pd.read_csv(draws_path)
    metadata = pd.read_csv(metadata_path)
    complete_output, completeness_status = _complete_draw_input(draws)
    if not complete_output and _env_bool("MODEL2B_REQUIRE_MANUSCRIPT_OUTPUTS", False):
        raise SystemExit(
            "Refusing to build manuscript counterfactual scenarios from incomplete Model 2B draws: "
            f"{completeness_status}."
        )
    missing = {"iso3", "country"}.difference(draws.columns)
    if missing:
        raise SystemExit(f"{draws_path.relative_to(ROOT)} is missing required columns: {', '.join(sorted(missing))}")
    missing = {"iso3", "country"}.difference(metadata.columns)
    if missing:
        raise SystemExit(f"{metadata_path.relative_to(ROOT)} is missing required columns: {', '.join(sorted(missing))}")

    requested = set(_env_countries())
    available = set(draws["iso3"].dropna().astype(str))
    countries = [iso3 for iso3 in metadata["iso3"].dropna().astype(str).tolist() if iso3 in available]
    if requested:
        countries = [iso3 for iso3 in countries if iso3 in requested]
    if not countries:
        raise SystemExit("No countries are available for Model 2B counterfactual scenario simulation.")

    max_draws = _env_int("MODEL2B_SCENARIO_MAX_DRAWS", 160)
    workers = min(
        len(countries),
        _env_int("MODEL2B_SCENARIO_COUNTRY_WORKERS", _env_int("MODEL2B_COUNTRY_WORKERS", os.cpu_count() or 1)),
    )
    payloads: list[tuple[str, Path, pd.DataFrame, dict[str, Any], int]] = []
    for iso3 in countries:
        metadata_row = metadata.loc[metadata["iso3"].astype(str).eq(iso3)].iloc[0]
        country_draws = draws.loc[draws["iso3"].astype(str).eq(iso3)].drop(columns=["iso3", "country"])
        data_path = _stan_data_path(iso3, contact_mode)
        payloads.append(
            (
                iso3,
                data_path,
                country_draws,
                _metadata_record(metadata_row),
                max_draws,
            )
        )

    if workers > 1:
        tables: list[pd.DataFrame | None] = [None for _ in payloads]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_country, payload): index for index, payload in enumerate(payloads)}
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                tables[index] = future.result()
                print(
                    f"Built Model 2B counterfactual scenarios for {payloads[index][0]} "
                    f"({completed}/{len(payloads)}).",
                    flush=True,
                )
        country_tables = [table for table in tables if table is not None]
    else:
        country_tables = []
        for completed, payload in enumerate(payloads, start=1):
            country_tables.append(_run_country(payload))
            print(
                f"Built Model 2B counterfactual scenarios for {payload[0]} "
                f"({completed}/{len(payloads)}).",
                flush=True,
            )

    combined = pd.concat(country_tables, ignore_index=True, sort=False)
    predictive = _full_model_predictive_intervals(contact_mode, countries)
    predictive_columns = [
        "reported_cases_predictive_median",
        "reported_cases_predictive_lower_95",
        "reported_cases_predictive_upper_95",
    ]
    combined = combined.merge(
        predictive,
        on=["iso3", "country", "year", "month"],
        how="left",
    )
    combined.loc[combined["scenario_id"].ne("full_model"), predictive_columns] = pd.NA
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS / f"model2b_counterfactual_scenario_monthly_{contact_mode}.csv"
    figure_path = FIGURE_DATA / f"figure_4_model2b_counterfactual_scenarios_{contact_mode}.csv"
    combined.to_csv(output_path, index=False)
    combined.to_csv(figure_path, index=False)
    if complete_output and contact_mode == "full":
        combined.to_csv(OUTPUTS / "model2b_counterfactual_scenario_monthly.csv", index=False)
        combined.to_csv(FIGURE_DATA / "figure_4_model2b_counterfactual_scenarios.csv", index=False)
    print(
        "Wrote Model 2B counterfactual scenario trajectories for "
        f"{len(countries)} countries and up to {max_draws} posterior draws per country "
        f"from {MODEL2B_OUTPUT_VERSION}."
    )


if __name__ == "__main__":
    main()
