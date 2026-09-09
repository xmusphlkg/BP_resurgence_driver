#!/usr/bin/env python3
"""Replay fitted Model 2C posterior draws under fixed counterfactual scenarios."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics.bayes.model2c_scenarios import (  # noqa: E402
    MODEL2C_OUTPUT_VERSION,
    model2c_counterfactual_outputs,
)
from pertussis_global_dynamics.paths import FIGURE_DATA, OUTPUTS, ROOT  # noqa: E402


MODEL2C_DEFAULT_PILOT_ISO3 = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")


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


def _env_countries(name: str = "MODEL2C_COUNTRIES") -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()]


def _contact_mode_suffix() -> str:
    raw = os.environ.get("MODEL2C_CONTACT_MODE", os.environ.get("MODEL2B_CONTACT_MODE", "full"))
    raw = raw.strip().lower().replace("-", "_")
    return "full" if raw == "full" else "rank_one"


def _output_tag() -> str:
    raw = os.environ.get("MODEL2C_OUTPUT_TAG", "").strip().lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")


def _output_suffix() -> str:
    tag = _output_tag()
    return tag if tag else _contact_mode_suffix()


def _manuscript_min_draws() -> int:
    return _env_int("MODEL2C_MIN_MANUSCRIPT_DRAWS", 2000)


def _complete_draw_input(draws: pd.DataFrame) -> tuple[bool, str]:
    if draws.empty or not {"iso3", "draw"}.issubset(draws.columns):
        return False, "missing_posterior_draws"
    expected_iso = set(MODEL2C_DEFAULT_PILOT_ISO3)
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


def _mode_candidates(filename: str, contact_mode: str) -> list[Path]:
    path = OUTPUTS / filename
    stem = path.stem
    suffix = path.suffix
    output_tag = _output_tag()
    suffix_mode = output_tag if output_tag else contact_mode
    return [OUTPUTS / f"{stem}_{suffix_mode}{suffix}"]


def _first_existing(paths: list[Path], message: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = ", ".join(str(path.relative_to(ROOT)) for path in paths)
    raise SystemExit(f"{message}. Checked: {checked}")


def _metadata_record(row: pd.Series) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}


def _stan_data_path(iso3: str, contact_mode: str) -> Path:
    suffix_mode = _output_suffix()
    return _first_existing(
        [OUTPUTS / f"model2c_stan_data_{iso3}_{suffix_mode}.json"],
        f"Missing Model 2C Stan data JSON for {iso3}",
    )


def _require_version(frame: pd.DataFrame, path: Path) -> None:
    if "model2c_output_version" not in frame.columns:
        raise SystemExit(f"{path.relative_to(ROOT)} is missing model2c_output_version.")
    versions = set(frame["model2c_output_version"].dropna().astype(str))
    if versions != {MODEL2C_OUTPUT_VERSION}:
        raise SystemExit(
            f"{path.relative_to(ROOT)} has Model 2C output version(s) "
            f"{sorted(versions)}; expected {MODEL2C_OUTPUT_VERSION}."
        )


def _run_country(
    payload: tuple[str, Path, pd.DataFrame, dict[str, Any], int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    iso3, data_path, country_draws, metadata, max_draws = payload
    with data_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    monthly, summary = model2c_counterfactual_outputs(data, metadata, country_draws, max_draws=max_draws)
    if monthly.empty or summary.empty:
        raise SystemExit(f"Model 2C scenario output is empty for {iso3}.")
    return monthly, summary


def main() -> None:
    contact_mode = _contact_mode_suffix()
    output_tag = _output_tag()
    suffix_mode = _output_suffix()
    draws_path = _first_existing(
        _mode_candidates("model2c_posterior_draws.csv", contact_mode),
        "Missing Model 2C posterior draws",
    )
    metadata_path = _first_existing(
        _mode_candidates("model2c_stan_data_metadata.csv", contact_mode),
        "Missing Model 2C Stan metadata",
    )
    draws = pd.read_csv(draws_path)
    metadata = pd.read_csv(metadata_path)
    _require_version(draws, draws_path)
    _require_version(metadata, metadata_path)

    complete_output, completeness_status = _complete_draw_input(draws)
    if not complete_output:
        print(
            "Model 2C scenario replay is not using a complete manuscript posterior: "
            f"{completeness_status}. Writing contact-mode-specific files only.",
            flush=True,
        )
    if not complete_output and _env_bool("MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS", False):
        raise SystemExit(
            "Refusing to build Model 2C manuscript scenarios from incomplete posterior draws: "
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
        raise SystemExit("No countries are available for Model 2C counterfactual scenario simulation.")

    max_draws = _env_int("MODEL2C_SCENARIO_MAX_DRAWS", 160)
    workers = min(
        len(countries),
        _env_int("MODEL2C_SCENARIO_COUNTRY_WORKERS", _env_int("MODEL2C_COUNTRY_WORKERS", os.cpu_count() or 1)),
    )
    payloads: list[tuple[str, Path, pd.DataFrame, dict[str, Any], int]] = []
    for iso3 in countries:
        metadata_row = metadata.loc[metadata["iso3"].astype(str).eq(iso3)].iloc[0]
        country_draws = draws.loc[draws["iso3"].astype(str).eq(iso3)].drop(columns=["iso3", "country"])
        payloads.append((iso3, _stan_data_path(iso3, contact_mode), country_draws, _metadata_record(metadata_row), max_draws))

    if workers > 1:
        monthly_tables: list[pd.DataFrame | None] = [None for _ in payloads]
        summary_tables: list[pd.DataFrame | None] = [None for _ in payloads]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_country, payload): index for index, payload in enumerate(payloads)}
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                monthly_tables[index], summary_tables[index] = future.result()
                print(
                    f"Built Model 2C counterfactual scenarios for {payloads[index][0]} "
                    f"({completed}/{len(payloads)}).",
                    flush=True,
                )
        monthly_parts = [table for table in monthly_tables if table is not None]
        summary_parts = [table for table in summary_tables if table is not None]
    else:
        monthly_parts = []
        summary_parts = []
        for completed, payload in enumerate(payloads, start=1):
            monthly, summary = _run_country(payload)
            monthly_parts.append(monthly)
            summary_parts.append(summary)
            print(
                f"Built Model 2C counterfactual scenarios for {payload[0]} "
                f"({completed}/{len(payloads)}).",
                flush=True,
            )

    monthly = pd.concat(monthly_parts, ignore_index=True, sort=False)
    summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    monthly_path = OUTPUTS / f"model2c_counterfactual_scenario_monthly_{suffix_mode}.csv"
    summary_path = OUTPUTS / f"model2c_counterfactual_scenario_summary_{suffix_mode}.csv"
    monthly.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)

    monthly_figure_path = FIGURE_DATA / f"figure_4_model2c_counterfactual_scenarios_{suffix_mode}.csv"
    summary_figure_path = FIGURE_DATA / f"figure_4_model2c_scenario_contrasts_{suffix_mode}.csv"
    monthly.to_csv(monthly_figure_path, index=False)
    summary.to_csv(summary_figure_path, index=False)
    if not output_tag and complete_output and contact_mode == "full":
        monthly.to_csv(OUTPUTS / "model2c_counterfactual_scenario_monthly.csv", index=False)
        summary.to_csv(OUTPUTS / "model2c_counterfactual_scenario_summary.csv", index=False)
        monthly.to_csv(FIGURE_DATA / "figure_4_model2c_counterfactual_scenarios.csv", index=False)
        summary.to_csv(FIGURE_DATA / "figure_4_model2c_scenario_contrasts.csv", index=False)

    print(
        "Wrote Model 2C scenario replay outputs for "
        f"{len(countries)} countries and up to {max_draws} posterior draws per country "
        f"from {MODEL2C_OUTPUT_VERSION}."
    )


if __name__ == "__main__":
    main()
