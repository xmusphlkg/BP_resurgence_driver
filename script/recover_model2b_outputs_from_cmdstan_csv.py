#!/usr/bin/env python3
"""Recover Model 2B posterior draws and monthly PPC from CmdStan CSV files."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODEL2B_OUTPUT_VERSION = "model2b_v2_20260616_layered_endemic_background_attribution"

DRAW_COLUMNS = [
    "log_beta",
    "log_reporting_rate",
    "contact_sensitivity",
    "waning_rate",
    "log_post_transmission_multiplier",
    "log_post_reporting_multiplier",
    "rebound_peak_month",
    "rebound_width",
    "rebound_release_gain",
    "outbreak_seeding_gain",
    "seasonal_strength",
    "natural_boosting_efficiency",
    "pathogen_pressure_scale",
    "epidemic_feedback_gain",
    "endemic_reseed_pressure",
    "endemic_background_force",
    "epidemic_pressure_retention",
    "post_2023_transmission_shift_multiplier",
    "excess_epidemic_pressure_scale",
    "outbreak_ignition_pulse_gain",
    "missed_natural_boosting_2020_2022",
    "natural_boosting_flow_2020_2022",
    "routine_vaccination_shortfall_2020_2022",
    "natural_waned_stock_2023_start",
    "net_demographic_adjustment_2015_2026",
    "nb_phi_month",
    "nb_phi_age",
    "init_natural_share",
    "init_vaccine_recent_fraction",
    "init_natural_recent_fraction",
    "nb_phi",
    "true_infections_2023_2025",
    "symptomatic_infections_2023_2025",
    "reported_cases_2023_2025",
    "reported_cases_2026",
]

ISO_RE = re.compile(r"model2b_stan_data_([A-Z]{3})(?:_(?:full|rank_one))?\.json")
CHAIN_RE = re.compile(r"_(\d+)\.csv$")


def infer_iso(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = ISO_RE.search(line)
            if match:
                return match.group(1)
            if line.startswith("lp__,"):
                break
    raise SystemExit(f"Could not infer ISO3 country from CmdStan CSV header: {path}")


def chain_number(path: Path) -> int:
    match = CHAIN_RE.search(path.name)
    return int(match.group(1)) if match else 1


def contact_mode_suffix() -> str:
    raw = os.environ.get("MODEL2B_CONTACT_MODE", "full").strip().lower().replace("-", "_")
    return "full" if raw == "full" else "rank_one"


def mode_candidates(filename: str, contact_mode: str) -> list[Path]:
    path = OUTPUTS / filename
    stem = path.stem
    suffix = path.suffix
    suffixed = OUTPUTS / f"{stem}_{contact_mode}{suffix}"
    if contact_mode == "rank_one":
        return [suffixed]
    return [suffixed]


def first_existing(paths: list[Path], message: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = ", ".join(str(path.relative_to(ROOT)) for path in paths)
    raise SystemExit(f"{message}. Checked: {checked}")


def load_metadata(contact_mode: str) -> pd.DataFrame:
    path = first_existing(
        mode_candidates("model2b_stan_data_metadata.csv", contact_mode),
        "Missing Model 2B Stan metadata",
    )
    metadata = pd.read_csv(path)
    if "iso3" not in metadata or "country" not in metadata:
        raise SystemExit(f"{path.relative_to(ROOT)} must contain iso3 and country columns")
    return metadata.loc[:, ["iso3", "country"]].drop_duplicates()


def load_stan_data(iso3: str, contact_mode: str) -> dict[str, object]:
    if contact_mode == "rank_one":
        candidates = [OUTPUTS / f"model2b_stan_data_{iso3}_rank_one.json"]
    else:
        candidates = [OUTPUTS / f"model2b_stan_data_{iso3}_{contact_mode}.json"]
    path = first_existing(candidates, f"Missing Stan data JSON for {iso3}")
    return json.loads(path.read_text(encoding="utf-8"))


def observed_map(data: dict[str, object]) -> dict[int, int]:
    observed: dict[int, int] = {}
    for index, value in zip(data.get("month_obs_idx", []), data.get("y_month", [])):
        observed[int(index)] = int(value)
    for index, value in zip(data.get("month_ppc_obs_idx", []), data.get("y_month_ppc", [])):
        observed[int(index)] = int(value)
    return observed


def read_chain(path: Path, n_months: int) -> pd.DataFrame:
    monthly_prefixes = ("mu_month.", "true_month.", "symptomatic_month.")
    monthly_columns = {
        f"mu_month.{index}" for index in range(1, n_months + 1)
    } | {
        f"true_month.{index}" for index in range(1, n_months + 1)
    } | {
        f"symptomatic_month.{index}" for index in range(1, n_months + 1)
    }
    wanted = set(DRAW_COLUMNS) | monthly_columns

    def use_column(column: str) -> bool:
        return column in wanted or column.startswith(monthly_prefixes)

    return pd.read_csv(path, comment="#", usecols=use_column)


def recover_country(iso3: str, country: str, csv_paths: list[Path], contact_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_stan_data(iso3, contact_mode)
    n_months = int(data["N_months"])
    year = np.asarray(data["year"], dtype=int)
    month = np.asarray(data["month"], dtype=int)
    observed = observed_map(data)

    draw_parts: list[pd.DataFrame] = []
    ppc_parts: list[pd.DataFrame] = []
    draw_offset = 0

    for path in sorted(csv_paths, key=chain_number):
        chain = chain_number(path)
        frame = read_chain(path, n_months)
        if frame.empty:
            continue
        frame = frame.reset_index(drop=True)
        n_draws = len(frame)
        draw_ids = np.arange(draw_offset + 1, draw_offset + n_draws + 1, dtype=int)
        draw_offset += n_draws

        draw_cols = [column for column in DRAW_COLUMNS if column in frame.columns]
        draws = frame.loc[:, draw_cols].copy()
        draws.insert(0, "draw", draw_ids)
        draws.insert(0, "iteration", np.arange(1, n_draws + 1, dtype=int))
        draws.insert(0, "chain", chain)
        draws.insert(0, "country", country)
        draws.insert(0, "iso3", iso3)
        draws["model2b_output_version"] = MODEL2B_OUTPUT_VERSION
        draw_parts.append(draws)

        mu = frame.loc[:, [f"mu_month.{index}" for index in range(1, n_months + 1)]].to_numpy(dtype=float)
        true = frame.loc[:, [f"true_month.{index}" for index in range(1, n_months + 1)]].to_numpy(dtype=float)
        symptomatic = frame.loc[:, [f"symptomatic_month.{index}" for index in range(1, n_months + 1)]].to_numpy(dtype=float)
        month_index = np.tile(np.arange(1, n_months + 1, dtype=int), n_draws)
        ppc = pd.DataFrame(
            {
                "iso3": iso3,
                "country": country,
                "draw": np.repeat(draw_ids, n_months),
                "year": np.tile(year, n_draws),
                "month": np.tile(month, n_draws),
                "observed_cases": [observed.get(int(index), np.nan) for index in month_index],
                "true_infections_predicted": true.reshape(-1),
                "symptomatic_infections_predicted": symptomatic.reshape(-1),
                "reported_cases_predicted": mu.reshape(-1),
            }
        )
        ppc["fit_role"] = np.where(ppc["year"].le(2025), "fit_likelihood", "posterior_predictive_check")
        ppc["model2b_output_version"] = MODEL2B_OUTPUT_VERSION
        ppc_parts.append(ppc)

    if not draw_parts or not ppc_parts:
        raise SystemExit(f"No usable CmdStan draws recovered for {iso3}")
    return pd.concat(draw_parts, ignore_index=True), pd.concat(ppc_parts, ignore_index=True)


def discover(source_root: Path) -> dict[str, list[Path]]:
    country_files: dict[str, list[Path]] = {}
    for directory in sorted(source_root.glob("model2b_transmission_attribution*")):
        if not directory.is_dir():
            continue
        csv_paths = sorted(directory.glob("*.csv"))
        if not csv_paths:
            continue
        iso3 = infer_iso(csv_paths[0])
        country_files.setdefault(iso3, []).extend(csv_paths)
    return country_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Directory containing CmdStan model2b_transmission_attribution* folders")
    args = parser.parse_args()

    contact_mode = contact_mode_suffix()
    metadata = load_metadata(contact_mode)
    country_by_iso = dict(zip(metadata["iso3"], metadata["country"]))
    country_files = discover(args.source_root)
    missing = sorted(set(country_by_iso) - set(country_files))
    if missing:
        raise SystemExit(f"Missing CmdStan CSV folders for: {', '.join(missing)}")

    draw_tables: list[pd.DataFrame] = []
    ppc_tables: list[pd.DataFrame] = []
    for iso3 in sorted(country_by_iso):
        draws, ppc = recover_country(iso3, country_by_iso[iso3], country_files[iso3], contact_mode)
        draw_tables.append(draws)
        ppc_tables.append(ppc)
        print(f"Recovered {iso3}: {draws['draw'].nunique()} posterior draws")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    draws_out = pd.concat(draw_tables, ignore_index=True)
    ppc_out = pd.concat(ppc_tables, ignore_index=True)
    draws_out.to_csv(OUTPUTS / f"model2b_bayes_posterior_draws_{contact_mode}.csv", index=False)
    ppc_out.to_csv(OUTPUTS / f"model2b_bayes_posterior_predictive_{contact_mode}.csv", index=False)
    print(f"Wrote outputs/model2b_bayes_posterior_draws_{contact_mode}.csv")
    print(f"Wrote outputs/model2b_bayes_posterior_predictive_{contact_mode}.csv")
    print("Recovered CmdStan draws are suffixed only; run recompute_model2b_attribution_from_draws.py to publish complete default outputs.")


if __name__ == "__main__":
    main()
