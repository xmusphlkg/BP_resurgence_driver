#!/usr/bin/env python3
"""Run isolated Model 2C supplement sensitivity fits.

The script is intentionally conservative: every sensitivity run must set a
MODEL2C_OUTPUT_TAG, so fitted draws, Stan data and replay outputs cannot
overwrite the manuscript-default Model 2C full-contact files.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SCRIPT = ROOT / "script"
PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class SensitivitySpec:
    tag: str
    suite: str
    countries: tuple[str, ...]
    env: dict[str, str]
    description: str


def sensitivity_specs() -> list[SensitivitySpec]:
    return [
        SensitivitySpec(
            tag="contact_upper_7",
            suite="contact",
            countries=("AUS", "BRA", "NZL"),
            env={"MODEL2C_CONTACT_SENSITIVITY_UPPER": "7.0"},
            description="Contact-sensitivity upper-bound sensitivity at 7.0 for boundary-focus countries.",
        ),
        SensitivitySpec(
            tag="contact_upper_10",
            suite="contact",
            countries=("AUS", "BRA", "NZL"),
            env={"MODEL2C_CONTACT_SENSITIVITY_UPPER": "10.0"},
            description="Contact-sensitivity upper-bound sensitivity at 10.0 for boundary-focus countries.",
        ),
        SensitivitySpec(
            tag="reporting_weak",
            suite="reporting",
            countries=("AUS", "BRA", "CHN", "THA"),
            env={"MODEL2C_REPORTING_PRIOR_MODE": "weak"},
            description="Weak post-2023 reporting-prior sensitivity for reporting-boundary countries.",
        ),
        SensitivitySpec(
            tag="reporting_null",
            suite="reporting",
            countries=("AUS", "BRA", "CHN", "THA"),
            env={"MODEL2C_REPORTING_PRIOR_MODE": "null"},
            description="Tight null reporting-prior sensitivity for reporting-boundary countries.",
        ),
        SensitivitySpec(
            tag="reporting_direct",
            suite="reporting",
            countries=("AUS", "BRA", "CHN", "THA"),
            env={"MODEL2C_REPORTING_PRIOR_MODE": "direct"},
            description="Direct-evidence reporting-prior sensitivity where denominator evidence is available.",
        ),
        SensitivitySpec(
            tag="waning_low",
            suite="waning",
            countries=("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"),
            env={"MODEL2C_WANING_PRIOR_MODE": "low"},
            description="Longer-protection waning-prior sensitivity across modelled countries.",
        ),
        SensitivitySpec(
            tag="waning_high",
            suite="waning",
            countries=("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"),
            env={"MODEL2C_WANING_PRIOR_MODE": "high"},
            description="Shorter-protection waning-prior sensitivity across modelled countries.",
        ),
        SensitivitySpec(
            tag="objective_unit_weights",
            suite="objective",
            countries=("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"),
            env={
                "MODEL2C_MONTH_WEIGHT_MODE": "unit",
                "MODEL2C_AGE_OBSERVATION_MODE": "unit_weight",
            },
            description="Unit-weight monthly and annual-age composite-objective sensitivity across modelled countries.",
        ),
        SensitivitySpec(
            tag="age_exclude_split_bins",
            suite="age",
            countries=("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"),
            env={"MODEL2C_AGE_OBSERVATION_MODE": "exclude_split_bins"},
            description="Age-observation sensitivity excluding source bins split across shared model age groups.",
        ),
        SensitivitySpec(
            tag="temporal_fit_2022",
            suite="temporal",
            countries=("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"),
            env={"MODEL2C_FIT_END_YEAR": "2022"},
            description="Pre-resurgence fit through 2022 with 2023-2026 observations reserved for temporal checks.",
        ),
    ]


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower().replace("-", "_") for item in value.replace(";", ",").split(",") if item.strip()}


def selected_specs(args: argparse.Namespace) -> list[SensitivitySpec]:
    suites = _split_csv(args.suite)
    tags = _split_csv(args.only_tags)
    specs = sensitivity_specs()
    if suites and "all" not in suites:
        specs = [spec for spec in specs if spec.suite in suites]
    if tags:
        specs = [spec for spec in specs if spec.tag in tags]
    if args.smoke and not tags:
        specs = [specs[0]] if specs else []
    if not specs:
        raise SystemExit("No sensitivity specifications matched the requested suite/tag filters.")
    return specs


def output_paths(spec: SensitivitySpec) -> dict[str, str]:
    tag = spec.tag
    return {
        "fit_summary": f"outputs/model2c_fit_summary_{tag}.csv",
        "posterior_draws": f"outputs/model2c_posterior_draws_{tag}.csv",
        "posterior_predictive": f"outputs/model2c_posterior_predictive_{tag}.csv",
        "observation_audit": f"outputs/model2c_observation_process_audit_{tag}.csv",
        "metadata": f"outputs/model2c_stan_data_metadata_{tag}.csv",
        "country_selection": f"outputs/model2c_country_selection_audit_{tag}.csv",
        "scenario_summary": f"outputs/model2c_counterfactual_scenario_summary_{tag}.csv",
        "scenario_monthly": f"outputs/model2c_counterfactual_scenario_monthly_{tag}.csv",
    }


def command_env(spec: SensitivitySpec, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(spec.env)
    env.update(
        {
            "MODEL2C_OUTPUT_TAG": spec.tag,
            "MODEL2C_CONTACT_MODE": "full",
            "MODEL2C_COUNTRIES": ",".join(spec.countries if not args.smoke else spec.countries[:1]),
            "MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS": "0",
            "MODEL2C_ALLOW_CONTACT_FALLBACK": os.environ.get("MODEL2C_ALLOW_CONTACT_FALLBACK", "0"),
        }
    )
    if args.smoke:
        env.setdefault("MODEL2C_CHAINS", "1")
        env.setdefault("MODEL2C_PARALLEL_CHAINS", "1")
        env.setdefault("MODEL2C_COUNTRY_WORKERS", "1")
        env.setdefault("MODEL2C_WARMUP", "10")
        env.setdefault("MODEL2C_SAMPLES", "10")
        env.setdefault("MODEL2C_SCENARIO_MAX_DRAWS", "10")
        env.setdefault("MODEL2C_MIN_MANUSCRIPT_DRAWS", "10")
    return env


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def manifest_row(spec: SensitivitySpec, args: argparse.Namespace, status: str) -> dict[str, str]:
    paths = output_paths(spec)
    return {
        "tag": spec.tag,
        "suite": spec.suite,
        "countries": ",".join(spec.countries if not args.smoke else spec.countries[:1]),
        "status": status,
        "prepare_only": str(bool(args.prepare_only)),
        "manifest_only": str(bool(args.manifest_only)),
        "smoke": str(bool(args.smoke)),
        "description": spec.description,
        "environment_overrides": ";".join(f"{key}={value}" for key, value in sorted(spec.env.items())),
        **paths,
    }


def _countries_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, usecols=["iso3"])
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(frame["iso3"].dropna().astype(str))


def existing_status(spec: SensitivitySpec) -> str:
    paths = output_paths(spec)
    fit = ROOT / paths["fit_summary"]
    summary = ROOT / paths["scenario_summary"]
    metadata = ROOT / paths["metadata"]
    expected = set(spec.countries)
    fit_countries = _countries_in(fit)
    summary_countries = _countries_in(summary)
    metadata_countries = _countries_in(metadata)
    if fit_countries:
        fit_complete = expected.issubset(fit_countries)
        if summary_countries:
            summary_complete = expected.issubset(summary_countries)
            if fit_complete and summary_complete:
                return "completed_with_scenario_replay"
            return "partial_completed_with_scenario_replay"
        if fit_complete:
            return "completed_fit_without_scenario_replay"
        return "partial_completed_fit_without_scenario_replay"
    if metadata_countries:
        if expected.issubset(metadata_countries):
            return "prepared"
        return "partial_prepared"
    return "planned"


def write_manifest(rows: list[dict[str, str]], *, suffix: str) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUTS / f"model2c_sensitivity_run_manifest{suffix}.csv", index=False)
    if not suffix:
        frame.to_csv(OUTPUTS / "model2c_sensitivity_run_manifest.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run isolated Model 2C supplement sensitivity fits.")
    parser.add_argument(
        "--suite",
        default="all",
        help="Comma-separated suites: contact, reporting, waning, objective, age, temporal, all.",
    )
    parser.add_argument("--only-tags", default="", help="Comma-separated sensitivity tags to run.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write sensitivity Stan data and metadata.")
    parser.add_argument("--smoke", action="store_true", help="Use one country and tiny MCMC settings for smoke testing.")
    parser.add_argument("--manifest-only", action="store_true", help="Refresh the sensitivity manifest from existing tagged outputs without running fits.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifest without running them.")
    args = parser.parse_args(argv)

    rows: list[dict[str, str]] = []
    suffix = "_smoke" if args.smoke else ""
    for spec in selected_specs(args):
        if args.manifest_only:
            rows.append(manifest_row(spec, args, existing_status(spec)))
            continue
        env = command_env(spec, args)
        country_args = ["--countries", *env["MODEL2C_COUNTRIES"].split(",")]
        if args.prepare_only:
            run_command(
                [str(PYTHON), str(SCRIPT / "run_model2c_bayes.py"), "--prepare-only", *country_args],
                env=env,
                dry_run=args.dry_run,
            )
            rows.append(manifest_row(spec, args, "prepared" if not args.dry_run else "dry_run_prepared"))
            continue

        run_command([str(PYTHON), str(SCRIPT / "run_model2c_bayes.py"), *country_args], env=env, dry_run=args.dry_run)
        run_command([str(PYTHON), str(SCRIPT / "build_model2c_counterfactual_scenarios.py")], env=env, dry_run=args.dry_run)
        rows.append(manifest_row(spec, args, "completed" if not args.dry_run else "dry_run_completed"))
    write_manifest(rows, suffix=suffix)
    print(f"Wrote Model 2C sensitivity manifest for {len(rows)} run(s).")


if __name__ == "__main__":
    main()
