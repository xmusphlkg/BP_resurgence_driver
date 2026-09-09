#!/usr/bin/env python3
"""Prepare or run isolated Model 2C structural refits.

Every run has a mandatory structural tag and uses a generated Stan source, so
neither the manuscript-default executable nor any untagged output is replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics import model2c  # noqa: E402
from pertussis_global_dynamics.bayes.model2c_structural import (  # noqa: E402
    STRUCTURAL_FLAG_DEFAULTS,
    augment_structural_data,
    write_structural_stan,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
BASE_STAN = model2c.STAN_MODEL
STRUCTURAL_STAN = BASE_STAN.with_name("model2c_structural_refits.stan")
DEFAULT_COUNTRIES = model2c.MODEL2C_DEFAULT_PILOT_ISO3
ORIGINAL_PREPARE_COUNTRY = model2c._prepare_country
ORIGINAL_DRAW_COLUMNS = tuple(model2c.MODEL2C_STAN_DRAW_COLUMNS)
ORIGINAL_PARAMETER_COLUMNS = tuple(model2c.MODEL2C_STAN_PARAMETER_COLUMNS)


@dataclass(frozen=True)
class StructuralSpec:
    tag: str
    label: str
    flags: dict[str, int]
    description: str


def structural_specs() -> list[StructuralSpec]:
    return [
        StructuralSpec(
            "structural_full",
            "full",
            {},
            "Tagged contemporaneous control with all default structural mechanisms enabled.",
        ),
        StructuralSpec(
            "structural_no_exposure_interruption",
            "no_exposure_interruption",
            {"use_exposure_interruption": 0},
            "Disable infection-contact and silent-exposure/boosting interruption from the contact-intensity path.",
        ),
        StructuralSpec(
            "structural_no_post2023_waning",
            "no_post_2023_waning",
            {"use_post_2023_waning": 0},
            "Disable vaccine, partial-vaccine, natural and deep-natural stock waning from January 2023 onward.",
        ),
        StructuralSpec(
            "structural_no_reporting_shift",
            "no_reporting_shift",
            {"use_reporting_shift": 0},
            "Fix the post-2023 level, peak and trend reporting multipliers to one in the observation mean.",
        ),
        StructuralSpec(
            "structural_flexible_seeding",
            "flexible_outbreak_seeding",
            {"use_flexible_outbreak_seeding": 1},
            "Replace the fitted two-pulse outbreak ignition path with four regularized time-basis Gaussian gains centred across 2023-2026.",
        ),
    ]


def _spec_map() -> dict[str, StructuralSpec]:
    return {spec.tag: spec for spec in structural_specs()}


def output_paths(tag: str) -> dict[str, str]:
    return {
        "fit_summary": f"outputs/model2c_fit_summary_{tag}.csv",
        "posterior_draws": f"outputs/model2c_posterior_draws_{tag}.csv",
        "posterior_predictive": f"outputs/model2c_posterior_predictive_{tag}.csv",
        "observation_audit": f"outputs/model2c_observation_process_audit_{tag}.csv",
        "metadata": f"outputs/model2c_stan_data_metadata_{tag}.csv",
        "scenario_summary": f"outputs/model2c_counterfactual_scenario_summary_{tag}.csv",
    }


def _install_spec(spec: StructuralSpec) -> None:
    flags = dict(STRUCTURAL_FLAG_DEFAULTS)
    flags.update(spec.flags)
    os.environ["MODEL2C_OUTPUT_TAG"] = spec.tag
    os.environ["MODEL2C_CONTACT_MODE"] = "full"
    os.environ["MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS"] = "0"
    write_structural_stan(BASE_STAN, STRUCTURAL_STAN)

    def prepare_country(
        inputs: dict[str, Any], iso3: str
    ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, float]]:
        path, data, metadata, params = ORIGINAL_PREPARE_COUNTRY(inputs, iso3)
        data = augment_structural_data(data, flags)
        metadata = {
            **metadata,
            "model2c_structural_refit_tag": spec.tag,
            "model2c_structural_refit_label": spec.label,
            "model2c_structural_stan_model": str(STRUCTURAL_STAN.relative_to(ROOT)),
            **{f"model2c_{name}": value for name, value in flags.items()},
        }
        path.write_text(json.dumps(model2c._json_ready(data), indent=2), encoding="utf-8")
        return path, data, metadata, params

    model2c._prepare_country = prepare_country
    model2c.STAN_MODEL = STRUCTURAL_STAN
    model2c.MODEL2C_STAN_DRAW_COLUMNS[:] = ORIGINAL_DRAW_COLUMNS
    model2c.MODEL2C_STAN_PARAMETER_COLUMNS[:] = ORIGINAL_PARAMETER_COLUMNS
    inactive_diagnostic_parameters: set[str] = set()
    if not flags["use_exposure_interruption"]:
        inactive_diagnostic_parameters.add("contact_sensitivity")
    if not flags["use_reporting_shift"]:
        inactive_diagnostic_parameters.update(
            {"log_post_reporting_multiplier", "post_reporting_peak_gain", "post_reporting_trend"}
        )
    if flags["use_flexible_outbreak_seeding"]:
        inactive_diagnostic_parameters.add("outbreak_seeding_gain")
        for name in [f"outbreak_seed_basis_gain[{index}]" for index in range(1, 5)]:
            model2c.MODEL2C_STAN_DRAW_COLUMNS.append(name)
            model2c.MODEL2C_STAN_PARAMETER_COLUMNS.append(name)
        model2c.MODEL2C_STAN_DRAW_COLUMNS.append("flexible_outbreak_seeding_total_gain")
    model2c.MODEL2C_STAN_PARAMETER_COLUMNS[:] = [
        name for name in model2c.MODEL2C_STAN_PARAMETER_COLUMNS
        if name not in inactive_diagnostic_parameters
    ]


def _countries_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["iso3"])["iso3"].dropna().astype(str))
    except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return set()


def _status(spec: StructuralSpec, expected: set[str]) -> str:
    paths = output_paths(spec.tag)
    fit = _countries_in(ROOT / paths["fit_summary"])
    ppc = _countries_in(ROOT / paths["posterior_predictive"])
    metadata = _countries_in(ROOT / paths["metadata"])
    if expected.issubset(fit) and expected.issubset(ppc):
        return "fit_complete"
    if fit or ppc:
        return "partial_fit"
    if expected.issubset(metadata):
        return "prepared"
    if metadata:
        return "partial_prepared"
    return "planned"


def _manifest(specs: list[StructuralSpec], countries: tuple[str, ...], requested_mode: str) -> pd.DataFrame:
    expected = set(countries)
    rows = []
    for spec in specs:
        flags = dict(STRUCTURAL_FLAG_DEFAULTS)
        flags.update(spec.flags)
        rows.append(
            {
                "tag": spec.tag,
                "structural_model": spec.label,
                "countries": ",".join(countries),
                "status": _status(spec, expected),
                "requested_mode": requested_mode,
                "description": spec.description,
                **flags,
                **output_paths(spec.tag),
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS / "model2c_structural_refit_manifest.csv", index=False)
    return frame


def _stanc_check() -> None:
    write_structural_stan(BASE_STAN, STRUCTURAL_STAN)
    from cmdstanpy import cmdstan_path

    stanc = Path(cmdstan_path()) / "bin" / "stanc"
    subprocess.run(
        [str(stanc), "--name=model2c_structural_refits", str(STRUCTURAL_STAN)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"Stan syntax check passed: {STRUCTURAL_STAN.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare or run isolated Model 2C structural refits.")
    parser.add_argument("--only-tags", default="", help="Comma-separated structural tags; default is all five.")
    parser.add_argument("--countries", nargs="*", help="ISO3 countries; default is the nine-country Model 2C panel.")
    parser.add_argument("--prepare-only", action="store_true", help="Write tagged data/metadata; do not compile or sample.")
    parser.add_argument("--manifest-only", action="store_true", help="Refresh status from existing outputs only.")
    parser.add_argument("--stanc-check", action="store_true", help="Generate and syntax-check Stan; never compile or sample.")
    args = parser.parse_args(argv)
    modes = sum((args.prepare_only, args.manifest_only, args.stanc_check))
    if modes > 1:
        parser.error("Choose only one of --prepare-only, --manifest-only and --stanc-check.")

    by_tag = _spec_map()
    requested = [value.strip().replace("-", "_") for value in args.only_tags.split(",") if value.strip()]
    unknown = set(requested) - set(by_tag)
    if unknown:
        parser.error("Unknown structural tag(s): " + ", ".join(sorted(unknown)))
    specs = [by_tag[tag] for tag in requested] if requested else structural_specs()
    countries = tuple(args.countries or DEFAULT_COUNTRIES)

    if args.stanc_check:
        _stanc_check()
        _manifest(specs, countries, "stanc_check")
        return
    if args.manifest_only:
        _manifest(specs, countries, "manifest_only")
        return

    for spec in specs:
        _install_spec(spec)
        forwarded = ["--countries", *countries]
        if args.prepare_only:
            forwarded.append("--prepare-only")
        model2c.main(forwarded)
    _manifest(specs, countries, "prepare_only" if args.prepare_only else "fit")


if __name__ == "__main__":
    main()
