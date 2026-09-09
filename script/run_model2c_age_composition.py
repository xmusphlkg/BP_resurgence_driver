from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics.bayes.model2c_age_composition import (  # noqa: E402
    augment_age_composition_data,
    write_age_composition_stan,
)
from pertussis_global_dynamics import model2c  # noqa: E402


BASE_STAN = model2c.STAN_MODEL
AGE_COMPOSITION_STAN = BASE_STAN.with_name("model2c_age_composition_sensitivity.stan")
DEFAULT_TAG = "age_composition_dm"


def _install_isolated_mode() -> None:
    os.environ.setdefault("MODEL2C_OUTPUT_TAG", DEFAULT_TAG)
    write_age_composition_stan(BASE_STAN, AGE_COMPOSITION_STAN)
    base_prepare = model2c._prepare_country

    def prepare_country(
        inputs: dict[str, Any], iso3: str
    ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, float]]:
        path, data, metadata, params = base_prepare(inputs, iso3)
        data = augment_age_composition_data(data)
        metadata = dict(metadata)
        metadata.update(
            {
                "model2c_likelihood_structure": "monthly_all_age_nb_plus_conditional_annual_age_dirichlet_multinomial",
                "model2c_age_composition_years": data["N_age_comp_years"],
                "model2c_age_composition_cells": data["N_age_comp_cells"],
                "model2c_age_composition_conditions_on_annual_total": True,
            }
        )
        path.write_text(json.dumps(model2c._json_ready(data), indent=2), encoding="utf-8")
        return path, data, metadata, params

    model2c._prepare_country = prepare_country
    model2c.STAN_MODEL = AGE_COMPOSITION_STAN


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Model 2C conditional annual age-composition sensitivity."
    )
    parser.add_argument("--countries", nargs="*", help="ISO3 countries to prepare or fit.")
    parser.add_argument("--prepare-only", action="store_true", help="Write and validate tagged Stan data only.")
    parser.add_argument("--compile-only", action="store_true", help="Generate and compile the alternative Stan model only.")
    args = parser.parse_args(argv)
    if args.prepare_only and args.compile_only:
        parser.error("Choose only one of --prepare-only and --compile-only.")

    _install_isolated_mode()
    if args.compile_only:
        CmdStanModel = model2c._require_cmdstanpy()
        compiled = CmdStanModel(stan_file=str(AGE_COMPOSITION_STAN))
        print(f"Compiled conditional age-composition model: {compiled.exe_file}")
        return

    forwarded: list[str] = []
    if args.countries:
        forwarded.extend(["--countries", *args.countries])
    if args.prepare_only:
        forwarded.append("--prepare-only")
    model2c.main(forwarded)


if __name__ == "__main__":
    main()
