#!/usr/bin/env python3
"""Recover a tagged one-country Model 2C sensitivity refit from CmdStan CSVs."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from cmdstanpy import from_csv

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics.data.model2_inputs import country_selection_audit, load_inputs  # noqa: E402
from pertussis_global_dynamics.model2c import (  # noqa: E402
    MODEL2C_OUTPUT_VERSION,
    _draws_dataframe,
    _fit_summary_row,
    _model2c_hmc_diagnostics,
    _prepare_country,
    _write_outputs,
)
from pertussis_global_dynamics.bayes.model2c_scenarios import (  # noqa: E402
    _select_scenario_draws,
    model2c_observation_audit,
    model2c_posterior_predictive,
)
from pertussis_global_dynamics.paths import OUTPUTS, ROOT  # noqa: E402
from run_model2c_sensitivity import sensitivity_specs  # noqa: E402


def normalise_tag(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower().replace("-", "_")).strip("_")


def spec_env(target_tag: str) -> dict[str, str]:
    target = normalise_tag(target_tag)
    for spec in sensitivity_specs():
        if spec.tag == target:
            return dict(spec.env)
    raise SystemExit(f"Unknown registered sensitivity tag: {target_tag}")


def write_selection(inputs: dict[str, pd.DataFrame], iso3: str, source_tag: str) -> None:
    selection = country_selection_audit(inputs, [iso3])
    selection = selection.rename(
        columns={
            "selected_for_current_model2": "selected_for_current_model2c",
            "model2_selection_reason": "model2c_selection_reason",
        }
    )
    selection["selected_for_current_model2"] = selection["selected_for_current_model2c"]
    selection["model2_selection_reason"] = selection["model2c_selection_reason"].astype(str).str.replace(
        "Model 2", "Model 2C", regex=False
    )
    selection["model2c_selection_reason"] = selection["model2_selection_reason"]
    selection["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
    selection["model2c_output_tag"] = source_tag
    selection.to_csv(OUTPUTS / f"model2c_country_selection_audit_{source_tag}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--iso3", required=True)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--max-treedepth", type=int, default=16)
    parser.add_argument("--scenario-max-draws", type=int, default=80)
    parser.add_argument("--build-scenarios", action="store_true")
    args = parser.parse_args()

    source_tag = normalise_tag(args.source_tag)
    target_tag = normalise_tag(args.target_tag)
    iso3 = args.iso3.strip().upper()
    for key, value in spec_env(target_tag).items():
        os.environ[key] = value
    os.environ.update(
        {
            "MODEL2C_OUTPUT_TAG": source_tag,
            "MODEL2C_CONTACT_MODE": "full",
            "MODEL2C_COUNTRIES": iso3,
            "MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS": "0",
            "MODEL2C_SCENARIO_MAX_DRAWS": str(args.scenario_max_draws),
        }
    )

    csv_paths = [path.resolve() for path in args.csv]
    missing = [path for path in csv_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing CmdStan CSV(s): " + ", ".join(str(path) for path in missing))
    fit = from_csv(path=[str(path) for path in csv_paths], method="sample")
    if fit is None:
        raise SystemExit("cmdstanpy.from_csv returned no fit.")

    inputs = load_inputs()
    _json_path, data, metadata, _params = _prepare_country(inputs, iso3)
    draws = _draws_dataframe(fit, metadata)
    country_draws = draws.drop(columns=["iso3", "country"])
    predictive_draws = _select_scenario_draws(country_draws, args.scenario_max_draws)
    hmc_chains, hmc_summary = _model2c_hmc_diagnostics(fit, metadata, max_treedepth=args.max_treedepth)
    tables = {
        "fit_summary": pd.DataFrame(
            [
                _fit_summary_row(
                    fit,
                    metadata,
                    args.chains,
                    args.warmup,
                    args.samples,
                    hmc_summary=hmc_summary,
                )
            ]
        ),
        "posterior_draws": draws,
        "posterior_predictive": model2c_posterior_predictive(data, metadata, predictive_draws),
        "observation_audit": model2c_observation_audit(data, metadata, country_draws),
        "metadata": pd.DataFrame([metadata]),
        "hmc_diagnostics": hmc_chains,
    }
    _write_outputs([tables])
    write_selection(inputs, iso3, source_tag)
    print(
        f"Recovered {source_tag}/{iso3}: "
        f"max_rhat={tables['fit_summary']['max_rhat'].iloc[0]:.5g}, "
        f"min_ess_bulk={tables['fit_summary']['min_ess_bulk'].iloc[0]:.5g}, "
        f"registered_hmc_gate_pass={tables['fit_summary']['registered_hmc_gate_pass'].iloc[0]}."
    )

    if args.build_scenarios:
        env = os.environ.copy()
        subprocess.run([sys.executable, "script/build_model2c_counterfactual_scenarios.py"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
