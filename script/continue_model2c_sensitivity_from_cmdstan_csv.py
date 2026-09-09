#!/usr/bin/env python3
"""Continue a completed tagged Model 2C fit from adapted CmdStan CSV states."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics.bayes.model2c_scenarios import (  # noqa: E402
    _select_scenario_draws,
    model2c_observation_audit,
    model2c_posterior_predictive,
)
from pertussis_global_dynamics.data.model2_inputs import country_selection_audit, load_inputs  # noqa: E402
from pertussis_global_dynamics.model2c import (  # noqa: E402
    MODEL2C_OUTPUT_VERSION,
    MODEL2C_STAN_PARAMETER_COLUMNS,
    STAN_MODEL,
    _draws_dataframe,
    _fit_summary_row,
    _model2c_hmc_diagnostics,
    _prepare_country,
    _write_outputs,
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


def read_step_size_and_metric(path: Path) -> tuple[float, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    step_size = np.nan
    metric: list[float] = []
    for index, line in enumerate(lines):
        match = re.search(r"# Step size = ([0-9.eE+-]+)", line)
        if match:
            step_size = float(match.group(1))
        if line.startswith("# Diagonal elements of inverse mass matrix:"):
            for metric_line in lines[index + 1 :]:
                if not metric_line.startswith("#"):
                    break
                values = metric_line.lstrip("#").strip().rstrip(",")
                if values:
                    metric.extend(float(value.strip()) for value in values.split(",") if value.strip())
            break
    if not np.isfinite(step_size) or not metric:
        raise SystemExit(f"Could not extract adapted step size and diagonal metric from {path}")
    return step_size, np.asarray(metric, dtype=float)


def last_initial_value(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, comment="#")
    if frame.empty:
        raise SystemExit(f"No posterior rows in {path}")
    row = frame.iloc[-1]
    init = {
        column: float(row[column])
        for column in MODEL2C_STAN_PARAMETER_COLUMNS
        if column in frame.columns and np.isfinite(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])
    }
    missing = sorted(set(MODEL2C_STAN_PARAMETER_COLUMNS).difference(init))
    if missing:
        raise SystemExit(f"{path} is missing sampled parameter columns for init: {', '.join(missing)}")
    return init


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
    parser.add_argument("--previous-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--max-treedepth", type=int, default=16)
    parser.add_argument("--scenario-max-draws", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260819)
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
    csv_paths = [path.resolve() for path in args.previous_csv]
    if any(not path.exists() for path in csv_paths):
        raise SystemExit("One or more previous CmdStan CSVs are missing.")

    inputs = load_inputs()
    json_path, data, metadata, _params = _prepare_country(inputs, iso3)
    steps_and_metrics = [read_step_size_and_metric(path) for path in csv_paths]
    step_sizes = [item[0] for item in steps_and_metrics]
    inv_metrics = [item[1] for item in steps_and_metrics]
    inits = [last_initial_value(path) for path in csv_paths]
    model = CmdStanModel(stan_file=str(STAN_MODEL))
    fit = model.sample(
        data=str(json_path),
        chains=len(csv_paths),
        parallel_chains=len(csv_paths),
        iter_warmup=0,
        iter_sampling=args.samples,
        seed=args.seed,
        inits=inits,
        adapt_engaged=False,
        step_size=step_sizes,
        inv_metric=inv_metrics,
        max_treedepth=args.max_treedepth,
        show_progress=False,
        show_console=False,
    )

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
                    len(csv_paths),
                    0,
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
        f"Continued {source_tag}/{iso3}: "
        f"max_rhat={tables['fit_summary']['max_rhat'].iloc[0]:.5g}, "
        f"min_ess_bulk={tables['fit_summary']['min_ess_bulk'].iloc[0]:.5g}, "
        f"registered_hmc_gate_pass={tables['fit_summary']['registered_hmc_gate_pass'].iloc[0]}."
    )

    if args.build_scenarios:
        subprocess.run([sys.executable, "script/build_model2c_counterfactual_scenarios.py"], cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
