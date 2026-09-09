"""Recover isolated Model 2C outputs from completed CmdStan chain CSV files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from cmdstanpy import from_csv

from _bootstrap import add_project_src_to_path


add_project_src_to_path()

from pertussis_global_dynamics.bayes import model2c_observation_audit, model2c_posterior_predictive
from pertussis_global_dynamics.model2c import _draws_dataframe, _fit_summary_row, _write_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--iso3", required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args()

    tag = args.tag.strip().lower().replace("-", "_")
    iso3 = args.iso3.strip().upper()
    csv_files = sorted(args.csv_dir.glob("*.csv"))
    if len(csv_files) != args.chains:
        raise SystemExit(f"Expected {args.chains} chain CSVs, found {len(csv_files)} in {args.csv_dir}")
    metadata_path = OUTPUTS / f"model2c_stan_data_metadata_{tag}.csv"
    data_path = OUTPUTS / f"model2c_stan_data_{iso3}_{tag}.json"
    if not metadata_path.exists() or not data_path.exists():
        raise SystemExit("Tagged metadata or Stan data JSON is missing; recovery is not safe.")
    metadata_frame = pd.read_csv(metadata_path)
    metadata_frame = metadata_frame.loc[metadata_frame["iso3"].eq(iso3)]
    if len(metadata_frame) != 1:
        raise SystemExit(f"Expected exactly one {iso3} metadata row, found {len(metadata_frame)}")
    metadata = metadata_frame.iloc[0].to_dict()
    data = json.loads(data_path.read_text(encoding="utf-8"))
    fit = from_csv([str(path) for path in csv_files])
    draws = _draws_dataframe(fit, metadata)
    country_draws = draws.drop(columns=["iso3", "country"])
    tables = {
        "fit_summary": pd.DataFrame([
            _fit_summary_row(fit, metadata, args.chains, args.warmup, args.samples)
        ]),
        "posterior_draws": draws,
        "posterior_predictive": model2c_posterior_predictive(data, metadata, country_draws),
        "observation_audit": model2c_observation_audit(data, metadata, country_draws),
        "metadata": metadata_frame,
    }
    os.environ["MODEL2C_OUTPUT_TAG"] = tag
    os.environ["MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS"] = "0"
    _write_outputs([tables])
    print(f"Recovered completed {iso3} tagged fit outputs for {tag} from {len(csv_files)} chains.")


if __name__ == "__main__":
    main()
