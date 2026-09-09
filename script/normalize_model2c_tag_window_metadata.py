#!/usr/bin/env python3
"""Recompute tagged Model 2C observation-window metadata from its Stan JSON.

This is primarily a provenance safeguard for long-running tagged fits: the
data JSON is authoritative even if the Python metadata helpers were improved
while an already-imported fitting process was still sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pertussis_global_dynamics.model2c_windows import model2c_stan_data_window_metadata  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def authoritative_windows(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    windows = model2c_stan_data_window_metadata(data)
    fit_end = windows.get("model2c_monthly_observation_end_year")
    windows["model2c_fit_end_year"] = fit_end
    windows["model2c_fit_window_semantics"] = (
        f"Likelihood fit through {fit_end}; later available monthly observations are reserved for temporal checks."
    )
    return windows


def normalize_frame(frame: pd.DataFrame, tag: str, outputs: Path) -> pd.DataFrame:
    normalized = frame.copy()
    if "iso3" not in normalized:
        raise ValueError("Tagged metadata table has no iso3 column.")
    for index, iso3 in normalized["iso3"].astype(str).items():
        path = outputs / f"model2c_stan_data_{iso3}_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        for key, value in authoritative_windows(path).items():
            normalized.loc[index, key] = value
    return normalized


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    paths = [
        args.outputs / f"model2c_stan_data_metadata_{args.tag}.csv",
        args.outputs / f"model2c_fit_summary_{args.tag}.csv",
    ]
    changed = 0
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        write_atomic(normalize_frame(frame, args.tag, args.outputs), path)
        print(f"Normalised {path}")
        changed += 1
    if not changed:
        raise SystemExit(f"No tagged metadata tables found for {args.tag}")


if __name__ == "__main__":
    main()
