from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def _weight_metrics(values: list[float]) -> dict[str, float]:
    weights = np.asarray(values, dtype=float)
    if weights.size == 0:
        return {"n": 0, "sum": 0.0, "mean": np.nan, "kish_ess": 0.0, "design_effect": np.nan}
    total = float(weights.sum())
    kish = total * total / float(np.square(weights).sum()) if np.square(weights).sum() > 0 else 0.0
    return {
        "n": int(weights.size),
        "sum": total,
        "mean": float(weights.mean()),
        "kish_ess": kish,
        "design_effect": float(weights.size / kish) if kish > 0 else np.nan,
    }


def diagnose_likelihood_weights(data: dict[str, Any], iso3: str, source: str) -> dict[str, Any]:
    month = _weight_metrics([float(value) for value in data.get("month_obs_weight", [])])
    age = _weight_metrics([float(value) for value in data.get("age_obs_weight", [])])
    month_year_totals: dict[int, int] = {}
    for idx, count in zip(data.get("month_obs_idx", []), data.get("y_month", [])):
        year_idx = int(data["year_index"][int(idx) - 1])
        month_year_totals[year_idx] = month_year_totals.get(year_idx, 0) + int(count)
    age_year_totals: dict[int, int] = {}
    for year_idx, count in zip(data.get("age_obs_year_idx", []), data.get("y_age", [])):
        key = int(year_idx)
        age_year_totals[key] = age_year_totals.get(key, 0) + int(count)
    shared = sorted(set(month_year_totals).intersection(age_year_totals))
    monthly_shared_total = sum(month_year_totals[year] for year in shared)
    age_shared_total = sum(age_year_totals[year] for year in shared)
    overlap_ratio = age_shared_total / monthly_shared_total if monthly_shared_total > 0 else np.nan
    return {
        "iso3": iso3,
        "source": source,
        "monthly_observations": month["n"],
        "monthly_weight_sum": month["sum"],
        "monthly_weight_mean_implied_temperature": month["mean"],
        "monthly_weight_kish_ess": month["kish_ess"],
        "monthly_weight_design_effect": month["design_effect"],
        "age_cells": age["n"],
        "age_weight_sum": age["sum"],
        "age_weight_mean_implied_temperature": age["mean"],
        "age_weight_kish_ess": age["kish_ess"],
        "age_weight_design_effect": age["design_effect"],
        "years_with_both_monthly_and_age_data": len(shared),
        "age_to_monthly_case_total_ratio_shared_years": overlap_ratio,
        "learning_rate_calibrated": False,
        "variance_calibration_status": "not_identifiable_without_pointwise_scores_or_refits",
        "interpretation": (
            "Kish metrics quantify weight concentration only; they are not a sandwich/Godambe "
            "calibration of the generalized-posterior learning rate."
        ),
    }


def _iso3_from_path(path: Path) -> str:
    match = re.search(r"model2c_stan_data_([A-Z]{3})_", path.name)
    return match.group(1) if match else "UNK"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit composite-likelihood weights without refitting Model 2C.")
    parser.add_argument("paths", nargs="*", type=Path, help="Tagged Model 2C Stan JSON files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS / "model2c_likelihood_weight_diagnostic.csv",
        help="Diagnostic CSV path.",
    )
    args = parser.parse_args(argv)
    paths = args.paths or sorted(OUTPUTS.glob("model2c_stan_data_???_full.json"))
    if not paths:
        raise SystemExit("No Model 2C Stan JSON files found.")
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(diagnose_likelihood_weights(data, _iso3_from_path(path), str(path.relative_to(ROOT))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("iso3").to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} likelihood-weight diagnostics to {args.output}")


if __name__ == "__main__":
    main()
