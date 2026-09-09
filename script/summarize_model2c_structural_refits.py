#!/usr/bin/env python3
"""Summarize in-fit and held-out prediction for isolated structural refits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_model2c_structural_refits import structural_specs


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
METRICS = (
    "rmse",
    "mae",
    "median_absolute_error",
    "mean_bias",
    "coverage_80",
    "coverage_95",
    "mean_interval_width_80",
    "mean_interval_width_95",
)


def _prediction_path(tag: str) -> Path:
    return OUTPUTS / f"model2c_posterior_predictive_{tag}.csv"


def summarize_prediction_file(path: Path, tag: str, structural_model: str) -> pd.DataFrame:
    columns = [
        "iso3", "country", "draw", "year", "month", "observed_cases",
        "reported_cases_predicted", "reported_cases_predictive", "fit_role",
    ]
    frame = pd.read_csv(path, usecols=lambda name: name in columns)
    required = set(columns) - {"reported_cases_predictive"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {', '.join(sorted(missing))}")
    frame = frame.loc[frame["observed_cases"].notna()].copy()
    records: list[dict[str, object]] = []
    month_keys = ["iso3", "country", "fit_role", "year", "month"]
    for keys, group in frame.groupby(month_keys, sort=True, dropna=False):
        observed = float(pd.to_numeric(group["observed_cases"], errors="coerce").dropna().iloc[0])
        latent = pd.to_numeric(group["reported_cases_predicted"], errors="coerce").dropna()
        predictive = pd.to_numeric(group.get("reported_cases_predictive", latent), errors="coerce").dropna()
        if latent.empty or predictive.empty:
            continue
        records.append(
            {
                **dict(zip(month_keys, keys)),
                "observed": observed,
                "posterior_median": float(latent.median()),
                "predictive_q025": float(predictive.quantile(0.025)),
                "predictive_q10": float(predictive.quantile(0.10)),
                "predictive_q90": float(predictive.quantile(0.90)),
                "predictive_q975": float(predictive.quantile(0.975)),
            }
        )
    monthly = pd.DataFrame(records)
    if monthly.empty:
        return pd.DataFrame()
    summaries = []
    for (iso3, country, fit_role), group in monthly.groupby(["iso3", "country", "fit_role"], sort=True):
        residual = group["posterior_median"] - group["observed"]
        summaries.append(
            {
                "tag": tag,
                "structural_model": structural_model,
                "iso3": iso3,
                "country": country,
                "fit_role": fit_role,
                "n_observed_months": len(group),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "median_absolute_error": float(np.median(np.abs(residual))),
                "mean_bias": float(np.mean(residual)),
                "coverage_80": float(np.mean((group["observed"] >= group["predictive_q10"]) & (group["observed"] <= group["predictive_q90"]))),
                "coverage_95": float(np.mean((group["observed"] >= group["predictive_q025"]) & (group["observed"] <= group["predictive_q975"]))),
                "mean_interval_width_80": float(np.mean(group["predictive_q90"] - group["predictive_q10"])),
                "mean_interval_width_95": float(np.mean(group["predictive_q975"] - group["predictive_q025"])),
            }
        )
    return pd.DataFrame(summaries)


def compare_with_control(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    control_tag = "structural_full" if summary["tag"].eq("structural_full").any() else "main_full"
    control = summary.loc[summary["tag"].eq(control_tag)].copy()
    keys = ["iso3", "country", "fit_role"]
    rows = []
    for tag, alternative in summary.loc[~summary["tag"].eq(control_tag)].groupby("tag", sort=True):
        merged = alternative.merge(control, on=keys, suffixes=("_alternative", "_control"))
        for record in merged.to_dict(orient="records"):
            for metric in METRICS:
                alt = float(record[f"{metric}_alternative"])
                ctl = float(record[f"{metric}_control"])
                rows.append(
                    {
                        "tag": tag,
                        "structural_model": record["structural_model_alternative"],
                        **{key: record[key] for key in keys},
                        "control_tag": control_tag,
                        "metric": metric,
                        "control_value": ctl,
                        "alternative_value": alt,
                        "alternative_minus_control": alt - ctl,
                        "relative_change_pct": np.nan if ctl == 0 else 100.0 * (alt - ctl) / abs(ctl),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-main-fallback", action="store_true", help="Use the frozen main full fit when tagged control is absent.")
    args = parser.parse_args(argv)
    frames = []
    for spec in structural_specs():
        path = _prediction_path(spec.tag)
        if path.exists():
            frames.append(summarize_prediction_file(path, spec.tag, spec.label))
    if args.include_main_fallback and not any(not frame.empty and frame["tag"].eq("structural_full").any() for frame in frames):
        main_path = OUTPUTS / "model2c_posterior_predictive_full.csv"
        if main_path.exists():
            frames.append(summarize_prediction_file(main_path, "main_full", "frozen_main_full"))
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    comparison = compare_with_control(summary)
    summary.to_csv(OUTPUTS / "model2c_structural_refit_predictive_metrics.csv", index=False)
    comparison.to_csv(OUTPUTS / "model2c_structural_refit_predictive_comparison.csv", index=False)
    print(f"Wrote {len(summary)} structural predictive metric rows and {len(comparison)} control comparisons.")


if __name__ == "__main__":
    main()
