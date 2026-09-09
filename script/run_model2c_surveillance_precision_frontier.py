#!/usr/bin/env python3
"""Estimate one-parameter precision frontiers for mechanism-resolving streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.bayes.model2c_surveillance_design import (  # noqa: E402
    MeasurementDesign,
    SURVEILLANCE_BOOTSTRAP_REPLICATES,
    nested_country_draw_bootstrap_auc,
    cross_validated_predictions,
    discrimination_metrics,
    feature_columns_for_package,
    simulate_observation_features,
)


FRONTIER_VERSION = "model2c_surveillance_precision_frontier_v2_infant_20260908"
POSITIVE_STRUCTURE = "matched_transmission_truth"
NEGATIVE_STRUCTURE = "matched_reporting_truth"
TESTING_LOG_SD_GRID = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00)
SEVERE_BASELINE_EVENT_GRID = (0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 50.00, 100.00, 200.00)
SEVERE_LOG_SD_GRID = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="outputs/model2c_surveillance_design/monitoring_packages_v4_source_preserved_20260908",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=SURVEILLANCE_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _settings() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in TESTING_LOG_SD_GRID:
        rows.append(
            {
                "frontier_type": "testing_log_sd",
                "parameter_value": value,
                "parameter_units": "log-scale SD",
                "stream": "testing",
                "testing_log_sd": value,
                "severe_baseline_events": 50.0,
                "severe_log_sd": 0.20,
            }
        )
    for value in SEVERE_BASELINE_EVENT_GRID:
        rows.append(
            {
                "frontier_type": "severe_baseline_events",
                "parameter_value": value,
                "parameter_units": "expected baseline infant events per sentinel catchment-year",
                "stream": "severe",
                "testing_log_sd": 0.20,
                "severe_baseline_events": value,
                "severe_log_sd": 0.20,
            }
        )
    for value in SEVERE_LOG_SD_GRID:
        rows.append(
            {
                "frontier_type": "severe_annual_log_sd",
                "parameter_value": value,
                "parameter_units": "annual log-scale SD",
                "stream": "severe",
                "testing_log_sd": 0.20,
                "severe_baseline_events": 50.0,
                "severe_log_sd": value,
            }
        )
    return rows


def _simulate_features(
    latent: pd.DataFrame,
    setting: dict[str, object],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    design_id = (
        f"frontier_{setting['frontier_type']}_"
        f"{str(setting['parameter_value']).replace('.', 'p')}"
    )
    design = MeasurementDesign(
        design_id=design_id,
        age_concentration=100.0,
        testing_log_sd=float(setting["testing_log_sd"]),
        severe_baseline_events=float(setting["severe_baseline_events"]),
        severe_log_sd=float(setting["severe_log_sd"]),
        serology_n_per_stratum=300,
    )
    rows: list[dict[str, object]] = []
    for row_number, (_, latent_row) in enumerate(latent.iterrows()):
        for replicate in range(1, replicates + 1):
            rows.append(
                simulate_observation_features(
                    latent_row,
                    design=design,
                    seed=seed + row_number * 10_007 + replicate * 101,
                    observation_replicate=replicate,
                )
            )
    return pd.DataFrame(rows)


def _evaluate_setting(
    latent: pd.DataFrame,
    setting: dict[str, object],
    *,
    replicates: int,
    folds: int,
    bootstrap: int,
    seed: int,
    setting_number: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    features = _simulate_features(latent, setting, replicates=replicates, seed=seed)
    features["truth_active"] = features["scenario_id"].eq(POSITIVE_STRUCTURE).astype(int)
    stream = str(setting["stream"])
    feature_columns = feature_columns_for_package(features, (stream,))
    predictions = cross_validated_predictions(
        features,
        feature_columns,
        n_folds=folds,
    )
    metrics = discrimination_metrics(
        predictions["truth_active"],
        predictions["probability_active"],
    )
    lower, upper = nested_country_draw_bootstrap_auc(
        predictions,
        n_bootstrap=bootstrap,
        seed=seed + setting_number * 1009,
    )
    country_rows: list[dict[str, object]] = []
    for iso3, country in predictions.groupby("iso3", sort=True):
        country_metrics = discrimination_metrics(
            country["truth_active"],
            country["probability_active"],
        )
        country_rows.append(
            {
                "frontier_type": setting["frontier_type"],
                "parameter_value": setting["parameter_value"],
                "stream": stream,
                "iso3": iso3,
                "auc": country_metrics["auc"],
                "balanced_accuracy": country_metrics["balanced_accuracy"],
                "n_country_draw_groups": country["group_id"].nunique(),
                "n_observations": len(country),
                "frontier_version": FRONTIER_VERSION,
            }
        )
    country_frame = pd.DataFrame(country_rows)
    n_country_success = int(country_frame["auc"].ge(0.70).sum())
    pooled_success = bool(metrics["auc"] >= 0.80 and n_country_success >= 6)
    conservative_success = bool(lower >= 0.80 and n_country_success >= 6)
    row = {
        **setting,
        "package_id": f"notifications+{stream}",
        "n_features": len(feature_columns),
        "n_country_draw_groups": predictions["group_id"].nunique(),
        "n_observations": len(predictions),
        **metrics,
        "auc_lower_95": lower,
        "auc_upper_95": upper,
        "n_countries_auc_ge_0_70": n_country_success,
        "operational_success": pooled_success,
        "conservative_success": conservative_success,
        "multiplicative_error_factor": (
            float(np.exp(float(setting["parameter_value"])))
            if setting["frontier_type"] == "testing_log_sd"
            else np.nan
        ),
        "frontier_version": FRONTIER_VERSION,
    }
    return row, country_rows


def _frontier_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frontier_type, group in results.groupby("frontier_type", sort=False):
        if frontier_type == "severe_baseline_events":
            direction = "minimum"
            choose = pd.Series.min
        else:
            direction = "maximum"
            choose = pd.Series.max
        row: dict[str, object] = {
            "frontier_type": frontier_type,
            "threshold_direction": direction,
            "parameter_units": group["parameter_units"].iloc[0],
            "grid_min": group["parameter_value"].min(),
            "grid_max": group["parameter_value"].max(),
            "operational_threshold": np.nan,
            "conservative_threshold": np.nan,
            "frontier_version": FRONTIER_VERSION,
        }
        operational = group.loc[group["operational_success"], "parameter_value"]
        conservative = group.loc[group["conservative_success"], "parameter_value"]
        if not operational.empty:
            row["operational_threshold"] = float(choose(operational))
        if not conservative.empty:
            row["conservative_threshold"] = float(choose(conservative))
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.replicates < 1 or args.folds < 2 or args.bootstrap < 1:
        raise SystemExit("Replicates, folds and bootstrap must be positive.")
    output_dir = ROOT / args.experiment_dir
    latent = pd.read_csv(output_dir / "latent_surveillance_signals.csv")
    latent = (
        latent.loc[latent["scenario_id"].isin([POSITIVE_STRUCTURE, NEGATIVE_STRUCTURE])]
        .sort_values(["iso3", "draw_rank", "scenario_id"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    settings = _settings()
    for setting_number, setting in enumerate(settings):
        row, country = _evaluate_setting(
            latent,
            setting,
            replicates=args.replicates,
            folds=args.folds,
            bootstrap=args.bootstrap,
            seed=args.seed,
            setting_number=setting_number,
        )
        rows.append(row)
        country_rows.extend(country)
        print(
            f"Evaluated {setting['frontier_type']}={setting['parameter_value']}: "
            f"AUC={row['auc']:.3f}",
            flush=True,
        )
    results = pd.DataFrame(rows)
    country = pd.DataFrame(country_rows)
    summary = _frontier_summary(results)
    results.to_csv(output_dir / "precision_frontier.csv", index=False)
    country.to_csv(output_dir / "precision_frontier_country.csv", index=False)
    summary.to_csv(output_dir / "precision_frontier_summary.csv", index=False)
    (output_dir / "precision_frontier.md").write_text(
        "# One-parameter surveillance precision frontiers\n\n"
        + _markdown_table(
            results[
                [
                    "frontier_type",
                    "parameter_value",
                    "auc",
                    "auc_lower_95",
                    "auc_upper_95",
                    "n_countries_auc_ge_0_70",
                    "operational_success",
                    "conservative_success",
                ]
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} precision-frontier settings.")


if __name__ == "__main__":
    main()
