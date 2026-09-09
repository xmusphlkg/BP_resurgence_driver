#!/usr/bin/env python3
"""Build reviewer-facing robustness summaries for surveillance design outputs."""

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
    PASSING_PRIMARY_COUNTRIES,
    SURVEILLANCE_BOOTSTRAP_REPLICATES,
    SURVEILLANCE_DESIGN_VERSION,
    cluster_bootstrap_auc,
    cross_validated_predictions,
    discrimination_metrics,
    leave_one_country_out_predictions,
    roc_auc,
)


MATCHED_STRUCTURES = ("matched_transmission_truth", "matched_reporting_truth")
KEY_PACKAGES = (
    "notifications",
    "notifications+age",
    "notifications+testing",
    "notifications+severe",
    "notifications+serology",
    "notifications+age+testing+severe+serology",
)


def _matched_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[frame["scenario_id"].isin(MATCHED_STRUCTURES)].copy()
    selected["truth_active"] = selected["scenario_id"].eq("matched_transmission_truth").astype(int)
    return selected


def direct_signal_summary(
    output_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design_number, design_id in enumerate(("low_precision", "main", "high_precision")):
        features = _matched_features(output_dir / f"observation_features_{design_id}.csv")
        populations = {
            "all_nine": set(features["iso3"]),
            "converged_six": set(PASSING_PRIMARY_COUNTRIES),
        }
        score_definitions = {
            "severe_2024": features["feature_severe_log_ratio_2024"],
            "mean_severe_2023_2025": features[
                [f"feature_severe_log_ratio_{year}" for year in (2023, 2024, 2025)]
            ].mean(axis=1),
            "inverse_testing_2024": -features["feature_testing_log_index_2024"],
            "inverse_mean_testing_2023_2025": -features[
                [f"feature_testing_log_index_{year}" for year in (2023, 2024, 2025)]
            ].mean(axis=1),
        }
        for population_number, (population_id, countries) in enumerate(populations.items()):
            mask = features["iso3"].isin(countries)
            for score_number, (score_id, score) in enumerate(score_definitions.items()):
                part = features.loc[mask, ["group_id", "truth_active"]].copy()
                part["probability_active"] = score.loc[mask].to_numpy(dtype=float)
                auc = roc_auc(part["truth_active"], part["probability_active"])
                if design_id == "main" and population_id == "all_nine":
                    lower, upper = cluster_bootstrap_auc(
                        part,
                        n_bootstrap=n_bootstrap,
                        seed=seed + design_number * 10_007 + population_number * 1_009 + score_number,
                    )
                else:
                    lower, upper = np.nan, np.nan
                rows.append(
                    {
                        "measurement_design": design_id,
                        "analysis_population": population_id,
                        "direct_signal": score_id,
                        "auc": auc,
                        "auc_lower_95": lower,
                        "auc_upper_95": upper,
                        "n_country_draw_groups": part["group_id"].nunique(),
                        "n_observations": len(part),
                        "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
                    }
                )
    return pd.DataFrame(rows)


def classifier_sensitivity(output_dir: Path) -> pd.DataFrame:
    features = _matched_features(output_dir / "observation_features_main.csv")
    package_results = pd.read_csv(output_dir / "package_discrimination.csv")
    rows: list[dict[str, object]] = []
    for population_id, countries in {
        "all_nine": set(features["iso3"]),
        "converged_six": set(PASSING_PRIMARY_COUNTRIES),
    }.items():
        part = features.loc[features["iso3"].isin(countries)].copy()
        for package in KEY_PACKAGES:
            package_row = package_results.loc[
                package_results["analysis_population"].eq(population_id)
                & package_results["measurement_design"].eq("main")
                & package_results["mechanism"].eq("transmission_vs_reporting")
                & package_results["package_id"].eq(package)
            ].iloc[0]
            streams = tuple(filter(None, str(package_row["package_streams"]).split("|")))
            prefixes = ["feature_notifications_", *(f"feature_{stream}_" for stream in streams)]
            columns = [
                column
                for column in part.columns
                if any(column.startswith(prefix) for prefix in prefixes)
            ]
            for shrinkage in (0.50, 0.80, 1.00):
                predictions = cross_validated_predictions(
                    part,
                    columns,
                    n_folds=5,
                    shrinkage=shrinkage,
                )
                metrics = discrimination_metrics(
                    predictions["truth_active"],
                    predictions["probability_active"],
                )
                rows.append(
                    {
                        "analysis_population": population_id,
                        "measurement_design": "main",
                        "package_id": package,
                        "classifier": "shrinkage_linear_discriminant",
                        "covariance_diagonal_shrinkage": shrinkage,
                        "n_features": len(columns),
                        **metrics,
                        "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
                    }
                )
    return pd.DataFrame(rows)


def country_holdout_sensitivity(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether the main result transports to a country absent from training."""
    features = _matched_features(output_dir / "observation_features_main.csv")
    package_results = pd.read_csv(output_dir / "package_discrimination.csv")
    pooled_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    for package in KEY_PACKAGES:
        package_row = package_results.loc[
            package_results["analysis_population"].eq("all_nine")
            & package_results["measurement_design"].eq("main")
            & package_results["mechanism"].eq("transmission_vs_reporting")
            & package_results["package_id"].eq(package)
        ].iloc[0]
        streams = tuple(filter(None, str(package_row["package_streams"]).split("|")))
        prefixes = ["feature_notifications_", *(f"feature_{stream}_" for stream in streams)]
        columns = [
            column
            for column in features.columns
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
        predictions = leave_one_country_out_predictions(
            features,
            columns,
            shrinkage=0.80,
        )
        metrics = discrimination_metrics(
            predictions["truth_active"],
            predictions["probability_active"],
        )
        per_country: list[float] = []
        for iso3, country in predictions.groupby("iso3", sort=True):
            country_metrics = discrimination_metrics(
                country["truth_active"],
                country["probability_active"],
            )
            per_country.append(float(country_metrics["auc"]))
            country_rows.append(
                {
                    "measurement_design": "main",
                    "package_id": package,
                    "iso3": iso3,
                    "validation_scheme": "leave_one_country_out",
                    "n_country_draw_groups": country["group_id"].nunique(),
                    "n_observations": len(country),
                    **country_metrics,
                    "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
                }
            )
        pooled_rows.append(
            {
                "measurement_design": "main",
                "package_id": package,
                "validation_scheme": "leave_one_country_out",
                "n_features": len(columns),
                "n_countries": len(per_country),
                "n_countries_auc_ge_0_70": int(np.sum(np.asarray(per_country) >= 0.70)),
                "median_country_auc": float(np.median(per_country)),
                "minimum_country_auc": float(np.min(per_country)),
                "maximum_country_auc": float(np.max(per_country)),
                "n_country_draw_groups": predictions["group_id"].nunique(),
                "n_observations": len(predictions),
                **metrics,
                "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
            }
        )
    return pd.DataFrame(pooled_rows), pd.DataFrame(country_rows)


def key_result_summary(output_dir: Path) -> pd.DataFrame:
    package = pd.read_csv(output_dir / "package_discrimination.csv")
    selected = package.loc[
        package["analysis_population"].eq("all_nine")
        & package["mechanism"].eq("transmission_vs_reporting")
        & package["package_id"].isin(KEY_PACKAGES)
    ].copy()
    selected["result_family"] = "matched_notification_package"
    return selected[
        [
            "result_family",
            "measurement_design",
            "package_id",
            "auc",
            "auc_lower_95",
            "auc_upper_95",
            "balanced_accuracy",
            "brier_score",
            "log_loss",
            "n_country_draw_groups",
            "n_observations",
            "surveillance_design_version",
        ]
    ].sort_values(["measurement_design", "package_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="outputs/model2c_surveillance_design/monitoring_packages_v4_source_preserved_20260908",
    )
    parser.add_argument("--bootstrap", type=int, default=SURVEILLANCE_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.experiment_dir
    if not output_dir.exists():
        raise SystemExit(f"Missing surveillance-design experiment: {output_dir}")
    direct = direct_signal_summary(output_dir, n_bootstrap=args.bootstrap, seed=args.seed)
    classifier = classifier_sensitivity(output_dir)
    country_holdout, country_holdout_detail = country_holdout_sensitivity(output_dir)
    key = key_result_summary(output_dir)
    direct.to_csv(output_dir / "direct_signal_discrimination.csv", index=False)
    classifier.to_csv(output_dir / "classifier_sensitivity.csv", index=False)
    country_holdout.to_csv(output_dir / "country_holdout_sensitivity.csv", index=False)
    country_holdout_detail.to_csv(output_dir / "country_holdout_country_discrimination.csv", index=False)
    key.to_csv(output_dir / "key_result_summary.csv", index=False)
    print(f"Wrote surveillance-design robustness summaries to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
