#!/usr/bin/env python3
"""Build nested country/draw AUC intervals for the matched NC benchmark.

The analysis reuses the frozen main-design observation features and computes
five-fold grouped-CV probabilities once per surveillance package. Uncertainty
is then evaluated without refitting: countries are sampled first and paired
country-draw groups are sampled second, retaining all scenario and observation
replicate rows attached to each selected group.
"""

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
    SURVEILLANCE_BOOTSTRAP_REPLICATES,
    SURVEILLANCE_BOOTSTRAP_SEED,
    cross_validated_predictions,
    feature_columns_for_package,
    nested_country_draw_bootstrap_auc,
    package_id,
    roc_auc,
    surveillance_packages,
)


OUTPUT_PATH = ROOT / "outputs" / "nc_country_draw_nested_auc.csv"
TABLE_PATH = ROOT / "manuscript" / "tables" / "nc_country_draw_nested_auc.csv"

ANALYSIS_VERSION = "nc_country_draw_nested_auc_v2_20260903"
MECHANISM = "transmission_vs_reporting"
POSITIVE_SCENARIO = "matched_transmission_truth"
NEGATIVE_SCENARIO = "matched_reporting_truth"
N_FOLDS = 5
SHRINKAGE = 0.80
BOOTSTRAP_REPLICATES = SURVEILLANCE_BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = SURVEILLANCE_BOOTSTRAP_SEED
EXPECTED_COUNTRIES = 9
EXPECTED_DRAWS_PER_COUNTRY = 80
EXPECTED_OBSERVATION_REPLICATES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="outputs/model2c_surveillance_design/monitoring_packages_v4_source_preserved_20260908",
        help="Frozen surveillance-design experiment directory, relative to the repository root.",
    )
    return parser.parse_args()


def matched_features(path: Path) -> pd.DataFrame:
    """Load and validate the frozen matched main-design feature rows."""
    features = pd.read_csv(path)
    required = {
        "iso3",
        "draw_id",
        "group_id",
        "scenario_id",
        "observation_replicate",
        "measurement_design",
        "surveillance_design_version",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise RuntimeError(
            "Observation features are missing required columns: "
            + ", ".join(missing)
        )
    selected = features.loc[
        features["scenario_id"].isin([POSITIVE_SCENARIO, NEGATIVE_SCENARIO])
    ].copy()
    selected["truth_active"] = (
        selected["scenario_id"].eq(POSITIVE_SCENARIO).astype(int)
    )

    if set(selected["measurement_design"].astype(str)) != {"main"}:
        raise RuntimeError("Expected only the main observation measurement design")
    countries = sorted(selected["iso3"].astype(str).unique())
    if len(countries) != EXPECTED_COUNTRIES:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNTRIES} countries, found {len(countries)}"
        )
    group_structure = selected.groupby("group_id", sort=False).agg(
        n_countries=("iso3", "nunique"),
        n_draw_ids=("draw_id", "nunique"),
        n_scenarios=("scenario_id", "nunique"),
    )
    if len(group_structure) != EXPECTED_COUNTRIES * EXPECTED_DRAWS_PER_COUNTRY:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_COUNTRIES * EXPECTED_DRAWS_PER_COUNTRY} "
            "paired country-draw groups"
        )
    if not (
        group_structure["n_countries"].eq(1).all()
        and group_structure["n_draw_ids"].eq(1).all()
        and group_structure["n_scenarios"].eq(2).all()
    ):
        raise RuntimeError(
            "Every group_id must identify one country-draw pair and both scenarios"
        )
    groups_per_country = (
        selected[["iso3", "group_id"]]
        .drop_duplicates()
        .groupby("iso3")["group_id"]
        .nunique()
    )
    if not groups_per_country.eq(EXPECTED_DRAWS_PER_COUNTRY).all():
        raise RuntimeError(
            f"Expected {EXPECTED_DRAWS_PER_COUNTRY} draw groups per country"
        )
    replicate_counts = selected.groupby(
        ["group_id", "scenario_id"],
        sort=False,
    )["observation_replicate"].nunique()
    if not replicate_counts.eq(EXPECTED_OBSERVATION_REPLICATES).all():
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_OBSERVATION_REPLICATES} observation replicates "
            "for every country-draw-scenario"
        )
    expected_rows = (
        EXPECTED_COUNTRIES
        * EXPECTED_DRAWS_PER_COUNTRY
        * 2
        * EXPECTED_OBSERVATION_REPLICATES
    )
    if len(selected) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} matched observation rows")
    return selected.sort_values(
        ["iso3", "draw_id", "scenario_id", "observation_replicate"]
    ).reset_index(drop=True)


def build_results(
    features: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Evaluate all 16 packages using fixed grouped-CV predictions."""
    rows: list[dict[str, object]] = []
    country_draw_counts = (
        features[["iso3", "group_id"]]
        .drop_duplicates()
        .groupby("iso3")["group_id"]
        .nunique()
    )
    surveillance_versions = sorted(
        features["surveillance_design_version"].astype(str).unique()
    )
    if len(surveillance_versions) != 1:
        raise RuntimeError("Expected one surveillance design version")

    for streams in surveillance_packages():
        feature_columns = feature_columns_for_package(features, streams)
        predictions = cross_validated_predictions(
            features,
            feature_columns,
            n_folds=N_FOLDS,
            shrinkage=SHRINKAGE,
        )
        pooled_auc = roc_auc(
            predictions["truth_active"],
            predictions["probability_active"],
        )
        lower, upper = nested_country_draw_bootstrap_auc(
            predictions,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        country_aucs = np.asarray(
            [
                roc_auc(
                    country["truth_active"],
                    country["probability_active"],
                )
                for _, country in predictions.groupby("iso3", sort=True)
            ],
            dtype=float,
        )
        rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "mechanism": MECHANISM,
                "analysis_population": "all_nine",
                "positive_structure_id": POSITIVE_SCENARIO,
                "negative_structure_id": NEGATIVE_SCENARIO,
                "measurement_design": "main",
                "validation_scheme": "fixed_5_fold_grouped_cv_predictions",
                "uncertainty_method": (
                    "country_first_draw_second_nested_bootstrap"
                ),
                "package_id": package_id(streams),
                "package_streams": "|".join(streams),
                "n_optional_streams": len(streams),
                "n_features": len(feature_columns),
                "n_countries": int(predictions["iso3"].nunique()),
                "n_country_draw_groups": int(predictions["group_id"].nunique()),
                "min_draw_groups_per_country": int(country_draw_counts.min()),
                "max_draw_groups_per_country": int(country_draw_counts.max()),
                "n_observations": len(predictions),
                "n_folds": N_FOLDS,
                "shrinkage": SHRINKAGE,
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": seed,
                "pooled_auc": pooled_auc,
                "nested_auc_lower_95": lower,
                "nested_auc_upper_95": upper,
                "country_auc_median": float(np.median(country_aucs)),
                "country_auc_minimum": float(np.min(country_aucs)),
                "country_auc_maximum": float(np.max(country_aucs)),
                "n_countries_auc_ge_0_70": int(np.sum(country_aucs >= 0.70)),
                "surveillance_design_version": surveillance_versions[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_optional_streams", "package_id"]
    ).reset_index(drop=True)


def validate_results(results: pd.DataFrame) -> None:
    """Check the generated table before publishing both CSV copies."""
    if len(results) != len(surveillance_packages()):
        raise RuntimeError("Nested AUC output must contain all 16 packages")
    if results["package_id"].nunique() != len(surveillance_packages()):
        raise RuntimeError("Nested AUC package identifiers must be unique")
    if not results["n_countries"].eq(EXPECTED_COUNTRIES).all():
        raise RuntimeError("Nested AUC output has an unexpected country count")
    expected_groups = EXPECTED_COUNTRIES * EXPECTED_DRAWS_PER_COUNTRY
    if not results["n_country_draw_groups"].eq(expected_groups).all():
        raise RuntimeError("Nested AUC output has an unexpected draw-group count")
    if not results["n_bootstrap"].eq(BOOTSTRAP_REPLICATES).all():
        raise RuntimeError("Nested AUC output must use 2000 bootstrap replicates")
    if not results["bootstrap_seed"].eq(BOOTSTRAP_SEED).all():
        raise RuntimeError("Nested AUC output has an unexpected bootstrap seed")
    auc_columns = [
        "pooled_auc",
        "nested_auc_lower_95",
        "nested_auc_upper_95",
        "country_auc_median",
        "country_auc_minimum",
        "country_auc_maximum",
    ]
    values = results[auc_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise RuntimeError("Nested AUC summaries must be finite and lie in [0, 1]")
    if not (
        results["nested_auc_lower_95"]
        .le(results["nested_auc_upper_95"])
        .all()
    ):
        raise RuntimeError("Nested AUC interval bounds are reversed")
    if not (
        results["country_auc_minimum"]
        .le(results["country_auc_median"])
        .all()
        and results["country_auc_median"]
        .le(results["country_auc_maximum"])
        .all()
    ):
        raise RuntimeError("Country AUC summaries are not ordered")


def main() -> None:
    args = parse_args()
    experiment_dir = ROOT / args.experiment_dir
    input_path = experiment_dir / "observation_features_main.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing frozen observation features: {input_path}")
    features = matched_features(input_path)
    results = build_results(features)
    validate_results(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    results.to_csv(TABLE_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {TABLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
