#!/usr/bin/env python3
"""Post-primary stress test for cross-loading between auxiliary streams.

For each matched country-draw trajectory and post-pandemic year, define

    T = latent testing log-index
    S = log(severe source / scenario-specific severe baseline).

Testing and severe measurements can then cross-load the other process:

    T' = T + gamma_t S
    S' = S + gamma_s T.

Both inputs on the right-hand side are the original frozen signals, so the
transformation is simultaneous and has no ordering effect. T and S are
dimensionless natural-link signals. Gamma is a modelled cross-loading fraction,
not an empirically estimated quantity. Gamma=0 reproduces the original streams.
When gamma_t=gamma_s=1, both transformed streams equal T+S. Because matched
notifications were constructed from approximately opposing transmission and
ascertainment contrasts, T+S is notification-like and contains little
remaining class contrast. Values above 1 are not analysed because they would
extrapolate beyond complete cross-loading and can reverse the dominant
contrast.

The analysis evaluates testing-only, severe-only, and both-stream
cross-loading under the main measurement precision. It also evaluates the
combined testing-plus-severe package over the complete Cartesian grid of the
two cross-loading coefficients. It is a measurement-specificity stress test,
not a dynamic mixed-mechanism epidemic generator or an estimate of historical
cross-loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.bayes.model2c_surveillance_design import (  # noqa: E402
    MEASUREMENT_DESIGNS,
    SURVEILLANCE_BOOTSTRAP_REPLICATES,
    nested_country_draw_bootstrap_auc,
    cross_validated_predictions,
    discrimination_metrics,
    feature_columns_for_package,
    leave_one_country_out_predictions,
    simulate_observation_features,
)


INPUT_PATH = (
    ROOT
    / "outputs"
    / "model2c_surveillance_design"
    / "monitoring_packages_v4_source_preserved_20260908"
    / "latent_surveillance_signals.csv"
)
OUTPUT_PATH = ROOT / "outputs" / "nc_measurement_contamination_frontier.csv"
TABLE_PATH = (
    ROOT / "manuscript" / "tables" / "nc_measurement_contamination_frontier.csv"
)
SURFACE_OUTPUT_PATH = (
    ROOT / "outputs" / "nc_measurement_contamination_surface.csv"
)
SURFACE_TABLE_PATH = (
    ROOT / "manuscript" / "tables" / "nc_measurement_contamination_surface.csv"
)
MARKDOWN_PATH = (
    ROOT / "manuscript" / "tables_md" / "nc_measurement_contamination_frontier.md"
)

ANALYSIS_VERSION = "nc_measurement_contamination_v2_source_preserved_infant_20260908"
SURFACE_ANALYSIS_VERSION = "nc_measurement_contamination_surface_v2_source_preserved_infant_20260908"
POSITIVE_SCENARIO = "matched_transmission_truth"
NEGATIVE_SCENARIO = "matched_reporting_truth"
CONTAMINATION_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
CONTAMINATION_REGIMES: dict[str, tuple[bool, bool]] = {
    "both_contaminated": (True, True),
    "testing_only": (True, False),
    "severe_only": (False, True),
}
PACKAGES: dict[str, tuple[str, ...]] = {
    "notifications": (),
    "notifications+testing": ("testing",),
    "notifications+severe": ("severe",),
    "notifications+testing+severe": ("testing", "severe"),
}
OBSERVATION_REPLICATES = 3
BASE_SEED = 20260727


def matched_latent() -> pd.DataFrame:
    latent = pd.read_csv(INPUT_PATH)
    selected = latent.loc[
        latent["scenario_id"].isin([POSITIVE_SCENARIO, NEGATIVE_SCENARIO])
    ].copy()
    counts = selected.groupby("group_id")["scenario_id"].nunique()
    if len(counts) != 720 or not counts.eq(2).all():
        raise RuntimeError(
            "Expected 720 paired country-draw groups with both matched scenarios"
        )
    notification_columns = [
        column
        for column in selected.columns
        if column.startswith("latent_notification_mean_")
    ]
    indexed = selected.set_index(["group_id", "scenario_id"])[notification_columns]
    differences = (
        indexed.groupby(level="group_id").max()
        - indexed.groupby(level="group_id").min()
    )
    if not np.allclose(differences.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-9):
        raise RuntimeError("Matched latent notifications are no longer exactly equal")
    return selected.sort_values(
        ["iso3", "draw_rank", "scenario_id"]
    ).reset_index(drop=True)


def contaminate_latent(
    latent: pd.DataFrame,
    *,
    testing_gamma: float,
    severe_gamma: float,
) -> pd.DataFrame:
    if not 0.0 <= testing_gamma <= 1.0:
        raise ValueError("Testing contamination must lie in [0, 1]")
    if not 0.0 <= severe_gamma <= 1.0:
        raise ValueError("Severe contamination must lie in [0, 1]")

    rows: list[pd.Series] = []
    for _, pair in latent.groupby("group_id", sort=True):
        positive = pair.loc[pair["scenario_id"].eq(POSITIVE_SCENARIO)].iloc[0].copy()
        negative = pair.loc[pair["scenario_id"].eq(NEGATIVE_SCENARIO)].iloc[0].copy()

        positive_baseline = max(
            float(positive["latent_severe_source_baseline"]),
            1e-12,
        )
        negative_baseline = max(
            float(negative["latent_severe_source_baseline"]),
            1e-12,
        )
        for year in (2023, 2024, 2025):
            testing_column = f"latent_testing_log_index_{year}"
            severe_column = f"latent_severe_source_{year}"
            positive_testing = float(positive[testing_column])
            negative_testing = float(negative[testing_column])
            positive_log_ratio = np.log(
                max(float(positive[severe_column]), 1e-12) / positive_baseline
            )
            negative_log_ratio = np.log(
                max(float(negative[severe_column]), 1e-12) / negative_baseline
            )

            positive[testing_column] = float(
                positive_testing + testing_gamma * positive_log_ratio
            )
            negative[testing_column] = float(
                negative_testing + testing_gamma * negative_log_ratio
            )
            positive[severe_column] = float(
                positive_baseline
                * np.exp(positive_log_ratio + severe_gamma * positive_testing)
            )
            negative[severe_column] = float(
                negative_baseline
                * np.exp(negative_log_ratio + severe_gamma * negative_testing)
            )

        rows.extend([positive, negative])

    contaminated = pd.DataFrame(rows).sort_values(
        ["iso3", "draw_rank", "scenario_id"]
    ).reset_index(drop=True)
    return contaminated


def simulate_features(latent: pd.DataFrame) -> pd.DataFrame:
    design = MEASUREMENT_DESIGNS["main"]
    rows: list[dict[str, object]] = []
    for row_number, (_, row) in enumerate(latent.iterrows()):
        for replicate in range(1, OBSERVATION_REPLICATES + 1):
            seed = BASE_SEED + row_number * 10_007 + replicate * 101
            rows.append(
                simulate_observation_features(
                    row,
                    design=design,
                    seed=seed,
                    observation_replicate=replicate,
                )
            )
    features = pd.DataFrame(rows).sort_values(
        ["iso3", "draw_rank", "scenario_id", "observation_replicate"]
    )
    features["truth_active"] = features["scenario_id"].eq(POSITIVE_SCENARIO).astype(int)
    return features


def signal_summary(latent: pd.DataFrame) -> dict[str, float]:
    indexed = latent.set_index(["group_id", "scenario_id"])
    positive = indexed.xs(POSITIVE_SCENARIO, level="scenario_id")
    negative = indexed.xs(NEGATIVE_SCENARIO, level="scenario_id")
    positive_severe_ratio = (
        positive["latent_severe_source_2024"]
        / positive["latent_severe_source_baseline"]
    )
    negative_severe_ratio = (
        negative["latent_severe_source_2024"]
        / negative["latent_severe_source_baseline"]
    )
    testing_ratio = np.exp(
        positive["latent_testing_log_index_2024"]
        - negative["latent_testing_log_index_2024"]
    )
    residual_ratio = np.exp(
        (
            positive["latent_testing_log_index_2024"]
            + np.log(positive_severe_ratio)
        )
        - (
            negative["latent_testing_log_index_2024"]
            + np.log(negative_severe_ratio)
        )
    )
    return {
        "median_2024_infant_severe_signal_ratio": float(
            np.median(positive_severe_ratio / negative_severe_ratio)
        ),
        "median_2024_ascertainment_ratio": float(np.median(testing_ratio)),
        "median_2024_testing_plus_severe_residual_ratio": float(
            np.median(residual_ratio)
        ),
        "max_abs_2024_testing_link_contrast": float(
            np.max(
                np.abs(
                    positive["latent_testing_log_index_2024"]
                    - negative["latent_testing_log_index_2024"]
                )
            )
        ),
        "max_abs_2024_severe_link_contrast": float(
            np.max(np.abs(np.log(positive_severe_ratio / negative_severe_ratio)))
        ),
    }


def evaluate_package(
    features: pd.DataFrame,
    *,
    package: str,
    streams: tuple[str, ...],
    bootstrap_seed: int,
) -> dict[str, object]:
    columns = feature_columns_for_package(features, streams)
    grouped = cross_validated_predictions(
        features,
        columns,
        n_folds=5,
        shrinkage=0.80,
    )
    grouped_metrics = discrimination_metrics(
        grouped["truth_active"],
        grouped["probability_active"],
    )
    lower, upper = nested_country_draw_bootstrap_auc(
        grouped,
        n_bootstrap=SURVEILLANCE_BOOTSTRAP_REPLICATES,
        seed=bootstrap_seed,
    )
    country_aucs = [
        discrimination_metrics(
            country["truth_active"],
            country["probability_active"],
        )["auc"]
        for _, country in grouped.groupby("iso3", sort=True)
    ]

    country_holdout = leave_one_country_out_predictions(
        features,
        columns,
        shrinkage=0.80,
    )
    holdout_metrics = discrimination_metrics(
        country_holdout["truth_active"],
        country_holdout["probability_active"],
    )
    holdout_country_aucs = [
        discrimination_metrics(
            country["truth_active"],
            country["probability_active"],
        )["auc"]
        for _, country in country_holdout.groupby("iso3", sort=True)
    ]
    return {
        "package_id": package,
        "n_features": len(columns),
        "grouped_cv_auc": grouped_metrics["auc"],
        "grouped_cv_auc_lower_95": lower,
        "grouped_cv_auc_upper_95": upper,
        "grouped_cv_balanced_accuracy": grouped_metrics["balanced_accuracy"],
        "grouped_cv_countries_auc_ge_0_70": int(
            np.sum(np.asarray(country_aucs) >= 0.70)
        ),
        "whole_country_holdout_auc": holdout_metrics["auc"],
        "whole_country_holdout_countries_auc_ge_0_70": int(
            np.sum(np.asarray(holdout_country_aucs) >= 0.70)
        ),
    }


def effective_metric_key(
    package: str,
    testing_gamma: float,
    severe_gamma: float,
) -> tuple[str, float, float]:
    return (
        package,
        testing_gamma if "testing" in package else 0.0,
        severe_gamma if "severe" in package else 0.0,
    )


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.loc[
        ~frame["package_id"].eq("notifications"),
        [
            "contamination_regime",
            "contamination_fraction",
            "median_2024_infant_severe_signal_ratio",
            "median_2024_ascertainment_ratio",
            "package_id",
            "grouped_cv_auc",
            "grouped_cv_auc_lower_95",
            "grouped_cv_auc_upper_95",
            "whole_country_holdout_auc",
        ],
    ].copy()
    display["Grouped CV AUC (95% interval)"] = display.apply(
        lambda row: (
            f"{row['grouped_cv_auc']:.3f} "
            f"({row['grouped_cv_auc_lower_95']:.3f}–"
            f"{row['grouped_cv_auc_upper_95']:.3f})"
        ),
        axis=1,
    )
    display = display[
        [
            "contamination_regime",
            "contamination_fraction",
            "median_2024_infant_severe_signal_ratio",
            "median_2024_ascertainment_ratio",
            "package_id",
            "Grouped CV AUC (95% interval)",
            "whole_country_holdout_auc",
        ]
    ]
    display.columns = [
        "Contamination regime",
        "γ",
        "Median severe-signal ratio",
        "Median ascertainment ratio",
        "Package",
        "Grouped CV AUC (95% interval)",
        "Whole-country-holdout AUC",
    ]
    for column in [
        "γ",
        "Median severe-signal ratio",
        "Median ascertainment ratio",
        "Whole-country-holdout AUC",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[column]).replace("|", "\\|") for column in columns
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def validate(
    frame: pd.DataFrame,
    original: pd.DataFrame,
) -> None:
    zero = frame.loc[frame["contamination_fraction"].eq(0.0)]
    for package in PACKAGES:
        values = zero.loc[zero["package_id"].eq(package), "grouped_cv_auc"]
        if len(values) != 3 or float(values.max() - values.min()) > 1e-12:
            raise RuntimeError(f"Gamma-zero results differ across regimes for {package}")

    # Check measurement invariance, not a prespecified performance outcome.
    # Source ascertainment plus infant burden need not be class-invariant at
    # rank one: loss of two-dimensional identifiability does not imply chance AUC.
    for regime, guard_package in [
        ("testing_only", "notifications+severe"),
        ("severe_only", "notifications+testing"),
    ]:
        values = frame.loc[
            frame["contamination_regime"].eq(regime)
            & frame["package_id"].eq(guard_package), "grouped_cv_auc"
        ]
        if float(values.max() - values.min()) > 1e-12:
            raise RuntimeError(f"An unchanged guard stream changed across gamma: {regime}")
    for column in ["grouped_cv_auc", "whole_country_holdout_auc"]:
        if not frame[column].between(0, 1).all():
            raise RuntimeError(f"Invalid discrimination metric: {column}")

    reconstructed = contaminate_latent(
        original,
        testing_gamma=0.0,
        severe_gamma=0.0,
    )
    numeric_columns = [
        column
        for column in original.columns
        if column.startswith("latent_testing_log_index_")
        or column.startswith("latent_severe_source_")
    ]
    if not np.allclose(
        reconstructed[numeric_columns].to_numpy(dtype=float),
        original[numeric_columns].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Gamma-zero transformation did not reproduce frozen signals")

    both_full_latent = contaminate_latent(
        original,
        testing_gamma=1.0,
        severe_gamma=1.0,
    )
    for year in (2023, 2024, 2025):
        transformed_testing = both_full_latent[
            f"latent_testing_log_index_{year}"
        ].to_numpy(dtype=float)
        transformed_severe = np.log(
            both_full_latent[f"latent_severe_source_{year}"].to_numpy(dtype=float)
            / both_full_latent["latent_severe_source_baseline"].to_numpy(dtype=float)
        )
        if not np.allclose(
            transformed_testing,
            transformed_severe,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Both-stream gamma-one transformation is not T+S in {year}"
            )


def validate_surface(surface: pd.DataFrame) -> None:
    expected_grid = {
        (testing_gamma, severe_gamma)
        for testing_gamma in CONTAMINATION_LEVELS
        for severe_gamma in CONTAMINATION_LEVELS
    }
    observed_grid = set(
        zip(
            surface["testing_cross_loading_fraction"],
            surface["severe_cross_loading_fraction"],
        )
    )
    if len(surface) != len(expected_grid) or observed_grid != expected_grid:
        raise RuntimeError("The cross-loading surface is not the complete 5 x 5 grid")
    if surface[
        [
            "testing_cross_loading_fraction",
            "severe_cross_loading_fraction",
        ]
    ].duplicated().any():
        raise RuntimeError("The cross-loading surface contains duplicate grid cells")
    if not surface["package_id"].eq("notifications+testing+severe").all():
        raise RuntimeError("The cross-loading surface must use the combined package")
    for estimate, lower, upper in [
        (
            "grouped_cv_auc",
            "grouped_cv_auc_lower_95",
            "grouped_cv_auc_upper_95",
        )
    ]:
        if not (
            surface[lower].between(0, 1).all()
            and surface[estimate].between(0, 1).all()
            and surface[upper].between(0, 1).all()
            and (surface[lower] <= surface[estimate]).all()
            and (surface[estimate] <= surface[upper]).all()
        ):
            raise RuntimeError("The cross-loading surface contains invalid AUC intervals")

    # The rank-one corner is a measurement degeneracy, not an AUC target.


def main() -> None:
    original = matched_latent()
    feature_cache: dict[tuple[float, float], pd.DataFrame] = {}
    metric_cache: dict[tuple[str, float, float], dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for regime, (contaminate_testing, contaminate_severe) in CONTAMINATION_REGIMES.items():
        for gamma in CONTAMINATION_LEVELS:
            testing_gamma = gamma if contaminate_testing else 0.0
            severe_gamma = gamma if contaminate_severe else 0.0
            feature_key = (testing_gamma, severe_gamma)
            if feature_key not in feature_cache:
                latent = contaminate_latent(
                    original,
                    testing_gamma=testing_gamma,
                    severe_gamma=severe_gamma,
                )
                feature_cache[feature_key] = simulate_features(latent)
            else:
                latent = contaminate_latent(
                    original,
                    testing_gamma=testing_gamma,
                    severe_gamma=severe_gamma,
                )
            signal = signal_summary(latent)

            for package_number, (package, streams) in enumerate(PACKAGES.items()):
                metric_key = effective_metric_key(
                    package,
                    testing_gamma,
                    severe_gamma,
                )
                if metric_key not in metric_cache:
                    metric_cache[metric_key] = evaluate_package(
                        feature_cache[feature_key],
                        package=package,
                        streams=streams,
                        bootstrap_seed=(
                            BASE_SEED
                            + package_number * 10_003
                            + int(metric_key[1] * 1000) * 101
                            + int(metric_key[2] * 1000) * 1009
                        ),
                    )
                rows.append(
                    {
                        "contamination_regime": regime,
                        "contamination_fraction": gamma,
                        "testing_cross_loading_fraction": testing_gamma,
                        "severe_cross_loading_fraction": severe_gamma,
                        **signal,
                        **metric_cache[metric_key],
                        "analysis_version": ANALYSIS_VERSION,
                        "contamination_definition": (
                            "For each country-draw trajectory and year, "
                            "T'=T+gamma_t*S and S'=S+gamma_s*T, with both "
                            "right-hand inputs taken from the frozen signals."
                        ),
                        "interpretation_boundary": (
                            "Post-primary measurement-specificity stress test "
                            "on frozen matched trajectories; not a dynamic "
                            "mixed-mechanism generator, estimated historical "
                            "contamination fraction, or external validation."
                        ),
                    }
                )
            print(
                f"Evaluated {regime}, gamma={gamma:.2f}: "
                f"severe ratio={signal['median_2024_infant_severe_signal_ratio']:.3f}, "
                f"ascertainment ratio={signal['median_2024_ascertainment_ratio']:.3f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values(
        ["contamination_regime", "contamination_fraction", "package_id"]
    )
    validate(frame, original)

    surface_rows: list[dict[str, object]] = []
    combined_package = "notifications+testing+severe"
    combined_streams = PACKAGES[combined_package]
    combined_package_number = list(PACKAGES).index(combined_package)
    for testing_gamma in CONTAMINATION_LEVELS:
        for severe_gamma in CONTAMINATION_LEVELS:
            feature_key = (testing_gamma, severe_gamma)
            latent = contaminate_latent(
                original,
                testing_gamma=testing_gamma,
                severe_gamma=severe_gamma,
            )
            if feature_key not in feature_cache:
                feature_cache[feature_key] = simulate_features(latent)
            metric_key = effective_metric_key(
                combined_package,
                testing_gamma,
                severe_gamma,
            )
            if metric_key not in metric_cache:
                metric_cache[metric_key] = evaluate_package(
                    feature_cache[feature_key],
                    package=combined_package,
                    streams=combined_streams,
                    bootstrap_seed=(
                        BASE_SEED
                        + combined_package_number * 10_003
                        + int(metric_key[1] * 1000) * 101
                        + int(metric_key[2] * 1000) * 1009
                    ),
                )
            surface_rows.append(
                {
                    "testing_cross_loading_fraction": testing_gamma,
                    "severe_cross_loading_fraction": severe_gamma,
                    **signal_summary(latent),
                    **metric_cache[metric_key],
                    "analysis_version": SURFACE_ANALYSIS_VERSION,
                    "grid_definition": (
                        "Complete 5 x 5 Cartesian grid of gamma_t and gamma_s "
                        "in {0, 0.25, 0.50, 0.75, 1.0}."
                    ),
                    "contamination_definition": (
                        "For each country-draw trajectory and year, "
                        "T'=T+gamma_t*S and S'=S+gamma_s*T, with both "
                        "right-hand inputs taken from the frozen signals."
                    ),
                    "interpretation_boundary": (
                        "Post-primary measurement-specificity response surface "
                        "on frozen matched trajectories; not a dynamic "
                        "mixed-mechanism generator, estimated historical "
                        "contamination fraction, or external validation."
                    ),
                }
            )
            print(
                "Evaluated combined-package surface, "
                f"gamma_t={testing_gamma:.2f}, gamma_s={severe_gamma:.2f}",
                flush=True,
            )

    surface = pd.DataFrame(surface_rows).sort_values(
        [
            "severe_cross_loading_fraction",
            "testing_cross_loading_fraction",
        ]
    )
    validate_surface(surface)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SURFACE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SURFACE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    frame.to_csv(TABLE_PATH, index=False)
    surface.to_csv(SURFACE_OUTPUT_PATH, index=False)
    surface.to_csv(SURFACE_TABLE_PATH, index=False)
    MARKDOWN_PATH.write_text(markdown_table(frame), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {SURFACE_OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
