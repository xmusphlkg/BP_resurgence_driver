#!/usr/bin/env python3
"""Evaluate surveillance performance as the constructed contrast is attenuated.

The primary matched-notification experiment compares two deliberately separated
trajectory classes. This post-primary frontier shrinks the between-class
difference in the testing and severe-outcome signals on their natural link
scales while leaving expected notifications exactly matched. It answers a
narrow robustness question: how much conditional discrimination remains when
the auxiliary signals are closer to the non-identifiable boundary?

The analysis is not a mixture-model fit to observed outbreaks and does not
estimate the historical contribution of transmission or ascertainment.
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
OUTPUT_PATH = ROOT / "outputs" / "nc_contrast_attenuation_frontier.csv"
TABLE_PATH = ROOT / "manuscript" / "tables" / "nc_contrast_attenuation_frontier.csv"
MARKDOWN_PATH = (
    ROOT / "manuscript" / "tables_md" / "nc_contrast_attenuation_frontier.md"
)

ANALYSIS_VERSION = "nc_contrast_attenuation_v2_source_preserved_infant_20260908"
POSITIVE_SCENARIO = "matched_transmission_truth"
NEGATIVE_SCENARIO = "matched_reporting_truth"
CONTRAST_LEVELS = (0.0, 0.125, 0.25, 0.50, 0.75, 1.0)
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
    differences = (
        selected.set_index(["group_id", "scenario_id"])[notification_columns]
        .groupby(level="group_id")
        .max()
        - selected.set_index(["group_id", "scenario_id"])[notification_columns]
        .groupby(level="group_id")
        .min()
    )
    if not np.allclose(differences.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-9):
        raise RuntimeError("Matched latent notifications are no longer exactly equal")
    return selected.sort_values(["iso3", "draw_rank", "scenario_id"]).reset_index(drop=True)


def symmetric_linear_pair(
    positive: pd.Series,
    negative: pd.Series,
    columns: list[str],
    level: float,
) -> tuple[pd.Series, pd.Series]:
    pos = positive.copy()
    neg = negative.copy()
    for column in columns:
        pos_value = float(positive[column])
        neg_value = float(negative[column])
        midpoint = 0.5 * (pos_value + neg_value)
        difference = pos_value - neg_value
        pos[column] = midpoint + 0.5 * level * difference
        neg[column] = midpoint - 0.5 * level * difference
    return pos, neg


def symmetric_log_pair(
    positive: pd.Series,
    negative: pd.Series,
    columns: list[str],
    level: float,
) -> tuple[pd.Series, pd.Series]:
    pos = positive.copy()
    neg = negative.copy()
    for column in columns:
        pos_value = max(float(positive[column]), 1e-12)
        neg_value = max(float(negative[column]), 1e-12)
        pos_log = np.log(pos_value)
        neg_log = np.log(neg_value)
        midpoint = 0.5 * (pos_log + neg_log)
        difference = pos_log - neg_log
        pos[column] = float(np.exp(midpoint + 0.5 * level * difference))
        neg[column] = float(np.exp(midpoint - 0.5 * level * difference))
    return pos, neg


def attenuate_latent(latent: pd.DataFrame, level: float) -> pd.DataFrame:
    if not 0.0 <= level <= 1.0:
        raise ValueError("Contrast level must lie in [0, 1]")
    testing_columns = [
        column
        for column in latent.columns
        if column.startswith("latent_testing_log_index_")
    ]
    severe_columns = [
        column
        for column in latent.columns
        if column.startswith("latent_severe_source_")
        and column != "latent_severe_source_baseline"
    ]
    rows: list[pd.Series] = []
    for _, pair in latent.groupby("group_id", sort=True):
        positive = pair.loc[pair["scenario_id"].eq(POSITIVE_SCENARIO)].iloc[0].copy()
        negative = pair.loc[pair["scenario_id"].eq(NEGATIVE_SCENARIO)].iloc[0].copy()
        positive, negative = symmetric_linear_pair(
            positive,
            negative,
            testing_columns,
            level,
        )
        positive, negative = symmetric_log_pair(
            positive,
            negative,
            severe_columns,
            level,
        )
        baseline = float(
            np.sqrt(
                max(float(positive["latent_severe_source_baseline"]), 1e-12)
                * max(float(negative["latent_severe_source_baseline"]), 1e-12)
            )
        )
        positive["latent_severe_source_baseline"] = baseline
        negative["latent_severe_source_baseline"] = baseline
        rows.extend([positive, negative])
    return pd.DataFrame(rows).sort_values(
        ["iso3", "draw_rank", "scenario_id"]
    ).reset_index(drop=True)


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
    severe_positive = (
        positive["latent_severe_source_2024"]
        / positive["latent_severe_source_baseline"]
    )
    severe_negative = (
        negative["latent_severe_source_2024"]
        / negative["latent_severe_source_baseline"]
    )
    ascertainment_ratio = np.exp(
        positive["latent_testing_log_index_2024"]
        - negative["latent_testing_log_index_2024"]
    )
    return {
        "median_2024_infant_severe_signal_ratio": float(
            np.median(severe_positive / severe_negative)
        ),
        "median_2024_ascertainment_ratio": float(np.median(ascertainment_ratio)),
    }


def evaluate_level(
    features: pd.DataFrame,
    *,
    contrast_level: float,
    signal: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for package_number, (package, streams) in enumerate(PACKAGES.items()):
        columns = feature_columns_for_package(features, streams)
        grouped = cross_validated_predictions(features, columns, n_folds=5, shrinkage=0.80)
        grouped_metrics = discrimination_metrics(
            grouped["truth_active"],
            grouped["probability_active"],
        )
        lower, upper = nested_country_draw_bootstrap_auc(
            grouped,
            n_bootstrap=SURVEILLANCE_BOOTSTRAP_REPLICATES,
            seed=BASE_SEED + package_number * 10_003 + int(contrast_level * 1000),
        )

        country_aucs = []
        for _, country in grouped.groupby("iso3", sort=True):
            country_aucs.append(
                discrimination_metrics(
                    country["truth_active"],
                    country["probability_active"],
                )["auc"]
            )

        country_holdout = leave_one_country_out_predictions(
            features,
            columns,
            shrinkage=0.80,
        )
        country_holdout_metrics = discrimination_metrics(
            country_holdout["truth_active"],
            country_holdout["probability_active"],
        )
        holdout_country_aucs = []
        for _, country in country_holdout.groupby("iso3", sort=True):
            holdout_country_aucs.append(
                discrimination_metrics(
                    country["truth_active"],
                    country["probability_active"],
                )["auc"]
            )

        rows.append(
            {
                "contrast_level": contrast_level,
                **signal,
                "package_id": package,
                "n_features": len(columns),
                "grouped_cv_auc": grouped_metrics["auc"],
                "grouped_cv_auc_lower_95": lower,
                "grouped_cv_auc_upper_95": upper,
                "grouped_cv_balanced_accuracy": grouped_metrics["balanced_accuracy"],
                "grouped_cv_countries_auc_ge_0_70": int(
                    np.sum(np.asarray(country_aucs) >= 0.70)
                ),
                "whole_country_holdout_auc": country_holdout_metrics["auc"],
                "whole_country_holdout_countries_auc_ge_0_70": int(
                    np.sum(np.asarray(holdout_country_aucs) >= 0.70)
                ),
                "analysis_version": ANALYSIS_VERSION,
                "interpretation_boundary": (
                    "Post-primary log-scale attenuation of the two constructed "
                    "matched-notification classes; not an estimated real-world "
                    "mixture or external validation."
                ),
            }
        )
    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    selected = frame.loc[
        frame["package_id"].isin(["notifications+testing", "notifications+severe"])
    ].copy()
    display = selected[
        [
            "contrast_level",
            "median_2024_infant_severe_signal_ratio",
            "median_2024_ascertainment_ratio",
            "package_id",
            "grouped_cv_auc",
            "grouped_cv_auc_lower_95",
            "grouped_cv_auc_upper_95",
            "whole_country_holdout_auc",
            "whole_country_holdout_countries_auc_ge_0_70",
        ]
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
            "contrast_level",
            "median_2024_infant_severe_signal_ratio",
            "median_2024_ascertainment_ratio",
            "package_id",
            "Grouped CV AUC (95% interval)",
            "whole_country_holdout_auc",
            "whole_country_holdout_countries_auc_ge_0_70",
        ]
    ]
    display.columns = [
        "Contrast fraction",
        "Median severe-signal ratio",
        "Median ascertainment ratio",
        "Package",
        "Grouped CV AUC (95% interval)",
        "Whole-country-holdout AUC",
        "Countries with holdout AUC ≥0.70",
    ]
    for column in [
        "Contrast fraction",
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


def validate(frame: pd.DataFrame) -> None:
    zero = frame.loc[frame["contrast_level"].eq(0.0)]
    if not np.allclose(zero["median_2024_infant_severe_signal_ratio"], 1.0):
        raise RuntimeError("Zero-contrast severe signals are not identical")
    if not np.allclose(zero["median_2024_ascertainment_ratio"], 1.0):
        raise RuntimeError("Zero-contrast testing signals are not identical")
    for column in ("grouped_cv_auc", "whole_country_holdout_auc"):
        if not frame[column].between(0, 1).all():
            raise RuntimeError(f"Invalid discrimination metric: {column}")
    # A source contrast can fail the scientific performance criterion. Do not
    # turn an observed failure into a software error or force a high AUC.


def main() -> None:
    original = matched_latent()
    rows: list[dict[str, object]] = []
    for level in CONTRAST_LEVELS:
        attenuated = attenuate_latent(original, level)
        signal = signal_summary(attenuated)
        features = simulate_features(attenuated)
        rows.extend(evaluate_level(features, contrast_level=level, signal=signal))
        print(
            f"Evaluated contrast={level:.3f}: "
            f"severe ratio={signal['median_2024_infant_severe_signal_ratio']:.3f}, "
            f"ascertainment ratio={signal['median_2024_ascertainment_ratio']:.3f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values(["contrast_level", "package_id"])
    validate(frame)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    frame.to_csv(TABLE_PATH, index=False)
    MARKDOWN_PATH.write_text(markdown_table(frame), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
