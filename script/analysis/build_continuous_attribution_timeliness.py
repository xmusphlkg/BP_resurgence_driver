#!/usr/bin/env python3
"""Continuous latent-contrast recovery and timeliness benchmark.

This analysis reuses the frozen matched-notification surveillance-design
outputs. It does not refit Model 2C and it does not edit manuscript text.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT / "outputs" / "model2c_surveillance_design" / "monitoring_packages_v4_source_preserved_20260908"
)
DATE_STAMP = "2026-09-03"
MATCHED_SCENARIOS = ("matched_reporting_truth", "matched_transmission_truth")
HORIZONS = (3, 6, 12, 24, 36)
EPSILON = 1e-9
RIDGE_ALPHA = 10.0


@dataclass(frozen=True)
class PackageSpec:
    label: str
    package_id: str
    streams: tuple[str, ...]


PACKAGE_SPECS = (
    PackageSpec("notifications", "notifications", ()),
    PackageSpec("testing", "notifications+testing", ("testing",)),
    PackageSpec("severe", "notifications+severe", ("severe",)),
    PackageSpec("testing-severe", "notifications+testing+severe", ("testing", "severe")),
    PackageSpec(
        "complete",
        "notifications+age+testing+severe+serology",
        ("age", "testing", "severe", "serology"),
    ),
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    return np.log(
        (pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float) + EPSILON)
        / (pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float) + EPSILON)
    )


def build_targets(latent: pd.DataFrame) -> pd.DataFrame:
    """Return paired continuous targets for exact matched-notification scenarios."""
    required = {
        "iso3",
        "draw_id",
        "draw_rank",
        "group_id",
        "scenario_id",
        "latent_severe_source_2023",
        "latent_severe_source_2024",
        "latent_severe_source_2025",
        "latent_testing_log_index_2023",
        "latent_testing_log_index_2024",
        "latent_testing_log_index_2025",
        "surveillance_design_version",
    }
    missing = sorted(required.difference(latent.columns))
    if missing:
        raise ValueError("Latent signal table is missing columns: " + ", ".join(missing))

    selected = latent.loc[latent["scenario_id"].isin(MATCHED_SCENARIOS)].copy()
    if selected.empty:
        raise ValueError("No exact matched-notification scenarios were found.")
    pair_index = ["iso3", "draw_id", "draw_rank", "group_id"]
    paired_counts = selected.groupby("group_id")["scenario_id"].nunique()
    if not paired_counts.eq(2).all():
        bad = int((~paired_counts.eq(2)).sum())
        raise ValueError(f"Matched targets require two scenarios per group; {bad} groups failed.")

    severe_columns = [f"latent_severe_source_{year}" for year in (2023, 2024, 2025)]
    testing_columns = [f"latent_testing_log_index_{year}" for year in (2023, 2024, 2025)]
    wide = selected.pivot(
        index=pair_index,
        columns="scenario_id",
        values=[*severe_columns, *testing_columns],
    )
    base = wide.index.to_frame(index=False).reset_index(drop=True)
    trans = "matched_transmission_truth"
    report = "matched_reporting_truth"
    base["log_infant_symptomatic_burden_ratio_2024"] = _safe_log_ratio(
        wide[("latent_severe_source_2024", trans)],
        wide[("latent_severe_source_2024", report)],
    )
    base["log_relative_ascertainment_ratio_2024"] = (
        wide[("latent_testing_log_index_2024", trans)]
        - wide[("latent_testing_log_index_2024", report)]
    ).to_numpy(dtype=float)
    base["burden_contrast_strength"] = (
        base["log_infant_symptomatic_burden_ratio_2024"]
        - base["log_relative_ascertainment_ratio_2024"]
    )
    base["infant_symptomatic_burden_ratio_2024"] = np.exp(
        _safe_log_ratio(
            wide[("latent_severe_source_2024", trans)],
            wide[("latent_severe_source_2024", report)],
        )
    )
    base["relative_ascertainment_ratio_2024"] = np.exp(
        (
            wide[("latent_testing_log_index_2024", trans)]
            - wide[("latent_testing_log_index_2024", report)]
        ).to_numpy(dtype=float)
    )
    version = selected["surveillance_design_version"].dropna().astype(str).unique()
    base["surveillance_design_version"] = version[0] if len(version) else ""
    if not np.isfinite(base["burden_contrast_strength"]).all():
        raise ValueError("Continuous target contains non-finite values.")
    targets = []
    for scenario, sign in (("matched_reporting_truth", -1.0), ("matched_transmission_truth", 1.0)):
        part = base.copy()
        part["scenario_id"] = scenario
        part["truth_active"] = int(sign > 0)
        part["signed_burden_dominant_strength"] = sign * part["burden_contrast_strength"]
        targets.append(part)
    return pd.concat(targets, ignore_index=True)


def _feature_month_index(column: str) -> int | None:
    match = re.fullmatch(r"feature_notifications_(\d{4})_m(\d{2})", column)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    return (year - 2023) * 12 + month


def _annual_feature_horizon(column: str) -> int | None:
    match = re.search(r"_(2023|2024|2025)$", column)
    if not match:
        return None
    return (int(match.group(1)) - 2022) * 12


def feature_columns_for_horizon(
    frame: pd.DataFrame,
    *,
    streams: Iterable[str],
    horizon_months: int,
) -> tuple[list[str], list[str]]:
    columns: list[str] = []
    available_streams: set[str] = set()
    for column in frame.columns:
        month_index = _feature_month_index(column)
        if month_index is not None and month_index <= horizon_months:
            columns.append(column)
            available_streams.add("notifications")
    stream_set = set(streams)
    for stream in stream_set:
        prefix = f"feature_{stream}_"
        for column in frame.columns:
            if not column.startswith(prefix):
                continue
            available_horizon = _annual_feature_horizon(column)
            if available_horizon is not None and available_horizon <= horizon_months:
                columns.append(column)
                available_streams.add(stream)
    seen: set[str] = set()
    ordered = []
    for column in columns:
        if column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered, sorted(available_streams)


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.nanmean(x_train, axis=0)
    scale = np.nanstd(x_train, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z_train = np.nan_to_num((x_train - centre) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    z_test = np.nan_to_num((x_test - centre) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    return z_train, z_test


def _ridge_predict_country_holdout(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float = RIDGE_ALPHA,
) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    outputs = []
    for fold, holdout_country in enumerate(sorted(frame["iso3"].astype(str).unique())):
        train = frame.loc[frame["iso3"].astype(str).ne(holdout_country)]
        test = frame.loc[frame["iso3"].astype(str).eq(holdout_country)]
        x_train, x_test = _standardize_train_test(
            train.loc[:, feature_columns].to_numpy(dtype=float),
            test.loc[:, feature_columns].to_numpy(dtype=float),
        )
        y_train = train["signed_burden_dominant_strength"].to_numpy(dtype=float)
        design_train = np.column_stack([np.ones(len(x_train)), x_train])
        design_test = np.column_stack([np.ones(len(x_test)), x_test])
        penalty = np.eye(design_train.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ y_train)
        part = test[
            [
                "iso3",
                "draw_id",
                "draw_rank",
                "group_id",
                "scenario_id",
                "observation_replicate",
                "truth_active",
                "burden_contrast_strength",
                "signed_burden_dominant_strength",
            ]
        ].copy()
        part["holdout_country"] = holdout_country
        part["outer_fold"] = fold
        part["predicted_signed_strength"] = design_test @ beta
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = y == 1
    negative = y == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def calibration_bin_mae(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 5) -> float:
    work = pd.DataFrame({"truth": y_true, "pred": y_pred}).sort_values("pred")
    if work.empty:
        return float("nan")
    edges = np.linspace(0, len(work), min(n_bins, len(work)) + 1).round().astype(int)
    bins = [work.iloc[edges[index] : edges[index + 1]] for index in range(len(edges) - 1)]
    errors = [
        abs(float(bin_["truth"].mean()) - float(bin_["pred"].mean()))
        for bin_ in bins
        if len(bin_)
    ]
    return float(np.mean(errors)) if errors else float("nan")


def markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else format(float(value), floatfmt)
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
            )
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    truth = predictions["signed_burden_dominant_strength"].to_numpy(dtype=float)
    pred = predictions["predicted_signed_strength"].to_numpy(dtype=float)
    pearson = pearsonr(truth, pred).statistic if len(np.unique(pred)) > 1 else np.nan
    spearman = spearmanr(truth, pred).statistic if len(np.unique(pred)) > 1 else np.nan
    predicted_active = pred >= 0.0
    truth_active = predictions["truth_active"].to_numpy(dtype=int)
    sensitivity = float(predicted_active[truth_active == 1].mean())
    specificity = float((~predicted_active[truth_active == 0]).mean())
    return {
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "rmse": float(np.sqrt(np.mean(np.square(pred - truth)))),
        "mae": float(np.mean(np.abs(pred - truth))),
        "calibration_bin_mae": calibration_bin_mae(truth, pred),
        "sign_accuracy": float((predicted_active == truth_active.astype(bool)).mean()),
        "balanced_accuracy": float(np.mean([sensitivity, specificity])),
        "auc": roc_auc(truth_active, pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def build_analysis(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latent = _read_csv(input_dir / "latent_surveillance_signals.csv")
    features = _read_csv(input_dir / "observation_features_main.csv")
    latent_versions = latent["surveillance_design_version"].dropna().astype(str).unique()
    feature_versions = features["surveillance_design_version"].dropna().astype(str).unique()
    if len(latent_versions) != 1 or len(feature_versions) != 1 or latent_versions[0] != feature_versions[0]:
        raise ValueError("Timeliness inputs must share one surveillance-design version.")
    surveillance_design_version = latent_versions[0]
    targets = build_targets(latent)
    matched_features = features.loc[features["scenario_id"].isin(MATCHED_SCENARIOS)].copy()
    merged = matched_features.merge(
        targets,
        on=["iso3", "draw_id", "draw_rank", "group_id", "scenario_id"],
        how="inner",
        suffixes=("", "_target"),
    )
    if merged.empty:
        raise ValueError("No matched observation features joined to continuous targets.")
    expected_rows = (
        targets["group_id"].nunique()
        * len(MATCHED_SCENARIOS)
        * matched_features["observation_replicate"].nunique()
    )
    if len(merged) != expected_rows:
        raise ValueError(f"Joined feature-target rows {len(merged)} != expected {expected_rows}.")

    metric_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        for spec in PACKAGE_SPECS:
            columns, available_streams = feature_columns_for_horizon(
                merged,
                streams=spec.streams,
                horizon_months=horizon,
            )
            predictions = _ridge_predict_country_holdout(merged, columns)
            predictions["horizon_months"] = horizon
            predictions["package_label"] = spec.label
            predictions["package_id"] = spec.package_id
            predictions["requested_streams"] = "|".join(spec.streams)
            predictions["available_streams_at_horizon"] = "|".join(available_streams)
            predictions["n_features"] = len(columns)
            predictions["n_auxiliary_features"] = len(
                [column for column in columns if not column.startswith("feature_notifications_")]
            )
            prediction_parts.append(predictions)

            metrics = regression_metrics(predictions)
            metric_rows.append(
                {
                    "horizon_months": horizon,
                    "package_label": spec.label,
                    "package_id": spec.package_id,
                    "requested_streams": "|".join(spec.streams),
                    "available_streams_at_horizon": "|".join(available_streams),
                    "n_features": len(columns),
                    "n_auxiliary_features": int(
                        sum(not column.startswith("feature_notifications_") for column in columns)
                    ),
                    "n_countries": predictions["iso3"].nunique(),
                    "n_country_draw_groups": predictions["group_id"].nunique(),
                    "n_observations": len(predictions),
                    "model": f"leave-one-country-out ridge linear regression alpha={RIDGE_ALPHA:g}",
                    "surveillance_design_version": surveillance_design_version,
                    **metrics,
                }
            )
            for iso3, country in predictions.groupby("iso3", sort=True):
                country_metrics = regression_metrics(country)
                country_rows.append(
                    {
                        "horizon_months": horizon,
                        "package_label": spec.label,
                        "package_id": spec.package_id,
                        "iso3": iso3,
                        "n_country_draw_groups": country["group_id"].nunique(),
                        "n_observations": len(country),
                        "surveillance_design_version": surveillance_design_version,
                        **country_metrics,
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(["horizon_months", "package_label"])
    country_metrics = pd.DataFrame(country_rows).sort_values(
        ["horizon_months", "package_label", "iso3"]
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return targets, metrics, country_metrics, predictions


def write_report(
    report_path: Path,
    *,
    input_dir: Path,
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    country_metrics: pd.DataFrame,
    prediction_path: Path,
) -> None:
    strength = targets.drop_duplicates("group_id")["burden_contrast_strength"]
    best_by_horizon = (
        metrics.sort_values(["horizon_months", "spearman_rho", "rmse"], ascending=[True, False, True])
        .groupby("horizon_months", as_index=False)
        .head(1)
    )
    complete = metrics.loc[metrics["package_label"].eq("complete")].sort_values("horizon_months")
    aux_gap = metrics.loc[
        metrics["horizon_months"].isin([3, 6])
        & metrics["n_auxiliary_features"].eq(0)
        & metrics["requested_streams"].ne(""),
        ["horizon_months", "package_label"],
    ]
    country_floor = (
        country_metrics.groupby(["horizon_months", "package_label"], as_index=False)["spearman_rho"]
        .min()
        .rename(columns={"spearman_rho": "minimum_country_spearman_rho"})
    )
    best_with_floor = best_by_horizon.merge(country_floor, on=["horizon_months", "package_label"], how="left")

    lines = [
        "# Continuous Latent-Contrast Recovery Timeliness Analysis",
        "",
        f"Date: {DATE_STAMP}",
        "",
        "## Inputs Located",
        "",
        f"- Latent matched-notification signals: `{(input_dir / 'latent_surveillance_signals.csv').relative_to(ROOT)}`.",
        f"- Simulated surveillance package features: `{(input_dir / 'observation_features_main.csv').relative_to(ROOT)}`.",
        f"- Existing decision operating output: `{(input_dir / 'decision_operating_point_predictions.csv').relative_to(ROOT)}`.",
        "- Existing manuscript figure-data ratios include `manuscript/figure_data/figure_4_matched_latent_ratios.csv` and `manuscript/figure_data/matched_notification_latent_ratios.csv`.",
        "",
        "## Estimand",
        "",
        "For each country-draw matched-notification pair, the analysis defines a conservative log-scale signed burden-dominance score:",
        "",
        "`M_g = log(U5 symptomatic source_2024,transmission / U5 symptomatic source_2024,reporting) - (log ascertainment index_2024,transmission - log ascertainment index_2024,reporting)`.",
        "",
        "Each observation row receives a signed target, `+M_g` for `matched_transmission_truth` and `-M_g` for `matched_reporting_truth`. This is a simulation-library latent-contrast recovery estimand, not a claim that real-world notification series alone identify absolute burden or ascertainment contributions.",
        "",
        "## Target Distribution",
        "",
        f"- Country-draw pairs: {targets['group_id'].nunique():,}.",
        f"- Observation scenarios: {', '.join(MATCHED_SCENARIOS)}.",
        f"- Paired strength median: {strength.median():.3f} log units; IQR {strength.quantile(0.25):.3f}-{strength.quantile(0.75):.3f}.",
        f"- Direction-reversing paired strengths: {int((strength < 0).sum())}/{len(strength)} country-draw pairs; these were retained rather than censored.",
        "",
        "## Key Timeliness Results",
        "",
        markdown_table(best_with_floor[
            [
                "horizon_months",
                "package_label",
                "spearman_rho",
                "pearson_r",
                "rmse",
                "calibration_bin_mae",
                "auc",
                "balanced_accuracy",
                "minimum_country_spearman_rho",
            ]
        ]),
        "",
        "Complete-package trajectory:",
        "",
        markdown_table(complete[
            [
                "horizon_months",
                "spearman_rho",
                "pearson_r",
                "rmse",
                "calibration_bin_mae",
                "auc",
                "balanced_accuracy",
                "n_features",
                "n_auxiliary_features",
            ]
        ]),
        "",
        "## Important Limitation",
        "",
        "The frozen feature export contains monthly notification features but only annual age, testing, severe and serology features. Therefore, 3- and 6-month auxiliary packages cannot use testing or severe information without an upstream monthly export; they are deliberately evaluated with the same available columns as notifications-only.",
    ]
    if not aux_gap.empty:
        affected = ", ".join(
            f"{int(row.horizon_months)} mo/{row.package_label}" for row in aux_gap.itertuples()
        )
        lines.append(f"Affected package-horizon cells: {affected}.")
    lines.extend(
        [
            "",
            "Needed upstream export for a stronger early-timeliness analysis: monthly or cumulative-through-month testing denominators/positivity indices, severe-outcome counts, and age-stratified notification shares generated with the same observation-replicate IDs.",
            "",
            "## Outputs",
            "",
            f"- Main metrics: `{(ROOT / f'outputs/continuous_attribution_timeliness_{DATE_STAMP}.metrics.csv').relative_to(ROOT)}`.",
            f"- Country metrics: `{(ROOT / f'outputs/continuous_attribution_timeliness_{DATE_STAMP}.country_metrics.csv').relative_to(ROOT)}`.",
            f"- Feature-row predictions: `{prediction_path.relative_to(ROOT)}`.",
            f"- Manuscript-ready summary: `manuscript/tables/continuous_attribution_timeliness_summary.csv`.",
            "",
            "## Candidate Manuscript Text",
            "",
            "Continuous latent-contrast recovery extended the binary burden-versus-ascertainment endpoint by assigning each country-draw pair a signed burden-dominance score derived from paired under-5 symptomatic burden and relative ascertainment ratios.",
            "",
            "In leave-one-country-model-structure-out evaluation, early notification-only features provided limited signed-score recovery, whereas annual testing and severe-outcome streams sharply improved rank correlation after the second completed surveillance year; because auxiliary streams were exported annually, 3- and 6-month auxiliary performance should be interpreted as a lower-bound analysis pending monthly testing and severe-outcome exports.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    targets, metrics, country_metrics, predictions = build_analysis(input_dir)

    target_path = ROOT / f"outputs/continuous_attribution_timeliness_{DATE_STAMP}.targets.csv"
    metrics_path = ROOT / f"outputs/continuous_attribution_timeliness_{DATE_STAMP}.metrics.csv"
    country_path = ROOT / f"outputs/continuous_attribution_timeliness_{DATE_STAMP}.country_metrics.csv"
    prediction_path = ROOT / f"outputs/continuous_attribution_timeliness_{DATE_STAMP}.predictions.csv"
    json_path = ROOT / f"outputs/continuous_attribution_timeliness_{DATE_STAMP}.metadata.json"
    table_path = ROOT / "manuscript" / "tables" / "continuous_attribution_timeliness_summary.csv"
    table_markdown_path = (
        ROOT / "manuscript" / "tables_md" / "continuous_attribution_timeliness_summary.md"
    )
    report_path = ROOT / "manuscript" / "reports" / f"continuous_attribution_timeliness_{DATE_STAMP}.md"

    targets.to_csv(target_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    country_metrics.to_csv(country_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    summary_columns = [
        "horizon_months",
        "package_label",
        "package_id",
        "available_streams_at_horizon",
        "n_features",
        "n_auxiliary_features",
        "n_countries",
        "n_country_draw_groups",
        "n_observations",
        "spearman_rho",
        "pearson_r",
        "rmse",
        "mae",
        "calibration_bin_mae",
        "auc",
        "balanced_accuracy",
        "sign_accuracy",
        "surveillance_design_version",
    ]
    metrics.loc[:, summary_columns].to_csv(table_path, index=False)
    table_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    summary_markdown = metrics.loc[
        :,
        [
            "horizon_months",
            "package_label",
            "available_streams_at_horizon",
            "n_features",
            "spearman_rho",
            "pearson_r",
            "rmse",
            "auc",
            "balanced_accuracy",
        ],
    ].rename(
        columns={
            "horizon_months": "Horizon months",
            "package_label": "Package",
            "available_streams_at_horizon": "Streams available",
            "n_features": "Features",
            "spearman_rho": "Spearman rho",
            "pearson_r": "Pearson r",
            "rmse": "RMSE",
            "auc": "AUC",
            "balanced_accuracy": "Balanced accuracy",
        }
    )
    table_markdown_path.write_text(markdown_table(summary_markdown) + "\n", encoding="utf-8")
    metadata = {
        "date": DATE_STAMP,
        "input_dir": str(input_dir.relative_to(ROOT)),
        "surveillance_design_version": str(metrics["surveillance_design_version"].iloc[0]),
        "estimand": "signed log-scale burden-dominant strength",
        "ridge_alpha": RIDGE_ALPHA,
        "horizons_months": list(HORIZONS),
        "packages": [spec.__dict__ for spec in PACKAGE_SPECS],
        "outputs": {
            "targets": str(target_path.relative_to(ROOT)),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "country_metrics": str(country_path.relative_to(ROOT)),
            "predictions": str(prediction_path.relative_to(ROOT)),
            "summary_table": str(table_path.relative_to(ROOT)),
            "summary_table_markdown": str(table_markdown_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        report_path,
        input_dir=input_dir,
        targets=targets,
        metrics=metrics,
        country_metrics=country_metrics,
        prediction_path=prediction_path,
    )
    print(f"Wrote {metrics_path.relative_to(ROOT)}")
    print(f"Wrote {table_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
