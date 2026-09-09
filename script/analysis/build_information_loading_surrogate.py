#!/usr/bin/env python3
"""Illustrate loading geometry in coherent matched effective-process coordinates.

The two coordinates are all-age symptomatic burden and matched effective
ascertainment, defined as matched notifications divided by that burden. They
are not the native testing index and infant severe-event source of the main
benchmark. This abstract illustration uses its contrast scales, but does not
independently validate source-scenario discrimination or its event thresholds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LATENT_PATH = (
    ROOT
    / "outputs"
    / "model2c_surveillance_design"
    / "monitoring_packages_v4_source_preserved_20260908"
    / "latent_surveillance_signals.csv"
)

DETAIL_PATH = ROOT / "outputs" / "information_loading_surrogate_2026-09-08.csv"
JSON_PATH = ROOT / "outputs" / "information_loading_surrogate_2026-09-08.json"
MD_PATH = ROOT / "outputs" / "information_loading_surrogate_2026-09-08.md"
SUMMARY_PATH = ROOT / "manuscript" / "tables" / "information_loading_surrogate_summary.csv"
REPORT_PATH = ROOT / "manuscript" / "reports" / "information_loading_framework_2026-09-08.md"

YEARS = (2023, 2024, 2025)
POSITIVE_SCENARIO = "matched_transmission_truth"
NEGATIVE_SCENARIO = "matched_reporting_truth"
BASE_SEED = 20260908
OBSERVATION_REPLICATES = 3
CV_FOLDS = 5

# Effective log-scale noise for the model-agnostic Gaussian surrogate. This
# includes observation error plus stream-specific heterogeneity not represented
# in a two-latent scaffold.
NOTIFICATION_SD = 0.55
TESTING_SD = 0.60
SEVERE_SD = 0.60


@dataclass(frozen=True)
class LatentContrasts:
    group_id: np.ndarray
    iso3: np.ndarray
    delta_log_i: np.ndarray
    delta_log_rho: np.ndarray


def roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Compute empirical AUC with average ranks for ties."""

    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = score[truth == 1]
    negative = score[truth == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")

    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = (start + end + 2) / 2.0
        start = end + 1

    rank_sum = float(ranks[: len(positive)].sum())
    numerator = rank_sum - len(positive) * (len(positive) + 1) / 2.0
    return numerator / (len(positive) * len(negative))


def grouped_lda_scores(
    features: np.ndarray,
    truth: np.ndarray,
    group_id: np.ndarray,
    *,
    seed: int,
    shrinkage: float = 0.80,
) -> np.ndarray:
    """Return grouped-CV linear-discriminant scores with train-fold scaling."""

    unique_groups = np.array(sorted(set(group_id.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    fold_for_group = {group: i % CV_FOLDS for i, group in enumerate(shuffled)}
    scores = np.zeros(len(truth), dtype=float)

    for fold in range(CV_FOLDS):
        is_test = np.array([fold_for_group[group] == fold for group in group_id])
        is_train = ~is_test
        x_train = features[is_train]
        x_test = features[is_test]
        y_train = truth[is_train]

        center = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (x_train - center) / scale
        x_test = (x_test - center) / scale

        mean_pos = x_train[y_train == 1].mean(axis=0)
        mean_neg = x_train[y_train == 0].mean(axis=0)
        covariance = np.cov(x_train, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]])
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            shrinkage * diagonal
            + (1.0 - shrinkage) * covariance
            + np.eye(covariance.shape[0]) * 1e-6
        )
        weights = np.linalg.solve(covariance, mean_pos - mean_neg)
        scores[is_test] = x_test @ weights

    return scores


def matrix_for_streams(
    streams: tuple[str, ...],
    *,
    testing_cross_loading: float = 0.0,
    severe_cross_loading: float = 0.0,
) -> np.ndarray:
    """Loadings on all-age log burden and matched log effective ascertainment.

    Legacy testing/severe keys mean idealised effective-ascertainment/burden
    readouts here, not the native-index/infant-event features in the main study.
    """

    rows: list[list[float]] = []
    if "notifications" in streams:
        rows.append([1.0, 1.0])
    if "testing" in streams:
        rows.append([testing_cross_loading, 1.0])
    if "severe" in streams:
        rows.append([1.0, severe_cross_loading])
    if not rows:
        raise ValueError("At least one stream is required")
    return np.asarray(rows, dtype=float)


def matrix_stats(matrix: np.ndarray) -> dict[str, float | int | str]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > 1e-10))
    if rank < 2:
        condition = float("inf")
        condition_text = "inf"
    else:
        condition = float(singular[0] / singular[-1])
        condition_text = f"{condition:.6g}"
    determinant = float("nan")
    if matrix.shape == (2, 2):
        determinant = float(np.linalg.det(matrix))
    return {
        "loading_rank": rank,
        "loading_condition_number": condition,
        "loading_condition_number_text": condition_text,
        "loading_determinant_2x2": determinant,
    }


def contrasts_from_frame(latent: pd.DataFrame) -> LatentContrasts:
    """Construct paired log I and log rho_effective with an exact product crosswalk."""

    selected = latent.loc[
        latent["scenario_id"].isin([POSITIVE_SCENARIO, NEGATIVE_SCENARIO])
    ].copy()
    indexed = selected.set_index(["group_id", "scenario_id"]).sort_index()
    if indexed.index.has_duplicates:
        raise ValueError("Duplicate group/scenario rows in matched latent library")
    positive = indexed.xs(POSITIVE_SCENARIO, level="scenario_id")
    negative = indexed.xs(NEGATIVE_SCENARIO, level="scenario_id")
    if not positive.index.equals(negative.index):
        raise ValueError("Every surrogate background must contain both source scenarios")
    if not positive["iso3"].equals(negative["iso3"]):
        raise ValueError("Paired source scenarios have inconsistent country labels")

    delta_log_i = []
    delta_log_rho = []
    for year in YEARS:
        burden_positive = positive[f"latent_all_age_symptomatic_burden_{year}"].to_numpy(dtype=float)
        burden_negative = negative[f"latent_all_age_symptomatic_burden_{year}"].to_numpy(dtype=float)
        columns = [f"latent_notification_mean_{year}_m{month:02d}" for month in range(1, 13)]
        mean_positive = positive[columns].sum(axis=1).to_numpy(dtype=float)
        mean_negative = negative[columns].sum(axis=1).to_numpy(dtype=float)
        for value in (burden_positive, burden_negative, mean_positive, mean_negative):
            if not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError("Effective log coordinates require finite positive burden and notifications")
        if not np.allclose(mean_positive, mean_negative, rtol=1e-12, atol=1e-10):
            raise ValueError("This illustration requires exactly matched notification means")
        delta_i = np.log(burden_positive) - np.log(burden_negative)
        delta_rho = np.log(mean_positive / burden_positive) - np.log(mean_negative / burden_negative)
        notification_contrast = np.log(mean_positive) - np.log(mean_negative)
        if not np.allclose(delta_i + delta_rho, notification_contrast, rtol=0, atol=1e-12):
            raise ValueError("Effective ascertainment and all-age burden must close the notification identity")
        delta_log_i.append(delta_i)
        delta_log_rho.append(delta_rho)

    return LatentContrasts(
        group_id=positive.index.to_numpy(dtype=str),
        iso3=positive["iso3"].to_numpy(dtype=str),
        delta_log_i=np.column_stack(delta_log_i),
        delta_log_rho=np.column_stack(delta_log_rho),
    )


def load_latent_contrasts() -> LatentContrasts:
    contrasts = contrasts_from_frame(pd.read_csv(LATENT_PATH))
    if len(contrasts.group_id) != 720:
        raise RuntimeError("Expected 720 matched source-scenario pairs")
    return contrasts


def simulate_features(
    contrasts: LatentContrasts,
    *,
    streams: tuple[str, ...],
    contrast_fraction: float = 1.0,
    testing_cross_loading: float = 0.0,
    severe_cross_loading: float = 0.0,
    testing_sd: float = TESTING_SD,
    severe_sd: float = SEVERE_SD,
    notification_sd: float = NOTIFICATION_SD,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired Gaussian observation features for grouped discrimination."""

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    truth: list[int] = []
    group_rows: list[str] = []
    country_rows: list[str] = []

    for idx, group in enumerate(contrasts.group_id):
        for class_sign, label in ((1.0, 1), (-1.0, 0)):
            latent_i = class_sign * 0.5 * contrast_fraction * contrasts.delta_log_i[idx]
            latent_rho = (
                class_sign * 0.5 * contrast_fraction * contrasts.delta_log_rho[idx]
            )
            for _ in range(OBSERVATION_REPLICATES):
                feature_parts: list[np.ndarray] = []
                if "notifications" in streams:
                    feature_parts.append(
                        latent_i
                        + latent_rho
                        + rng.normal(0.0, notification_sd, len(YEARS))
                    )
                if "testing" in streams:
                    feature_parts.append(
                        testing_cross_loading * latent_i
                        + latent_rho
                        + rng.normal(0.0, testing_sd, len(YEARS))
                    )
                if "severe" in streams:
                    feature_parts.append(
                        latent_i
                        + severe_cross_loading * latent_rho
                        + rng.normal(0.0, severe_sd, len(YEARS))
                    )
                rows.append(np.concatenate(feature_parts))
                truth.append(label)
                group_rows.append(group)
                country_rows.append(contrasts.iso3[idx])

    return (
        np.vstack(rows),
        np.asarray(truth, dtype=int),
        np.asarray(group_rows, dtype=str),
        np.asarray(country_rows, dtype=str),
    )


def evaluate_surrogate(
    contrasts: LatentContrasts,
    *,
    analysis_family: str,
    setting_name: str,
    streams: tuple[str, ...],
    contrast_fraction: float = 1.0,
    testing_cross_loading: float = 0.0,
    severe_cross_loading: float = 0.0,
    testing_sd: float = TESTING_SD,
    severe_sd: float = SEVERE_SD,
    seed_offset: int = 0,
) -> dict[str, object]:
    seed = BASE_SEED + seed_offset
    features, truth, group_id, iso3 = simulate_features(
        contrasts,
        streams=streams,
        contrast_fraction=contrast_fraction,
        testing_cross_loading=testing_cross_loading,
        severe_cross_loading=severe_cross_loading,
        testing_sd=testing_sd,
        severe_sd=severe_sd,
        seed=seed,
    )
    scores = grouped_lda_scores(features, truth, group_id, seed=seed)
    country_aucs = [
        roc_auc(truth[iso3 == country], scores[iso3 == country])
        for country in sorted(set(iso3.tolist()))
    ]
    matrix = matrix_for_streams(
        streams,
        testing_cross_loading=testing_cross_loading,
        severe_cross_loading=severe_cross_loading,
    )
    stats = matrix_stats(matrix)
    projected = matrix @ np.vstack(
        [
            contrast_fraction * contrasts.delta_log_i[:, 1],
            contrast_fraction * contrasts.delta_log_rho[:, 1],
        ]
    )
    signal_norm = np.sqrt(np.sum(projected * projected, axis=0))
    package_id = "notifications"
    if streams != ("notifications",):
        package_id = "notifications+" + "+".join(
            stream for stream in streams if stream != "notifications"
        )
    return {
        "analysis_family": analysis_family,
        "setting_name": setting_name,
        "package_id": package_id,
        "streams": "+".join(streams),
        "contrast_fraction": contrast_fraction,
        "testing_cross_loading_fraction": testing_cross_loading,
        "severe_cross_loading_fraction": severe_cross_loading,
        "testing_log_sd": testing_sd,
        "severe_log_sd": severe_sd,
        "severe_baseline_events": np.nan,
        "surrogate_auc": roc_auc(truth, scores),
        "surrogate_countries_auc_ge_0_70": int(np.sum(np.asarray(country_aucs) >= 0.70)),
        "surrogate_country_auc_min": float(np.min(country_aucs)),
        "surrogate_country_auc_median": float(np.median(country_aucs)),
        "median_2024_signal_norm": float(np.median(signal_norm)),
        "median_2024_delta_log_i": float(
            np.median(contrast_fraction * contrasts.delta_log_i[:, 1])
        ),
        "median_2024_delta_log_rho": float(
            np.median(contrast_fraction * contrasts.delta_log_rho[:, 1])
        ),
        **stats,
        "n_country_draw_groups": len(contrasts.group_id),
        "n_observations": int(len(truth)),
        "random_seed": seed,
        "interpretation_boundary": (
            "Abstract Gaussian illustration of all-age burden and matched effective "
            "ascertainment. Optional readouts differ from the main native-testing "
            "and infant-event estimands; no independent validation is claimed."
        ),
    }




def build_detail_table(contrasts: LatentContrasts) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    packages = {
        "notifications": ("notifications",),
        "notifications+testing": ("notifications", "testing"),
        "notifications+severe": ("notifications", "severe"),
        "notifications+testing+severe": ("notifications", "testing", "severe"),
    }

    for level in (0.0, 0.125, 0.25, 0.50, 0.75, 1.0):
        for package_number, streams in enumerate(packages.values()):
            rows.append(
                evaluate_surrogate(
                    contrasts,
                    analysis_family="attenuation",
                    setting_name=f"contrast_fraction={level:g}",
                    streams=streams,
                    contrast_fraction=level,
                    seed_offset=1000 + int(level * 1000) + package_number * 37,
                )
            )

    for gamma_t in (0.0, 0.25, 0.50, 0.75, 1.0):
        for gamma_s in (0.0, 0.25, 0.50, 0.75, 1.0):
            rows.append(
                evaluate_surrogate(
                    contrasts,
                    analysis_family="cross_loading_surface",
                    setting_name=f"gamma_t={gamma_t:g};gamma_s={gamma_s:g}",
                    streams=("notifications", "testing", "severe"),
                    testing_cross_loading=gamma_t,
                    severe_cross_loading=gamma_s,
                    seed_offset=3000 + int(gamma_t * 100) * 11 + int(gamma_s * 100),
                )
            )

    for gamma in (0.0, 0.25, 0.50, 0.75, 1.0):
        for package_number, streams in enumerate(
            [
                ("notifications", "testing"),
                ("notifications", "severe"),
                ("notifications", "testing", "severe"),
            ]
        ):
            rows.append(
                evaluate_surrogate(
                    contrasts,
                    analysis_family="cross_loading_both",
                    setting_name=f"gamma={gamma:g}",
                    streams=streams,
                    testing_cross_loading=gamma,
                    severe_cross_loading=gamma,
                    seed_offset=4000 + int(gamma * 1000) + package_number * 53,
                )
            )

    for sigma in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        rows.append(
            evaluate_surrogate(
                contrasts,
                analysis_family="testing_log_sd",
                setting_name=f"testing_log_sd={sigma:g}",
                streams=("notifications", "testing"),
                testing_sd=sigma,
                seed_offset=5000 + int(sigma * 1000),
            )
        )

    for event_count in (0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        severe_sd = math.sqrt(0.20**2 + 1.0 / event_count)
        row = evaluate_surrogate(
            contrasts,
            analysis_family="severe_baseline_events",
            setting_name=f"severe_baseline_events={event_count:g}",
            streams=("notifications", "severe"),
            severe_sd=severe_sd,
            seed_offset=6000 + int(event_count * 100),
        )
        row["severe_baseline_events"] = event_count
        rows.append(row)

    for drift_sd in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0):
        severe_sd = math.sqrt(drift_sd**2 + 1.0 / 50.0)
        rows.append(
            evaluate_surrogate(
                contrasts,
                analysis_family="severe_annual_log_sd",
                setting_name=f"severe_annual_log_sd={drift_sd:g}",
                streams=("notifications", "severe"),
                severe_sd=severe_sd,
                seed_offset=7000 + int(drift_sd * 1000),
            )
        )
        rows[-1]["parameter_value"] = drift_sd

    detail = pd.DataFrame(rows)
    detail["parameter_value"] = np.nan
    detail.loc[detail["analysis_family"].eq("attenuation"), "parameter_value"] = detail[
        "contrast_fraction"
    ]
    detail.loc[detail["analysis_family"].eq("cross_loading_both"), "parameter_value"] = detail[
        "testing_cross_loading_fraction"
    ]
    detail.loc[
        detail["analysis_family"].eq("cross_loading_surface"),
        "parameter_value",
    ] = detail["testing_cross_loading_fraction"]
    detail.loc[detail["analysis_family"].eq("testing_log_sd"), "parameter_value"] = detail[
        "testing_log_sd"
    ]
    detail.loc[
        detail["analysis_family"].eq("severe_baseline_events"),
        "parameter_value",
    ] = detail["severe_baseline_events"]
    detail.loc[
        detail["analysis_family"].eq("severe_annual_log_sd"),
        "parameter_value",
    ] = detail.loc[
        detail["analysis_family"].eq("severe_annual_log_sd"),
        "setting_name",
    ].str.extract(r"=([0-9.]+)")[0].astype(float)

    detail["coordinate_system"] = "all_age_burden_and_matched_effective_ascertainment"
    detail["main_benchmark_comparison"] = "different_auxiliary_estimands_no_numerical_validation"
    return detail


def nearest_threshold(frame: pd.DataFrame, family: str, package_id: str) -> str:
    subset = frame.loc[
        frame["analysis_family"].eq(family) & frame["package_id"].eq(package_id)
    ].copy()
    subset = subset.sort_values("parameter_value")
    passing = subset.loc[subset["surrogate_auc"].ge(0.80)]
    if passing.empty:
        return "no surrogate point reached AUC >= 0.80"
    if family == "severe_baseline_events":
        row = passing.iloc[0]
        return f"{row['parameter_value']:.3g} baseline events (AUC {row['surrogate_auc']:.3f})"
    if family in {"testing_log_sd", "severe_annual_log_sd"}:
        row = passing.iloc[-1]
        return f"{row['parameter_value']:.3g} log-scale SD (AUC {row['surrogate_auc']:.3f})"
    row = passing.iloc[0]
    return f"{row['parameter_value']:.3g} (AUC {row['surrogate_auc']:.3f})"


def build_summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    def auc_at(family: str, package_id: str, value: float) -> float:
        subset = detail.loc[
            detail["analysis_family"].eq(family)
            & detail["package_id"].eq(package_id)
            & np.isclose(detail["parameter_value"].astype(float), value)
        ]
        return float(subset.iloc[0]["surrogate_auc"])

    scope = "Effective-process illustration; auxiliary estimands differ from the primary source-scenario benchmark."
    rows = [
        {
            "evidence_component": "Matched-product geometry",
            "surrogate_result": f"Notification AUC {auc_at('attenuation', 'notifications', 1.0):.3f}",
            "current_model_result": scope,
            "model_agnostic_interpretation": "rho_effective equals matched notifications divided by all-age symptomatic burden; paired log contrasts sum to zero by construction.",
            "remaining_model_dependence": "All-age burden contrasts are taken from the selected pertussis model library; effective ascertainment is constructed, not observed.",
        },
        {
            "evidence_component": "Signal-contrast attenuation",
            "surrogate_result": f"Idealised effective-ascertainment plus burden readouts: AUC {auc_at('attenuation', 'notifications+testing+severe', 0.25):.3f} at 25% retained contrast",
            "current_model_result": scope,
            "model_agnostic_interpretation": "Projected contrast and noise determine discrimination; no equivalence with the primary attenuation threshold is asserted.",
            "remaining_model_dependence": "Contrast magnitudes, Gaussian noise and grouped classifier define this conditional result.",
        },
        {
            "evidence_component": "Cross-loading and rank loss",
            "surrogate_result": f"AUC {auc_at('cross_loading_both', 'notifications+testing+severe', 1.0):.3f} when both idealised optional streams load [1,1]",
            "current_model_result": scope,
            "model_agnostic_interpretation": "Full shared loading projects this matched effective-process contrast to zero. Rank loss alone does not imply chance discrimination for arbitrary source-scenario contrasts.",
            "remaining_model_dependence": "Primary native ascertainment and infant burden need not sum to the matched notification signal; their cross-loading result can differ.",
        },
        {
            "evidence_component": "Effective-ascertainment readout noise",
            "surrogate_result": nearest_threshold(detail, "testing_log_sd", "notifications+testing"),
            "current_model_result": scope,
            "model_agnostic_interpretation": "The information scale depends on noise relative to the projected effective-process contrast.",
            "remaining_model_dependence": "This is not a threshold for observed test volume or the primary native ascertainment index.",
        },
        {
            "evidence_component": "Stylised count-noise scale",
            "surrogate_result": nearest_threshold(detail, "severe_baseline_events", "notifications+severe"),
            "current_model_result": scope,
            "model_agnostic_interpretation": "The 1/event-count term is a Gaussian noise-scale illustration, not a discrete sparse-event model.",
            "remaining_model_dependence": "The approximation is unreliable at very low counts and cannot be translated to infant sentinel size or operational thresholds.",
        },
    ]
    return pd.DataFrame(rows)


def format_auc(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.3f}"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a Markdown table without optional dependencies."""

    display = frame.copy()
    display = display.reset_index()
    headers = [str(column) for column in display.columns]

    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def markdown_records(frame: pd.DataFrame) -> str:
    """Render a record table without optional dependencies."""

    headers = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: LatentContrasts,
) -> None:
    contrast_corr = float(
        np.corrcoef(contrasts.delta_log_i[:, 1], contrasts.delta_log_rho[:, 1])[0, 1]
    )
    median_delta_i = float(np.median(contrasts.delta_log_i[:, 1]))
    median_delta_rho = float(np.median(contrasts.delta_log_rho[:, 1]))
    attenuation_display = detail.loc[
        detail["analysis_family"].eq("attenuation")
        & detail["package_id"].isin(
            ["notifications", "notifications+testing", "notifications+severe", "notifications+testing+severe"]
        )
    ].pivot(index="parameter_value", columns="package_id", values="surrogate_auc")
    cross_display = detail.loc[
        detail["analysis_family"].eq("cross_loading_both")
        & detail["package_id"].isin(
            ["notifications+testing", "notifications+severe", "notifications+testing+severe"]
        )
    ].pivot(index="parameter_value", columns="package_id", values="surrogate_auc")
    precision_display = detail.loc[
        detail["analysis_family"].isin(
            ["testing_log_sd", "severe_baseline_events", "severe_annual_log_sd"]
        )
    ][["analysis_family", "parameter_value", "package_id", "surrogate_auc"]].copy()
    attenuation_md = markdown_table(attenuation_display.map(format_auc))
    cross_md = markdown_table(cross_display.map(format_auc))
    precision_md = markdown_records(
        precision_display.assign(
            surrogate_auc=precision_display["surrogate_auc"].map(format_auc)
        )
    )
    summary_md = markdown_records(summary)

    maximum_product_residual = float(np.max(np.abs(contrasts.delta_log_i + contrasts.delta_log_rho)))
    report = rf"""# Loading geometry in matched effective-process coordinates

## Scope

This is an abstract two-component observation illustration. Its coordinates are all-age symptomatic burden and matched effective ascertainment. It does not use the primary experiment's native source-testing index or infant severe-event source. Its results cannot independently validate the main source-scenario AUCs or information thresholds.

## Consistent product identity

Let $I_y$ be all-age symptomatic burden and $M_y$ matched annual expected notifications. Define $\rho_y^{{eff}}=M_y/I_y$ and $x_y=(\log I_y,\log\rho_y^{{eff}})^T$. The observation illustration is $z_y=A x_y+\epsilon_y$. Notifications have loading $[1,1]$; idealised effective-ascertainment and burden readouts have loadings $[0,1]$ and $[1,0]$. The computational package keys `testing` and `severe` are retained for compatibility but refer only to those idealised readouts here.

Across {len(contrasts.group_id)} paired backgrounds, the maximum absolute paired log-product residual was {maximum_product_residual:.3g}. In 2024 the median log burden contrast was {median_delta_i:.3f}, the median log effective-ascertainment contrast was {median_delta_rho:.3f}, and their correlation was {contrast_corr:.3f}. Their inverse relation is an algebraic consequence of exact matching, not recovered biological evidence.

The optional loading block is `[[gamma_T, 1], [1, gamma_S]]`, with determinant `gamma_T*gamma_S - 1`. When both coefficients are one, it measures only the matched product; this particular contrast then has zero projected signal. Generally, rank deficiency prevents identification of two arbitrary coordinates but need not prevent classification of particular scenarios. Full rank does not guarantee adequate precision, and unequal or correlated noise requires $A^T\Sigma^{{-1}}A$.

## Generated results

### Contrast attenuation

{attenuation_md}

### Shared effective-process loading

{cross_md}

### Gaussian noise-scale sensitivity

{precision_md}

The event-count rows use the stylised Gaussian variance term `drift_SD^2 + 1/event_count`. This approximation does not represent discrete sparse infant events and is unreliable at very small counts; it supplies no minimum sentinel size. Noise assumptions were not calibrated to laboratory or hospital data. The grouped classifier and paired contrasts are fixed conditional design choices.

## Supplementary summary

{summary_md}

These results describe this effective-process illustration. They neither force the main native-source cross-loading result to decline nor establish numerical agreement with Figure 3. No difference between their AUCs is treated as validation error.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    MD_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    for path in [DETAIL_PATH, JSON_PATH, MD_PATH, SUMMARY_PATH, REPORT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    contrasts = load_latent_contrasts()
    detail = build_detail_table(contrasts)
    summary = build_summary_table(detail)

    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    metadata = {
        "analysis_version": "information_loading_surrogate_v2_effective_coordinates_20260908",
        "input_latent_path": str(LATENT_PATH.relative_to(ROOT)),
        "n_country_draw_groups": int(len(contrasts.group_id)),
        "years": list(YEARS),
        "positive_scenario": POSITIVE_SCENARIO,
        "negative_scenario": NEGATIVE_SCENARIO,
        "base_seed": BASE_SEED,
        "observation_replicates": OBSERVATION_REPLICATES,
        "cv_folds": CV_FOLDS,
        "notification_sd": NOTIFICATION_SD,
        "testing_sd": TESTING_SD,
        "severe_sd": SEVERE_SD,
        "coordinate_system": "all_age_symptomatic_burden_and_matched_effective_ascertainment",
        "maximum_log_product_residual": float(np.max(np.abs(contrasts.delta_log_i + contrasts.delta_log_rho))),
        "boundary": "Abstract effective-process illustration with different auxiliary estimands; not independent validation of the primary native-source benchmark.",
    }
    JSON_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown_outputs(detail, summary, contrasts)

    print(f"Wrote {DETAIL_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
