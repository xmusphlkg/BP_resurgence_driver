"""Simulation-based surveillance design for Model 2C.

The module compares the complete Model 2C trajectory with paired trajectories
in which one candidate process is neutralised.  It converts latent trajectories
to noisy surveillance features and evaluates held-out structural
discrimination without refitting the 27-parameter Stan model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

from pertussis_global_dynamics.bayes.model2c_recovery import negative_binomial_2_rng


SURVEILLANCE_DESIGN_VERSION = "model2c_surveillance_design_v8_source_preserved_infant_20260908"
DEFAULT_SURVEILLANCE_TAG = "monitoring_packages_v4_source_preserved_20260908"
SURVEILLANCE_BOOTSTRAP_REPLICATES = 2000
SURVEILLANCE_BOOTSTRAP_SEED = 20260730

MECHANISM_CONTRASTS: dict[str, str] = {
    "missed_exposure_boosting": "no_missed_exposure_boosting",
    "routine_immunization_disruption": "no_routine_immunization_disruption",
    "vaccine_waning": "low_vaccine_waning",
    "post_2023_transmission_context": "no_post_2023_transmission_context",
    "reporting_testing_shift": "no_reporting_testing_shift",
}

DECISION_CONTRASTS: dict[str, tuple[str, str]] = {
    "transmission_vs_reporting": (
        "matched_transmission_truth",
        "matched_reporting_truth",
    ),
}


def structural_contrasts() -> dict[str, tuple[str, str]]:
    contrasts = {
        mechanism: ("full_model", counterfactual)
        for mechanism, counterfactual in MECHANISM_CONTRASTS.items()
    }
    return {**contrasts, **DECISION_CONTRASTS}

OPTIONAL_STREAMS = ("age", "testing", "severe", "serology")
DEFAULT_CLASSIFIER = "shrinkage_linear_discriminant"
CLASSIFIER_IDS = (
    DEFAULT_CLASSIFIER,
    "ridge_logistic",
    "random_forest_stumps",
    "gradient_boosted_stumps",
    "random_features_neural",
)
AGE_GROUPS = ("under5", "school_age", "adult")
POST_YEARS = (2023, 2024, 2025)
SEROLOGY_YEARS = (2023, 2025)
BASELINE_YEARS = tuple(range(2015, 2020))

PASSING_PRIMARY_COUNTRIES = ("AUS", "CHN", "NZL", "SWE", "THA", "USA")


@dataclass(frozen=True)
class MeasurementDesign:
    design_id: str
    age_concentration: float
    testing_log_sd: float
    severe_baseline_events: float
    severe_log_sd: float
    serology_n_per_stratum: int
    serology_sensitivity: float = 0.80
    serology_specificity: float = 0.98
    severe_dispersion: float = 20.0
    severe_baseline_observation_years: float | None = None


MEASUREMENT_DESIGNS: dict[str, MeasurementDesign] = {
    "low_precision": MeasurementDesign(
        "low_precision",
        age_concentration=30.0,
        testing_log_sd=0.35,
        severe_baseline_events=10.0,
        severe_log_sd=0.35,
        serology_n_per_stratum=100,
    ),
    "main": MeasurementDesign(
        "main",
        age_concentration=100.0,
        testing_log_sd=0.20,
        severe_baseline_events=50.0,
        severe_log_sd=0.20,
        serology_n_per_stratum=300,
    ),
    "high_precision": MeasurementDesign(
        "high_precision",
        age_concentration=300.0,
        testing_log_sd=0.10,
        severe_baseline_events=200.0,
        severe_log_sd=0.10,
        serology_n_per_stratum=1000,
    ),
}


def surveillance_packages() -> tuple[tuple[str, ...], ...]:
    """Return the prespecified 16 optional-stream combinations."""
    packages: list[tuple[str, ...]] = []
    for size in range(len(OPTIONAL_STREAMS) + 1):
        packages.extend(combinations(OPTIONAL_STREAMS, size))
    return tuple(packages)


def package_id(streams: Sequence[str]) -> str:
    ordered = tuple(stream for stream in OPTIONAL_STREAMS if stream in set(streams))
    return "notifications" if not ordered else "notifications+" + "+".join(ordered)


def feature_columns_for_package(frame: pd.DataFrame, streams: Sequence[str]) -> list[str]:
    prefixes = ["feature_notifications_"]
    prefixes.extend(f"feature_{stream}_" for stream in streams)
    return [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(max(numerator, 1e-12) / max(denominator, 1e-12))


def _annual_sum(frame: pd.DataFrame, column: str, year: int) -> float:
    return float(frame.loc[frame["year"].eq(year), column].sum())


def _baseline_calendar_month_mean(frame: pd.DataFrame, column: str, month: int) -> float:
    mask = frame["year"].isin(BASELINE_YEARS) & frame["month"].eq(month)
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _population_by_group(data: dict[str, Any], year_value: int) -> dict[str, float]:
    year = np.asarray(data["year"], dtype=int)
    population = np.asarray(data["age_population"], dtype=float)
    mask = year == int(year_value)
    if not mask.any():
        raise ValueError(f"Prepared data do not contain year {year_value}.")
    annual = population[mask].mean(axis=0)
    return {
        "under5": float(annual[:3].sum()),
        "school_age": float(annual[3:5].sum()),
        "adult": float(annual[5:].sum()),
    }


def trajectory_to_latent_signals(
    prediction: pd.DataFrame,
    data: dict[str, Any],
    draw: pd.Series | dict[str, Any],
    *,
    iso3: str,
    draw_id: str,
    draw_rank: int,
    scenario_id: str,
) -> dict[str, Any]:
    """Convert one deterministic trajectory to surveillance signal parameters."""
    required = {
        "year",
        "month",
        "reported_cases_predicted",
        "symptomatic_infections_predicted",
        "symptomatic_infections_infant_predicted",
        "symptomatic_infections_under5_predicted",
        "true_infections_under5_predicted",
        "true_infections_school_age_predicted",
        "true_infections_adult_predicted",
        "reported_cases_under5_predicted",
        "reported_cases_school_age_predicted",
        "reported_cases_adult_predicted",
        "natural_immunity_under5_predicted",
        "natural_immunity_school_age_predicted",
        "natural_immunity_adult_predicted",
    }
    missing = sorted(required.difference(prediction.columns))
    if missing:
        raise ValueError(f"Prediction is missing surveillance-design columns: {', '.join(missing)}")

    row: dict[str, Any] = {
        "iso3": str(iso3),
        "draw_id": str(draw_id),
        "draw_rank": int(draw_rank),
        "scenario_id": str(scenario_id),
        "group_id": f"{iso3}:{draw_id}",
        "nb_phi_month": float(draw.get("nb_phi_month", draw.get("nb_phi", 20.0))),
        "nb_phi_age": float(draw.get("nb_phi_age", 20.0)),
        "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
    }

    for month_value in range(1, 13):
        row[f"latent_notification_baseline_m{month_value:02d}"] = _baseline_calendar_month_mean(
            prediction,
            "reported_cases_predicted",
            month_value,
        )
    for year_value in POST_YEARS:
        for month_value in range(1, 13):
            selected = prediction.loc[
                prediction["year"].eq(year_value) & prediction["month"].eq(month_value),
                "reported_cases_predicted",
            ]
            row[f"latent_notification_mean_{year_value}_m{month_value:02d}"] = (
                float(selected.iloc[0]) if not selected.empty else 0.0
            )

    source_reported_column = (
        "source_reported_cases_predicted"
        if "source_reported_cases_predicted" in prediction
        else "reported_cases_predicted"
    )
    baseline_reported = np.mean(
        [_annual_sum(prediction, source_reported_column, year_value) for year_value in BASELINE_YEARS]
    )
    effective_baseline_reported = np.mean(
        [_annual_sum(prediction, "reported_cases_predicted", year_value) for year_value in BASELINE_YEARS]
    )
    baseline_symptomatic = np.mean(
        [_annual_sum(prediction, "symptomatic_infections_predicted", year_value) for year_value in BASELINE_YEARS]
    )
    baseline_ascertainment = _safe_ratio(baseline_reported, baseline_symptomatic)
    baseline_severe_source = np.mean(
        [
            _annual_sum(prediction, "symptomatic_infections_infant_predicted", year_value)
            for year_value in BASELINE_YEARS
        ]
    )
    row["latent_severe_source_baseline"] = float(max(baseline_severe_source, 1e-9))
    row["severe_source_age_scope"] = "under_1_year"
    row["testing_signal_scope"] = "native_source_ascertainment_before_notification_matching"

    reported_columns = {
        "under5": "reported_cases_under5_predicted",
        "school_age": "reported_cases_school_age_predicted",
        "adult": "reported_cases_adult_predicted",
    }
    infection_columns = {
        "under5": "true_infections_under5_predicted",
        "school_age": "true_infections_school_age_predicted",
        "adult": "true_infections_adult_predicted",
    }
    natural_immunity_columns = {
        "under5": "natural_immunity_under5_predicted",
        "school_age": "natural_immunity_school_age_predicted",
        "adult": "natural_immunity_adult_predicted",
    }
    sensitivity = float(draw.get("serology_sensitivity", 0.80))
    specificity = float(draw.get("serology_specificity", 0.98))

    for year_value in POST_YEARS:
        annual_reported = {
            group: _annual_sum(prediction, column, year_value)
            for group, column in reported_columns.items()
        }
        reported_total = max(sum(annual_reported.values()), 1e-12)
        for group, value in annual_reported.items():
            row[f"latent_age_share_{group}_{year_value}"] = float(value / reported_total)

        post_reported = _annual_sum(prediction, source_reported_column, year_value)
        post_symptomatic = _annual_sum(prediction, "symptomatic_infections_predicted", year_value)
        post_ascertainment = _safe_ratio(post_reported, post_symptomatic)
        row[f"latent_testing_log_index_{year_value}"] = float(
            np.log(_safe_ratio(post_ascertainment, baseline_ascertainment))
        )
        effective_ascertainment = _safe_ratio(
            _annual_sum(prediction, "reported_cases_predicted", year_value), post_symptomatic
        )
        row[f"latent_effective_ascertainment_log_index_{year_value}"] = float(
            np.log(_safe_ratio(effective_ascertainment, _safe_ratio(effective_baseline_reported, baseline_symptomatic)))
        )
        row[f"latent_under5_symptomatic_burden_{year_value}"] = _annual_sum(
            prediction, "symptomatic_infections_under5_predicted", year_value
        )
        row[f"latent_all_age_symptomatic_burden_{year_value}"] = post_symptomatic
        row[f"latent_severe_source_{year_value}"] = _annual_sum(
            prediction,
            "symptomatic_infections_infant_predicted",
            year_value,
        )

        if year_value in SEROLOGY_YEARS:
            populations = _population_by_group(data, year_value)
            for group, column in infection_columns.items():
                infection_count = _annual_sum(prediction, column, year_value)
                recent_infection_probability = float(
                    1.0 - np.exp(-max(infection_count, 0.0) / max(populations[group], 1.0))
                )
                assay_probability = (
                    sensitivity * recent_infection_probability
                    + (1.0 - specificity) * (1.0 - recent_infection_probability)
                )
                row[f"latent_serology_probability_{group}_{year_value}"] = float(
                    np.clip(assay_probability, 1e-6, 1.0 - 1e-6)
                )
                state = prediction.loc[
                    prediction["year"].eq(year_value) & prediction["month"].eq(1),
                    natural_immunity_columns[group],
                ]
                natural_immunity_share = (
                    float(state.iloc[0]) / max(populations[group], 1.0)
                    if not state.empty
                    else 0.0
                )
                immunity_marker_probability = (
                    sensitivity * natural_immunity_share
                    + (1.0 - specificity) * (1.0 - natural_immunity_share)
                )
                row[f"latent_serology_immunity_probability_{group}_{year_value}"] = float(
                    np.clip(immunity_marker_probability, 1e-6, 1.0 - 1e-6)
                )
    return row


def simulate_observation_features(
    latent: pd.Series | dict[str, Any],
    *,
    design: MeasurementDesign,
    seed: int,
    observation_replicate: int,
) -> dict[str, Any]:
    """Sample all candidate surveillance streams from one latent signal row."""
    # Separate streams preserve testing draws across notification matching
    # strengths even when count samplers consume different random numbers.
    rng_notifications, rng_age, rng_testing, rng_severe, rng_serology, rng_baseline = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(6)
    )
    row: dict[str, Any] = {
        "iso3": str(latent["iso3"]),
        "draw_id": str(latent["draw_id"]),
        "draw_rank": int(latent["draw_rank"]),
        "scenario_id": str(latent["scenario_id"]),
        "group_id": str(latent["group_id"]),
        "observation_replicate": int(observation_replicate),
        "measurement_design": design.design_id,
        "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
    }

    phi_month = max(float(latent["nb_phi_month"]), 0.2)
    simulated_monthly: dict[tuple[int, int], int] = {}
    for year_value in POST_YEARS:
        for month_value in range(1, 13):
            mean = float(latent[f"latent_notification_mean_{year_value}_m{month_value:02d}"])
            observed = int(negative_binomial_2_rng(np.asarray([mean]), phi_month, rng_notifications)[0])
            simulated_monthly[(year_value, month_value)] = observed
            baseline = float(latent[f"latent_notification_baseline_m{month_value:02d}"])
            row[f"feature_notifications_{year_value}_m{month_value:02d}"] = float(
                np.log((observed + 0.5) / (baseline + 0.5))
            )

    baseline_events_observed = design.severe_baseline_events
    if design.severe_baseline_observation_years is not None:
        baseline_years = float(design.severe_baseline_observation_years)
        if not np.isfinite(baseline_years) or baseline_years <= 0:
            raise ValueError("severe_baseline_observation_years must be positive.")
        baseline_events_observed = float(negative_binomial_2_rng(
            np.asarray([design.severe_baseline_events * baseline_years]),
            max(design.severe_dispersion * baseline_years, 0.2), rng_baseline,
        )[0]) / baseline_years
    row["severe_baseline_events_observed"] = baseline_events_observed
    row["severe_baseline_events_expected"] = design.severe_baseline_events

    for year_value in POST_YEARS:
        shares = np.asarray(
            [
                float(latent[f"latent_age_share_{group}_{year_value}"])
                for group in AGE_GROUPS
            ],
            dtype=float,
        )
        shares = np.maximum(shares, 1e-6)
        shares /= shares.sum()
        probabilities = rng_age.dirichlet(shares * max(design.age_concentration, 1e-3))
        annual_total = max(
            sum(simulated_monthly[(year_value, month_value)] for month_value in range(1, 13)),
            1,
        )
        age_counts = rng_age.multinomial(annual_total, probabilities)
        row[f"feature_age_under5_vs_adult_{year_value}"] = float(
            np.log((age_counts[0] + 0.5) / (age_counts[2] + 0.5))
        )
        row[f"feature_age_school_age_vs_adult_{year_value}"] = float(
            np.log((age_counts[1] + 0.5) / (age_counts[2] + 0.5))
        )

        testing_signal = float(latent[f"latent_testing_log_index_{year_value}"])
        row[f"feature_testing_log_index_{year_value}"] = float(
            testing_signal + rng_testing.normal(0.0, design.testing_log_sd)
        )

        severe_scale = design.severe_baseline_events / max(
            float(latent["latent_severe_source_baseline"]),
            1e-9,
        )
        severe_mean = max(float(latent[f"latent_severe_source_{year_value}"]) * severe_scale, 0.0)
        row[f"expected_severe_events_before_drift_{year_value}"] = severe_mean
        severe_mean *= float(
            np.exp(rng_severe.normal(-0.5 * design.severe_log_sd**2, design.severe_log_sd))
        )
        severe_observed = int(
            negative_binomial_2_rng(
                np.asarray([severe_mean]),
                max(design.severe_dispersion, 0.2),
                rng_severe,
            )[0]
        )
        row[f"observed_severe_events_{year_value}"] = severe_observed
        row[f"feature_severe_log_ratio_{year_value}"] = float(
            np.log((severe_observed + 0.5) / (baseline_events_observed + 0.5))
        )

    for year_value in SEROLOGY_YEARS:
        for group in AGE_GROUPS:
            probability = float(latent[f"latent_serology_probability_{group}_{year_value}"])
            positive = int(rng_serology.binomial(design.serology_n_per_stratum, probability))
            observed_probability = (positive + 0.5) / (design.serology_n_per_stratum + 1.0)
            row[f"feature_serology_logit_{group}_{year_value}"] = float(logit(observed_probability))
            immunity_probability = float(
                latent[f"latent_serology_immunity_probability_{group}_{year_value}"]
            )
            immunity_positive = int(
                rng_serology.binomial(design.serology_n_per_stratum, immunity_probability)
            )
            observed_immunity_probability = (
                immunity_positive + 0.5
            ) / (design.serology_n_per_stratum + 1.0)
            row[f"feature_serology_immunity_logit_{group}_{year_value}"] = float(
                logit(observed_immunity_probability)
            )
    return row


def assign_group_folds(groups: Iterable[str], n_folds: int = 5) -> dict[str, int]:
    """Assign stable folds while keeping paired scenarios and replicates together."""
    unique = sorted(set(str(group) for group in groups))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    return {group: index % n_folds for index, group in enumerate(unique)}


def _fit_shrinkage_lda(
    x: np.ndarray,
    y: np.ndarray,
    *,
    shrinkage: float = 0.80,
) -> dict[str, np.ndarray | float]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both structural classes.")
    centre = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z = np.nan_to_num((x - centre) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    mean0 = z[y == 0].mean(axis=0)
    mean1 = z[y == 1].mean(axis=0)
    residual = np.vstack([z[y == 0] - mean0, z[y == 1] - mean1])
    covariance = residual.T @ residual / max(len(z) - 2, 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    ridge = max(float(np.nanmean(np.diag(covariance))), 1.0) * 1e-5
    inverse = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
    weight = inverse @ (mean1 - mean0)
    intercept = float(-0.5 * (mean1 + mean0) @ weight)
    return {
        "centre": centre,
        "scale": scale,
        "weight": weight,
        "intercept": intercept,
    }


def _standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z = np.nan_to_num((x - centre) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    return z, centre, scale


def _apply_standardization(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    centre = np.asarray(model["centre"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    return np.nan_to_num((x - centre) / scale, nan=0.0, posinf=20.0, neginf=-20.0)


def _predict_shrinkage_lda(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    weight = np.asarray(model["weight"], dtype=float)
    z = _apply_standardization(model, x)
    score = z @ weight + float(model["intercept"])
    return expit(np.clip(score, -35.0, 35.0))


def _fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2_penalty: float = 1.0,
    max_iter: int = 80,
) -> dict[str, Any]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both structural classes.")
    z, centre, scale = _standardize_features(x)
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1], dtype=float)
    beta[0] = logit(np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6))
    penalty = np.full(design.shape[1], float(l2_penalty), dtype=float)
    penalty[0] = 0.0
    for _ in range(max_iter):
        probability = expit(np.clip(design @ beta, -35.0, 35.0))
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (y - probability) - penalty * beta
        hessian = (design.T * weights) @ design + np.diag(penalty + 1e-8)
        step = np.linalg.solve(hessian, gradient)
        beta += np.clip(step, -2.0, 2.0)
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return {"centre": centre, "scale": scale, "beta": beta}


def _predict_ridge_logistic(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = _apply_standardization(model, x)
    design = np.column_stack([np.ones(len(z)), z])
    return expit(np.clip(design @ np.asarray(model["beta"], dtype=float), -35.0, 35.0))


def _fit_rbf_prototype(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both structural classes.")
    z, centre, scale = _standardize_features(x)
    mean0 = z[y == 0].mean(axis=0)
    mean1 = z[y == 1].mean(axis=0)
    nearest = np.where(y[:, None] == 1, mean1, mean0)
    sigma2 = float(np.median(np.sum((z - nearest) ** 2, axis=1)))
    sigma2 = max(sigma2, 1e-3)
    return {"centre": centre, "scale": scale, "mean0": mean0, "mean1": mean1, "sigma2": sigma2}


def _predict_rbf_prototype(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = _apply_standardization(model, x)
    mean0 = np.asarray(model["mean0"], dtype=float)
    mean1 = np.asarray(model["mean1"], dtype=float)
    d0 = np.sum((z - mean0) ** 2, axis=1)
    d1 = np.sum((z - mean1) ** 2, axis=1)
    score = (d0 - d1) / (2.0 * float(model["sigma2"]))
    return expit(np.clip(score, -35.0, 35.0))


def _best_probability_stump(z: np.ndarray, y: np.ndarray) -> tuple[int, float, float, float]:
    best: tuple[float, int, float, float, float] | None = None
    for feature in range(z.shape[1]):
        values = z[:, feature]
        thresholds = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        for threshold in thresholds:
            left = values <= threshold
            right = ~left
            if left.sum() < 2 or right.sum() < 2:
                continue
            left_probability = (float(y[left].sum()) + 1.0) / (float(left.sum()) + 2.0)
            right_probability = (float(y[right].sum()) + 1.0) / (float(right.sum()) + 2.0)
            fitted = np.where(left, left_probability, right_probability)
            loss = float(np.mean(np.square(y - fitted)))
            candidate = (loss, feature, float(threshold), left_probability, right_probability)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        base = (float(y.sum()) + 1.0) / (float(len(y)) + 2.0)
        return 0, 0.0, base, base
    _, feature, threshold, left_probability, right_probability = best
    return feature, threshold, left_probability, right_probability


def _fit_random_forest_stumps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_estimators: int = 80,
) -> dict[str, Any]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both structural classes.")
    z, centre, scale = _standardize_features(x)
    rng = np.random.default_rng(seed)
    n_features = z.shape[1]
    max_features = max(1, int(round(np.sqrt(n_features))))
    stumps: list[tuple[np.ndarray, int, float, float, float]] = []
    for _ in range(n_estimators):
        rows = rng.integers(0, len(z), size=len(z))
        features = np.sort(rng.choice(n_features, size=max_features, replace=False))
        feature, threshold, left, right = _best_probability_stump(z[rows][:, features], y[rows])
        stumps.append((features, feature, threshold, left, right))
    return {"centre": centre, "scale": scale, "stumps": stumps}


def _predict_random_forest_stumps(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = _apply_standardization(model, x)
    probabilities = []
    for features, feature, threshold, left, right in model["stumps"]:
        values = z[:, np.asarray(features, dtype=int)][:, int(feature)]
        probabilities.append(np.where(values <= threshold, left, right))
    return np.clip(np.mean(np.vstack(probabilities), axis=0), 1e-6, 1.0 - 1e-6)


def _best_residual_stump(z: np.ndarray, residual: np.ndarray) -> tuple[int, float, float, float]:
    best: tuple[float, int, float, float, float] | None = None
    for feature in range(z.shape[1]):
        values = z[:, feature]
        thresholds = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        for threshold in thresholds:
            left = values <= threshold
            right = ~left
            if left.sum() < 2 or right.sum() < 2:
                continue
            left_value = float(residual[left].mean())
            right_value = float(residual[right].mean())
            fitted = np.where(left, left_value, right_value)
            loss = float(np.mean(np.square(residual - fitted)))
            candidate = (loss, feature, float(threshold), left_value, right_value)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        value = float(residual.mean())
        return 0, 0.0, value, value
    _, feature, threshold, left_value, right_value = best
    return feature, threshold, left_value, right_value


def _fit_gradient_boosted_stumps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 60,
    learning_rate: float = 0.15,
) -> dict[str, Any]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both structural classes.")
    z, centre, scale = _standardize_features(x)
    base = logit(np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6))
    score = np.full(len(y), base, dtype=float)
    stumps: list[tuple[int, float, float, float]] = []
    for _ in range(n_estimators):
        residual = y - expit(np.clip(score, -35.0, 35.0))
        feature, threshold, left, right = _best_residual_stump(z, residual)
        update = np.where(z[:, feature] <= threshold, left, right)
        score += learning_rate * update
        stumps.append((feature, threshold, left, right))
    return {
        "centre": centre,
        "scale": scale,
        "base_score": float(base),
        "learning_rate": float(learning_rate),
        "stumps": stumps,
    }


def _predict_gradient_boosted_stumps(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = _apply_standardization(model, x)
    score = np.full(len(z), float(model["base_score"]), dtype=float)
    for feature, threshold, left, right in model["stumps"]:
        score += float(model["learning_rate"]) * np.where(z[:, int(feature)] <= threshold, left, right)
    return expit(np.clip(score, -35.0, 35.0))


def _fit_random_features_neural(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_hidden: int = 24,
) -> dict[str, Any]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both structural classes.")
    z, centre, scale = _standardize_features(x)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0 / np.sqrt(max(z.shape[1], 1)), size=(z.shape[1], n_hidden))
    bias = rng.normal(0.0, 0.25, size=n_hidden)
    hidden = np.tanh(z @ weights + bias)
    output = _fit_ridge_logistic(hidden, y, l2_penalty=0.5)
    return {"centre": centre, "scale": scale, "weights": weights, "bias": bias, "output": output}


def _predict_random_features_neural(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = _apply_standardization(model, x)
    hidden = np.tanh(z @ np.asarray(model["weights"], dtype=float) + np.asarray(model["bias"], dtype=float))
    return _predict_ridge_logistic(model["output"], hidden)


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    shrinkage: float = 0.80,
    seed: int = 20260726,
) -> dict[str, Any]:
    if classifier == DEFAULT_CLASSIFIER:
        return {"classifier": classifier, "model": _fit_shrinkage_lda(x, y, shrinkage=shrinkage)}
    if classifier == "ridge_logistic":
        return {"classifier": classifier, "model": _fit_ridge_logistic(x, y)}
    if classifier == "random_forest_stumps":
        return {"classifier": classifier, "model": _fit_random_forest_stumps(x, y, seed=seed)}
    if classifier == "gradient_boosted_stumps":
        return {"classifier": classifier, "model": _fit_gradient_boosted_stumps(x, y)}
    if classifier == "random_features_neural":
        return {"classifier": classifier, "model": _fit_random_features_neural(x, y, seed=seed)}
    raise ValueError(f"Unknown classifier: {classifier}")


def _predict_classifier(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    classifier = str(model["classifier"])
    fitted = model["model"]
    if classifier == DEFAULT_CLASSIFIER:
        return _predict_shrinkage_lda(fitted, x)
    if classifier == "ridge_logistic":
        return _predict_ridge_logistic(fitted, x)
    if classifier == "random_forest_stumps":
        return _predict_random_forest_stumps(fitted, x)
    if classifier == "gradient_boosted_stumps":
        return _predict_gradient_boosted_stumps(fitted, x)
    if classifier == "random_features_neural":
        return _predict_random_features_neural(fitted, x)
    raise ValueError(f"Unknown classifier: {classifier}")


def roc_auc(y_true: Sequence[int], probability: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    positive = y == 1
    negative = y == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = rankdata(p, method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def discrimination_metrics(y_true: Sequence[int], probability: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    predicted = p >= 0.5
    sensitivity = float(predicted[y == 1].mean()) if np.any(y == 1) else np.nan
    specificity = float((~predicted[y == 0]).mean()) if np.any(y == 0) else np.nan
    return {
        "auc": roc_auc(y, p),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": float(np.mean(np.square(p - y))),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def operating_threshold(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    policy: str,
    target: float = 0.90,
) -> float:
    """Choose a deterministic operating threshold from training predictions.

    ``policy`` is one of ``youden``, ``sensitivity`` or ``specificity``.  Ties
    are resolved conservatively and deterministically: the largest threshold is
    used for sensitivity targets and the smallest threshold for specificity
    targets.  The function is deliberately separate from model fitting so that
    callers can enforce inner-training-only threshold selection.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold training data must contain both classes.")
    if policy not in {"youden", "sensitivity", "specificity"}:
        raise ValueError(f"Unknown threshold policy: {policy}")
    if not 0.0 < target <= 1.0:
        raise ValueError("Threshold target must lie in (0, 1].")
    candidates = np.unique(np.concatenate(([0.0], p, [1.0 + 1e-12])))
    rows: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = p >= threshold
        sensitivity = float(predicted[y == 1].mean())
        specificity = float((~predicted[y == 0]).mean())
        rows.append((float(threshold), sensitivity, specificity))
    if policy == "youden":
        best_value = max(sensitivity + specificity - 1.0 for _, sensitivity, specificity in rows)
        eligible = [row for row in rows if np.isclose(row[1] + row[2] - 1.0, best_value)]
        return float(np.median([row[0] for row in eligible]))
    if policy == "sensitivity":
        eligible = [row for row in rows if row[1] >= target]
        return float(max(row[0] for row in eligible))
    eligible = [row for row in rows if row[2] >= target]
    return float(min(row[0] for row in eligible))


def nested_country_operating_predictions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    policies: Sequence[str] = ("youden", "sensitivity", "specificity"),
    target: float = 0.90,
    n_inner_folds: int = 5,
    shrinkage: float = 0.80,
    classifier: str = DEFAULT_CLASSIFIER,
    classifier_seed: int = 20260726,
) -> pd.DataFrame:
    """Evaluate training-selected operating points in whole-country holdouts.

    For each outer held-out country, thresholds are selected from grouped
    out-of-fold predictions among the remaining countries.  The classifier is
    then fitted once to all outer-training rows and evaluated on the untouched
    country.  Feature centring and scaling therefore remain training-only at
    both levels.
    """
    required = {"group_id", "truth_active", "iso3", "scenario_id", "observation_replicate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Feature table is missing columns: " + ", ".join(missing))
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    countries = sorted(frame["iso3"].astype(str).unique())
    if len(countries) < 3:
        raise ValueError("Nested country operating points require at least three countries.")
    outputs: list[pd.DataFrame] = []
    for outer_fold, country in enumerate(countries):
        train = frame.loc[frame["iso3"].astype(str).ne(country)].copy()
        test = frame.loc[frame["iso3"].astype(str).eq(country)].copy()
        inner = cross_validated_predictions(
            train,
            feature_columns,
            n_folds=n_inner_folds,
            shrinkage=shrinkage,
            classifier=classifier,
            classifier_seed=classifier_seed + outer_fold * 1009,
        )
        thresholds = {
            policy: operating_threshold(
                inner["truth_active"],
                inner["probability_active"],
                policy=policy,
                target=target,
            )
            for policy in policies
        }
        model = _fit_classifier(
            train.loc[:, feature_columns].to_numpy(dtype=float),
            train["truth_active"].astype(int).to_numpy(),
            shrinkage=shrinkage,
            classifier=classifier,
            seed=classifier_seed + outer_fold * 1009 + 503,
        )
        probability = _predict_classifier(
            model,
            test.loc[:, feature_columns].to_numpy(dtype=float),
        )
        base = test[
            [
                "iso3", "draw_id", "draw_rank", "group_id", "scenario_id",
                "observation_replicate", "truth_active",
            ]
        ].copy()
        base["outer_fold"] = outer_fold
        base["holdout_country"] = country
        base["probability_active"] = probability
        base["classifier"] = classifier
        for policy, threshold in thresholds.items():
            part = base.copy()
            part["threshold_policy"] = policy
            part["threshold_target"] = target if policy != "youden" else np.nan
            part["threshold"] = threshold
            part["predicted_active"] = part["probability_active"].ge(threshold).astype(int)
            outputs.append(part)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["threshold_policy", "iso3", "draw_rank", "scenario_id", "observation_replicate"]
    )


def cross_validated_predictions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    n_folds: int = 5,
    shrinkage: float = 0.80,
    classifier: str = DEFAULT_CLASSIFIER,
    classifier_seed: int = 20260726,
) -> pd.DataFrame:
    """Return group-held-out probabilities for a balanced structural contrast."""
    required = {"group_id", "truth_active", "iso3", "scenario_id", "observation_replicate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {', '.join(missing)}")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    folds = assign_group_folds(frame["group_id"], n_folds=n_folds)
    work = frame.copy()
    work["fold"] = work["group_id"].map(folds).astype(int)
    outputs: list[pd.DataFrame] = []
    for fold in range(n_folds):
        train = work.loc[work["fold"].ne(fold)]
        test = work.loc[work["fold"].eq(fold)]
        x_train = train.loc[:, feature_columns].to_numpy(dtype=float)
        y_train = train["truth_active"].astype(int).to_numpy()
        model = _fit_classifier(
            x_train,
            y_train,
            shrinkage=shrinkage,
            classifier=classifier,
            seed=classifier_seed + fold * 1009,
        )
        probability = _predict_classifier(model, test.loc[:, feature_columns].to_numpy(dtype=float))
        part = test[
            ["iso3", "draw_id", "draw_rank", "group_id", "scenario_id", "observation_replicate", "truth_active", "fold"]
        ].copy()
        part["probability_active"] = probability
        part["classifier"] = classifier
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["iso3", "draw_rank", "scenario_id", "observation_replicate"]
    )


def leave_one_country_out_predictions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    shrinkage: float = 0.80,
    classifier: str = DEFAULT_CLASSIFIER,
    classifier_seed: int = 20260726,
) -> pd.DataFrame:
    """Return predictions when each complete country is excluded from training."""
    required = {"group_id", "truth_active", "iso3", "scenario_id", "observation_replicate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {', '.join(missing)}")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    countries = sorted(frame["iso3"].astype(str).unique())
    if len(countries) < 3:
        raise ValueError("Leave-one-country-out validation requires at least three countries.")
    outputs: list[pd.DataFrame] = []
    for fold, country in enumerate(countries):
        train = frame.loc[frame["iso3"].astype(str).ne(country)]
        test = frame.loc[frame["iso3"].astype(str).eq(country)]
        model = _fit_classifier(
            train.loc[:, feature_columns].to_numpy(dtype=float),
            train["truth_active"].astype(int).to_numpy(),
            shrinkage=shrinkage,
            classifier=classifier,
            seed=classifier_seed + fold * 1009,
        )
        probability = _predict_classifier(
            model,
            test.loc[:, feature_columns].to_numpy(dtype=float),
        )
        part = test[
            [
                "iso3",
                "draw_id",
                "draw_rank",
                "group_id",
                "scenario_id",
                "observation_replicate",
                "truth_active",
            ]
        ].copy()
        part["fold"] = fold
        part["holdout_country"] = country
        part["probability_active"] = probability
        part["classifier"] = classifier
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["iso3", "draw_rank", "scenario_id", "observation_replicate"]
    )


def cluster_bootstrap_auc(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = SURVEILLANCE_BOOTSTRAP_REPLICATES,
    seed: int = 20260726,
) -> tuple[float, float]:
    groups = sorted(predictions["group_id"].astype(str).unique())
    if len(groups) < 2:
        return np.nan, np.nan
    indices = {group: np.flatnonzero(predictions["group_id"].astype(str).to_numpy() == group) for group in groups}
    y = predictions["truth_active"].astype(int).to_numpy()
    p = predictions["probability_active"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        value = roc_auc(y[selected], p[selected])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def nested_country_draw_bootstrap_auc(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = SURVEILLANCE_BOOTSTRAP_REPLICATES,
    seed: int = SURVEILLANCE_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a country-first, draw-second percentile interval for AUC.

    The out-of-fold probabilities are treated as fixed. Each bootstrap
    replicate samples the observed countries with replacement and then, for
    every sampled country occurrence, independently samples that country's
    paired draw groups with replacement. All scenarios and observation
    replicates belonging to a sampled draw group are retained.
    """
    required = {
        "iso3",
        "draw_id",
        "group_id",
        "truth_active",
        "probability_active",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            "Predictions are missing nested-bootstrap columns: "
            + ", ".join(missing)
        )
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if predictions.empty:
        return np.nan, np.nan
    if predictions.loc[:, list(required)].isna().any().any():
        raise ValueError("Nested-bootstrap columns must not contain missing values.")

    work = predictions.copy()
    work["iso3"] = work["iso3"].astype(str)
    work["draw_id"] = work["draw_id"].astype(str)
    work["group_id"] = work["group_id"].astype(str)
    y = work["truth_active"].to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("truth_active must contain only zeroes and ones.")
    y = y.astype(int)
    probability = work["probability_active"].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("probability_active must contain only finite values.")

    group_mapping = work[["group_id", "iso3", "draw_id"]].drop_duplicates()
    group_country_counts = group_mapping.groupby("group_id")["iso3"].nunique()
    if group_country_counts.gt(1).any():
        raise ValueError("Each group_id must belong to exactly one country.")
    group_draw_counts = group_mapping.groupby("group_id")["draw_id"].nunique()
    if group_draw_counts.gt(1).any():
        raise ValueError("Each group_id must identify exactly one draw.")

    countries = sorted(work["iso3"].unique())
    if len(countries) < 2:
        return np.nan, np.nan
    groups = sorted(work["group_id"].unique())
    group_lookup = {group: index for index, group in enumerate(groups)}
    row_group_indices = work["group_id"].map(group_lookup).to_numpy(dtype=int)
    country_group_indices = [
        np.asarray(
            [
                group_lookup[group]
                for group in sorted(
                    group_mapping.loc[
                        group_mapping["iso3"].eq(country),
                        "group_id",
                    ].unique()
                )
            ],
            dtype=int,
        )
        for country in countries
    ]
    if any(len(country_groups) == 0 for country_groups in country_group_indices):
        raise ValueError("Every sampled country must contain at least one draw group.")

    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(work), dtype=int)
    values: list[float] = []
    for _ in range(n_bootstrap):
        group_multiplicity = np.zeros(len(groups), dtype=int)
        sampled_countries = rng.integers(
            0,
            len(countries),
            size=len(countries),
        )
        for country_index in sampled_countries:
            available_groups = country_group_indices[country_index]
            sampled_groups = available_groups[
                rng.integers(
                    0,
                    len(available_groups),
                    size=len(available_groups),
                )
            ]
            np.add.at(group_multiplicity, sampled_groups, 1)
        selected = np.repeat(
            row_indices,
            group_multiplicity[row_group_indices],
        )
        value = roc_auc(y[selected], probability[selected])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def shapley_stream_gains(package_results: pd.DataFrame) -> pd.DataFrame:
    """Compute Shapley-average marginal AUC gains across all stream subsets."""
    required = {"mechanism", "analysis_population", "measurement_design", "package_streams", "auc"}
    missing = sorted(required.difference(package_results.columns))
    if missing:
        raise ValueError(f"Package results are missing Shapley columns: {', '.join(missing)}")
    rows: list[dict[str, Any]] = []
    group_columns = ["mechanism", "analysis_population", "measurement_design"]
    n_streams = len(OPTIONAL_STREAMS)
    for keys, group in package_results.groupby(group_columns, sort=False):
        lookup: dict[frozenset[str], float] = {}
        for item in group.itertuples(index=False):
            streams = frozenset(filter(None, str(item.package_streams).split("|")))
            lookup[streams] = float(item.auc)
        if len(lookup) != 2**n_streams:
            continue
        for stream in OPTIONAL_STREAMS:
            gain = 0.0
            others = [candidate for candidate in OPTIONAL_STREAMS if candidate != stream]
            for size in range(len(others) + 1):
                weight = factorial(size) * factorial(n_streams - size - 1) / factorial(n_streams)
                for subset_tuple in combinations(others, size):
                    subset = frozenset(subset_tuple)
                    gain += weight * (lookup[subset | {stream}] - lookup[subset])
            rows.append(
                {
                    "mechanism": keys[0],
                    "analysis_population": keys[1],
                    "measurement_design": keys[2],
                    "stream": stream,
                    "shapley_auc_gain": float(gain),
                }
            )
    return pd.DataFrame(rows)


def minimum_successful_packages(
    package_results: pd.DataFrame,
    country_results: pd.DataFrame,
    *,
    pooled_auc_threshold: float = 0.80,
    country_auc_threshold: float = 0.70,
    country_count_threshold: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["mechanism", "analysis_population", "measurement_design"]
    country_count = (
        country_results.assign(country_success=lambda x: x["auc"] >= country_auc_threshold)
        .groupby(keys + ["package_id"], as_index=False)["country_success"]
        .sum()
        .rename(columns={"country_success": "n_countries_auc_ge_threshold"})
    )
    merged = package_results.merge(country_count, on=keys + ["package_id"], how="left")
    for group_keys, group in merged.groupby(keys, sort=False):
        eligible = group.loc[
            group["auc"].ge(pooled_auc_threshold)
            & group["n_countries_auc_ge_threshold"].ge(country_count_threshold)
        ].copy()
        if eligible.empty:
            best = group.sort_values(["auc", "n_optional_streams"], ascending=[False, True]).iloc[0]
            status = "no_package_met_success_rule"
        else:
            minimum_size = int(eligible["n_optional_streams"].min())
            best = eligible.loc[eligible["n_optional_streams"].eq(minimum_size)].sort_values(
                ["auc", "log_loss"],
                ascending=[False, True],
            ).iloc[0]
            status = "success_rule_met"
        rows.append(
            {
                "mechanism": group_keys[0],
                "analysis_population": group_keys[1],
                "measurement_design": group_keys[2],
                "recommendation_status": status,
                "recommended_package_id": best["package_id"],
                "recommended_package_streams": best["package_streams"],
                "n_optional_streams": int(best["n_optional_streams"]),
                "pooled_auc": float(best["auc"]),
                "pooled_auc_lower_95": float(best["auc_lower_95"]),
                "pooled_auc_upper_95": float(best["auc_upper_95"]),
                "n_countries_auc_ge_threshold": int(best["n_countries_auc_ge_threshold"]),
                "pooled_auc_threshold": pooled_auc_threshold,
                "country_auc_threshold": country_auc_threshold,
                "country_count_threshold": country_count_threshold,
            }
        )
    return pd.DataFrame(rows)
