"""Simulation-based recoverability utilities for Model 2C.

The recovery workflow is deliberately isolated from manuscript-default output
paths.  Synthetic observations are generated from a fitted posterior draw,
written to a tagged recovery directory, and can later be refitted with the
unchanged Model 2C Stan program.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.bayes.model2c_scenarios import _predict_monthly


RECOVERY_PARAMETERS = (
    "log_beta",
    "log_reporting_rate",
    "contact_sensitivity",
    "waning_rate",
    "log_post_transmission_multiplier",
    "log_post_reporting_multiplier",
    "rebound_peak_month",
    "rebound_width",
    "rebound_asymmetry",
    "rebound_release_gain",
    "secondary_rebound_delay",
    "secondary_rebound_width",
    "secondary_rebound_release_gain",
    "outbreak_seeding_gain",
    "seasonal_strength",
    "natural_boosting_efficiency",
    "pathogen_pressure_scale",
    "epidemic_feedback_gain",
    "endemic_reseed_pressure",
    "epidemic_pressure_retention",
    "post_reporting_peak_gain",
    "post_reporting_trend",
    "nb_phi_month",
    "nb_phi_age",
    "init_natural_share",
    "init_vaccine_recent_fraction",
    "init_natural_recent_fraction",
)

MECHANISM_PARAMETERS = {
    "post_reporting_shift": "log_post_reporting_multiplier",
    "post_transmission_shift": "log_post_transmission_multiplier",
}


@dataclass(frozen=True)
class RecoveryScenario:
    scenario_id: str
    scenario_class: str
    reporting_shift_enabled: bool
    transmission_context_enabled: bool
    description: str


def recovery_scenarios() -> tuple[RecoveryScenario, ...]:
    return (
        RecoveryScenario(
            "multi_mechanism",
            "multi_mechanism",
            True,
            True,
            "Posterior-draw reporting and post-2023 transmission mechanisms retained.",
        ),
        RecoveryScenario(
            "reporting_only",
            "single_mechanism",
            True,
            False,
            "Reporting shift retained; post-2023 transmission context neutralized.",
        ),
        RecoveryScenario(
            "transmission_only",
            "single_mechanism",
            False,
            True,
            "Post-2023 transmission context retained; reporting shift neutralized.",
        ),
        RecoveryScenario(
            "null_both",
            "mechanism_null",
            False,
            False,
            "Reporting shift and post-2023 transmission context both neutralized.",
        ),
    )


def apply_recovery_scenario(draw: pd.Series, scenario: RecoveryScenario) -> pd.Series:
    """Return a truth draw with the requested mechanisms retained/neutralized."""
    truth = draw.copy()
    if not scenario.reporting_shift_enabled:
        truth["log_post_reporting_multiplier"] = 0.0
        truth["post_reporting_peak_gain"] = 0.0
        truth["post_reporting_peak_multiplier"] = 1.0
        truth["post_reporting_trend"] = 0.0
        truth["post_reporting_trend_per_year"] = 0.0
    if not scenario.transmission_context_enabled:
        truth["log_post_transmission_multiplier"] = 0.0
        truth["post_2023_transmission_shift_multiplier"] = 1.0
        truth["rebound_release_gain"] = 0.0
        truth["secondary_rebound_release_gain"] = 0.0
        truth["pathogen_pressure_scale"] = 0.0
        truth["excess_epidemic_pressure_scale"] = 0.0
        truth["outbreak_seeding_gain"] = 0.0
        truth["outbreak_ignition_pulse_gain"] = 0.0
        truth["secondary_outbreak_ignition_pulse_gain"] = 0.0
    return truth


def negative_binomial_2_rng(mean: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Draw NB2 counts using its Gamma-Poisson representation."""
    mean = np.maximum(np.asarray(mean, dtype=float), 0.0)
    phi = max(float(phi), 1e-9)
    rate = rng.gamma(shape=phi, scale=mean / phi)
    return rng.poisson(rate).astype(int)


def synthetic_stan_data(
    base_data: dict[str, Any],
    truth: pd.Series,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Generate synthetic monthly and annual-age observations for one truth draw."""
    predicted = _predict_monthly(base_data, truth, include_age_predictions=True)
    if not isinstance(predicted, pd.DataFrame):
        raise TypeError("Model 2C prediction did not return a monthly frame.")
    rng = np.random.default_rng(seed)
    phi_month = float(truth.get("nb_phi_month", truth.get("nb_phi", 20.0)))
    phi_age = float(truth.get("nb_phi_age", 20.0))

    out = dict(base_data)
    fit_idx = np.asarray(base_data.get("month_obs_idx", []), dtype=int) - 1
    ppc_idx = np.asarray(base_data.get("month_ppc_obs_idx", []), dtype=int) - 1
    monthly_mean = predicted["reported_cases_predicted"].to_numpy(dtype=float)
    out["y_month"] = negative_binomial_2_rng(monthly_mean[fit_idx], phi_month, rng).tolist()
    out["y_month_ppc"] = negative_binomial_2_rng(monthly_mean[ppc_idx], phi_month, rng).tolist()

    year_index = np.asarray(base_data["year_index"], dtype=int) - 1
    n_years = int(base_data["N_years"])
    n_age = int(base_data["N_age"])
    annual_age_mean = np.zeros((n_years, n_age), dtype=float)
    for age in range(n_age):
        monthly_age = predicted[f"reported_cases_age_{age + 1}_predicted"].to_numpy(dtype=float)
        np.add.at(annual_age_mean[:, age], year_index, monthly_age)
    age_year_idx = np.asarray(base_data.get("age_obs_year_idx", []), dtype=int) - 1
    age_idx = np.asarray(base_data.get("age_obs_age_idx", []), dtype=int) - 1
    selected_age_mean = annual_age_mean[age_year_idx, age_idx]
    out["y_age"] = negative_binomial_2_rng(selected_age_mean, phi_age, rng).tolist()
    expected = {
        "month_fit_mean": monthly_mean[fit_idx],
        "month_ppc_mean": monthly_mean[ppc_idx],
        "age_mean": selected_age_mean,
    }
    return out, expected


def summarise_recovery_run(
    truth: pd.Series,
    posterior: pd.DataFrame,
    *,
    run_id: str,
    iso3: str,
    scenario_id: str,
    replicate: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise interval coverage, direction, and mechanism discrimination."""
    parameter_rows: list[dict[str, Any]] = []
    for parameter in RECOVERY_PARAMETERS:
        if parameter not in truth or parameter not in posterior:
            continue
        values = pd.to_numeric(posterior[parameter], errors="coerce").dropna()
        if values.empty:
            continue
        truth_value = float(truth[parameter])
        lower, median, upper = values.quantile([0.025, 0.5, 0.975]).tolist()
        null = 0.0 if parameter.startswith("log_post_") else np.nan
        truth_direction = int(np.sign(truth_value - null)) if np.isfinite(null) else 0
        posterior_direction = int(np.sign(median - null)) if np.isfinite(null) else 0
        parameter_rows.append(
            {
                "run_id": run_id,
                "iso3": iso3,
                "scenario_id": scenario_id,
                "replicate": replicate,
                "parameter": parameter,
                "truth": truth_value,
                "posterior_mean": float(values.mean()),
                "posterior_median": float(median),
                "lower_95": float(lower),
                "upper_95": float(upper),
                "covered_95": bool(lower <= truth_value <= upper),
                "bias": float(values.mean() - truth_value),
                "absolute_error": float(abs(values.mean() - truth_value)),
                "truth_direction": truth_direction,
                "posterior_direction": posterior_direction,
                "direction_correct": bool(truth_direction == 0 or truth_direction == posterior_direction),
            }
        )

    rope = float(np.log(1.05))
    mechanism_rows: list[dict[str, Any]] = []
    for mechanism, parameter in MECHANISM_PARAMETERS.items():
        if parameter not in truth or parameter not in posterior:
            continue
        values = pd.to_numeric(posterior[parameter], errors="coerce").dropna()
        truth_value = float(truth[parameter])
        truth_active = bool(abs(truth_value) > rope)
        numeric = values.to_numpy(dtype=float)
        positive_probability = float(np.mean(numeric > rope))
        negative_probability = float(np.mean(numeric < -rope))
        inactive_probability = float(np.mean(np.abs(numeric) <= rope))
        active_probability = max(positive_probability, negative_probability)
        classification = (
            "active"
            if active_probability >= 0.80
            else "inactive"
            if inactive_probability >= 0.80
            else "ambiguous"
        )
        mechanism_rows.append(
            {
                "run_id": run_id,
                "iso3": iso3,
                "scenario_id": scenario_id,
                "replicate": replicate,
                "mechanism": mechanism,
                "parameter": parameter,
                "truth": truth_value,
                "rope_half_width": rope,
                "truth_active": truth_active,
                "posterior_active_probability": active_probability,
                "posterior_inactive_probability": inactive_probability,
                "classification": classification,
                "discrimination_correct": bool(classification == ("active" if truth_active else "inactive")),
            }
        )
    return pd.DataFrame(parameter_rows), pd.DataFrame(mechanism_rows)


def recovery_aggregate(parameter_summary: pd.DataFrame, mechanism_summary: pd.DataFrame) -> pd.DataFrame:
    """Build compact aggregate metrics without treating simulation replicates as data."""
    rows: list[dict[str, Any]] = []
    if not parameter_summary.empty:
        for parameter, group in parameter_summary.groupby("parameter", sort=True):
            rows.append(
                {
                    "metric_family": "parameter_recovery",
                    "estimand": parameter,
                    "n_runs": int(len(group)),
                    "coverage_95": float(group["covered_95"].mean()),
                    "mean_bias": float(group["bias"].mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(group["bias"].to_numpy(dtype=float))))),
                    "direction_accuracy": float(group["direction_correct"].mean()),
                    "discrimination_accuracy": np.nan,
                }
            )
    if not mechanism_summary.empty:
        for mechanism, group in mechanism_summary.groupby("mechanism", sort=True):
            rows.append(
                {
                    "metric_family": "mechanism_discrimination",
                    "estimand": mechanism,
                    "n_runs": int(len(group)),
                    "coverage_95": np.nan,
                    "mean_bias": np.nan,
                    "rmse": np.nan,
                    "direction_accuracy": np.nan,
                    "discrimination_accuracy": float(group["discrimination_correct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def safe_recovery_tag(value: str) -> str:
    clean = "".join(character if character.isalnum() or character == "_" else "_" for character in value.lower())
    clean = "_".join(part for part in clean.split("_") if part)
    if not clean or clean in {"main", "full", "rank_one"}:
        raise ValueError("Recovery tag must be non-empty and distinct from manuscript-default suffixes.")
    return clean


def assert_isolated_recovery_dir(path: Path, outputs_root: Path) -> None:
    resolved = path.resolve()
    recovery_root = (outputs_root / "model2c_recovery").resolve()
    if recovery_root not in resolved.parents:
        raise ValueError(f"Recovery output must be a child of {recovery_root}.")
