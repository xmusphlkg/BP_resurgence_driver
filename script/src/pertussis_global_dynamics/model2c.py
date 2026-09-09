from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.bayes.model2b_data import (
    MODEL2B_END_YEAR,
    build_model2b_stan_data,
    model2b_default_initial_params,
    model2b_initial_values,
)
from pertussis_global_dynamics.bayes.model2c_scenarios import (
    MODEL2C_OUTPUT_VERSION,
    _select_scenario_draws,
    model2c_observation_audit,
    model2c_posterior_predictive,
)
from pertussis_global_dynamics.data.model2_inputs import (
    build_country_context,
    country_selection_audit,
    ensure_directories,
    load_inputs,
    select_pilot_countries,
)
from pertussis_global_dynamics.model2c_windows import (
    MODEL2C_FIT_END_YEAR,
    MODEL2C_STATE_END_YEAR,
    MODEL2C_STATE_START_YEAR,
    model2c_stan_data_window_metadata,
    model2c_window_metadata,
)
from pertussis_global_dynamics.paths import OUTPUTS, ROOT, SCRIPT


STAN_MODEL = SCRIPT / "stan" / "model2c_transmission_scenario_replay.stan"
MODEL2C_DEFAULT_PILOT_ISO3 = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
MODEL2C_DEFAULT_START_YEAR = MODEL2C_STATE_START_YEAR
MODEL2C_STAN_DRAW_COLUMNS = [
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
    "endemic_background_force",
    "epidemic_pressure_retention",
    "post_reporting_peak_gain",
    "post_reporting_peak_multiplier",
    "post_reporting_trend",
    "post_reporting_trend_per_year",
    "post_2023_transmission_shift_multiplier",
    "excess_epidemic_pressure_scale",
    "outbreak_ignition_pulse_gain",
    "secondary_outbreak_ignition_pulse_gain",
    "missed_natural_boosting_2020_2022",
    "natural_boosting_flow_2020_2022",
    "natural_exposure_recharge_flow_2020_2022",
    "routine_dtp1_shortfall_2020_2022",
    "routine_dtp3_shortfall_2020_2022",
    "routine_vaccination_shortfall_2020_2022",
    "natural_waned_stock_2023_start",
    "net_demographic_adjustment_2015_2026",
    "nb_phi_month",
    "nb_phi_age",
    "init_natural_share",
    "init_vaccine_recent_fraction",
    "init_natural_recent_fraction",
    "nb_phi",
    "true_infections_2023_2025",
    "symptomatic_infections_2023_2025",
    "reported_cases_2023_2025",
    "reported_cases_2026",
]
MODEL2C_STAN_PARAMETER_COLUMNS = [
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
]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            return max(minimum, int(raw))
        except ValueError:
            return default
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            return default
    return default


def _env_countries(name: str = "MODEL2C_COUNTRIES") -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()]


def _manuscript_min_draws() -> int:
    return _env_int("MODEL2C_MIN_MANUSCRIPT_DRAWS", 2000)


def _contact_mode_suffix() -> str:
    raw = os.environ.get("MODEL2C_CONTACT_MODE", os.environ.get("MODEL2B_CONTACT_MODE", "full"))
    raw = raw.strip().lower().replace("-", "_")
    return "full" if raw == "full" else "rank_one"


def _output_tag() -> str:
    raw = os.environ.get("MODEL2C_OUTPUT_TAG", "").strip().lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")


def _init_source_tag() -> str:
    raw = os.environ.get("MODEL2C_INIT_SOURCE_TAG", "").strip().lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")


def _output_suffix() -> str:
    tag = _output_tag()
    if tag:
        return tag
    return _contact_mode_suffix()


def _model2c_year_window() -> tuple[int, int]:
    start_year = _env_int("MODEL2C_START_YEAR", MODEL2C_DEFAULT_START_YEAR, minimum=1900)
    end_year = _env_int("MODEL2C_END_YEAR", MODEL2C_STATE_END_YEAR, minimum=start_year)
    return start_year, end_year


def _default_country_workers(country_count: int, chains: int) -> int:
    total_workers = _env_int("PERTUSSIS_WORKERS", os.cpu_count() or 1)
    parallel_chains = min(chains, _env_int("MODEL2C_PARALLEL_CHAINS", chains))
    return max(1, min(country_count, total_workers // max(parallel_chains, 1)))


def _country_workers(country_count: int, chains: int) -> int:
    default = _default_country_workers(country_count, chains)
    return min(country_count, _env_int("MODEL2C_COUNTRY_WORKERS", default))


def _complete_manuscript_output(combined: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    draws = combined.get("posterior_draws", pd.DataFrame())
    if draws.empty or not {"iso3", "draw"}.issubset(draws.columns):
        return False, "missing_posterior_draws"
    expected_iso = set(MODEL2C_DEFAULT_PILOT_ISO3)
    observed_iso = set(draws["iso3"].dropna().astype(str))
    missing = sorted(expected_iso.difference(observed_iso))
    if missing:
        return False, "missing_countries_" + ",".join(missing)
    draw_counts = draws.loc[draws["iso3"].isin(expected_iso)].groupby("iso3")["draw"].nunique()
    min_draws = _manuscript_min_draws()
    too_few = sorted(draw_counts.loc[draw_counts.lt(min_draws)].index.astype(str).tolist())
    if too_few:
        return False, f"posterior_draws_below_{min_draws}_" + ",".join(too_few)
    return True, "complete_nine_country_manuscript_output"


def _require_cmdstanpy():
    try:
        from cmdstanpy import CmdStanModel, cmdstan_path
    except ImportError as exc:
        raise SystemExit(
            "CmdStanPy is required for Model 2C. Install dependencies with "
            "`.venv/bin/python -m pip install -r requirements.txt`, then install CmdStan with "
            "`.venv/bin/python -m cmdstanpy.install_cmdstan`."
        ) from exc
    try:
        cmdstan_path()
    except ValueError as exc:
        raise SystemExit(
            "CmdStan is not installed. Run `.venv/bin/python -m cmdstanpy.install_cmdstan` "
            "or set CMDSTAN to an existing CmdStan installation."
        ) from exc
    return CmdStanModel


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _metadata_to_model2c(metadata: dict[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    converted = dict(metadata)
    converted["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
    converted["model2c_model_role"] = "scenario_first_counterfactual_transmission_model"
    converted["model2c_counterfactual_interpretation"] = (
        "Posterior replay scenario contrasts, not additive mechanism shares."
    )
    output_tag = _output_tag()
    converted["model2c_output_tag"] = output_tag or "main"
    converted["model2c_start_year"] = int(converted.get("start_year", MODEL2C_DEFAULT_START_YEAR))
    converted["model2c_end_year"] = int(converted.get("end_year", MODEL2B_END_YEAR))
    converted.update(model2c_window_metadata())
    converted["model2c_fit_end_year"] = int(converted.get("fit_end_year", MODEL2C_FIT_END_YEAR))
    converted["model2c_fit_window_semantics"] = (
        f"Likelihood fit through {converted['model2c_fit_end_year']}; later available monthly observations are "
        "reserved for temporal checks."
    )
    converted["model2c_state_start_year"] = int(converted.get("start_year", MODEL2C_STATE_START_YEAR))
    converted["model2c_state_end_year"] = int(converted.get("end_year", MODEL2C_STATE_END_YEAR))
    converted["model2c_contact_matrix_source"] = converted.get("model2b_contact_matrix_source")
    converted["model2c_contact_matrix_fallback"] = converted.get("model2b_contact_matrix_fallback")
    converted["model2c_month_obs_weight_min"] = converted.get("model2b_month_obs_weight_min")
    converted["model2c_month_obs_weight_mean"] = converted.get("model2b_month_obs_weight_mean")
    converted["model2c_month_obs_weight_max"] = converted.get("model2b_month_obs_weight_max")
    converted["model2c_initial_immunity_method"] = converted.get("model2b_initial_immunity_method")
    converted["model2c_age_state_transition_method"] = converted.get("model2b_age_state_transition_method")
    if data is not None:
        converted.update(model2c_stan_data_window_metadata(data))
    return converted


def _stan_data_path(iso3: str, contact_mode: str) -> Path:
    tag = _output_tag()
    suffix = f"_{tag}" if tag else ("_rank_one" if contact_mode == "rank_one" else "_full")
    return OUTPUTS / f"model2c_stan_data_{iso3}{suffix}.json"


def _prepare_country(inputs: dict[str, pd.DataFrame], iso3: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, float]]:
    contact_mode = _contact_mode_suffix()
    os.environ["MODEL2B_CONTACT_MODE"] = contact_mode
    context = build_country_context(iso3, inputs)
    params = model2b_default_initial_params()
    start_year, end_year = _model2c_year_window()
    data, metadata = build_model2b_stan_data(
        context,
        inputs,
        params,
        start_year=start_year,
        end_year=end_year,
    )
    metadata = _metadata_to_model2c(metadata, data)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = _stan_data_path(iso3, contact_mode)
    path.write_text(json.dumps(_json_ready(data), indent=2), encoding="utf-8")
    return path, data, metadata, params


def _ebfmi(energy: np.ndarray) -> float:
    energy = np.asarray(energy, dtype=float)
    if energy.size < 2 or not np.isfinite(energy).all():
        return float("nan")
    variance = float(np.var(energy, ddof=1))
    if variance <= 0:
        return float("nan")
    return float(np.mean(np.diff(energy) ** 2) / variance)


def _model2c_hmc_diagnostics(
    fit: Any,
    metadata: dict[str, Any],
    *,
    max_treedepth: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract per-chain HMC diagnostics from a completed CmdStan fit."""
    try:
        method = fit.method_variables()
        energy = np.asarray(method["energy__"], dtype=float)
        divergent = np.asarray(method["divergent__"], dtype=float)
        treedepth = np.asarray(method["treedepth__"], dtype=float)
    except Exception as exc:
        empty = pd.DataFrame()
        return empty, {
            "hmc_diagnostics_available": False,
            "hmc_diagnostic_error": f"{type(exc).__name__}: {exc}",
            "total_divergent_transitions": np.nan,
            "total_treedepth_saturations": np.nan,
            "max_treedepth_observed": np.nan,
            "minimum_ebfmi": np.nan,
        }
    if energy.ndim == 1:
        energy = energy[:, None]
        divergent = divergent[:, None]
        treedepth = treedepth[:, None]
    rows: list[dict[str, Any]] = []
    for chain_index in range(energy.shape[1]):
        rows.append(
            {
                "iso3": metadata["iso3"],
                "country": metadata["country"],
                "chain": chain_index + 1,
                "post_warmup_draws": int(energy.shape[0]),
                "divergent_transitions": int(np.sum(divergent[:, chain_index])),
                "treedepth_saturations": int(np.sum(treedepth[:, chain_index] >= int(max_treedepth))),
                "max_treedepth_observed": int(np.max(treedepth[:, chain_index])),
                "ebfmi": _ebfmi(energy[:, chain_index]),
                "max_treedepth": int(max_treedepth),
                "model2c_output_version": MODEL2C_OUTPUT_VERSION,
            }
        )
    chains = pd.DataFrame(rows)
    return chains, {
        "hmc_diagnostics_available": True,
        "hmc_diagnostic_error": "",
        "total_divergent_transitions": int(chains["divergent_transitions"].sum()),
        "total_treedepth_saturations": int(chains["treedepth_saturations"].sum()),
        "max_treedepth_observed": int(chains["max_treedepth_observed"].max()),
        "minimum_ebfmi": float(chains["ebfmi"].min()),
    }


def _fit_summary_row(
    fit: Any,
    metadata: dict[str, Any],
    chains: int,
    warmup: int,
    samples: int,
    *,
    hmc_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = fit.summary()
    core_summary = summary.loc[summary.index.intersection(MODEL2C_STAN_PARAMETER_COLUMNS)]
    max_rhat_all = float(summary["R_hat"].dropna().max()) if "R_hat" in summary else float("nan")
    min_ess_bulk_all = float(summary["ESS_bulk"].dropna().min()) if "ESS_bulk" in summary else float("nan")
    max_rhat = float(core_summary["R_hat"].dropna().max()) if "R_hat" in core_summary else float("nan")
    min_ess_bulk = float(core_summary["ESS_bulk"].dropna().min()) if "ESS_bulk" in core_summary else float("nan")
    hmc = hmc_summary or {
        "hmc_diagnostics_available": False,
        "hmc_diagnostic_error": "not_extracted",
        "total_divergent_transitions": np.nan,
        "total_treedepth_saturations": np.nan,
        "max_treedepth_observed": np.nan,
        "minimum_ebfmi": np.nan,
    }
    divergent = hmc.get("total_divergent_transitions", np.nan)
    depth_saturations = hmc.get("total_treedepth_saturations", np.nan)
    minimum_ebfmi = hmc.get("minimum_ebfmi", np.nan)
    registered_gate_pass = bool(
        np.isfinite(max_rhat)
        and np.isfinite(min_ess_bulk)
        and max_rhat < 1.01
        and min_ess_bulk >= 400.0
        and bool(hmc.get("hmc_diagnostics_available", False))
        and np.isfinite(divergent)
        and np.isfinite(depth_saturations)
        and np.isfinite(minimum_ebfmi)
        and int(divergent) == 0
        and int(depth_saturations) == 0
        and float(minimum_ebfmi) >= 0.3
    )
    return {
        **metadata,
        "chains": int(chains),
        "iter_warmup": int(warmup),
        "iter_sampling": int(samples),
        "posterior_draws": int(chains * samples),
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "max_rhat_all_reported_quantities": max_rhat_all,
        "min_ess_bulk_all_reported_quantities": min_ess_bulk_all,
        **hmc,
        "registered_hmc_gate_pass": registered_gate_pass,
        "registered_hmc_gate_rule": "core R-hat <1.01; core bulk ESS >=400; zero divergence; zero tree-depth saturation; minimum per-chain E-BFMI >=0.3",
        "diagnostic_scope": "core_sampled_parameters",
        "stan_model": str(STAN_MODEL.relative_to(ROOT)),
        "fit_status": "sampled",
        "model2c_output_version": MODEL2C_OUTPUT_VERSION,
    }


def _draws_dataframe(fit: Any, metadata: dict[str, Any]) -> pd.DataFrame:
    # CmdStanPy accepts the base variable name for vector-valued parameters,
    # not an individual element such as ``gain[1]``.  Structural refits add
    # indexed parameters to the manuscript draw contract, so request each base
    # variable once and then retain the explicitly registered expanded columns.
    requested_variables = list(
        dict.fromkeys(re.sub(r"\[[0-9]+\]$", "", column) for column in MODEL2C_STAN_DRAW_COLUMNS)
    )
    raw = fit.draws_pd(vars=["chain__", "iter__", "draw__"] + requested_variables)
    columns = [column for column in MODEL2C_STAN_DRAW_COLUMNS if column in raw.columns]
    draws = raw.loc[:, ["chain__", "iter__", "draw__"] + columns].copy()
    draws = draws.rename(columns={"chain__": "chain", "iter__": "iteration", "draw__": "draw"})
    draws["chain"] = draws["chain"].astype(int)
    draws["iteration"] = draws["iteration"].astype(int)
    draws["draw"] = draws["draw"].astype(int)
    draws.insert(0, "country", metadata["country"])
    draws.insert(0, "iso3", metadata["iso3"])
    draws["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
    return draws


def _chain_initial_values(base_init: dict[str, float], data: dict[str, Any], chains: int, seed: int) -> list[dict[str, float]]:
    bounds = {
        "log_beta": (-16.11809565, -1.38629436, 0.08),
        "log_reporting_rate": (-6.21460810, 0.40546511, 0.08),
        "contact_sensitivity": (0.0, float(data.get("contact_sensitivity_upper", 5.0)), 0.10),
        "waning_rate": (0.005, 0.40, 0.01),
        "log_post_transmission_multiplier": (-0.69314718, 0.91629073, 0.06),
        "log_post_reporting_multiplier": (-1.38629436, 1.38629436, 0.06),
        "rebound_peak_month": (1.01, 47.99, 0.40),
        "rebound_width": (0.76, 13.99, 0.25),
        "rebound_asymmetry": (-0.74, 0.74, 0.08),
        "rebound_release_gain": (0.05, 3.99, 0.12),
        "secondary_rebound_delay": (2.01, 23.99, 0.45),
        "secondary_rebound_width": (1.01, 15.99, 0.30),
        "secondary_rebound_release_gain": (0.001, 2.99, 0.08),
        "outbreak_seeding_gain": (0.0001, 0.299, 0.004),
        "seasonal_strength": (0.01, 2.49, 0.08),
        "natural_boosting_efficiency": (0.001, 0.999, 0.04),
        "pathogen_pressure_scale": (0.001, 2.49, 0.08),
        "epidemic_feedback_gain": (0.01, 24.99, 0.45),
        "endemic_reseed_pressure": (0.0001, 0.079, 0.002),
        "epidemic_pressure_retention": (0.151, 0.849, 0.04),
        "post_reporting_peak_gain": (0.001, 2.99, 0.08),
        "post_reporting_trend": (-0.79, 0.79, 0.05),
        "nb_phi_month": (0.21, 499.0, 1.25),
        "nb_phi_age": (0.21, 499.0, 1.25),
        "init_natural_share": (0.001, 0.499, 0.025),
        "init_vaccine_recent_fraction": (0.051, 0.949, 0.03),
        "init_natural_recent_fraction": (0.051, 0.949, 0.03),
    }
    rng = np.random.default_rng(seed)
    initial_values: list[dict[str, float]] = []
    for chain_index in range(chains):
        chain_init = dict(base_init)
        if chain_index > 0:
            for name, (lower, upper, scale) in bounds.items():
                if name in chain_init:
                    chain_init[name] = float(np.clip(float(chain_init[name]) + rng.normal(0.0, scale), lower, upper))
        initial_values.append(chain_init)
    return initial_values


def _posterior_initial_values(iso3: str, chains: int, seed: int) -> list[dict[str, float]] | None:
    tag = _init_source_tag()
    if not tag:
        return None
    path = OUTPUTS / f"model2c_posterior_draws_{tag}.csv"
    if not path.exists():
        print(
            f"MODEL2C_INIT_SOURCE_TAG={tag} requested but {path.relative_to(ROOT)} is missing; "
            "using default initial values.",
            flush=True,
        )
        return None
    needed = {"iso3", "chain", *MODEL2C_STAN_PARAMETER_COLUMNS}
    try:
        draws = pd.read_csv(path, usecols=lambda column: column in needed)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(
            f"Could not read posterior initial values from {path.relative_to(ROOT)} "
            f"({type(exc).__name__}: {exc}); using default initial values.",
            flush=True,
        )
        return None
    rows = draws.loc[draws["iso3"].astype(str).str.upper().eq(iso3.upper())].copy()
    if rows.empty:
        print(
            f"MODEL2C_INIT_SOURCE_TAG={tag} has no posterior rows for {iso3}; using default initial values.",
            flush=True,
        )
        return None
    parameters = [column for column in MODEL2C_STAN_PARAMETER_COLUMNS if column in rows.columns]
    if not parameters:
        print(
            f"MODEL2C_INIT_SOURCE_TAG={tag} has no sampled parameter columns for {iso3}; "
            "using default initial values.",
            flush=True,
        )
        return None
    rng = np.random.default_rng(seed)
    chain_values = sorted(pd.to_numeric(rows.get("chain", pd.Series(dtype=float)), errors="coerce").dropna().unique())
    initial_values: list[dict[str, float]] = []
    for chain_index in range(chains):
        if chain_values:
            source_chain = chain_values[chain_index % len(chain_values)]
            pool = rows.loc[pd.to_numeric(rows["chain"], errors="coerce").eq(source_chain)]
        else:
            pool = rows
        if pool.empty:
            pool = rows
        record = pool.iloc[int(rng.integers(0, len(pool)))]
        init = {
            parameter: float(record[parameter])
            for parameter in parameters
            if np.isfinite(pd.to_numeric(pd.Series([record[parameter]]), errors="coerce").iloc[0])
        }
        if len(init) != len(parameters):
            return None
        initial_values.append(init)
    print(
        f"Using posterior initial values for {iso3} from model2c_posterior_draws_{tag}.csv.",
        flush=True,
    )
    return initial_values


def _run_country(
    inputs: dict[str, pd.DataFrame],
    iso3: str,
    *,
    chains: int,
    warmup: int,
    samples: int,
    seed: int,
    adapt_delta: float,
    adapt_engaged: bool,
    step_size: float,
    max_treedepth: int,
    model: Any | None = None,
    model_exe_file: str | None = None,
) -> dict[str, pd.DataFrame]:
    if model is None:
        from cmdstanpy import CmdStanModel

        if model_exe_file is None:
            raise ValueError("Either model or model_exe_file must be provided.")
        model = CmdStanModel(exe_file=str(model_exe_file))
    json_path, data, metadata, params = _prepare_country(inputs, iso3)
    if (
        str(metadata.get("contact_mode_requested", "")).lower() == "full"
        and bool(metadata.get("model2c_contact_matrix_fallback", False))
        and not _env_bool("MODEL2C_ALLOW_CONTACT_FALLBACK", False)
    ):
        raise SystemExit(
            f"Refusing to run Model 2C full-contact main analysis for {iso3}: "
            f"{metadata.get('model2c_contact_matrix_source', 'contact_matrix_fallback')}. "
            "Set MODEL2C_ALLOW_CONTACT_FALLBACK=1 only for explicit sensitivity debugging."
        )
    init = model2b_initial_values(params)
    if "log_reporting_rate_prior_mean" in data:
        init["log_reporting_rate"] = float(data["log_reporting_rate_prior_mean"])
    init["log_post_transmission_multiplier"] = min(
        0.91629073,
        max(-0.69314718, float(init.get("log_post_transmission_multiplier", 0.0))),
    )
    init["rebound_asymmetry"] = float(np.clip(init.get("rebound_asymmetry", 0.0), -0.74, 0.74))
    init["secondary_rebound_delay"] = float(np.clip(init.get("secondary_rebound_delay", 9.0), 2.01, 23.99))
    init["secondary_rebound_width"] = float(np.clip(init.get("secondary_rebound_width", 5.0), 1.01, 15.99))
    init["secondary_rebound_release_gain"] = float(
        np.clip(init.get("secondary_rebound_release_gain", 0.35), 0.001, 2.99)
    )
    init["post_reporting_peak_gain"] = float(np.clip(init.get("post_reporting_peak_gain", 0.15), 0.001, 2.99))
    init["post_reporting_trend"] = float(np.clip(init.get("post_reporting_trend", 0.0), -0.79, 0.79))
    init["init_natural_share"] = float(data.get("init_natural_share_prior_mean", init["init_natural_share"]))
    init["init_vaccine_recent_fraction"] = float(
        data.get("init_vaccine_recent_fraction_prior_mean", init["init_vaccine_recent_fraction"])
    )
    init["init_natural_recent_fraction"] = float(
        data.get("init_natural_recent_fraction_prior_mean", init["init_natural_recent_fraction"])
    )
    posterior_init = _posterior_initial_values(str(metadata["iso3"]), chains, seed)
    sample_kwargs = {
        "data": str(json_path),
        "chains": chains,
        "parallel_chains": min(chains, _env_int("MODEL2C_PARALLEL_CHAINS", chains)),
        "iter_warmup": warmup,
        "iter_sampling": samples,
        "seed": seed,
        "inits": posterior_init or _chain_initial_values(init, data, chains, seed),
        "adapt_engaged": adapt_engaged,
        "max_treedepth": max_treedepth,
        "show_progress": False,
        "show_console": _env_bool("MODEL2C_SHOW_CONSOLE", False),
    }
    if adapt_engaged:
        sample_kwargs["adapt_delta"] = adapt_delta
    if np.isfinite(step_size):
        sample_kwargs["step_size"] = step_size
    fit = model.sample(**sample_kwargs)
    draws = _draws_dataframe(fit, metadata)
    country_draws = draws.drop(columns=["iso3", "country"])
    hmc_chains, hmc_summary = _model2c_hmc_diagnostics(
        fit,
        metadata,
        max_treedepth=max_treedepth,
    )
    predictive_draws = _select_scenario_draws(
        country_draws,
        max(1, _env_int("MODEL2C_SCENARIO_MAX_DRAWS", 160)),
    )
    return {
        "fit_summary": pd.DataFrame([
            _fit_summary_row(
                fit,
                metadata,
                chains,
                warmup,
                samples,
                hmc_summary=hmc_summary,
            )
        ]),
        "posterior_draws": draws,
        "posterior_predictive": model2c_posterior_predictive(data, metadata, predictive_draws),
        "observation_audit": model2c_observation_audit(data, metadata, country_draws),
        "metadata": pd.DataFrame([metadata]),
        "hmc_diagnostics": hmc_chains,
    }


def _write_outputs(tables: list[dict[str, pd.DataFrame]]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    names = [
        "fit_summary",
        "posterior_draws",
        "posterior_predictive",
        "observation_audit",
        "metadata",
        "hmc_diagnostics",
    ]
    combined = {
        name: pd.concat([item[name] for item in tables if not item[name].empty], ignore_index=True, sort=False)
        if any(not item[name].empty for item in tables)
        else pd.DataFrame()
        for name in names
    }
    contact_mode = _contact_mode_suffix()
    complete_output, completeness_status = _complete_manuscript_output(combined)
    output_tag = _output_tag()
    if not complete_output:
        print(
            "Model 2C run is not a complete manuscript output: "
            f"{completeness_status}. Writing partial files only.",
            flush=True,
        )
    suffix_mode = _output_suffix()
    output_paths = {
        "fit_summary": "model2c_fit_summary.csv",
        "posterior_draws": "model2c_posterior_draws.csv",
        "posterior_predictive": "model2c_posterior_predictive.csv",
        "observation_audit": "model2c_observation_process_audit.csv",
        "metadata": "model2c_stan_data_metadata.csv",
        "hmc_diagnostics": "model2c_hmc_diagnostics.csv",
    }
    for name, filename in output_paths.items():
        stem = filename.removesuffix(".csv")
        combined[name].to_csv(OUTPUTS / f"{stem}_{suffix_mode}.csv", index=False)
        if not output_tag and complete_output and contact_mode == "full":
            combined[name].to_csv(OUTPUTS / filename, index=False)
    if not complete_output and _env_bool("MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS", False):
        raise SystemExit(
            "Refusing to update Model 2C manuscript-default outputs from an incomplete run: "
            f"{completeness_status}."
        )


def prepare_model2c_data(countries: list[str]) -> pd.DataFrame:
    ensure_directories()
    inputs = load_inputs()
    rows = []
    for iso3 in countries:
        _path, _data, metadata, _params = _prepare_country(inputs, iso3)
        rows.append(metadata)
    metadata = pd.DataFrame(rows)
    contact_mode = _contact_mode_suffix()
    output_tag = _output_tag()
    suffix_mode = _output_suffix()
    metadata.to_csv(OUTPUTS / f"model2c_stan_data_metadata_{suffix_mode}.csv", index=False)
    if not output_tag and contact_mode == "full":
        metadata.to_csv(OUTPUTS / "model2c_stan_data_metadata.csv", index=False)
    selection = country_selection_audit(inputs, countries)
    selection = selection.rename(
        columns={
            "selected_for_current_model2": "selected_for_current_model2c",
            "model2_selection_reason": "model2c_selection_reason",
        }
    )
    selection["selected_for_current_model2"] = selection["selected_for_current_model2c"]
    selection["model2_selection_reason"] = selection["model2c_selection_reason"].astype(str).str.replace(
        "Model 2", "Model 2C", regex=False
    )
    selection["model2c_selection_reason"] = selection["model2_selection_reason"]
    selection["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
    selection["model2c_output_tag"] = output_tag or "main"
    if output_tag:
        selection.to_csv(OUTPUTS / f"model2c_country_selection_audit_{output_tag}.csv", index=False)
    else:
        selection.to_csv(OUTPUTS / "model2c_country_selection_audit.csv", index=False)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian Model 2C scenario-first transmission model.")
    parser.add_argument("--countries", nargs="*", help="ISO3 countries to fit. Defaults to dynamic-pilot selection.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write CmdStan JSON data files.")
    args = parser.parse_args(argv)

    ensure_directories()
    inputs = load_inputs()
    countries = args.countries or _env_countries() or select_pilot_countries(inputs, limit=_env_int("MODEL2C_COUNTRY_LIMIT", 9))
    if args.prepare_only:
        prepare_model2c_data(countries)
        print(f"Wrote Model 2C Stan data for {len(countries)} countries.")
        return

    CmdStanModel = _require_cmdstanpy()
    chains = _env_int("MODEL2C_CHAINS", 4)
    warmup = _env_int("MODEL2C_WARMUP", 500, minimum=0)
    samples = _env_int("MODEL2C_SAMPLES", 500)
    adapt_delta = _env_float("MODEL2C_ADAPT_DELTA", 0.95)
    adapt_engaged = warmup > 0 and _env_bool("MODEL2C_ADAPT_ENGAGED", True)
    step_size = _env_float("MODEL2C_STEP_SIZE", float("nan"))
    max_treedepth = _env_int("MODEL2C_MAX_TREEDEPTH", 12)
    seed = _env_int("MODEL2C_SEED", 20260617)
    model = CmdStanModel(stan_file=str(STAN_MODEL))
    country_workers = _country_workers(len(countries), chains)

    if country_workers > 1:
        model_exe_file = getattr(model, "exe_file", str(STAN_MODEL.with_suffix("")))
        tables_by_index: list[dict[str, pd.DataFrame] | None] = [None for _ in countries]
        with ProcessPoolExecutor(max_workers=country_workers) as executor:
            futures = {
                executor.submit(
                    _run_country,
                    inputs,
                    iso3,
                    chains=chains,
                    warmup=warmup,
                    samples=samples,
                    seed=seed + index,
                    adapt_delta=adapt_delta,
                    adapt_engaged=adapt_engaged,
                    step_size=step_size,
                    max_treedepth=max_treedepth,
                    model=None,
                    model_exe_file=model_exe_file,
                ): (index, iso3)
                for index, iso3 in enumerate(countries)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, iso3 = futures[future]
                tables_by_index[index] = future.result()
                print(
                    f"Model 2C completed {iso3} ({completed}/{len(countries)}) "
                    f"with {country_workers} country workers.",
                    flush=True,
                )
        tables = [table for table in tables_by_index if table is not None]
    else:
        tables = [
            _run_country(
                inputs,
                iso3,
                chains=chains,
                warmup=warmup,
                samples=samples,
                seed=seed + index,
                adapt_delta=adapt_delta,
                adapt_engaged=adapt_engaged,
                step_size=step_size,
                max_treedepth=max_treedepth,
                model=model,
            )
            for index, iso3 in enumerate(countries)
        ]
    _write_outputs(tables)
    selection = country_selection_audit(inputs, countries)
    selection = selection.rename(
        columns={
            "selected_for_current_model2": "selected_for_current_model2c",
            "model2_selection_reason": "model2c_selection_reason",
        }
    )
    selection["selected_for_current_model2"] = selection["selected_for_current_model2c"]
    selection["model2_selection_reason"] = selection["model2c_selection_reason"].astype(str).str.replace(
        "Model 2", "Model 2C", regex=False
    )
    selection["model2c_selection_reason"] = selection["model2_selection_reason"]
    selection["model2c_output_version"] = MODEL2C_OUTPUT_VERSION
    output_tag = _output_tag()
    selection["model2c_output_tag"] = output_tag or "main"
    if output_tag:
        selection.to_csv(OUTPUTS / f"model2c_country_selection_audit_{output_tag}.csv", index=False)
    else:
        selection.to_csv(OUTPUTS / "model2c_country_selection_audit.csv", index=False)
    print(f"Wrote Model 2C Bayesian scenario outputs for {len(countries)} countries.")
