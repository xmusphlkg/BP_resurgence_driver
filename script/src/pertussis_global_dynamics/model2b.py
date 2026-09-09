from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from pertussis_global_dynamics.bayes.model2b_attribution import (
    MODEL2B_OUTPUT_VERSION,
    MODEL2B_OUTCOMES,
    MODEL2B_PRIMARY_MECHANISMS,
    model2b_country_outputs,
    summarize_model2b_pooled_attribution,
)
from pertussis_global_dynamics.bayes.model2b_data import (
    model2b_default_initial_params,
    model2b_initial_values,
    write_model2b_stan_data,
)
from pertussis_global_dynamics.data.model2_inputs import (
    build_country_context,
    country_selection_audit,
    ensure_directories,
    load_inputs,
    select_pilot_countries,
)
from pertussis_global_dynamics.paths import FIGURE_DATA, OUTPUTS, ROOT, SCRIPT


STAN_MODEL = SCRIPT / "stan" / "model2b_transmission_attribution.stan"
MODEL2B_DEFAULT_PILOT_ISO3 = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
MODEL2B_STAN_DRAW_COLUMNS = [
    "log_beta",
    "log_reporting_rate",
    "contact_sensitivity",
    "waning_rate",
    "log_post_transmission_multiplier",
    "log_post_reporting_multiplier",
    "rebound_peak_month",
    "rebound_width",
    "rebound_release_gain",
    "outbreak_seeding_gain",
    "seasonal_strength",
    "natural_boosting_efficiency",
    "pathogen_pressure_scale",
    "epidemic_feedback_gain",
    "endemic_reseed_pressure",
    "endemic_background_force",
    "epidemic_pressure_retention",
    "post_2023_transmission_shift_multiplier",
    "excess_epidemic_pressure_scale",
    "outbreak_ignition_pulse_gain",
    "missed_natural_boosting_2020_2022",
    "natural_boosting_flow_2020_2022",
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
MODEL2B_STAN_PARAMETER_COLUMNS = [
    "log_beta",
    "log_reporting_rate",
    "contact_sensitivity",
    "waning_rate",
    "log_post_transmission_multiplier",
    "log_post_reporting_multiplier",
    "rebound_peak_month",
    "rebound_width",
    "rebound_release_gain",
    "outbreak_seeding_gain",
    "seasonal_strength",
    "natural_boosting_efficiency",
    "pathogen_pressure_scale",
    "epidemic_feedback_gain",
    "endemic_reseed_pressure",
    "epidemic_pressure_retention",
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


def _manuscript_min_draws() -> int:
    return _env_int("MODEL2B_MIN_MANUSCRIPT_DRAWS", 2000)


def _complete_manuscript_output(combined: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    draws = combined.get("posterior_draws", pd.DataFrame())
    if draws.empty or not {"iso3", "draw"}.issubset(draws.columns):
        return False, "missing_posterior_draws"
    expected_iso = set(MODEL2B_DEFAULT_PILOT_ISO3)
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


def _env_countries(name: str = "MODEL2B_COUNTRIES") -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()]


def _default_country_workers(country_count: int, chains: int) -> int:
    total_workers = _env_int("PERTUSSIS_WORKERS", os.cpu_count() or 1)
    parallel_chains = min(chains, _env_int("MODEL2B_PARALLEL_CHAINS", chains))
    return max(1, min(country_count, total_workers // max(parallel_chains, 1)))


def _country_workers(country_count: int, chains: int) -> int:
    default = _default_country_workers(country_count, chains)
    return min(country_count, _env_int("MODEL2B_COUNTRY_WORKERS", default))


def _require_cmdstanpy():
    try:
        from cmdstanpy import CmdStanModel, cmdstan_path
    except ImportError as exc:
        raise SystemExit(
            "CmdStanPy is required for Model 2B. Install dependencies with "
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


def _fit_summary_row(fit: Any, metadata: dict[str, Any], chains: int, warmup: int, samples: int) -> dict[str, Any]:
    summary = fit.summary()
    core_summary = summary.loc[summary.index.intersection(MODEL2B_STAN_PARAMETER_COLUMNS)]
    max_rhat_all = float(summary["R_hat"].dropna().max()) if "R_hat" in summary else float("nan")
    min_ess_bulk_all = float(summary["ESS_bulk"].dropna().min()) if "ESS_bulk" in summary else float("nan")
    max_rhat = float(core_summary["R_hat"].dropna().max()) if "R_hat" in core_summary else float("nan")
    min_ess_bulk = float(core_summary["ESS_bulk"].dropna().min()) if "ESS_bulk" in core_summary else float("nan")
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
        "diagnostic_scope": "core_sampled_parameters",
        "stan_model": str(STAN_MODEL.relative_to(ROOT)),
        "fit_status": "sampled",
        "model2b_month_obs_weight_min": metadata.get("model2b_month_obs_weight_min", float("nan")),
        "model2b_month_obs_weight_mean": metadata.get("model2b_month_obs_weight_mean", float("nan")),
        "model2b_month_obs_weight_max": metadata.get("model2b_month_obs_weight_max", float("nan")),
        "model2b_output_version": MODEL2B_OUTPUT_VERSION,
    }


def _draws_dataframe(fit: Any, metadata: dict[str, Any]) -> pd.DataFrame:
    raw = fit.draws_pd(vars=["chain__", "iter__", "draw__"] + MODEL2B_STAN_DRAW_COLUMNS)
    columns = [column for column in MODEL2B_STAN_DRAW_COLUMNS if column in raw.columns]
    draws = raw.loc[:, ["chain__", "iter__", "draw__"] + columns].copy()
    draws = draws.rename(columns={"chain__": "chain", "iter__": "iteration", "draw__": "draw"})
    draws["chain"] = draws["chain"].astype(int)
    draws["iteration"] = draws["iteration"].astype(int)
    draws["draw"] = draws["draw"].astype(int)
    draws.insert(0, "country", metadata["country"])
    draws.insert(0, "iso3", metadata["iso3"])
    draws["model2b_output_version"] = MODEL2B_OUTPUT_VERSION
    return draws


def _prepare_country(inputs: dict[str, pd.DataFrame], iso3: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, float]]:
    context = build_country_context(iso3, inputs)
    params = model2b_default_initial_params()
    json_path, data, metadata = write_model2b_stan_data(context, inputs, params, output_dir=OUTPUTS)
    return json_path, data, metadata, params


def _run_country(
    inputs: dict[str, pd.DataFrame],
    iso3: str,
    *,
    chains: int,
    warmup: int,
    samples: int,
    seed: int,
    adapt_delta: float,
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
        and bool(metadata.get("model2b_contact_matrix_fallback", False))
        and not _env_bool("MODEL2B_ALLOW_CONTACT_FALLBACK", False)
    ):
        raise SystemExit(
            f"Refusing to run Model 2B full-contact main analysis for {iso3}: "
            f"{metadata.get('model2b_contact_matrix_source', 'contact_matrix_fallback')}. "
            "Set MODEL2B_ALLOW_CONTACT_FALLBACK=1 only for explicit sensitivity debugging."
        )
    init = model2b_initial_values(params)
    if "log_reporting_rate_prior_mean" in data:
        init["log_reporting_rate"] = float(data["log_reporting_rate_prior_mean"])
    init["log_post_transmission_multiplier"] = min(
        0.91629073,
        max(-0.69314718, float(init.get("log_post_transmission_multiplier", 0.0))),
    )
    init["init_natural_share"] = float(data.get("init_natural_share_prior_mean", init["init_natural_share"]))
    init["init_vaccine_recent_fraction"] = float(
        data.get("init_vaccine_recent_fraction_prior_mean", init["init_vaccine_recent_fraction"])
    )
    init["init_natural_recent_fraction"] = float(
        data.get("init_natural_recent_fraction_prior_mean", init["init_natural_recent_fraction"])
    )
    fit = model.sample(
        data=str(json_path),
        chains=chains,
        parallel_chains=min(chains, _env_int("MODEL2B_PARALLEL_CHAINS", chains)),
        iter_warmup=warmup,
        iter_sampling=samples,
        seed=seed,
        inits=[init for _ in range(chains)],
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=False,
        show_console=_env_bool("MODEL2B_SHOW_CONSOLE", False),
    )
    draws = _draws_dataframe(fit, metadata)
    country_draws = draws.drop(columns=["iso3", "country"])
    (
        shapley,
        factor_summary,
        layered_attribution_summary,
        observation_audit,
        transmission_context_audit,
        posterior_predictive,
    ) = model2b_country_outputs(
        data,
        metadata,
        country_draws,
    )
    fit_summary = pd.DataFrame([_fit_summary_row(fit, metadata, chains, warmup, samples)])
    return {
        "fit_summary": fit_summary,
        "posterior_draws": draws,
        "shapley_draws": shapley,
        "factor_summary": factor_summary,
        "layered_attribution_summary": layered_attribution_summary,
        "observation_audit": observation_audit,
        "transmission_context_audit": transmission_context_audit,
        "posterior_predictive": posterior_predictive,
        "metadata": pd.DataFrame([metadata]),
    }


def _write_outputs(tables: list[dict[str, pd.DataFrame]]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    names = [
        "fit_summary",
        "posterior_draws",
        "posterior_predictive",
        "observation_audit",
        "transmission_context_audit",
        "shapley_draws",
        "factor_summary",
        "layered_attribution_summary",
        "metadata",
    ]
    combined = {
        name: pd.concat([item[name] for item in tables if not item[name].empty], ignore_index=True, sort=False)
        if any(not item[name].empty for item in tables)
        else pd.DataFrame()
        for name in names
    }
    contact_mode = _contact_mode_suffix()
    complete_output, completeness_status = _complete_manuscript_output(combined)
    if not complete_output:
        print(
            "Model 2B run is not a complete manuscript output: "
            f"{completeness_status}. Writing partial files only.",
            flush=True,
        )
    output_paths = {
        "fit_summary": "model2b_bayes_fit_summary.csv",
        "posterior_draws": "model2b_bayes_posterior_draws.csv",
        "posterior_predictive": "model2b_bayes_posterior_predictive.csv",
        "observation_audit": "model2b_bayes_observation_process_audit.csv",
        "transmission_context_audit": "model2b_bayes_transmission_context_audit.csv",
        "shapley_draws": "model2b_bayes_shapley_draws.csv",
        "factor_summary": "model2b_bayes_factor_weight_summary.csv",
        "layered_attribution_summary": "model2b_bayes_layered_attribution_summary.csv",
        "metadata": "model2b_stan_data_metadata.csv",
    }
    suffix_mode = contact_mode if complete_output else f"partial_{contact_mode}"
    for name, filename in output_paths.items():
        stem = filename.removesuffix(".csv")
        combined[name].to_csv(OUTPUTS / f"{stem}_{suffix_mode}.csv", index=False)
        if complete_output and contact_mode == "full":
            combined[name].to_csv(OUTPUTS / filename, index=False)
    factor_summary = combined["factor_summary"]
    if not factor_summary.empty and "outcome_target" in factor_summary:
        figure_data = factor_summary.loc[
            factor_summary["outcome_target"].eq(MODEL2B_OUTCOMES["true_infections"])
        ].copy()
    else:
        figure_data = pd.DataFrame()
    figure_data.to_csv(FIGURE_DATA / f"figure_4_bayesian_transmission_attribution_{suffix_mode}.csv", index=False)
    if complete_output and contact_mode == "full":
        figure_data.to_csv(FIGURE_DATA / "figure_4_bayesian_transmission_attribution.csv", index=False)
    pooled = summarize_model2b_pooled_attribution(combined["shapley_draws"])
    pooled.to_csv(OUTPUTS / f"model2b_bayes_pooled_9country_attribution_{suffix_mode}.csv", index=False)
    if complete_output and contact_mode == "full":
        pooled.to_csv(OUTPUTS / "model2b_bayes_pooled_9country_attribution.csv", index=False)
    _write_contact_sensitivity_summary()
    if not complete_output and _env_bool("MODEL2B_REQUIRE_MANUSCRIPT_OUTPUTS", False):
        raise SystemExit(
            "Refusing to update Model 2B manuscript-default outputs from an incomplete run: "
            f"{completeness_status}."
        )


def _contact_mode_suffix() -> str:
    raw = os.environ.get("MODEL2B_CONTACT_MODE", "full").strip().lower().replace("-", "_")
    return "full" if raw == "full" else "rank_one"


def _factor_summary_status(path: Path, outcome: str) -> str:
    if not path.exists():
        return "missing_" + path.name
    frame = pd.read_csv(path)
    required = {"iso3", "factor_id", "outcome_target", "n_posterior_draws", "model2b_output_version"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        return f"{path.name}_missing_columns_" + ",".join(missing_columns)
    versions = set(frame["model2b_output_version"].dropna().astype(str))
    if versions != {MODEL2B_OUTPUT_VERSION}:
        return f"{path.name}_output_version_mismatch"
    true = frame.loc[frame["outcome_target"].eq(outcome)].copy()
    expected_iso = set(MODEL2B_DEFAULT_PILOT_ISO3)
    expected_factors = set(MODEL2B_PRIMARY_MECHANISMS)
    observed_iso = set(true["iso3"].dropna().astype(str))
    missing_iso = sorted(expected_iso.difference(observed_iso))
    if missing_iso:
        return f"{path.name}_missing_countries_" + ",".join(missing_iso)
    expected_pairs = {(iso3, factor_id) for iso3 in expected_iso for factor_id in expected_factors}
    observed_pairs = {
        (str(row.iso3), str(row.factor_id))
        for row in true.loc[true["iso3"].isin(expected_iso)].itertuples(index=False)
    }
    missing_pairs = sorted(expected_pairs.difference(observed_pairs))
    if missing_pairs:
        preview = ",".join(f"{iso3}:{factor_id}" for iso3, factor_id in missing_pairs[:5])
        return f"{path.name}_missing_factor_rows_" + preview
    true["n_posterior_draws"] = pd.to_numeric(true["n_posterior_draws"], errors="coerce")
    min_draws = _manuscript_min_draws()
    too_few = sorted(
        true.loc[true["n_posterior_draws"].lt(min_draws), "iso3"].dropna().astype(str).unique().tolist()
    )
    if too_few:
        return f"{path.name}_posterior_draws_below_{min_draws}_" + ",".join(too_few)
    return "ok"


def _write_contact_sensitivity_summary() -> None:
    columns = [
        "iso3",
        "country",
        "factor_id",
        "factor_label",
        "rank_one_share_percent",
        "full_contact_share_percent",
        "share_difference_full_minus_rank_one",
        "rank_one_rank",
        "full_contact_rank",
        "rank_change",
        "mechanism_rank_stable",
        "share_stability_band",
        "sensitivity_status",
        "model2b_output_version",
    ]
    rank_path = OUTPUTS / "model2b_bayes_factor_weight_summary_rank_one.csv"
    full_path = OUTPUTS / "model2b_bayes_factor_weight_summary_full.csv"
    output_path = OUTPUTS / "model2b_full_contact_sensitivity_summary.csv"
    missing = [path.name for path in (rank_path, full_path) if not path.exists()]
    if missing:
        pd.DataFrame(
            [
                {
                    "iso3": "ALL",
                    "country": "All countries",
                    "factor_id": "all_primary_mechanisms",
                    "factor_label": "All primary mechanisms",
                    "sensitivity_status": "missing_" + ";".join(missing),
                    "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                }
            ],
            columns=columns,
        ).to_csv(output_path, index=False)
        return

    outcome = MODEL2B_OUTCOMES["true_infections"]
    rank_status = _factor_summary_status(rank_path, outcome)
    full_status = _factor_summary_status(full_path, outcome)
    invalid = [status for status in (rank_status, full_status) if status != "ok"]
    if invalid:
        pd.DataFrame(
            [
                {
                    "iso3": "ALL",
                    "country": "All countries",
                    "factor_id": "all_primary_mechanisms",
                    "factor_label": "All primary mechanisms",
                    "sensitivity_status": "invalid_" + ";".join(invalid),
                    "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                }
            ],
            columns=columns,
        ).to_csv(output_path, index=False)
        return

    rank = pd.read_csv(rank_path)
    full = pd.read_csv(full_path)
    rank = rank.loc[rank["outcome_target"].eq(outcome)].copy()
    full = full.loc[full["outcome_target"].eq(outcome)].copy()
    for frame in (rank, full):
        frame["share_percent_median"] = pd.to_numeric(frame["share_percent_median"], errors="coerce")
        frame["positive_rank_basis"] = frame["share_percent_median"].where(frame["share_percent_median"].gt(0), float("-inf"))
        frame["mechanism_rank"] = (
            frame.sort_values(
                ["iso3", "positive_rank_basis", "share_percent_median", "factor_id"],
                ascending=[True, False, False, True],
                kind="stable",
            )
            .groupby("iso3")
            .cumcount()
            + 1
        )

    merged = rank.merge(full, on=["iso3", "factor_id"], how="outer", suffixes=("_rank_one", "_full"))
    rows = []
    for row in merged.itertuples(index=False):
        rank_share = getattr(row, "share_percent_median_rank_one", float("nan"))
        full_share = getattr(row, "share_percent_median_full", float("nan"))
        share_diff = full_share - rank_share if pd.notna(rank_share) and pd.notna(full_share) else float("nan")
        rank_one_rank = getattr(row, "mechanism_rank_rank_one", float("nan"))
        full_contact_rank = getattr(row, "mechanism_rank_full", float("nan"))
        rank_change = full_contact_rank - rank_one_rank if pd.notna(rank_one_rank) and pd.notna(full_contact_rank) else float("nan")
        rows.append(
            {
                "iso3": getattr(row, "iso3"),
                "country": getattr(row, "country_rank_one", None) or getattr(row, "country_full", None),
                "factor_id": getattr(row, "factor_id"),
                "factor_label": getattr(row, "factor_label_rank_one", None) or getattr(row, "factor_label_full", None),
                "rank_one_share_percent": rank_share,
                "full_contact_share_percent": full_share,
                "share_difference_full_minus_rank_one": share_diff,
                "rank_one_rank": rank_one_rank,
                "full_contact_rank": full_contact_rank,
                "rank_change": rank_change,
                "mechanism_rank_stable": bool(rank_change == 0) if pd.notna(rank_change) else False,
                "share_stability_band": "stable_within_5pp" if pd.notna(share_diff) and abs(share_diff) <= 5.0 else "shift_gt_5pp",
                "sensitivity_status": "compared_rank_one_vs_full",
                "model2b_output_version": MODEL2B_OUTPUT_VERSION,
            }
        )
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def prepare_model2b_data(countries: list[str]) -> pd.DataFrame:
    ensure_directories()
    inputs = load_inputs()
    rows = []
    for iso3 in countries:
        _path, _data, metadata, _params = _prepare_country(inputs, iso3)
        rows.append(metadata)
    metadata = pd.DataFrame(rows)
    contact_mode = _contact_mode_suffix()
    metadata.to_csv(OUTPUTS / f"model2b_stan_data_metadata_{contact_mode}.csv", index=False)
    if contact_mode == "full":
        metadata.to_csv(OUTPUTS / "model2b_stan_data_metadata.csv", index=False)
    selection = country_selection_audit(inputs, countries)
    selection = selection.rename(
        columns={
            "selected_for_current_model2": "selected_for_current_model2b",
            "model2_selection_reason": "model2b_selection_reason",
        }
    )
    selection["selected_for_current_model2"] = selection["selected_for_current_model2b"]
    selection["model2_selection_reason"] = selection["model2b_selection_reason"]
    selection["model2b_selection_reason"] = selection["model2b_selection_reason"].astype(str).str.replace(
        "Model 2", "Bayesian Model 2B", regex=False
    )
    selection["model2_selection_reason"] = selection["model2b_selection_reason"]
    selection.to_csv(OUTPUTS / "model2b_country_selection_audit.csv", index=False)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian Model 2B transmission attribution.")
    parser.add_argument("--countries", nargs="*", help="ISO3 countries to fit. Defaults to dynamic-pilot selection.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write CmdStan JSON data files.")
    args = parser.parse_args(argv)

    ensure_directories()
    inputs = load_inputs()
    countries = args.countries or _env_countries() or select_pilot_countries(inputs, limit=_env_int("MODEL2B_COUNTRY_LIMIT", 9))
    if args.prepare_only:
        prepare_model2b_data(countries)
        print(f"Wrote Model 2B Stan data for {len(countries)} countries.")
        return

    CmdStanModel = _require_cmdstanpy()
    chains = _env_int("MODEL2B_CHAINS", 4)
    warmup = _env_int("MODEL2B_WARMUP", 500, minimum=0)
    samples = _env_int("MODEL2B_SAMPLES", 500)
    adapt_delta = _env_float("MODEL2B_ADAPT_DELTA", 0.95)
    max_treedepth = _env_int("MODEL2B_MAX_TREEDEPTH", 12)
    seed = _env_int("MODEL2B_SEED", 20260611)
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
                    f"Model 2B completed {iso3} ({completed}/{len(countries)}) "
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
                max_treedepth=max_treedepth,
                model=model,
            )
            for index, iso3 in enumerate(countries)
        ]
    _write_outputs(tables)
    selection = country_selection_audit(inputs, countries)
    selection = selection.rename(
        columns={
            "selected_for_current_model2": "selected_for_current_model2b",
            "model2_selection_reason": "model2b_selection_reason",
        }
    )
    selection["selected_for_current_model2"] = selection["selected_for_current_model2b"]
    selection["model2b_selection_reason"] = selection["model2b_selection_reason"].astype(str).str.replace(
        "Model 2", "Bayesian Model 2B", regex=False
    )
    selection["model2_selection_reason"] = selection["model2b_selection_reason"]
    selection.to_csv(OUTPUTS / "model2b_country_selection_audit.csv", index=False)
    print(f"Wrote Model 2B Bayesian attribution outputs for {len(countries)} countries.")
