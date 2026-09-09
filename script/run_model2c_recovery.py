#!/usr/bin/env python3
"""Prepare, fit, and summarise isolated Model 2C recovery experiments.

No sampling occurs unless ``--fit`` is supplied.  Every artifact is written
below ``outputs/model2c_recovery/<tag>/`` and cannot replace Model 2C main or
sensitivity outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.parallel import exclusive_run_lock, limit_native_threads, process_budget

if __name__ in {"__main__", "__mp_main__"}:
    limit_native_threads(1)

import numpy as np
import pandas as pd

from pertussis_global_dynamics.bayes.model2c_recovery import (  # noqa: E402
    RECOVERY_PARAMETERS,
    apply_recovery_scenario,
    assert_isolated_recovery_dir,
    recovery_aggregate,
    recovery_scenarios,
    safe_recovery_tag,
    summarise_recovery_run,
    synthetic_stan_data,
)


OUTPUTS = ROOT / "outputs"
STAN_FILE = ROOT / "script" / "stan" / "model2c_transmission_scenario_replay.stan"
DEFAULT_COUNTRIES = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
FEASIBILITY_REPORT_DIR = ROOT / "manuscript" / "reports"
REQUIRED_PREFLIGHT_VALUE_LEVEL = "high_for_boundary_decision_not_primary_claim"
REQUIRED_LONG_PILOT_CHAINS = 4
MAX_LONG_PILOT_ACTIVE_STAN_CHAINS = 4
PARAMETER_SUMMARY_COLUMNS = (
    "run_id", "iso3", "scenario_id", "replicate", "parameter", "truth", "posterior_mean",
    "posterior_median", "lower_95", "upper_95", "covered_95", "bias", "absolute_error",
    "truth_direction", "posterior_direction", "direction_correct",
)
MECHANISM_SUMMARY_COLUMNS = (
    "run_id", "iso3", "scenario_id", "replicate", "mechanism", "parameter", "truth",
    "rope_half_width", "truth_active", "posterior_active_probability",
    "posterior_inactive_probability", "classification", "discrimination_correct",
)
AGGREGATE_COLUMNS = (
    "metric_family", "estimand", "n_runs", "coverage_95", "mean_bias", "rmse",
    "direction_accuracy", "discrimination_accuracy",
)


def _fit_requires_feasibility_preflight(args: argparse.Namespace) -> bool:
    """Return whether a recovery fit is computationally/interpretively consequential.

    Four-chain recovery fits are used for the registered synthetic-recovery
    diagnostic gate, where between-chain diagnostics can affect manuscript
    scope.  They therefore require a documented feasibility and scientific-value
    preflight even when the caller forgets the explicit guard flag.
    """
    return bool(getattr(args, "fit", False)) and int(getattr(args, "chains", 0) or 0) >= REQUIRED_LONG_PILOT_CHAINS


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _run_id(iso3: str, scenario_id: str, replicate: int) -> str:
    return f"{iso3}_{scenario_id}_r{replicate:03d}"


def _base_data_path(iso3: str, suffix: str) -> Path:
    return OUTPUTS / f"model2c_stan_data_{iso3}_{suffix}.json"


def _write_manifest_atomic(manifest: pd.DataFrame, path: Path) -> None:
    """Replace the manifest atomically; only the parent process calls this."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(path)


def _fit_one_run(payload: dict[str, Any]) -> dict[str, str]:
    """Fit one independent synthetic dataset and return a parent-update message."""
    if bool(payload.get("chainwise", False)):
        return _fit_one_run_chainwise(payload)

    from cmdstanpy import CmdStanModel

    row = payload["row"]
    model = CmdStanModel(exe_file=str(payload["exe_file"]))
    fit = model.sample(
        data=str(ROOT / str(row["data_path"])),
        chains=int(payload["chains"]),
        parallel_chains=min(int(payload["parallel_chains"]), int(payload["chains"])),
        iter_warmup=int(payload["warmup"]),
        iter_sampling=int(payload["samples"]),
        seed=int(row["fit_seed"]),
        adapt_delta=float(payload["adapt_delta"]),
        max_treedepth=int(payload["max_treedepth"]),
        show_progress=False,
    )
    _write_fit_products(row, fit, int(payload["max_treedepth"]), completion_details={"mode": "joint"})
    return {"run_id": str(row["run_id"]), "status": "fit_complete", "error": ""}


def _write_fit_products(
    row: dict[str, Any],
    fit: Any,
    max_treedepth: int,
    *,
    completion_details: dict[str, Any],
) -> None:
    """Persist posterior draws, summaries, HMC diagnostics and completion marker."""
    draws_path = ROOT / str(row["draws_path"])
    summary_path = ROOT / str(row["fit_summary_path"])
    raw = fit.draws_pd(vars=list(RECOVERY_PARAMETERS))
    raw.insert(0, "run_id", row["run_id"])
    raw.to_csv(draws_path, index=False)
    fit.summary().to_csv(summary_path)
    method = fit.method_variables()
    energy = np.asarray(method["energy__"], dtype=float)
    divergent = np.asarray(method["divergent__"])
    treedepth = np.asarray(method["treedepth__"])
    if energy.ndim == 1:
        energy = energy[:, None]
        divergent = divergent[:, None]
        treedepth = treedepth[:, None]

    def ebfmi(chain_energy: np.ndarray) -> float:
        variance = float(np.var(chain_energy, ddof=1))
        if chain_energy.size < 2 or not np.isfinite(chain_energy).all() or variance <= 0:
            return math.nan
        return float(np.mean(np.diff(chain_energy) ** 2) / variance)

    hmc_path = ROOT / str(row["hmc_diagnostics_path"])
    hmc_rows = []
    for chain_index in range(energy.shape[1]):
        hmc_rows.append({
            "run_id": str(row["run_id"]),
            "chain": chain_index + 1,
            "post_warmup_draws": int(energy.shape[0]),
            "divergent_transitions": int(np.sum(divergent[:, chain_index])),
            "treedepth_saturations": int(np.sum(treedepth[:, chain_index] >= int(max_treedepth))),
            "minimum_ebfmi": ebfmi(energy[:, chain_index]),
            "max_treedepth": int(max_treedepth),
        })
    pd.DataFrame(hmc_rows).to_csv(hmc_path, index=False)
    marker_path = ROOT / str(row["completion_marker_path"])
    marker_temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    marker_payload = {"run_id": str(row["run_id"]), "status": "fit_complete", **completion_details}
    marker_temporary.write_text(
        json.dumps(_json_ready(marker_payload)) + "\n",
        encoding="utf-8",
    )
    marker_temporary.replace(marker_path)


def _fit_one_chain_csv(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one chain for a synthetic recovery fit, with optional retry."""
    from cmdstanpy import CmdStanModel

    row = payload["row"]
    chain = int(payload["chain"])
    run_id = str(row["run_id"])
    retries = int(payload.get("chain_retries", 0))
    timeout_seconds = float(payload.get("chain_timeout_seconds", 0) or 0)
    timeout = timeout_seconds if timeout_seconds > 0 else None
    chain_dir = ROOT / str(payload["chain_dir"])
    errors: list[str] = []
    model = CmdStanModel(exe_file=str(payload["exe_file"]))
    for attempt in range(retries + 1):
        seed = int(row["fit_seed"]) + chain * 10_009 + attempt * 1_000_003
        attempt_dir = chain_dir / f"chain_{chain:02d}" / f"attempt_{attempt + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            fit = model.sample(
                data=str(ROOT / str(row["data_path"])),
                chains=1,
                parallel_chains=1,
                chain_ids=[chain],
                iter_warmup=int(payload["warmup"]),
                iter_sampling=int(payload["samples"]),
                seed=seed,
                adapt_delta=float(payload["adapt_delta"]),
                max_treedepth=int(payload["max_treedepth"]),
                output_dir=str(attempt_dir),
                show_progress=False,
                timeout=timeout,
            )
            csv_files = list(fit.runset.csv_files)
            if len(csv_files) != 1:
                raise RuntimeError(f"expected one chain CSV for {run_id} chain {chain}, found {len(csv_files)}")
            return {
                "run_id": run_id,
                "chain": chain,
                "attempt": attempt + 1,
                "csv_path": str(Path(csv_files[0]).resolve()),
                "seed": seed,
            }
        except Exception as exc:  # pragma: no cover - exercised by integration runs
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"{run_id} chain {chain} failed after {retries + 1} attempt(s): {' | '.join(errors)}")


def _fit_one_run_chainwise(payload: dict[str, Any]) -> dict[str, str]:
    """Fit one synthetic dataset as independently retriable single-chain jobs."""
    from cmdstanpy import from_csv

    row = payload["row"]
    run_id = str(row["run_id"])
    chain_dir = ROOT / str(payload["chain_dir"])
    chain_dir.mkdir(parents=True, exist_ok=True)
    chain_payloads = []
    for chain in range(1, int(payload["chains"]) + 1):
        chain_payload = dict(payload)
        chain_payload["chain"] = chain
        chain_payloads.append(chain_payload)

    max_workers = max(1, min(int(payload["parallel_chains"]), int(payload["chains"])))
    if max_workers == 1:
        chain_results = [_fit_one_chain_csv(chain_payload) for chain_payload in chain_payloads]
    else:
        chain_results = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_fit_one_chain_csv, chain_payload): chain_payload for chain_payload in chain_payloads}
            for future in as_completed(futures):
                chain_results.append(future.result())

    chain_results = sorted(chain_results, key=lambda item: int(item["chain"]))
    csv_paths = [item["csv_path"] for item in chain_results]
    fit = from_csv(path=csv_paths, method="sample")
    if fit is None:
        raise RuntimeError(f"failed to load chainwise CSVs for {run_id}")
    _write_fit_products(
        row,
        fit,
        int(payload["max_treedepth"]),
        completion_details={
            "mode": "chainwise",
            "chain_attempts": [{"chain": item["chain"], "attempt": item["attempt"]} for item in chain_results],
        },
    )
    return {"run_id": run_id, "status": "fit_complete", "error": ""}


def _execute_fit_tasks(
    payloads: Iterable[dict[str, Any]],
    *,
    run_workers: int,
    worker: Callable[[dict[str, Any]], dict[str, str]] = _fit_one_run,
) -> Iterable[tuple[dict[str, Any], dict[str, str] | None, Exception | None]]:
    """Yield task outcomes sequentially or across independent worker processes."""
    tasks = list(payloads)
    if run_workers <= 1:
        for payload in tasks:
            try:
                yield payload, worker(payload), None
            except Exception as exc:  # pragma: no cover - exercised through the same parent handler
                yield payload, None, exc
        return
    with ProcessPoolExecutor(max_workers=run_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(worker, payload): payload for payload in tasks}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                yield payload, future.result(), None
            except Exception as exc:
                yield payload, None, exc


def _completion_paths(row: pd.Series | dict[str, Any]) -> list[Path]:
    return [
        ROOT / str(row["draws_path"]),
        ROOT / str(row["fit_summary_path"]),
        ROOT / str(row["completion_marker_path"]),
        ROOT / str(row["hmc_diagnostics_path"]),
    ]


def _mark_incomplete_running_fits_interrupted(manifest: pd.DataFrame, reason: str) -> pd.DataFrame:
    updated = manifest.copy()
    for index, row in updated.iterrows():
        if str(row.get("status", "")) != "fit_running":
            continue
        if all(path.exists() for path in _completion_paths(row)):
            updated.loc[index, "status"] = "fit_complete"
            updated.loc[index, "error"] = ""
        else:
            updated.loc[index, "status"] = "fit_interrupted"
            updated.loc[index, "error"] = reason
    return updated


def _enforce_feasibility_preflight(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "require_feasibility_preflight", False)) and not _fit_requires_feasibility_preflight(args):
        return
    preflight_path = FEASIBILITY_REPORT_DIR / f"model2c_recovery_feasibility_preflight_{args.tag}.csv"
    if not preflight_path.exists():
        raise SystemExit(
            "Long-HMC recovery fits require a feasibility preflight. "
            f"Missing {_display_path(preflight_path)}."
        )
    audit = pd.read_csv(preflight_path)
    if audit.empty:
        raise SystemExit(f"Feasibility preflight is empty: {_display_path(preflight_path)}")
    record = audit.iloc[-1]
    action = str(record.get("recommended_next_action", ""))
    allowed_actions = {
        "continue_bounded_chainwise_pilot_fits",
        "resolve_active_or_interrupted_fit_with_chainwise_retry_before_starting_new_fit",
    }
    if action not in allowed_actions:
        raise SystemExit(
            "Feasibility preflight does not authorize additional long-HMC pilot fitting: "
            f"{action or 'missing recommended_next_action'}."
        )
    value_level = str(record.get("scientific_value_level", ""))
    if value_level != REQUIRED_PREFLIGHT_VALUE_LEVEL:
        raise SystemExit(
            "Feasibility preflight did not document sufficient scientific value for long-HMC recovery fitting: "
            f"{value_level or 'missing scientific_value_level'}."
        )
    if str(record.get("pilot_fit_authorized", "")).lower() not in {"true", "1", "yes"}:
        raise SystemExit("Feasibility preflight does not authorize a bounded pilot fit.")
    authorized_chains = int(record.get("authorized_chain_count", REQUIRED_LONG_PILOT_CHAINS) or REQUIRED_LONG_PILOT_CHAINS)
    authorized_active = int(
        record.get("authorized_max_active_stan_chains", MAX_LONG_PILOT_ACTIVE_STAN_CHAINS)
        or MAX_LONG_PILOT_ACTIVE_STAN_CHAINS
    )
    authorized_workers = int(record.get("authorized_run_workers", 1) or 1)
    authorized_fits = int(record.get("authorized_max_fits_per_invocation", 1) or 1)
    if args.run_workers == 0:
        current_budget = process_budget(
            0, tasks=max(1, args.max_fits or 1), slots_per_worker=args.parallel_chains,
            memory_gib_per_worker=float(record.get("memory_gib_per_chain", 0.5)) * args.parallel_chains,
        )
        args.run_workers = min(current_budget.workers, authorized_workers)
    if args.chains != authorized_chains:
        raise SystemExit(
            "Feasibility preflight authorizes exactly "
            f"{authorized_chains} chains for the long-HMC recovery diagnostic gate; received {args.chains}."
        )
    if args.run_workers * args.parallel_chains > authorized_active:
        raise SystemExit(
            "Feasibility preflight limits active Stan chains to "
            f"{authorized_active}; received run-workers × parallel-chains = "
            f"{args.run_workers * args.parallel_chains}."
        )
    if not args.chainwise:
        raise SystemExit("Feasibility preflight requires --chainwise so slow chains can be bounded and retried.")
    if args.max_fits is None or args.max_fits > authorized_fits:
        if authorized_fits == 1:
            raise SystemExit("Feasibility preflight allows only one long-HMC recovery fit per invocation.")
        raise SystemExit(f"Feasibility preflight limits this pilot batch to {authorized_fits} fits.")
    if args.run_workers > authorized_workers:
        raise SystemExit(f"Feasibility preflight limits independent recovery workers to {authorized_workers}.")
    if authorized_workers > 1 and args.chain_timeout_seconds <= 0:
        raise SystemExit("Parallel pilot fits require a positive per-chain timeout.")
    if int(record.get("fit_running", 0) or 0) > 0 and action.startswith("continue_"):
        raise SystemExit("Feasibility preflight saw active fits; refresh or resolve them before starting another fit.")


def prepare_experiment(args: argparse.Namespace, experiment_dir: Path) -> pd.DataFrame:
    draws_path = Path(args.posterior_draws)
    if not draws_path.is_absolute():
        draws_path = ROOT / draws_path
    draws = pd.read_csv(draws_path)
    required_draw_columns = {"iso3", *RECOVERY_PARAMETERS}
    missing = sorted(required_draw_columns.difference(draws.columns))
    if missing:
        raise SystemExit(f"Posterior draw file is missing recovery columns: {', '.join(missing)}")

    data_dir = experiment_dir / "stan_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    truth_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    expected_rows: list[dict[str, Any]] = []
    scenarios = recovery_scenarios()

    for country_number, iso3 in enumerate(args.countries):
        country_draws = draws.loc[draws["iso3"].astype(str).eq(iso3)].reset_index(drop=True)
        if country_draws.empty:
            raise SystemExit(f"No posterior draws found for {iso3} in {draws_path}.")
        base_path = _base_data_path(iso3, args.base_data_suffix)
        if not base_path.exists():
            raise SystemExit(f"Missing prepared Model 2C base data: {base_path}")
        base_data = json.loads(base_path.read_text(encoding="utf-8"))

        for replicate in range(1, args.replicates + 1):
            draw_rng = np.random.default_rng(args.seed + country_number * 100_003 + replicate * 1_009)
            source_position = int(draw_rng.integers(0, len(country_draws)))
            source_draw = country_draws.iloc[source_position].copy()
            for scenario_number, scenario in enumerate(scenarios):
                run_id = _run_id(iso3, scenario.scenario_id, replicate)
                truth = apply_recovery_scenario(source_draw, scenario)
                simulation_seed = args.seed + country_number * 1_000_003 + replicate * 10_007 + scenario_number * 101
                synthetic, expected = synthetic_stan_data(base_data, truth, seed=simulation_seed)
                data_path = data_dir / f"{run_id}.json"
                data_path.write_text(json.dumps(_json_ready(synthetic), indent=2), encoding="utf-8")

                truth_row = {
                    "run_id": run_id,
                    "tag": args.tag,
                    "iso3": iso3,
                    "scenario_id": scenario.scenario_id,
                    "scenario_class": scenario.scenario_class,
                    "replicate": replicate,
                    "source_draw": int(source_draw.get("draw", source_position + 1)),
                    "simulation_seed": simulation_seed,
                    "truth_reporting_shift_enabled": scenario.reporting_shift_enabled,
                    "truth_transmission_context_enabled": scenario.transmission_context_enabled,
                    **{name: float(truth[name]) for name in RECOVERY_PARAMETERS},
                }
                truth_rows.append(truth_row)
                expected_rows.append(
                    {
                        "run_id": run_id,
                        "month_fit_expected_total": float(np.sum(expected["month_fit_mean"])),
                        "month_ppc_expected_total": float(np.sum(expected["month_ppc_mean"])),
                        "annual_age_expected_total": float(np.sum(expected["age_mean"])),
                        "synthetic_month_fit_total": int(np.sum(synthetic["y_month"])),
                        "synthetic_month_ppc_total": int(np.sum(synthetic["y_month_ppc"])),
                        "synthetic_age_total": int(np.sum(synthetic["y_age"])),
                    }
                )
                manifest_rows.append(
                    {
                        "run_id": run_id,
                        "tag": args.tag,
                        "iso3": iso3,
                        "scenario_id": scenario.scenario_id,
                        "scenario_class": scenario.scenario_class,
                        "replicate": replicate,
                        "status": "prepared_not_fitted",
                        "data_path": str(data_path.relative_to(ROOT)),
                        "draws_path": str((experiment_dir / "fit_draws" / f"{run_id}.csv").relative_to(ROOT)),
                        "fit_summary_path": str((experiment_dir / "fit_summaries" / f"{run_id}.csv").relative_to(ROOT)),
                        "completion_marker_path": str(
                            (experiment_dir / "fit_summaries" / f"{run_id}.complete.json").relative_to(ROOT)
                        ),
                        "hmc_diagnostics_path": str(
                            (experiment_dir / "fit_summaries" / f"{run_id}.hmc.csv").relative_to(ROOT)
                        ),
                        "simulation_seed": simulation_seed,
                        "fit_seed": args.seed + country_number * 100_019 + replicate * 1_021 + scenario_number,
                        "description": scenario.description,
                    }
                )

    truth = pd.DataFrame(truth_rows)
    manifest = pd.DataFrame(manifest_rows)
    truth.to_csv(experiment_dir / "truth.csv", index=False)
    pd.DataFrame(expected_rows).to_csv(experiment_dir / "synthetic_observation_totals.csv", index=False)
    _write_manifest_atomic(manifest, experiment_dir / "manifest.csv")
    write_report(experiment_dir, manifest, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    return manifest


def prepare_from_source_experiment(args: argparse.Namespace, experiment_dir: Path) -> pd.DataFrame:
    """Reuse prespecified synthetic data from a frozen recovery experiment."""
    source_dir = Path(args.source_experiment)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    source_manifest = pd.read_csv(source_dir / "manifest.csv")
    source_truth = pd.read_csv(source_dir / "truth.csv")
    source_totals = pd.read_csv(source_dir / "synthetic_observation_totals.csv")
    selected = source_manifest.loc[
        source_manifest["iso3"].astype(str).isin(args.countries)
        & pd.to_numeric(source_manifest["replicate"]).le(args.replicates)
    ].copy()
    expected = len(args.countries) * args.replicates * 4
    if len(selected) != expected:
        raise SystemExit(f"Expected {expected} frozen source runs but found {len(selected)}.")
    selected["tag"] = args.tag
    selected["status"] = "prepared_not_fitted"
    selected["error"] = ""
    selected["draws_path"] = selected["run_id"].map(
        lambda value: str((experiment_dir / "fit_draws" / f"{value}.csv").relative_to(ROOT))
    )
    selected["fit_summary_path"] = selected["run_id"].map(
        lambda value: str((experiment_dir / "fit_summaries" / f"{value}.csv").relative_to(ROOT))
    )
    selected["completion_marker_path"] = selected["run_id"].map(
        lambda value: str((experiment_dir / "fit_summaries" / f"{value}.complete.json").relative_to(ROOT))
    )
    selected["hmc_diagnostics_path"] = selected["run_id"].map(
        lambda value: str((experiment_dir / "fit_summaries" / f"{value}.hmc.csv").relative_to(ROOT))
    )
    run_ids = set(selected["run_id"].astype(str))
    truth = source_truth.loc[source_truth["run_id"].astype(str).isin(run_ids)].copy()
    truth["tag"] = args.tag
    totals = source_totals.loc[source_totals["run_id"].astype(str).isin(run_ids)].copy()
    truth.to_csv(experiment_dir / "truth.csv", index=False)
    totals.to_csv(experiment_dir / "synthetic_observation_totals.csv", index=False)
    _write_manifest_atomic(selected, experiment_dir / "manifest.csv")
    write_report(experiment_dir, selected, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    return selected


def fit_experiment(args: argparse.Namespace, experiment_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    from cmdstanpy import CmdStanModel

    fit_draws_dir = experiment_dir / "fit_draws"
    fit_summaries_dir = experiment_dir / "fit_summaries"
    fit_draws_dir.mkdir(parents=True, exist_ok=True)
    fit_summaries_dir.mkdir(parents=True, exist_ok=True)
    model = CmdStanModel(stan_file=str(STAN_FILE))
    model_exe_file = str(model.exe_file)
    budget = process_budget(
        args.run_workers, tasks=max(1, args.max_fits or len(manifest)),
        slots_per_worker=min(args.parallel_chains, args.chains),
        memory_gib_per_worker=args.memory_gib_per_chain * min(args.parallel_chains, args.chains),
    )
    args.run_workers = budget.workers
    (experiment_dir / "execution_plan.json").write_text(json.dumps({
        "parent_pid": os.getpid(),
        "budget": budget.to_dict(), "chains_per_fit": args.chains,
        "parallel_chains_per_fit": args.parallel_chains, "warmup": args.warmup,
        "samples": args.samples, "max_fits": args.max_fits,
        "chain_timeout_seconds": args.chain_timeout_seconds,
        "chain_retries": args.chain_retries, "native_threads": 1,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Recovery budget: {args.run_workers} fits × {args.parallel_chains} chains × 1 thread; "
          f"at most {args.max_fits or len(manifest)} new fits", flush=True)
    updated = manifest.copy()
    payloads: list[dict[str, Any]] = []
    for index, row in updated.iterrows():
        draws_path, fit_summary_path, completion_marker_path, hmc_diagnostics_path = _completion_paths(row)
        if all(path.exists() for path in (draws_path, fit_summary_path, completion_marker_path, hmc_diagnostics_path)) and not args.force:
            updated.loc[index, "status"] = "fit_complete"
            updated.loc[index, "error"] = ""
            continue
        if args.force and completion_marker_path.exists():
            completion_marker_path.unlink()
        if args.max_fits is not None and len(payloads) >= args.max_fits:
            updated.loc[index, "status"] = "fit_pending"
            updated.loc[index, "error"] = f"deferred by --max-fits={args.max_fits}"
            continue
        updated.loc[index, "status"] = "fit_running"
        updated.loc[index, "error"] = ""
        payloads.append(
            {
                "row": row.to_dict(),
                "exe_file": model_exe_file,
                "chains": args.chains,
                "parallel_chains": args.parallel_chains,
                "warmup": args.warmup,
                "samples": args.samples,
                "adapt_delta": args.adapt_delta,
                "max_treedepth": args.max_treedepth,
                "chainwise": args.chainwise,
                "chain_timeout_seconds": args.chain_timeout_seconds,
                "chain_retries": args.chain_retries,
                "chain_dir": str((fit_summaries_dir / "chain_csv" / str(row["run_id"])).relative_to(ROOT)),
            }
        )
    manifest_path = experiment_dir / "manifest.csv"
    _write_manifest_atomic(updated, manifest_path)
    index_by_run = {str(value): index for index, value in updated["run_id"].items()}
    try:
        for payload, result, error in _execute_fit_tasks(payloads, run_workers=args.run_workers):
            run_id = str(payload["row"]["run_id"])
            index = index_by_run[run_id]
            if error is None and result is not None:
                updated.loc[index, "status"] = result["status"]
                updated.loc[index, "error"] = result.get("error", "")
                print(f"Completed recovery fit {run_id}", flush=True)
            else:
                updated.loc[index, "status"] = "fit_failed"
                updated.loc[index, "error"] = f"{type(error).__name__}: {error}"[:1000]
                print(f"Recovery fit failed {run_id}: {error}", file=sys.stderr, flush=True)
            _write_manifest_atomic(updated, manifest_path)
    except KeyboardInterrupt:
        updated = _mark_incomplete_running_fits_interrupted(updated, "interrupted before completion")
        _write_manifest_atomic(updated, manifest_path)
        raise
    return updated


def refresh_manifest_fit_status(manifest: pd.DataFrame) -> pd.DataFrame:
    refreshed = manifest.copy()
    for index, row in refreshed.iterrows():
        if all(path.exists() for path in _completion_paths(row)):
            refreshed.loc[index, "status"] = "fit_complete"
            refreshed.loc[index, "error"] = ""
        elif str(row.get("status", "")) == "fit_running":
            refreshed.loc[index, "status"] = "fit_interrupted"
            refreshed.loc[index, "error"] = "missing completion artifacts after status refresh"
    return refreshed


def summarise_experiment(experiment_dir: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = refresh_manifest_fit_status(manifest)
    _write_manifest_atomic(manifest, experiment_dir / "manifest.csv")
    truth = pd.read_csv(experiment_dir / "truth.csv").set_index("run_id", drop=False)
    parameter_parts: list[pd.DataFrame] = []
    mechanism_parts: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        draws_path = ROOT / str(row.draws_path)
        if not draws_path.exists():
            continue
        posterior = pd.read_csv(draws_path)
        parameter, mechanism = summarise_recovery_run(
            truth.loc[row.run_id],
            posterior,
            run_id=row.run_id,
            iso3=row.iso3,
            scenario_id=row.scenario_id,
            replicate=int(row.replicate),
        )
        parameter_parts.append(parameter)
        mechanism_parts.append(mechanism)
    parameter_summary = (
        pd.concat(parameter_parts, ignore_index=True)
        if parameter_parts
        else pd.DataFrame(columns=PARAMETER_SUMMARY_COLUMNS)
    )
    mechanism_summary = (
        pd.concat(mechanism_parts, ignore_index=True)
        if mechanism_parts
        else pd.DataFrame(columns=MECHANISM_SUMMARY_COLUMNS)
    )
    aggregate = recovery_aggregate(parameter_summary, mechanism_summary)
    if aggregate.empty:
        aggregate = pd.DataFrame(columns=AGGREGATE_COLUMNS)
    parameter_summary.to_csv(experiment_dir / "parameter_recovery.csv", index=False)
    mechanism_summary.to_csv(experiment_dir / "mechanism_discrimination.csv", index=False)
    aggregate.to_csv(experiment_dir / "aggregate_recovery.csv", index=False)
    write_report(experiment_dir, manifest, parameter_summary, mechanism_summary, aggregate)
    return parameter_summary, mechanism_summary, aggregate


def write_report(
    experiment_dir: Path,
    manifest: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    completed = int(manifest["status"].eq("fit_complete").sum()) if "status" in manifest else 0
    lines = [
        "# Model 2C simulation-based recovery",
        "",
        f"Tagged experiment directory: `{experiment_dir.relative_to(ROOT)}`.",
        f"Prepared runs: {len(manifest)}; completed refits: {completed}.",
        "",
        "## Protocol",
        "",
        "For each country and replicate, one fitted posterior draw supplies the truth. The same truth draw is replayed under four prespecified structures: both reporting and transmission mechanisms, reporting only, transmission only, and neither. Monthly all-age and annual age-group counts are independently generated with the model's NB2 observation distributions, then refitted with the unchanged Model 2C Stan likelihood.",
        "",
        "Primary diagnostics are 95% interval coverage, bias/RMSE, direction recovery for log-scale shift parameters, and active/inactive mechanism discrimination using a ±log(1.05) region of practical equivalence. Posterior mechanism activity probabilities ≥0.80 are classified active, ≤0.20 inactive, and intermediate values ambiguous.",
        "",
        "This is a recoverability diagnostic conditional on the observed covariate histories and posterior-draw truth distribution; it is not external validation and does not calibrate the composite likelihood learning rate.",
    ]
    if aggregate.empty:
        lines += ["", "## Status", "", "Synthetic inputs are prepared; sampling and empirical coverage calculations have not yet been run."]
    else:
        lines += ["", "## Aggregate results", "", _markdown_table(aggregate)]
    (experiment_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render Markdown without pandas' optional ``tabulate`` dependency."""
    if frame.empty:
        return "(none)"

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Model 2C parameter/mechanism recovery experiments.")
    parser.add_argument("--tag", default="posterior_truth_v1")
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--posterior-draws", default="outputs/model2c_posterior_draws.csv")
    parser.add_argument(
        "--source-experiment",
        default="",
        help="Reuse frozen synthetic JSON/truth rows from an existing recovery directory.",
    )
    parser.add_argument("--base-data-suffix", default="full")
    parser.add_argument("--prepare-only", action="store_true", help="Generate synthetic JSON and stop before Stan sampling.")
    parser.add_argument("--fit", action="store_true", help="Run the computationally expensive Stan refits.")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse the existing manifest and synthetic inputs; never regenerate completed work.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-fits",
        type=int,
        default=None,
        help="Run at most this many missing refits in the current invocation; completed fits are still skipped.",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--parallel-chains", type=int, default=4)
    parser.add_argument(
        "--run-workers",
        type=int,
        default=1,
        help="Independent synthetic refits; 0 selects a resource/preflight-bounded count. Active chains = workers × parallel-chains.",
    )
    parser.add_argument("--memory-gib-per-chain", type=float, default=0.5,
                        help="Planning estimate for concurrent chain memory, not an OS limit.")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--adapt-delta", type=float, default=0.95)
    parser.add_argument("--max-treedepth", type=int, default=12)
    parser.add_argument(
        "--chainwise",
        action="store_true",
        help=(
            "Run each chain as an independently retriable single-chain CmdStan job, then combine the "
            "successful chain CSVs for the same 4-chain diagnostic gate."
        ),
    )
    parser.add_argument(
        "--chain-timeout-seconds",
        type=float,
        default=0.0,
        help="Per-chain wall-clock timeout in chainwise mode; 0 disables the timeout.",
    )
    parser.add_argument(
        "--chain-retries",
        type=int,
        default=0,
        help="Additional attempts per chain in chainwise mode, using a new seed for each retry.",
    )
    parser.add_argument(
        "--require-feasibility-preflight",
        action="store_true",
        help=(
            "Require the tag-specific feasibility preflight report to authorize bounded long-HMC "
            "recovery fitting before any Stan job is started."
        ),
    )
    args = parser.parse_args(argv)
    args.tag = safe_recovery_tag(args.tag)
    args.countries = [str(country).upper() for country in args.countries]
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if args.run_workers < 0:
        parser.error("--run-workers must be nonnegative")
    if args.memory_gib_per_chain <= 0:
        parser.error("--memory-gib-per-chain must be positive")
    if args.chains < 1 or args.parallel_chains < 1 or args.parallel_chains > args.chains:
        parser.error("Require 1 <= parallel-chains <= chains")
    if args.max_fits is not None and args.max_fits < 1:
        parser.error("--max-fits must be at least 1")
    if args.chain_timeout_seconds < 0:
        parser.error("--chain-timeout-seconds must be non-negative")
    if args.chain_retries < 0:
        parser.error("--chain-retries must be non-negative")
    if not args.chainwise and (args.chain_timeout_seconds > 0 or args.chain_retries > 0):
        parser.error("--chain-timeout-seconds and --chain-retries require --chainwise")
    if args.prepare_only and args.fit:
        parser.error("--prepare-only and --fit are mutually exclusive")
    if args.require_feasibility_preflight and not args.fit:
        parser.error("--require-feasibility-preflight is only meaningful with --fit")
    if args.resume and (args.prepare_only or args.force):
        parser.error("--resume cannot regenerate prepared inputs or force-overwrite completed fits")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    limit_native_threads(1)
    experiment_dir = OUTPUTS / "model2c_recovery" / args.tag
    assert_isolated_recovery_dir(experiment_dir, OUTPUTS)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(experiment_dir / ".run.lock"):
        run_experiment(args, experiment_dir)


def run_experiment(args: argparse.Namespace, experiment_dir: Path) -> None:
    # Check the amended resource envelope before changing any scientific inputs.
    if args.fit:
        _enforce_feasibility_preflight(args)
    manifest_path = experiment_dir / "manifest.csv"
    if args.summarize_only or args.resume:
        if not manifest_path.exists():
            raise SystemExit(f"Missing recovery manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = (
            prepare_from_source_experiment(args, experiment_dir)
            if args.source_experiment
            else prepare_experiment(args, experiment_dir)
        )
    if args.fit:
        manifest = fit_experiment(args, experiment_dir, manifest)
    summarise_experiment(experiment_dir, manifest)
    print(f"Wrote Model 2C recovery experiment to {experiment_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
