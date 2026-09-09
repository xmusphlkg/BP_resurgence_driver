#!/usr/bin/env python3
"""Compare streams after partial median latent-log-contrast matching.

The primary matched-notification benchmark contains unequal native contrasts:
the infant severe-outcome contrast is larger and more consistently directed
than the native ascertainment-linked contrast.  This analysis rescales the
paired latent contrast of each auxiliary stream so that its median absolute
class difference relative to a reference Gaussian SD is the same for both
streams. For severe outcomes that SD describes log-rate drift ONLY: it excludes
negative-binomial count variance and log-count transformation effects. This is
not equal total signal-to-noise, pairwise effect, or direction matching.
Expected notifications remain exactly matched.

This is a conditional design sensitivity, not a re-fit of the transmission
model and not an estimate of historical mechanism shares.  Whole-country
holdout is reported alongside grouped CV because the former is the transport
estimand for a new country structure.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.parallel import exclusive_run_lock, limit_native_threads, process_budget

if __name__ in {"__main__", "__mp_main__"}:
    limit_native_threads(1)

import numpy as np
import pandas as pd

from pertussis_global_dynamics.bayes.model2c_surveillance_design import (  # noqa: E402
    MEASUREMENT_DESIGNS,
    SURVEILLANCE_BOOTSTRAP_REPLICATES,
    cross_validated_predictions,
    discrimination_metrics,
    feature_columns_for_package,
    leave_one_country_out_predictions,
    nested_country_draw_bootstrap_auc,
    simulate_observation_features,
)


INPUT_PATH = (
    ROOT
    / "outputs"
    / "model2c_surveillance_design"
    / "monitoring_packages_v4_source_preserved_20260908"
    / "latent_surveillance_signals.csv"
)
OUTPUT_PATH = ROOT / "outputs" / "nc_equal_contrast_comparison.csv"
TABLE_PATH = ROOT / "manuscript" / "tables" / "nc_equal_contrast_comparison.csv"
MARKDOWN_PATH = ROOT / "manuscript" / "tables_md" / "nc_equal_contrast_comparison.md"

POSITIVE_SCENARIO = "matched_transmission_truth"
NEGATIVE_SCENARIO = "matched_reporting_truth"
TARGET_STANDARDISED_CONTRASTS = (0.5, 1.0, 2.0, 4.0)
OBSERVATION_REPLICATES = 3
BASE_SEED = 20260909
STREAMS = {
    "testing": "testing",
    "severe": "severe",
}
INTERPRETATION_BOUNDARY = (
    "Equal pooled median absolute latent log contrast relative to a reference "
    "Gaussian error/log-rate-drift SD; severe-event count noise, transformation "
    "effects and contrast directions remain unequal. Not equal total SNR or "
    "historical attribution."
)


def matched_latent() -> pd.DataFrame:
    latent = pd.read_csv(INPUT_PATH)
    latent = latent.loc[
        latent["scenario_id"].isin([POSITIVE_SCENARIO, NEGATIVE_SCENARIO])
    ].copy()
    counts = latent.groupby("group_id")["scenario_id"].nunique()
    if len(counts) != 720 or not counts.eq(2).all():
        raise RuntimeError("Expected 720 paired country-draw groups")
    notification_columns = [
        column
        for column in latent.columns
        if column.startswith("latent_notification_mean_")
    ]
    indexed = latent.set_index(["group_id", "scenario_id"])
    differences = indexed[notification_columns].groupby(level=0).max() - indexed[
        notification_columns
    ].groupby(level=0).min()
    if not np.allclose(differences.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-9):
        raise RuntimeError("Expected notification means are not exactly matched")
    return latent.sort_values(["iso3", "draw_rank", "scenario_id"]).reset_index(drop=True)


def stream_deltas(latent: pd.DataFrame, stream: str) -> np.ndarray:
    indexed = latent.set_index(["group_id", "scenario_id"])
    positive = indexed.xs(POSITIVE_SCENARIO, level="scenario_id")
    negative = indexed.xs(NEGATIVE_SCENARIO, level="scenario_id")
    values: list[np.ndarray] = []
    for year in (2023, 2024, 2025):
        if stream == "testing":
            values.append(
                (
                    positive[f"latent_testing_log_index_{year}"]
                    - negative[f"latent_testing_log_index_{year}"]
                ).to_numpy(dtype=float)
            )
        elif stream == "severe":
            positive_log = np.log(
                positive[f"latent_severe_source_{year}"]
                / positive["latent_severe_source_baseline"]
            )
            negative_log = np.log(
                negative[f"latent_severe_source_{year}"]
                / negative["latent_severe_source_baseline"]
            )
            values.append((positive_log - negative_log).to_numpy(dtype=float))
        else:
            raise ValueError(f"Unknown stream: {stream}")
    return np.concatenate(values)


def rescale_stream_contrast(
    latent: pd.DataFrame,
    *,
    stream: str,
    target_standardised_contrast: float,
    measurement_sd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rescale median log contrast; measurement_sd is a reference SD, not total noise."""

    if target_standardised_contrast <= 0 or measurement_sd <= 0:
        raise ValueError("Target contrast and measurement SD must be positive")
    native_delta = stream_deltas(latent, stream)
    native_median = float(np.median(np.abs(native_delta)))
    if not np.isfinite(native_median) or native_median <= 0:
        raise RuntimeError(f"Native {stream} contrast has no usable variation")
    factor = target_standardised_contrast * measurement_sd / native_median

    rows: list[pd.Series] = []
    for _, pair in latent.groupby("group_id", sort=True):
        positive = pair.loc[pair["scenario_id"].eq(POSITIVE_SCENARIO)].iloc[0].copy()
        negative = pair.loc[pair["scenario_id"].eq(NEGATIVE_SCENARIO)].iloc[0].copy()
        for year in (2023, 2024, 2025):
            if stream == "testing":
                positive_value = float(positive[f"latent_testing_log_index_{year}"])
                negative_value = float(negative[f"latent_testing_log_index_{year}"])
                midpoint = 0.5 * (positive_value + negative_value)
                delta = factor * (positive_value - negative_value)
                positive[f"latent_testing_log_index_{year}"] = midpoint + 0.5 * delta
                negative[f"latent_testing_log_index_{year}"] = midpoint - 0.5 * delta
            else:
                positive_baseline = max(
                    float(positive["latent_severe_source_baseline"]), 1e-12
                )
                negative_baseline = max(
                    float(negative["latent_severe_source_baseline"]), 1e-12
                )
                positive_log = np.log(
                    max(float(positive[f"latent_severe_source_{year}"]), 1e-12)
                    / positive_baseline
                )
                negative_log = np.log(
                    max(float(negative[f"latent_severe_source_{year}"]), 1e-12)
                    / negative_baseline
                )
                midpoint = 0.5 * (positive_log + negative_log)
                delta = factor * (positive_log - negative_log)
                positive[f"latent_severe_source_{year}"] = positive_baseline * np.exp(
                    midpoint + 0.5 * delta
                )
                negative[f"latent_severe_source_{year}"] = negative_baseline * np.exp(
                    midpoint - 0.5 * delta
                )
        rows.extend([positive, negative])

    rescaled = pd.DataFrame(rows).sort_values(
        ["iso3", "draw_rank", "scenario_id"]
    ).reset_index(drop=True)
    achieved = float(
        np.median(np.abs(stream_deltas(rescaled, stream))) / measurement_sd
    )
    return rescaled, {
        "native_median_absolute_log_contrast": native_median,
        "native_median_absolute_standardised_contrast": native_median / measurement_sd,
        "contrast_rescaling_factor": factor,
        "achieved_median_absolute_standardised_contrast": achieved,
    }


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


def evaluate(features: pd.DataFrame, *, target: float, stream: str,
             n_bootstrap: int = SURVEILLANCE_BOOTSTRAP_REPLICATES) -> dict[str, object]:
    package = f"notifications+{stream}"
    columns = feature_columns_for_package(features, (stream,))
    grouped = cross_validated_predictions(features, columns, n_folds=5, shrinkage=0.80)
    grouped_metrics = discrimination_metrics(
        grouped["truth_active"], grouped["probability_active"]
    )
    grouped_lower, grouped_upper = nested_country_draw_bootstrap_auc(
        grouped,
        n_bootstrap=n_bootstrap,
        seed=BASE_SEED + int(target * 1000) + (0 if stream == "testing" else 100),
    )
    holdout = leave_one_country_out_predictions(features, columns, shrinkage=0.80)
    holdout_metrics = discrimination_metrics(
        holdout["truth_active"], holdout["probability_active"]
    )
    country_auc = [
        discrimination_metrics(country["truth_active"], country["probability_active"])["auc"]
        for _, country in holdout.groupby("iso3", sort=True)
    ]
    return {
        "target_standardised_contrast": target,
        "stream": stream,
        "package_id": package,
        "grouped_cv_auc": grouped_metrics["auc"],
        "grouped_cv_auc_lower_95": grouped_lower,
        "grouped_cv_auc_upper_95": grouped_upper,
        "whole_country_holdout_auc": holdout_metrics["auc"],
        "holdout_country_auc_median": float(np.median(country_auc)),
        "holdout_country_auc_minimum": float(np.min(country_auc)),
        "holdout_country_auc_maximum": float(np.max(country_auc)),
        "holdout_countries_auc_ge_0_70": int(np.sum(np.asarray(country_auc) >= 0.70)),
        "n_country_structures": int(features["iso3"].nunique()),
        "n_paired_backgrounds": int(features["group_id"].nunique()),
        "analysis_version": "nc_equal_contrast_v1_source_preserved_20260909",
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    display["Grouped CV AUC (95% UI)"] = display.apply(
        lambda row: (
            f"{row['grouped_cv_auc']:.3f} "
            f"({row['grouped_cv_auc_lower_95']:.3f}–"
            f"{row['grouped_cv_auc_upper_95']:.3f})"
        ),
        axis=1,
    )
    display["Whole-country holdout AUC"] = display["whole_country_holdout_auc"].map(
        lambda value: f"{value:.3f}"
    )
    display["Holdout country AUC median (range)"] = display.apply(
        lambda row: (
            f"{row['holdout_country_auc_median']:.3f} "
            f"({row['holdout_country_auc_minimum']:.3f}–"
            f"{row['holdout_country_auc_maximum']:.3f})"
        ),
        axis=1,
    )
    display = display[
        [
            "target_standardised_contrast",
            "stream",
            "Grouped CV AUC (95% UI)",
            "Whole-country holdout AUC",
            "Holdout country AUC median (range)",
            "holdout_countries_auc_ge_0_70",
        ]
    ]
    display.columns = [
        "Median latent contrast / reference SD",
        "Auxiliary stream",
        "Grouped CV AUC (95% UI)",
        "Whole-country holdout AUC",
        "Holdout country AUC median (range)",
        "Countries with holdout AUC ≥0.70",
    ]
    display["Median latent contrast / reference SD"] = display["Median latent contrast / reference SD"].map(
        lambda value: f"{value:.1f}"
    )
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n\nReference SD = 0.20: Gaussian testing error or severe log-rate drift only. " + INTERPRETATION_BOUNDARY + "\n"


_WORKER_LATENT: pd.DataFrame | None = None


def _initialise_worker() -> None:
    global _WORKER_LATENT
    _WORKER_LATENT = matched_latent()


def _evaluate_task(task: tuple[float, str, int]) -> dict:
    target, stream, n_bootstrap = task
    started = time.monotonic()
    if _WORKER_LATENT is None:
        _initialise_worker()
    design = MEASUREMENT_DESIGNS["main"]
    measurement_sd = design.testing_log_sd if stream == "testing" else design.severe_log_sd
    latent, scale_summary = rescale_stream_contrast(
        _WORKER_LATENT, stream=stream,
        target_standardised_contrast=target, measurement_sd=measurement_sd,
    )
    result = evaluate(simulate_features(latent), target=target, stream=stream, n_bootstrap=n_bootstrap)
    result.update(scale_summary)
    result["measurement_noise_sd"] = measurement_sd
    return {"result": result, "worker_pid": os.getpid(), "elapsed_seconds": time.monotonic() - started}


def _fingerprint(n_bootstrap: int) -> str:
    digest = hashlib.sha256()
    for path in (INPUT_PATH, Path(__file__), SRC / "pertussis_global_dynamics/bayes/model2c_surveillance_design.py"):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps({"bootstrap": n_bootstrap, "numpy": np.__version__, "pandas": pd.__version__,
                              "python": sys.version, "native_threads": 1}, sort_keys=True).encode())
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    """Only the parent writes outputs; task seeds do not depend on execution order."""
    output = (args.output_dir.resolve() / "results.csv") if args.output_dir else OUTPUT_PATH
    run_dir = args.output_dir.resolve() if args.output_dir else ROOT / "outputs/nc_equal_contrast_parallel"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tasks = [(target, stream, args.bootstrap) for target in TARGET_STANDARDISED_CONTRASTS for stream in STREAMS]
    budget = process_budget(args.workers, tasks=len(tasks), memory_gib_per_worker=args.memory_gib_per_worker)
    fingerprint = _fingerprint(args.bootstrap)
    rows, completed, pending = [], [], []
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    status = {"status": "running", "fingerprint": fingerprint, "budget": budget.to_dict(),
              "bootstrap_replicates": args.bootstrap, "completed_tasks": [], "total_tasks": len(tasks),
              "parent_pid": os.getpid(), "resumed_tasks": 0}

    def task_path(task):
        return checkpoint_dir / f"{task[0]:g}_{task[1]}.json"

    for task in tasks:
        path = task_path(task)
        if args.resume and path.exists():
            saved = json.loads(path.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError(f"Checkpoint inputs/code differ: {path}; use a new --output-dir")
            if (saved["result"]["stream"] != task[1]
                    or saved["result"]["target_standardised_contrast"] != task[0]):
                raise ValueError(f"Checkpoint belongs to a different task: {path}")
            rows.append(saved["result"])
            completed.append(saved)
            status["resumed_tasks"] += 1
        else:
            pending.append(task)

    status["completed_tasks"] = completed
    _atomic_json(run_dir / "status.json", status)
    print(f"Process budget: {budget.workers} workers × 1 native thread; "
          f"{budget.available_cpus} CPUs, {budget.available_memory_bytes / 1024**3:.1f} GiB available; "
          f"{len(pending)} pending tasks", flush=True)

    def accept(task, outcome):
        outcome["fingerprint"] = fingerprint
        _atomic_json(task_path(task), outcome)
        rows.append(outcome["result"])
        completed.append(outcome)
        _atomic_json(run_dir / "status.json", status)
        print(f"[{len(completed)}/{len(tasks)}] {task[1]}, contrast={task[0]:g}: "
              f"AUC={outcome['result']['whole_country_holdout_auc']:.3f}; "
              f"pid={outcome['worker_pid']}, {outcome['elapsed_seconds']:.1f}s", flush=True)

    try:
        if pending and budget.workers == 1:
            _initialise_worker()
            for task in pending:
                accept(task, _evaluate_task(task))
        elif pending:
            with ProcessPoolExecutor(max_workers=min(budget.workers, len(pending)),
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialise_worker) as executor:
                futures = {executor.submit(_evaluate_task, task): task for task in pending}
                for future in as_completed(futures):
                    accept(futures[future], future.result())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(run_dir / "status.json", status)
        raise

    frame = pd.DataFrame(rows).sort_values(
        ["target_standardised_contrast", "stream"]
    )
    if not np.allclose(
        frame["achieved_median_absolute_standardised_contrast"],
        frame["target_standardised_contrast"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("Equal-contrast rescaling did not reach its target")
    _atomic_csv(output, frame)
    if not args.output_dir:
        _atomic_csv(TABLE_PATH, frame)
        MARKDOWN_PATH.write_text(markdown_table(frame), encoding="utf-8")
    else:
        (run_dir / "results.md").write_text(markdown_table(frame), encoding="utf-8")
    status.update(status="complete", elapsed_seconds=time.monotonic() - started, output=str(output))
    _atomic_json(run_dir / "status.json", status)
    print(f"Wrote {output}; {status['elapsed_seconds']:.1f}s elapsed", flush=True)
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-labels", action="store_true",
        help="Refresh interpretation metadata and the display from existing results; no simulation.",
    )
    parser.add_argument("--workers", type=int, default=0, help="Process count; 0 selects a CPU/memory-bounded count.")
    parser.add_argument("--memory-gib-per-worker", type=float, default=1.0, help="Conservative planning estimate, not an OS memory limit.")
    parser.add_argument("--output-dir", type=Path, help="Isolated run directory; never updates manuscript results.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed tasks only if code, inputs and configuration match.")
    parser.add_argument("--bootstrap", type=int, default=SURVEILLANCE_BOOTSTRAP_REPLICATES,
                        help="Bootstrap replicates; non-default values require an isolated output directory.")
    args = parser.parse_args(argv)
    if args.refresh_labels and (args.output_dir or args.resume):
        parser.error("--refresh-labels updates the canonical display only; cannot combine with --output-dir or --resume")
    if args.workers < 0 or args.memory_gib_per_worker <= 0 or args.bootstrap < 1:
        parser.error("Worker count must be nonnegative; memory and bootstrap counts must be positive")
    if args.bootstrap != SURVEILLANCE_BOOTSTRAP_REPLICATES and not args.output_dir:
        parser.error("Non-default bootstrap counts require --output-dir to protect manuscript results")
    if args.refresh_labels:
        existing = pd.read_csv(OUTPUT_PATH)
        existing["interpretation_boundary"] = INTERPRETATION_BOUNDARY
        existing.to_csv(OUTPUT_PATH, index=False)
        existing.to_csv(TABLE_PATH, index=False)
        MARKDOWN_PATH.write_text(markdown_table(existing), encoding="utf-8")
        print("Refreshed partial-scale-matching labels; no numerical analysis rerun.")
    else:
        run_dir = args.output_dir.resolve() if args.output_dir else ROOT / "outputs/nc_equal_contrast_parallel"
        with exclusive_run_lock(run_dir / ".run.lock"):
            run_comparison(args)


if __name__ == "__main__":
    main()
