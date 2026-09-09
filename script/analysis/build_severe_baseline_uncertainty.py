#!/usr/bin/env python3
"""Audit finite baseline measurement for the infant sentinel observation model.

This is a post-review sensitivity analysis on the frozen calibration backgrounds.
It samples a baseline count independently from follow-up counts and retains the
same follow-up random draws across baseline observation windows.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script" / "src"))
from script.run_model2c_surveillance_design import _observation_seed
from pertussis_global_dynamics.bayes.model2c_surveillance_design import (
    DEFAULT_SURVEILLANCE_TAG, SURVEILLANCE_DESIGN_VERSION, MEASUREMENT_DESIGNS,
    cross_validated_predictions, feature_columns_for_package,
    discrimination_metrics, nested_country_draw_bootstrap_auc,
    simulate_observation_features,
)


def main() -> None:
    directory = ROOT / "outputs/model2c_surveillance_design" / DEFAULT_SURVEILLANCE_TAG
    latent = pd.read_csv(directory / "latent_surveillance_signals.csv")
    matched = latent.loc[latent.scenario_id.isin([
        "matched_transmission_truth", "matched_reporting_truth"
    ])].copy()
    results, counts = [], []
    for events in (0.1, 0.5, 1.0, 2.0):
        for baseline_years in (None, 1.0, 5.0):
            design = replace(
                MEASUREMENT_DESIGNS["main"], design_id="baseline_uncertainty",
                severe_baseline_events=events, severe_log_sd=1.0,
                severe_baseline_observation_years=baseline_years,
            )
            rows = [simulate_observation_features(
                row, design=design,
                seed=_observation_seed(row, "baseline_uncertainty", 20260908, replicate),
                observation_replicate=replicate,
            ) for _, row in matched.iterrows() for replicate in (1, 2, 3)]
            features = pd.DataFrame(rows)
            features["truth_active"] = features.scenario_id.eq("matched_transmission_truth").astype(int)
            predictions = cross_validated_predictions(
                features, feature_columns_for_package(features, ("severe",)), n_folds=5
            )
            metrics = discrimination_metrics(predictions.truth_active, predictions.probability_active)
            lower, upper = nested_country_draw_bootstrap_auc(predictions, n_bootstrap=2000, seed=20260908)
            metadata = {
                "baseline_expected_infant_events_per_catchment_year": events,
                "baseline_observation_years": baseline_years,
                "baseline_measurement": "known expectation" if baseline_years is None else "observed negative-binomial count",
                "annual_log_rate_drift_sd": 1.0, "followup_years": 3,
                "n_country_backgrounds": 9, "n_paired_draw_backgrounds": 720,
                "n_observations": len(features), "bootstrap_replicates": 2000,
                "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
            }
            results.append({**metadata, **metrics, "auc_lower_95": lower, "auc_upper_95": upper})
            for scenario, group in features.groupby("scenario_id"):
                counts.append({
                    **metadata, "scenario_id": scenario,
                    "median_expected_2024_events": group.expected_severe_events_before_drift_2024.median(),
                    "q025_expected_2024_events": group.expected_severe_events_before_drift_2024.quantile(.025),
                    "q975_expected_2024_events": group.expected_severe_events_before_drift_2024.quantile(.975),
                    "fraction_zero_followup_events_all_three_years": group[[f"observed_severe_events_{y}" for y in (2023, 2024, 2025)]].sum(axis=1).eq(0).mean(),
                    "fraction_zero_measured_baseline": group.severe_baseline_events_observed.eq(0).mean(),
                })
            print(f"b={events:g}, baseline years={baseline_years}, AUC={metrics['auc']:.4f}", flush=True)
    for name, frame in [
        ("severe_baseline_uncertainty", pd.DataFrame(results)),
        ("severe_expected_and_observed_counts", pd.DataFrame(counts)),
    ]:
        frame.to_csv(directory / f"{name}.csv", index=False)
        frame.to_csv(ROOT / "manuscript/tables" / f"{name}.csv", index=False)
    write_report(results)


def write_report(results) -> None:
    report = [
        "# Infant sentinel baseline measurement sensitivity (2026-09-08)", "",
        "Post-review sensitivity; conditional on the existing nine calibration ensembles. "
        "The primary frontier treats the baseline expected rate b as known. Here an independent "
        "baseline count over L=1 or 5 years is NB2(bL, phi L), divided by L. "
        "Follow-up expectations are b times the infant symptomatic-burden ratio for each year; "
        "b is not the expected follow-up event count. All classifiers see the full three-year "
        "2023–2025 window. Follow-up random draws are held fixed across baseline windows. "
        "Annual drift is independent with log SD 1.0; persistent drift and an unknown catchment "
        "denominator remain outside this sensitivity. Intervals resample countries and paired "
        "backgrounds with fixed out-of-fold predictions, not classifier refits.", "",
        "\n".join([
            "| Baseline expected events/year | Baseline window | AUC | 95% interval |",
            "| --- | --- | --- | --- |",
            *[f"| {r['baseline_expected_infant_events_per_catchment_year']:g} | {r['baseline_measurement']} ({r['baseline_observation_years']}) | {r['auc']:.3f} | {r['auc_lower_95']:.3f}–{r['auc_upper_95']:.3f} |" for r in results],
        ]), "",
        "Expected event ranges, baseline zeros and three-year all-zero follow-up fractions "
        "are provided by scenario in severe_expected_and_observed_counts.csv.", "",
    ]
    (ROOT / "manuscript/reports/severe_baseline_uncertainty_2026-09-08.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
