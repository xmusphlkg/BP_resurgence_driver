#!/usr/bin/env python3
"""Country-macro AUC and paired conditional uncertainty for review comparisons.

Resample countries, then complete calibration pairs within each sampled country;
retain all observation replicates. Identical resampling weights are used for
every comparison. Predictions stay fixed: intervals exclude model refitting,
classifier training and selection uncertainty. Country AUCs are calculated
before averaging, so separately fitted score scales are never pooled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "script/src"))
from pertussis_global_dynamics.bayes.model2c_surveillance_design import (  # noqa: E402
    CLASSIFIER_IDS, DEFAULT_CLASSIFIER, cross_validated_predictions,
    feature_columns_for_package, leave_one_country_out_predictions, roc_auc,
)

OUT = ROOT / "outputs/country_macro_revision_20260921"
TABLES = ROOT / "manuscript/tables"
N_BOOT = 1000
SEED = 20260923
PACKAGES = {
    "notifications": (),
    "age": ("age",),
    "testing": ("testing",),
    "severe": ("severe",),
    "testing_severe": ("testing", "severe"),
    "testing_severe_recent_serology": ("testing", "severe", "serology"),
    "testing_severe_serology": ("testing", "severe", "serology"),
    "all_streams": ("age", "testing", "severe", "serology"),
}


def macro_bootstrap(pred: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    countries = sorted(pred.iso3.unique())
    rng = np.random.default_rng(SEED)
    values = np.empty(N_BOOT, dtype=float)
    point = {}
    matrices = {}
    for iso3 in countries:
        part = pred.loc[pred.iso3.eq(iso3)]
        groups = sorted(part.group_id.unique())
        positives = [part.loc[part.group_id.eq(g) & part.truth_active.eq(1), "probability_active"].to_numpy() for g in groups]
        negatives = [part.loc[part.group_id.eq(g) & part.truth_active.eq(0), "probability_active"].to_numpy() for g in groups]
        if len({len(x) for x in positives + negatives}) != 1 or not len(positives[0]):
            raise ValueError("Incomplete or unequal paired observation replicates")
        matrix = np.empty((len(groups), len(groups)), dtype=float)
        for i, p in enumerate(positives):
            for j, n in enumerate(negatives):
                matrix[i, j] = ((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])).mean()
        matrices[iso3] = matrix
        point[iso3] = float(matrix.mean())
        if not np.isclose(point[iso3], roc_auc(part.truth_active, part.probability_active)):
            raise ValueError("Pairwise AUC does not reproduce the rank statistic")
    for replicate in range(N_BOOT):
        sampled_countries = rng.choice(countries, size=len(countries), replace=True)
        country_values = []
        for sampled in sampled_countries:
            matrix = matrices[sampled]
            weights = np.bincount(rng.integers(matrix.shape[0], size=matrix.shape[0]), minlength=matrix.shape[0]).astype(float)
            country_values.append(float(weights @ matrix @ weights / (matrix.shape[0] ** 2)))
        values[replicate] = float(np.nanmean(country_values))
    return pd.Series(point, name="auc"), values


def evaluate(features: pd.DataFrame, dataset: str, package: str,
             classifier: str = DEFAULT_CLASSIFIER) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DataFrame]:
    frame = features.loc[features.scenario_id.isin(["matched_transmission_truth", "matched_reporting_truth"])].copy()
    frame["truth_active"] = frame.scenario_id.eq("matched_transmission_truth").astype(int)
    columns = feature_columns_for_package(frame, PACKAGES[package])
    if package == "testing_severe_recent_serology":
        columns = [c for c in columns if not c.startswith("feature_serology_immunity_")]
    hold = leave_one_country_out_predictions(frame, columns, classifier=classifier)
    hold["probability_active"] = hold.probability_active.clip(1e-9, 1 - 1e-9)
    country, boot = macro_bootstrap(hold)
    grouped = cross_validated_predictions(frame, columns, classifier=classifier)
    metadata = {"dataset": dataset, "package": package, "classifier": classifier}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    row = dict(metadata, n_features=len(columns), observation_replicates=int(frame.observation_replicate.nunique()), macro_auc=float(country.mean()),
               macro_auc_lower_95=float(lo), macro_auc_upper_95=float(hi),
               country_median=float(country.median()), country_min=float(country.min()),
               country_max=float(country.max()), country_n_ge_070=int((country >= .70).sum()),
               pooled_holdout_auc=roc_auc(hold.truth_active, hold.probability_active),
               grouped_auc=roc_auc(grouped.truth_active, grouped.probability_active.clip(1e-9, 1-1e-9)))
    detail = country.rename_axis("iso3").reset_index().assign(**metadata)
    return row, detail, boot, hold.assign(**metadata)


def pretty_dataset(value: str) -> str:
    return {
        "process_anchored": "process-anchored",
        "idealized": "idealised",
        "native_map": "native map",
        "effective_map": "effective map",
    }.get(value, value)


def pretty_package(value: str) -> str:
    return {
        "notifications": "notifications",
        "age": "age composition",
        "testing": "testing",
        "severe": "severe outcomes",
        "testing_severe": "testing + severe",
        "testing_severe_recent_serology": "testing + severe + recent serology",
        "testing_severe_serology": "testing + severe + serology",
        "all_streams": "all streams",
    }.get(value, value)


def pretty_classifier(value: str) -> str:
    return {
        "shrinkage_linear_discriminant": "shrinkage linear discriminant",
        "ridge_logistic": "ridge logistic",
        "random_forest_stumps": "random-forest stumps",
        "gradient_boosted_stumps": "boosted stumps",
        "random_features_neural": "random-feature neural",
    }.get(value, value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "process_anchored": ROOT / "outputs/agent_measurement_model_upgrade_20260911/observation_features_calibrated.csv",
        "idealized": ROOT / "outputs/model2c_surveillance_design/monitoring_packages_v4_source_preserved_20260908/observation_features_main.csv",
        "native_map": ROOT / "outputs/observation_process_robustness_20260921/features_native.csv",
        "effective_map": ROOT / "outputs/observation_process_robustness_20260921/features_effective.csv",
    }
    summaries, countries, predictions, bootrows = [], [], [], []
    bootstrap = {}
    points = {}
    for dataset, path in sources.items():
        features = pd.read_csv(path)
        configs = ([(p, DEFAULT_CLASSIFIER) for p in PACKAGES]
                   if dataset == "process_anchored" else
                   [(p, c) for c in CLASSIFIER_IDS for p in ("testing", "severe")]
                   if dataset == "idealized" else [("testing", DEFAULT_CLASSIFIER)])
        for package, classifier in configs:
            row, detail, boot, pred = evaluate(features, dataset, package, classifier)
            summaries.append(row)
            countries.append(detail)
            predictions.append(pred)
            key = (dataset, package, classifier)
            bootstrap[key] = boot
            points[key] = row["macro_auc"]
            bootrows.append(pd.DataFrame({"replicate": np.arange(N_BOOT), "macro_auc": boot}).assign(
                dataset=dataset, package=package, classifier=classifier))
            print(dataset, package, classifier, f"macro={row['macro_auc']:.3f}", flush=True)
    contrasts = []
    def contrast(label, left, right):
        delta = bootstrap[left] - bootstrap[right]
        lo, hi = np.quantile(delta, [.025, .975])
        contrasts.append({"comparison": label, "macro_auc_difference": points[left] - points[right],
                          "lower_95": float(lo), "upper_95": float(hi),
                          "bootstrap_replicates": N_BOOT})
    c = DEFAULT_CLASSIFIER
    for package in ("testing", "testing_severe", "testing_severe_recent_serology", "testing_severe_serology", "all_streams"):
        contrast(f"process: {package} minus severe", ("process_anchored", package, c), ("process_anchored", "severe", c))
    contrast("effective minus native testing map", ("effective_map", "testing", c), ("native_map", "testing", c))
    for classifier in CLASSIFIER_IDS:
        contrast(f"idealized severe minus testing: {classifier}", ("idealized", "severe", classifier), ("idealized", "testing", classifier))
    pd.DataFrame(summaries).to_csv(OUT / "macro_auc_summary.csv", index=False)
    pd.concat(countries, ignore_index=True).to_csv(OUT / "country_auc.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.concat(bootrows, ignore_index=True).to_csv(OUT / "macro_bootstrap_replicates.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "paired_macro_differences.csv", index=False)
    pd.DataFrame(summaries).to_csv(TABLES / "country_macro_auc.csv", index=False)
    pd.DataFrame(contrasts).to_csv(TABLES / "paired_macro_auc_differences.csv", index=False)
    lines = ["| Comparison | Macro AUC difference | 95% conditional UI |", "| --- | ---: | --- |"]
    lines += [f"| {r['comparison'].replace('_', ' ')} | {r['macro_auc_difference']:.3f} | {r['lower_95']:.3f}–{r['upper_95']:.3f} |" for r in contrasts]
    (ROOT / "manuscript/tables_md/paired_macro_auc_differences.md").write_text("\n".join(lines) + "\n")
    macro_lines = ["| Dataset | Package | Classifier | Macro AUC (95% conditional UI) | Country median (range) | Countries ≥0.70 |", "| --- | --- | --- | --- | --- | ---: |"]
    for r in summaries:
        macro_lines.append(f"| {pretty_dataset(r['dataset'])} | {pretty_package(r['package'])} | {pretty_classifier(r['classifier'])} | {r['macro_auc']:.3f} ({r['macro_auc_lower_95']:.3f}–{r['macro_auc_upper_95']:.3f}) | {r['country_median']:.3f} ({r['country_min']:.3f}–{r['country_max']:.3f}) | {r['country_n_ge_070']}/9 |")
    (ROOT / "manuscript/tables_md/country_macro_auc.md").write_text("\n".join(macro_lines) + "\n")
    report = ["# Country-macro revision analysis (21 September 2026)", "", "All packages include notifications. Process-anchored and idealised designs retain all three observation replicates; native/effective-map replays retain their one replicate. The recent-serology package excludes the exploratory infection-derived-immunity feature. Common country-first, draw-pair-second resamples hold predictions fixed; these are conditional intervals, not external validation or estimates of training uncertainty.", "", *macro_lines, "", *lines]
    (ROOT / "manuscript/reports/country_macro_revision_2026-09-21.md").write_text("\n".join(report) + "\n")
    (OUT / "manifest.json").write_text(json.dumps({
        "script": str(Path(__file__).relative_to(ROOT)),
        "sources": {k: str(v.relative_to(ROOT)) for k, v in sources.items()},
        "bootstrap_replicates": N_BOOT, "seed": SEED,
        "estimand": "Equal-country mean of within-country AUCs; fixed scenario orientation",
        "resampling": "Country first, complete draw pair second; all observation repeats retained; common resamples across comparisons",
        "limitations": "Conditional on fixed predictions and nine-country model library; excludes model refitting, training and selection uncertainty; no cost data",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
