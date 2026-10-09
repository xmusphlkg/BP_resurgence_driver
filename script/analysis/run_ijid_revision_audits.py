#!/usr/bin/env python3
"""Run the targeted audits requested for the IJID revision.

The audits use the frozen matched-notification library.  They are deliberately
small additions to the primary analysis:

* stratify the process-anchored discrimination by the pair's larger/smaller
  2024 infant (0–11 month) symptomatic-burden ratio, while retaining country structure;
* regenerate notification counts with fresh, independent negative-binomial
  noise under the exact matching contract and repeat the full preprocessing,
  grouped-CV and leave-one-country-out evaluation;
* export the existing country-level and model-library sensitivity evidence in a
  single revision audit manifest.

These analyses are conditional on the frozen nine-country generator library.
They do not refit Stan models and do not constitute external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.bayes.model2c_surveillance_design import (  # noqa: E402
    DEFAULT_CLASSIFIER,
    _fit_classifier,
    _predict_classifier,
    cross_validated_predictions,
    discrimination_metrics,
    feature_columns_for_package,
    leave_one_country_out_predictions,
    roc_auc,
)


TAG = "monitoring_packages_v4_source_preserved_20260908"
FEATURES = ROOT / "outputs" / "model2c_surveillance_design" / TAG / "observation_features_main.csv"
LATENT = ROOT / "outputs" / "model2c_surveillance_design" / TAG / "latent_surveillance_signals.csv"
RATIOS = ROOT / "data" / "derived" / "matched_latent_ratios.csv"
COUNTRY_AUC = ROOT / "outputs" / "country_macro_revision_20260921" / "country_auc.csv"
HOLDOUT = ROOT / "outputs" / "country_macro_revision_20260921" / "holdout_predictions.csv"
MODEL_LIBRARY = ROOT / "outputs" / "model2c_surveillance_design" / TAG / "model_library_surveillance_sensitivity.csv"
OUT = ROOT / "outputs" / "ijid_revision_audits_20261008"
TABLES = ROOT / "outputs" / "derived_tables"
FIGURE_DATA = ROOT / "outputs" / "figure_data"
REPORT = ROOT / "outputs" / "local_reports" / "ijid_revision_audits_2026-10-08.md"

POSITIVE = "matched_transmission_truth"
NEGATIVE = "matched_reporting_truth"
SCENARIOS = (POSITIVE, NEGATIVE)
POST_YEARS = (2023, 2024, 2025)
MONTHS = tuple(range(1, 13))
DEFAULT_SEED = 20261008
DEFAULT_NULL_REPLICATES = 1000
STRATA = (("1.0–1.5", 1.0, 1.5), ("1.5–2.0", 1.5, 2.0), ("2.0–4.0", 2.0, 4.0), ("≥4.0", 4.0, np.inf))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def workspace_status() -> str:
    """Return the complete porcelain status used for provenance hashing."""
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except Exception:
        return "unavailable"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES)
    latent = pd.read_csv(LATENT)
    features = features.loc[features["scenario_id"].isin(SCENARIOS)].copy()
    latent = latent.loc[latent["scenario_id"].isin(SCENARIOS)].copy()
    if len(features) != 720 * 2 * 3:
        raise ValueError(f"Expected 4,320 feature rows, found {len(features)}")
    pair_counts = features.groupby("group_id")["scenario_id"].nunique()
    if len(pair_counts) != 720 or not pair_counts.eq(2).all():
        raise ValueError("The frozen feature table does not contain 720 complete two-class pairs")
    truth = features["scenario_id"].eq(POSITIVE).astype(int)
    features["truth_active"] = truth
    # The primary severe-outcome contrast is defined for infants (0–11
    # months), not the broader under-five group.  Reuse the audited ratio
    # export that drives the matched-latent-ratio figure so the strata have
    # the same estimand as the main 8.71-fold burden result.
    if not RATIOS.exists():
        raise FileNotFoundError(f"Missing audited infant-burden ratio file: {RATIOS}")
    ratio_frame = pd.read_csv(RATIOS)
    required_ratio_columns = {"group_id", "infant_symptomatic_burden_ratio_2024"}
    if not required_ratio_columns.issubset(ratio_frame.columns):
        raise ValueError("Audited ratio export is missing the infant 2024 burden ratio")
    if ratio_frame["group_id"].nunique() != 720:
        raise ValueError("Audited infant-burden ratio export does not contain 720 pair backgrounds")
    burden = ratio_frame[["group_id", "infant_symptomatic_burden_ratio_2024"]].rename(
        columns={"infant_symptomatic_burden_ratio_2024": "burden_ratio_high_low"}
    )
    signed_ratio = pd.to_numeric(burden["burden_ratio_high_low"], errors="coerce")
    burden["burden_ratio_high_low"] = np.maximum(signed_ratio, 1.0 / signed_ratio)
    if (burden["burden_ratio_high_low"] < 1.0).any() or not np.isfinite(burden["burden_ratio_high_low"]).all():
        raise ValueError("Infant burden ratios must be finite and at least one")
    features = features.merge(burden, on="group_id", how="left", validate="many_to_one")
    if features["burden_ratio_high_low"].isna().any():
        raise ValueError("Missing burden contrast for one or more feature rows")
    return features, latent, burden


def country_holdout(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """LOCO predictions without imposing the primary table's >=3-country guard."""
    countries = sorted(frame["iso3"].astype(str).unique())
    if len(countries) < 2:
        raise ValueError("At least two countries are required for a country holdout")
    outputs: list[pd.DataFrame] = []
    for fold, country in enumerate(countries):
        train = frame.loc[frame["iso3"].astype(str).ne(country)]
        test = frame.loc[frame["iso3"].astype(str).eq(country)]
        model = _fit_classifier(
            train.loc[:, columns].to_numpy(dtype=float),
            train["truth_active"].astype(int).to_numpy(),
            classifier=DEFAULT_CLASSIFIER,
            shrinkage=0.80,
            seed=20260726 + fold * 1009,
        )
        part = test[["iso3", "draw_id", "draw_rank", "group_id", "scenario_id", "observation_replicate", "truth_active"]].copy()
        part["fold"] = fold
        part["holdout_country"] = country
        part["probability_active"] = _predict_classifier(model, test.loc[:, columns].to_numpy(dtype=float))
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def country_auc_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in predictions.groupby("iso3", sort=True):
        rows.append({"iso3": country, "auc": roc_auc(group["truth_active"], group["probability_active"]), "n_pairs": group["group_id"].nunique(), "n_rows": len(group)})
    return pd.DataFrame(rows)


def macro_bootstrap(predictions: pd.DataFrame, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    countries = sorted(predictions["iso3"].astype(str).unique())
    point = float(country_auc_table(predictions)["auc"].mean())
    rng = np.random.default_rng(seed)
    by_country = {}
    for country, group in predictions.groupby("iso3", sort=True):
        group = group.reset_index(drop=True)
        pair_indices = {
            pair: values.to_numpy(dtype=int)
            for pair, values in group.groupby("group_id", sort=False).groups.items()
        }
        by_country[country] = {
            "pairs": tuple(pair_indices),
            "indices": pair_indices,
            "y": group["truth_active"].to_numpy(dtype=int),
            "p": group["probability_active"].to_numpy(dtype=float),
        }
    values = []
    for _ in range(n_bootstrap):
        aucs = []
        for country in countries:
            group = by_country[country]
            pairs = group["pairs"]
            sampled = rng.choice(pairs, size=len(pairs), replace=True)
            indices = np.concatenate([group["indices"][pair] for pair in sampled])
            aucs.append(roc_auc(group["y"][indices], group["p"][indices]))
        values.append(float(np.nanmean(aucs)))
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def burden_stratified(features: pd.DataFrame, *, n_bootstrap: int = 1000, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify frozen primary predictions; never fit on a stratum."""
    frozen = pd.read_csv(HOLDOUT)
    frozen = frozen.loc[frozen.dataset.eq("process_anchored") & frozen.classifier.eq(DEFAULT_CLASSIFIER)]
    latent = pd.read_csv(LATENT)
    latent = latent.loc[latent.scenario_id.isin(SCENARIOS)].copy()
    assert set(latent.severe_source_age_scope) == {"under_1_year"}
    latent["total"] = latent[[f"latent_severe_source_{y}" for y in POST_YEARS]].sum(axis=1)
    total = latent.pivot(index="group_id", columns="scenario_id", values="total")
    ratio = total.max(axis=1) / total.min(axis=1)
    pairs = features[["group_id", "iso3", "burden_ratio_high_low"]].drop_duplicates().copy()
    pairs["burden_ratio_2023_2025"] = pairs.group_id.map(ratio)
    pairs.to_csv(FIGURE_DATA / "burden_ratio_by_pair.csv", index=False)
    pairs.to_csv(OUT / "burden_ratio_by_pair.csv", index=False)
    rows, detail, evaluated = [], [], []
    for window, column in (("2024", "burden_ratio_high_low"), ("2023–2025", "burden_ratio_2023_2025")):
        for label, lower, upper in STRATA:
            selected = pairs.loc[pairs[column].ge(lower) & pairs[column].lt(upper)]
            for package in ("testing", "severe"):
                hold = frozen.loc[frozen.package.eq(package)].merge(selected, on=["group_id", "iso3"], validate="many_to_one")
                if hold.empty:
                    continue
                macro, lo, hi = macro_bootstrap(hold, n_bootstrap, seed + len(rows)*37)
                metadata = dict(burden_window=window, burden_stratum=label, package_id="notifications+"+package)
                rows.append(dict(**metadata, n_pairs=len(selected), n_countries=selected.iso3.nunique(),
                                 countries=",".join(sorted(selected.iso3.unique())), country_macro_auc=macro,
                                 country_macro_lower_95=lo, country_macro_upper_95=hi,
                                 bootstrap_replicates=n_bootstrap,
                                 validation_note="Frozen primary nine-country LOCO scores; within-country pair bootstrap; no stratum refitting"))
                detail.extend(dict(**metadata, **r) for r in country_auc_table(hold).to_dict("records"))
                evaluated.append(hold.assign(**metadata))
    pd.concat(evaluated).to_csv(OUT / "burden_stratified_frozen_predictions.csv", index=False)
    counts=[]
    for window, column in (("2024", "burden_ratio_high_low"), ("2023–2025", "burden_ratio_2023_2025")):
        for label, lower, upper in STRATA:
            subset=pairs.loc[pairs[column].ge(lower) & pairs[column].lt(upper)]
            for country in sorted(pairs.iso3.unique()):
                counts.append(dict(burden_window=window,burden_stratum=label,iso3=country,n_pairs=int(subset.iso3.eq(country).sum())))
    pd.DataFrame(counts).to_csv(OUT / "burden_stratum_country_counts.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(detail)


def gamma_poisson(rng: np.random.Generator, mean: float, phi: float) -> int:
    mean = max(float(mean), 0.0)
    phi = max(float(phi), 0.2)
    if mean <= 0:
        return 0
    return int(rng.poisson(rng.gamma(shape=phi, scale=mean / phi)))


def regenerated_notifications(latent: pd.DataFrame, seed: int, observation_replicates: int = 3) -> pd.DataFrame:
    """Regenerate notification features from identical means and fixed baselines."""
    metadata = ["iso3", "draw_id", "draw_rank", "group_id", "scenario_id"]
    columns = [f"feature_notifications_{y}_m{m:02d}" for y in POST_YEARS for m in MONTHS]
    means = latent[[c.replace("feature_notifications", "latent_notification_mean") for c in columns]].to_numpy(float)
    baseline = np.tile(latent[[f"latent_notification_baseline_m{m:02d}" for m in MONTHS]].to_numpy(float), (1,3))
    phi = np.maximum(latent.nb_phi_month.to_numpy(float)[:,None], .2)
    frames = []
    for replicate, child in enumerate(np.random.SeedSequence(seed).spawn(observation_replicates), 1):
        rng = np.random.default_rng(child)
        counts = rng.poisson(rng.gamma(shape=phi, scale=means/phi))
        frame = latent[metadata].reset_index(drop=True).copy()
        frame["truth_active"] = frame.scenario_id.eq(POSITIVE).astype(int)
        frame["observation_replicate"] = replicate
        frame[columns] = np.log((counts+.5)/(baseline+.5))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def regenerated_noise_null(latent: pd.DataFrame, *, n_replicates: int, seed: int, n_bootstrap: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for replicate in range(n_replicates):
        frame = regenerated_notifications(latent, seed + replicate)
        columns = feature_columns_for_package(frame, ())
        grouped = cross_validated_predictions(frame, columns, classifier=DEFAULT_CLASSIFIER)
        holdout = leave_one_country_out_predictions(frame, columns, classifier=DEFAULT_CLASSIFIER)
        grouped_metrics = discrimination_metrics(grouped["truth_active"], grouped["probability_active"])
        macro = float(country_auc_table(holdout)["auc"].mean())
        lower = upper = float("nan")
        if replicate % 50 == 0:
            print(f"Notification null {replicate}/{n_replicates}", flush=True)
        rows.append({
            "replicate": replicate,
            "seed": seed + replicate,
            "grouped_auc": float(grouped_metrics["auc"]),
            "pooled_holdout_auc": float(roc_auc(holdout["truth_active"], holdout["probability_active"])),
            "country_macro_auc": macro,
            "country_macro_lower_95": lower,
            "country_macro_upper_95": upper,
            "n_pairs": frame["group_id"].nunique(),
            "n_countries": frame["iso3"].nunique(),
        })
    result = pd.DataFrame(rows)
    primary = pd.read_csv(ROOT / "outputs/country_macro_revision_20260921/macro_auc_summary.csv")
    primary = primary.loc[primary.dataset.eq("process_anchored") & primary.package.eq("notifications")].iloc[0]
    observed = {"grouped_auc": primary.grouped_auc, "pooled_holdout_auc": primary.pooled_holdout_auc, "country_macro_auc": primary.macro_auc}
    summary_rows = []
    for metric in ("grouped_auc", "pooled_holdout_auc", "country_macro_auc"):
        values = result[metric].to_numpy(float)
        summary_rows.append({
            "metric": metric,
            "n_replicates": len(values),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "observed_auc": float(observed[metric]),
            "observed_percentile": float(100*np.mean(values <= observed[metric])),
            "upper_tail_exceedances": int(np.sum(values >= observed[metric])),
            "upper_tail_probability_plus_one": float((1+np.sum(values >= observed[metric]))/(len(values)+1)),
            "two_sided_probability_plus_one": float((1+np.sum(abs(values-.5) >= abs(observed[metric]-.5)))/(len(values)+1)),
            "fraction_ge_0_525": float(np.mean(values >= 0.525)),
            "fraction_le_0_475": float(np.mean(values <= 0.475)),
        })
    return result, pd.DataFrame(summary_rows)


def matching_contract(latent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = latent.loc[latent["scenario_id"].isin(SCENARIOS)].copy()
    for column in [
        "nb_phi_month",
        *[f"latent_notification_baseline_m{m:02d}" for m in MONTHS],
        *[
            f"latent_notification_mean_{year}_m{m:02d}"
            for year in POST_YEARS
            for m in MONTHS
        ],
    ]:
        pivot = selected.pivot(index="group_id", columns="scenario_id", values=column)
        diff = (pivot[POSITIVE] - pivot[NEGATIVE]).abs()
        rows.append({"quantity": column, "max_absolute_pair_difference": float(diff.max()), "n_nonzero_differences": int((diff > 1e-10).sum())})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-replicates", type=int, default=DEFAULT_NULL_REPLICATES)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    features, latent, burden = read_inputs()

    burden_summary, burden_country = burden_stratified(features, n_bootstrap=args.bootstrap_replicates, seed=args.seed)
    burden_summary.to_csv(OUT / "burden_stratified_auc.csv", index=False)
    burden_summary.to_csv(TABLES / "burden_stratified_auc.csv", index=False)
    burden_country.to_csv(OUT / "burden_stratified_country_auc.csv", index=False)
    burden_country.to_csv(TABLES / "burden_stratified_country_auc.csv", index=False)

    null_replicates, null_summary = regenerated_noise_null(
        latent, n_replicates=args.null_replicates, seed=args.seed, n_bootstrap=min(args.bootstrap_replicates, 500)
    )
    null_replicates.to_csv(OUT / "notification_regenerated_noise_null_replicates.csv", index=False)
    null_summary.to_csv(OUT / "notification_regenerated_noise_null_summary.csv", index=False)
    null_summary.to_csv(TABLES / "notification_regenerated_noise_null_summary.csv", index=False)
    contract = matching_contract(latent)
    contract.to_csv(OUT / "notification_noise_contract.csv", index=False)
    contract.to_csv(TABLES / "notification_noise_contract.csv", index=False)

    if COUNTRY_AUC.exists():
        country_auc = pd.read_csv(COUNTRY_AUC)
        country_auc.to_csv(OUT / "country_level_primary_auc.csv", index=False)
        country_auc.to_csv(TABLES / "country_level_primary_auc.csv", index=False)
    if MODEL_LIBRARY.exists():
        model_library = pd.read_csv(MODEL_LIBRARY)
        model_library.to_csv(OUT / "model_library_sensitivity_status.csv", index=False)

    status = workspace_status()
    manifest = {
        "analysis_name": "ijid_revision_audits",
        "created_utc": "2026-10-08",
        "git_revision_at_run": git_revision(),
        "python": platform.python_version(),
        "seed": args.seed,
        "notification_null_replicates": args.null_replicates,
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary_estimand": "country-macro AUC under process-anchored native-map measurement, conditional on nine pre-fitted generators",
        "burden_ratio_definition": "larger/smaller infant symptomatic burden for 2024 and summed 2023–2025",
        "stratification_prediction_source": "frozen process_anchored primary nine-country LOCO predictions; no retraining",
        "burden_ratio_source": "manuscript/figure_data/figure_4_matched_latent_ratios.csv: infant_symptomatic_burden_ratio_2024",
        "burden_strata": [label for label, _, _ in STRATA],
        "notification_null": "fresh independent gamma-Poisson/negative-binomial count noise with identical matched means; 3 observation replicates per null draw; 2015–2019 monthly baseline means fixed; grouped-CV and leave-one-country-out evaluation repeated",
        "notification_null_baseline_handling": "baseline monthly means are held fixed; only post-period notification counts are regenerated",
        "notification_null_observation_replicates": 3,
        "script_sha256": sha256(Path(__file__)),
        "workspace_dirty": bool(status.strip()),
        "workspace_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (FEATURES, LATENT, RATIOS, HOLDOUT, ROOT / "outputs/agent_measurement_model_upgrade_20260911/observation_features_calibrated.csv")},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(OUT.glob("*.csv"))},
        "interpretation_boundary": "AUC intervals and stratified results are conditional on fixed generators, folds, classifier and observation model; they are not external validation or field thresholds.",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# IJID revision audits (2026-10-08)",
        "",
        "These audits use the frozen nine-country matched-notification library. They target the two main concerns raised before submission: whether severe-outcome discrimination is driven by large latent burden contrasts, and whether the notification-only AUC of 0.525 is a finite realization of a null process.",
        "",
        "## Burden-stratified discrimination",
        "",
        "Ratios use 2024 and summed 2023–2025 infant burden. Scores are frozen primary process-anchored nine-country LOCO predictions, with no stratum refitting. Intervals resample complete pairs within represented countries, keeping country composition fixed. The previous audit used idealised features and refitted within strata; its AUCs are superseded and were not primary-rule performance.",
        "",
        markdown_table(burden_summary),
        "",
        "## Regenerated notification-noise null",
        "",
        "Each replicate regenerates both scenario rows from the same matched expected means and pair-specific dispersion, with a fresh seed, then reruns fold-specific preprocessing, grouped five-fold evaluation and leave-one-country-out transfer. The null summary should be interpreted as an implementation and finite-noise audit, not as a replacement for the primary process-anchored result.",
        "",
        markdown_table(null_summary),
        "",
        "## Contract and provenance",
        "",
        markdown_table(contract),
        "",
        f"Inputs are SHA-256 frozen in `{(OUT / 'analysis_manifest.json').relative_to(ROOT)}`. The run used seed {args.seed}, {args.null_replicates} regenerated-noise replicates and {args.bootstrap_replicates} within-country pair bootstrap replicates for stratification. The null distribution uses repeated complete classifier evaluations without an unnecessary inner bootstrap.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(burden_summary.to_string(index=False))
    print(null_summary.to_string(index=False))


if __name__ == "__main__":
    main()
