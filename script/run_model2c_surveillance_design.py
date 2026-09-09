#!/usr/bin/env python3
"""Run the Model 2C simulation-based surveillance-design experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.bayes.model2c_recovery import safe_recovery_tag  # noqa: E402
from pertussis_global_dynamics.bayes.model2c_scenarios import _predict_monthly  # noqa: E402
from pertussis_global_dynamics.bayes.model2c_surveillance_design import (  # noqa: E402
    MEASUREMENT_DESIGNS,
    DEFAULT_SURVEILLANCE_TAG,
    MECHANISM_CONTRASTS,
    OPTIONAL_STREAMS,
    PASSING_PRIMARY_COUNTRIES,
    SURVEILLANCE_BOOTSTRAP_REPLICATES,
    SURVEILLANCE_DESIGN_VERSION,
    cross_validated_predictions,
    discrimination_metrics,
    feature_columns_for_package,
    minimum_successful_packages,
    nested_country_draw_bootstrap_auc,
    package_id,
    shapley_stream_gains,
    simulate_observation_features,
    surveillance_packages,
    structural_contrasts,
    trajectory_to_latent_signals,
)


OUTPUTS = ROOT / "outputs"
DEFAULT_COUNTRIES = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
BASE_SCENARIOS = ("full_model", *MECHANISM_CONTRASTS.values())


@lru_cache(maxsize=64)
def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _match_notification_trajectory(
    target: pd.DataFrame,
    latent_source: pd.DataFrame,
    *,
    match_strength: float = 1.0,
) -> pd.DataFrame:
    """Keep latent states while removing a fraction of notification deviation.

    A strength of one reproduces exact matching.  At lower strengths, the
    source notification mean is moved towards the common target on the log
    scale, leaving ``1 - match_strength`` of its native log deviation.
    """
    if not 0.0 <= match_strength <= 1.0:
        raise ValueError("match_strength must lie in [0, 1].")
    matched = latent_source.copy()
    matched["source_reported_cases_predicted"] = latent_source.get(
        "source_reported_cases_predicted", latent_source["reported_cases_predicted"]
    ).to_numpy(dtype=float)
    target_total = target["reported_cases_predicted"].to_numpy(dtype=float)
    source_total = matched["reported_cases_predicted"].to_numpy(dtype=float)
    exact_ratio = np.divide(
        target_total,
        source_total,
        out=np.ones_like(target_total),
        where=source_total > 1e-12,
    )
    ratio = np.power(np.clip(exact_ratio, 1e-12, None), match_strength)
    matched["reported_cases_predicted"] = source_total * ratio
    reported_columns = [
        column
        for column in matched.columns
        if column.startswith("reported_cases_") and column.endswith("_predicted")
        and column != "reported_cases_predicted"
    ]
    for column in reported_columns:
        matched[column] = matched[column].to_numpy(dtype=float) * ratio
    return matched


def _strength_suffix(match_strength: float) -> str:
    if np.isclose(match_strength, 1.0):
        return ""
    return f"_m{int(round(match_strength * 100)):03d}"


def _matched_scenario_ids(match_strengths: list[float]) -> set[str]:
    ids: set[str] = set()
    for match_strength in match_strengths:
        suffix = _strength_suffix(float(match_strength))
        ids.add(f"matched_reporting_truth{suffix}")
        ids.add(f"matched_transmission_truth{suffix}")
    return ids


def _parse_match_strengths(values: list[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                strength = float(item)
            except ValueError as exc:
                raise ValueError(f"Invalid notification match strength: {item}") from exc
            if strength < 0.0 or strength > 1.0:
                raise ValueError("notification match strengths must lie in [0, 1]")
            parsed.append(strength)
    if not parsed:
        raise ValueError("At least one notification match strength is required.")
    return sorted(set(parsed), reverse=True)


def _observation_seed_offset(row: pd.Series) -> int:
    scenario = str(row["scenario_id"])
    if scenario.startswith("matched_reporting_truth"):
        scenario = "matched_reporting_truth"
    elif scenario.startswith("matched_transmission_truth"):
        scenario = "matched_transmission_truth"
    return _stable_seed(str(row["iso3"]), str(row["draw_id"]), scenario)


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16, person=b"pertussis-obs-v8").digest(), "little")


def _observation_seed(row: pd.Series, design_id: str, seed: int, replicate: int) -> int:
    return _stable_seed(seed, design_id, _observation_seed_offset(row), replicate)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_matches(path: Path, contract: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()) == contract
    except (ValueError, OSError):
        return False


def _write_contract(path: Path, contract: dict[str, Any]) -> None:
    path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")


def _latent_contract(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        ROOT / args.posterior_draws,
        *[OUTPUTS / f"model2c_stan_data_{iso3}_{args.base_data_suffix}.json" for iso3 in args.countries],
        Path(__file__),
        SRC / "pertussis_global_dynamics/bayes/model2c_scenarios.py",
        SRC / "pertussis_global_dynamics/bayes/model2c_surveillance_design.py",
    ]
    return {
        "version": SURVEILLANCE_DESIGN_VERSION,
        "countries": args.countries,
        "draws_per_country": args.draws_per_country,
        "notification_match_strengths": args.notification_match_strengths,
        "input_sha256": {str(path.relative_to(ROOT)): _sha256_file(path) for path in paths},
    }


def _prediction_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = _read_json(str(payload["data_path"]))
    draw = pd.Series(payload["draw"])
    predictions: dict[str, pd.DataFrame] = {}
    for scenario_id in BASE_SCENARIOS:
        prediction = _predict_monthly(
            data,
            draw,
            scenario_id=scenario_id,
            include_age_predictions=True,
        )
        if not isinstance(prediction, pd.DataFrame):
            raise TypeError("Model 2C surveillance-design prediction did not return a frame.")
        predictions[scenario_id] = prediction
    for match_strength in payload["notification_match_strengths"]:
        suffix = _strength_suffix(float(match_strength))
        predictions[f"matched_reporting_truth{suffix}"] = _match_notification_trajectory(
            predictions["full_model"],
            predictions["no_post_2023_transmission_context"],
            match_strength=float(match_strength),
        )
        predictions[f"matched_transmission_truth{suffix}"] = _match_notification_trajectory(
            predictions["full_model"],
            predictions["no_reporting_testing_shift"],
            match_strength=float(match_strength),
        )
    return [
        trajectory_to_latent_signals(
            prediction,
            data,
            draw,
            iso3=str(payload["iso3"]),
            draw_id=str(payload["draw_id"]),
            draw_rank=int(payload["draw_rank"]),
            scenario_id=scenario_id,
        )
        for scenario_id, prediction in predictions.items()
    ]


def _selected_draws(
    draws: pd.DataFrame,
    countries: list[str],
    draws_per_country: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for iso3 in countries:
        country = draws.loc[draws["iso3"].astype(str).eq(iso3)].reset_index(drop=True)
        if country.empty:
            raise SystemExit(f"No posterior draws are available for {iso3}.")
        count = min(draws_per_country, len(country))
        positions = np.unique(np.linspace(0, len(country) - 1, count).round().astype(int))
        for rank, position in enumerate(positions):
            draw = country.iloc[int(position)].copy()
            chain = draw.get("chain", "")
            draw_number = draw.get("draw", int(position) + 1)
            draw_id = f"{chain}:{draw_number}:{int(position)}"
            selected.append(
                {
                    "iso3": iso3,
                    "draw_id": draw_id,
                    "draw_rank": rank,
                    "draw": draw.to_dict(),
                }
            )
    return selected


def build_latent_cache(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    cache_path = output_dir / "latent_surveillance_signals.csv"
    contract_path = output_dir / "latent_surveillance_signals.metadata.json"
    contract = _latent_contract(args)
    if cache_path.exists() and not args.force_predictions:
        cached = pd.read_csv(cache_path)
        required_scenarios = set(BASE_SCENARIOS).union(
            _matched_scenario_ids(args.notification_match_strengths)
        )
        cached_scenarios = set(cached["scenario_id"].astype(str).unique())
        if (
            cached["surveillance_design_version"].eq(SURVEILLANCE_DESIGN_VERSION).all()
            and required_scenarios.issubset(cached_scenarios)
            and _contract_matches(contract_path, contract)
        ):
            print(f"Using cached latent surveillance signals from {cache_path.relative_to(ROOT)}")
            return cached
        print(
            "Cached latent surveillance signals do not match requested scenario set; rebuilding.",
            flush=True,
        )

    draws_path = ROOT / args.posterior_draws
    draws = pd.read_csv(draws_path)
    selected = _selected_draws(draws, args.countries, args.draws_per_country)
    payloads: list[dict[str, Any]] = []
    for item in selected:
        data_path = OUTPUTS / f"model2c_stan_data_{item['iso3']}_{args.base_data_suffix}.json"
        if not data_path.exists():
            raise SystemExit(f"Missing prepared country data: {data_path}")
        payloads.append(
            {
                **item,
                "data_path": str(data_path),
                "notification_match_strengths": args.notification_match_strengths,
            }
        )

    rows: list[dict[str, Any]] = []
    if args.prediction_workers == 1:
        for index, payload in enumerate(payloads, start=1):
            rows.extend(_prediction_worker(payload))
            if index % 100 == 0 or index == len(payloads):
                print(f"Built {index}/{len(payloads)} latent trajectories", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.prediction_workers) as executor:
            futures = {executor.submit(_prediction_worker, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), start=1):
                payload = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"Latent trajectories failed for {payload['iso3']} "
                        f"draw {payload['draw_id']}: {exc}"
                    ) from exc
                if index % 100 == 0 or index == len(payloads):
                    print(f"Built {index}/{len(payloads)} latent trajectories", flush=True)
    cache = pd.DataFrame(rows).sort_values(["iso3", "draw_rank", "scenario_id"])
    cache.to_csv(cache_path, index=False)
    _write_contract(contract_path, contract)
    return cache


def build_observation_features(
    latent: pd.DataFrame,
    *,
    design_ids: list[str],
    observation_replicates: int,
    seed: int,
    output_dir: Path,
    force: bool,
    legacy_exact_feature_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if legacy_exact_feature_dir is not None:
        raise ValueError("Legacy observation reuse cannot verify the source-preserved measurement contract; regenerate observations.")
    outputs: dict[str, pd.DataFrame] = {}
    for design_id in design_ids:
        design = MEASUREMENT_DESIGNS[design_id]
        path = output_dir / f"observation_features_{design_id}.csv"
        contract_path = path.with_suffix(".metadata.json")
        contract = {
            "version": SURVEILLANCE_DESIGN_VERSION,
            "latent_sha256": (
                _sha256_file(output_dir / "latent_surveillance_signals.csv")
                if (output_dir / "latent_surveillance_signals.csv").exists()
                else hashlib.sha256(latent.to_csv(index=False).encode()).hexdigest()
            ),
            "measurement_design": asdict(design),
            "seed": seed,
            "observation_replicates": observation_replicates,
            "rng": "blake2b-128 keyed by design/background/class/replicate; independent stream children",
            "sampler_sha256": _sha256_file(SRC / "pertussis_global_dynamics/bayes/model2c_surveillance_design.py"),
        }
        if path.exists() and not force:
            frame = pd.read_csv(path)
            required_scenarios = set(latent["scenario_id"].astype(str).unique())
            cached_scenarios = set(frame["scenario_id"].astype(str).unique())
            if (
                frame["surveillance_design_version"].eq(SURVEILLANCE_DESIGN_VERSION).all()
                and required_scenarios.issubset(cached_scenarios)
                and _contract_matches(contract_path, contract)
            ):
                outputs[design_id] = frame
                continue
            print(
                f"Cached observation features for {design_id} do not match requested scenario set; rebuilding.",
                flush=True,
            )
        rows: list[dict[str, Any]] = []
        for _, row in latent.iterrows():
            for replicate in range(1, observation_replicates + 1):
                simulation_seed = _observation_seed(row, design_id, seed, replicate)
                rows.append(
                    simulate_observation_features(
                        row,
                        design=design,
                        seed=simulation_seed,
                        observation_replicate=replicate,
                    )
                )
        frame = pd.DataFrame(rows).sort_values(
            ["iso3", "draw_rank", "scenario_id", "observation_replicate"]
        )
        frame.to_csv(path, index=False)
        _write_contract(contract_path, contract)
        outputs[design_id] = frame
        print(f"Wrote {len(frame)} observation rows for {design_id}", flush=True)
    return outputs


def _evaluate_one_population(
    features: pd.DataFrame,
    *,
    countries: set[str],
    population_id: str,
    design_id: str,
    n_folds: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    package_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    selected = features.loc[features["iso3"].isin(countries)].copy()
    for mechanism_number, (mechanism, structures) in enumerate(structural_contrasts().items()):
        positive_structure, negative_structure = structures
        pair = selected.loc[
            selected["scenario_id"].isin([positive_structure, negative_structure])
        ].copy()
        pair["truth_active"] = pair["scenario_id"].eq(positive_structure).astype(int)
        for package_number, streams in enumerate(surveillance_packages()):
            columns = feature_columns_for_package(pair, streams)
            predictions = cross_validated_predictions(
                pair,
                columns,
                n_folds=n_folds,
            )
            metrics = discrimination_metrics(
                predictions["truth_active"],
                predictions["probability_active"],
            )
            if n_bootstrap > 0:
                lower, upper = nested_country_draw_bootstrap_auc(
                    predictions,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed + mechanism_number * 10_007 + package_number * 101,
                )
            else:
                lower, upper = np.nan, np.nan
            streams_text = "|".join(streams)
            package_rows.append(
                {
                    "mechanism": mechanism,
                    "positive_structure_id": positive_structure,
                    "negative_structure_id": negative_structure,
                    "analysis_population": population_id,
                    "measurement_design": design_id,
                    "package_id": package_id(streams),
                    "package_streams": streams_text,
                    "n_optional_streams": len(streams),
                    "n_features": len(columns),
                    "n_country_draw_groups": predictions["group_id"].nunique(),
                    "n_observations": len(predictions),
                    **metrics,
                    "auc_lower_95": lower,
                    "auc_upper_95": upper,
                    "auc_ui_width": upper - lower if np.isfinite(lower) and np.isfinite(upper) else np.nan,
                    "n_bootstrap": int(n_bootstrap),
                    "interval_method": (
                        "country-first, calibration-ensemble-member-second nested bootstrap"
                        if n_bootstrap > 0 else "not_bootstrapped"
                    ),
                    "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
                }
            )
            for iso3, country in predictions.groupby("iso3", sort=True):
                country_metrics = discrimination_metrics(
                    country["truth_active"],
                    country["probability_active"],
                )
                country_rows.append(
                    {
                        "mechanism": mechanism,
                        "analysis_population": population_id,
                        "measurement_design": design_id,
                        "package_id": package_id(streams),
                        "package_streams": streams_text,
                        "iso3": iso3,
                        "n_country_draw_groups": country["group_id"].nunique(),
                        "n_observations": len(country),
                        **country_metrics,
                        "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
                    }
                )
    package_results = pd.DataFrame(package_rows)
    country_results = pd.DataFrame(country_rows)
    shapley = shapley_stream_gains(package_results)
    shapley["surveillance_design_version"] = SURVEILLANCE_DESIGN_VERSION
    return package_results, country_results, shapley


def evaluate_designs(
    feature_tables: dict[str, pd.DataFrame],
    *,
    countries: list[str],
    n_folds: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    package_parts: list[pd.DataFrame] = []
    country_parts: list[pd.DataFrame] = []
    shapley_parts: list[pd.DataFrame] = []
    populations = {
        "all_nine": set(countries),
        "converged_six": set(countries).intersection(PASSING_PRIMARY_COUNTRIES),
    }
    for design_number, (design_id, features) in enumerate(feature_tables.items()):
        for population_number, (population_id, population_countries) in enumerate(populations.items()):
            package, country, shapley = _evaluate_one_population(
                features,
                countries=population_countries,
                population_id=population_id,
                design_id=design_id,
                n_folds=n_folds,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=seed + design_number * 1_000_003 + population_number * 100_003,
            )
            package_parts.append(package)
            country_parts.append(country)
            shapley_parts.append(shapley)
            print(f"Evaluated {design_id}, {population_id}", flush=True)
    package_results = pd.concat(package_parts, ignore_index=True)
    country_results = pd.concat(country_parts, ignore_index=True)
    shapley_results = pd.concat(shapley_parts, ignore_index=True)

    recommendation_parts: list[pd.DataFrame] = []
    for population_id, country_threshold in (("all_nine", 6), ("converged_six", 4)):
        recommendation_parts.append(
            minimum_successful_packages(
                package_results.loc[package_results["analysis_population"].eq(population_id)],
                country_results.loc[country_results["analysis_population"].eq(population_id)],
                country_count_threshold=country_threshold,
            )
        )
    recommendations = pd.concat(recommendation_parts, ignore_index=True)
    recommendations["surveillance_design_version"] = SURVEILLANCE_DESIGN_VERSION
    return package_results, country_results, shapley_results, recommendations


def write_report(
    output_dir: Path,
    package_results: pd.DataFrame,
    country_results: pd.DataFrame,
    shapley_results: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    main = package_results.loc[
        package_results["analysis_population"].eq("all_nine")
        & package_results["measurement_design"].eq("main")
    ].copy()
    current = main.loc[main["package_id"].eq("notifications+age")].set_index("mechanism")
    complete = main.loc[
        main["package_id"].eq("notifications+age+testing+severe+serology")
    ].set_index("mechanism")
    recommended = recommendations.loc[
        recommendations["analysis_population"].eq("all_nine")
        & recommendations["measurement_design"].eq("main")
    ].set_index("mechanism")
    top_stream = (
        shapley_results.loc[
            shapley_results["analysis_population"].eq("all_nine")
            & shapley_results["measurement_design"].eq("main")
        ]
        .sort_values(["mechanism", "shapley_auc_gain"], ascending=[True, False])
        .groupby("mechanism")
        .first()
    )
    lines = [
        "# Model 2C surveillance-design results",
        "",
        f"Analysis version: `{SURVEILLANCE_DESIGN_VERSION}`.",
        "",
        "The experiment uses paired complete and one-process-neutralised trajectories,",
        "group-held-out posterior draws, and the 16 prespecified monitoring packages.",
        "AUC measures structural distinguishability conditional on the mechanism library;",
        "it does not identify the historical cause of a national resurgence.",
        "",
        "## Main results",
        "",
        "| Mechanism contrast | Current package AUC | Complete package AUC | Largest average marginal gain | Recommendation |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for mechanism in structural_contrasts():
        current_auc = float(current.loc[mechanism, "auc"])
        complete_auc = float(complete.loc[mechanism, "auc"])
        stream = str(top_stream.loc[mechanism, "stream"])
        gain = float(top_stream.loc[mechanism, "shapley_auc_gain"])
        recommendation = recommended.loc[mechanism]
        lines.append(
            f"| {mechanism} | {current_auc:.3f} | {complete_auc:.3f} | "
            f"{stream} ({gain:+.3f}) | {recommendation['recommended_package_id']} "
            f"({recommendation['recommendation_status']}) |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "The prespecified operational success rule requires pooled AUC at least 0.80",
        "and country-specific AUC at least 0.70 in six of nine backgrounds. A package",
        "that fails this rule should not be used to rank that mechanism from routine",
        "surveillance alone.",
        "",
        "## Reproducibility",
        "",
        "- `latent_surveillance_signals.csv` contains the deterministic model signals.",
        "- `observation_features_*.csv` contains the simulated monitoring observations.",
        "- `package_discrimination.csv` and `country_discrimination.csv` contain held-out results.",
        "- `stream_shapley_gains.csv` contains average marginal AUC gains.",
        "- `minimum_package_recommendations.csv` applies the prespecified success rule.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model 2C surveillance-design simulations.")
    parser.add_argument("--tag", default=DEFAULT_SURVEILLANCE_TAG)
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--posterior-draws", default="outputs/model2c_posterior_draws.csv")
    parser.add_argument("--base-data-suffix", default="full")
    parser.add_argument("--draws-per-country", type=int, default=80)
    parser.add_argument("--observation-replicates", type=int, default=3)
    parser.add_argument("--measurement-designs", nargs="+", default=["low_precision", "main", "high_precision"])
    parser.add_argument("--prediction-workers", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=SURVEILLANCE_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--notification-match-strengths",
        nargs="*",
        default=["1.00"],
        help=(
            "Notification match strengths in [0, 1]. Accepts either space-separated "
            "values or comma-separated lists, e.g. 1.00,0.95,0.90,0.80."
        ),
    )
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--force-observations", action="store_true")
    parser.add_argument(
        "--legacy-exact-feature-dir",
        type=Path,
        default=None,
        help=(
            "Deprecated: legacy observation reuse is rejected because it cannot verify "
            "the current source-preserved measurement contract."
        ),
    )
    args = parser.parse_args(argv)
    args.tag = safe_recovery_tag(args.tag)
    args.countries = [str(value).upper() for value in args.countries]
    unknown = sorted(set(args.measurement_designs).difference(MEASUREMENT_DESIGNS))
    if unknown:
        parser.error(f"Unknown measurement designs: {', '.join(unknown)}")
    if args.draws_per_country < args.folds:
        parser.error("--draws-per-country must be at least the number of folds")
    if args.observation_replicates < 1:
        parser.error("--observation-replicates must be at least 1")
    if args.prediction_workers < 1:
        parser.error("--prediction-workers must be at least 1")
    try:
        args.notification_match_strengths = _parse_match_strengths(args.notification_match_strengths)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = OUTPUTS / "model2c_surveillance_design" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    latent = build_latent_cache(args, output_dir)
    feature_tables = build_observation_features(
        latent,
        design_ids=args.measurement_designs,
        observation_replicates=args.observation_replicates,
        seed=args.seed,
        output_dir=output_dir,
        force=args.force_observations,
        legacy_exact_feature_dir=args.legacy_exact_feature_dir,
    )
    package, country, shapley, recommendations = evaluate_designs(
        feature_tables,
        countries=args.countries,
        n_folds=args.folds,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    package.to_csv(output_dir / "package_discrimination.csv", index=False)
    country.to_csv(output_dir / "country_discrimination.csv", index=False)
    shapley.to_csv(output_dir / "stream_shapley_gains.csv", index=False)
    recommendations.to_csv(output_dir / "minimum_package_recommendations.csv", index=False)
    write_report(output_dir, package, country, shapley, recommendations)
    print(f"Wrote surveillance-design experiment to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
