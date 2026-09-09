#!/usr/bin/env python3
"""Run non-Bayesian robustness checks for the global reported-case screen.

This script deliberately writes to a new output namespace.  It does not replace
the primary resurgence table.  Its intervals are conditional wild-bootstrap
intervals for the fitted periodic baseline; they are not uncertainty intervals
for infection burden or surveillance-process changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "clean" / "cases_country_age_time.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "global_baseline_robustness"
DEFAULT_REPORT = ROOT / "manuscript" / "reports" / "global_baseline_robustness.md"
LOW_BASELINE_THRESHOLD = 0.1


def build_preferred_annual_series(cases: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the primary screen's annual aggregation and source preference."""
    required = {"iso3", "country", "year", "time_resolution", "age_group", "cases_total", "population"}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"Cases input is missing columns: {sorted(missing)}")

    all_cases = cases.loc[
        cases["age_group"].eq("all_ages"),
        ["iso3", "country", "year", "time_resolution", "cases_total", "population"],
    ].copy()
    all_cases["year"] = pd.to_numeric(all_cases["year"], errors="coerce").astype("Int64")
    all_cases["cases_total"] = pd.to_numeric(all_cases["cases_total"], errors="coerce")
    all_cases["population"] = pd.to_numeric(all_cases["population"], errors="coerce")

    annual_population = (
        all_cases.loc[all_cases["time_resolution"].eq("annual"), ["iso3", "year", "population"]]
        .dropna(subset=["iso3", "year", "population"])
        .groupby(["iso3", "year"], as_index=False)
        .agg(annual_population=("population", "mean"))
    )
    all_cases = all_cases.merge(annual_population, on=["iso3", "year"], how="left")
    all_cases["population"] = all_cases["population"].fillna(all_cases["annual_population"])
    all_cases = all_cases.drop(columns="annual_population")

    time_resolved = all_cases.loc[~all_cases["time_resolution"].eq("annual")]
    annual = all_cases.loc[all_cases["time_resolution"].eq("annual")]
    annual_from_time = time_resolved.groupby(["iso3", "country", "year"], as_index=False).agg(
        cases_total=("cases_total", "sum"), population=("population", "mean"), source_rows=("cases_total", "size")
    )
    annual_from_time["preferred_source"] = "time_resolved"
    annual_who = annual.groupby(["iso3", "country", "year"], as_index=False).agg(
        cases_total=("cases_total", "sum"), population=("population", "mean"), source_rows=("cases_total", "size")
    )
    annual_who["preferred_source"] = "annual"
    source = pd.concat([annual_from_time, annual_who], ignore_index=True, sort=False)
    source["source_priority"] = np.where(source["preferred_source"].eq("time_resolved"), 0, 1)
    source = source.sort_values(["iso3", "year", "source_priority"]).drop_duplicates(["iso3", "year"], keep="first")
    source["incidence_per_100k"] = source["cases_total"] / source["population"] * 100_000.0
    return source.sort_values(["iso3", "year"]).reset_index(drop=True)


def _design(years: np.ndarray, centre: float, n_columns: int | None = None) -> np.ndarray:
    full = np.column_stack(
        [
            np.ones(len(years)),
            years - centre,
            np.sin(2 * np.pi * years / 4.0),
            np.cos(2 * np.pi * years / 4.0),
            np.sin(2 * np.pi * years / 5.0),
            np.cos(2 * np.pi * years / 5.0),
        ]
    )
    return full if n_columns is None else full[:, :n_columns]


def _stable_seed(base_seed: int, iso3: str, prediction_years: list[int]) -> int:
    token = f"{base_seed}|{iso3}|{','.join(map(str, prediction_years))}"
    value = 0
    for index, char in enumerate(token):
        value = (value + (index + 1) * ord(char)) % (2**32 - 1)
    return value


def fit_periodic_baseline(
    group: pd.DataFrame,
    prediction_years: list[int],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Fit the primary periodic model and an HC2 wild residual bootstrap."""
    fit = group.loc[group["year"].between(2010, 2019)].dropna(subset=["incidence_per_100k", "year"]).copy()
    fit_window = "2010-2019"
    if len(fit) < 6:
        fit = group.loc[group["year"].between(2000, 2019)].dropna(subset=["incidence_per_100k", "year"]).copy()
        fit_window = "2000-2019 fallback"
    base = {
        "periodic_fit_window": fit_window,
        "periodic_fit_n_years": len(fit),
        "periodic_fit_n_parameters": 0,
        "periodic_expected_incidence_per_100k": np.nan,
        "periodic_expected_lcl": np.nan,
        "periodic_expected_ucl": np.nan,
        "periodic_bootstrap_successes": 0,
        "periodic_bootstrap_status": "not_evaluable",
    }
    if fit.empty or not prediction_years:
        return base
    if fit["incidence_per_100k"].clip(lower=0).sum() <= 0:
        return {
            **base,
            "periodic_expected_incidence_per_100k": 0.0,
            "periodic_expected_lcl": 0.0,
            "periodic_expected_ucl": 0.0,
            "periodic_bootstrap_status": "all_zero_baseline_no_model_uncertainty",
        }

    years = fit["year"].to_numpy(dtype=float)
    centre = float(years.mean())
    y = np.log1p(fit["incidence_per_100k"].clip(lower=0).to_numpy(dtype=float))
    n_columns = 6 if len(fit) >= 6 else min(len(fit), 4)
    x = _design(years, centre, n_columns)
    pinv = np.linalg.pinv(x)
    coef = pinv @ y
    prediction = np.asarray(prediction_years, dtype=float)
    xp = _design(prediction, centre, n_columns)
    cap = float(np.nanpercentile(fit["incidence_per_100k"], 95) * 4 + 1)

    def inverse_mean(log_values: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(np.clip(log_values, -20.0, 20.0)), 0.0, cap).mean(axis=-1)

    point = float(inverse_mean(xp @ coef))
    result = {
        **base,
        "periodic_fit_n_parameters": int(n_columns),
        "periodic_expected_incidence_per_100k": point,
    }
    fitted = x @ coef
    residual = y - fitted
    hat = np.sum(x * pinv.T, axis=1)
    if len(fit) <= n_columns or np.allclose(residual, 0.0, atol=1e-12):
        return {
            **result,
            "periodic_expected_lcl": point,
            "periodic_expected_ucl": point,
            "periodic_bootstrap_status": "not_estimable_saturated_or_zero_residual_fit",
        }

    adjusted = residual / np.sqrt(np.clip(1.0 - hat, 1e-8, None))
    rng = np.random.default_rng(seed)
    multipliers = rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, len(fit)))
    y_star = fitted[None, :] + multipliers * adjusted[None, :]
    coef_star = y_star @ pinv.T
    expected_star = inverse_mean(coef_star @ xp.T)
    finite = expected_star[np.isfinite(expected_star)]
    if not len(finite):
        return {**result, "periodic_bootstrap_status": "bootstrap_failed"}
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        **result,
        "periodic_expected_lcl": float(lower),
        "periodic_expected_ucl": float(upper),
        "periodic_bootstrap_successes": int(len(finite)),
        "periodic_bootstrap_status": "hc2_rademacher_wild_residual_bootstrap",
    }


def analyse_configuration(
    annual: pd.DataFrame,
    *,
    requested_cutoff: int,
    provisional_policy: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if provisional_policy not in {"include", "exclude_cutoff"}:
        raise ValueError("provisional_policy must be include or exclude_cutoff")
    effective_cutoff = requested_cutoff - int(provisional_policy == "exclude_cutoff")
    prediction_years = list(range(2023, effective_cutoff + 1))
    if not prediction_years:
        raise ValueError("The requested configuration contains no post-pandemic years")
    baseline_required = set(range(2010, 2020))
    post_required = set(prediction_years)

    rows: list[dict[str, object]] = []
    for (iso3, country), group in annual.groupby(["iso3", "country"], dropna=False):
        group = group.loc[group["year"].le(requested_cutoff)].copy()
        baseline = group.loc[group["year"].between(2010, 2019)]
        post = group.loc[group["year"].isin(prediction_years)]
        baseline_inc = baseline["incidence_per_100k"].dropna()
        post_inc = post["incidence_per_100k"].dropna()
        descriptive_mean = float(baseline_inc.mean()) if len(baseline_inc) else np.nan
        descriptive_median = float(baseline_inc.median()) if len(baseline_inc) else np.nan
        recent = baseline.loc[baseline["year"].between(2015, 2019), "incidence_per_100k"].dropna()
        recent_mean = float(recent.mean()) if len(recent) else np.nan
        post_mean = float(post_inc.mean()) if len(post_inc) else np.nan
        observed_prediction_years = sorted(post.loc[post["incidence_per_100k"].notna(), "year"].astype(int).unique())
        periodic = fit_periodic_baseline(
            group,
            observed_prediction_years,
            n_bootstrap=n_bootstrap,
            seed=_stable_seed(seed, str(iso3), observed_prediction_years),
        )
        expected = float(periodic["periodic_expected_incidence_per_100k"])
        lcl = float(periodic["periodic_expected_lcl"])
        ucl = float(periodic["periodic_expected_ucl"])
        raw_ratio = post_mean / expected if np.isfinite(post_mean) and expected > 0 else np.nan
        ratio_lcl = post_mean / ucl if np.isfinite(post_mean) and ucl > 0 else np.nan
        ratio_ucl = (
            post_mean / lcl
            if np.isfinite(post_mean) and lcl > 0
            else np.inf
            if np.isfinite(post_mean) and post_mean > 0 and np.isfinite(lcl) and lcl <= 0
            else np.nan
        )
        baseline_years = set(baseline.loc[baseline["incidence_per_100k"].notna(), "year"].astype(int))
        post_years = set(post.loc[post["incidence_per_100k"].notna(), "year"].astype(int))
        required_rows = group.loc[group["year"].isin(baseline_required | post_required)]
        population_complete = bool(
            required_rows["population"].notna().all() and (required_rows["population"] > 0).all()
        )
        balanced = baseline_required.issubset(baseline_years) and post_required.issubset(post_years) and population_complete
        stable_source = bool(balanced and required_rows["preferred_source"].nunique(dropna=True) == 1)
        low = bool(np.isfinite(expected) and expected < LOW_BASELINE_THRESHOLD)
        absolute_change = post_mean - expected if np.isfinite(post_mean) and np.isfinite(expected) else np.nan
        mean_post_cases = float(post["cases_total"].mean()) if len(post) else np.nan
        guarded = bool(
            np.isfinite(raw_ratio)
            and raw_ratio >= 2.0
            and np.isfinite(absolute_change)
            and absolute_change >= 1.0
            and np.isfinite(mean_post_cases)
            and mean_post_cases >= 10.0
            and not low
        )
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "requested_data_cutoff": requested_cutoff,
                "provisional_year_policy": provisional_policy,
                "effective_post_end_year": effective_cutoff,
                "post_years_estimand": f"2023-{effective_cutoff} available-year arithmetic mean incidence",
                "baseline_observed_years_2010_2019": len(baseline_years),
                "post_observed_years": len(post_years),
                "balanced_panel": balanced,
                "stable_reporter_panel": stable_source,
                "preferred_source_classes": ";".join(sorted(required_rows["preferred_source"].dropna().astype(str).unique())),
                "baseline_zero_incidence_years": int((baseline_inc == 0).sum()),
                "post_mean_cases": mean_post_cases,
                "post_mean_incidence_per_100k": post_mean,
                **periodic,
                "periodic_resurgence_ratio": raw_ratio,
                "periodic_resurgence_ratio_lcl": ratio_lcl,
                "periodic_resurgence_ratio_ucl": ratio_ucl,
                "periodic_resurgence_ratio_ucl_unbounded": bool(np.isinf(ratio_ucl)),
                "periodic_ratio_with_0_1_floor": post_mean / max(expected, LOW_BASELINE_THRESHOLD)
                if np.isfinite(post_mean) and np.isfinite(expected)
                else np.nan,
                "periodic_ratio_reportable_above_0_1": raw_ratio if not low else np.nan,
                "low_periodic_baseline_below_0_1_per_100k": low,
                "absolute_post_minus_periodic_incidence_per_100k": absolute_change,
                "guarded_resurgence_signal": guarded,
                "descriptive_mean_baseline_incidence_per_100k": descriptive_mean,
                "descriptive_mean_resurgence_ratio": post_mean / descriptive_mean
                if np.isfinite(post_mean) and descriptive_mean > 0
                else np.nan,
                "descriptive_median_baseline_incidence_per_100k": descriptive_median,
                "descriptive_median_resurgence_ratio": post_mean / descriptive_median
                if np.isfinite(post_mean) and descriptive_median > 0
                else np.nan,
                "recent_2015_2019_mean_baseline_incidence_per_100k": recent_mean,
                "recent_2015_2019_mean_resurgence_ratio": post_mean / recent_mean
                if np.isfinite(post_mean) and recent_mean > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarise_panels(country_results: pd.DataFrame) -> pd.DataFrame:
    estimators = {
        "periodic_raw": ("periodic_expected_incidence_per_100k", "periodic_resurgence_ratio"),
        "periodic_floor_0_1": ("periodic_expected_incidence_per_100k", "periodic_ratio_with_0_1_floor"),
        "descriptive_mean_2010_2019": (
            "descriptive_mean_baseline_incidence_per_100k",
            "descriptive_mean_resurgence_ratio",
        ),
        "descriptive_median_2010_2019": (
            "descriptive_median_baseline_incidence_per_100k",
            "descriptive_median_resurgence_ratio",
        ),
        "descriptive_mean_2015_2019": (
            "recent_2015_2019_mean_baseline_incidence_per_100k",
            "recent_2015_2019_mean_resurgence_ratio",
        ),
    }
    panels = {
        "all_evaluable": lambda x: x["post_mean_incidence_per_100k"].notna(),
        "balanced": lambda x: x["balanced_panel"].astype(bool),
        "stable_reporter": lambda x: x["stable_reporter_panel"].astype(bool),
    }
    rows: list[dict[str, object]] = []
    config_cols = ["requested_data_cutoff", "provisional_year_policy", "effective_post_end_year"]
    for config, frame in country_results.groupby(config_cols, dropna=False):
        for panel_name, select in panels.items():
            panel = frame.loc[select(frame)].copy()
            for estimator, (baseline_col, ratio_col) in estimators.items():
                ratios = pd.to_numeric(panel[ratio_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                baseline = pd.to_numeric(panel[baseline_col], errors="coerce")
                post = pd.to_numeric(panel["post_mean_incidence_per_100k"], errors="coerce")
                weights = pd.to_numeric(panel["post_observed_years"], errors="coerce").clip(lower=1)
                valid = baseline.notna() & post.notna()
                weighted_ratio = (
                    float(np.average(post[valid], weights=weights[valid]))
                    / float(np.average(baseline[valid].clip(lower=LOW_BASELINE_THRESHOLD if estimator == "periodic_floor_0_1" else 0), weights=weights[valid]))
                    if valid.any() and np.average(baseline[valid].clip(lower=LOW_BASELINE_THRESHOLD if estimator == "periodic_floor_0_1" else 0), weights=weights[valid]) > 0
                    else np.nan
                )
                rows.append(
                    {
                        "requested_data_cutoff": config[0],
                        "provisional_year_policy": config[1],
                        "effective_post_end_year": config[2],
                        "panel": panel_name,
                        "baseline_estimator": estimator,
                        "n_countries_in_panel": len(panel),
                        "n_evaluable_ratios": len(ratios),
                        "median_country_ratio": float(ratios.median()) if len(ratios) else np.nan,
                        "iqr_country_ratio_lower": float(ratios.quantile(0.25)) if len(ratios) else np.nan,
                        "iqr_country_ratio_upper": float(ratios.quantile(0.75)) if len(ratios) else np.nan,
                        "available_country_year_weighted_mean_incidence_ratio": weighted_ratio,
                        "n_guarded_resurgence_signals": int(panel["guarded_resurgence_signal"].sum()),
                        "n_low_periodic_baselines": int(panel["low_periodic_baseline_below_0_1_per_100k"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    concordance: pd.DataFrame,
    path: Path,
    *,
    n_bootstrap: int,
    seed: int,
    input_path: Path,
) -> None:
    primary = results.loc[
        results["requested_data_cutoff"].eq(2025) & results["provisional_year_policy"].eq("include")
    ]
    primary_summary = summary.loc[
        summary["requested_data_cutoff"].eq(2025)
        & summary["provisional_year_policy"].eq("include")
        & summary["baseline_estimator"].isin(["periodic_raw", "descriptive_mean_2010_2019"])
    ]
    lines = [
        "# Global reported-case baseline robustness analysis",
        "",
        "## Scope and estimands",
        "",
        f"- Frozen input: `{input_path.relative_to(ROOT)}`.",
        "- Country numerator: arithmetic mean of available annual reported-case incidence from 2023 through the effective cutoff.",
        "- Periodic comparator: mean predicted incidence for the same post-pandemic years from the primary `log1p` model (intercept, centred trend, 4-year and 5-year sine/cosine terms), fitted to 2010–2019; 2000–2019 is used only when fewer than six 2010–2019 observations are present.",
        f"- Baseline uncertainty: {n_bootstrap:,} HC2 leverage-adjusted Rademacher wild residual replicates (seed {seed}). Ratio limits invert the baseline limits while holding the observed post-pandemic numerator fixed.",
        "- Alternative comparators: 2010–2019 arithmetic mean, 2010–2019 median, and 2015–2019 arithmetic mean incidence.",
        "- Balanced panel: complete incidence and positive population denominators in every year of 2010–2019 and every post-pandemic estimand year.",
        "- Stable-reporter panel: balanced-panel membership plus one preferred source class (annual or time-resolved) over all required years.",
        "- Low-baseline safeguard: flag expected periodic incidence below 0.1 per 100,000; retain the raw ratio for audit, also report a denominator-floored ratio and suppress the reportable raw ratio.",
        "- Guarded signal: raw ratio at least 2, absolute increase at least 1 per 100,000, mean annual post-pandemic cases at least 10, and periodic baseline at least 0.1 per 100,000.",
        "",
        "## 2025-cutoff results",
        "",
        f"The primary 2023–2025 available-year screen had post-period observations for {primary['post_mean_incidence_per_100k'].notna().sum()} countries and an estimable positive periodic ratio for {primary['periodic_resurgence_ratio'].notna().sum()}; {primary['balanced_panel'].sum()} met the balanced definition and {primary['stable_reporter_panel'].sum()} met the stable-reporter definition. {primary['low_periodic_baseline_below_0_1_per_100k'].sum()} had periodic expected incidence below 0.1 per 100,000, and {primary['guarded_resurgence_signal'].sum()} met the guarded signal definition.",
        "",
        f"Of the periodic ratio intervals, {primary['periodic_resurgence_ratio_ucl_unbounded'].sum()} had an unbounded upper limit because the bootstrap baseline lower limit reached zero. The robustness implementation reproduced all {(concordance['concordance_status'] == 'pass').sum()} comparable primary periodic point estimates (maximum absolute difference {concordance.loc[concordance['concordance_status'].eq('pass'), 'expected_absolute_difference'].max():.3g}); the remaining {(concordance['concordance_status'] != 'pass').sum()} primary rows used a descriptive fallback or lacked a comparable positive periodic estimate.",
        "",
        "Panel summaries (country-level median ratios):",
        "",
        "| Panel | Baseline | N ratios | Median ratio | IQR | Guarded signals |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_summary.itertuples(index=False):
        lines.append(
            f"| {row.panel} | {row.baseline_estimator} | {row.n_evaluable_ratios} | {row.median_country_ratio:.2f} | {row.iqr_country_ratio_lower:.2f}–{row.iqr_country_ratio_upper:.2f} | {row.n_guarded_resurgence_signals} |"
        )
    lines += [
        "",
        "## Cutoff and provisional-year sensitivity",
        "",
        "`exclude_cutoff` is a conservative leave-last-calendar-year-out analysis. The frozen source has no row-level provisional-status field, so this is not a claim that all excluded values are provisional.",
        "",
        "| Requested cutoff | Policy | Effective end | Panel | Baseline | N | Median ratio |",
        "|---:|---|---:|---|---|---:|---:|",
    ]
    compact = summary.loc[
        summary["panel"].isin(["all_evaluable", "stable_reporter"])
        & summary["baseline_estimator"].isin(["periodic_raw", "descriptive_mean_2010_2019"])
    ]
    for row in compact.itertuples(index=False):
        lines.append(
            f"| {row.requested_data_cutoff} | {row.provisional_year_policy} | {row.effective_post_end_year} | {row.panel} | {row.baseline_estimator} | {row.n_evaluable_ratios} | {row.median_country_ratio:.2f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "These are notification-based descriptive estimands. Wild-bootstrap limits quantify conditional regression residual uncertainty only; they do not incorporate population uncertainty, stochastic case generation, source revision, testing intensity, case-definition changes, missing-not-at-random reporting, or model-selection uncertainty. A saturated or zero-residual periodic fit yields a flagged non-estimable interval rather than artificial precision. Equal-country panel summaries are not global burden estimates.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-cutoffs", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument(
        "--provisional-policies",
        choices=["include", "exclude_cutoff"],
        nargs="+",
        default=["include", "exclude_cutoff"],
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def build_primary_concordance(results: pd.DataFrame, primary_path: Path) -> pd.DataFrame:
    """Audit that the like-for-like 2025 point estimates reproduce the primary screen."""
    primary = pd.read_csv(primary_path)
    robustness = results.loc[
        results["requested_data_cutoff"].eq(2025) & results["provisional_year_policy"].eq("include"),
        ["iso3", "periodic_expected_incidence_per_100k", "periodic_resurgence_ratio"],
    ].rename(
        columns={
            "periodic_expected_incidence_per_100k": "robustness_periodic_expected",
            "periodic_resurgence_ratio": "robustness_periodic_ratio",
        }
    )
    audit = primary.loc[
        :, ["iso3", "periodic_expected_postpandemic_incidence_per_100k", "resurgence_ratio", "baseline_method"]
    ].merge(robustness, on="iso3", how="left")
    audit["expected_absolute_difference"] = (
        audit["robustness_periodic_expected"] - audit["periodic_expected_postpandemic_incidence_per_100k"]
    ).abs()
    comparable = pd.to_numeric(
        audit["periodic_expected_postpandemic_incidence_per_100k"], errors="coerce"
    ).gt(0)
    audit["concordance_status"] = np.where(
        ~comparable,
        "not_comparable_primary_used_descriptive_fallback_or_missing",
        np.where(audit["expected_absolute_difference"].le(1e-10), "pass", "fail"),
    )
    return audit


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    cases = pd.read_csv(args.input, low_memory=False)
    annual = build_preferred_annual_series(cases)
    configurations = []
    for cutoff in sorted(set(args.data_cutoffs)):
        for policy in dict.fromkeys(args.provisional_policies):
            if policy == "exclude_cutoff" and cutoff <= 2023:
                continue
            configurations.append((cutoff, policy))
    result_frames = [
        analyse_configuration(
            annual,
            requested_cutoff=cutoff,
            provisional_policy=policy,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        for cutoff, policy in configurations
    ]
    results = pd.concat(result_frames, ignore_index=True)
    summary = summarise_panels(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    country_path = args.output_dir / "country_baseline_robustness.csv"
    summary_path = args.output_dir / "panel_cutoff_sensitivity_summary.csv"
    concordance_path = args.output_dir / "primary_point_estimate_concordance.csv"
    metadata_path = args.output_dir / "run_metadata.json"
    results.to_csv(country_path, index=False)
    summary.to_csv(summary_path, index=False)
    primary_path = ROOT / "outputs" / "resurgence_index_by_country.csv"
    concordance = build_primary_concordance(results, primary_path)
    concordance.to_csv(concordance_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "input": str(args.input.relative_to(ROOT)),
                "data_cutoffs": sorted(set(args.data_cutoffs)),
                "provisional_policies": list(dict.fromkeys(args.provisional_policies)),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "low_baseline_threshold_per_100k": LOW_BASELINE_THRESHOLD,
                "country_rows": len(results),
                "annual_preferred_rows": len(annual),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        results,
        summary,
        concordance,
        args.report,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        input_path=args.input,
    )
    print(f"Wrote {country_path.relative_to(ROOT)} ({len(results)} rows)")
    print(f"Wrote {summary_path.relative_to(ROOT)} ({len(summary)} rows)")
    print(f"Wrote {concordance_path.relative_to(ROOT)} ({len(concordance)} rows)")
    print(f"Wrote {metadata_path.relative_to(ROOT)}")
    print(f"Wrote {args.report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
