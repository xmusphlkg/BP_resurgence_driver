#!/usr/bin/env python3
"""Extract Figure 1 context from frozen outputs without fitting any model.

Country comparisons retain the robustness table's unregularised ratios and
one common post-period numerator. Monthly calibration, comparator sensitivity
and conditional periodic-baseline intervals describe the nine-model library.
Separate temporal prediction checks, vaccination histories, geographic
coverage and selection summaries remain supporting data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INCOME_DIR = Path("outputs/model2c_surveillance_design/monitoring_packages_v4_source_preserved_20260908")
INPUTS = {
    "annual_regional_rates": Path("manuscript/figure_data/figure_1_global_trends.csv"),
    "country_baseline_robustness": Path("outputs/global_baseline_robustness/country_baseline_robustness.csv"),
    "selection_audit": Path("outputs/model2c_country_selection_audit.csv"),
    "income_summary": INCOME_DIR / "income_group_readiness_summary.csv",
    "income_setting_audit": INCOME_DIR / "income_group_setting_audit.csv",
    "monthly_calibration": Path("manuscript/figure_data/figure_3_model2c_monthly_posterior_predictive.csv"),
    "temporal_monthly": Path("outputs/model2c_temporal_validation/monthly_temporal_validation.csv"),
    "temporal_summary": Path("outputs/model2c_temporal_validation/temporal_validation_summary.csv"),
    "vaccination_coverage": Path("data/clean/vaccine_coverage.csv"),
}
REGIONS = {
    "AFRO": "Africa", "AMRO": "Americas", "EMRO": "Eastern Mediterranean",
    "EURO": "Europe", "SEARO": "South-East Asia", "WPRO": "Western Pacific",
}
PRIMARY_ISO3 = {"AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"}
INCIDENCE_UNIT = "reported cases per 100000 population per year"
BASELINES = {
    "mean_2010_2019": (
        "2010–19 mean", "descriptive_mean_baseline_incidence_per_100k",
        "descriptive_mean_resurgence_ratio",
    ),
    "mean_2015_2019": (
        "2015–19 mean", "recent_2015_2019_mean_baseline_incidence_per_100k",
        "recent_2015_2019_mean_resurgence_ratio",
    ),
    "periodic": (
        "Periodic comparator", "periodic_expected_incidence_per_100k",
        "periodic_resurgence_ratio",
    ),
}


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def flags(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype("string").str.lower().map({"true": True, "false": False})
    if values.isna().any():
        raise ValueError(f"{column}: missing or non-boolean selection flag")
    return values.astype(bool)


def build_regional_context(trends: pd.DataFrame) -> pd.DataFrame:
    require_columns(trends, ["series_type", "who_region", "year", "reported_cases", "population", "incidence_per_100k"], "regional rates")
    frame = trends.loc[trends.series_type.eq("who_region")].copy()
    frame["who_region"] = frame.who_region.str.removeprefix("WHO_")
    frame = frame.loc[frame.year.isin([*range(2015, 2020), 2023, 2024, 2025])]
    if frame.duplicated(["who_region", "year"]).any() or set(frame.who_region) != set(REGIONS):
        raise ValueError("Regional input must have unique region-years for six WHO regions")
    expected_rates = frame.reported_cases / frame.population * 100_000
    if (frame.population <= 0).any() or not np.allclose(frame.incidence_per_100k, expected_rates, rtol=1e-10, atol=1e-10):
        raise ValueError("Regional incidence unit must be reported cases per 100000 population")
    rows = []
    for region, label in REGIONS.items():
        group = frame.loc[frame.who_region.eq(region)]
        pre = group.loc[group.year.between(2015, 2019)].sort_values("year")
        post = group.loc[group.year.between(2023, 2025)].sort_values("year")
        if len(pre) != 5 or len(post) != 3 or group.incidence_per_100k.isna().any():
            raise ValueError(f"{region}: expected five baseline and three post-period annual rates")
        baseline, numerator = float(pre.incidence_per_100k.mean()), float(post.incidence_per_100k.mean())
        rows.append({
            "who_region": region, "region_label": label,
            "baseline_incidence": baseline, "post_incidence": numerator,
            "ratio": (numerator + 0.01) / (baseline + 0.01),
            "ratio_offset_per_100k": 0.01,
            "baseline_years": ";".join(pre.year.astype(int).astype(str)),
            "post_years": ";".join(post.year.astype(int).astype(str)),
            "n_baseline_years": len(pre), "n_post_years": len(post),
            "incidence_unit": INCIDENCE_UNIT,
        })
    return pd.DataFrame(rows)


def build_baseline_context(robustness: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    require_columns(selection, ["iso3", "selected_for_current_model2c"], "selection")
    selected = set(selection.loc[flags(selection, "selected_for_current_model2c"), "iso3"])
    if selected != PRIMARY_ISO3:
        raise ValueError("Figure 1 baseline comparison requires the frozen nine primary backgrounds")
    require_columns(robustness, [
        "iso3", "country", "requested_data_cutoff", "provisional_year_policy",
        "post_mean_incidence_per_100k", "post_observed_years", "post_years_estimand",
        "periodic_fit_window", "periodic_fit_n_years", "preferred_source_classes",
        "periodic_resurgence_ratio_lcl", "periodic_resurgence_ratio_ucl",
        "periodic_resurgence_ratio_ucl_unbounded", "periodic_bootstrap_status",
        "periodic_bootstrap_successes",
        *[column for values in BASELINES.values() for column in values[1:]],
    ], "baseline robustness")
    frame = robustness.loc[
        robustness.requested_data_cutoff.eq(2025)
        & robustness.provisional_year_policy.eq("include") & robustness.iso3.isin(selected)
    ].sort_values("iso3")
    if len(frame) != 9 or frame.iso3.duplicated().any() or set(frame.iso3) != selected:
        raise ValueError("Expected one cutoff-2025/include robustness row per primary country")
    if not frame.post_observed_years.eq(3).all():
        raise ValueError("Missing post years: exact available years need verification before labelling")
    lower = frame.periodic_resurgence_ratio_lcl.to_numpy(dtype=float)
    upper = frame.periodic_resurgence_ratio_ucl.to_numpy(dtype=float)
    unbounded = flags(frame, "periodic_resurgence_ratio_ucl_unbounded").to_numpy()
    if (not np.isfinite(lower).all() or (lower <= 0).any()
            or np.isnan(upper).any() or (upper < lower).any()
            or not np.array_equal(np.isposinf(upper), unbounded)):
        raise ValueError("Periodic bootstrap intervals must preserve ordered positive bounds and unbounded flags")
    if (not frame.periodic_bootstrap_successes.eq(2000).all()
            or not frame.periodic_bootstrap_status.eq("hc2_rademacher_wild_residual_bootstrap").all()):
        raise ValueError("Periodic bootstrap intervals require 2000 archived HC2 wild-residual replicates")
    rows = []
    for record in frame.to_dict("records"):
        numerator = float(record["post_mean_incidence_per_100k"])
        for baseline_id, (label, denominator_column, ratio_column) in BASELINES.items():
            denominator, ratio = float(record[denominator_column]), float(record[ratio_column])
            if not np.isfinite([numerator, denominator, ratio]).all() or denominator <= 0:
                raise ValueError(f"{record['iso3']}/{baseline_id}: invalid rates or ratio")
            if not np.isclose(ratio, numerator / denominator, rtol=1e-10, atol=1e-10):
                raise ValueError(f"{record['iso3']}/{baseline_id}: common numerator/denominator identity failed")
            periodic = baseline_id == "periodic"
            rows.append({
                "iso3": record["iso3"], "country": record["country"],
                "baseline_id": baseline_id, "baseline_label": label, "ratio": ratio,
                "baseline_incidence": denominator, "post_incidence": numerator,
                "post_years": "2023;2024;2025", "n_post_years": 3,
                "post_years_estimand": record["post_years_estimand"],
                "incidence_unit": INCIDENCE_UNIT, "ratio_offset_per_100k": 0.0,
                "preferred_source_classes": record["preferred_source_classes"],
                "periodic_fit_window": record["periodic_fit_window"] if periodic else "",
                "periodic_fit_n_years": record["periodic_fit_n_years"] if periodic else None,
                "audit_periodic_ratio_lcl": record["periodic_resurgence_ratio_lcl"] if periodic else None,
                "audit_periodic_ratio_ucl": record["periodic_resurgence_ratio_ucl"] if periodic else None,
                "audit_periodic_ratio_ucl_unbounded": record["periodic_resurgence_ratio_ucl_unbounded"] if periodic else None,
                "audit_periodic_bootstrap_status": record["periodic_bootstrap_status"] if periodic else "",
                "audit_periodic_bootstrap_successes": record["periodic_bootstrap_successes"] if periodic else None,
            })
    return pd.DataFrame(rows)


def build_selection_context(
    selection: pd.DataFrame, income_summary: pd.DataFrame, income_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag_columns = ["time_resolved_cases", "age_group_cases", "eligible_with_current_local_high_resolution_data", "selected_for_current_model2c"]
    for name, frame in [("selection", selection), ("income setting audit", income_audit)]:
        require_columns(frame, ["iso3", *flag_columns], name)
        if len(frame) != 217 or frame.iso3.duplicated().any():
            raise ValueError(f"{name}: expected 217 distinct screened settings")
    if set(selection.iso3) != set(income_audit.iso3):
        raise ValueError("Selection and income audits must cover identical settings")
    selected = selection.set_index("iso3").sort_index()
    audit = income_audit.set_index("iso3").sort_index().copy()
    for column in flag_columns:
        if not flags(selected, column).equals(flags(audit, column)):
            raise ValueError(f"Income audit and current selection disagree on {column}")
        audit[column] = flags(audit, column)
    eligible = audit.eligible_with_current_local_high_resolution_data
    primary = audit.selected_for_current_model2c
    if not eligible.equals(audit.time_resolved_cases & audit.age_group_cases):
        raise ValueError("Joint eligibility must mean both time-resolved and age-specific notifications")
    if int(eligible.sum()) != 10 or set(audit.index[primary]) != PRIMARY_ISO3 or (primary & ~eligible).any():
        raise ValueError("Expected nested 217 screened / 10 joint eligible / 9 primary settings")
    sensitivity = eligible & ~primary
    if set(audit.index[sensitivity]) != {"SGP"}:
        raise ValueError("Singapore must remain eligible and outside the frozen primary set")
    stage_sets = [
        ("screened", "Settings screened", audit.index, "Global notification calibration-data screen; not a global auxiliary-stream absence audit."),
        ("joint_eligible", "Jointly eligible", audit.index[eligible], "Both time-resolved all-age and annual age-specific notifications available in the archived screen."),
        ("primary_backgrounds", "Primary model backgrounds", audit.index[primary], "Frozen primary set; non-random selection based on data availability."),
        ("sensitivity_only", "Singapore: separate sensitivity", audit.index[sensitivity], "Eligible but outside the frozen nine-country primary set; separately tagged sensitivity fit."),
    ]
    stage_rows = [{
        "stage_id": stage_id, "stage_label": label, "n_settings": len(iso3s),
        "denominator_n": 217, "percent_of_screen": 100 * len(iso3s) / 217,
        "stage_order": order, "iso3_list": ";".join(iso3s), "note": note,
    } for order, (stage_id, label, iso3s, note) in enumerate(stage_sets, start=1)]
    require_columns(income_summary, [
        "world_bank_income_group", "n_settings", "n_time_resolved", "n_age_specific",
        "n_joint_high_resolution", "n_selected_calibration_backgrounds",
        "joint_high_resolution_percent", "income_classification_date", "interpretation_boundary",
    ], "income summary")
    require_columns(audit, ["income_group_for_summary", "income_classification_date"], "income audit")
    if income_summary.world_bank_income_group.duplicated().any():
        raise ValueError("Income summary contains duplicate groups")
    if set(income_summary.world_bank_income_group) != set(audit.income_group_for_summary):
        raise ValueError("Income summary groups differ from the setting audit")
    income = income_summary.copy()
    checks = {
        "n_time_resolved": "time_resolved_cases", "n_age_specific": "age_group_cases",
        "n_joint_high_resolution": "eligible_with_current_local_high_resolution_data",
        "n_selected_calibration_backgrounds": "selected_for_current_model2c",
    }
    for index, record in income.iterrows():
        group = audit.loc[audit.income_group_for_summary.eq(record.world_bank_income_group)]
        expected = {"n_settings": len(group), **{key: int(group[col].sum()) for key, col in checks.items()}}
        if any(int(record[key]) != value for key, value in expected.items()):
            raise ValueError(f"Income summary count mismatch: {record.world_bank_income_group}")
        if not group.income_classification_date.eq(record.income_classification_date).all():
            raise ValueError("Income classification vintages differ")
        if not np.isclose(record.joint_high_resolution_percent, 100 * expected["n_joint_high_resolution"] / len(group)):
            raise ValueError("Income eligibility percentage uses a different denominator")
        income.loc[index, "n_sensitivity_only"] = int((group.eligible_with_current_local_high_resolution_data & ~group.selected_for_current_model2c).sum())
        income.loc[index, "n_not_joint_eligible"] = len(group) - expected["n_joint_high_resolution"]
    income[["n_sensitivity_only", "n_not_joint_eligible"]] = income[["n_sensitivity_only", "n_not_joint_eligible"]].astype(int)
    income["income_label"] = income.world_bank_income_group
    return pd.DataFrame(stage_rows), income


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_primary_coverage(income_audit: pd.DataFrame) -> pd.DataFrame:
    """Count library members, not source availability, across region and income."""
    require_columns(income_audit, ["iso3", "who_region", "income_group_for_summary",
                                  "selected_for_current_model2c"], "coverage")
    primary = income_audit.loc[flags(income_audit, "selected_for_current_model2c")]
    if set(primary.iso3) != PRIMARY_ISO3 or primary.iso3.duplicated().any():
        raise ValueError("Coverage requires exactly the nine primary backgrounds")
    income_groups = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    if not primary.income_group_for_summary.isin(income_groups).all() or not primary.who_region.isin(REGIONS).all():
        raise ValueError("Primary backgrounds have unknown region or income group")
    return pd.DataFrame([
        {"who_region": region, "region_label": label, "income_group": income,
         "n_primary": int(((primary.who_region == region) & (primary.income_group_for_summary == income)).sum()),
         "iso3_list": ";".join(sorted(primary.loc[(primary.who_region == region) &
                                   (primary.income_group_for_summary == income), "iso3"]))}
        for region, label in REGIONS.items() for income in income_groups
    ])


def write_if_changed(path: Path, content: bytes) -> None:
    if not path.exists() or path.read_bytes() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def build_monthly_context(monthly: pd.DataFrame) -> pd.DataFrame:
    """Promote the 2015–25 portion of Supplementary Figure 17, retaining gaps."""
    columns = ["iso3", "country", "year", "month", "date", "fit_role", "observed_cases",
               "reported_median", "reported_lower", "reported_upper",
               "reported_predictive_lower", "reported_predictive_upper",
               "expected_ui_method", "predictive_ui_method"]
    require_columns(monthly, columns, "monthly calibration")
    frame = monthly.loc[monthly.iso3.isin(PRIMARY_ISO3) & monthly.year.between(2015, 2025), columns].copy()
    if frame.duplicated(["iso3", "year", "month"]).any() or set(frame.iso3) != PRIMARY_ISO3:
        raise ValueError("Monthly calibration requires unique country-months for nine countries")
    if not frame.groupby("iso3").size().eq(132).all() or not frame.month.between(1, 12).all():
        raise ValueError("Monthly context must retain the complete 2015–25 grid, including missing observations")
    expected_dates = pd.to_datetime(dict(year=frame.year, month=frame.month, day=15))
    if not pd.to_datetime(frame.date).eq(expected_dates).all():
        raise ValueError("Monthly date and year/month keys disagree")
    if not frame.fit_role.isin(["Monthly likelihood", "State history"]).all():
        raise ValueError("Unexpected fit role within the main monthly display window")
    valid = frame.fit_role.eq("Monthly likelihood")
    numeric = ["observed_cases", "reported_median", "reported_lower", "reported_upper",
               "reported_predictive_lower", "reported_predictive_upper"]
    if not np.isfinite(frame.loc[valid, numeric]).all().all() or (frame.loc[valid, numeric] < 0).any().any():
        raise ValueError("Observed calibration months require finite non-negative counts and intervals")
    for low, high in [("reported_lower", "reported_upper"),
                      ("reported_predictive_lower", "reported_predictive_upper")]:
        if (frame.loc[valid, low] > frame.loc[valid, high]).any():
            raise ValueError("Monthly interval endpoints are reversed")
    if not frame.expected_ui_method.eq("calibration_ensemble_percentile_2.5_97.5").all() or not frame.predictive_ui_method.eq("negative_binomial_predictive_percentile_2.5_97.5").all():
        raise ValueError("Monthly expected and predictive interval provenance changed")
    # Missing observations remain explicit rows, breaking plotted lines/ribbons.
    frame.loc[~valid, numeric] = np.nan
    frame["observation_available"] = valid
    frame["analysis_role"] = "primary_calibration_through_2025"
    return frame.sort_values(["iso3", "year", "month"]).reset_index(drop=True)


def build_temporal_coverage(monthly: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Verify held-out coverage against month-level intervals before rendering."""
    bounds = {50: ("pred_q0_25", "pred_q0_75"), 80: ("pred_q0_1", "pred_q0_9"),
              95: ("pred_q0_025", "pred_q0_975")}
    require_columns(monthly, ["iso3", "year", "month", "observed_cases",
        *[c for pair in bounds.values() for c in pair], *[f"covered_{n}" for n in bounds]], "temporal months")
    require_columns(summary, ["scope", "iso3", "country", "horizon_band", "n_months", "n_observed_months", "n_missing_months",
        *[f"coverage_{n}" for n in bounds]], "temporal summary")
    if (set(monthly.iso3) != PRIMARY_ISO3 or monthly.duplicated(["iso3", "year", "month"]).any()
            or not monthly.year.between(2023, 2025).all() or not monthly.month.between(1, 12).all()
            or not monthly.groupby("iso3").size().eq(36).all()):
        raise ValueError("Temporal check requires 36 held-out months per primary country in 2023–25")
    expected = {}
    observed = monthly.observed_cases.notna()
    for nominal, (low, high) in bounds.items():
        if (not np.isfinite(monthly[[low, high]]).all().all()
                or not np.isfinite(monthly.loc[observed, "observed_cases"]).all()
                or (monthly[low] > monthly[high]).any()):
            raise ValueError("Temporal intervals and observations must be finite and ordered")
        covered = monthly.observed_cases.between(monthly[low], monthly[high])
        if not covered.loc[observed].equals(flags(monthly.loc[observed], f"covered_{nominal}")):
            raise ValueError("Archived temporal coverage flags disagree with predictive intervals")
        expected[nominal] = covered
    selected = summary.loc[summary.scope.isin(["country_all_horizons", "pooled_all_horizons"])].copy()
    if (len(selected) != 10 or selected.iso3.duplicated().any()
            or set(selected.iso3) != PRIMARY_ISO3 | {"ALL"}
            or not selected.horizon_band.eq("all").all()):
        raise ValueError("Temporal coverage summary requires nine countries and one pooled row")
    rows = []
    for r in selected.itertuples(index=False):
        mask = pd.Series(True, index=monthly.index) if r.iso3 == "ALL" else monthly.iso3.eq(r.iso3)
        if int(mask.sum()) != r.n_months:
            raise ValueError("Temporal coverage denominator differs from the month-level archive")
        available = mask & observed
        if int(available.sum()) != r.n_observed_months or int((mask & ~observed).sum()) != r.n_missing_months:
            raise ValueError("Temporal observed/missing counts disagree with the archive")
        if not available.any():
            raise ValueError("Temporal coverage has no evaluable observations")
        for nominal in bounds:
            covered = int(expected[nominal].loc[available].sum())
            coverage = covered / int(available.sum())
            if not np.isclose(coverage, getattr(r, f"coverage_{nominal}"), rtol=0, atol=1e-12):
                raise ValueError("Temporal summary disagrees with month-level coverage")
            rows.append({"iso3": r.iso3, "country": r.country, "nominal_coverage_percent": nominal,
                         "observed_coverage_percent": coverage * 100, "n_covered": covered,
                         "n_months": int(available.sum()), "n_calendar_months": int(mask.sum()),
                         "n_missing_months": int((mask & ~observed).sum()), "coverage_gap_pp": coverage * 100 - nominal,
                         "fit_end_year": 2022, "evaluation_years": "2023;2024;2025",
                         "analysis_role": "separate_temporal_fit_through_2022"})
    return pd.DataFrame(rows).sort_values(["iso3", "nominal_coverage_percent"]).reset_index(drop=True)


def build_vaccination_context(coverage: pd.DataFrame) -> pd.DataFrame:
    """Summarise the same WUENIC series used in Supplementary Figure 13."""
    require_columns(coverage, ["iso3", "country", "year", "antigen", "coverage", "coverage_type", "source"], "vaccination coverage")
    source_type = "WHO/UNICEF Estimates of National Immunization Coverage"
    frame = coverage.loc[coverage.iso3.isin(PRIMARY_ISO3) & coverage.antigen.eq("DTPCV3")
                         & coverage.coverage_type.eq(source_type) & coverage.year.between(2019, 2024)].copy()
    if (set(frame.iso3) != PRIMARY_ISO3 or frame.duplicated(["iso3", "year"]).any()
            or not frame.groupby("iso3").size().eq(6).all()):
        raise ValueError("Vaccination context requires unique 2019–2024 WUENIC DTP3 estimates for all nine countries")
    if not np.isfinite(frame.coverage).all() or not frame.coverage.between(0, 1).all():
        raise ValueError("WUENIC coverage must be a finite proportion in [0, 1]")
    rows = []
    for iso3, group in frame.groupby("iso3", sort=True):
        group = group.sort_values("year")
        windows = [("reference_2019", "2019", group.loc[group.year.eq(2019)]),
                   ("minimum_2020_2024", "2020–24 minimum", group.loc[group.year.between(2020, 2024)]),
                   ("latest_2024", "2024", group.loc[group.year.eq(2024)])]
        for statistic, label, window in windows:
            value = window.coverage.min()
            selected = window.loc[window.coverage.eq(value)]
            rows.append({"iso3": iso3, "country": group.iloc[0].country,
                         "statistic_id": statistic, "statistic_label": label,
                         "coverage_percent": 100 * float(value),
                         "window_years": ";".join(window.year.astype(int).astype(str)),
                         "value_years": ";".join(selected.year.astype(int).astype(str)),
                         "n_available_years": len(window), "antigen": "DTPCV3",
                         "coverage_type": source_type, "source": ";".join(sorted(set(window.source))),
                         "carried_forward": False, "analysis_role": "routine_vaccination_context"})
    return pd.DataFrame(rows)


def build_context_data(root: Path = ROOT, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    output_dir = output_dir or root / "manuscript/figure_data"
    source_bytes = {name: (root / path).read_bytes() for name, path in INPUTS.items()}
    frames = {name: pd.read_csv(io.BytesIO(content)) for name, content in source_bytes.items()}
    regional = build_regional_context(frames["annual_regional_rates"])
    baselines = build_baseline_context(frames["country_baseline_robustness"], frames["selection_audit"])
    selection, income = build_selection_context(frames["selection_audit"], frames["income_summary"], frames["income_setting_audit"])
    coverage = build_primary_coverage(frames["income_setting_audit"])
    monthly = build_monthly_context(frames["monthly_calibration"])
    temporal = build_temporal_coverage(frames["temporal_monthly"], frames["temporal_summary"])
    vaccination = build_vaccination_context(frames["vaccination_coverage"])
    result = {"regional": regional, "baselines": baselines, "selection": selection, "income": income,
              "coverage": coverage, "monthly": monthly, "temporal_coverage": temporal,
              "vaccination": vaccination}
    metadata = {
        "schema_version": 5, "builder": "script/build_figure1_context_data.py",
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "inputs": [{"role": name, "path": str(path), "sha256": sha256(source_bytes[name])} for name, path in INPUTS.items()],
        "definitions": {
            "regional": "Arithmetic mean of frozen WHO-region annual reported incidence per 100000 in 2023-2025 versus 2015-2019; ratio=(post_mean+0.01)/(baseline_mean+0.01), matching the previous Figure 1b. Not a ratio of pooled counts and populations.",
            "baselines": "Nine frozen primary settings only; requested_data_cutoff=2025 and provisional_year_policy=include. Retain source ratios exactly: the same available-year 2023-2025 mean reported incidence divided by each of the 2010-2019 mean, 2015-2019 mean, and periodic expected incidence. No offset, floor, refit or bootstrap rerun.",
            "range": "The span of the three baseline point estimates in panel b is comparator sensitivity, not a confidence interval. Panel c uses the periodic bootstrap bounds retained in audit-prefixed columns; infinity and unbounded flags are not clipped in the source table. An arrow marks an unbounded upper interval in the plot.",
            "periodic_uncertainty": "Main panel c promotes the nine-country interval comparison from Supplementary Figure 27f. Preserve 95% conditional intervals from 2000 archived HC2 leverage-adjusted Rademacher wild-residual bootstrap replicates of the periodic baseline, holding the observed 2023-2025 numerator fixed. These intervals describe baseline estimation conditional on the fitted periodic model; they do not include uncertainty in reporting completeness or alternative model structures. No new bootstrap or model fit is performed.",
            "periodic": "Use the archived periodic baseline and preserve its fit-window audit field. The source fits 2010-2019 or a 2000-2019 fallback when fewer than six years are available; do not infer an identical historical window without checking this field.",
            "years": "All nine countries have three available post years in this frozen configuration. Presence of a 2025 observation is not proof of a complete/final annual report. No 2026 incidence is used, including where the selection audit has a later latest-case-year field.",
            "selection": "Nested sets: 217 screened settings, 10 with both required notification resolutions, 9 frozen primary backgrounds; Singapore is the remaining eligible setting and a separate sensitivity case. Counts of these stages must not be summed.",
            "income": "Preserve the frozen income summary and verify its counts and classification date against all 217 setting-level records. This is calibration representation, not global auxiliary-stream absence or income-stratified classifier performance.",
            "coverage": "Six WHO regions crossed with four income groups; counts and ISO3 lists refer only to the nine primary models. Zero means no model background in that cell, not national absence of surveillance data. Singapore remains in the original eligibility audit but is outside the primary benchmark.",
            "monthly": "The 2015-2025 portion of the existing monthly calibration export underlying Supplementary Figure 17. Retain the complete country-month grid, preserve missing observations as gaps, and omit partial 2026 checks. Points are observed notifications; the curve is the median expected report; the displayed 95% band uses negative-binomial predictive-replicate percentiles. Country-specific linear count scales are not incidence rates or population-normalised comparisons. These are in-sample fits, distinct from the separate temporal prediction check in the supplementary methods.",
            "temporal_coverage": "Supporting data for the supplementary temporal-prediction methods, not an active main-figure panel. The separate fit-through-2022 experiment is evaluated over available observations in the 36-month 2023-2025 window per country. Nine missing UK months are excluded, giving 27 UK observations and 315 pooled country-months. Verify archived interval-coverage flags and summary proportions against the month-level predictive endpoints. Retain actual coverage for nominal 50%, 80% and 95% intervals and the observed-minus-nominal difference in percentage points. Coverage is descriptive, not an independent-country uncertainty interval. No model, predictive draws or observation data are regenerated.",
            "vaccination": "Supporting programme context and the archived earlier panel-c candidate, not an active main-figure panel. Use the WUENIC DTP3 annual estimates also underlying Supplementary Figure 13: 2019, the minimum over 2020-2024, and 2024. All nine countries have the complete required six-year series. Convert fractions to percent without clipping, interpolation, administrative-estimate substitution or 2025 carry-forward. The minimum can occur in 2024 and can equal the endpoint. Connecting segments span these selected values, not uncertainty. This supplies routine programme context, not a causal estimate of vaccination effects on resurgence.",
        },
        "outputs": [],
    }
    for name, frame in result.items():
        filename = f"figure_1_context_{name}.csv"
        content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        write_if_changed(output_dir / filename, content)
        metadata["outputs"].append({"file": filename, "rows": len(frame), "sha256": sha256(content)})
    write_if_changed(output_dir / "figure_1_context.metadata.json", (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript/figure_data")
    args = parser.parse_args()
    result = build_context_data(output_dir=args.output_dir)
    print("Figure 1 context written or unchanged: " + ", ".join(f"{name}={len(frame)} rows" for name, frame in result.items()))


if __name__ == "__main__":
    main()
