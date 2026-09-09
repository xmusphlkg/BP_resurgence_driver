from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.model2_constants import (
    MODEL2_EXCLUDED_MAIN_ANALYSIS,
    PREFERRED_COVERAGE_TYPES,
    PRIMARY_DTP_COVERAGE_TYPE,
)
from pertussis_global_dynamics.paths import CLEAN, DATA_INTERMEDIATE, FITTED, METADATA, MODELS, OUTPUTS, RAW, REPORTS, ROOT


def ensure_directories() -> None:
    for path in (DATA_INTERMEDIATE, OUTPUTS, REPORTS, MODELS, FITTED):
        path.mkdir(parents=True, exist_ok=True)


def optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_monthly_covid_policy() -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "month",
        "monthly_stringency_index",
        "monthly_school_closure",
        "monthly_workplace_closure",
        "monthly_stay_home_requirement",
        "oxcgrt_days",
        "data_available",
        "source",
        "notes",
    ]
    raw_path = RAW / "covid_policy" / "OxCGRT_compact_national_v1.csv"
    if not raw_path.exists():
        return pd.DataFrame(columns=columns)
    usecols = [
        "CountryName",
        "CountryCode",
        "Jurisdiction",
        "Date",
        "StringencyIndex_Average",
        "C1M_School closing",
        "C2M_Workplace closing",
        "C6M_Stay at home requirements",
    ]
    raw = pd.read_csv(raw_path, usecols=usecols)
    raw = raw.loc[raw["Jurisdiction"].astype(str).eq("NAT_TOTAL")].copy()
    raw["date"] = pd.to_datetime(raw["Date"].astype(str), format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw["year"] = raw["date"].dt.year.astype(int)
    raw["month"] = raw["date"].dt.month.astype(int)
    for column in [
        "StringencyIndex_Average",
        "C1M_School closing",
        "C2M_Workplace closing",
        "C6M_Stay at home requirements",
    ]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    monthly = raw.groupby(["CountryCode", "CountryName", "year", "month"], as_index=False).agg(
        monthly_stringency_index=("StringencyIndex_Average", "mean"),
        monthly_school_closure=("C1M_School closing", "mean"),
        monthly_workplace_closure=("C2M_Workplace closing", "mean"),
        monthly_stay_home_requirement=("C6M_Stay at home requirements", "mean"),
        oxcgrt_days=("StringencyIndex_Average", "count"),
    )
    monthly = monthly.rename(columns={"CountryCode": "iso3", "CountryName": "country"})
    monthly["data_available"] = monthly["oxcgrt_days"].gt(0)
    monthly["source"] = "OxCGRT compact national v1 daily data"
    monthly["notes"] = (
        "Monthly mean of national daily OxCGRT indicators; used by the monthly ODE to avoid applying annual "
        "COVID contact-shock averages to pre-intervention months."
    )
    return monthly.loc[:, columns].sort_values(["iso3", "year", "month"]).reset_index(drop=True)


def metadata_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    return text in {"1", "true", "t", "yes", "y"}


def baseline_maternal_coverage_from_metadata(metadata: dict[str, Any]) -> float:
    available = metadata_bool(metadata.get("maternal_coverage_available", False))
    coverage = pd.to_numeric(pd.Series([metadata.get("maternal_coverage_latest", np.nan)]), errors="coerce").iloc[0]
    if available and np.isfinite(coverage):
        return float(np.clip(coverage, 0.0, 0.98))
    maternal_program = metadata_bool(metadata.get("maternal_program", False))
    return 0.55 if maternal_program else 0.08


def registry_path(path_text: str) -> Path:
    historical_prefixes = {
        "data_raw/": "data/raw/",
        "data_clean/": "data/clean/",
        "data_intermediate/": "data/intermediate/",
        "metadata/": "data/metadata/",
        "models/": "script/models/",
        "R/": "script/R/",
    }
    normalized = path_text.strip()
    for old, new in historical_prefixes.items():
        if normalized.startswith(old):
            normalized = f"{new}{normalized[len(old):]}"
            break
    return ROOT / normalized


def worker_count(cap: int | None = None) -> int:
    raw = os.environ.get("PERTUSSIS_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    detected = max(1, os.cpu_count() or 1)
    return detected if cap is None else max(1, min(detected, cap))


def model2_country_limit(default: int = 7) -> int:
    raw = os.environ.get("MODEL2_COUNTRY_LIMIT")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return default
    return default


def model2_bootstrap_replicates(default: int = 48) -> int:
    raw = os.environ.get("MODEL2_BOOTSTRAP_REPLICATES")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return default
    return default


def choose_dtp_coverage(coverage: pd.DataFrame, dose: str) -> pd.DataFrame:
    dtp = coverage.loc[
        coverage["antigen"].astype(str).eq(f"DTPCV{dose}")
        & coverage["dose"].astype(str).eq(str(dose))
    ].copy()
    dtp["year"] = pd.to_numeric(dtp["year"], errors="coerce").astype("Int64")
    dtp["coverage"] = pd.to_numeric(dtp["coverage"], errors="coerce")
    dtp = dtp.loc[dtp["coverage"].notna()].copy()
    dtp = dtp.loc[dtp["coverage_type"].astype(str).eq(PRIMARY_DTP_COVERAGE_TYPE)].copy()
    dtp = dtp.sort_values(["iso3", "year"])
    dtp = dtp.drop_duplicates(["iso3", "year"], keep="first")
    return dtp.loc[:, ["iso3", "year", "coverage"]].rename(columns={"coverage": f"dtp{dose}_coverage"})


def pre_mean_dtp_from_coverage(dtp: pd.DataFrame, iso3: str, dose: str) -> float:
    """Return the 2010-2019 DTP reference from the full coverage table."""
    column = f"dtp{dose}_coverage"
    if dtp.empty or column not in dtp:
        return np.nan
    frame = dtp.loc[dtp["iso3"].astype(str).eq(str(iso3))].copy()
    if frame.empty:
        return np.nan
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    pre = frame.loc[frame["year"].between(2010, 2019), column].dropna()
    if pre.empty:
        pre = frame[column].dropna()
    return float(pre.mean()) if not pre.empty else np.nan


def pre_mean_dtp1_from_coverage(dtp1: pd.DataFrame, iso3: str) -> float:
    return pre_mean_dtp_from_coverage(dtp1, iso3, "1")


def pre_mean_dtp3_from_coverage(dtp3: pd.DataFrame, iso3: str) -> float:
    return pre_mean_dtp_from_coverage(dtp3, iso3, "3")


def coverage_disruption_weight_from_drop(drop: float) -> float:
    if not np.isfinite(drop):
        return np.nan
    return float(np.clip(0.75 + max(float(drop), 0.0) * 4.0, 0.0, 3.0))


def model2_coverage_source_sensitivity(inputs: dict[str, pd.DataFrame], selected_countries: list[str]) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "selected_for_current_model2",
        "excluded_from_default_model2",
        "antigen",
        "dose",
        "coverage_type",
        "source_priority",
        "is_primary_model2_source",
        "baseline_window",
        "pandemic_window",
        "available_years_2010_2022",
        "overlap_years_with_primary_2010_2022",
        "n_baseline_years",
        "n_pandemic_years",
        "pre_pandemic_mean_dtp3_coverage",
        "pandemic_mean_dtp3_coverage",
        "dtp3_coverage_drop",
        "coverage_disruption_weight_from_source",
        "primary_coverage_type",
        "primary_pre_pandemic_mean_dtp3_coverage",
        "primary_pandemic_mean_dtp3_coverage",
        "primary_dtp3_coverage_drop",
        "primary_coverage_disruption_weight",
        "dtp3_drop_difference_vs_primary",
        "coverage_disruption_weight_difference_vs_primary",
        "mean_abs_coverage_difference_vs_primary_2010_2022",
        "max_abs_coverage_difference_vs_primary_2010_2022",
        "max_source_coverage_2010_2022",
        "coverage_values_above_one_2010_2022",
        "coverage_source_sensitivity_level",
        "source_sensitivity_note",
        "journal_use_recommendation",
    ]
    coverage = inputs.get("coverage", pd.DataFrame()).copy()
    if coverage.empty or not selected_countries:
        return pd.DataFrame(columns=columns)

    coverage["iso3"] = coverage["iso3"].astype(str)
    coverage["year"] = pd.to_numeric(coverage["year"], errors="coerce").astype("Int64")
    coverage["coverage"] = pd.to_numeric(coverage["coverage"], errors="coerce")
    coverage["dose"] = coverage["dose"].astype(str)
    coverage["coverage_type"] = coverage["coverage_type"].astype(str)
    dtp3 = coverage.loc[
        coverage["iso3"].isin(selected_countries)
        & coverage["antigen"].astype(str).eq("DTPCV3")
        & coverage["dose"].eq("3")
    ].copy()
    if dtp3.empty:
        return pd.DataFrame(columns=columns)

    dtp3["coverage_priority"] = dtp3["coverage_type"].map(PREFERRED_COVERAGE_TYPES).fillna(9).astype(int)
    selected_set = set(selected_countries)
    rows: list[dict[str, Any]] = []
    for iso3 in selected_countries:
        country_rows = dtp3.loc[dtp3["iso3"].eq(iso3)].copy()
        if country_rows.empty:
            continue
        country = str(country_rows["country"].dropna().iloc[0]) if country_rows["country"].notna().any() else iso3
        primary = (
            country_rows.loc[country_rows["coverage_type"].astype(str).eq(PRIMARY_DTP_COVERAGE_TYPE)]
            .sort_values(["year", "coverage_priority"])
            .drop_duplicates(["iso3", "year"], keep="first")
            .loc[:, ["year", "coverage", "coverage_type"]]
            .rename(columns={"coverage": "primary_coverage", "coverage_type": "primary_selected_type"})
        )
        if primary.empty:
            continue
        primary_window = primary.loc[primary["year"].between(2010, 2022)]
        primary_type = PRIMARY_DTP_COVERAGE_TYPE
        primary_pre = float(primary.loc[primary["year"].between(2010, 2019), "primary_coverage"].mean())
        primary_pandemic = float(primary.loc[primary["year"].between(2020, 2022), "primary_coverage"].mean())
        primary_drop = primary_pre - primary_pandemic if np.isfinite(primary_pre) and np.isfinite(primary_pandemic) else np.nan
        primary_weight = coverage_disruption_weight_from_drop(primary_drop)

        for coverage_type, type_rows in country_rows.groupby("coverage_type", sort=False):
            type_rows = (
                type_rows.sort_values(["year", "coverage_priority"])
                .drop_duplicates(["iso3", "year", "coverage_type"], keep="first")
                .copy()
            )
            baseline = type_rows.loc[type_rows["year"].between(2010, 2019) & type_rows["coverage"].notna()]
            pandemic = type_rows.loc[type_rows["year"].between(2020, 2022) & type_rows["coverage"].notna()]
            analysis_window = type_rows.loc[type_rows["year"].between(2010, 2022) & type_rows["coverage"].notna()]
            pre_mean = float(baseline["coverage"].mean()) if not baseline.empty else np.nan
            pandemic_mean = float(pandemic["coverage"].mean()) if not pandemic.empty else np.nan
            drop = pre_mean - pandemic_mean if np.isfinite(pre_mean) and np.isfinite(pandemic_mean) else np.nan
            weight = coverage_disruption_weight_from_drop(drop)
            compare = (
                type_rows.loc[:, ["year", "coverage"]]
                .rename(columns={"coverage": "source_coverage"})
                .merge(primary, on="year", how="inner")
            )
            compare = compare.loc[
                compare["year"].between(2010, 2022)
                & compare["source_coverage"].notna()
                & compare["primary_coverage"].notna()
            ].copy()
            if not compare.empty:
                abs_diff = (compare["source_coverage"] - compare["primary_coverage"]).abs()
                mean_abs_diff = float(abs_diff.mean())
                max_abs_diff = float(abs_diff.max())
            else:
                mean_abs_diff = np.nan
                max_abs_diff = np.nan
            drop_diff = drop - primary_drop if np.isfinite(drop) and np.isfinite(primary_drop) else np.nan
            weight_diff = weight - primary_weight if np.isfinite(weight) and np.isfinite(primary_weight) else np.nan
            is_primary = str(coverage_type) == primary_type
            has_overlap = len(baseline) >= 6 and len(pandemic) >= 2 and len(compare) >= 8
            if is_primary:
                level = "primary_reference"
                note = "Reference DTP3 source selected by Model 2 coverage priority for the 2010-2022 disruption window."
                recommendation = "Use as the main DTP3 coverage-disruption covariate."
            elif not has_overlap:
                level = "limited_overlap"
                note = "Alternative source lacks enough overlapping non-missing baseline, pandemic, or comparison years for a stable source-sensitivity calculation."
                recommendation = "Do not use for strong sensitivity claims; report only as source-availability audit."
            elif abs(weight_diff) <= 0.02 and (not np.isfinite(max_abs_diff) or max_abs_diff <= 0.02):
                level = "low"
                note = "Alternative source gives a very similar DTP3 disruption weight to the primary source."
                recommendation = "Source choice is unlikely to materially change the routine-immunization disruption covariate."
            elif abs(weight_diff) <= 0.10 and (not np.isfinite(max_abs_diff) or max_abs_diff <= 0.06):
                level = "moderate"
                note = "Alternative source changes the DTP3 disruption weight modestly; keep routine-immunization disruption wording source-qualified."
                recommendation = "Mention source sensitivity when interpreting coverage-disruption contributions."
            else:
                level = "high"
                note = "Alternative source materially changes the DTP3 disruption weight or has large pointwise differences from the primary source."
                recommendation = "Treat coverage-disruption attribution as source-sensitive until curated national coverage series are added."

            rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "selected_for_current_model2": iso3 in selected_set,
                    "excluded_from_default_model2": iso3 in MODEL2_EXCLUDED_MAIN_ANALYSIS,
                    "antigen": "DTPCV3",
                    "dose": "3",
                    "coverage_type": str(coverage_type),
                    "source_priority": int(PREFERRED_COVERAGE_TYPES.get(str(coverage_type), 9)),
                    "is_primary_model2_source": bool(is_primary),
                    "baseline_window": "2010-2019",
                    "pandemic_window": "2020-2022",
                    "available_years_2010_2022": int(analysis_window["year"].nunique()),
                    "overlap_years_with_primary_2010_2022": int(compare["year"].nunique()) if not compare.empty else 0,
                    "n_baseline_years": int(baseline["year"].nunique()),
                    "n_pandemic_years": int(pandemic["year"].nunique()),
                    "pre_pandemic_mean_dtp3_coverage": pre_mean,
                    "pandemic_mean_dtp3_coverage": pandemic_mean,
                    "dtp3_coverage_drop": drop,
                    "coverage_disruption_weight_from_source": weight,
                    "primary_coverage_type": primary_type,
                    "primary_pre_pandemic_mean_dtp3_coverage": primary_pre,
                    "primary_pandemic_mean_dtp3_coverage": primary_pandemic,
                    "primary_dtp3_coverage_drop": primary_drop,
                    "primary_coverage_disruption_weight": primary_weight,
                    "dtp3_drop_difference_vs_primary": drop_diff,
                    "coverage_disruption_weight_difference_vs_primary": weight_diff,
                    "mean_abs_coverage_difference_vs_primary_2010_2022": mean_abs_diff,
                    "max_abs_coverage_difference_vs_primary_2010_2022": max_abs_diff,
                    "max_source_coverage_2010_2022": float(analysis_window["coverage"].max()) if not analysis_window.empty else np.nan,
                    "coverage_values_above_one_2010_2022": int(analysis_window["coverage"].gt(1.0).sum()),
                    "coverage_source_sensitivity_level": level,
                    "source_sensitivity_note": note,
                    "journal_use_recommendation": recommendation,
                }
            )

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["iso3", "source_priority", "coverage_type"])
        .reset_index(drop=True)
    )


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "cases": pd.read_csv(CLEAN / "cases_country_age_time.csv"),
        "age_cases": pd.read_csv(CLEAN / "cases_age_model_groups.csv"),
        "age_source_cases": pd.read_csv(CLEAN / "cases_age_specific.csv"),
        "age_pattern_evidence": optional_csv(CLEAN / "age_pattern_evidence.csv"),
        "coverage": pd.read_csv(CLEAN / "vaccine_coverage.csv"),
        "population_age": pd.read_csv(CLEAN / "population_age.csv"),
        "covid": pd.read_csv(CLEAN / "covid_policy.csv"),
        "covid_monthly": load_monthly_covid_policy(),
        "metadata": pd.read_csv(CLEAN / "country_metadata.csv"),
        "tiers": pd.read_csv(OUTPUTS / "country_tier_assignment.csv"),
        "resurgence": pd.read_csv(OUTPUTS / "resurgence_index_by_country.csv"),
        "diagnostic_timeline": optional_csv(CLEAN / "surveillance_diagnostic_timeline.csv"),
        "diagnostic_sources": optional_csv(CLEAN / "surveillance_diagnostic_sources.csv"),
        "surveillance_outcomes": optional_csv(CLEAN / "surveillance_outcome_evidence.csv"),
        "surveillance_outcome_sources": optional_csv(CLEAN / "surveillance_outcome_sources.csv"),
        "serology_evidence": optional_csv(CLEAN / "serology_anti_pt_igg_evidence.csv"),
        "serology_sources": optional_csv(CLEAN / "serology_anti_pt_igg_sources.csv"),
        "contact_mobility": optional_csv(CLEAN / "contact_mobility.csv"),
        "contact_validation_evidence": optional_csv(CLEAN / "contact_validation_evidence.csv"),
        "contact_validation_sources": optional_csv(CLEAN / "contact_validation_sources.csv"),
        "strain_amr_timeline": optional_csv(CLEAN / "pertussis_strain_amr_timeline.csv"),
        "strain_amr_sources": optional_csv(CLEAN / "pertussis_strain_amr_sources.csv"),
        "maternal_coverage": optional_csv(CLEAN / "maternal_immunization_coverage.csv"),
        "gbd_burden": optional_csv(CLEAN / "gbd_pertussis_burden.csv"),
    }


def periodic_baseline(years: np.ndarray, incidence: np.ndarray, predict_years: np.ndarray) -> tuple[np.ndarray, str]:
    years = np.asarray(years, dtype=float)
    incidence = np.asarray(incidence, dtype=float)
    mask = (years >= 2010) & (years <= 2019) & np.isfinite(incidence)
    fit_years = years[mask]
    fit_incidence = incidence[mask]
    if len(fit_years) < 6 or np.nansum(fit_incidence) <= 0:
        fallback = np.nanmean(fit_incidence) if len(fit_incidence) else np.nan
        return np.full(len(predict_years), max(float(fallback), 0.0) if np.isfinite(fallback) else np.nan), "descriptive fallback baseline"

    centered = fit_years - fit_years.mean()
    y = np.log1p(np.clip(fit_incidence, 0.0, None))
    x_full = np.column_stack(
        [
            np.ones(len(fit_years)),
            centered,
            np.sin(2 * np.pi * fit_years / 4.0),
            np.cos(2 * np.pi * fit_years / 4.0),
            np.sin(2 * np.pi * fit_years / 5.0),
            np.cos(2 * np.pi * fit_years / 5.0),
        ]
    )
    x = x_full[:, : min(x_full.shape[1], max(2, len(fit_years) - 1))]
    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
    predict_years = np.asarray(predict_years, dtype=float)
    xp_full = np.column_stack(
        [
            np.ones(len(predict_years)),
            predict_years - fit_years.mean(),
            np.sin(2 * np.pi * predict_years / 4.0),
            np.cos(2 * np.pi * predict_years / 4.0),
            np.sin(2 * np.pi * predict_years / 5.0),
            np.cos(2 * np.pi * predict_years / 5.0),
        ]
    )
    pred = np.expm1(np.clip(xp_full[:, : len(coefs)] @ coefs, -20.0, 20.0))
    cap = np.nanpercentile(fit_incidence, 95) * 6.0 + 1.0
    return np.clip(pred, 0.0, cap), "periodic log-incidence baseline with 2010-2019 4- and 5-year cycles"


def build_country_context(iso3: str, inputs: dict[str, pd.DataFrame]) -> CountryContext:
    cases = inputs["cases"].copy()
    cases["year"] = pd.to_numeric(cases["year"], errors="coerce").astype("Int64")
    country_all_age = cases.loc[cases["iso3"].eq(iso3) & cases["age_group"].eq("all_ages")].copy()
    time_resolved = country_all_age.loc[~country_all_age["time_resolution"].eq("annual")].copy()
    annual_source = country_all_age.loc[country_all_age["time_resolution"].eq("annual")].copy()
    if not time_resolved.empty:
        annual = time_resolved.groupby(["iso3", "country", "year"], as_index=False).agg(
            cases_total=("cases_total", "sum"),
            population=("population", "mean"),
            source=("source", "first"),
            source_type=("source_type", "first"),
            source_dataset=("source_dataset", "first"),
        )
        annual["time_resolution"] = "annual_from_time_resolved"
        annual["date_start"] = annual["year"].astype(str) + "-01-01"
        annual["date_end"] = annual["year"].astype(str) + "-12-31"
    else:
        annual = country_all_age.loc[country_all_age["time_resolution"].eq("annual")].copy()
    annual = annual.loc[annual["year"].between(2000, 2025)].copy()
    if annual.empty:
        raise ValueError(f"No annual case rows for {iso3}")
    annual["cases_total"] = pd.to_numeric(annual["cases_total"], errors="coerce")
    annual["population"] = pd.to_numeric(annual["population"], errors="coerce")
    annual = annual.dropna(subset=["year", "cases_total", "population"])
    annual = annual.loc[annual["population"].gt(0)].copy()
    annual["incidence_per_100k"] = annual["cases_total"] / annual["population"] * 100000.0

    baseline_annual = annual_source.loc[annual_source["year"].between(2000, 2025)].copy()
    if not baseline_annual.empty:
        baseline_annual["cases_total"] = pd.to_numeric(baseline_annual["cases_total"], errors="coerce")
        baseline_annual["population"] = pd.to_numeric(baseline_annual["population"], errors="coerce")
        baseline_annual = baseline_annual.dropna(subset=["year", "cases_total", "population"])
        baseline_annual = baseline_annual.loc[baseline_annual["population"].gt(0)].copy()
        baseline_annual["incidence_per_100k"] = baseline_annual["cases_total"] / baseline_annual["population"] * 100000.0

    dtp1 = choose_dtp_coverage(inputs["coverage"], "1")
    dtp3 = choose_dtp_coverage(inputs["coverage"], "3")
    coverage_pre_mean_dtp1 = pre_mean_dtp1_from_coverage(dtp1, iso3)
    coverage_pre_mean_dtp3 = pre_mean_dtp3_from_coverage(dtp3, iso3)
    annual = annual.merge(dtp1, on=["iso3", "year"], how="left").merge(dtp3, on=["iso3", "year"], how="left")
    annual = annual.sort_values("year")
    for column in ("dtp1_coverage", "dtp3_coverage"):
        annual[column] = pd.to_numeric(annual[column], errors="coerce")
        carry_2025 = annual["year"].eq(2025) & annual[column].isna()
        reference_2024 = annual.loc[annual["year"].eq(2024), column].dropna()
        if carry_2025.any() and not reference_2024.empty:
            annual.loc[carry_2025, column] = float(reference_2024.iloc[-1])
        annual[column] = annual[column].interpolate(limit_direction="both").ffill().bfill()

    covid = inputs["covid"].loc[inputs["covid"]["iso3"].eq(iso3)].copy()
    covid["year"] = pd.to_numeric(covid["year"], errors="coerce").astype("Int64")
    annual = annual.merge(
        covid.loc[:, ["iso3", "year", "stringency_index", "school_closure", "workplace_closure", "stay_home_requirement"]],
        on=["iso3", "year"],
        how="left",
    )
    for column in ("stringency_index", "school_closure", "workplace_closure", "stay_home_requirement"):
        annual[column] = pd.to_numeric(annual[column], errors="coerce").fillna(0.0)

    baseline_pre = baseline_annual.loc[baseline_annual["year"].between(2010, 2019)] if not baseline_annual.empty else pd.DataFrame()
    baseline_for_periodic = baseline_annual if len(baseline_pre) >= 6 else annual
    expected, note = periodic_baseline(
        baseline_for_periodic["year"].to_numpy(dtype=float),
        baseline_for_periodic["incidence_per_100k"].to_numpy(dtype=float),
        annual["year"].to_numpy(dtype=float),
    )
    annual["periodic_expected_incidence_per_100k"] = expected
    pre = annual.loc[annual["year"].between(2010, 2019)]
    annual_pre_mean_dtp3 = (
        float(pre["dtp3_coverage"].mean())
        if pre["dtp3_coverage"].notna().any()
        else float(annual["dtp3_coverage"].mean())
    )
    annual_pre_mean_dtp1 = (
        float(pre["dtp1_coverage"].mean())
        if pre["dtp1_coverage"].notna().any()
        else float(annual["dtp1_coverage"].mean())
    )
    pre_mean_dtp1 = coverage_pre_mean_dtp1 if np.isfinite(coverage_pre_mean_dtp1) else annual_pre_mean_dtp1
    pre_mean_dtp3 = coverage_pre_mean_dtp3 if np.isfinite(coverage_pre_mean_dtp3) else annual_pre_mean_dtp3
    pre_for_incidence = baseline_pre if len(baseline_pre) >= 6 else pre
    pre_mean_incidence = (
        float(pre_for_incidence["incidence_per_100k"].mean())
        if not pre_for_incidence.empty and pre_for_incidence["incidence_per_100k"].notna().any()
        else float(annual["incidence_per_100k"].mean())
    )

    pop_age = inputs["population_age"].loc[inputs["population_age"]["iso3"].eq(iso3)].copy()
    diagnostic = inputs.get("diagnostic_timeline", pd.DataFrame())
    diagnostic_timeline = (
        diagnostic.loc[diagnostic["iso3"].astype(str).eq(iso3)].copy()
        if not diagnostic.empty and "iso3" in diagnostic
        else pd.DataFrame()
    )
    surveillance = inputs.get("surveillance_outcomes", pd.DataFrame())
    surveillance_outcomes = (
        surveillance.loc[surveillance["iso3"].astype(str).eq(iso3)].copy()
        if not surveillance.empty and "iso3" in surveillance
        else pd.DataFrame()
    )
    pathogen = inputs.get("strain_amr_timeline", pd.DataFrame())
    pathogen_timeline = (
        pathogen.loc[pathogen["iso3"].astype(str).eq(iso3)].copy()
        if not pathogen.empty and "iso3" in pathogen
        else pd.DataFrame()
    )
    covid_monthly = inputs.get("covid_monthly", pd.DataFrame())
    monthly_covid = (
        covid_monthly.loc[covid_monthly["iso3"].astype(str).eq(iso3)].copy()
        if not covid_monthly.empty and "iso3" in covid_monthly
        else pd.DataFrame()
    )
    meta_row = inputs["metadata"].loc[inputs["metadata"]["iso3"].eq(iso3)].head(1)
    metadata = meta_row.iloc[0].to_dict() if not meta_row.empty else {"iso3": iso3, "country": annual["country"].iloc[0]}
    return CountryContext(
        iso3=iso3,
        country=str(annual["country"].iloc[0]),
        annual=annual,
        time_resolved=time_resolved,
        population_age=pop_age,
        metadata=metadata,
        pre_mean_dtp3=max(min(pre_mean_dtp3, 0.99), 0.01) if np.isfinite(pre_mean_dtp3) else 0.90,
        pre_mean_incidence=max(pre_mean_incidence, 0.001) if np.isfinite(pre_mean_incidence) else 0.001,
        baseline_note=note,
        monthly_covid=monthly_covid,
        diagnostic_timeline=diagnostic_timeline,
        pathogen_timeline=pathogen_timeline,
        surveillance_outcomes=surveillance_outcomes,
        pre_mean_dtp1=max(min(pre_mean_dtp1, 0.99), 0.01) if np.isfinite(pre_mean_dtp1) else 0.90,
    )


def select_pilot_countries(inputs: dict[str, pd.DataFrame], limit: int = 9) -> list[str]:
    tiers = inputs["tiers"].copy()
    if "latest_case_year" not in tiers.columns:
        latest = inputs["resurgence"].loc[:, ["iso3", "latest_case_year", "latest_reported_cases"]].drop_duplicates("iso3")
        tiers = tiers.merge(latest, on="iso3", how="left")
    metadata = inputs["metadata"].loc[:, ["iso3", "who_region", "maternal_program", "adolescent_booster"]].drop_duplicates("iso3")
    tiers = tiers.merge(metadata, on="iso3", how="left")
    preferred = {
        "USA",
        "GBR",
        "AUS",
        "CAN",
        "NLD",
        "DNK",
        "SWE",
        "FIN",
        "ESP",
        "POL",
        "CZE",
        "FRA",
        "DEU",
        "ITA",
        "JPN",
        "KOR",
        "NZL",
        "CHN",
    }
    has_time_age = pd.Series(True, index=tiers.index)
    if {"time_resolved_cases", "age_group_cases"}.issubset(tiers.columns):
        has_time_age = tiers["time_resolved_cases"].fillna(False).astype(bool) & tiers["age_group_cases"].fillna(False).astype(bool)
    candidates = tiers.loc[
        tiers["post_to_baseline_rr"].notna()
        & tiers["latest_case_year"].fillna(0).ge(2024)
        & has_time_age
    ].copy()
    if candidates.empty:
        candidates = tiers.loc[tiers["post_to_baseline_rr"].notna()].copy()
    candidates["score"] = (
        candidates["post_to_baseline_rr"].fillna(0).clip(upper=25)
        + candidates["tier_score"].fillna(0)
        + candidates["model2_pilot_candidate"].fillna(False).astype(float) * 10
        # Keep the default pilot stable while allowing auxiliary countries with
        # newly integrated high-resolution source series to enter the main set.
        + candidates["iso3"].isin(preferred).astype(float) * 3
    )
    chosen: list[str] = []
    for iso3 in candidates.sort_values("score", ascending=False)["iso3"]:
        if str(iso3) in MODEL2_EXCLUDED_MAIN_ANALYSIS:
            continue
        if iso3 not in chosen:
            chosen.append(str(iso3))
        if len(chosen) == limit:
            break
    return chosen


def country_selection_audit(inputs: dict[str, pd.DataFrame], selected_countries: list[str]) -> pd.DataFrame:
    tiers = inputs["tiers"].copy()
    if "latest_case_year" not in tiers.columns:
        latest = inputs["resurgence"].loc[:, ["iso3", "latest_case_year", "latest_reported_cases"]].drop_duplicates("iso3")
        tiers = tiers.merge(latest, on="iso3", how="left")
    metadata = inputs["metadata"].loc[
        :,
        ["iso3", "who_region", "maternal_program", "adolescent_booster", "routine_scheduler_code"],
    ].drop_duplicates("iso3")
    tiers = tiers.merge(metadata, on="iso3", how="left", suffixes=("", "_metadata"))
    for column in ("time_resolved_cases", "age_group_cases", "model2_pilot_candidate"):
        if column not in tiers:
            tiers[column] = False
        tiers[column] = tiers[column].fillna(False).astype(bool)
    tiers["selected_for_current_model2"] = tiers["iso3"].astype(str).isin(selected_countries)
    tiers["excluded_from_default_model2"] = tiers["iso3"].astype(str).isin(MODEL2_EXCLUDED_MAIN_ANALYSIS)
    tiers["default_model2_exclusion_reason"] = tiers["iso3"].astype(str).map(MODEL2_EXCLUDED_MAIN_ANALYSIS).fillna("")
    tiers["has_recent_post_pandemic_case_year"] = pd.to_numeric(tiers.get("latest_case_year"), errors="coerce").fillna(0).ge(2024)
    tiers["has_resurgence_metric"] = pd.to_numeric(tiers.get("post_to_baseline_rr"), errors="coerce").notna()
    tiers["eligible_with_current_local_high_resolution_data"] = (
        tiers["time_resolved_cases"]
        & tiers["age_group_cases"]
        & tiers["has_recent_post_pandemic_case_year"]
        & tiers["has_resurgence_metric"]
    )

    def reason(row: pd.Series) -> str:
        if bool(row["selected_for_current_model2"]):
            return "Selected: local time-resolved all-age and annual age-specific PertussisIncidence data support the current dynamic pilot."
        if bool(row["excluded_from_default_model2"]):
            return str(row["default_model2_exclusion_reason"])
        missing = []
        if not bool(row["time_resolved_cases"]):
            missing.append("time-resolved cases")
        if not bool(row["age_group_cases"]):
            missing.append("age-specific cases")
        if not bool(row["has_recent_post_pandemic_case_year"]):
            missing.append("recent 2024/2025 case year")
        if not bool(row["has_resurgence_metric"]):
            missing.append("post-pandemic baseline ratio")
        if missing:
            return "Not selected: missing " + ", ".join(missing) + "."
        return "Eligible but not selected under current MODEL2_COUNTRY_LIMIT."

    tiers["model2_selection_reason"] = tiers.apply(reason, axis=1)
    rr = pd.to_numeric(tiers.get("post_to_baseline_rr"), errors="coerce")
    tiers["weak_resurgence_or_negative_control_candidate"] = tiers["has_recent_post_pandemic_case_year"] & rr.notna() & rr.le(1.5)
    tiers["selection_bias_note"] = np.where(
        tiers["selected_for_current_model2"],
        "Selected countries are data-available dynamic pilots, not a random global sample.",
        "Excluded countries remain in the global descriptive screen until source-specific high-resolution data are added.",
    )
    ordered = [
        "iso3",
        "country",
        "who_region",
        "recommended_tier",
        "tier_score",
        "post_to_baseline_rr",
        "latest_case_year",
        "latest_reported_cases",
        "time_resolved_cases",
        "age_group_cases",
        "eligible_with_current_local_high_resolution_data",
        "selected_for_current_model2",
        "excluded_from_default_model2",
        "weak_resurgence_or_negative_control_candidate",
        "model2_selection_reason",
        "default_model2_exclusion_reason",
        "data_gaps",
        "maternal_program",
        "adolescent_booster",
        "routine_scheduler_code",
        "selection_bias_note",
    ]
    for column in ordered:
        if column not in tiers:
            tiers[column] = pd.NA
    return (
        tiers.loc[:, ordered]
        .sort_values(
            ["selected_for_current_model2", "eligible_with_current_local_high_resolution_data", "tier_score", "post_to_baseline_rr"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
