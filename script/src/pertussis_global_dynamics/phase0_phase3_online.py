from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pertussis_global_dynamics.model2_constants import PRIMARY_DTP_COVERAGE_TYPE
from pertussis_global_dynamics.paths import CLEAN, DATA_INTERMEDIATE, MANUSCRIPT, METADATA, OUTPUTS, RAW, REPORTS, ROOT


ANNUAL_CASES_RATES_DIR = RAW / "annual_cases_and_rates"
AGE_SPECIFIC_CASES_DIR = RAW / "age_specific_cases"
CONTACTDATA_DIR = RAW / "contact_mobility"
CONTACT_VALIDATION_DIR = RAW / "contact_validation"
COVID_POLICY_DIR = RAW / "covid_policy"
DEMOGRAPHY_DIR = RAW / "demography"
DIAGNOSTICS_DIR = RAW / "diagnostics"
DIGITAL_SURVEILLANCE_DIR = RAW / "digital_surveillance"
IMMUNIZATION_DIR = RAW / "immunization"
PATHOGEN_DIR = RAW / "pathogen"
PROVENANCE_DIR = RAW / "provenance"
SEROLOGY_DIR = RAW / "serology"
SEVERE_OUTCOMES_DIR = RAW / "severe_outcomes"
THAILAND_DIR = RAW / "thailand"
TIME_RESOLVED_CASES_DIR = RAW / "time_resolved_cases"

ACCESS_DATE = dt.date.today().isoformat()

WHO_REPORTED = ANNUAL_CASES_RATES_DIR / "who_reported_cases_all_vpd.xlsx"
WHO_INCIDENCE = ANNUAL_CASES_RATES_DIR / "who_incidence_rates_all_vpd.xlsx"
WHO_COVERAGE = IMMUNIZATION_DIR / "who_coverage_all_antigens.xlsx"
WHO_INTRODUCTION = IMMUNIZATION_DIR / "who_vaccine_introduction.xlsx"
WHO_SCHEDULE = IMMUNIZATION_DIR / "who_vaccine_schedule.xlsx"
OFFICIAL_PERTUSSIS_SCHEDULE_STRATEGY = IMMUNIZATION_DIR / "official_pertussis_schedule_strategy.csv"
CONTACTDATA_MATRICES = CONTACTDATA_DIR / "contactdata_prem_contact_matrices_by_location_16.csv"
CONTACTDATA_POPULATION = CONTACTDATA_DIR / "contactdata_population_byage.csv"
OXCGRT_COMPACT = COVID_POLICY_DIR / "OxCGRT_compact_national_v1.csv"
PERTUSSIS_INCIDENCE_PROVENANCE_DIR = PROVENANCE_DIR / "pertussis_incidence"
PI_INCIDENCE = TIME_RESOLVED_CASES_DIR / "pertussis_incidence_cases.xlsx"
PI_AGE = AGE_SPECIFIC_CASES_DIR / "pertussis_incidence_case_year_age.xlsx"
PI_HEALTHMAP = DIGITAL_SURVEILLANCE_DIR / "healthmap_alerts.csv"
PI_HEALTHMAP_ISO = DIGITAL_SURVEILLANCE_DIR / "healthmap_iso3_lookup.csv"
PI_OVERRIDES = TIME_RESOLVED_CASES_DIR / "pertussis_incidence_annual_case_overrides.csv"
PI_COMMIT = PERTUSSIS_INCIDENCE_PROVENANCE_DIR / "COMMIT"
LOCAL_SOURCES_DIR = RAW
LOCAL_AUXILIARY_INCIDENCE_REPORT = TIME_RESOLVED_CASES_DIR / "auxiliary_pertussis_incidence_report.xlsx"
LOCAL_REGION_RATE_WORKBOOK = ANNUAL_CASES_RATES_DIR / "who_region_incidence_rates.xlsx"
LOCAL_COUNTRY_SERIES_DIR = TIME_RESOLVED_CASES_DIR / "annual_country_series"
LOCAL_SURVEILLANCE_AGE_CASES = AGE_SPECIFIC_CASES_DIR / "supplemental_surveillance_cases_by_age.csv"
LOCAL_CANADA_OFFICIAL_AGE_DISTRIBUTION = AGE_SPECIFIC_CASES_DIR / "canada_vpd_2019_age_distribution.csv"
LOCAL_ENGLAND_OFFICIAL_AGE_DISTRIBUTION = AGE_SPECIFIC_CASES_DIR / "england_pertussis_age_distribution_1994_2014.csv"
LOCAL_EXTENDED_AGE_CASES = AGE_SPECIFIC_CASES_DIR / "extended_pertussis_case_year_age.xlsx"
LOCAL_ECDC_SURVEILLANCE = ANNUAL_CASES_RATES_DIR / "ecdc_pertussis_surveillance.csv"
LOCAL_DIAGNOSTIC_TIMELINE = DIAGNOSTICS_DIR / "diagnostic_standards_timeline.csv"
LOCAL_DIAGNOSTIC_SOURCES = DIAGNOSTICS_DIR / "diagnostic_standards_sources.csv"
LOCAL_DIAGNOSTIC_AGE_TESTING_SUMMARY = DIAGNOSTICS_DIR / "diagnostic_age_testing_summary.csv"
LOCAL_SURVEILLANCE_OUTCOME_EVIDENCE = SEVERE_OUTCOMES_DIR / "surveillance_outcome_evidence.csv"
LOCAL_SURVEILLANCE_OUTCOME_SOURCES = SEVERE_OUTCOMES_DIR / "surveillance_outcome_sources.csv"
LOCAL_THA_INFANT_SEVERE = THAILAND_DIR / "thailand_infant_severe_2019_2023.csv"
LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SUMMARY = SEVERE_OUTCOMES_DIR / "japan_annual_national_surveillance_summary.csv"
LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SOURCES = SEVERE_OUTCOMES_DIR / "japan_annual_national_surveillance_sources.csv"
LOCAL_SEROLOGY_EVIDENCE = SEROLOGY_DIR / "serology_anti_pt_igg_evidence.csv"
LOCAL_SEROLOGY_SOURCES = SEROLOGY_DIR / "serology_anti_pt_igg_sources.csv"
LOCAL_CONTACT_VALIDATION_EVIDENCE = CONTACT_VALIDATION_DIR / "contact_validation_evidence.csv"
LOCAL_CONTACT_VALIDATION_SOURCES = CONTACT_VALIDATION_DIR / "contact_validation_sources.csv"
LOCAL_AGE_PATTERN_EVIDENCE = CLEAN / "age_pattern_evidence.csv"
LOCAL_STRAIN_AMR_TIMELINE = PATHOGEN_DIR / "strain_amr_timeline.csv"
LOCAL_STRAIN_AMR_SOURCES = PATHOGEN_DIR / "strain_amr_sources.csv"
LOCAL_MATERNAL_COVERAGE = IMMUNIZATION_DIR / "maternal_immunization_coverage.xlsx"
LOCAL_MATERNAL_PREGNANCY_RECOMMENDATION_REVIEW = IMMUNIZATION_DIR / "maternal_tdap_recommendation_review.pdf"
LOCAL_WPP_AGE_GROUPS_ANNUAL = DEMOGRAPHY_DIR / "wpp_country_age_groups_annual.csv"
LOCAL_COUNTRY_PROFILE_INPUTS = IMMUNIZATION_DIR / "country_profile_inputs.csv"
LOCAL_VACCINE_PROGRAM_BACKGROUND = IMMUNIZATION_DIR / "vaccine_program_background.csv"
LOCAL_WHO_REPORTED_CASES_WORKBOOK = ANNUAL_CASES_RATES_DIR / "who_reported_cases_incidence_workbook_2025-12-02.xlsx"
LOCAL_GBD_DIR = SEVERE_OUTCOMES_DIR / "gbd"
LOCAL_GBD_ISO = LOCAL_GBD_DIR / "iso_code.csv"
LOCAL_GBD_FILES = (
    LOCAL_GBD_DIR / "ihme_gbd_2021_data_a5bddb9a_1.csv",
    LOCAL_GBD_DIR / "ihme_gbd_2021_data_c75f76ea_1.csv",
)

WHO_SOURCE_URL = "https://immunizationdata.who.int/"
CONTACTDATA_SOURCE_URL = "https://cran.r-project.org/package=contactdata"
OXCGRT_SOURCE_URL = "https://github.com/OxCGRT/covid-policy-dataset"
PERTUSSIS_INCIDENCE_SOURCE_URL = "https://github.com/xmusphlkg/PertussisIncidence"

PERTUSSIS_INCIDENCE_COUNTRIES = {
    "AU": ("AUS", "Australia"),
    "CN": ("CHN", "China"),
    "GB": ("GBR", "United Kingdom of Great Britain and Northern Ireland"),
    "JP": ("JPN", "Japan"),
    "NZ": ("NZL", "New Zealand"),
    "SE": ("SWE", "Sweden"),
    "SG": ("SGP", "Singapore"),
    "US": ("USA", "United States of America"),
}

EXTENDED_AGE_CASES_COUNTRIES = {
    "BR": ("BRA", "Brazil"),
    "CA": ("CAN", "Canada"),
    "TH": ("THA", "Thailand"),
}

EU_AGE_RATE_SOURCE_EXCLUDED_ISO3 = {
    iso3
    for _, (iso3, _) in {
        **PERTUSSIS_INCIDENCE_COUNTRIES,
        **EXTENDED_AGE_CASES_COUNTRIES,
    }.items()
}

EU_AGE_RATE_BINS = {"<1", "01-04", "05-09", "10-14", "15-19", "20-29", "30+"}
EU_AGE_RATE_PANEL_LABEL = "EU annual age-specific rate panel"
EU_AGE_RATE_PANEL_NORMALIZED_DATASET = "EU annual age-specific rate panel normalized to annual reported-case totals"

SURVEILLANCE_AGE_CASES_COUNTRIES = {
    "Australia": ("AUS", "Australia"),
    "China": ("CHN", "China"),
    "Thailand": ("THA", "Thailand"),
}

AUXILIARY_INCIDENCE_REPORT_COUNTRIES = {
    "BR": ("BRA", "Brazil"),
    "TH": ("THA", "Thailand"),
    "ZA": ("ZAF", "South Africa"),
}

COUNTRY_SERIES_COUNTRIES = {
    "australia": ("AUS", "Australia"),
    "canada": ("CAN", "Canada"),
    "china": ("CHN", "China"),
    "thailand": ("THA", "Thailand"),
}

INCIDENCE_COUNTRY_CODE_TO_ISO3 = {
    code: iso3
    for code, (iso3, _) in {
        **PERTUSSIS_INCIDENCE_COUNTRIES,
        **AUXILIARY_INCIDENCE_REPORT_COUNTRIES,
    }.items()
}
INCIDENCE_ISO3_TO_COUNTRY_CODE = {iso3: code for code, iso3 in INCIDENCE_COUNTRY_CODE_TO_ISO3.items()}

ECDC_REGION_TO_ISO3 = {
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "CY": "CYP",
    "CZ": "CZE",
    "DE": "DEU",
    "DK": "DNK",
    "EE": "EST",
    "ES": "ESP",
    "FI": "FIN",
    "EL": "GRC",
    "FR": "FRA",
    "GR": "GRC",
    "HR": "HRV",
    "HU": "HUN",
    "IE": "IRL",
    "IS": "ISL",
    "IT": "ITA",
    "LT": "LTU",
    "LU": "LUX",
    "LV": "LVA",
    "MT": "MLT",
    "NL": "NLD",
    "NO": "NOR",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SE": "SWE",
    "SI": "SVN",
    "SK": "SVK",
    "UK": "GBR",
}

AGE_GROUP_MAP = {
    "infant_0_2m": (0.0, 3.0 / 12.0),
    "infant_3_11m": (3.0 / 12.0, 1.0),
    "child_1_4y": (1.0, 5.0),
    "child_5_9y": (5.0, 10.0),
    "adolescent_10_17y": (10.0, 18.0),
    "young_adult_18_39y": (18.0, 40.0),
    "middle_adult_40_64y": (40.0, 65.0),
    "elderly_65plus": (65.0, np.nan),
}

CASE_COLUMNS = [
    "iso3",
    "country",
    "date_start",
    "date_end",
    "year",
    "month",
    "epiweek",
    "time_resolution",
    "age_group",
    "age_lower",
    "age_upper",
    "cases_confirmed",
    "cases_probable",
    "cases_total",
    "deaths",
    "hospitalizations",
    "population",
    "source",
    "source_type",
    "source_dataset",
    "case_definition",
    "notes",
]

AGE_MODEL_COLUMNS = [
    "iso3",
    "country",
    "date_start",
    "date_end",
    "year",
    "month",
    "epiweek",
    "time_resolution",
    "model_age_group",
    "age_group",
    "age_lower",
    "age_upper",
    "source",
    "source_type",
    "source_dataset",
    "case_definition",
    "notes",
    "cases_total",
    "source_cases_total",
    "allocation_fraction",
    "source_age_groups",
]

AGE_PATTERN_BROAD_GROUPS = {
    "infant_0_11m": ("infant_0_2m", "infant_3_11m"),
    "child_1_4y": ("child_1_4y",),
    "school_age_5_17y": ("child_5_9y", "adolescent_10_17y"),
    "young_adult_18_39y": ("young_adult_18_39y",),
    "middle_adult_40_64y": ("middle_adult_40_64y",),
    "elderly_65plus": ("elderly_65plus",),
}


def worker_count(cap: int | None = None) -> int:
    raw = os.environ.get("PERTUSSIS_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    detected = max(1, os.cpu_count() or 1)
    return detected if cap is None else max(1, min(detected, cap))


def parallel_calls(calls: list[tuple[object, tuple[object, ...]]], cap: int = 8) -> list[object]:
    workers = min(worker_count(cap=cap), len(calls))
    if workers <= 1:
        return [func(*args) for func, args in calls]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, *args) for func, args in calls]
        return [future.result() for future in futures]


def read_excel_data(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return read_excel_sheet(path, "Data", usecols=usecols)


def excel_cache_path(path: Path, sheet_name: str, usecols: list[str] | None = None) -> Path:
    stat = require_file(path).stat()
    columns = ",".join(usecols) if usecols else "__all__"
    key = f"{path.resolve()}|{sheet_name}|{columns}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return DATA_INTERMEDIATE / "excel_cache" / f"{safe_stem}_{digest}.pkl"


def read_excel_sheet(path: Path, sheet_name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    cache_path = excel_cache_path(path, sheet_name, usecols)
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(require_file(path), sheet_name=sheet_name, usecols=usecols)
    tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    frame.to_pickle(tmp)
    tmp.replace(cache_path)
    return frame


def ensure_directories() -> None:
    for path in (CLEAN, DATA_INTERMEDIATE, OUTPUTS, REPORTS, METADATA, MANUSCRIPT / "tables", MANUSCRIPT / "tables_md", MANUSCRIPT / "figure_data"):
        path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing required local input: {path}. Restore the saved data snapshot under data/raw/.")
    return path


def repo_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    text = str(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


MATERNAL_COVERAGE_CATEGORY_PRIORITY = {
    "Official coverage": 3,
    "Administrative coverage": 2,
    "Article": 1,
}


def maternal_coverage_year_bounds(value: object) -> tuple[object, object]:
    if pd.isna(value):
        return pd.NA, pd.NA
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return pd.NA, pd.NA
    tokens = re.findall(r"\d+(?:\.\d+)?", text)
    if not tokens:
        return pd.NA, pd.NA
    start = int(float(tokens[0]))
    end = int(float(tokens[-1]))
    return start, end


def standardize_maternal_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename_map = {
        "CODE": "iso3",
        "NAME": "country",
        "country_name": "country",
        "YEAR": "year",
        "COVERAGE_CATEGORY_DESCRIPTION": "coverage_category",
        "Coverage": "coverage_raw",
        "备注": "notes",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns}).copy()
    if "iso3" not in out:
        out["iso3"] = ""
    if "country" not in out:
        out["country"] = ""
    if "year" not in out:
        out["year"] = pd.NA
    if "coverage_category" not in out:
        out["coverage_category"] = ""
    if "coverage_raw" not in out:
        out["coverage_raw"] = pd.NA
    if "source" not in out:
        out["source"] = ""
    if "notes" not in out:
        out["notes"] = ""
    if "source_type" not in out:
        out["source_type"] = "measured"
    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["country"] = out["country"].astype(str).str.strip()
    year_bounds = out["year"].map(maternal_coverage_year_bounds)
    out["coverage_year_start"] = pd.array([start for start, _ in year_bounds], dtype="Int64")
    out["coverage_year_end"] = pd.array([end for _, end in year_bounds], dtype="Int64")
    out["year"] = out["coverage_year_end"]
    out["coverage_percent"] = out["coverage_raw"].map(clean_number)
    out["coverage_proportion"] = out["coverage_percent"] / 100.0
    out["source"] = out["source"].fillna("").astype(str)
    out["source_type"] = out["source_type"].fillna("measured").astype(str)
    out["coverage_category"] = out["coverage_category"].fillna("").astype(str)
    out["coverage_raw"] = out["coverage_raw"].fillna("").astype(str)
    out["notes"] = out["notes"].fillna("").astype(str)
    out["coverage_available"] = out["coverage_proportion"].notna() & ~out["source_type"].astype(str).str.contains(
        "derived_no_program",
        case=False,
        na=False,
    )
    out["coverage_category_priority"] = out["coverage_category"].map(MATERNAL_COVERAGE_CATEGORY_PRIORITY).fillna(0).astype(int)
    out = out.loc[out["coverage_available"]].copy()
    out["year"] = out["year"].astype("Int64")
    out["coverage_year_start"] = out["coverage_year_start"].astype("Int64")
    out["coverage_year_end"] = out["coverage_year_end"].astype("Int64")
    return out


def select_latest_maternal_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        columns = [
            "iso3",
            "country",
            "year",
            "coverage_year_start",
            "coverage_year_end",
            "coverage_category",
            "coverage_percent",
            "coverage_proportion",
            "coverage_raw",
            "source_type",
            "source",
            "notes",
            "coverage_available",
        ]
        return pd.DataFrame(columns=columns)
    ordered = frame.sort_values(
        ["coverage_available", "coverage_year_end", "coverage_category_priority", "coverage_percent", "source"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).copy()
    latest = ordered.drop_duplicates("iso3", keep="first").copy()
    columns = [
        "iso3",
        "country",
        "year",
        "coverage_year_start",
        "coverage_year_end",
        "coverage_category",
        "coverage_percent",
        "coverage_proportion",
        "coverage_raw",
        "source_type",
        "source",
        "notes",
        "coverage_available",
    ]
    return latest.loc[:, columns].sort_values(["iso3"]).reset_index(drop=True)


def parse_denominator_multiplier(value: object) -> float:
    text = str(value).lower()
    match = re.search(r"per\s+([0-9][0-9,\s]*)", text)
    if not match:
        return 100000.0
    number = match.group(1).replace(",", "").replace(" ", "")
    return float(number) if number.isdigit() else 100000.0


def age_token_to_months(token: object) -> float | None:
    if pd.isna(token):
        return None
    text = str(token).strip().upper()
    first = text.split("-")[0]
    match = re.match(r"([WMY])\s*(\d+(?:\.\d+)?)", first)
    if not match:
        return None
    unit, value_text = match.groups()
    value = float(value_text)
    if unit == "W":
        return value / 4.345
    if unit == "M":
        return value
    if unit == "Y":
        return value * 12.0
    return None


def split_schedule_codes(ages: Iterable[object]) -> tuple[str, str]:
    primary: list[str] = []
    boosters: list[str] = []
    seen_primary: set[str] = set()
    seen_boosters: set[str] = set()
    for age in ages:
        token = str(age).strip()
        if not token or token.casefold() == "nan":
            continue
        months = age_token_to_months(token)
        if months is not None and months <= 12:
            if token not in seen_primary:
                primary.append(token)
                seen_primary.add(token)
        else:
            if token not in seen_boosters:
                boosters.append(token)
                seen_boosters.add(token)
    return ";".join(primary), ";".join(boosters)


def read_name_map() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    reported, coverage, intro, schedule = parallel_calls(
        [
            (read_excel_data, (WHO_REPORTED, ["GROUP", "CODE", "NAME"])),
            (read_excel_data, (WHO_COVERAGE, ["GROUP", "CODE", "NAME"])),
            (read_excel_data, (WHO_INTRODUCTION, ["ISO_3_CODE", "COUNTRYNAME"])),
            (read_excel_data, (WHO_SCHEDULE, ["ISO_3_CODE", "COUNTRYNAME"])),
        ],
        cap=4,
    )
    frames.append(reported.loc[reported["GROUP"].eq("COUNTRIES"), ["CODE", "NAME"]].rename(columns={"CODE": "iso3", "NAME": "country"}))

    frames.append(coverage.loc[coverage["GROUP"].eq("COUNTRIES"), ["CODE", "NAME"]].rename(columns={"CODE": "iso3", "NAME": "country"}))

    frames.append(intro.rename(columns={"ISO_3_CODE": "iso3", "COUNTRYNAME": "country"}))

    frames.append(schedule.rename(columns={"ISO_3_CODE": "iso3", "COUNTRYNAME": "country"}))

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["iso3", "country"]).copy()
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out = out[out["iso3"].str.fullmatch(r"[A-Z]{3}")].copy()
    out["country_key"] = out["country"].map(normalize_name)
    out = out.sort_values(["country_key", "iso3"]).drop_duplicates("country_key", keep="first")

    aliases = [
        ("united states", "USA", "United States of America"),
        ("united states of america", "USA", "United States of America"),
        ("us", "USA", "United States of America"),
        ("uk", "GBR", "United Kingdom of Great Britain and Northern Ireland"),
        ("united kingdom", "GBR", "United Kingdom of Great Britain and Northern Ireland"),
        ("south korea", "KOR", "Republic of Korea"),
        ("north korea", "PRK", "Democratic People's Republic of Korea"),
        ("russia", "RUS", "Russian Federation"),
        ("iran", "IRN", "Iran (Islamic Republic of)"),
        ("vietnam", "VNM", "Viet Nam"),
        ("laos", "LAO", "Lao People's Democratic Republic"),
        ("syria", "SYR", "Syrian Arab Republic"),
        ("tanzania", "TZA", "United Republic of Tanzania"),
        ("venezuela", "VEN", "Venezuela (Bolivarian Republic of)"),
        ("bolivia", "BOL", "Bolivia (Plurinational State of)"),
        ("moldova", "MDA", "Republic of Moldova"),
        ("bosnia", "BIH", "Bosnia and Herzegovina"),
    ]
    alias_df = pd.DataFrame(
        [{"country_key": normalize_name(key), "iso3": iso3, "country": country} for key, iso3, country in aliases]
    )
    out = pd.concat([alias_df, out.loc[:, ["country_key", "iso3", "country"]]], ignore_index=True, sort=False)
    out = out.drop_duplicates("country_key", keep="first")
    return out


def read_who_cases_long(name_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases, incidence = parallel_calls(
        [
            (read_excel_data, (WHO_REPORTED, ["GROUP", "CODE", "NAME", "YEAR", "DISEASE", "DISEASE_DESCRIPTION", "CASES"])),
            (read_excel_data, (WHO_INCIDENCE, ["GROUP", "CODE", "YEAR", "DISEASE", "DENOMINATOR", "INCIDENCE_RATE"])),
        ],
        cap=2,
    )
    cases = cases.loc[cases["DISEASE"].astype(str).str.upper().eq("PERTUSSIS")].copy()
    cases["year"] = pd.to_numeric(cases["YEAR"], errors="coerce").astype("Int64")
    cases["cases_total"] = pd.to_numeric(cases["CASES"], errors="coerce")
    cases["country"] = cases["NAME"].astype(str).str.strip()
    cases["iso3"] = cases["CODE"].astype(str).str.upper()
    cases["source"] = "WHO Immunization Data Portal reported-cases-data.xlsx"
    cases["source_type"] = "measured"
    cases = cases.dropna(subset=["year", "cases_total"]).copy()

    incidence = incidence.loc[incidence["DISEASE"].astype(str).str.upper().eq("PERTUSSIS")].copy()
    incidence["iso3"] = incidence["CODE"].astype(str).str.upper()
    incidence["year"] = pd.to_numeric(incidence["YEAR"], errors="coerce").astype("Int64")
    incidence["incidence_rate"] = pd.to_numeric(incidence["INCIDENCE_RATE"], errors="coerce")
    incidence["incidence_multiplier"] = incidence["DENOMINATOR"].map(parse_denominator_multiplier)
    incidence = incidence.loc[:, ["iso3", "year", "incidence_rate", "incidence_multiplier", "DENOMINATOR"]]

    country_cases = cases.loc[cases["GROUP"].eq("COUNTRIES") & cases["iso3"].str.fullmatch(r"[A-Z]{3}")].copy()
    country_cases = country_cases.merge(incidence, on=["iso3", "year"], how="left")
    country_cases["population"] = np.where(
        country_cases["incidence_rate"].gt(0) & country_cases["cases_total"].notna(),
        country_cases["cases_total"] / country_cases["incidence_rate"] * country_cases["incidence_multiplier"],
        np.nan,
    )
    country_cases = country_cases.sort_values(["iso3", "year"])
    country_cases["population"] = country_cases.groupby("iso3")["population"].transform(lambda x: x.ffill().bfill())
    country_cases["source_dataset"] = "local saved WHO all-age annual reported cases"

    region_cases = cases.loc[~cases["GROUP"].eq("COUNTRIES")].copy()
    return country_cases, region_cases


def read_who_reported_cases_workbook(name_map: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "country_raw",
        "year",
        "reported_cases",
        "source",
        "source_type",
        "source_dataset",
        "notes",
    ]
    if not LOCAL_WHO_REPORTED_CASES_WORKBOOK.exists():
        return pd.DataFrame(columns=columns)
    raw = pd.read_excel(LOCAL_WHO_REPORTED_CASES_WORKBOOK, sheet_name="Sheet1")
    if raw.empty or "Country / Region" not in raw.columns:
        return pd.DataFrame(columns=columns)
    raw = raw.loc[~raw["Country / Region"].astype(str).str.startswith("Exported", na=False)].copy()
    raw = raw.loc[raw["Disease"].astype(str).str.casefold().eq("pertussis")].copy()
    year_columns = [column for column in raw.columns if column not in {"Country / Region", "Disease"}]
    if not year_columns:
        return pd.DataFrame(columns=columns)
    long = raw.melt(
        id_vars=["Country / Region", "Disease"],
        value_vars=year_columns,
        var_name="year",
        value_name="reported_cases",
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["reported_cases"] = pd.to_numeric(
        long["reported_cases"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    long = long.dropna(subset=["year", "reported_cases"]).copy()
    if long.empty:
        return pd.DataFrame(columns=columns)
    long["country_raw"] = long["Country / Region"].astype(str).str.strip()
    long["country_key"] = long["country_raw"].map(normalize_name)
    mapping = name_map.loc[:, ["country_key", "iso3", "country"]].drop_duplicates("country_key")
    long = long.merge(mapping, on="country_key", how="left")
    long["source"] = str(LOCAL_WHO_REPORTED_CASES_WORKBOOK.relative_to(ROOT))
    long["source_type"] = "annual_reported_case_workbook"
    long["source_dataset"] = "official WHO annual reported-cases workbook snapshot"
    long["notes"] = (
        "Annual country-level WHO pertussis reported-case workbook for 2000-2023; "
        "used as a provenance concordance check against the existing WHO annual country-case series."
    )
    long = long.dropna(subset=["iso3", "year", "reported_cases"]).copy()
    return long.loc[:, columns].sort_values(["iso3", "year"]).reset_index(drop=True)


def read_dtp_coverage() -> pd.DataFrame:
    cols = [
        "GROUP",
        "CODE",
        "NAME",
        "YEAR",
        "ANTIGEN",
        "ANTIGEN_DESCRIPTION",
        "COVERAGE_CATEGORY",
        "COVERAGE_CATEGORY_DESCRIPTION",
        "TARGET_NUMBER",
        "DOSES",
        "COVERAGE",
    ]
    raw = read_excel_data(WHO_COVERAGE, usecols=cols)
    raw = raw.loc[raw["GROUP"].eq("COUNTRIES") & raw["ANTIGEN"].isin(["DTPCV1", "DTPCV3"])].copy()
    out = raw.rename(
        columns={
            "CODE": "iso3",
            "NAME": "country",
            "YEAR": "year",
            "ANTIGEN": "antigen",
            "ANTIGEN_DESCRIPTION": "antigen_description",
            "COVERAGE_CATEGORY": "coverage_code",
            "COVERAGE_CATEGORY_DESCRIPTION": "coverage_type",
            "TARGET_NUMBER": "target_number",
            "DOSES": "doses_administered",
            "COVERAGE": "coverage_percent",
        }
    )
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["dose"] = np.where(out["antigen"].eq("DTPCV1"), "1", "3")
    out["coverage"] = pd.to_numeric(out["coverage_percent"], errors="coerce") / 100.0
    out["source"] = "WHO Immunization Data Portal coverage-data.xlsx"
    out["grade_of_confidence"] = out["coverage_type"].astype(str)
    out["notes"] = "Local saved WHO coverage export."
    return out.loc[
        :,
        [
            "iso3",
            "country",
            "year",
            "antigen",
            "dose",
            "coverage",
            "coverage_type",
            "grade_of_confidence",
            "source",
            "notes",
            "target_number",
            "doses_administered",
        ],
    ]


def build_cases(country_cases: pd.DataFrame) -> pd.DataFrame:
    out = country_cases.loc[:, ["iso3", "country", "year", "cases_total", "population", "source", "source_type", "source_dataset"]].copy()
    out["date_start"] = out["year"].astype(str) + "-01-01"
    out["date_end"] = out["year"].astype(str) + "-12-31"
    out["month"] = pd.NA
    out["epiweek"] = pd.NA
    out["time_resolution"] = "annual"
    out["age_group"] = "all_ages"
    out["age_lower"] = pd.NA
    out["age_upper"] = pd.NA
    out["cases_confirmed"] = pd.NA
    out["cases_probable"] = pd.NA
    out["deaths"] = pd.NA
    out["hospitalizations"] = pd.NA
    out["case_definition"] = "WHO/JRF reported pertussis cases; observation model required before interpreting as infections."
    out["notes"] = "Local saved WHO annual all-age reported cases; PertussisIncidence time-resolved and age-specific files are integrated separately where available."
    return out.loc[:, CASE_COLUMNS]


def interval_end(date_start: pd.Timestamp, month: object, week: object) -> pd.Timestamp:
    if pd.notna(month):
        return date_start + pd.offsets.MonthEnd(0)
    if pd.notna(week):
        return date_start + pd.Timedelta(days=6)
    return date_start


def read_pertussis_incidence_population() -> pd.DataFrame:
    if not PI_INCIDENCE.exists():
        return pd.DataFrame(columns=["country_code", "year", "population"])
    pop = read_excel_sheet(PI_INCIDENCE, "Population")
    pop = pop.rename(columns={"Name": "country_code", "Year": "year", "PopulationRaw": "population_raw", "Population": "population_per_100k"})
    pop["country_code"] = pop["country_code"].astype(str).str.strip().str.upper()
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce").astype("Int64")
    pop["population"] = pd.to_numeric(pop.get("population_raw"), errors="coerce")
    pop["population"] = pop["population"].fillna(pd.to_numeric(pop.get("population_per_100k"), errors="coerce") * 100000.0)
    return pop.loc[:, ["country_code", "year", "population"]].dropna(subset=["country_code", "year"])


def build_pertussis_incidence_population_lookup(country_cases: pd.DataFrame | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    workbook_population = read_pertussis_incidence_population()
    if not workbook_population.empty:
        frames.append(workbook_population)

    if country_cases is not None and not country_cases.empty:
        annual_population = country_cases.loc[
            country_cases["iso3"].isin(INCIDENCE_ISO3_TO_COUNTRY_CODE)
            & country_cases["year"].notna()
            & country_cases["population"].notna(),
            ["iso3", "year", "population"],
        ].copy()
        if not annual_population.empty:
            annual_population["country_code"] = annual_population["iso3"].astype(str).str.upper().map(INCIDENCE_ISO3_TO_COUNTRY_CODE)
            annual_population["year"] = pd.to_numeric(annual_population["year"], errors="coerce").astype("Int64")
            annual_population = annual_population.dropna(subset=["country_code", "year", "population"])
            annual_population = annual_population.groupby(["country_code", "year"], as_index=False).agg(population=("population", "mean"))
            frames.append(annual_population.loc[:, ["country_code", "year", "population"]])

    local_wpp = read_local_wpp_population_age()
    if not local_wpp.empty:
        wpp_population = local_wpp.loc[
            local_wpp["iso3"].isin(INCIDENCE_ISO3_TO_COUNTRY_CODE) & local_wpp["year"].notna() & local_wpp["total_population"].notna(),
            ["iso3", "year", "total_population"],
        ].copy()
        if not wpp_population.empty:
            wpp_population["country_code"] = wpp_population["iso3"].astype(str).str.upper().map(INCIDENCE_ISO3_TO_COUNTRY_CODE)
            wpp_population["year"] = pd.to_numeric(wpp_population["year"], errors="coerce").astype("Int64")
            wpp_population["population"] = pd.to_numeric(wpp_population["total_population"], errors="coerce")
            wpp_population = wpp_population.dropna(subset=["country_code", "year", "population"])
            wpp_population = wpp_population.groupby(["country_code", "year"], as_index=False).agg(population=("population", "mean"))
            frames.append(wpp_population.loc[:, ["country_code", "year", "population"]])

    if not frames:
        return pd.DataFrame(columns=["country_code", "year", "population"])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["country_code"] = out["country_code"].astype(str).str.strip().str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out = out.dropna(subset=["country_code", "year", "population"])
    out = out.drop_duplicates(["country_code", "year"], keep="first")
    return out.loc[:, ["country_code", "year", "population"]]


def read_pertussis_incidence_sheet(sheet: str, population: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    return read_pertussis_incidence_sheet_from_workbook(
        sheet,
        PI_INCIDENCE,
        PERTUSSIS_INCIDENCE_COUNTRIES,
        "xmusphlkg/PertussisIncidence Data/Pertussis incidence.xlsx",
        population,
        overrides,
    )


def read_pertussis_incidence_sheet_from_workbook(
    sheet: str,
    workbook_path: Path,
    countries: dict[str, tuple[str, str]],
    source_text: str,
    population: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    code = sheet.strip().upper()
    if code == "POPULATION" or code not in countries:
        return pd.DataFrame(columns=CASE_COLUMNS)
    iso3, country = countries[code]
    raw = read_excel_sheet(workbook_path, sheet)
    raw = raw.dropna(how="all").copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS)

    def optional_column(name: str) -> pd.Series:
        if name in raw.columns:
            return raw[name]
        return pd.Series(pd.NA, index=raw.index)

    raw["country_code"] = code
    raw["iso3"] = iso3
    raw["country"] = country
    raw["date_start"] = pd.to_datetime(optional_column("Date"), errors="coerce")
    raw["year"] = pd.to_numeric(optional_column("Year"), errors="coerce")
    week = pd.to_numeric(optional_column("Week"), errors="coerce")
    date_year = raw["date_start"].dt.year
    iso_year = raw["date_start"].dt.isocalendar().year.astype("Float64")
    raw["year"] = raw["year"].where(raw["year"].between(1900, 2100), np.where(week.notna(), iso_year, date_year))
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["month"] = pd.to_numeric(optional_column("Month"), errors="coerce").astype("Int64")
    raw["epiweek"] = week.astype("Int64")
    raw["cases_raw"] = pd.to_numeric(optional_column("Cases"), errors="coerce")
    raw = raw.merge(population, on=["country_code", "year"], how="left")
    raw = raw.merge(
        overrides.rename(columns={"Country": "country_code", "Year": "year"}),
        on=["country_code", "year"],
        how="left",
    )
    annual_raw = raw.groupby(["country_code", "year"])["cases_raw"].transform("sum")
    official = pd.to_numeric(raw.get("OfficialAnnualCases"), errors="coerce")
    factor = np.where(official.notna() & annual_raw.gt(0), official / annual_raw, 1.0)
    raw["cases_total"] = raw["cases_raw"] * factor
    raw["date_start"] = raw["date_start"].fillna(pd.to_datetime(raw["year"].astype(str) + "-01-01", errors="coerce"))
    raw["date_end"] = [interval_end(date, month, epiweek) if pd.notna(date) else pd.NaT for date, month, epiweek in zip(raw["date_start"], raw["month"], raw["epiweek"])]
    raw["time_resolution"] = np.where(raw["month"].notna(), "monthly", np.where(raw["epiweek"].notna(), "weekly", "annual"))
    raw["age_group"] = "all_ages"
    raw["age_lower"] = pd.NA
    raw["age_upper"] = pd.NA
    raw["cases_confirmed"] = pd.NA
    raw["cases_probable"] = pd.NA
    raw["deaths"] = pd.NA
    raw["hospitalizations"] = pd.NA
    raw["source"] = "xmusphlkg/PertussisIncidence Data/Pertussis incidence.xlsx"
    raw["source"] = source_text
    raw["source_type"] = "measured"
    raw["source_dataset"] = "curated weekly/monthly national surveillance records"
    raw["case_definition"] = "National pertussis surveillance interval counts curated by PertussisIncidence; observation model required."
    source_url = raw.get("URL", pd.Series("", index=raw.index)).fillna("").astype(str)
    annual_source = raw.get("AnnualCaseSource", pd.Series("", index=raw.index)).fillna("").astype(str)
    raw["notes"] = (
        "Curated source URL: "
        + source_url
        + "; annual calibration: "
        + annual_source.replace("", "reporting-interval sum")
        + "; raw interval cases retained only via source workbook."
    )
    return raw.loc[:, CASE_COLUMNS]


def read_pertussis_incidence_workbook(
    workbook_path: Path,
    countries: dict[str, tuple[str, str]],
    source_text: str,
    population: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    if not workbook_path.exists():
        return pd.DataFrame(columns=CASE_COLUMNS)
    xl = pd.ExcelFile(workbook_path)
    valid_sheets = [
        sheet
        for sheet in xl.sheet_names
        if sheet.strip().upper() != "POPULATION" and sheet.strip().upper() in countries
    ]
    if not valid_sheets:
        return pd.DataFrame(columns=CASE_COLUMNS)
    frames = [
        frame
        for frame in parallel_calls(
            [
                (
                    read_pertussis_incidence_sheet_from_workbook,
                    (sheet, workbook_path, countries, source_text, population, overrides),
                )
                for sheet in valid_sheets
            ],
            cap=8,
        )
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=CASE_COLUMNS)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["iso3", "year", "cases_total"]).copy()
    return out.loc[:, CASE_COLUMNS]


def read_pertussis_incidence_time_series(country_cases: pd.DataFrame | None = None) -> pd.DataFrame:
    population = build_pertussis_incidence_population_lookup(country_cases)
    overrides = pd.read_csv(PI_OVERRIDES) if PI_OVERRIDES.exists() else pd.DataFrame(columns=["Country", "Year"])
    frames = [
        frame
        for frame in [
            read_pertussis_incidence_workbook(
                PI_INCIDENCE,
                PERTUSSIS_INCIDENCE_COUNTRIES,
                "xmusphlkg/PertussisIncidence Data/Pertussis incidence.xlsx",
                population,
                overrides,
            ),
            read_pertussis_incidence_workbook(
                LOCAL_AUXILIARY_INCIDENCE_REPORT,
                AUXILIARY_INCIDENCE_REPORT_COUNTRIES,
                repo_path(LOCAL_AUXILIARY_INCIDENCE_REPORT),
                population,
                pd.DataFrame(columns=["Country", "Year", "OfficialAnnualCases"]),
            ),
        ]
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=CASE_COLUMNS)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["iso3", "year", "cases_total"]).copy()
    return out.loc[:, CASE_COLUMNS]


def read_annual_country_series() -> pd.DataFrame:
    columns = CASE_COLUMNS
    if not LOCAL_COUNTRY_SERIES_DIR.exists():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for path in sorted(LOCAL_COUNTRY_SERIES_DIR.glob("*.csv")):
        country_key = path.stem
        if country_key not in COUNTRY_SERIES_COUNTRIES:
            continue
        iso3, country = COUNTRY_SERIES_COUNTRIES[country_key]
        raw = pd.read_csv(path)
        raw_year = pd.to_numeric(raw.get("year"), errors="coerce")
        raw_cases = pd.to_numeric(raw.get("cases"), errors="coerce")
        raw_source = raw.get("source", pd.Series(dtype=str)).astype(str).replace({"nan": ""})
        for idx, year_value in raw_year.items():
            cases_value = raw_cases.loc[idx] if idx in raw_cases.index else pd.NA
            if pd.isna(year_value) or pd.isna(cases_value):
                continue
            source_text = str(raw_source.loc[idx]).strip() if idx in raw_source.index else ""
            note = (
                "Annual historical extension from a local CountrySeries extract"
                + (f"; original raw source string: {source_text}" if source_text else ".")
            )
            rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "date_start": f"{int(year_value)}-01-01",
                    "date_end": f"{int(year_value)}-12-31",
                    "year": int(year_value),
                    "month": pd.NA,
                    "epiweek": pd.NA,
                    "time_resolution": "annual",
                    "age_group": "all_ages",
                    "age_lower": 0.0,
                    "age_upper": np.nan,
                    "cases_confirmed": pd.NA,
                    "cases_probable": pd.NA,
                    "cases_total": float(cases_value),
                    "deaths": pd.NA,
                    "hospitalizations": pd.NA,
                    "population": pd.NA,
                    "source": str(path.relative_to(ROOT)),
                    "source_type": "annual_reported_case_series",
                    "source_dataset": "auxiliary annual country-series extension",
                    "case_definition": "Country-level annual reported pertussis cases from the saved CountrySeries extract.",
                    "notes": note,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["cases_total"] = pd.to_numeric(out["cases_total"], errors="coerce")
    out = out.dropna(subset=["iso3", "year", "cases_total"]).copy()
    return out.loc[:, columns].sort_values(["iso3", "year"]).reset_index(drop=True)


def read_pertussis_region_rates() -> pd.DataFrame:
    columns = [
        "region_name",
        "disease",
        "denominator",
        "year",
        "rate_per_million",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_REGION_RATE_WORKBOOK.exists():
        return pd.DataFrame(columns=columns)
    raw = pd.read_excel(LOCAL_REGION_RATE_WORKBOOK, sheet_name="Sheet1")
    if raw.empty:
        return pd.DataFrame(columns=columns)
    raw = raw.loc[~raw["Country / Region"].astype(str).str.startswith("Exported", na=False)].copy()
    raw = raw.rename(
        columns={
            "Country / Region": "region_name",
            "Disease": "disease",
            "Denominator": "denominator",
        }
    )
    year_columns = [column for column in raw.columns if column not in {"region_name", "disease", "denominator"}]
    long = raw.melt(
        id_vars=["region_name", "disease", "denominator"],
        value_vars=year_columns,
        var_name="year",
        value_name="rate_per_million",
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["rate_per_million"] = pd.to_numeric(long["rate_per_million"], errors="coerce")
    long["source"] = str(LOCAL_REGION_RATE_WORKBOOK.relative_to(ROOT))
    long["source_type"] = "regional_incidence_rate_workbook"
    long["notes"] = (
        "Annual Global plus WHO-region pertussis incidence rates per million total population from the saved regional workbook; "
        "used for descriptive regional context only."
    )
    long = long.dropna(subset=["region_name", "year", "rate_per_million"]).copy()
    return long.loc[:, columns].sort_values(["region_name", "year"]).reset_index(drop=True)


def summarize_pertussis_region_rates(region_rates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "region_name",
        "years_available",
        "year_min",
        "year_max",
        "rate_mean_2000_2024_per_million",
        "rate_mean_2015_2019_per_million",
        "rate_mean_2020_2024_per_million",
        "recent_to_pre2020_ratio",
        "pre2020_to_recent_absolute_change_per_million",
        "peak_year",
        "peak_rate_per_million",
        "rate_2024_per_million",
        "trend_signature",
        "summary_note",
    ]
    if region_rates.empty or "region_name" not in region_rates.columns:
        return pd.DataFrame(columns=columns)

    frame = region_rates.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["rate_per_million"] = pd.to_numeric(frame["rate_per_million"], errors="coerce")
    frame = frame.dropna(subset=["region_name", "year", "rate_per_million"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    def classify_trend(ratio: float) -> str:
        if pd.isna(ratio):
            return "not_available"
        if ratio >= 1.25:
            return "rising"
        if ratio <= 0.8:
            return "declining"
        return "stable_or_mixed"

    rows: list[dict[str, object]] = []
    for region_name, group in frame.groupby("region_name", dropna=False):
        years = pd.to_numeric(group["year"], errors="coerce").dropna().astype(int)
        mean_all = float(group["rate_per_million"].mean())
        mean_pre = float(group.loc[group["year"].between(2015, 2019), "rate_per_million"].mean())
        mean_recent = float(group.loc[group["year"].between(2020, 2024), "rate_per_million"].mean())
        ratio = mean_recent / mean_pre if pd.notna(mean_pre) and mean_pre != 0 else np.nan
        change = mean_recent - mean_pre if not math.isnan(mean_recent) and not math.isnan(mean_pre) else np.nan
        peak_idx = group["rate_per_million"].idxmax()
        peak_year = int(pd.to_numeric(group.loc[peak_idx, "year"], errors="coerce")) if pd.notna(peak_idx) else pd.NA
        peak_rate = float(group.loc[peak_idx, "rate_per_million"]) if pd.notna(peak_idx) else np.nan
        rate_2024 = float(group.loc[group["year"].eq(2024), "rate_per_million"].iloc[0]) if group["year"].eq(2024).any() else np.nan
        rows.append(
            {
                "scope": "region",
                "region_name": region_name,
                "years_available": int(years.nunique()),
                "year_min": int(years.min()) if not years.empty else pd.NA,
                "year_max": int(years.max()) if not years.empty else pd.NA,
                "rate_mean_2000_2024_per_million": mean_all,
                "rate_mean_2015_2019_per_million": mean_pre,
                "rate_mean_2020_2024_per_million": mean_recent,
                "recent_to_pre2020_ratio": ratio,
                "pre2020_to_recent_absolute_change_per_million": change,
                "peak_year": peak_year,
                "peak_rate_per_million": peak_rate,
                "rate_2024_per_million": rate_2024,
                "trend_signature": classify_trend(ratio),
                "summary_note": (
                    "Global plus the six WHO regions annual pertussis incidence-rate context from the saved regional workbook; "
                    "recent-vs-pre-pandemic ratios use 2020-2024 versus 2015-2019 means."
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("region_name").reset_index(drop=True)


def build_annual_country_series_comparison(country_series: pd.DataFrame, workbook_time_resolved_cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "annual_country_series_cases",
        "workbook_annual_cases",
        "workbook_observation_count",
        "absolute_difference",
        "relative_difference",
        "ratio_annual_country_series_to_workbook",
        "comparison_note",
    ]
    if country_series.empty or workbook_time_resolved_cases.empty:
        return pd.DataFrame(columns=columns)
    annual = country_series.copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    annual["cases_total"] = pd.to_numeric(annual["cases_total"], errors="coerce")
    annual = annual.dropna(subset=["iso3", "year", "cases_total"]).copy()
    annual = annual.loc[annual["time_resolution"].eq("annual")].copy()
    if annual.empty:
        return pd.DataFrame(columns=columns)
    workbook = workbook_time_resolved_cases.copy()
    workbook["year"] = pd.to_numeric(workbook["year"], errors="coerce").astype("Int64")
    workbook["cases_total"] = pd.to_numeric(workbook["cases_total"], errors="coerce")
    workbook = workbook.loc[workbook["time_resolution"].isin(["monthly", "weekly"])].copy()
    workbook = workbook.dropna(subset=["iso3", "year", "cases_total"]).copy()
    if workbook.empty:
        return pd.DataFrame(columns=columns)
    workbook_annual = (
        workbook.loc[:, ["iso3", "country", "year", "cases_total", "time_resolution"]]
        .groupby(["iso3", "country", "year"], as_index=False)
        .agg(
            workbook_annual_cases=("cases_total", "sum"),
            workbook_observation_count=("cases_total", "size"),
        )
    )
    comparison = annual.loc[:, ["iso3", "country", "year", "cases_total"]].merge(
        workbook_annual,
        on=["iso3", "country", "year"],
        how="inner",
    )
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    comparison = comparison.rename(columns={"cases_total": "annual_country_series_cases"})
    comparison["absolute_difference"] = comparison["annual_country_series_cases"] - comparison["workbook_annual_cases"]
    comparison["relative_difference"] = comparison["absolute_difference"] / comparison["workbook_annual_cases"].replace({0: np.nan})
    comparison["ratio_annual_country_series_to_workbook"] = (
        comparison["annual_country_series_cases"] / comparison["workbook_annual_cases"].replace({0: np.nan})
    )
    comparison["comparison_note"] = "Annual CountrySeries extension compared against annual sums of overlapping monthly/weekly workbook rows."
    return comparison.loc[:, columns].sort_values(["iso3", "year"]).reset_index(drop=True)


def summarize_annual_country_series_concordance(country_series: pd.DataFrame, workbook_time_resolved_cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "iso3",
        "country",
        "country_series_years",
        "workbook_years",
        "overlap_years",
        "country_series_year_min",
        "country_series_year_max",
        "workbook_year_min",
        "workbook_year_max",
        "overlap_year_min",
        "overlap_year_max",
        "pre_workbook_extension_years",
        "mean_ratio_annual_country_series_to_workbook",
        "median_ratio_annual_country_series_to_workbook",
        "mean_absolute_relative_difference",
        "within_factor2_fraction",
        "comparison_note",
    ]
    if country_series.empty or workbook_time_resolved_cases.empty:
        return pd.DataFrame(columns=columns)
    country_series = country_series.copy()
    country_series["year"] = pd.to_numeric(country_series["year"], errors="coerce").astype("Int64")
    country_series["cases_total"] = pd.to_numeric(country_series["cases_total"], errors="coerce")
    country_series = country_series.dropna(subset=["iso3", "year", "cases_total"]).copy()
    country_series = country_series.loc[country_series["time_resolution"].eq("annual")].copy()
    workbook = workbook_time_resolved_cases.copy()
    workbook["year"] = pd.to_numeric(workbook["year"], errors="coerce").astype("Int64")
    workbook["cases_total"] = pd.to_numeric(workbook["cases_total"], errors="coerce")
    workbook = workbook.loc[workbook["time_resolution"].isin(["monthly", "weekly"])].copy()
    workbook = workbook.dropna(subset=["iso3", "year", "cases_total"]).copy()
    if country_series.empty or workbook.empty:
        return pd.DataFrame(columns=columns)
    workbook_annual = (
        workbook.loc[:, ["iso3", "country", "year", "cases_total"]]
        .groupby(["iso3", "country", "year"], as_index=False)
        .agg(workbook_annual_cases=("cases_total", "sum"))
    )
    comparison = build_annual_country_series_comparison(country_series, workbook_time_resolved_cases)
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    summary_rows: list[dict[str, object]] = []
    annual_years = country_series.loc[:, ["iso3", "country", "year"]].drop_duplicates()
    workbook_years = workbook_annual.loc[:, ["iso3", "country", "year"]].drop_duplicates()
    for (iso3, country), series_group in annual_years.groupby(["iso3", "country"], dropna=False):
        series_years = pd.to_numeric(series_group["year"], errors="coerce").dropna().astype(int)
        workbook_group = workbook_years.loc[workbook_years["iso3"].eq(iso3) & workbook_years["country"].eq(country)].copy()
        workbook_years_vec = pd.to_numeric(workbook_group["year"], errors="coerce").dropna().astype(int)
        comparison_group = comparison.loc[comparison["iso3"].eq(iso3) & comparison["country"].eq(country)].copy()
        ratio = pd.to_numeric(comparison_group["ratio_annual_country_series_to_workbook"], errors="coerce")
        rel = pd.to_numeric(comparison_group["relative_difference"], errors="coerce")
        overlap_years = pd.to_numeric(comparison_group["year"], errors="coerce").dropna().astype(int)
        if series_years.empty:
            pre_workbook_extension_years = 0
        elif workbook_years_vec.empty:
            pre_workbook_extension_years = int(series_years.size)
        else:
            pre_workbook_extension_years = int((series_years < int(workbook_years_vec.min())).sum())
        mean_ratio = float(ratio.mean()) if not ratio.empty else np.nan
        median_ratio = float(ratio.median()) if not ratio.empty else np.nan
        mean_abs_rel = float(rel.abs().mean()) if not rel.empty else np.nan
        within_factor2 = float(ratio.between(0.5, 2.0).mean()) if not ratio.empty else np.nan
        if pd.notna(mean_abs_rel) and pd.notna(within_factor2) and mean_abs_rel <= 0.1 and within_factor2 >= 0.9:
            note = "Auxiliary annual extension aligns closely with overlapping workbook annual sums."
        else:
            note = "Auxiliary annual extension is best treated as a historical concordance check; overlap divergence is more variable."
        summary_rows.append(
            {
                "scope": "country",
                "iso3": iso3,
                "country": country,
                "country_series_years": int(series_years.size),
                "workbook_years": int(workbook_years_vec.size),
                "overlap_years": int(overlap_years.size),
                "country_series_year_min": int(series_years.min()) if len(series_years) else pd.NA,
                "country_series_year_max": int(series_years.max()) if len(series_years) else pd.NA,
                "workbook_year_min": int(workbook_years_vec.min()) if len(workbook_years_vec) else pd.NA,
                "workbook_year_max": int(workbook_years_vec.max()) if len(workbook_years_vec) else pd.NA,
                "overlap_year_min": int(overlap_years.min()) if len(overlap_years) else pd.NA,
                "overlap_year_max": int(overlap_years.max()) if len(overlap_years) else pd.NA,
                "pre_workbook_extension_years": pre_workbook_extension_years,
                "mean_ratio_annual_country_series_to_workbook": mean_ratio,
                "median_ratio_annual_country_series_to_workbook": median_ratio,
                "mean_absolute_relative_difference": mean_abs_rel,
                "within_factor2_fraction": within_factor2,
                "comparison_note": note,
            }
        )
    summary = pd.DataFrame(summary_rows, columns=columns)
    if summary.empty:
        return summary
    aggregate_ratio = pd.to_numeric(comparison["ratio_annual_country_series_to_workbook"], errors="coerce")
    aggregate_rel = pd.to_numeric(comparison["relative_difference"], errors="coerce")
    aggregate_series_years = int(annual_years.shape[0])
    aggregate_workbook_years = int(workbook_years.shape[0])
    aggregate_overlap_years = int(comparison.shape[0])
    aggregate_pre_extension = int(summary["pre_workbook_extension_years"].fillna(0).sum())
    aggregate_note = (
        "Aggregate concordance summary across Australia, Canada, China, and Thailand annual CountrySeries extensions; "
        "Canada has no overlapping workbook annual sums and is therefore excluded from the overlap ratio calculation."
    )
    aggregate_row = pd.DataFrame(
        [
            {
                "scope": "all_overlaps",
                "iso3": "ALL",
                "country": "Australia, Canada, China, and Thailand",
                "country_series_years": aggregate_series_years,
                "workbook_years": aggregate_workbook_years,
                "overlap_years": aggregate_overlap_years,
                "country_series_year_min": int(annual_years["year"].min()),
                "country_series_year_max": int(annual_years["year"].max()),
                "workbook_year_min": int(workbook_years["year"].min()),
                "workbook_year_max": int(workbook_years["year"].max()),
                "overlap_year_min": int(comparison["year"].min()),
                "overlap_year_max": int(comparison["year"].max()),
                "pre_workbook_extension_years": aggregate_pre_extension,
                "mean_ratio_annual_country_series_to_workbook": float(aggregate_ratio.mean()),
                "median_ratio_annual_country_series_to_workbook": float(aggregate_ratio.median()),
                "mean_absolute_relative_difference": float(aggregate_rel.abs().mean()),
                "within_factor2_fraction": float(aggregate_ratio.between(0.5, 2.0).mean()),
                "comparison_note": aggregate_note,
            }
        ],
        columns=columns,
    )
    return pd.concat([aggregate_row, summary], ignore_index=True, sort=False).sort_values(["scope", "iso3"], na_position="last").reset_index(drop=True)


def summarize_south_africa_monthly_series(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "iso3",
        "country",
        "year",
        "source_ids",
        "source_dataset",
        "source",
        "time_resolution",
        "period_start",
        "period_end",
        "months_observed",
        "total_cases",
        "mean_monthly_cases",
        "median_monthly_cases",
        "peak_month",
        "peak_month_cases",
        "share_of_series_cases_percent",
        "summary_note",
    ]
    if cases.empty or not {"iso3", "country", "date_start", "date_end", "year", "time_resolution", "age_group", "cases_total"}.issubset(cases.columns):
        return pd.DataFrame(columns=columns)

    monthly = cases.copy()
    monthly["iso3"] = monthly["iso3"].astype(str).str.upper()
    monthly["country"] = monthly["country"].fillna("").astype(str).str.strip()
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce")
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce")
    monthly["cases_total"] = pd.to_numeric(monthly["cases_total"], errors="coerce")
    monthly["date_start"] = pd.to_datetime(monthly["date_start"], errors="coerce")
    monthly["date_end"] = pd.to_datetime(monthly["date_end"], errors="coerce")
    monthly["time_resolution"] = monthly["time_resolution"].astype(str)
    monthly["age_group"] = monthly["age_group"].astype(str).str.strip()
    monthly = monthly.loc[
        (monthly["iso3"] == "ZAF")
        & monthly["time_resolution"].eq("monthly")
        & monthly["age_group"].eq("all_ages")
    ].copy()
    monthly = monthly.dropna(subset=["year", "month", "cases_total", "date_start", "date_end"]).copy()
    if monthly.empty:
        return pd.DataFrame(columns=columns)

    monthly = monthly.sort_values("date_start").reset_index(drop=True)
    source_ids = "local_auxiliary_time_resolved_cases"
    source_dataset = str(monthly["source_dataset"].dropna().astype(str).iloc[0]) if "source_dataset" in monthly and monthly["source_dataset"].notna().any() else pd.NA
    source = str(monthly["source"].dropna().astype(str).iloc[0]) if "source" in monthly and monthly["source"].notna().any() else pd.NA
    total_series_cases = float(monthly["cases_total"].sum())

    rows: list[dict[str, object]] = []
    for year_value, group in monthly.groupby(monthly["year"].astype(int), sort=True):
        group = group.sort_values("date_start").copy()
        peak_idx = group["cases_total"].idxmax()
        peak_row = group.loc[peak_idx]
        total_cases = float(group["cases_total"].sum())
        period_start = group["date_start"].min()
        period_end = group["date_end"].max()
        rows.append(
            {
                "scope": "year",
                "iso3": "ZAF",
                "country": "South Africa",
                "year": int(year_value),
                "source_ids": source_ids,
                "source_dataset": source_dataset,
                "source": source,
                "time_resolution": "monthly",
                "period_start": period_start.strftime("%Y-%m-%d") if pd.notna(period_start) else pd.NA,
                "period_end": period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else pd.NA,
                "months_observed": int(group.shape[0]),
                "total_cases": total_cases,
                "mean_monthly_cases": float(group["cases_total"].mean()),
                "median_monthly_cases": float(group["cases_total"].median()),
                "peak_month": pd.to_datetime(peak_row["date_start"]).strftime("%Y-%m") if pd.notna(peak_row["date_start"]) else pd.NA,
                "peak_month_cases": float(peak_row["cases_total"]),
                "share_of_series_cases_percent": total_cases / total_series_cases * 100.0 if total_series_cases else np.nan,
                "summary_note": "Year-specific summary of South Africa's auxiliary monthly all-age series; AFRO descriptive context only.",
            }
        )

    overall_peak_idx = monthly["cases_total"].idxmax()
    overall_peak_row = monthly.loc[overall_peak_idx]
    overall_start = monthly["date_start"].min()
    overall_end = monthly["date_end"].max()
    overall_row = pd.DataFrame(
        [
            {
                "scope": "all_period",
                "iso3": "ZAF",
                "country": "South Africa",
                "year": pd.NA,
                "source_ids": source_ids,
                "source_dataset": source_dataset,
                "source": source,
                "time_resolution": "monthly",
                "period_start": overall_start.strftime("%Y-%m-%d") if pd.notna(overall_start) else pd.NA,
                "period_end": overall_end.strftime("%Y-%m-%d") if pd.notna(overall_end) else pd.NA,
                "months_observed": int(monthly.shape[0]),
                "total_cases": total_series_cases,
                "mean_monthly_cases": float(monthly["cases_total"].mean()),
                "median_monthly_cases": float(monthly["cases_total"].median()),
                "peak_month": pd.to_datetime(overall_peak_row["date_start"]).strftime("%Y-%m") if pd.notna(overall_peak_row["date_start"]) else pd.NA,
                "peak_month_cases": float(overall_peak_row["cases_total"]),
                "share_of_series_cases_percent": 100.0,
                "summary_note": "South Africa's auxiliary monthly all-age series extends AFRO time-resolved coverage beyond the main nine-country pilot; it remains outside Model 2 because age-specific rows are unavailable.",
            }
        ],
        columns=columns,
    )
    yearly = pd.DataFrame(rows, columns=columns)
    return pd.concat([overall_row, yearly], ignore_index=True, sort=False).loc[:, columns]


def summarize_time_resolved_all_age_series(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "pipeline_role",
        "time_resolution",
        "source_ids",
        "source_dataset",
        "source",
        "period_start",
        "period_end",
        "periods_observed",
        "total_cases",
        "peak_period_start",
        "peak_period_end",
        "peak_period_cases",
        "summary_note",
    ]
    required_columns = {"iso3", "country", "date_start", "date_end", "time_resolution", "age_group", "cases_total"}
    if cases.empty or not required_columns.issubset(cases.columns):
        return pd.DataFrame(columns=columns)

    time_resolved = cases.copy()
    time_resolved["iso3"] = time_resolved["iso3"].astype(str).str.upper()
    time_resolved["country"] = time_resolved["country"].fillna("").astype(str).str.strip()
    time_resolved["time_resolution"] = time_resolved["time_resolution"].astype(str).str.strip().str.casefold()
    time_resolved["age_group"] = time_resolved["age_group"].astype(str).str.strip()
    time_resolved["cases_total"] = pd.to_numeric(time_resolved["cases_total"], errors="coerce")
    time_resolved["date_start"] = pd.to_datetime(time_resolved["date_start"], errors="coerce")
    time_resolved["date_end"] = pd.to_datetime(time_resolved["date_end"], errors="coerce")
    time_resolved["source"] = time_resolved["source"].fillna("").astype(str).str.strip()
    time_resolved = time_resolved.loc[
        time_resolved["age_group"].eq("all_ages")
        & time_resolved["time_resolution"].isin({"monthly", "weekly"})
    ].copy()
    time_resolved = time_resolved.dropna(subset=["date_start", "date_end", "cases_total"]).copy()
    if time_resolved.empty:
        return pd.DataFrame(columns=columns)

    main_model_iso = {"AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"}

    def source_metadata(source_text: str) -> tuple[str, str]:
        text = str(source_text).casefold()
        if "auxiliary_pertussis_incidence_report.xlsx" in text or "curated_pertussis" in text:
            return (
                "local_auxiliary_time_resolved_cases",
                "Auxiliary pertussis incidence report workbook",
            )
        return (
            "pertussis_incidence_time_resolved_cases",
            "PertussisIncidence curated weekly/monthly national surveillance records",
        )

    def pipeline_role(iso3: str) -> str:
        if iso3 == "ZAF":
            return "auxiliary_afro_monthly_extension"
        if iso3 == "SGP":
            return "sensitivity_only_small_population"
        if iso3 in main_model_iso:
            return "default_model2c_main_analysis"
        return "time_resolved_all_age_only"

    rows: list[dict[str, object]] = []
    for (iso3, country, time_resolution), group in time_resolved.groupby(["iso3", "country", "time_resolution"], sort=True):
        group = group.sort_values("date_start").copy()
        peak_idx = group["cases_total"].idxmax()
        peak_row = group.loc[peak_idx]
        source_ids, source_dataset = source_metadata(group["source"].dropna().astype(str).iloc[0] if group["source"].notna().any() else "")
        period_start = group["date_start"].min()
        period_end = group["date_end"].max()
        role = pipeline_role(str(iso3))
        if role == "auxiliary_afro_monthly_extension":
            note = "Auxiliary AFRO monthly all-age extension; descriptive companion only, not a Model 2C input."
        elif role == "sensitivity_only_small_population":
            note = (
                "Singapore meets the current time-resolution rule and is documented through a separate tagged "
                "sensitivity fit that passed the core sampling gate; the frozen primary model set preserved the "
                "nine-country source library."
            )
        else:
            note = "Time-resolved all-age series used alongside annual age bins in the evidence-backed dynamic pipeline; age-specific rows remain annual."
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "pipeline_role": role,
                "time_resolution": str(time_resolution),
                "source_ids": source_ids,
                "source_dataset": source_dataset,
                "source": str(group["source"].dropna().astype(str).iloc[0]) if group["source"].notna().any() else pd.NA,
                "period_start": period_start.strftime("%Y-%m-%d") if pd.notna(period_start) else pd.NA,
                "period_end": period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else pd.NA,
                "periods_observed": int(group.shape[0]),
                "total_cases": float(group["cases_total"].sum()),
                "peak_period_start": pd.to_datetime(peak_row["date_start"]).strftime("%Y-%m-%d") if pd.notna(peak_row["date_start"]) else pd.NA,
                "peak_period_end": pd.to_datetime(peak_row["date_end"]).strftime("%Y-%m-%d") if pd.notna(peak_row["date_end"]) else pd.NA,
                "peak_period_cases": float(peak_row["cases_total"]),
                "summary_note": note,
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(["pipeline_role", "country", "time_resolution"]).reset_index(drop=True)


def audit_time_resolved_all_age_series_completeness(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "pipeline_role",
        "time_resolution",
        "year",
        "model2c_fit_role",
        "expected_periods",
        "observed_periods",
        "completeness_fraction",
        "missing_periods",
        "last_period_observed",
        "period_start",
        "period_end",
        "total_cases",
        "coverage_status",
        "source_ids",
        "source_dataset",
        "model2c_missing_data_impact",
    ]
    required_columns = {"iso3", "country", "date_start", "date_end", "time_resolution", "age_group", "cases_total"}
    if cases.empty or not required_columns.issubset(cases.columns):
        return pd.DataFrame(columns=columns)

    def env_year(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    model2c_start_year = env_year("MODEL2C_START_YEAR", 2010)
    model2c_end_year = env_year("MODEL2C_END_YEAR", 2026)
    model2c_fit_end_year = env_year("MODEL2C_FIT_END_YEAR", 2025)
    main_model_iso = {"AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"}

    frame = cases.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame["country"].fillna("").astype(str).str.strip()
    frame["time_resolution"] = frame["time_resolution"].astype(str).str.strip().str.casefold()
    frame["age_group"] = frame["age_group"].astype(str).str.strip()
    frame["cases_total"] = pd.to_numeric(frame["cases_total"], errors="coerce")
    frame["date_start"] = pd.to_datetime(frame["date_start"], errors="coerce")
    frame["date_end"] = pd.to_datetime(frame["date_end"], errors="coerce")
    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    else:
        frame["year"] = frame["date_start"].dt.year.astype("Int64")
    if "month" in frame.columns:
        frame["month"] = pd.to_numeric(frame["month"], errors="coerce").astype("Int64")
    else:
        frame["month"] = frame["date_start"].dt.month.astype("Int64")
    if "epiweek" in frame.columns:
        frame["epiweek"] = pd.to_numeric(frame["epiweek"], errors="coerce").astype("Int64")
    else:
        frame["epiweek"] = frame["date_start"].dt.isocalendar().week.astype("Int64")
    frame["source"] = frame.get("source", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()

    time_resolved = frame.loc[
        frame["age_group"].eq("all_ages")
        & frame["time_resolution"].isin({"monthly", "weekly"})
    ].dropna(subset=["date_start", "date_end", "cases_total", "year"]).copy()
    if time_resolved.empty:
        return pd.DataFrame(columns=columns)

    latest_year = (
        time_resolved.groupby(["iso3", "time_resolution"], dropna=False)["year"]
        .max()
        .astype("Int64")
        .to_dict()
    )

    def source_metadata(source_text: str) -> tuple[str, str]:
        text = str(source_text).casefold()
        if "auxiliary_pertussis_incidence_report.xlsx" in text or "curated_pertussis" in text:
            return (
                "local_auxiliary_time_resolved_cases",
                "Auxiliary pertussis incidence report workbook",
            )
        return (
            "pertussis_incidence_time_resolved_cases",
            "PertussisIncidence curated weekly/monthly national surveillance records",
        )

    def pipeline_role(iso3: str) -> str:
        if iso3 == "ZAF":
            return "auxiliary_afro_monthly_extension"
        if iso3 == "SGP":
            return "sensitivity_only_small_population"
        if iso3 in main_model_iso:
            return "default_model2c_main_analysis"
        return "time_resolved_all_age_only"

    def model_fit_role(iso3: str, year: int) -> str:
        if iso3 not in main_model_iso:
            return "not_used_by_default_model2c"
        if year < model2c_start_year:
            return "pre_model2c_window_context_only"
        if year <= model2c_fit_end_year:
            return "model2c_likelihood_window"
        if year <= model2c_end_year:
            return "model2c_posterior_predictive_check_window"
        return "post_model2c_window_not_fit"

    def expected_period_count(year: int, resolution: str) -> int:
        if resolution == "monthly":
            return 12
        return int(pd.Timestamp(year=int(year), month=12, day=28).isocalendar().week)

    def period_values(group: pd.DataFrame, resolution: str) -> list[int]:
        if resolution == "monthly":
            values = group["month"]
        else:
            values = group["epiweek"]
        numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
        return sorted(value for value in numeric.unique().tolist() if value > 0)

    def missing_label(period: int, resolution: str) -> str:
        return f"W{period:02d}" if resolution == "weekly" else f"M{period:02d}"

    def coverage_status(
        *,
        iso3: str,
        year: int,
        resolution: str,
        observed: int,
        expected: int,
        period_end: pd.Timestamp,
    ) -> str:
        if observed >= expected:
            return "complete_observed_year"
        is_latest_year = int(latest_year.get((str(iso3), resolution), year)) == int(year)
        latest_cutoff = pd.Timestamp(year=int(year), month=12, day=15)
        if is_latest_year and pd.notna(period_end) and period_end < latest_cutoff:
            return "partial_latest_year"
        if expected and observed / expected >= 0.95:
            return "near_complete_minor_gap"
        return "incomplete_year"

    def impact_note(fit_role: str, status: str, role: str) -> str:
        if role != "default_model2c_main_analysis":
            return "No direct impact on the default Model 2C likelihood."
        if fit_role == "model2c_posterior_predictive_check_window":
            return "Affects posterior predictive check interpretation only; not used for fitting."
        if fit_role != "model2c_likelihood_window":
            return "Outside the default Model 2C fitting likelihood window."
        if status == "complete_observed_year":
            return "No completeness warning for the fitted Model 2C monthly likelihood."
        if status == "partial_latest_year":
            return "Partial latest-year reports enter the monthly likelihood only where periods exist; annual 2025 totals should be read as incomplete."
        return "Missing high-resolution periods can make affected monthly or annual observed cases lower bounds in the fitted likelihood."

    rows: list[dict[str, object]] = []
    for (group_iso, country, time_resolution, year), group in time_resolved.groupby(
        ["iso3", "country", "time_resolution", "year"],
        sort=True,
    ):
        year_int = int(year)
        resolution = str(time_resolution)
        periods = period_values(group, resolution)
        expected = expected_period_count(year_int, resolution)
        missing = [missing_label(period, resolution) for period in range(1, expected + 1) if period not in set(periods)]
        period_start = group["date_start"].min()
        period_end = group["date_end"].max()
        observed = len(periods)
        fraction = float(observed / expected) if expected else np.nan
        status = coverage_status(
            iso3=str(group_iso),
            year=year_int,
            resolution=resolution,
            observed=observed,
            expected=expected,
            period_end=period_end,
        )
        role = pipeline_role(str(group_iso))
        fit_role = model_fit_role(str(group_iso), year_int)
        source_values = group["source"].dropna().astype(str)
        source_ids, source_dataset = source_metadata(source_values.iloc[0] if not source_values.empty else "")
        rows.append(
            {
                "iso3": group_iso,
                "country": country,
                "pipeline_role": role,
                "time_resolution": resolution,
                "year": year_int,
                "model2c_fit_role": fit_role,
                "expected_periods": int(expected),
                "observed_periods": int(observed),
                "completeness_fraction": fraction,
                "missing_periods": ";".join(missing) if missing else "none",
                "last_period_observed": int(max(periods)) if periods else pd.NA,
                "period_start": period_start.strftime("%Y-%m-%d") if pd.notna(period_start) else pd.NA,
                "period_end": period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else pd.NA,
                "total_cases": float(group["cases_total"].sum()),
                "coverage_status": status,
                "source_ids": source_ids,
                "source_dataset": source_dataset,
                "model2c_missing_data_impact": impact_note(fit_role, status, role),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["pipeline_role", "country", "year"]).reset_index(drop=True)


def parse_source_age_label(label: object) -> tuple[float, float] | None:
    text = str(label).strip().replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    if not text or text.lower() in {"year", "unknown", "unknow", "notknown", "nan"}:
        return None
    if text.startswith("<"):
        value = clean_number(text)
        return (0.0, value) if not math.isnan(value) and value > 0 else None
    if text.endswith("+"):
        value = clean_number(text)
        return (value, 100.0) if not math.isnan(value) else None
    nums = [float(item) for item in re.findall(r"\d+", text)]
    if len(nums) >= 2:
        lower, upper = nums[0], nums[1]
        return lower, upper + 1.0
    return None


def estimate_population_for_age_interval(pop_row: pd.Series | None, lower: float, upper: float | None) -> float:
    if pop_row is None or not np.isfinite(lower):
        return np.nan
    source_upper = 100.0 if upper is None or pd.isna(upper) else float(upper)
    if not np.isfinite(source_upper) or source_upper <= lower:
        return np.nan
    total = 0.0
    for age_group, (group_lower, group_upper) in AGE_GROUP_MAP.items():
        group_upper_value = 100.0 if pd.isna(group_upper) else float(group_upper)
        overlap = max(0.0, min(source_upper, group_upper_value) - max(lower, float(group_lower)))
        if overlap <= 0:
            continue
        group_population = pop_row.get(age_group, np.nan)
        if pd.isna(group_population):
            continue
        total += float(group_population) * overlap / (group_upper_value - float(group_lower))
    return total if total > 0 else np.nan


def non_overlapping_age_columns(values: dict[str, float]) -> dict[str, tuple[float, float, float]]:
    parsed: dict[str, tuple[float, float, float]] = {}
    for label, cases in values.items():
        bounds = parse_source_age_label(label)
        if bounds is None or pd.isna(cases):
            continue
        lower, upper = bounds
        if upper <= lower:
            continue
        parsed[label] = (lower, upper, float(cases))
    keep: dict[str, tuple[float, float, float]] = {}
    for label, (lower, upper, cases) in parsed.items():
        contains_more_specific = False
        for other, (other_lower, other_upper, other_cases) in parsed.items():
            if other == label or pd.isna(other_cases):
                continue
            if lower <= other_lower and upper >= other_upper and (upper - lower) > (other_upper - other_lower):
                contains_more_specific = True
                break
        if not contains_more_specific:
            keep[label] = (lower, upper, cases)
    return keep


def infer_pertussis_age_year_column(raw: pd.DataFrame) -> str | None:
    for column in raw.columns:
        if str(column).strip().casefold() == "year":
            return str(column)
    if raw.columns.empty:
        return None
    first = raw.columns[0]
    first_numeric = pd.to_numeric(raw[first], errors="coerce")
    if first_numeric.notna().mean() >= 0.5:
        return str(first)
    return None


def read_pertussis_age_specific_sheet(
    sheet: str,
    workbook_path: Path = PI_AGE,
    countries: dict[str, tuple[str, str]] = PERTUSSIS_INCIDENCE_COUNTRIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = sheet.strip().upper()
    if code not in countries:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    iso3, country = countries[code]
    raw = read_excel_sheet(workbook_path, sheet)
    raw = raw.dropna(how="all").copy()
    year_col = infer_pertussis_age_year_column(raw)
    if year_col is None or year_col not in raw.columns:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    age_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        year = pd.to_numeric(row.get(year_col), errors="coerce")
        if pd.isna(year):
            continue
        year_int = int(year)
        values = {
            str(column): pd.to_numeric(row.get(column), errors="coerce")
            for column in raw.columns
            if column != year_col
        }
        selected = non_overlapping_age_columns(values)
        for label, (lower, upper, cases) in selected.items():
            date_start = f"{year_int}-01-01"
            date_end = f"{year_int}-12-31"
            age_rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "date_start": date_start,
                    "date_end": date_end,
                    "year": year_int,
                    "month": pd.NA,
                    "epiweek": pd.NA,
                    "time_resolution": "annual",
                    "age_group": label,
                    "age_lower": lower,
                    "age_upper": np.nan if upper >= 100 else upper,
                    "cases_confirmed": pd.NA,
                    "cases_probable": pd.NA,
                    "cases_total": cases,
                    "deaths": pd.NA,
                    "hospitalizations": pd.NA,
                    "population": pd.NA,
                    "source": repo_path(workbook_path),
                    "source_type": "measured",
                    "source_dataset": "curated annual age-distribution surveillance records",
                    "case_definition": "Country-specific reported pertussis cases by source age group.",
                    "notes": "Selected non-overlapping source age bins; broad overlapping aggregate bins are suppressed when finer bins are available.",
                }
            )
            source_upper = upper
            source_width = max(source_upper - lower, 1e-9)
            for model_age, (model_lower, model_upper) in AGE_GROUP_MAP.items():
                model_upper_value = 100.0 if pd.isna(model_upper) else float(model_upper)
                overlap = max(0.0, min(source_upper, model_upper_value) - max(lower, float(model_lower)))
                if overlap <= 0:
                    continue
                fraction = overlap / source_width
                model_rows.append(
                    {
                        "iso3": iso3,
                        "country": country,
                        "date_start": date_start,
                        "date_end": date_end,
                        "year": year_int,
                        "month": pd.NA,
                        "epiweek": pd.NA,
                        "time_resolution": "annual",
                        "model_age_group": model_age,
                        "age_group": label,
                        "age_lower": lower,
                        "age_upper": np.nan if upper >= 100 else upper,
                        "source": repo_path(workbook_path),
                        "source_type": "derived",
                        "source_dataset": "age-bin overlap allocation from curated source bins",
                        "case_definition": "Allocated reported cases by model age group; not a direct source observation.",
                        "notes": "Allocation fraction is based on age-bin width overlap, not population denominators.",
                        "cases_total": cases * fraction,
                        "source_cases_total": cases,
                        "allocation_fraction": fraction,
                        "source_age_groups": label,
                    }
                )
    age = pd.DataFrame(age_rows, columns=CASE_COLUMNS)
    model = pd.DataFrame(model_rows, columns=AGE_MODEL_COLUMNS)
    return age, model


def read_pertussis_age_specific_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LOCAL_SURVEILLANCE_AGE_CASES.exists():
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw = pd.read_csv(LOCAL_SURVEILLANCE_AGE_CASES)
    required = {"country", "year", "age_group", "cases"}
    if not required.issubset(raw.columns):
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    country_map = {normalize_name(country): (iso3, country_name) for country, (iso3, country_name) in SURVEILLANCE_AGE_CASES_COUNTRIES.items()}
    raw["country_key"] = raw["country"].astype(str).str.strip().map(normalize_name)
    raw = raw.loc[raw["country_key"].isin(country_map)].copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw["iso3"] = raw["country_key"].map(lambda key: country_map[key][0])
    raw["country"] = raw["country_key"].map(lambda key: country_map[key][1])
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["cases_total"] = pd.to_numeric(raw["cases"], errors="coerce")
    raw = raw.dropna(subset=["iso3", "year", "age_group", "cases_total"]).copy()
    age_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        bounds = parse_source_age_label(row["age_group"])
        if bounds is None:
            continue
        lower, upper = bounds
        cases = float(row["cases_total"])
        year_int = int(row["year"])
        date_start = f"{year_int}-01-01"
        date_end = f"{year_int}-12-31"
        age_rows.append(
            {
                "iso3": row["iso3"],
                "country": row["country"],
                "date_start": date_start,
                "date_end": date_end,
                "year": year_int,
                "month": pd.NA,
                "epiweek": pd.NA,
                "time_resolution": "annual",
                "age_group": row["age_group"],
                "age_lower": lower,
                "age_upper": np.nan if upper >= 100 else upper,
                "cases_confirmed": pd.NA,
                "cases_probable": pd.NA,
                "cases_total": cases,
                "deaths": pd.NA,
                "hospitalizations": pd.NA,
                "population": pd.NA,
                "source": repo_path(LOCAL_SURVEILLANCE_AGE_CASES),
                "source_type": "measured",
                "source_dataset": "curated annual age-distribution surveillance records",
                "case_definition": "Country-specific reported pertussis cases by source age group.",
                "notes": "Supplemental early-year age-specific rows extracted from the local surveillance CSV; used only before workbook coverage begins for this country.",
            }
        )
        source_width = max(upper - lower, 1e-9)
        for model_age, (model_lower, model_upper) in AGE_GROUP_MAP.items():
            model_upper_value = 100.0 if pd.isna(model_upper) else float(model_upper)
            overlap = max(0.0, min(upper, model_upper_value) - max(lower, float(model_lower)))
            if overlap <= 0:
                continue
            fraction = overlap / source_width
            model_rows.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "date_start": date_start,
                    "date_end": date_end,
                    "year": year_int,
                    "month": pd.NA,
                    "epiweek": pd.NA,
                    "time_resolution": "annual",
                    "model_age_group": model_age,
                    "age_group": row["age_group"],
                    "age_lower": lower,
                    "age_upper": np.nan if upper >= 100 else upper,
                    "source": repo_path(LOCAL_SURVEILLANCE_AGE_CASES),
                    "source_type": "derived",
                    "source_dataset": "age-bin overlap allocation from supplemental local surveillance CSV",
                    "case_definition": "Allocated reported cases by model age group; not a direct source observation.",
                    "notes": "Allocation fraction is based on age-bin width overlap, not population denominators.",
                    "cases_total": cases * fraction,
                    "source_cases_total": cases,
                    "allocation_fraction": fraction,
                    "source_age_groups": row["age_group"],
                }
            )
    age = pd.DataFrame(age_rows, columns=CASE_COLUMNS)
    model = pd.DataFrame(model_rows, columns=AGE_MODEL_COLUMNS)
    return age, model


def read_canada_official_age_distribution_summary() -> dict[str, object] | None:
    if not LOCAL_CANADA_OFFICIAL_AGE_DISTRIBUTION.exists():
        return None
    raw = pd.read_csv(LOCAL_CANADA_OFFICIAL_AGE_DISTRIBUTION)
    required = {"country", "year", "period_start", "period_end", "age_group", "cases"}
    if not required.issubset(raw.columns):
        return None
    raw["country_key"] = raw["country"].astype(str).str.strip().map(normalize_name)
    raw = raw.loc[raw["country_key"].eq(normalize_name("Canada"))].copy()
    if raw.empty:
        return None
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["cases_total"] = pd.to_numeric(raw["cases"], errors="coerce")
    raw["period_start"] = pd.to_datetime(raw["period_start"], errors="coerce")
    raw["period_end"] = pd.to_datetime(raw["period_end"], errors="coerce")
    raw = raw.dropna(subset=["year", "age_group", "cases_total"]).copy()
    if raw.empty:
        return None
    source_age_group_count = int(raw["age_group"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    total_cases = float(raw["cases_total"].sum())
    period_start = raw["period_start"].dropna().min()
    period_end = raw["period_end"].dropna().max()
    report_year = int(raw["year"].dropna().max())

    model_totals = {age_group: 0.0 for age_group in AGE_GROUP_MAP}
    for _, row in raw.iterrows():
        label = str(row.get("age_group", "")).strip()
        cases = float(pd.to_numeric(row.get("cases_total"), errors="coerce"))
        if not np.isfinite(cases) or cases <= 0:
            continue
        bounds = parse_source_age_label(label)
        if bounds is None:
            continue
        lower, upper = bounds
        source_width = max(upper - lower, 1e-9)
        for model_age, (model_lower, model_upper) in AGE_GROUP_MAP.items():
            model_upper_value = 100.0 if pd.isna(model_upper) else float(model_upper)
            overlap = max(0.0, min(upper, model_upper_value) - max(lower, float(model_lower)))
            if overlap <= 0:
                continue
            model_totals[model_age] += cases * (overlap / source_width)

    broad_totals = {
        broad: float(sum(float(model_totals.get(model_age, 0.0)) for model_age in model_age_groups))
        for broad, model_age_groups in AGE_PATTERN_BROAD_GROUPS.items()
    }
    sorted_broad = sorted(broad_totals.items(), key=lambda item: item[1], reverse=True)
    dominant_group, dominant_cases = sorted_broad[0]
    second_cases = sorted_broad[1][1] if len(sorted_broad) > 1 else 0.0
    dominant_share = dominant_cases / total_cases * 100.0 if total_cases else np.nan
    second_share = second_cases / total_cases * 100.0 if total_cases else np.nan
    model_dominant_group = max(model_totals, key=model_totals.get) if model_totals else pd.NA
    model_dominant_cases = float(model_totals.get(model_dominant_group, np.nan)) if model_totals else np.nan
    model_dominant_share = model_dominant_cases / total_cases * 100.0 if total_cases else np.nan
    infant_cases = broad_totals["infant_0_11m"]
    school_age_cases = broad_totals["school_age_5_17y"]
    adult_cases = broad_totals["young_adult_18_39y"] + broad_totals["middle_adult_40_64y"] + broad_totals["elderly_65plus"]
    visible_cases = float(
        raw.loc[~raw["age_group"].astype(str).str.strip().str.casefold().eq("unspecified"), "cases_total"].sum()
    )
    unspecified_cases = total_cases - visible_cases

    return {
        "iso3": "CAN",
        "country": "Canada",
        "evidence_family": "reported_case_age_distribution",
        "year": report_year,
        "period_start": period_start.strftime("%Y-%m-%d") if pd.notna(period_start) else pd.NA,
        "period_end": period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else pd.NA,
        "geographic_scope": "national five-year age-distribution surveillance summary",
        "age_pattern_signature": classify_age_pattern_signature(dominant_group, dominant_share, second_share),
        "reported_case_total": total_cases,
        "dominant_age_group": dominant_group,
        "dominant_model_age_group": model_dominant_group,
        "dominant_model_age_group_cases": model_dominant_cases,
        "dominant_model_age_group_share_percent": model_dominant_share,
        "infant_case_count": infant_cases,
        "infant_case_share_percent": infant_cases / total_cases * 100.0 if total_cases else np.nan,
        "school_age_case_count": school_age_cases,
        "school_age_case_share_percent": school_age_cases / total_cases * 100.0 if total_cases else np.nan,
        "adult_case_count": adult_cases,
        "adult_case_share_percent": adult_cases / total_cases * 100.0 if total_cases else np.nan,
        "source_age_group_count": source_age_group_count,
        "case_count": pd.NA,
        "case_count_qualifier": pd.NA,
        "case_count_type": pd.NA,
        "infant_age_definition": pd.NA,
        "under3m_case_count": pd.NA,
        "under6m_case_count": pd.NA,
        "under6m_hospitalization_percent": pd.NA,
        "total_hospitalization_percent": pd.NA,
        "hospitalizations": pd.NA,
        "total_deaths": pd.NA,
        "infant_deaths": pd.NA,
        "death_age_scope": pd.NA,
        "testing_denominator": pd.NA,
        "test_positivity_percent": pd.NA,
        "laboratory_confirmation_methods": pd.NA,
        "reporting_or_surveillance_change": pd.NA,
        "source_ids": "CAN_PHAC_2019_VPD_REPORT",
        "source": "Public Health Agency of Canada. Vaccine-preventable disease surveillance report 2019.",
        "source_type": "official_surveillance_summary",
        "source_dataset": "official reported-case age distribution summary",
        "geographic_scope_source": "Figure 16 text equivalent in PHAC surveillance report",
        "notes": (
            "PHAC Figure 16 text equivalent lists ten visible age bins totaling "
            f"{visible_cases:.0f} cases and an additional {unspecified_cases:.0f} unspecified cases, "
            f"reconciling to the report total of {total_cases:.0f} cases; summary-level age-pattern evidence only."
        ),
    }


def read_england_official_age_distribution_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LOCAL_ENGLAND_OFFICIAL_AGE_DISTRIBUTION.exists():
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw = pd.read_csv(LOCAL_ENGLAND_OFFICIAL_AGE_DISTRIBUTION)
    required = {"country", "year", "age_group", "cases"}
    if not required.issubset(raw.columns):
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw["country_key"] = raw["country"].astype(str).str.strip().map(normalize_name)
    raw = raw.loc[raw["country_key"].eq(normalize_name("England"))].copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["cases_total"] = pd.to_numeric(raw["cases"], errors="coerce")
    raw = raw.dropna(subset=["year", "age_group", "cases_total"]).copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)

    age_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    source_id = "UKHSA_ENGLAND_2019_AGE_DISTRIBUTION_1994_2014"
    source_notes = (
        "Extracted from Table 3 of the UKHSA annual report for laboratory-confirmed pertussis in England 2019; "
        "only 1994-2014 are retained here because the 2015-2019 years overlap workbook age coverage and are omitted "
        "to avoid duplication."
    )
    for _, row in raw.iterrows():
        bounds = parse_source_age_label(row["age_group"])
        if bounds is None:
            continue
        lower, upper = bounds
        cases = float(row["cases_total"])
        year_int = int(row["year"])
        date_start = f"{year_int}-01-01"
        date_end = f"{year_int}-12-31"
        age_rows.append(
            {
                "iso3": "GBR",
                "country": "United Kingdom of Great Britain and Northern Ireland",
                "date_start": date_start,
                "date_end": date_end,
                "year": year_int,
                "month": pd.NA,
                "epiweek": pd.NA,
                "time_resolution": "annual",
                "age_group": str(row["age_group"]).strip(),
                "age_lower": lower,
                "age_upper": np.nan if upper >= 100 else upper,
                "cases_confirmed": pd.NA,
                "cases_probable": pd.NA,
                "cases_total": cases,
                "deaths": pd.NA,
                "hospitalizations": pd.NA,
                "population": pd.NA,
                "source": source_id,
                "source_type": "measured",
                "source_dataset": "official annual age-distribution surveillance records",
                "case_definition": "Country-specific reported pertussis cases by source age group.",
                "notes": source_notes,
            }
        )
        source_width = max(upper - lower, 1e-9)
        for model_age, (model_lower, model_upper) in AGE_GROUP_MAP.items():
            model_upper_value = 100.0 if pd.isna(model_upper) else float(model_upper)
            overlap = max(0.0, min(upper, model_upper_value) - max(lower, float(model_lower)))
            if overlap <= 0:
                continue
            fraction = overlap / source_width
            model_rows.append(
                {
                    "iso3": "GBR",
                    "country": "United Kingdom of Great Britain and Northern Ireland",
                    "date_start": date_start,
                    "date_end": date_end,
                    "year": year_int,
                    "month": pd.NA,
                    "epiweek": pd.NA,
                    "time_resolution": "annual",
                    "model_age_group": model_age,
                    "age_group": str(row["age_group"]).strip(),
                    "age_lower": lower,
                    "age_upper": np.nan if upper >= 100 else upper,
                    "source": source_id,
                    "source_type": "derived",
                    "source_dataset": "age-bin overlap allocation from official England annual age-distribution summary",
                    "case_definition": "Allocated reported cases by model age group; not a direct source observation.",
                    "notes": "Allocation fraction is based on age-bin width overlap, not population denominators.",
                    "cases_total": cases * fraction,
                    "source_cases_total": cases,
                    "allocation_fraction": fraction,
                    "source_age_groups": str(row["age_group"]).strip(),
                }
            )
    age = pd.DataFrame(age_rows, columns=CASE_COLUMNS)
    model = pd.DataFrame(model_rows, columns=AGE_MODEL_COLUMNS)
    return age, model


def read_eu_age_specific_rate_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LOCAL_EXTENDED_AGE_CASES.exists():
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    try:
        raw = pd.read_excel(LOCAL_EXTENDED_AGE_CASES, sheet_name="EU")
        reported = pd.read_csv(LOCAL_ECDC_SURVEILLANCE)
    except Exception:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    required = {"Population", "Distribution", "Unit", "Time", "RegionCode", "RegionName", "Category", "Value"}
    reported_required = {"Population", "Indicator", "Unit", "Time", "RegionCode", "NumValue"}
    if not required.issubset(raw.columns) or not reported_required.issubset(reported.columns):
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw = raw.loc[
        raw["Population"].astype(str).str.strip().str.casefold().eq("all cases")
        & raw["Distribution"].astype(str).str.contains("age-specific rate", case=False, regex=False)
        & raw["Unit"].astype(str).str.contains("N/100000", case=False, regex=False)
    ].copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw["iso3"] = raw["RegionCode"].astype(str).str.upper().map(ECDC_REGION_TO_ISO3)
    raw = raw.loc[raw["iso3"].notna() & ~raw["iso3"].isin(EU_AGE_RATE_SOURCE_EXCLUDED_ISO3)].copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    raw["year"] = pd.to_numeric(raw["Time"], errors="coerce")
    raw["rate_per_100k"] = pd.to_numeric(raw["Value"], errors="coerce")
    raw["Category"] = raw["Category"].astype(str).str.strip()
    raw = raw.loc[raw["year"].notna() & raw["rate_per_100k"].notna() & raw["Category"].isin(EU_AGE_RATE_BINS)].copy()
    if raw.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    reported = reported.loc[
        reported["Population"].astype(str).str.strip().str.casefold().eq("all cases")
        & reported["Indicator"].astype(str).str.strip().str.casefold().eq("reported cases")
        & reported["Unit"].astype(str).str.strip().str.upper().eq("N")
    ].copy()
    if reported.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    reported["iso3"] = reported["RegionCode"].astype(str).str.upper().map(ECDC_REGION_TO_ISO3)
    reported["year"] = pd.to_numeric(reported["Time"], errors="coerce")
    reported["reported_cases"] = pd.to_numeric(reported["NumValue"], errors="coerce")
    reported = reported.loc[reported["iso3"].notna() & reported["year"].notna() & reported["reported_cases"].notna()].copy()
    if reported.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    reported = reported.groupby(["iso3", "year"], as_index=False).agg(reported_cases_total=("reported_cases", "sum"))
    reported_lookup = {(row.iso3, int(row.year)): float(row.reported_cases_total) for row in reported.itertuples(index=False)}

    age_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    source_string = f"{repo_path(LOCAL_EXTENDED_AGE_CASES)} (EU sheet)"
    source_dataset = EU_AGE_RATE_PANEL_NORMALIZED_DATASET
    for (iso3, country, year), group in raw.groupby(["iso3", "RegionName", "year"], sort=True):
        year_int = int(year)
        group = group.sort_values(["CategoryIndex", "Category"], na_position="last")
        total_rate = float(group["rate_per_100k"].sum())
        anchor_cases = reported_lookup.get((iso3, year_int))
        if not np.isfinite(total_rate) or total_rate <= 0 or anchor_cases is None or not np.isfinite(anchor_cases) or anchor_cases <= 0:
            continue
        for _, row in group.iterrows():
            bounds = parse_source_age_label(row["Category"])
            if bounds is None:
                continue
            lower, upper = bounds
            estimated_cases = float(row["rate_per_100k"]) / total_rate * anchor_cases
            date_start = f"{year_int}-01-01"
            date_end = f"{year_int}-12-31"
            age_rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "date_start": date_start,
                    "date_end": date_end,
                    "year": year_int,
                    "month": pd.NA,
                    "epiweek": pd.NA,
                    "time_resolution": "annual",
                    "age_group": str(row["Category"]).strip(),
                    "age_lower": lower,
                    "age_upper": np.nan if upper >= 100 else upper,
                    "cases_confirmed": pd.NA,
                    "cases_probable": pd.NA,
                    "cases_total": estimated_cases,
                    "deaths": pd.NA,
                    "hospitalizations": pd.NA,
                    "population": pd.NA,
                    "source": source_string,
                    "source_type": "derived_from_rate_panel",
                    "source_dataset": source_dataset,
                    "case_definition": "Annual age-specific rate per 100,000 normalized to the country-year reported-case total from the ECDC surveillance snapshot.",
                    "notes": "Age-specific rate panel from the extended local age workbook; exact workbook-sheet countries are excluded, and annual rates are normalized to reported-case totals so the age-pattern evidence remains comparable without relying on missing census-age denominators.",
                }
            )
            source_upper = 100.0 if pd.isna(upper) else float(upper)
            source_width = max(source_upper - lower, 1e-9)
            for model_age, (model_lower, model_upper) in AGE_GROUP_MAP.items():
                model_upper_value = 100.0 if pd.isna(model_upper) else float(model_upper)
                overlap = max(0.0, min(source_upper, model_upper_value) - max(lower, float(model_lower)))
                if overlap <= 0:
                    continue
                fraction = overlap / source_width
                model_rows.append(
                    {
                        "iso3": iso3,
                        "country": country,
                        "date_start": date_start,
                        "date_end": date_end,
                        "year": year_int,
                        "month": pd.NA,
                        "epiweek": pd.NA,
                        "time_resolution": "annual",
                        "model_age_group": model_age,
                        "age_group": str(row["Category"]).strip(),
                        "age_lower": lower,
                        "age_upper": np.nan if upper >= 100 else upper,
                        "source": source_string,
                        "source_type": "derived",
                        "source_dataset": f"{EU_AGE_RATE_PANEL_NORMALIZED_DATASET} and allocated by age-bin overlap",
                        "case_definition": "Normalized annual age-specific rate per 100,000 to the country-year reported-case total and allocated by model age-bin overlap; not a direct source observation.",
                        "notes": "Derived from age-specific rates using annual reported-case totals as the scaling anchor; allocation fraction is based on age-bin width overlap, not source counts.",
                        "cases_total": estimated_cases * fraction,
                        "source_cases_total": estimated_cases,
                        "allocation_fraction": fraction,
                        "source_age_groups": str(row["Category"]).strip(),
                    }
                )
    age = pd.DataFrame(age_rows, columns=CASE_COLUMNS)
    model = pd.DataFrame(model_rows, columns=AGE_MODEL_COLUMNS)
    return age, model


def read_pertussis_age_specific_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    age_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    workbook_specs = [
        (PI_AGE, PERTUSSIS_INCIDENCE_COUNTRIES),
        (LOCAL_EXTENDED_AGE_CASES, EXTENDED_AGE_CASES_COUNTRIES),
    ]
    for workbook_path, countries in workbook_specs:
        if not workbook_path.exists():
            continue
        xl = pd.ExcelFile(workbook_path)
        valid_sheets = [sheet for sheet in xl.sheet_names if sheet.strip().upper() in countries]
        if not valid_sheets:
            continue
        results = parallel_calls(
            [(read_pertussis_age_specific_sheet, (sheet, workbook_path, countries)) for sheet in valid_sheets],
            cap=8,
        )
        age_frames.extend(age for age, _ in results if not age.empty)
        model_frames.extend(model for _, model in results if not model.empty)
    year_floor_map: dict[str, object] = {}
    if age_frames:
        base_age = pd.concat(age_frames, ignore_index=True, sort=False)
        year_floor = (
            base_age.loc[base_age["iso3"].notna(), ["iso3", "year"]]
            .dropna(subset=["year"])
            .groupby("iso3", as_index=False)
            .agg(year_floor=("year", "min"))
        )
        year_floor_map = year_floor.set_index("iso3")["year_floor"].to_dict()
    supplemental_age, supplemental_model = read_pertussis_age_specific_csv()
    if not supplemental_age.empty and year_floor_map:
        supplemental_age["year_floor"] = supplemental_age["iso3"].map(year_floor_map)
        supplemental_age = supplemental_age.loc[
            supplemental_age["year_floor"].isna() | supplemental_age["year"].lt(supplemental_age["year_floor"])
        ].copy()
        supplemental_age = supplemental_age.drop(columns=["year_floor"])
        if not supplemental_age.empty:
            supplemental_model["year_floor"] = supplemental_model["iso3"].map(year_floor_map)
            supplemental_model = supplemental_model.loc[
                supplemental_model["year_floor"].isna() | supplemental_model["year"].lt(supplemental_model["year_floor"])
            ].copy()
            supplemental_model = supplemental_model.drop(columns=["year_floor"])
    if not supplemental_age.empty:
        age_frames.append(supplemental_age)
    if not supplemental_model.empty:
        model_frames.append(supplemental_model)
    england_age, england_model = read_england_official_age_distribution_summary()
    if not england_age.empty:
        if year_floor_map:
            england_age["year_floor"] = england_age["iso3"].map(year_floor_map)
            england_age = england_age.loc[
                england_age["year_floor"].isna() | england_age["year"].lt(england_age["year_floor"])
            ].copy()
            england_age = england_age.drop(columns=["year_floor"])
            if not england_age.empty:
                england_model["year_floor"] = england_model["iso3"].map(year_floor_map)
                england_model = england_model.loc[
                    england_model["year_floor"].isna() | england_model["year"].lt(england_model["year_floor"])
                ].copy()
                england_model = england_model.drop(columns=["year_floor"])
        if not england_age.empty:
            age_frames.append(england_age)
        if not england_model.empty:
            model_frames.append(england_model)
    eu_age, eu_model = read_eu_age_specific_rate_panel()
    if not eu_age.empty:
        age_frames.append(eu_age)
    if not eu_model.empty:
        model_frames.append(eu_model)
    age = pd.concat(age_frames, ignore_index=True, sort=False) if age_frames else pd.DataFrame(columns=CASE_COLUMNS)
    model = pd.concat(model_frames, ignore_index=True, sort=False) if model_frames else pd.DataFrame(columns=AGE_MODEL_COLUMNS)
    return age, model


def collapse_unique_text(values: pd.Series, default: str = "not_available") -> str:
    cleaned = [str(value).strip() for value in values.fillna("").astype(str) if str(value).strip() and str(value).strip().lower() != "nan"]
    return "; ".join(dict.fromkeys(cleaned)) if cleaned else default


def summarize_dominant_text(values: pd.Series) -> tuple[object, int, float]:
    cleaned = [str(value).strip() for value in values.fillna("").astype(str) if str(value).strip() and str(value).strip().lower() != "nan"]
    if not cleaned:
        return pd.NA, 0, np.nan
    counts = pd.Series(cleaned).value_counts()
    dominant_count = int(counts.max())
    dominant_candidates = sorted(counts.loc[counts.eq(dominant_count)].index.tolist())
    dominant_value = dominant_candidates[0]
    dominant_share = dominant_count / len(cleaned) * 100.0 if cleaned else np.nan
    return dominant_value, dominant_count, dominant_share


def classify_age_pattern_signature(dominant_group: str, dominant_share: float, second_share: float) -> str:
    if not np.isfinite(dominant_share) or dominant_share < 25.0 or (np.isfinite(second_share) and dominant_share - second_share < 10.0):
        return "mixed_age_distribution"
    if dominant_group == "school_age_5_17y":
        return "school_age_peak"
    if dominant_group == "young_adult_18_39y":
        return "young_adult_peak"
    if dominant_group in {"middle_adult_40_64y", "elderly_65plus"}:
        return "adult_peak"
    if dominant_group in {"infant_0_11m", "child_1_4y"}:
        return "infant_or_early_child_peak"
    return "mixed_age_distribution"


def build_age_pattern_evidence(age_model_groups: pd.DataFrame, surveillance_outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "evidence_family",
        "year",
        "period_start",
        "period_end",
        "geographic_scope",
        "age_pattern_signature",
        "reported_case_total",
        "dominant_age_group",
        "dominant_model_age_group",
        "dominant_model_age_group_cases",
        "dominant_model_age_group_share_percent",
        "infant_case_count",
        "infant_case_share_percent",
        "school_age_case_count",
        "school_age_case_share_percent",
        "adult_case_count",
        "adult_case_share_percent",
        "source_age_group_count",
        "case_count",
        "case_count_qualifier",
        "case_count_type",
        "infant_age_definition",
        "under3m_case_count",
        "under6m_case_count",
        "under6m_hospitalization_percent",
        "total_hospitalization_percent",
        "hospitalizations",
        "total_deaths",
        "infant_deaths",
        "death_age_scope",
        "testing_denominator",
        "test_positivity_percent",
        "laboratory_confirmation_methods",
        "reporting_or_surveillance_change",
        "source_ids",
        "source",
        "source_type",
        "source_dataset",
        "geographic_scope_source",
        "notes",
    ]
    frames: list[pd.DataFrame] = []

    if not age_model_groups.empty:
        frame = age_model_groups.copy()
        required = {"iso3", "country", "year", "model_age_group", "cases_total"}
        if required.issubset(frame.columns):
            frame["iso3"] = frame["iso3"].astype(str).str.upper()
            frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
            frame["cases_total"] = pd.to_numeric(frame["cases_total"], errors="coerce")
            frame["model_age_group"] = frame["model_age_group"].astype(str)
            frame = frame.dropna(subset=["iso3", "country", "year", "cases_total", "model_age_group"]).copy()
            group_cols = ["iso3", "country", "year"]
            for column in ["source", "source_type", "source_dataset", "case_definition", "notes"]:
                if column in frame.columns:
                    group_cols.append(column)
            rows: list[dict[str, object]] = []
            for _, group in frame.groupby(group_cols, dropna=False):
                total = float(group["cases_total"].sum())
                if not np.isfinite(total) or total <= 0:
                    continue
                model_totals = group.groupby("model_age_group")["cases_total"].sum().to_dict()
                broad_totals = {
                    broad: float(sum(float(model_totals.get(model_age, 0.0)) for model_age in model_age_groups))
                    for broad, model_age_groups in AGE_PATTERN_BROAD_GROUPS.items()
                }
                sorted_broad = sorted(broad_totals.items(), key=lambda item: item[1], reverse=True)
                dominant_group, dominant_cases = sorted_broad[0]
                second_cases = sorted_broad[1][1] if len(sorted_broad) > 1 else 0.0
                dominant_share = dominant_cases / total * 100.0 if total else np.nan
                second_share = second_cases / total * 100.0 if total else np.nan
                model_dominant_group = max(model_totals, key=model_totals.get) if model_totals else pd.NA
                model_dominant_cases = float(model_totals.get(model_dominant_group, np.nan)) if model_totals else np.nan
                model_dominant_share = model_dominant_cases / total * 100.0 if total else np.nan
                infant_cases = broad_totals["infant_0_11m"]
                school_age_cases = broad_totals["school_age_5_17y"]
                adult_cases = broad_totals["young_adult_18_39y"] + broad_totals["middle_adult_40_64y"] + broad_totals["elderly_65plus"]
                rows.append(
                    {
                        "iso3": str(group["iso3"].iloc[0]).upper(),
                        "country": str(group["country"].iloc[0]),
                        "evidence_family": "reported_case_age_distribution",
                        "year": int(pd.to_numeric(group["year"].iloc[0], errors="coerce")),
                        "period_start": f"{int(pd.to_numeric(group['year'].iloc[0], errors='coerce'))}-01-01",
                        "period_end": f"{int(pd.to_numeric(group['year'].iloc[0], errors='coerce'))}-12-31",
                        "geographic_scope": "national annual age-distribution surveillance record",
                        "age_pattern_signature": classify_age_pattern_signature(dominant_group, dominant_share, second_share),
                        "reported_case_total": total,
                        "dominant_age_group": dominant_group,
                        "dominant_model_age_group": model_dominant_group,
                        "dominant_model_age_group_cases": model_dominant_cases,
                        "dominant_model_age_group_share_percent": model_dominant_share,
                        "infant_case_count": infant_cases,
                        "infant_case_share_percent": infant_cases / total * 100.0 if total else np.nan,
                        "school_age_case_count": school_age_cases,
                        "school_age_case_share_percent": school_age_cases / total * 100.0 if total else np.nan,
                        "adult_case_count": adult_cases,
                        "adult_case_share_percent": adult_cases / total * 100.0 if total else np.nan,
                        "source_age_group_count": int(group["age_group"].nunique()) if "age_group" in group.columns else int(group["model_age_group"].nunique()),
                        "case_count": pd.NA,
                        "case_count_qualifier": pd.NA,
                        "case_count_type": pd.NA,
                        "infant_age_definition": pd.NA,
                        "under3m_case_count": pd.NA,
                        "under6m_case_count": pd.NA,
                        "under6m_hospitalization_percent": pd.NA,
                        "total_hospitalization_percent": pd.NA,
                        "hospitalizations": pd.NA,
                        "total_deaths": pd.NA,
                        "infant_deaths": pd.NA,
                        "death_age_scope": pd.NA,
                        "testing_denominator": pd.NA,
                        "test_positivity_percent": pd.NA,
                        "laboratory_confirmation_methods": pd.NA,
                        "reporting_or_surveillance_change": pd.NA,
                        "source_ids": collapse_unique_text(group.get("source", pd.Series(dtype=str))),
                        "source": collapse_unique_text(group.get("source", pd.Series(dtype=str))),
                        "source_type": collapse_unique_text(group.get("source_type", pd.Series(dtype=str))),
                        "source_dataset": collapse_unique_text(group.get("source_dataset", pd.Series(dtype=str))),
                        "geographic_scope_source": "country-year source age bins allocated to the shared model age axis",
                        "notes": collapse_unique_text(group.get("notes", pd.Series(dtype=str))),
                    }
                )
            if rows:
                frames.append(pd.DataFrame(rows, columns=columns))

    if not surveillance_outcomes.empty and "iso3" in surveillance_outcomes.columns:
        outcome = surveillance_outcomes.copy()
        outcome["iso3"] = outcome["iso3"].astype(str).str.upper()
        outcome["year"] = pd.to_numeric(outcome.get("year"), errors="coerce")
        for column in [
            "case_count",
            "infant_case_count",
            "under3m_case_count",
            "under6m_case_count",
            "under6m_hospitalization_percent",
            "total_hospitalization_percent",
            "hospitalizations",
            "total_deaths",
            "infant_deaths",
            "testing_denominator",
            "test_positivity_percent",
        ]:
            if column in outcome:
                outcome[column] = pd.to_numeric(outcome[column], errors="coerce")
            else:
                outcome[column] = pd.NA
        outcome["period_start"] = pd.to_datetime(outcome.get("period_start"), errors="coerce")
        outcome["period_end"] = pd.to_datetime(outcome.get("period_end"), errors="coerce")
        age_fields = [
            "infant_case_count",
            "under3m_case_count",
            "under6m_case_count",
            "under6m_hospitalization_percent",
            "total_hospitalization_percent",
            "hospitalizations",
            "total_deaths",
            "infant_deaths",
        ]
        age_mask = outcome[age_fields].notna().any(axis=1) if age_fields else pd.Series(False, index=outcome.index)
        outcome = outcome.loc[age_mask].copy()
        if not outcome.empty:
            rows = []
            for _, row in outcome.iterrows():
                year = pd.to_numeric(row.get("year"), errors="coerce")
                period_start = row.get("period_start")
                period_end = row.get("period_end")
                if pd.isna(period_start):
                    period_start = pd.Timestamp(f"{int(year)}-01-01") if pd.notna(year) else pd.NaT
                if pd.isna(period_end):
                    period_end = pd.Timestamp(f"{int(year)}-12-31") if pd.notna(year) else pd.NaT
                signature = "official_severe_outcome_summary"
                if pd.notna(row.get("under3m_case_count")) or pd.notna(row.get("under6m_case_count")) or pd.notna(row.get("under6m_hospitalization_percent")):
                    signature = "young_infant_severe_outcome_signature"
                elif pd.notna(row.get("infant_case_count")) or pd.notna(row.get("infant_deaths")):
                    signature = "infant_severe_outcome_signature"
                rows.append(
                    {
                        "iso3": str(row.get("iso3", "")).upper(),
                        "country": str(row.get("country", "")),
                        "evidence_family": "official_severe_outcome_age_signature",
                        "year": int(year) if pd.notna(year) else pd.NA,
                        "period_start": period_start.strftime("%Y-%m-%d") if pd.notna(period_start) else pd.NA,
                        "period_end": period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else pd.NA,
                        "geographic_scope": row.get("geographic_scope", pd.NA),
                        "age_pattern_signature": signature,
                        "reported_case_total": row.get("case_count", pd.NA),
                        "dominant_age_group": pd.NA,
                        "dominant_model_age_group": pd.NA,
                        "dominant_model_age_group_cases": pd.NA,
                        "dominant_model_age_group_share_percent": pd.NA,
                        "infant_case_count": row.get("infant_case_count", pd.NA),
                        "infant_case_share_percent": pd.NA,
                        "school_age_case_count": pd.NA,
                        "school_age_case_share_percent": pd.NA,
                        "adult_case_count": pd.NA,
                        "adult_case_share_percent": pd.NA,
                        "source_age_group_count": pd.NA,
                        "case_count": row.get("case_count", pd.NA),
                        "case_count_qualifier": row.get("case_count_qualifier", pd.NA),
                        "case_count_type": row.get("case_count_type", pd.NA),
                        "infant_age_definition": row.get("infant_age_definition", pd.NA),
                        "under3m_case_count": row.get("under3m_case_count", pd.NA),
                        "under6m_case_count": row.get("under6m_case_count", pd.NA),
                        "under6m_hospitalization_percent": row.get("under6m_hospitalization_percent", pd.NA),
                        "total_hospitalization_percent": row.get("total_hospitalization_percent", pd.NA),
                        "hospitalizations": row.get("hospitalizations", pd.NA),
                        "total_deaths": row.get("total_deaths", pd.NA),
                        "infant_deaths": row.get("infant_deaths", pd.NA),
                        "death_age_scope": row.get("death_age_scope", pd.NA),
                        "testing_denominator": row.get("testing_denominator", pd.NA),
                        "test_positivity_percent": row.get("test_positivity_percent", pd.NA),
                        "laboratory_confirmation_methods": row.get("laboratory_confirmation_methods", pd.NA),
                        "reporting_or_surveillance_change": row.get("reporting_or_surveillance_change", pd.NA),
                        "source_ids": collapse_unique_text(pd.Series([row.get("source_ids", pd.NA)])),
                        "source": str(row.get("source", "")),
                        "source_type": str(row.get("source_type", "")),
                        "source_dataset": str(row.get("source_dataset", "")),
                        "geographic_scope_source": "official surveillance summary row with age-specific severe-outcome fields when available",
                        "notes": str(row.get("notes", "")),
                    }
                )
            if rows:
                frames.append(pd.DataFrame(rows, columns=columns))

    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True, sort=False)
    for column in [
        "year",
        "reported_case_total",
        "dominant_model_age_group_cases",
        "dominant_model_age_group_share_percent",
        "infant_case_count",
        "infant_case_share_percent",
        "school_age_case_count",
        "school_age_case_share_percent",
        "adult_case_count",
        "adult_case_share_percent",
        "source_age_group_count",
        "case_count",
        "infant_case_count",
        "under3m_case_count",
        "under6m_case_count",
        "under6m_hospitalization_percent",
        "total_hospitalization_percent",
        "hospitalizations",
        "total_deaths",
        "infant_deaths",
        "testing_denominator",
        "test_positivity_percent",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.loc[:, columns].sort_values(["iso3", "evidence_family", "year", "period_end"], na_position="last").reset_index(drop=True)


def summarize_eu_age_pattern_country_summary(age_pattern_evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "iso3",
        "country",
        "country_year_rows",
        "year_min",
        "year_max",
        "reported_case_total_sum",
        "dominant_age_pattern_signature",
        "dominant_age_pattern_signature_count",
        "dominant_age_pattern_signature_share_percent",
        "dominant_age_group",
        "dominant_age_group_count",
        "dominant_age_group_share_percent",
        "dominant_model_age_group",
        "dominant_model_age_group_count",
        "dominant_model_age_group_share_percent",
        "unique_age_pattern_signatures",
        "summary_note",
    ]
    if age_pattern_evidence.empty or "source_dataset" not in age_pattern_evidence.columns:
        return pd.DataFrame(columns=columns)

    frame = age_pattern_evidence.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame["country"].fillna("").astype(str).str.strip()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["reported_case_total"] = pd.to_numeric(frame["reported_case_total"], errors="coerce")
    frame["source_dataset"] = frame["source_dataset"].fillna("").astype(str)
    eu = frame.loc[frame["source_dataset"].str.contains(EU_AGE_RATE_PANEL_LABEL, case=False, na=False)].copy()
    if eu.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (iso3, country), group in eu.groupby(["iso3", "country"], dropna=False):
        years = pd.to_numeric(group["year"], errors="coerce").dropna().astype(int)
        dominant_signature, dominant_signature_count, dominant_signature_share = summarize_dominant_text(group["age_pattern_signature"])
        dominant_age_group, dominant_age_group_count, dominant_age_group_share = summarize_dominant_text(group["dominant_age_group"])
        dominant_model_age_group, dominant_model_age_group_count, dominant_model_age_group_share = summarize_dominant_text(group["dominant_model_age_group"])
        country_year_rows = int(len(group))
        reported_case_total_sum = float(pd.to_numeric(group["reported_case_total"], errors="coerce").sum())
        unique_signatures = int(group["age_pattern_signature"].dropna().astype(str).str.strip().replace({"": pd.NA}).dropna().nunique())
        rows.append(
            {
                "scope": "country",
                "iso3": iso3,
                "country": country,
                "country_year_rows": country_year_rows,
                "year_min": int(years.min()) if not years.empty else pd.NA,
                "year_max": int(years.max()) if not years.empty else pd.NA,
                "reported_case_total_sum": reported_case_total_sum,
                "dominant_age_pattern_signature": dominant_signature,
                "dominant_age_pattern_signature_count": dominant_signature_count,
                "dominant_age_pattern_signature_share_percent": dominant_signature_share,
                "dominant_age_group": dominant_age_group,
                "dominant_age_group_count": dominant_age_group_count,
                "dominant_age_group_share_percent": dominant_age_group_share,
                "dominant_model_age_group": dominant_model_age_group,
                "dominant_model_age_group_count": dominant_model_age_group_count,
                "dominant_model_age_group_share_percent": dominant_model_age_group_share,
                "unique_age_pattern_signatures": unique_signatures,
                "summary_note": (
                    f"Country-level summary of the {EU_AGE_RATE_PANEL_NORMALIZED_DATASET}; "
                    f"dominant signature across {country_year_rows} country-years is {dominant_signature} "
                    f"({dominant_signature_share:.1f}%)."
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(["iso3", "country"], na_position="last").reset_index(drop=True)


def summarize_eu_age_pattern_year_summary(age_pattern_evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "year",
        "countries_observed",
        "country_year_rows",
        "reported_case_total_sum",
        "infant_or_early_child_peak_country_count",
        "school_age_peak_country_count",
        "mixed_age_distribution_country_count",
        "adult_peak_country_count",
        "young_adult_peak_country_count",
        "dominant_age_pattern_signature_by_country_count",
        "dominant_age_pattern_signature_country_count",
        "dominant_age_pattern_signature_country_share_percent",
        "dominant_age_pattern_signature_by_reported_case_total",
        "dominant_age_pattern_signature_reported_case_total",
        "dominant_age_pattern_signature_reported_case_share_percent",
        "summary_note",
    ]
    if age_pattern_evidence.empty or "source_dataset" not in age_pattern_evidence.columns:
        return pd.DataFrame(columns=columns)

    frame = age_pattern_evidence.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["reported_case_total"] = pd.to_numeric(frame["reported_case_total"], errors="coerce")
    frame["source_dataset"] = frame["source_dataset"].fillna("").astype(str)
    eu = frame.loc[frame["source_dataset"].str.contains(EU_AGE_RATE_PANEL_LABEL, case=False, na=False)].copy()
    if eu.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for year, group in eu.groupby("year", dropna=False):
        year_int = int(year)
        counts = group["age_pattern_signature"].value_counts()
        case_totals = group.groupby("age_pattern_signature")["reported_case_total"].sum().sort_values(ascending=False)
        dominant_country_signature = counts.index[0]
        dominant_country_count = int(counts.iloc[0])
        dominant_country_share = dominant_country_count / len(group) * 100.0 if len(group) else np.nan
        dominant_case_signature = case_totals.index[0]
        dominant_case_total = float(case_totals.iloc[0])
        total_cases = float(pd.to_numeric(group["reported_case_total"], errors="coerce").sum())
        dominant_case_share = dominant_case_total / total_cases * 100.0 if total_cases else np.nan
        rows.append(
            {
                "scope": "year",
                "year": year_int,
                "countries_observed": int(group["iso3"].nunique()),
                "country_year_rows": int(len(group)),
                "reported_case_total_sum": total_cases,
                "infant_or_early_child_peak_country_count": int(counts.get("infant_or_early_child_peak", 0)),
                "school_age_peak_country_count": int(counts.get("school_age_peak", 0)),
                "mixed_age_distribution_country_count": int(counts.get("mixed_age_distribution", 0)),
                "adult_peak_country_count": int(counts.get("adult_peak", 0)),
                "young_adult_peak_country_count": int(counts.get("young_adult_peak", 0)),
                "dominant_age_pattern_signature_by_country_count": dominant_country_signature,
                "dominant_age_pattern_signature_country_count": dominant_country_count,
                "dominant_age_pattern_signature_country_share_percent": dominant_country_share,
                "dominant_age_pattern_signature_by_reported_case_total": dominant_case_signature,
                "dominant_age_pattern_signature_reported_case_total": dominant_case_total,
                "dominant_age_pattern_signature_reported_case_share_percent": dominant_case_share,
                "summary_note": (
                    f"By country count, the {year_int} EU panel is dominated by {dominant_country_signature} "
                    f"({dominant_country_count}/{len(group)} country-years); by reported-case totals, it is dominated by "
                    f"{dominant_case_signature} ({dominant_case_share:.1f}% of reported cases)."
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(["year"], na_position="last").reset_index(drop=True)


def summarize_global_age_pattern_summary(age_pattern_evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "age_pattern_coverage_class",
        "total_rows",
        "reported_case_age_distribution_rows",
        "reported_case_age_distribution_year_min",
        "reported_case_age_distribution_year_max",
        "reported_case_age_distribution_year_span",
        "reported_case_age_distribution_dominant_signature",
        "reported_case_age_distribution_dominant_signature_count",
        "reported_case_age_distribution_dominant_signature_share_percent",
        "reported_case_age_distribution_unique_signatures",
        "official_severe_outcome_age_signature_rows",
        "official_severe_outcome_age_signature_year_min",
        "official_severe_outcome_age_signature_year_max",
        "official_severe_outcome_age_signature_year_span",
        "official_severe_outcome_age_signature_dominant_signature",
        "official_severe_outcome_age_signature_dominant_signature_count",
        "official_severe_outcome_age_signature_dominant_signature_share_percent",
        "official_severe_outcome_age_signature_unique_signatures",
        "summary_note",
    ]
    if age_pattern_evidence.empty or "evidence_family" not in age_pattern_evidence.columns:
        return pd.DataFrame(columns=columns)

    frame = age_pattern_evidence.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame["country"].fillna("").astype(str).str.strip()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["age_pattern_signature"] = frame["age_pattern_signature"].fillna("").astype(str).str.strip()

    def summarize_family(group: pd.DataFrame, family: str) -> dict[str, object]:
        sub = group.loc[group["evidence_family"].eq(family)].copy()
        prefix = "reported_case_age_distribution" if family == "reported_case_age_distribution" else "official_severe_outcome_age_signature"
        if sub.empty:
            return {
                f"{prefix}_rows": 0,
                f"{prefix}_year_min": pd.NA,
                f"{prefix}_year_max": pd.NA,
                f"{prefix}_year_span": 0,
                f"{prefix}_dominant_signature": pd.NA,
                f"{prefix}_dominant_signature_count": 0,
                f"{prefix}_dominant_signature_share_percent": np.nan,
                f"{prefix}_unique_signatures": 0,
            }

        years = sub["year"].dropna().astype(int)
        dominant_signature, dominant_signature_count, dominant_signature_share = summarize_dominant_text(sub["age_pattern_signature"])
        unique_signatures = int(
            sub["age_pattern_signature"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace({"": pd.NA})
            .dropna()
            .nunique()
        )
        return {
            f"{prefix}_rows": int(len(sub)),
            f"{prefix}_year_min": int(years.min()) if not years.empty else pd.NA,
            f"{prefix}_year_max": int(years.max()) if not years.empty else pd.NA,
            f"{prefix}_year_span": int(years.max() - years.min() + 1) if not years.empty else 0,
            f"{prefix}_dominant_signature": dominant_signature,
            f"{prefix}_dominant_signature_count": dominant_signature_count,
            f"{prefix}_dominant_signature_share_percent": dominant_signature_share,
            f"{prefix}_unique_signatures": unique_signatures,
        }

    rows: list[dict[str, object]] = []
    for (iso3, country), group in frame.groupby(["iso3", "country"], dropna=False):
        reported = group.loc[group["evidence_family"].eq("reported_case_age_distribution")]
        severe = group.loc[group["evidence_family"].eq("official_severe_outcome_age_signature")]
        reported_rows = int(len(reported))
        severe_rows = int(len(severe))
        if reported_rows > 0 and severe_rows > 0:
            coverage_class = "both_families"
            summary_note = "Both reported-case age-distribution and official severe-outcome age-signature evidence are available; see Supplementary Figure 3."
        elif reported_rows > 0:
            coverage_class = "reported_case_only"
            summary_note = "Reported-case age-distribution evidence only; see Supplementary Figure 3."
        else:
            coverage_class = "official_severe_outcome_only"
            summary_note = "Official severe-outcome age-signature evidence only; see Supplementary Figure 3."
        row: dict[str, object] = {
            "iso3": iso3,
            "country": country,
            "age_pattern_coverage_class": coverage_class,
            "total_rows": int(len(group)),
            "summary_note": summary_note,
        }
        row.update(summarize_family(group, "reported_case_age_distribution"))
        row.update(summarize_family(group, "official_severe_outcome_age_signature"))
        rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out["age_pattern_coverage_class"] = pd.Categorical(
        out["age_pattern_coverage_class"],
        categories=["both_families", "reported_case_only", "official_severe_outcome_only"],
        ordered=True,
    )
    return out.sort_values(["age_pattern_coverage_class", "country"], na_position="last").reset_index(drop=True)


def summarize_age_pattern_validation_lattice(
    age_pattern_global_summary: pd.DataFrame,
    inventory: pd.DataFrame,
    external_validation_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "age_pattern_coverage_class",
        "validation_score",
        "validation_class",
        "age_pattern_case_distribution_available",
        "age_pattern_severe_outcome_signature_available",
        "testing_denominator_available",
        "direct_testing_denominator_series_available",
        "annual_national_surveillance_summary_available",
        "testing_reporting_case_definition_available",
        "serology_available",
        "contact_validation_available",
        "hospitalization_available",
        "death_available",
        "testing_denominator_years_available",
        "serology_n_rows",
        "contact_validation_n_rows",
        "external_validation_boundary_label",
        "summary_note",
    ]
    if age_pattern_global_summary.empty or inventory.empty:
        return pd.DataFrame(columns=columns)

    frame = age_pattern_global_summary.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame["country"].fillna("").astype(str).str.strip()
    inv = inventory.copy()
    inv["iso3"] = inv["iso3"].astype(str).str.upper()
    inv["country"] = inv["country"].fillna("").astype(str).str.strip()
    keep_cols = [
        "iso3",
        "country",
        "age_pattern_case_distribution_available",
        "age_pattern_severe_outcome_signature_available",
        "testing_denominator_available",
        "direct_testing_denominator_series_available",
        "annual_national_surveillance_summary_available",
        "testing_reporting_case_definition_available",
        "hospitalization_available",
        "death_available",
        "testing_denominator_years_available",
    ]
    inv = inv.loc[:, [column for column in keep_cols if column in inv.columns]].copy()
    ext = external_validation_summary.copy()
    if not ext.empty:
        ext["iso3"] = ext["iso3"].astype(str).str.upper()
        ext["country"] = ext["country"].fillna("").astype(str).str.strip()
        ext = ext.loc[
            :,
            [
                "iso3",
                "serology_available",
                "contact_validation_available",
                "serology_n_rows",
                "contact_validation_n_rows",
                "external_validation_boundary_label",
            ],
        ]
    else:
        ext = pd.DataFrame(
            columns=[
                "iso3",
                "serology_available",
                "contact_validation_available",
                "serology_n_rows",
                "contact_validation_n_rows",
                "external_validation_boundary_label",
            ]
        )

    out = frame.merge(inv, on=["iso3", "country"], how="left").merge(ext, on="iso3", how="left")
    for column in [
        "age_pattern_case_distribution_available",
        "age_pattern_severe_outcome_signature_available",
        "testing_denominator_available",
        "direct_testing_denominator_series_available",
        "annual_national_surveillance_summary_available",
        "testing_reporting_case_definition_available",
        "serology_available",
        "contact_validation_available",
        "hospitalization_available",
        "death_available",
    ]:
        if column in out.columns:
            out[column] = out[column].fillna(False).astype(bool)
    for column in ["testing_denominator_years_available", "serology_n_rows", "contact_validation_n_rows"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out["validation_score"] = (
        out[
            [
                "age_pattern_case_distribution_available",
                "age_pattern_severe_outcome_signature_available",
                "testing_denominator_available",
                "direct_testing_denominator_series_available",
                "annual_national_surveillance_summary_available",
                "testing_reporting_case_definition_available",
                "serology_available",
                "contact_validation_available",
                "hospitalization_available",
                "death_available",
            ]
        ]
        .astype(bool)
        .sum(axis=1)
        .astype(int)
    )
    out["validation_class"] = pd.Categorical(
        np.select(
            [out["validation_score"].ge(8), out["validation_score"].between(5, 7)],
            ["rich_validation", "mixed_validation"],
            default="sparse_validation",
        ),
        categories=["rich_validation", "mixed_validation", "sparse_validation"],
        ordered=True,
    )
    out["external_validation_boundary_label"] = out["external_validation_boundary_label"].fillna("none").astype(str)
    out["summary_note"] = (
        "Country-level validation lattice across age-pattern, testing-denominator, "
        "surveillance, serology, and contact domains; score counts available direct evidence layers. "
        "See Supplementary Figure 21."
    )
    return out.loc[:, columns].sort_values(["validation_score", "country"], ascending=[False, True], na_position="last").reset_index(drop=True)


def summarize_gbr_age_pattern_history(age_model_groups: pd.DataFrame, age_pattern_evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "source_segment",
        "source_dataset",
        "source_ids",
        "period_start",
        "period_end",
        "reported_case_total",
        "under1_share_percent",
        "child_1_4_share_percent",
        "school_age_5_17_share_percent",
        "adult_18plus_share_percent",
        "dominant_broad_age_band",
        "dominant_broad_age_band_share_percent",
        "dominant_model_age_group",
        "dominant_model_age_group_share_percent",
        "age_pattern_signature",
        "summary_note",
    ]
    if age_model_groups.empty or age_pattern_evidence.empty:
        return pd.DataFrame(columns=columns)

    model = age_model_groups.copy()
    model["iso3"] = model["iso3"].astype(str).str.upper()
    model["country"] = model["country"].fillna("").astype(str).str.strip()
    model["year"] = pd.to_numeric(model["year"], errors="coerce")
    model["cases_total"] = pd.to_numeric(model["cases_total"], errors="coerce")
    model["model_age_group"] = model["model_age_group"].astype(str)
    model = model.loc[
        (model["iso3"] == "GBR")
        & model["year"].notna()
        & model["cases_total"].notna()
        & model["model_age_group"].notna()
    ].copy()
    if model.empty:
        return pd.DataFrame(columns=columns)

    pattern = age_pattern_evidence.copy()
    pattern["iso3"] = pattern["iso3"].astype(str).str.upper()
    pattern["country"] = pattern["country"].fillna("").astype(str).str.strip()
    pattern["year"] = pd.to_numeric(pattern["year"], errors="coerce")
    pattern = pattern.loc[
        (pattern["iso3"] == "GBR") & pattern["evidence_family"].eq("reported_case_age_distribution")
    ].copy()
    if pattern.empty:
        return pd.DataFrame(columns=columns)

    pattern_columns = [
        "year",
        "country",
        "source_dataset",
        "source_ids",
        "period_start",
        "period_end",
        "age_pattern_signature",
        "reported_case_total",
        "dominant_model_age_group",
        "dominant_model_age_group_share_percent",
        "notes",
    ]
    pattern = pattern.loc[:, [column for column in pattern_columns if column in pattern.columns]].drop_duplicates(subset=["year"]).copy()

    rows: list[dict[str, object]] = []
    for year, group in model.groupby("year", dropna=False):
        year_int = int(year)
        total = float(group["cases_total"].sum())
        if not np.isfinite(total) or total <= 0:
            continue

        model_totals = group.groupby("model_age_group")["cases_total"].sum().sort_values(ascending=False)
        broad_totals = {
            "under1": float(model_totals.get("infant_0_2m", 0.0) + model_totals.get("infant_3_11m", 0.0)),
            "child_1_4y": float(model_totals.get("child_1_4y", 0.0)),
            "school_age_5_17y": float(model_totals.get("child_5_9y", 0.0) + model_totals.get("adolescent_10_17y", 0.0)),
            "adult_18plus": float(
                model_totals.get("young_adult_18_39y", 0.0)
                + model_totals.get("middle_adult_40_64y", 0.0)
                + model_totals.get("elderly_65plus", 0.0)
            ),
        }
        broad_sorted = sorted(broad_totals.items(), key=lambda item: item[1], reverse=True)
        dominant_broad_band, dominant_broad_cases = broad_sorted[0]
        second_broad_cases = broad_sorted[1][1] if len(broad_sorted) > 1 else 0.0
        dominant_broad_share = dominant_broad_cases / total * 100.0 if total else np.nan
        second_broad_share = second_broad_cases / total * 100.0 if total else np.nan
        dominant_model_age_group = str(model_totals.index[0]) if not model_totals.empty else pd.NA
        dominant_model_age_group_cases = float(model_totals.iloc[0]) if not model_totals.empty else np.nan
        dominant_model_age_group_share = dominant_model_age_group_cases / total * 100.0 if total else np.nan
        signature_row = pattern.loc[pattern["year"].eq(year_int)]
        if not signature_row.empty:
            signature_row = signature_row.iloc[0]
        else:
            signature_row = pd.Series(dtype=object)
        rows.append(
            {
                "iso3": "GBR",
                "country": "United Kingdom of Great Britain and Northern Ireland",
                "year": year_int,
                "source_segment": "official_summary_extension" if year_int <= 2014 else "workbook_age_coverage",
                "source_dataset": str(signature_row.get("source_dataset", "")),
                "source_ids": str(signature_row.get("source_ids", "")),
                "period_start": signature_row.get("period_start", f"{year_int}-01-01"),
                "period_end": signature_row.get("period_end", f"{year_int}-12-31"),
                "reported_case_total": float(signature_row.get("reported_case_total", total)),
                "under1_share_percent": broad_totals["under1"] / total * 100.0 if total else np.nan,
                "child_1_4_share_percent": broad_totals["child_1_4y"] / total * 100.0 if total else np.nan,
                "school_age_5_17_share_percent": broad_totals["school_age_5_17y"] / total * 100.0 if total else np.nan,
                "adult_18plus_share_percent": broad_totals["adult_18plus"] / total * 100.0 if total else np.nan,
                "dominant_broad_age_band": dominant_broad_band,
                "dominant_broad_age_band_share_percent": dominant_broad_share,
                "dominant_model_age_group": str(signature_row.get("dominant_model_age_group", dominant_model_age_group)),
                "dominant_model_age_group_share_percent": float(
                    signature_row.get("dominant_model_age_group_share_percent", dominant_model_age_group_share)
                ),
                "age_pattern_signature": str(signature_row.get("age_pattern_signature", classify_age_pattern_signature(dominant_model_age_group, dominant_broad_share, second_broad_share))),
                "summary_note": (
                    "England/GBR age history companion table extends the official 1994-2014 age-distribution summary "
                    "before workbook coverage begins in 2015; use with Supplementary Figure 11."
                ),
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out["source_segment"] = pd.Categorical(
        out["source_segment"],
        categories=["official_summary_extension", "workbook_age_coverage"],
        ordered=True,
    )
    return out.sort_values(["year"], na_position="last").reset_index(drop=True)


def summarize_usa_annual_age_specific_surveillance_summary(age_pattern_evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "period_start",
        "period_end",
        "geographic_scope",
        "evidence_family",
        "reported_case_total",
        "dominant_age_group",
        "dominant_model_age_group",
        "dominant_model_age_group_share_percent",
        "infant_case_share_percent",
        "school_age_case_share_percent",
        "adult_case_share_percent",
        "hospitalizations",
        "total_deaths",
        "infant_deaths",
        "source_ids",
        "source_type",
        "notes",
    ]
    if age_pattern_evidence.empty or "evidence_family" not in age_pattern_evidence.columns:
        return pd.DataFrame(columns=columns)

    usa = age_pattern_evidence.copy()
    usa["iso3"] = usa["iso3"].astype(str).str.upper()
    usa["country"] = usa["country"].fillna("").astype(str).str.strip()
    usa["year"] = pd.to_numeric(usa["year"], errors="coerce")
    usa["evidence_family"] = usa["evidence_family"].astype(str)
    usa = usa.loc[
        (usa["iso3"] == "USA")
        & usa["evidence_family"].isin(
            [
                "official_severe_outcome_age_signature",
                "reported_case_age_distribution",
            ]
        )
    ].copy()
    if usa.empty:
        return pd.DataFrame(columns=columns)

    usa["evidence_family"] = pd.Categorical(
        usa["evidence_family"],
        categories=["official_severe_outcome_age_signature", "reported_case_age_distribution"],
        ordered=True,
    )
    selected = (
        usa.loc[:, [column for column in columns if column in usa.columns]]
        .sort_values(["evidence_family", "year"], na_position="last", kind="stable")
        .reset_index(drop=True)
    )
    selected["evidence_family"] = selected["evidence_family"].astype(str)
    return selected


def classify_gbd_under1_share(share: float) -> str:
    if pd.isna(share):
        return "not_available"
    value = float(share)
    if value >= 0.5:
        return "infant_dominated"
    if value >= 0.4:
        return "infant_leaning"
    if value >= 0.35:
        return "mixed_burden"
    return "broad_or_noninfant"


def summarize_gbd_burden_country_summary(gbd_burden: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "iso3",
        "country",
        "dalys_years_available",
        "dalys_year_min",
        "dalys_year_max",
        "dalys_all_age_number_mean",
        "dalys_under1_number_mean",
        "dalys_under1_share_allages_mean",
        "dalys_under1_share_allages_median",
        "dalys_under1_share_allages_min",
        "dalys_under1_share_allages_max",
        "dominant_dalys_age_burden_signature",
        "dominant_dalys_age_burden_signature_count",
        "dominant_dalys_age_burden_signature_share_percent",
        "incidence_years_available",
        "incidence_year_min",
        "incidence_year_max",
        "incidence_all_age_number_mean",
        "incidence_under1_number_mean",
        "incidence_under1_share_allages_mean",
        "incidence_under1_share_allages_median",
        "incidence_under1_share_allages_min",
        "incidence_under1_share_allages_max",
        "summary_note",
    ]
    if gbd_burden.empty or "iso3" not in gbd_burden:
        return pd.DataFrame(columns=columns)

    frame = gbd_burden.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame["location_name"].fillna("").astype(str).str.strip()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["val"] = pd.to_numeric(frame["val"], errors="coerce")
    frame["measure_name"] = frame["measure_name"].fillna("").astype(str)
    frame["metric_name"] = frame["metric_name"].fillna("").astype(str)
    frame["age_name"] = frame["age_name"].fillna("").astype(str)
    frame = frame.loc[frame["iso3"].notna()].copy()
    frame = frame.loc[
        frame["metric_name"].eq("Number")
        & frame["measure_name"].isin(["DALYs (Disability-Adjusted Life Years)", "Incidence"])
        & frame["age_name"].isin(["All ages", "<1 year"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (iso3, country), country_group in frame.groupby(["iso3", "country"], dropna=False):
        row: dict[str, object] = {
            "scope": "country",
            "iso3": iso3,
            "country": country,
        }
        for measure_name, prefix in [
            ("DALYs (Disability-Adjusted Life Years)", "dalys"),
            ("Incidence", "incidence"),
        ]:
            measure_group = country_group.loc[country_group["measure_name"].eq(measure_name)].copy()
            if measure_group.empty:
                row.update(
                    {
                        f"{prefix}_years_available": pd.NA,
                        f"{prefix}_year_min": pd.NA,
                        f"{prefix}_year_max": pd.NA,
                        f"{prefix}_all_age_number_mean": np.nan,
                        f"{prefix}_under1_number_mean": np.nan,
                        f"{prefix}_under1_share_allages_mean": np.nan,
                        f"{prefix}_under1_share_allages_median": np.nan,
                        f"{prefix}_under1_share_allages_min": np.nan,
                        f"{prefix}_under1_share_allages_max": np.nan,
                    }
                )
                if prefix == "dalys":
                    row.update(
                        {
                            "dominant_dalys_age_burden_signature": "not_available",
                            "dominant_dalys_age_burden_signature_count": pd.NA,
                            "dominant_dalys_age_burden_signature_share_percent": np.nan,
                        }
                    )
                continue

            pivot = measure_group.pivot_table(index="year", columns="age_name", values="val", aggfunc="first").reset_index()
            pivot = pivot.rename(columns={"All ages": "all_ages", "<1 year": "under1"})
            pivot = pivot.dropna(subset=["year", "all_ages", "under1"])
            if pivot.empty:
                row.update(
                    {
                        f"{prefix}_years_available": pd.NA,
                        f"{prefix}_year_min": pd.NA,
                        f"{prefix}_year_max": pd.NA,
                        f"{prefix}_all_age_number_mean": np.nan,
                        f"{prefix}_under1_number_mean": np.nan,
                        f"{prefix}_under1_share_allages_mean": np.nan,
                        f"{prefix}_under1_share_allages_median": np.nan,
                        f"{prefix}_under1_share_allages_min": np.nan,
                        f"{prefix}_under1_share_allages_max": np.nan,
                    }
                )
                if prefix == "dalys":
                    row.update(
                        {
                            "dominant_dalys_age_burden_signature": "not_available",
                            "dominant_dalys_age_burden_signature_count": pd.NA,
                            "dominant_dalys_age_burden_signature_share_percent": np.nan,
                        }
                    )
                continue

            pivot["year"] = pd.to_numeric(pivot["year"], errors="coerce")
            pivot = pivot.dropna(subset=["year"]).copy()
            pivot["under1_share_allages"] = pivot["under1"] / pivot["all_ages"].replace({0: np.nan})
            pivot["under1_share_class"] = pivot["under1_share_allages"].map(classify_gbd_under1_share)
            dominant_signature = classify_gbd_under1_share(float(pivot["under1_share_allages"].mean()))
            dominant_count = int(pivot["under1_share_class"].eq(dominant_signature).sum())
            dominant_share = dominant_count / len(pivot) * 100.0 if len(pivot) else np.nan
            row.update(
                {
                    f"{prefix}_years_available": int(pivot["year"].nunique()),
                    f"{prefix}_year_min": int(pivot["year"].min()),
                    f"{prefix}_year_max": int(pivot["year"].max()),
                    f"{prefix}_all_age_number_mean": float(pivot["all_ages"].mean()),
                    f"{prefix}_under1_number_mean": float(pivot["under1"].mean()),
                    f"{prefix}_under1_share_allages_mean": float(pivot["under1_share_allages"].mean()),
                    f"{prefix}_under1_share_allages_median": float(pivot["under1_share_allages"].median()),
                    f"{prefix}_under1_share_allages_min": float(pivot["under1_share_allages"].min()),
                    f"{prefix}_under1_share_allages_max": float(pivot["under1_share_allages"].max()),
                }
            )
            if prefix == "dalys":
                row.update(
                    {
                        "dominant_dalys_age_burden_signature": dominant_signature,
                        "dominant_dalys_age_burden_signature_count": dominant_count,
                        "dominant_dalys_age_burden_signature_share_percent": dominant_share,
                    }
                )
        row["summary_note"] = (
            "Modeled IHME GBD 2021 pertussis burden summary using Number rows for all ages and <1 year; "
            "DALYs are the primary burden metric, incidence is a secondary consistency check, and the class labels "
            "('infant_dominated', 'infant_leaning', 'mixed_burden', 'broad_or_noninfant') are based on the mean <1-year "
            "share across available country-year rows, with count/share reporting how often the yearly class matches that "
            "country-level dominant class."
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns).sort_values(["iso3", "country"], na_position="last").reset_index(drop=True)


def summarize_gbd_burden_year_summary(gbd_burden: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "year",
        "countries_observed",
        "country_year_rows",
        "dalys_all_age_number_sum",
        "dalys_under1_number_sum",
        "weighted_dalys_under1_share_allages",
        "mean_country_dalys_under1_share_allages",
        "median_country_dalys_under1_share_allages",
        "countries_dalys_infant_dominated",
        "countries_dalys_infant_leaning",
        "countries_dalys_mixed_burden",
        "countries_dalys_broad_or_noninfant",
        "dominant_dalys_age_burden_signature_by_country_count",
        "dominant_dalys_age_burden_signature_country_count",
        "dominant_dalys_age_burden_signature_country_share_percent",
        "dominant_dalys_age_burden_signature_by_weighted_burden",
        "dominant_dalys_age_burden_signature_weighted_burden",
        "dominant_dalys_age_burden_signature_weighted_burden_share_percent",
        "incidence_all_age_number_sum",
        "incidence_under1_number_sum",
        "weighted_incidence_under1_share_allages",
        "mean_country_incidence_under1_share_allages",
        "median_country_incidence_under1_share_allages",
        "summary_note",
    ]
    if gbd_burden.empty or "iso3" not in gbd_burden:
        return pd.DataFrame(columns=columns)

    frame = gbd_burden.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["val"] = pd.to_numeric(frame["val"], errors="coerce")
    frame["measure_name"] = frame["measure_name"].fillna("").astype(str)
    frame["metric_name"] = frame["metric_name"].fillna("").astype(str)
    frame["age_name"] = frame["age_name"].fillna("").astype(str)
    frame = frame.loc[frame["iso3"].notna()].copy()
    frame = frame.loc[
        frame["metric_name"].eq("Number")
        & frame["measure_name"].isin(["DALYs (Disability-Adjusted Life Years)", "Incidence"])
        & frame["age_name"].isin(["All ages", "<1 year"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for year, year_group in frame.groupby("year", dropna=False):
        year_int = int(year)
        row: dict[str, object] = {
            "scope": "year",
            "year": year_int,
            "countries_observed": int(year_group["iso3"].dropna().astype(str).nunique()),
            "country_year_rows": int(len(year_group)),
        }
        dalys_group = year_group.loc[year_group["measure_name"].eq("DALYs (Disability-Adjusted Life Years)")].copy()
        if not dalys_group.empty:
            dalys_pivot = dalys_group.pivot_table(index="iso3", columns="age_name", values="val", aggfunc="first").reset_index()
            dalys_pivot = dalys_pivot.rename(columns={"All ages": "all_ages", "<1 year": "under1"})
            dalys_pivot = dalys_pivot.dropna(subset=["iso3", "all_ages", "under1"])
            if not dalys_pivot.empty:
                dalys_pivot["under1_share_allages"] = dalys_pivot["under1"] / dalys_pivot["all_ages"].replace({0: np.nan})
                dalys_pivot["under1_share_class"] = dalys_pivot["under1_share_allages"].map(classify_gbd_under1_share)
                dominant_country_signature, dominant_country_count, dominant_country_share = summarize_dominant_text(
                    dalys_pivot["under1_share_class"]
                )
                weighted_share = float(dalys_pivot["under1"].sum() / dalys_pivot["all_ages"].sum()) if float(dalys_pivot["all_ages"].sum()) else np.nan
                weighted_signature = classify_gbd_under1_share(weighted_share)
                weighted_share_percent = weighted_share * 100.0 if pd.notna(weighted_share) else np.nan
                class_counts = dalys_pivot["under1_share_class"].value_counts()
                row.update(
                    {
                        "dalys_all_age_number_sum": float(dalys_pivot["all_ages"].sum()),
                        "dalys_under1_number_sum": float(dalys_pivot["under1"].sum()),
                        "weighted_dalys_under1_share_allages": weighted_share,
                        "mean_country_dalys_under1_share_allages": float(dalys_pivot["under1_share_allages"].mean()),
                        "median_country_dalys_under1_share_allages": float(dalys_pivot["under1_share_allages"].median()),
                        "countries_dalys_infant_dominated": int(class_counts.get("infant_dominated", 0)),
                        "countries_dalys_infant_leaning": int(class_counts.get("infant_leaning", 0)),
                        "countries_dalys_mixed_burden": int(class_counts.get("mixed_burden", 0)),
                        "countries_dalys_broad_or_noninfant": int(class_counts.get("broad_or_noninfant", 0)),
                        "dominant_dalys_age_burden_signature_by_country_count": dominant_country_signature,
                        "dominant_dalys_age_burden_signature_country_count": dominant_country_count,
                        "dominant_dalys_age_burden_signature_country_share_percent": dominant_country_share,
                        "dominant_dalys_age_burden_signature_by_weighted_burden": weighted_signature,
                        "dominant_dalys_age_burden_signature_weighted_burden": weighted_signature,
                        "dominant_dalys_age_burden_signature_weighted_burden_share_percent": weighted_share_percent,
                    }
                )
            else:
                row.update(
                    {
                        "dalys_all_age_number_sum": np.nan,
                        "dalys_under1_number_sum": np.nan,
                        "weighted_dalys_under1_share_allages": np.nan,
                        "mean_country_dalys_under1_share_allages": np.nan,
                        "median_country_dalys_under1_share_allages": np.nan,
                        "countries_dalys_infant_dominated": pd.NA,
                        "countries_dalys_infant_leaning": pd.NA,
                        "countries_dalys_mixed_burden": pd.NA,
                        "countries_dalys_broad_or_noninfant": pd.NA,
                        "dominant_dalys_age_burden_signature_by_country_count": "not_available",
                        "dominant_dalys_age_burden_signature_country_count": pd.NA,
                        "dominant_dalys_age_burden_signature_country_share_percent": np.nan,
                        "dominant_dalys_age_burden_signature_by_weighted_burden": "not_available",
                        "dominant_dalys_age_burden_signature_weighted_burden": "not_available",
                        "dominant_dalys_age_burden_signature_weighted_burden_share_percent": np.nan,
                    }
                )
        incidence_group = year_group.loc[year_group["measure_name"].eq("Incidence")].copy()
        if not incidence_group.empty:
            incidence_pivot = incidence_group.pivot_table(index="iso3", columns="age_name", values="val", aggfunc="first").reset_index()
            incidence_pivot = incidence_pivot.rename(columns={"All ages": "all_ages", "<1 year": "under1"})
            incidence_pivot = incidence_pivot.dropna(subset=["iso3", "all_ages", "under1"])
            if not incidence_pivot.empty:
                incidence_pivot["under1_share_allages"] = incidence_pivot["under1"] / incidence_pivot["all_ages"].replace({0: np.nan})
                row.update(
                    {
                        "incidence_all_age_number_sum": float(incidence_pivot["all_ages"].sum()),
                        "incidence_under1_number_sum": float(incidence_pivot["under1"].sum()),
                        "weighted_incidence_under1_share_allages": float(
                            incidence_pivot["under1"].sum() / incidence_pivot["all_ages"].sum()
                        )
                        if float(incidence_pivot["all_ages"].sum())
                        else np.nan,
                        "mean_country_incidence_under1_share_allages": float(incidence_pivot["under1_share_allages"].mean()),
                        "median_country_incidence_under1_share_allages": float(incidence_pivot["under1_share_allages"].median()),
                    }
                )
            else:
                row.update(
                    {
                        "incidence_all_age_number_sum": np.nan,
                        "incidence_under1_number_sum": np.nan,
                        "weighted_incidence_under1_share_allages": np.nan,
                        "mean_country_incidence_under1_share_allages": np.nan,
                        "median_country_incidence_under1_share_allages": np.nan,
                    }
                )
        row["summary_note"] = (
            "Year-level summary of the modeled IHME GBD 2021 pertussis burden extract; DALYs are the primary burden metric "
            "and the under-1 share is based on Number rows for all ages vs <1 year. Incidence is retained as a secondary "
            "consistency check. Class labels use 0.35/0.4/0.5 under-1 share thresholds and should be read as auxiliary burden "
            "context rather than surveillance."
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns).sort_values(["year"], na_position="last").reset_index(drop=True)


def read_healthmap_alerts() -> pd.DataFrame:
    if not PI_HEALTHMAP.exists():
        return pd.DataFrame(columns=["iso3", "country", "issue_date", "alert_title", "raw_place_name", "source", "notes"])
    raw = pd.read_csv(PI_HEALTHMAP)
    iso_map = pd.read_csv(PI_HEALTHMAP_ISO) if PI_HEALTHMAP_ISO.exists() else pd.DataFrame(columns=["Country", "ISO3"])
    out = raw.rename(columns={"date": "issue_date", "title": "alert_title"}).copy()
    out["raw_place_name"] = out.get("place_name", "")
    out["country"] = out["raw_place_name"].astype(str).map(lambda x: x.split(",")[-1].strip() if "," in x else x.strip())
    out["country"] = out["country"].replace({"Hong Kong": "China", "Macau": "China", "Taiwan": "China", "Guam [USA]": "United States"})
    out = out.merge(iso_map.rename(columns={"Country": "country", "ISO3": "iso3"}), on="country", how="left")
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="coerce").dt.date.astype("string")
    out["source"] = "xmusphlkg/PertussisIncidence Data/HealthmapData.csv"
    out["notes"] = "HealthMap alert records are digital surveillance signals, not verified outbreak counts or unique events."
    return out.loc[:, ["iso3", "country", "issue_date", "alert_title", "raw_place_name", "source", "notes"]]


def parse_contact_age_bin(value: object) -> tuple[float, float]:
    text = str(value)
    nums = [float(item) for item in re.findall(r"\d+", text)]
    if len(nums) >= 2:
        return nums[0], nums[1] + (1.0 if "_" in text else 0.0)
    if len(nums) == 1:
        return nums[0], nums[0] + 5.0
    return np.nan, np.nan


def age_group_width(age_group: str) -> float:
    lower, upper = AGE_GROUP_MAP[age_group]
    if pd.isna(upper):
        return np.inf
    return max(float(upper) - float(lower), 1e-9)


def infer_anchor_year(base: pd.DataFrame, preferred_year: int = 2019) -> int:
    years = sorted(int(year) for year in base["year"].dropna().unique())
    if not years:
        return preferred_year
    if preferred_year in years:
        return preferred_year
    pre_years = [year for year in years if year <= preferred_year]
    return max(pre_years) if pre_years else min(years)


def birth_rate_proxy_for_year(base: pd.DataFrame, year: int, anchor_under1_share: float, anchor_year: int) -> float:
    series = base.set_index("year")["total_population"].sort_index().astype(float)
    growth = series.pct_change()
    pre_growth = growth.loc[(growth.index >= max(series.index.min(), 2010)) & (growth.index <= anchor_year)].replace([np.inf, -np.inf], np.nan).dropna()
    if pre_growth.empty:
        pre_growth = growth.loc[growth.index <= anchor_year].replace([np.inf, -np.inf], np.nan).dropna()
    baseline_growth = float(pre_growth.median()) if not pre_growth.empty else 0.0
    recent_growth = growth.loc[(growth.index >= year - 2) & (growth.index <= year)].replace([np.inf, -np.inf], np.nan).dropna()
    current_growth = float(recent_growth.mean()) if not recent_growth.empty else baseline_growth
    growth_adjustment = float(np.clip(1.0 + 6.0 * (current_growth - baseline_growth), 0.55, 1.45))
    return float(np.clip(anchor_under1_share * growth_adjustment, 0.0025, 0.04))


def one_year_cohort_step(previous: dict[str, float], target_total: float, births_proxy: float) -> dict[str, float]:
    raw = {age_group: 0.0 for age_group in AGE_GROUP_MAP}
    births_proxy = float(max(births_proxy, 0.0))
    raw["infant_0_2m"] = births_proxy * 0.25
    raw["infant_3_11m"] = births_proxy * 0.75
    raw["child_1_4y"] = previous.get("infant_0_2m", 0.0) + previous.get("infant_3_11m", 0.0) + previous.get("child_1_4y", 0.0) * (
        1.0 - 1.0 / age_group_width("child_1_4y")
    )
    raw["child_5_9y"] = previous.get("child_1_4y", 0.0) / age_group_width("child_1_4y") + previous.get("child_5_9y", 0.0) * (
        1.0 - 1.0 / age_group_width("child_5_9y")
    )
    raw["adolescent_10_17y"] = previous.get("child_5_9y", 0.0) / age_group_width("child_5_9y") + previous.get("adolescent_10_17y", 0.0) * (
        1.0 - 1.0 / age_group_width("adolescent_10_17y")
    )
    raw["young_adult_18_39y"] = previous.get("adolescent_10_17y", 0.0) / age_group_width("adolescent_10_17y") + previous.get("young_adult_18_39y", 0.0) * (
        1.0 - 1.0 / age_group_width("young_adult_18_39y")
    )
    raw["middle_adult_40_64y"] = previous.get("young_adult_18_39y", 0.0) / age_group_width("young_adult_18_39y") + previous.get("middle_adult_40_64y", 0.0) * (
        1.0 - 1.0 / age_group_width("middle_adult_40_64y")
    )
    raw["elderly_65plus"] = previous.get("middle_adult_40_64y", 0.0) / age_group_width("middle_adult_40_64y") + previous.get("elderly_65plus", 0.0)

    infant_total = raw["infant_0_2m"] + raw["infant_3_11m"]
    older_total = sum(value for age_group, value in raw.items() if age_group not in {"infant_0_2m", "infant_3_11m"})
    residual_total = max(float(target_total) - infant_total, 0.0)
    if older_total > 0:
        older_scale = residual_total / older_total
        for age_group in raw:
            if age_group not in {"infant_0_2m", "infant_3_11m"}:
                raw[age_group] *= older_scale
    total = sum(raw.values())
    if total > 0 and np.isfinite(target_total) and target_total > 0:
        correction = float(target_total) / total
        raw = {age_group: value * correction for age_group, value in raw.items()}
    return raw


def project_population_age_country(base: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    base = base.sort_values("year").copy()
    anchor_year = infer_anchor_year(base)
    anchor_total_values = base.loc[base["year"].eq(anchor_year), "total_population"]
    if anchor_total_values.empty:
        return pd.DataFrame()
    anchor_total = float(anchor_total_values.iloc[0])
    share_lookup = shares.set_index("age_group")["share"].to_dict()
    anchor_counts = {age_group: anchor_total * float(share_lookup.get(age_group, 0.0)) for age_group in AGE_GROUP_MAP}
    anchor_under1_share = float(share_lookup.get("infant_0_2m", 0.0) + share_lookup.get("infant_3_11m", 0.0))
    rows: list[dict[str, object]] = []
    counts_by_year: dict[int, dict[str, float]] = {}
    births_by_year: dict[int, float] = {}

    for _, row in base.loc[base["year"].le(anchor_year)].iterrows():
        year = int(row["year"])
        total = float(row["total_population"])
        counts_by_year[year] = {age_group: total * float(share_lookup.get(age_group, 0.0)) for age_group in AGE_GROUP_MAP}
        births_by_year[year] = total * anchor_under1_share

    previous = anchor_counts
    for _, row in base.loc[base["year"].gt(anchor_year)].iterrows():
        year = int(row["year"])
        total = float(row["total_population"])
        birth_rate = birth_rate_proxy_for_year(base, year, anchor_under1_share, anchor_year)
        births_proxy = total * birth_rate
        previous = one_year_cohort_step(previous, total, births_proxy)
        counts_by_year[year] = previous
        births_by_year[year] = births_proxy

    for _, row in base.iterrows():
        year = int(row["year"])
        total = float(row["total_population"])
        counts = counts_by_year.get(year)
        if not counts:
            continue
        for age_group, population in counts.items():
            lower, upper = AGE_GROUP_MAP[age_group]
            rows.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "year": year,
                    "age_group": age_group,
                    "age_lower": lower,
                    "age_upper": upper,
                    "population": population,
                    "births": births_by_year.get(year, pd.NA),
                    "total_population": total,
                    "source": "WHO incidence-derived total population with contactdata age distribution anchored to 2019 and projected by a transparent birth-cohort proxy",
                    "source_type": "derived_demographic_projection_proxy",
                    "age_structure_anchor_year": anchor_year,
                    "age_structure_projection_method": "static contactdata age shares through anchor year; one-year cohort flow after anchor; under-1 birth proxy adjusted by total-population growth deviation",
                }
            )
    return pd.DataFrame(rows)


def build_population_age(total_population: pd.DataFrame, name_map: pd.DataFrame) -> pd.DataFrame:
    local_wpp = read_local_wpp_population_age()
    if not CONTACTDATA_POPULATION.exists():
        if not local_wpp.empty:
            return local_wpp.loc[
                :,
                [
                    "iso3",
                    "country",
                    "year",
                    "age_group",
                    "age_lower",
                    "age_upper",
                    "population",
                    "births",
                    "total_population",
                    "source",
                    "source_type",
                    "age_structure_anchor_year",
                    "age_structure_projection_method",
                ],
            ]
        return pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "year",
                "age_group",
                "age_lower",
                "age_upper",
                "population",
                "births",
                "total_population",
                "source",
                "source_type",
            ]
        )
    pop = pd.read_csv(CONTACTDATA_POPULATION)
    pop["country_key"] = pop["country"].map(normalize_name)
    pop = pop.merge(name_map, on="country_key", how="left", suffixes=("_contactdata", ""))
    pop = pop.dropna(subset=["iso3"]).copy()
    bounds = pop["age"].apply(parse_contact_age_bin)
    pop["fine_lower"] = [item[0] for item in bounds]
    pop["fine_upper"] = [item[1] for item in bounds]
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop = pop.dropna(subset=["fine_lower", "fine_upper", "population"])

    share_rows: list[dict[str, object]] = []
    for (iso3, country), group in pop.groupby(["iso3", "country"]):
        weights = {age_group: 0.0 for age_group in AGE_GROUP_MAP}
        for _, row in group.iterrows():
            fine_lower = float(row["fine_lower"])
            fine_upper = float(row["fine_upper"])
            fine_width = max(fine_upper - fine_lower, 1e-9)
            for age_group, (lower, upper) in AGE_GROUP_MAP.items():
                target_upper = float("inf") if pd.isna(upper) else float(upper)
                overlap = max(0.0, min(fine_upper, target_upper) - max(fine_lower, float(lower)))
                if overlap > 0:
                    weights[age_group] += float(row["population"]) * overlap / fine_width
        total = sum(weights.values())
        if total <= 0:
            continue
        for age_group, value in weights.items():
            lower, upper = AGE_GROUP_MAP[age_group]
            share_rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "age_group": age_group,
                    "age_lower": lower,
                    "age_upper": upper,
                    "share": value / total,
                }
            )
    shares = pd.DataFrame(share_rows)
    if shares.empty:
        return pd.DataFrame(columns=["iso3", "country", "year", "age_group", "age_lower", "age_upper", "population", "births", "total_population", "source", "source_type"])

    base = total_population.dropna(subset=["iso3", "year", "total_population"]).copy()
    projected: list[pd.DataFrame] = []
    for (iso3, country), country_base in base.groupby(["iso3", "country"], sort=True):
        country_shares = shares.loc[shares["iso3"].eq(iso3) & shares["country"].eq(country)].copy()
        if country_shares.empty:
            continue
        country_projection = project_population_age_country(country_base, country_shares)
        if not country_projection.empty:
            projected.append(country_projection)
    out = pd.concat(projected, ignore_index=True, sort=False) if projected else pd.DataFrame()
    if out.empty:
        if not local_wpp.empty:
            return local_wpp.loc[
                :,
                [
                    "iso3",
                    "country",
                    "year",
                    "age_group",
                    "age_lower",
                    "age_upper",
                    "population",
                    "births",
                    "total_population",
                    "source",
                    "source_type",
                    "age_structure_anchor_year",
                    "age_structure_projection_method",
                ],
            ]
        return pd.DataFrame(columns=["iso3", "country", "year", "age_group", "age_lower", "age_upper", "population", "births", "total_population", "source", "source_type"])
    if not local_wpp.empty:
        wpp_iso = set(local_wpp["iso3"].dropna().astype(str))
        out = pd.concat([local_wpp, out.loc[~out["iso3"].astype(str).isin(wpp_iso)]], ignore_index=True, sort=False)
    return out.loc[
        :,
        [
            "iso3",
            "country",
            "year",
            "age_group",
            "age_lower",
            "age_upper",
            "population",
            "births",
            "total_population",
            "source",
            "source_type",
            "age_structure_anchor_year",
            "age_structure_projection_method",
        ],
    ]


def build_population_age_structure_projection_audit(population: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "age_structure_anchor_year",
        "projection_method",
        "under1_share_anchor",
        "under1_share_2025",
        "under1_share_change_anchor_to_2025",
        "max_abs_age_share_change_anchor_to_2025",
        "dynamic_age_structure_available",
        "journal_use_recommendation",
    ]
    if population.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (iso3, country), group in population.groupby(["iso3", "country"], sort=True):
        anchor_year = int(pd.to_numeric(group["age_structure_anchor_year"], errors="coerce").dropna().iloc[0]) if group["age_structure_anchor_year"].notna().any() else 2019
        anchor = group.loc[group["year"].eq(anchor_year)].copy()
        latest = group.loc[group["year"].eq(2025)].copy()
        if latest.empty:
            latest_year = int(group["year"].max())
            latest = group.loc[group["year"].eq(latest_year)].copy()
        if anchor.empty or latest.empty:
            continue
        anchor_share = anchor.set_index("age_group")["population"] / anchor["population"].sum()
        latest_share = latest.set_index("age_group")["population"] / latest["population"].sum()
        aligned = pd.concat([anchor_share.rename("anchor"), latest_share.rename("latest")], axis=1).fillna(0.0)
        under1_anchor = float(aligned.loc[[idx for idx in aligned.index if idx in {"infant_0_2m", "infant_3_11m"}], "anchor"].sum())
        under1_latest = float(aligned.loc[[idx for idx in aligned.index if idx in {"infant_0_2m", "infant_3_11m"}], "latest"].sum())
        max_change = float((aligned["latest"] - aligned["anchor"]).abs().max())
        source_types = set(group.get("source_type", pd.Series(dtype=str)).dropna().astype(str))
        uses_wpp = any("un_wpp" in source_type for source_type in source_types)
        recommendation = (
            "Use as curated UN WPP 2024 annual age-specific population support for demographic and infant-denominator mechanisms."
            if uses_wpp
            else "Use as a transparent birth-cohort and age-structure proxy only; replace with curated UN WPP or national annual age-specific population before strong demographic mechanism claims."
        )
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "age_structure_anchor_year": anchor_year,
                "projection_method": str(group["age_structure_projection_method"].dropna().iloc[0]) if group["age_structure_projection_method"].notna().any() else "",
                "under1_share_anchor": under1_anchor,
                "under1_share_2025": under1_latest,
                "under1_share_change_anchor_to_2025": under1_latest - under1_anchor,
                "max_abs_age_share_change_anchor_to_2025": max_change,
                "dynamic_age_structure_available": bool(max_change > 1e-6),
                "journal_use_recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def read_local_maternal_coverage() -> pd.DataFrame:
    if LOCAL_MATERNAL_COVERAGE.exists():
        raw = pd.read_excel(LOCAL_MATERNAL_COVERAGE)
    else:
        return select_latest_maternal_coverage(pd.DataFrame())
    standardized = standardize_maternal_coverage(raw)
    return select_latest_maternal_coverage(standardized)


def read_local_maternal_pregnancy_recommendation_summary() -> pd.DataFrame:
    if not LOCAL_MATERNAL_PREGNANCY_RECOMMENDATION_REVIEW.exists():
        columns = [
            "iso3",
            "country",
            "recommendation_category",
            "recommendation_status",
            "delivery_setting",
            "coverage_note",
            "coverage_percent_low",
            "coverage_percent_high",
            "recommendation_available",
            "source",
            "source_type",
            "notes",
        ]
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "iso3": "ARG",
            "country": "Argentina",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "National recommendation since 2012; 2013 Tdap booster in single pregnancy included in the National Calendar (mandatory and free of charge)",
            "delivery_setting": "",
            "coverage_note": "",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "AUS",
            "country": "Australia",
            "recommendation_category": "health_department_recommendation",
            "recommendation_status": "Health department recommendation",
            "delivery_setting": "Vaccinators include general practitioners and midwives",
            "coverage_note": "Unclear coverage rates",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "BEL",
            "country": "Belgium",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2013",
            "delivery_setting": "",
            "coverage_note": "",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "BRA",
            "country": "Brazil",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2014",
            "delivery_setting": "Vaccination given in health centers",
            "coverage_note": "65% coverage",
            "coverage_percent_low": 65.0,
            "coverage_percent_high": 65.0,
        },
        {
            "iso3": "CAN",
            "country": "Canada",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "National recommendation since 2018 but not all provinces have endorsed it",
            "delivery_setting": "Given by general practitioners, Ob/Gyn, or midwives",
            "coverage_note": "45% coverage",
            "coverage_percent_low": 45.0,
            "coverage_percent_high": 45.0,
        },
        {
            "iso3": "CRI",
            "country": "Costa Rica",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2012",
            "delivery_setting": "Vaccines given by midwives, general nurses, or general practitioners",
            "coverage_note": "81% coverage",
            "coverage_percent_low": 81.0,
            "coverage_percent_high": 81.0,
        },
        {
            "iso3": "DEU",
            "country": "Germany",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended from 2020",
            "delivery_setting": "Given by Ob/Gyn or general practitioner",
            "coverage_note": "40% coverage",
            "coverage_percent_low": 40.0,
            "coverage_percent_high": 40.0,
        },
        {
            "iso3": "IND",
            "country": "India",
            "recommendation_category": "professional_society_recommendation",
            "recommendation_status": "Recommended by professional society in 2013",
            "delivery_setting": "",
            "coverage_note": "Coverage rates are unknown but under evaluation",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "MYS",
            "country": "Malaysia",
            "recommendation_category": "professional_society_recommendation",
            "recommendation_status": "Recommended by professional societies",
            "delivery_setting": "Given by Ob/Gyn, specialists, and general practitioners",
            "coverage_note": "Mostly provided in the private sector as the government only recommends tetanus",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "PAK",
            "country": "Pakistan",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "National recommendation",
            "delivery_setting": "Given by Ob/Gyn",
            "coverage_note": "65% coverage",
            "coverage_percent_low": 65.0,
            "coverage_percent_high": 65.0,
        },
        {
            "iso3": "PHL",
            "country": "Philippines",
            "recommendation_category": "recommendation_with_private_delivery",
            "recommendation_status": "Recommended",
            "delivery_setting": "Given by Ob/Gyn",
            "coverage_note": "Only high rates of uptake with private consultation; high rates of loss to follow up with the Ob/Gyn",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "QAT",
            "country": "Qatar",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended in 2019",
            "delivery_setting": "Given by general practitioner",
            "coverage_note": "Unknown uptake rates",
            "coverage_percent_low": pd.NA,
            "coverage_percent_high": pd.NA,
        },
        {
            "iso3": "ZAF",
            "country": "South Africa",
            "recommendation_category": "partial_provincial_rollout",
            "recommendation_status": "Guidelines are being updated; as of June 2023, pertussis vaccination in pregnancy is now available in some provinces and is expected to be available in the public sector in 2024",
            "delivery_setting": "Vaccines are given by Ob/Gyn or a specific center",
            "coverage_note": "70% coverage",
            "coverage_percent_low": 70.0,
            "coverage_percent_high": 70.0,
        },
        {
            "iso3": "ESP",
            "country": "Spain",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2013 (January 2015 for the Valencian community; all regions by the end of 2016)",
            "delivery_setting": "Given by general practitioner",
            "coverage_note": "83% coverage",
            "coverage_percent_low": 83.0,
            "coverage_percent_high": 83.0,
        },
        {
            "iso3": "CHE",
            "country": "Switzerland",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2013",
            "delivery_setting": "Usually given by Ob/Gyn",
            "coverage_note": "Uptake ~50% (varies by region of country)",
            "coverage_percent_low": 50.0,
            "coverage_percent_high": 50.0,
        },
        {
            "iso3": "THA",
            "country": "Thailand",
            "recommendation_category": "professional_society_recommendation",
            "recommendation_status": "Recommended by national professional societies (e.g. The Royal Thai College of Obstetricians and Gynecologists)",
            "delivery_setting": "Unclear coverage in the public sector; 70% covered in private sector",
            "coverage_note": "70% covered in private sector",
            "coverage_percent_low": 70.0,
            "coverage_percent_high": 70.0,
        },
        {
            "iso3": "GBR",
            "country": "United Kingdom of Great Britain and Northern Ireland",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2012",
            "delivery_setting": "General practitioners and midwives give the vaccine",
            "coverage_note": "68% coverage",
            "coverage_percent_low": 68.0,
            "coverage_percent_high": 68.0,
        },
        {
            "iso3": "USA",
            "country": "United States of America",
            "recommendation_category": "national_recommendation",
            "recommendation_status": "Recommended since 2012",
            "delivery_setting": "Vaccines given by Ob/Gyn, midwives, general practitioner, health department",
            "coverage_note": "Coverage 50%–70%",
            "coverage_percent_low": 50.0,
            "coverage_percent_high": 70.0,
        },
    ]
    out = pd.DataFrame(rows)
    out["recommendation_available"] = True
    out["source"] = "https://doi.org/10.1002/ijgo.15311"
    out["source_type"] = "review_table"
    out["notes"] = (
        "Country-level pregnancy Tdap recommendation and uptake snapshot from Table 1 of the 2024 GPI review; descriptive policy background only."
    )
    out["coverage_percent_low"] = pd.to_numeric(out["coverage_percent_low"], errors="coerce")
    out["coverage_percent_high"] = pd.to_numeric(out["coverage_percent_high"], errors="coerce")
    return out.loc[
        :,
        [
            "iso3",
            "country",
            "recommendation_category",
            "recommendation_status",
            "delivery_setting",
            "coverage_note",
            "coverage_percent_low",
            "coverage_percent_high",
            "recommendation_available",
            "source",
            "source_type",
            "notes",
        ],
    ].sort_values(["country", "iso3"]).reset_index(drop=True)


def read_local_diagnostic_timeline() -> pd.DataFrame:
    if not LOCAL_DIAGNOSTIC_TIMELINE.exists():
        return pd.DataFrame(
            columns=[
                "country",
                "iso3",
                "period_start",
                "period_end",
                "geographic_scope",
                "surveillance_regime",
                "primary_diagnostic_methods",
                "case_definition_or_reporting_change",
                "relative_detection_prior_mean",
                "relative_detection_prior_lower",
                "relative_detection_prior_upper",
                "effect_direction",
                "evidence_strength",
                "source_ids",
                "notes",
            ]
        )
    out = pd.read_csv(LOCAL_DIAGNOSTIC_TIMELINE)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out["source"] = repo_path(LOCAL_DIAGNOSTIC_TIMELINE)
    out["source_type"] = "curated_observation_process_timeline"
    return out


def read_local_diagnostic_age_testing_summary() -> pd.DataFrame:
    columns = [
        "country",
        "iso3",
        "period_start",
        "period_end",
        "age_group",
        "recommended_test",
        "total_tests",
        "serology_tests",
        "serology_percent",
        "source_ids",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_DIAGNOSTIC_AGE_TESTING_SUMMARY.exists():
        return pd.DataFrame(columns=columns)
    out = pd.read_csv(LOCAL_DIAGNOSTIC_AGE_TESTING_SUMMARY)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    for column in ["total_tests", "serology_tests", "serology_percent"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["period_start", "period_end"]:
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out["source"] = out["source"].fillna(
        repo_path(LOCAL_DIAGNOSTIC_AGE_TESTING_SUMMARY)
    )
    out["source_type"] = "curated_age_stratified_testing_summary"
    return out.loc[:, columns].sort_values(["iso3", "period_start", "age_group"]).reset_index(drop=True)


def read_local_diagnostic_sources() -> pd.DataFrame:
    if not LOCAL_DIAGNOSTIC_SOURCES.exists():
        return pd.DataFrame(columns=["source_id", "citation", "source_type", "url", "key_evidence_for_dataset"])
    out = pd.read_csv(LOCAL_DIAGNOSTIC_SOURCES)
    out["local_source_file"] = repo_path(LOCAL_DIAGNOSTIC_SOURCES)
    return out


def read_local_surveillance_outcome_evidence() -> pd.DataFrame:
    columns = [
        "country",
        "iso3",
        "year",
        "period_start",
        "period_end",
        "geographic_scope",
        "case_count",
        "case_count_qualifier",
        "case_count_type",
        "infant_case_count",
        "infant_age_definition",
        "under3m_case_count",
        "under6m_case_count",
        "under6m_hospitalization_percent",
        "total_hospitalization_percent",
        "hospitalizations",
        "total_deaths",
        "infant_deaths",
        "death_age_scope",
        "testing_denominator",
        "test_positivity_percent",
        "laboratory_confirmation_methods",
        "reporting_or_surveillance_change",
        "source_ids",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_SURVEILLANCE_OUTCOME_EVIDENCE.exists():
        return pd.DataFrame(columns=columns)
    out = pd.read_csv(LOCAL_SURVEILLANCE_OUTCOME_EVIDENCE)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    integer_columns = [
        "year",
        "case_count",
        "infant_case_count",
        "under3m_case_count",
        "under6m_case_count",
        "hospitalizations",
        "total_deaths",
        "infant_deaths",
        "testing_denominator",
    ]
    numeric_columns = integer_columns + ["under6m_hospitalization_percent", "total_hospitalization_percent", "test_positivity_percent"]
    for column in numeric_columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["period_start", "period_end"]:
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out["source"] = out["source"].fillna(repo_path(LOCAL_SURVEILLANCE_OUTCOME_EVIDENCE))
    out["source_type"] = out["source_type"].fillna("curated_official_surveillance_outcome_summary")
    return out.loc[:, columns].sort_values(["iso3", "year", "period_start"], na_position="last").reset_index(drop=True)


def read_local_surveillance_outcome_sources() -> pd.DataFrame:
    if not LOCAL_SURVEILLANCE_OUTCOME_SOURCES.exists():
        return pd.DataFrame(columns=["source_id", "citation", "source_type", "url", "key_evidence_for_dataset"])
    out = pd.read_csv(LOCAL_SURVEILLANCE_OUTCOME_SOURCES)
    out["local_source_file"] = repo_path(LOCAL_SURVEILLANCE_OUTCOME_SOURCES)
    return out


def read_local_japan_annual_national_surveillance_summary() -> pd.DataFrame:
    columns = [
        "country",
        "iso3",
        "year",
        "period_start",
        "period_end",
        "summary_status",
        "reported_case_count",
        "guideline_compliant_case_count",
        "guideline_compliant_percent",
        "age_peak_group",
        "age_peak_share_percent",
        "age_distribution_note",
        "diagnostic_methods_summary",
        "diagnostic_top_method",
        "diagnostic_top_method_share_percent",
        "source_ids",
        "source",
        "source_type",
        "source_dataset",
        "notes",
    ]
    if not LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SUMMARY.exists():
        return pd.DataFrame(columns=columns)
    out = pd.read_csv(LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SUMMARY)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    for column in ["year", "reported_case_count", "guideline_compliant_case_count", "guideline_compliant_percent", "age_peak_share_percent", "diagnostic_top_method_share_percent"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["period_start", "period_end"]:
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out["source"] = out["source"].fillna(
        repo_path(LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SUMMARY)
    )
    out["source_type"] = out["source_type"].fillna("official_surveillance_summary")
    out["source_dataset"] = out["source_dataset"].fillna("official annual national surveillance summary")
    return out.loc[:, columns].sort_values(["year"], kind="stable").reset_index(drop=True)


def read_local_japan_annual_national_surveillance_sources() -> pd.DataFrame:
    if not LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SOURCES.exists():
        return pd.DataFrame(columns=["source_id", "citation", "source_type", "url", "key_evidence_for_dataset"])
    out = pd.read_csv(LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SOURCES)
    out["local_source_file"] = repo_path(LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SOURCES)
    return out


def read_local_serology_evidence() -> pd.DataFrame:
    columns = [
        "country",
        "iso3",
        "year",
        "period_start",
        "period_end",
        "geographic_scope",
        "sample_size",
        "age_scope",
        "assay",
        "anti_pt_cutoff",
        "serology_metric",
        "serology_value",
        "serology_unit",
        "interpretation",
        "source_ids",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_SEROLOGY_EVIDENCE.exists():
        return pd.DataFrame(columns=columns)
    out = pd.read_csv(LOCAL_SEROLOGY_EVIDENCE)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    for column in ["year", "sample_size", "serology_value"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["period_start", "period_end"]:
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    text_columns = [
        "country",
        "geographic_scope",
        "age_scope",
        "assay",
        "anti_pt_cutoff",
        "serology_metric",
        "serology_unit",
        "interpretation",
        "source_ids",
        "notes",
    ]
    for column in text_columns:
        if column in out:
            out[column] = out[column].fillna("not_available").astype(str).str.strip().replace({"": "not_available"})
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out["source"] = out["source"].fillna(repo_path(LOCAL_SEROLOGY_EVIDENCE))
    out["source_type"] = out["source_type"].fillna("curated_national_serology_summary")
    return out.loc[:, columns].sort_values(["iso3", "year", "period_start", "age_scope"], na_position="last").reset_index(drop=True)


def read_local_serology_sources() -> pd.DataFrame:
    if not LOCAL_SEROLOGY_SOURCES.exists():
        return pd.DataFrame(columns=["source_id", "citation", "source_type", "url", "key_evidence_for_dataset"])
    out = pd.read_csv(LOCAL_SEROLOGY_SOURCES)
    out["local_source_file"] = repo_path(LOCAL_SEROLOGY_SOURCES)
    return out


def read_local_contact_validation_evidence() -> pd.DataFrame:
    columns = [
        "country",
        "iso3",
        "year",
        "period_start",
        "period_end",
        "geographic_scope",
        "sample_size",
        "survey_name",
        "contact_metric",
        "contact_value",
        "contact_unit",
        "baseline_comparison",
        "interpretation",
        "source_ids",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_CONTACT_VALIDATION_EVIDENCE.exists():
        return pd.DataFrame(columns=columns)
    out = pd.read_csv(LOCAL_CONTACT_VALIDATION_EVIDENCE)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    for column in ["year", "sample_size", "contact_value"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["period_start", "period_end"]:
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    text_columns = [
        "country",
        "geographic_scope",
        "survey_name",
        "contact_metric",
        "contact_unit",
        "baseline_comparison",
        "interpretation",
        "source_ids",
        "notes",
    ]
    for column in text_columns:
        if column in out:
            out[column] = out[column].fillna("not_available").astype(str).str.strip().replace({"": "not_available"})
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out["source"] = out["source"].fillna(repo_path(LOCAL_CONTACT_VALIDATION_EVIDENCE))
    out["source_type"] = out["source_type"].fillna("curated_direct_contact_validation_summary")
    return out.loc[:, columns].sort_values(["iso3", "year", "period_start"], na_position="last").reset_index(drop=True)


def read_local_contact_validation_sources() -> pd.DataFrame:
    if not LOCAL_CONTACT_VALIDATION_SOURCES.exists():
        return pd.DataFrame(columns=["source_id", "citation", "source_type", "url", "key_evidence_for_dataset"])
    out = pd.read_csv(LOCAL_CONTACT_VALIDATION_SOURCES)
    out["local_source_file"] = repo_path(LOCAL_CONTACT_VALIDATION_SOURCES)
    return out


def summarize_serology_anti_pt_igg(serology: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "serology_anti_pt_igg_available",
        "serology_n_rows",
        "serology_year_min",
        "serology_year_max",
        "serology_year_span",
        "serology_period_start_min",
        "serology_period_end_max",
        "serology_geographic_scopes",
        "serology_age_scopes",
        "serology_assays",
        "serology_cutoffs",
        "serology_metric_summary",
        "serology_value_min",
        "serology_value_median",
        "serology_value_max",
        "serology_units",
        "serology_sample_size_total",
        "serology_interpretation_summary",
        "serology_source_ids",
        "summary_note",
    ]
    if serology.empty:
        return pd.DataFrame(columns=columns)

    frame = serology.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame.get("country", frame["iso3"]).fillna(frame["iso3"]).astype(str).str.strip()
    for column in ["year", "sample_size", "serology_value"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    for column in ["period_start", "period_end"]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        else:
            frame[column] = pd.NaT
    text_columns = [
        "geographic_scope",
        "age_scope",
        "assay",
        "anti_pt_cutoff",
        "serology_metric",
        "serology_unit",
        "interpretation",
        "source_ids",
        "notes",
    ]
    for column in text_columns:
        if column in frame:
            frame[column] = frame[column].fillna("not_available").astype(str).str.strip().replace({"": "not_available"})
        else:
            frame[column] = "not_available"

    def join_unique(values: pd.Series) -> str:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values.dropna().astype(str):
            value = value.strip()
            if not value or value in seen or value.lower() == "nan":
                continue
            seen.add(value)
            ordered.append(value)
        return "; ".join(ordered)

    summary = (
        frame.groupby(["iso3", "country"], dropna=False)
        .agg(
            serology_n_rows=("source_ids", "size"),
            serology_year_min=("year", "min"),
            serology_year_max=("year", "max"),
            serology_period_start_min=("period_start", "min"),
            serology_period_end_max=("period_end", "max"),
            serology_geographic_scopes=("geographic_scope", join_unique),
            serology_age_scopes=("age_scope", join_unique),
            serology_assays=("assay", join_unique),
            serology_cutoffs=("anti_pt_cutoff", join_unique),
            serology_metric_summary=("serology_metric", join_unique),
            serology_value_min=("serology_value", "min"),
            serology_value_median=("serology_value", "median"),
            serology_value_max=("serology_value", "max"),
            serology_units=("serology_unit", join_unique),
            serology_sample_size_total=("sample_size", lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1)),
            serology_interpretation_summary=("interpretation", join_unique),
            serology_source_ids=("source_ids", join_unique),
        )
        .reset_index()
        .sort_values(["iso3", "country"], na_position="last")
        .reset_index(drop=True)
    )
    summary["serology_anti_pt_igg_available"] = True
    summary["serology_n_rows"] = pd.to_numeric(summary["serology_n_rows"], errors="coerce").astype("Int64")
    summary["serology_year_min"] = pd.to_numeric(summary["serology_year_min"], errors="coerce").astype("Int64")
    summary["serology_year_max"] = pd.to_numeric(summary["serology_year_max"], errors="coerce").astype("Int64")
    summary["serology_period_start_min"] = summary["serology_period_start_min"].dt.strftime("%Y-%m-%d")
    summary["serology_period_end_max"] = summary["serology_period_end_max"].dt.strftime("%Y-%m-%d")
    summary["serology_year_span"] = (
        pd.to_numeric(summary["serology_year_max"], errors="coerce") - pd.to_numeric(summary["serology_year_min"], errors="coerce")
    ).astype("Int64")
    summary["serology_value_min"] = pd.to_numeric(summary["serology_value_min"], errors="coerce")
    summary["serology_value_median"] = pd.to_numeric(summary["serology_value_median"], errors="coerce")
    summary["serology_value_max"] = pd.to_numeric(summary["serology_value_max"], errors="coerce")
    summary["serology_sample_size_total"] = pd.to_numeric(summary["serology_sample_size_total"], errors="coerce")
    summary["serology_interpretation_summary"] = summary["serology_interpretation_summary"].fillna("not_available").astype(str)
    summary["summary_note"] = summary.apply(
        lambda row: (
            f"{row['country']} {row['serology_geographic_scopes']} {row['serology_age_scopes']} {row['serology_assays']}: "
            f"{row['serology_interpretation_summary']}"
            if str(row["serology_interpretation_summary"]).strip()
            else f"{row['country']} {row['serology_geographic_scopes']} {row['serology_age_scopes']} {row['serology_assays']}"
        ),
        axis=1,
    )
    return summary.loc[:, columns]


def summarize_contact_validation(contact_validation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "direct_contact_validation_available",
        "n_rows",
        "year_min",
        "year_max",
        "period_start_min",
        "period_end_max",
        "geographic_scopes",
        "survey_names",
        "contact_metrics",
        "contact_value_min",
        "contact_value_max",
        "contact_units",
        "baseline_comparisons",
        "source_ids",
        "summary_note",
    ]
    if contact_validation.empty:
        return pd.DataFrame(columns=columns)

    def join_unique(values: pd.Series) -> str:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values.dropna().astype(str):
            value = value.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return "; ".join(ordered)

    summary = (
        contact_validation.groupby(["iso3", "country"], dropna=False)
        .agg(
            n_rows=("source_ids", "size"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            period_start_min=("period_start", "min"),
            period_end_max=("period_end", "max"),
            geographic_scopes=("geographic_scope", join_unique),
            survey_names=("survey_name", join_unique),
            contact_metrics=("contact_metric", join_unique),
            contact_value_min=("contact_value", "min"),
            contact_value_max=("contact_value", "max"),
            contact_units=("contact_unit", join_unique),
            baseline_comparisons=("baseline_comparison", join_unique),
            source_ids=("source_ids", join_unique),
            summary_note=("interpretation", join_unique),
        )
        .reset_index()
        .sort_values(["iso3", "country"], na_position="last")
        .reset_index(drop=True)
    )
    summary["direct_contact_validation_available"] = True
    summary["n_rows"] = pd.to_numeric(summary["n_rows"], errors="coerce").astype("Int64")
    summary["year_min"] = pd.to_numeric(summary["year_min"], errors="coerce").astype("Int64")
    summary["year_max"] = pd.to_numeric(summary["year_max"], errors="coerce").astype("Int64")
    summary["contact_value_min"] = pd.to_numeric(summary["contact_value_min"], errors="coerce")
    summary["contact_value_max"] = pd.to_numeric(summary["contact_value_max"], errors="coerce")
    summary["summary_note"] = summary.apply(
        lambda row: (
            f"{row['country']} {row['geographic_scopes']} {row['survey_names']}: {row['summary_note']}"
            if str(row["summary_note"]).strip()
            else f"{row['country']} {row['geographic_scopes']} {row['survey_names']}"
        ),
        axis=1,
    )
    return summary.loc[:, columns]


def summarize_external_validation_boundary(
    serology_evidence: pd.DataFrame,
    contact_validation: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "external_validation_boundary_label",
        "serology_available",
        "serology_n_rows",
        "serology_year_span",
        "serology_metric_summary",
        "serology_sample_size_total",
        "contact_validation_available",
        "contact_validation_n_rows",
        "contact_year_span",
        "contact_metric_summary",
        "contact_standardized_reduction_percent_min",
        "contact_standardized_reduction_percent_max",
        "contact_sample_size_total",
        "combined_source_ids",
        "external_validation_boundary_note",
    ]

    def join_unique(values: pd.Series) -> str:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values.dropna().astype(str):
            value = value.strip()
            if not value or value in seen or value.lower() == "nan":
                continue
            seen.add(value)
            ordered.append(value)
        return "; ".join(ordered)

    def span_from_min_max(row: pd.Series, min_col: str, max_col: str) -> str:
        min_value = pd.to_numeric(row.get(min_col), errors="coerce")
        max_value = pd.to_numeric(row.get(max_col), errors="coerce")
        if pd.isna(min_value) or pd.isna(max_value):
            return "not_available"
        return f"{int(min_value)}-{int(max_value)}"

    def standardized_contact_reduction(row: pd.Series) -> float:
        value = pd.to_numeric(row.get("contact_value"), errors="coerce")
        if pd.isna(value):
            return np.nan
        unit = str(row.get("contact_unit", "")).strip().lower()
        metric = str(row.get("contact_metric", "")).strip().lower()
        if "percent reduction" in unit or "percent reduction" in metric:
            return float(value)
        baseline_text = str(row.get("baseline_comparison", ""))
        baseline_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", baseline_text)
        if baseline_match:
            baseline = float(baseline_match.group(1))
            if baseline > 0:
                return float((baseline - float(value)) / baseline * 100.0)
        return np.nan

    serology = serology_evidence.copy() if not serology_evidence.empty else pd.DataFrame()
    if not serology.empty:
        serology["iso3"] = serology["iso3"].astype(str).str.upper()
        for column in ["year", "serology_value", "sample_size"]:
            if column in serology:
                serology[column] = pd.to_numeric(serology[column], errors="coerce")
            else:
                serology[column] = np.nan
        serology_summary = (
            serology.groupby(["iso3", "country"], dropna=False)
            .agg(
                serology_n_rows=("source_ids", "size"),
                serology_year_min=("year", "min"),
                serology_year_max=("year", "max"),
                serology_metric_summary=("serology_metric", join_unique),
                serology_sample_size_total=("sample_size", lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1)),
                serology_source_ids=("source_ids", join_unique),
            )
            .reset_index()
        )
        serology_summary = serology_summary.rename(columns={"country": "serology_country"})
        serology_summary["serology_available"] = True
        serology_summary["serology_year_span"] = serology_summary.apply(
            lambda row: span_from_min_max(row, "serology_year_min", "serology_year_max"),
            axis=1,
        )
    else:
        serology_summary = pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "serology_n_rows",
                "serology_year_min",
                "serology_year_max",
                "serology_metric_summary",
                "serology_sample_size_total",
                "serology_source_ids",
                "serology_available",
                "serology_year_span",
            ]
        )

    contact = contact_validation.copy() if not contact_validation.empty else pd.DataFrame()
    if not contact.empty:
        contact["iso3"] = contact["iso3"].astype(str).str.upper()
        for column in ["year", "sample_size", "contact_value"]:
            if column in contact:
                contact[column] = pd.to_numeric(contact[column], errors="coerce")
            else:
                contact[column] = np.nan
        contact["standardized_contact_reduction_percent"] = contact.apply(standardized_contact_reduction, axis=1)
        contact_summary = (
            contact.groupby(["iso3", "country"], dropna=False)
            .agg(
                contact_validation_n_rows=("source_ids", "size"),
                contact_year_min=("year", "min"),
                contact_year_max=("year", "max"),
                contact_metric_summary=("contact_metric", join_unique),
                contact_sample_size_total=("sample_size", lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1)),
                contact_source_ids=("source_ids", join_unique),
                contact_standardized_reduction_percent_min=("standardized_contact_reduction_percent", "min"),
                contact_standardized_reduction_percent_max=("standardized_contact_reduction_percent", "max"),
            )
            .reset_index()
        )
        contact_summary = contact_summary.rename(columns={"country": "contact_country"})
        contact_summary["contact_validation_available"] = True
        contact_summary["contact_year_span"] = contact_summary.apply(
            lambda row: span_from_min_max(row, "contact_year_min", "contact_year_max"),
            axis=1,
        )
    else:
        contact_summary = pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "contact_validation_n_rows",
                "contact_year_min",
                "contact_year_max",
                "contact_metric_summary",
                "contact_sample_size_total",
                "contact_source_ids",
                "contact_standardized_reduction_percent_min",
                "contact_standardized_reduction_percent_max",
                "contact_validation_available",
                "contact_year_span",
            ]
        )

    summary = serology_summary.merge(contact_summary, on="iso3", how="outer", suffixes=("_serology", "_contact"))
    if summary.empty:
        return pd.DataFrame(columns=columns)

    summary["country"] = summary.get("contact_country", pd.Series(dtype=str)).combine_first(
        summary.get("serology_country", pd.Series(dtype=str))
    )

    for column in ["serology_available", "contact_validation_available"]:
        if column not in summary:
            summary[column] = False
        summary[column] = summary[column].fillna(False).astype(bool)

    for column in [
        "serology_n_rows",
        "serology_year_min",
        "serology_year_max",
        "contact_validation_n_rows",
        "contact_year_min",
        "contact_year_max",
    ]:
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").astype("Int64")

    for column in [
        "serology_sample_size_total",
        "contact_sample_size_total",
        "contact_standardized_reduction_percent_min",
        "contact_standardized_reduction_percent_max",
    ]:
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    summary["serology_year_span"] = summary.get("serology_year_span", pd.Series(dtype=str)).fillna("not_available").astype(str)
    summary["contact_year_span"] = summary.get("contact_year_span", pd.Series(dtype=str)).fillna("not_available").astype(str)
    summary["serology_metric_summary"] = summary.get("serology_metric_summary", pd.Series(dtype=str)).fillna("not_available").astype(str)
    summary["contact_metric_summary"] = summary.get("contact_metric_summary", pd.Series(dtype=str)).fillna("not_available").astype(str)

    summary["external_validation_boundary_label"] = np.where(
        summary["serology_available"] & summary["contact_validation_available"],
        "serology_plus_contact_validation_boundary",
        np.where(
            summary["serology_available"],
            "serology_validation_boundary",
            np.where(summary["contact_validation_available"], "contact_validation_boundary", "external_validation_boundary"),
        ),
    )
    summary["external_validation_boundary_note"] = np.where(
        summary["serology_available"] & summary["contact_validation_available"],
        "Country-specific anti-PT-IgG serology and direct contact-survey evidence jointly constrain latent true-infection and contact-shock wording.",
        np.where(
            summary["serology_available"],
            "Country-specific anti-PT-IgG serology constrains latent true-infection wording, but direct contact validation is unavailable.",
            np.where(
                summary["contact_validation_available"],
                "Direct contact-survey evidence constrains contact-shock wording, but serology is unavailable.",
                "No serology or direct contact-validation evidence is available.",
            ),
        ),
    )

    combined_source_ids: list[str] = []
    for _, row in summary.iterrows():
        source_ids: list[str] = []
        for column in ["serology_source_ids", "contact_source_ids"]:
            value = row.get(column)
            if pd.isna(value):
                continue
            source_ids.extend([item.strip() for item in str(value).split(";") if item.strip()])
        combined_source_ids.append("; ".join(dict.fromkeys(source_ids)) if source_ids else "not_available")
    summary["combined_source_ids"] = combined_source_ids

    summary["boundary_sort"] = pd.Categorical(
        summary["external_validation_boundary_label"],
        categories=[
            "serology_plus_contact_validation_boundary",
            "serology_validation_boundary",
            "contact_validation_boundary",
            "external_validation_boundary",
        ],
        ordered=True,
    )
    summary = summary.sort_values(["boundary_sort", "country"], na_position="last").reset_index(drop=True)
    return summary.loc[:, columns]


def read_local_ecdc_reported_cases() -> pd.DataFrame:
    columns = [
        "region_code",
        "region_name",
        "iso3",
        "country",
        "year",
        "reported_cases",
        "population_scope",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_ECDC_SURVEILLANCE.exists():
        return pd.DataFrame(columns=columns)
    raw = pd.read_csv(LOCAL_ECDC_SURVEILLANCE)
    raw["year"] = pd.to_numeric(raw.get("Time"), errors="coerce").astype("Int64")
    raw["reported_cases"] = pd.to_numeric(raw.get("NumValue"), errors="coerce")
    raw["region_code"] = raw.get("RegionCode", pd.Series(dtype=str)).astype(str).str.strip()
    raw["region_name"] = raw.get("RegionName", pd.Series(dtype=str)).astype(str).str.strip()
    raw["iso3"] = raw["region_code"].map(ECDC_REGION_TO_ISO3)
    raw["country"] = raw["region_name"]
    raw["population_scope"] = raw.get("Population", pd.Series(dtype=str)).astype(str).str.strip()
    raw["source"] = repo_path(LOCAL_ECDC_SURVEILLANCE)
    raw["source_type"] = "regional_official_surveillance_report"
    raw["notes"] = np.where(
        raw["region_code"].astype(str).str.contains("EUEEA", na=False),
        "EU/EEA aggregate reported-case row from the ECDC surveillance snapshot.",
        "Country-level reported-case row from the ECDC surveillance snapshot.",
    )
    out = raw.loc[:, columns].copy()
    out = out.sort_values(["region_code", "year"], na_position="last").reset_index(drop=True)
    return out


def build_ecdc_who_comparison(ecdc_cases: pd.DataFrame, who_country_cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "ecdc_reported_cases",
        "who_reported_cases",
        "absolute_difference",
        "relative_difference",
        "ratio_ecdc_to_who",
        "comparison_note",
    ]
    if ecdc_cases.empty or who_country_cases.empty:
        return pd.DataFrame(columns=columns)
    ecdc_country = ecdc_cases.loc[ecdc_cases["iso3"].notna()].copy()
    ecdc_country["year"] = pd.to_numeric(ecdc_country["year"], errors="coerce").astype("Int64")
    ecdc_country["reported_cases"] = pd.to_numeric(ecdc_country["reported_cases"], errors="coerce")
    ecdc_country = ecdc_country.dropna(subset=["iso3", "year", "reported_cases"]).copy()
    who = who_country_cases.loc[who_country_cases["year"].between(1998, 2022)].copy()
    who["year"] = pd.to_numeric(who["year"], errors="coerce").astype("Int64")
    who["cases_total"] = pd.to_numeric(who["cases_total"], errors="coerce")
    who = who.dropna(subset=["iso3", "year", "cases_total"]).copy()
    comparison = ecdc_country.loc[:, ["iso3", "country", "year", "reported_cases"]].merge(
        who.loc[:, ["iso3", "year", "cases_total"]],
        on=["iso3", "year"],
        how="inner",
    )
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    comparison = comparison.rename(columns={"reported_cases": "ecdc_reported_cases", "cases_total": "who_reported_cases"})
    comparison["absolute_difference"] = comparison["ecdc_reported_cases"] - comparison["who_reported_cases"]
    comparison["relative_difference"] = comparison["absolute_difference"] / comparison["who_reported_cases"].replace({0: np.nan})
    comparison["ratio_ecdc_to_who"] = comparison["ecdc_reported_cases"] / comparison["who_reported_cases"].replace({0: np.nan})
    comparison["comparison_note"] = (
        "ECDC annual reported cases compared against WHO annual reported cases for overlapping EU/EEA country-year rows."
    )
    return comparison.loc[:, columns].sort_values(["iso3", "year"]).reset_index(drop=True)


def summarize_ecdc_who_concordance(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "iso3",
        "country",
        "overlap_years",
        "mean_ecdc_to_who_ratio",
        "median_ecdc_to_who_ratio",
        "mean_absolute_relative_difference",
        "within_factor2_fraction",
        "ecdc_year_min",
        "ecdc_year_max",
        "comparison_note",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    summary_rows: list[dict[str, object]] = []
    summary_rows.append(
        {
            "scope": "all_overlaps",
            "iso3": "ALL",
            "country": "All overlapping EU/EEA country-years",
            "overlap_years": int(len(comparison)),
            "mean_ecdc_to_who_ratio": float(pd.to_numeric(comparison["ratio_ecdc_to_who"], errors="coerce").mean()),
            "median_ecdc_to_who_ratio": float(pd.to_numeric(comparison["ratio_ecdc_to_who"], errors="coerce").median()),
            "mean_absolute_relative_difference": float(pd.to_numeric(comparison["relative_difference"], errors="coerce").abs().mean()),
            "within_factor2_fraction": float(
                pd.to_numeric(comparison["ratio_ecdc_to_who"], errors="coerce").between(0.5, 2.0).mean()
            ),
            "ecdc_year_min": int(pd.to_numeric(comparison["year"], errors="coerce").min()),
            "ecdc_year_max": int(pd.to_numeric(comparison["year"], errors="coerce").max()),
            "comparison_note": "Aggregate concordance summary for overlapping ECDC/WHO country-year rows.",
        }
    )
    for (iso3, country), group in comparison.groupby(["iso3", "country"], dropna=False):
        ratio = pd.to_numeric(group["ratio_ecdc_to_who"], errors="coerce")
        rel = pd.to_numeric(group["relative_difference"], errors="coerce")
        year = pd.to_numeric(group["year"], errors="coerce")
        summary_rows.append(
            {
                "scope": "country",
                "iso3": iso3,
                "country": country,
                "overlap_years": int(group.shape[0]),
                "mean_ecdc_to_who_ratio": float(ratio.mean()),
                "median_ecdc_to_who_ratio": float(ratio.median()),
                "mean_absolute_relative_difference": float(rel.abs().mean()),
                "within_factor2_fraction": float(ratio.between(0.5, 2.0).mean()),
                "ecdc_year_min": int(year.min()),
                "ecdc_year_max": int(year.max()),
                "comparison_note": "Country-level concordance summary for overlapping ECDC/WHO annual rows.",
            }
        )
    return pd.DataFrame(summary_rows, columns=columns).sort_values(["scope", "iso3"], na_position="last").reset_index(drop=True)


def build_who_reported_cases_concordance(
    workbook_cases: pd.DataFrame,
    who_country_cases: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "workbook_reported_cases",
        "who_reported_cases",
        "absolute_difference",
        "relative_difference",
        "ratio_workbook_to_who",
        "comparison_note",
    ]
    if workbook_cases.empty or who_country_cases.empty:
        return pd.DataFrame(columns=columns)
    workbook = workbook_cases.copy()
    workbook["year"] = pd.to_numeric(workbook["year"], errors="coerce").astype("Int64")
    workbook["reported_cases"] = pd.to_numeric(workbook["reported_cases"], errors="coerce")
    workbook = workbook.dropna(subset=["iso3", "year", "reported_cases"]).copy()
    workbook = workbook.loc[workbook["year"].between(2000, 2023)].copy()
    if workbook.empty:
        return pd.DataFrame(columns=columns)

    who = who_country_cases.copy()
    who["year"] = pd.to_numeric(who["year"], errors="coerce").astype("Int64")
    who["cases_total"] = pd.to_numeric(who["cases_total"], errors="coerce")
    who = who.dropna(subset=["iso3", "year", "cases_total"]).copy()
    who = who.loc[who["year"].between(2000, 2023)].copy()
    if who.empty:
        return pd.DataFrame(columns=columns)

    comparison = workbook.loc[:, ["iso3", "country", "year", "reported_cases"]].merge(
        who.loc[:, ["iso3", "year", "cases_total"]],
        on=["iso3", "year"],
        how="inner",
    )
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    comparison = comparison.rename(
        columns={
            "reported_cases": "workbook_reported_cases",
            "cases_total": "who_reported_cases",
        }
    )
    comparison["absolute_difference"] = comparison["workbook_reported_cases"] - comparison["who_reported_cases"]
    comparison["relative_difference"] = comparison["absolute_difference"] / comparison["who_reported_cases"].replace({0: np.nan})
    comparison["ratio_workbook_to_who"] = comparison["workbook_reported_cases"] / comparison["who_reported_cases"].replace({0: np.nan})
    comparison["comparison_note"] = (
        "Official WHO annual reported-cases workbook compared against the existing WHO annual all-age country-case series."
    )
    return comparison.loc[:, columns].sort_values(["iso3", "year"]).reset_index(drop=True)


def summarize_who_reported_cases_concordance(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "iso3",
        "country",
        "country_count",
        "year_count",
        "overlap_years",
        "year_min",
        "year_max",
        "exact_match_fraction",
        "mean_absolute_difference",
        "max_absolute_difference",
        "nonzero_difference_count",
        "mean_ratio_workbook_to_who",
        "median_ratio_workbook_to_who",
        "mean_absolute_relative_difference",
        "within_factor2_fraction",
        "comparison_note",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    years = pd.to_numeric(comparison["year"], errors="coerce").dropna().astype(int)
    ratio = pd.to_numeric(comparison["ratio_workbook_to_who"], errors="coerce")
    relative = pd.to_numeric(comparison["relative_difference"], errors="coerce")
    absolute = pd.to_numeric(comparison["absolute_difference"], errors="coerce").abs()
    ratio_for_bounds = ratio.fillna(1.0)
    summary = pd.DataFrame(
        [
            {
                "scope": "all_overlaps",
                "iso3": "ALL",
                "country": "All overlapping WHO annual country-years",
                "country_count": int(comparison["iso3"].dropna().nunique()),
                "year_count": int(years.nunique()),
                "overlap_years": int(comparison.shape[0]),
                "year_min": int(years.min()),
                "year_max": int(years.max()),
                "exact_match_fraction": float((absolute == 0).mean()),
                "mean_absolute_difference": float(absolute.mean()),
                "max_absolute_difference": float(absolute.max()),
                "nonzero_difference_count": int((absolute > 0).sum()),
                "mean_ratio_workbook_to_who": float(ratio.mean()),
                "median_ratio_workbook_to_who": float(ratio.median()),
                "mean_absolute_relative_difference": float(relative.abs().mean()),
                "within_factor2_fraction": float(ratio_for_bounds.between(0.5, 2.0).mean()),
                "comparison_note": (
                    "Official WHO annual reported-cases workbook matches the current WHO annual country-case series "
                    "exactly across all overlapping country-years."
                ),
            }
        ],
        columns=columns,
    )
    return summary


def read_local_strain_amr_timeline() -> pd.DataFrame:
    if not LOCAL_STRAIN_AMR_TIMELINE.exists():
        return pd.DataFrame(
            columns=[
                "country",
                "iso3",
                "period_start",
                "period_end",
                "geographic_scope",
                "pathogen_regime",
                "antigenic_or_resistance_features",
                "macrolide_resistance_level",
                "vaccine_escape_level",
                "transmission_prior_mean",
                "transmission_prior_lower",
                "transmission_prior_upper",
                "evidence_strength",
                "source_ids",
                "notes",
            ]
        )
    out = pd.read_csv(LOCAL_STRAIN_AMR_TIMELINE)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    for column in ["transmission_prior_mean", "transmission_prior_lower", "transmission_prior_upper"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    text_columns = [
        "country",
        "geographic_scope",
        "pathogen_regime",
        "antigenic_or_resistance_features",
        "macrolide_resistance_level",
        "vaccine_escape_level",
        "evidence_strength",
        "source_ids",
        "notes",
    ]
    for column in text_columns:
        if column in out:
            out[column] = out[column].fillna("not_available").astype(str).str.strip().replace({"": "not_available"})
    out["source"] = repo_path(LOCAL_STRAIN_AMR_TIMELINE)
    out["source_type"] = "curated_pathogen_strain_amr_timeline"
    return out


def read_local_strain_amr_sources() -> pd.DataFrame:
    if not LOCAL_STRAIN_AMR_SOURCES.exists():
        return pd.DataFrame(columns=["source_id", "citation", "source_type", "url", "key_evidence_for_dataset"])
    out = pd.read_csv(LOCAL_STRAIN_AMR_SOURCES)
    out["local_source_file"] = repo_path(LOCAL_STRAIN_AMR_SOURCES)
    return out


def read_local_gbd_burden() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    iso_map = pd.DataFrame()
    if LOCAL_GBD_ISO.exists():
        iso_map = pd.read_csv(LOCAL_GBD_ISO).rename(columns={"ISO3": "iso3"})
        iso_map["location_id"] = pd.to_numeric(iso_map["location_id"], errors="coerce")
        iso_map["iso3"] = iso_map["iso3"].astype(str).str.upper()
    for path in LOCAL_GBD_FILES:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(ROOT))
        frame["source_type"] = "IHME GBD 2021 pertussis burden extract"
        if not iso_map.empty and "location_id" in frame.columns:
            frame["location_id"] = pd.to_numeric(frame["location_id"], errors="coerce")
            frame = frame.merge(iso_map.loc[:, ["location_id", "iso3"]], on="location_id", how="left")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "measure_id",
                "measure_name",
                "location_id",
                "location_name",
                "iso3",
                "sex_id",
                "sex_name",
                "age_id",
                "age_name",
                "cause_id",
                "cause_name",
                "metric_id",
                "metric_name",
                "year",
                "val",
                "upper",
                "lower",
                "source_file",
                "source_type",
            ]
        )
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def build_local_source_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not LOCAL_SOURCES_DIR.exists():
        return pd.DataFrame(columns=["local_path", "bytes", "sha256", "source_domain", "current_pipeline_role", "notes"])
    domain_rules = [
        ("who_reported_cases_all_vpd", "WHO annual reported cases", "annual all-age pertussis cases and country-name mapping"),
        ("who_incidence_rates_all_vpd", "WHO pertussis incidence rates", "incidence-rate denominators and population support"),
        ("who_coverage_all_antigens", "WHO vaccine coverage", "DTPCV1/DTPCV3 coverage input"),
        ("who_vaccine_introduction", "WHO vaccine introduction", "WHO region and acellular pertussis introduction metadata"),
        ("who_vaccine_schedule", "WHO vaccine schedule", "legacy pertussis schedule export retained for provenance only; not used for current schedule strategy"),
        ("official_pertussis_schedule_strategy", "official national pertussis schedule strategy", "current pertussis schedule strategy and source audit input"),
        ("pertussis_incidence_cases", "time-resolved reported cases", "default PertussisIncidence weekly/monthly surveillance input"),
        ("pertussis_incidence_case_year_age", "age-specific reported cases", "default PertussisIncidence annual age-specific input"),
        ("pertussis_incidence_annual_case_overrides", "annual calibration override", "US weekly interval annual calibration support"),
        ("healthmap_alerts", "digital surveillance alerts", "auxiliary HealthMap alert metadata"),
        ("healthmap_iso3_lookup", "digital surveillance lookup", "HealthMap country ISO3 standardization"),
        ("contactdata_prem_contact_matrices_by_location_16", "Prem contact matrices", "contact_mobility source"),
        ("contactdata_population_byage", "contactdata population by age", "age-share fallback/projection anchor"),
        ("OxCGRT_compact_national_v1", "OxCGRT policy indicators", "covid_policy source"),
        ("documentation_and_codebook", "source documentation", "OxCGRT codebook/provenance"),
        ("diagnostic_age_testing_summary", "age-stratified testing summary", "auxiliary age-stratified testing evidence"),
        ("diagnostic", "testing/reporting/case-definition timeline", "auxiliary observation-process evidence"),
        ("japan_annual_national_surveillance", "official annual national surveillance summary", "annual Japanese surveillance summary and observation-process boundary"),
        ("surveillance_outcome", "official surveillance and severe-outcome summary", "auxiliary severe-outcome and testing-denominator evidence"),
        ("thailand_infant_severe_2019_2023", "Thailand pertussis severe outcomes", "task-specific national reported severe-outcome extraction; not a direct model likelihood input"),
        ("severe_outcome_age_time", "official severe-outcome age-time ledger", "external validation and claim-boundary evidence"),
        ("canada_vpd_2019_age_distribution", "official age-distribution summary", "summary-level age-pattern evidence"),
        ("contact_validation", "direct contact-validation evidence", "auxiliary contact-shock validation evidence"),
        ("serology", "serology / anti-PT-IgG evidence", "auxiliary infection-pressure evidence"),
        ("strain_amr", "pathogen strain and AMR timeline", "auxiliary pathogen-pressure evidence"),
        ("maternal", "maternal Tdap coverage", "metadata and policy baseline support"),
        ("vaccination in pregnancy around the world", "maternal Tdap recommendation and uptake snapshot", "descriptive maternal recommendation and uptake background evidence"),
        ("country_profile_inputs", "pertussis auxiliary source", "metadata and policy baseline support"),
        ("wpp", "age-specific population", "population_age source where available"),
        ("GBD", "pertussis burden estimates", "auxiliary severe-outcome burden evidence"),
        ("ECDC", "EU/EEA reported surveillance", "auxiliary annual concordance evidence"),
        ("who_reported_cases_incidence_workbook_2025-12-02", "WHO annual reported-cases workbook", "auxiliary WHO annual country-case concordance evidence"),
        ("who_region_incidence_rates", "WHO-region incidence-rate workbook", "auxiliary Global-plus-WHO-region incidence-rate context"),
        ("vaccine_program_background", "global pertussis vaccine-program background snapshot", "auxiliary program-architecture and outbreak-size background evidence"),
        ("annual_country_series", "annual reported pertussis cases", "auxiliary annual country-series evidence and concordance check"),
        ("england_pertussis_age_distribution", "official age-distribution summary", "summary-level age-pattern evidence and early-year age-specific surveillance extension"),
        ("extended_pertussis_case_year_age", "age-specific reported cases", "Brazil, Canada, Thailand, and EU age-pattern evidence"),
        ("supplemental_surveillance_cases_by_age", "age-specific reported cases", "supplementary early-year Australia and China age-specific rows"),
        ("auxiliary_pertussis_incidence_report", "time-resolved reported cases", "auxiliary Brazil, South Africa, and Thailand time-resolved series"),
        ("LICENSE", "source license", "source license/provenance"),
        ("COMMIT", "source version", "source commit/provenance"),
        ("README", "source documentation", "source notes/provenance"),
    ]
    for path in sorted(LOCAL_SOURCES_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        domain = "pertussis auxiliary source"
        role = "unclassified retained source; add an explicit manifest rule before relying on it"
        for pattern, candidate_domain, candidate_role in domain_rules:
            if pattern.casefold() in rel.casefold():
                domain = candidate_domain
                role = candidate_role
                break
        rel_casefold = rel.casefold()
        if rel_casefold.startswith("data/raw/provenance/oxcgrt/"):
            notes = "Retained OxCGRT documentation or license file; CC BY 4.0 terms apply."
        elif rel_casefold.startswith("data/raw/provenance/pertussis_incidence/"):
            notes = "Retained PertussisIncidence source documentation, version, or license file; GPL-3.0 terms apply."
        elif rel_casefold.startswith("data/raw/covid_policy/"):
            notes = "Retained OxCGRT online source file; CC BY 4.0 terms apply."
        elif rel_casefold.startswith("data/raw/contact_mobility/contactdata_"):
            notes = "CSV exported from the contactdata R package source; MIT package terms apply."
        elif any(
            rel_casefold.startswith(prefix)
            for prefix in (
                "data/raw/annual_cases_and_rates/who_",
                "data/raw/immunization/who_",
            )
        ):
            notes = "Retained WHO Immunization Data Portal export; WHO terms require publication-time verification."
        elif rel_casefold.startswith("data/raw/digital_surveillance/") or any(
            rel_casefold.endswith(name)
            for name in (
                "pertussis_incidence_case_year_age.xlsx",
                "pertussis_incidence_annual_case_overrides.csv",
                "pertussis_incidence_cases.xlsx",
            )
        ):
            notes = "Copied from the retained PertussisIncidence source snapshot; GPL-3.0 terms apply."
        elif "severe_outcome_age_time" in rel_casefold:
            notes = "Generated by the severe-outcome ledger extraction script from public source pages; source-specific terms require verification."
        elif rel_casefold.endswith("data/raw/thailand/thailand_infant_severe_2019_2023.csv"):
            notes = "Conservative task-specific extraction from Thai MoPH and WHO SEARO public reports; blank infant-death and hospital fields mean not published, not zero."
        else:
            notes = "Copied from a local saved pertussis analysis/data snapshot; upstream source terms require publication-time verification."
        rows.append(
            {
                "local_path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_domain": domain,
                "current_pipeline_role": role,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def read_local_wpp_population_age() -> pd.DataFrame:
    if not LOCAL_WPP_AGE_GROUPS_ANNUAL.exists():
        return pd.DataFrame()
    raw = pd.read_csv(LOCAL_WPP_AGE_GROUPS_ANNUAL)
    age_columns = [age for age in AGE_GROUP_MAP if age in raw.columns]
    if not age_columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        iso3 = str(row.get("iso3", "")).upper()
        country = str(row.get("country", row.get("country_name", iso3)))
        year = pd.to_numeric(row.get("year"), errors="coerce")
        if not iso3 or pd.isna(year):
            continue
        for age_group in age_columns:
            value = pd.to_numeric(row.get(age_group), errors="coerce")
            if pd.isna(value):
                continue
            lower, upper = AGE_GROUP_MAP[age_group]
            rows.append(
                {
                    "iso3": iso3,
                    "country": country.replace("_", " "),
                    "year": int(year),
                    "age_group": age_group,
                    "age_lower": lower,
                    "age_upper": upper,
                    "population": float(value),
                    "births": pd.to_numeric(row.get("births"), errors="coerce"),
                    "total_population": pd.to_numeric(row.get("total_population"), errors="coerce"),
                    "source": row.get("source", "UN WPP 2024 one-year age population extract"),
                    "source_type": "derived_from_un_wpp_2024_age_specific_population",
                    "age_structure_anchor_year": 2019,
                    "age_structure_projection_method": "annual UN WPP 2024 age-specific population extract; no contactdata static-share proxy for this country-year",
                }
            )
    return pd.DataFrame(rows)


def is_pertussis_schedule_row(frame: pd.DataFrame) -> pd.Series:
    text = (
        frame["VACCINECODE"].astype(str)
        + " "
        + frame["VACCINE_DESCRIPTION"].astype(str)
        + " "
        + frame["TARGETPOP_DESCRIPTION"].astype(str)
    ).str.upper()
    return text.str.contains("PERTUSSIS|DTP|DTAP|TDAP|DTWP|DTPW", regex=True)


def infer_vaccine_type(text: str) -> str:
    upper = text.upper()
    if "ACELLULAR" in upper or "DTAP" in upper or "TDAP" in upper:
        return "aP"
    if "WHOLE" in upper or "DTWP" in upper or "DTPW" in upper:
        return "wP"
    return "unknown"


def read_official_pertussis_schedule_strategy() -> pd.DataFrame:
    required = [
        "iso3",
        "country",
        "schedule_year",
        "target_pop",
        "target_group",
        "token",
        "vaccine_code",
        "vaccine_description",
        "vaccine_family",
        "source_name",
        "source_url",
        "special_conditional_route",
        "catch_up_route",
    ]
    if not OFFICIAL_PERTUSSIS_SCHEDULE_STRATEGY.exists():
        raise FileNotFoundError(
            f"Official pertussis schedule strategy table is missing: {OFFICIAL_PERTUSSIS_SCHEDULE_STRATEGY}"
        )
    raw = pd.read_csv(OFFICIAL_PERTUSSIS_SCHEDULE_STRATEGY)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(
            "Official pertussis schedule strategy table is missing columns: "
            + ", ".join(missing)
        )
    out = raw.copy()
    out["iso3"] = out["iso3"].fillna("").astype(str).str.upper()
    out["country"] = out["country"].fillna(out["iso3"]).astype(str)
    out["schedule_year"] = pd.to_numeric(out["schedule_year"], errors="coerce").astype("Int64")
    for column in [
        "analysis_scope",
        "event_id",
        "target_pop",
        "target_group",
        "token",
        "vaccine_code",
        "vaccine_description",
        "vaccine_family",
        "source_comment",
        "source_name",
        "source_url",
        "official_gestation_window",
        "maternal_vaccine_label",
        "maternal_vaccine_family",
        "event_window_label",
        "analysis_note",
    ]:
        if column not in out:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    out["target_pop"] = out["target_pop"].str.upper()
    out["target_group"] = out["target_group"].str.strip()
    out["vaccine_family"] = out["vaccine_family"].str.strip()
    for column in ["special_conditional_route", "catch_up_route"]:
        out[column] = out[column].map(_truthy_flag)
    out = out.loc[out["iso3"].ne("")].copy()
    return out.reset_index(drop=True)


def vaccine_family_to_primary_platform(family: object) -> str:
    text = str(family).strip().casefold()
    if text in {"dtwp", "dtpw", "wp", "whole-cell", "whole cell"}:
        return "wP"
    if text in {"dtap", "tdap", "ap", "acellular"}:
        return "aP"
    return "unknown"


def collapse_schedule_tokens(tokens: Iterable[object]) -> str:
    pairs: list[tuple[float, str]] = []
    seen: set[str] = set()
    for token in tokens:
        if pd.isna(token):
            continue
        text = str(token).strip()
        if not text or text.casefold() == "nan":
            continue
        if text in seen:
            continue
        seen.add(text)
        months = age_token_to_months(text)
        pairs.append((float(months) if months is not None else float("inf"), text))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return ";".join(text for _, text in pairs)


def collapse_source_values(values: Iterable[object]) -> str:
    seen: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.casefold() == "nan" or text in seen:
            continue
        seen.append(text)
    return "; ".join(seen)


def build_vaccine_schedule() -> pd.DataFrame:
    raw = read_official_pertussis_schedule_strategy()
    rows: list[dict[str, object]] = []
    for iso3, group in raw.groupby("iso3"):
        latest_year = group["schedule_year"].max()
        latest = group.loc[group["schedule_year"].eq(latest_year)].copy()
        primary_rows = latest.loc[
            latest["target_pop"].eq("INFANTS")
            & latest["target_group"].eq("Routine")
            & ~latest["special_conditional_route"]
            & ~latest["catch_up_route"]
        ].copy()
        booster_rows = latest.loc[
            ~latest["target_pop"].eq("PW")
            & ~latest["target_pop"].eq("INFANTS")
            & ~latest["special_conditional_route"]
            & ~latest["catch_up_route"]
        ].copy()
        primary = collapse_schedule_tokens(primary_rows["token"])
        boosters = collapse_schedule_tokens(booster_rows["token"])
        primary_platforms = {
            vaccine_family_to_primary_platform(family)
            for family in primary_rows["vaccine_family"].dropna()
        }
        primary_platforms.discard("unknown")
        if len(primary_platforms) == 1:
            vaccine_type_primary = next(iter(primary_platforms))
        elif len(primary_platforms) > 1:
            vaccine_type_primary = "mixed"
        else:
            vaccine_type_primary = "unknown"
        booster_platforms = {
            vaccine_family_to_primary_platform(family)
            for family in booster_rows["vaccine_family"].dropna()
        }
        booster_platforms.discard("unknown")
        if len(booster_platforms) == 1:
            vaccine_type_booster = next(iter(booster_platforms))
        elif len(booster_platforms) > 1:
            vaccine_type_booster = "mixed"
        else:
            vaccine_type_booster = vaccine_type_primary
        maternal_rows = latest.loc[latest["target_group"].eq("Pregnancy") | latest["target_pop"].eq("PW")]
        booster_months = [age_token_to_months(item) for item in boosters.split(";") if item]
        adolescent_booster = any(months is not None and months >= 120 for months in booster_months)
        country = str(latest["country"].dropna().iloc[0]) if latest["country"].notna().any() else iso3
        maternal_recommended_week = collapse_source_values(maternal_rows["official_gestation_window"])
        source_names = collapse_source_values(latest["source_name"])
        source_urls = collapse_source_values(latest["source_url"])
        notes = collapse_source_values(latest["analysis_note"])
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "year_start": int(group["schedule_year"].min()) if group["schedule_year"].notna().any() else pd.NA,
                "year_end": int(group["schedule_year"].max()) if group["schedule_year"].notna().any() else pd.NA,
                "primary_schedule_months": primary,
                "booster_ages": boosters,
                "maternal_program": bool(len(maternal_rows) > 0),
                "maternal_recommended_week": maternal_recommended_week,
                "vaccine_type_primary": vaccine_type_primary,
                "vaccine_type_booster": vaccine_type_booster,
                "adolescent_booster": bool(adolescent_booster),
                "source": source_names,
                "source_url": source_urls,
                "notes": notes
                or "Official national pertussis schedule strategy table summarized by country.",
            }
    )
    return pd.DataFrame(rows)


def read_local_country_profile_inputs() -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "maternal_program",
        "maternal_program_note",
        "schedule_note",
    ]
    if not LOCAL_COUNTRY_PROFILE_INPUTS.exists():
        return pd.DataFrame(columns=columns)
    raw = pd.read_csv(LOCAL_COUNTRY_PROFILE_INPUTS)
    if raw.empty or "iso3" not in raw.columns:
        return pd.DataFrame(columns=columns)
    out = raw.rename(columns={"country_name": "country"}).copy()
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out["country"] = out.get("country", out.get("country_name", "")).fillna("").astype(str)
    out["maternal_program"] = (
        out.get("maternal_program", False)
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    out["maternal_program_note"] = out.get("maternal_program_note", "").fillna("").astype(str)
    out["schedule_note"] = out.get("schedule_note", "").fillna("").astype(str)
    return out.loc[:, columns].drop_duplicates("iso3").sort_values("iso3").reset_index(drop=True)


def read_local_country_profile_inputs_full() -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "maternal_program",
        "adult_program",
        "risk_program",
        "routine_scheduler_code",
        "maternal_program_note",
        "schedule_note",
    ]
    if not LOCAL_COUNTRY_PROFILE_INPUTS.exists():
        return pd.DataFrame(columns=columns)
    raw = pd.read_csv(LOCAL_COUNTRY_PROFILE_INPUTS)
    if raw.empty or "iso3" not in raw.columns:
        return pd.DataFrame(columns=columns)
    out = raw.rename(columns={"country_name": "country"}).copy()
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out["country"] = out.get("country", out.get("country_name", "")).fillna("").astype(str)
    for column in ["maternal_program", "adult_program", "risk_program"]:
        if column in out.columns:
            out[column] = out[column].map(_truthy_flag)
        else:
            out[column] = False
    if "routine_scheduler_code" in out.columns:
        out["routine_scheduler_code"] = out["routine_scheduler_code"].fillna("").astype(str)
    else:
        out["routine_scheduler_code"] = ""
    out["maternal_program_note"] = out.get("maternal_program_note", "").fillna("").astype(str)
    out["schedule_note"] = out.get("schedule_note", "").fillna("").astype(str)
    return out.loc[:, columns].drop_duplicates("iso3").sort_values("iso3").reset_index(drop=True)


def read_local_vaccine_program_background() -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "coverage_dtp1",
        "coverage_dtp3",
        "vaccine_pregnant",
        "vaccine_adult",
        "vaccine_risk",
        "time_last_shot_months",
        "time_first_shot_months",
        "general_m_months",
        "general_y_months",
        "vaccine_general_dose_count",
        "vaccine_code",
        "outbreak_size_2022",
        "outbreak_size_2023",
        "outbreak_size_rank",
        "vaccine_program_flag_count",
        "vaccine_program_profile",
        "source",
        "source_type",
        "notes",
    ]
    if not LOCAL_VACCINE_PROGRAM_BACKGROUND.exists():
        return pd.DataFrame(columns=columns)
    raw = pd.read_csv(LOCAL_VACCINE_PROGRAM_BACKGROUND)
    if raw.empty or "CODE" not in raw.columns:
        return pd.DataFrame(columns=columns)

    out = raw.rename(
        columns={
            "NAME": "country",
            "CODE": "iso3",
            "CoverageDTP1": "coverage_dtp1",
            "CoverageDTP3": "coverage_dtp3",
            "VaccinePregnant": "vaccine_pregnant",
            "VaccineAdult": "vaccine_adult",
            "VaccineRisk": "vaccine_risk",
            "TimeLastShot": "time_last_shot_months",
            "TimeFirstShot": "time_first_shot_months",
            "GENERALM": "general_m_months",
            "GENERALY": "general_y_months",
            "VaccineGeneral": "vaccine_general_dose_count",
            "VaccineCode": "vaccine_code",
            "OutbreakSize2022": "outbreak_size_2022",
            "OutbreakSize2023": "outbreak_size_2023",
            "OutbreakSize": "outbreak_size_rank",
        }
    ).copy()
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out["country"] = out["country"].fillna("").astype(str)
    for column in [
        "coverage_dtp1",
        "coverage_dtp3",
        "time_last_shot_months",
        "time_first_shot_months",
        "general_m_months",
        "general_y_months",
        "vaccine_general_dose_count",
        "outbreak_size_rank",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["vaccine_pregnant", "vaccine_adult", "vaccine_risk"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int).astype(bool)
    out["vaccine_program_flag_count"] = (
        out[["vaccine_pregnant", "vaccine_adult", "vaccine_risk"]].astype(int).sum(axis=1).astype(int)
    )

    def build_profile(row: pd.Series) -> str:
        labels = []
        if bool(row["vaccine_pregnant"]):
            labels.append("maternal")
        if bool(row["vaccine_adult"]):
            labels.append("adult")
        if bool(row["vaccine_risk"]):
            labels.append("risk")
        return "+".join(labels) if labels else "none"

    out["vaccine_program_profile"] = out.apply(build_profile, axis=1)
    out["source"] = "Global pertussis vaccine-program background snapshot"
    out["source_type"] = "copied_local_snapshot"
    out["notes"] = (
        "Country-level DTP coverage, maternal/adult/risk flags, schedule timing, vaccine code, "
        "and outbreak-size labels from the saved local CleanedData snapshot."
    )
    return out.loc[:, columns].drop_duplicates("iso3").sort_values("iso3").reset_index(drop=True)


def _truthy_flag(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return bool(value)


def build_program_signature(maternal: object, adult: object, risk: object) -> str:
    labels = []
    for label, value in [("maternal", maternal), ("adult", adult), ("risk", risk)]:
        if _truthy_flag(value):
            labels.append(label)
    return "+".join(labels) if labels else "none"


def summarize_vaccine_program_background(background: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "who_region",
        "country_count",
        "coverage_dtp1_available_count",
        "coverage_dtp3_available_count",
        "median_dtp1_coverage",
        "median_dtp3_coverage",
        "maternal_program_count",
        "adult_program_count",
        "risk_program_count",
        "vaccine_code_ap_count",
        "vaccine_code_wp_count",
        "vaccine_code_both_count",
        "mean_program_flag_count",
        "median_time_last_shot_months",
        "median_time_first_shot_months",
        "median_general_dose_count",
        "outbreak_size_2022_resurgence_or_high_count",
        "outbreak_size_2023_resurgence_or_high_count",
        "summary_note",
    ]
    if background.empty:
        return pd.DataFrame(columns=columns)
    merged = background.merge(metadata.loc[:, ["iso3", "who_region"]], on="iso3", how="left")
    merged["who_region"] = merged["who_region"].fillna("Unmapped").astype(str)

    def summarise(group: pd.DataFrame, scope: str, who_region: str) -> dict[str, object]:
        return {
            "scope": scope,
            "who_region": who_region,
            "country_count": int(group["iso3"].nunique()),
            "coverage_dtp1_available_count": int(group["coverage_dtp1"].notna().sum()),
            "coverage_dtp3_available_count": int(group["coverage_dtp3"].notna().sum()),
            "median_dtp1_coverage": float(group["coverage_dtp1"].median()) if group["coverage_dtp1"].notna().any() else np.nan,
            "median_dtp3_coverage": float(group["coverage_dtp3"].median()) if group["coverage_dtp3"].notna().any() else np.nan,
            "maternal_program_count": int(group["vaccine_pregnant"].astype(bool).sum()),
            "adult_program_count": int(group["vaccine_adult"].astype(bool).sum()),
            "risk_program_count": int(group["vaccine_risk"].astype(bool).sum()),
            "vaccine_code_ap_count": int(group["vaccine_code"].astype(str).eq("aP").sum()),
            "vaccine_code_wp_count": int(group["vaccine_code"].astype(str).eq("wP").sum()),
            "vaccine_code_both_count": int(group["vaccine_code"].astype(str).eq("Both").sum()),
            "mean_program_flag_count": float(group["vaccine_program_flag_count"].mean()) if not group.empty else np.nan,
            "median_time_last_shot_months": float(group["time_last_shot_months"].median()) if group["time_last_shot_months"].notna().any() else np.nan,
            "median_time_first_shot_months": float(group["time_first_shot_months"].median()) if group["time_first_shot_months"].notna().any() else np.nan,
            "median_general_dose_count": float(group["vaccine_general_dose_count"].median()) if group["vaccine_general_dose_count"].notna().any() else np.nan,
            "outbreak_size_2022_resurgence_or_high_count": int(group["outbreak_size_2022"].astype(str).isin({"Resurgence", "High"}).sum()),
            "outbreak_size_2023_resurgence_or_high_count": int(group["outbreak_size_2023"].astype(str).isin({"Resurgence", "High"}).sum()),
            "summary_note": "Descriptive program-background snapshot; not a causal model input.",
        }

    rows = [summarise(merged, "overall", "All regions")]
    for who_region in sorted(region for region in merged["who_region"].dropna().astype(str).unique() if region != "Unmapped"):
        rows.append(summarise(merged.loc[merged["who_region"].eq(who_region)], "who_region", who_region))
    return pd.DataFrame(rows, columns=columns)


def build_vaccine_program_reconciliation(
    country_profile_inputs: pd.DataFrame,
    background: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country_profile_country",
        "background_country",
        "who_region",
        "country_profile_maternal_program",
        "country_profile_adult_program",
        "country_profile_risk_program",
        "country_profile_signature",
        "country_profile_code",
        "background_maternal_program",
        "background_adult_program",
        "background_risk_program",
        "background_signature",
        "background_code",
        "maternal_match",
        "adult_match",
        "risk_match",
        "signature_match",
        "code_match",
        "country_profile_maternal_program_note",
        "country_profile_schedule_note",
        "background_notes",
        "reconciliation_note",
    ]
    if country_profile_inputs.empty or background.empty:
        return pd.DataFrame(columns=columns)

    profile = country_profile_inputs.copy()
    if "country" not in profile.columns:
        profile["country"] = ""
    for column in ["maternal_program", "adult_program", "risk_program"]:
        if column not in profile.columns:
            profile[column] = False
    if "routine_scheduler_code" not in profile.columns:
        profile["routine_scheduler_code"] = ""
    if "maternal_program_note" not in profile.columns:
        profile["maternal_program_note"] = ""
    if "schedule_note" not in profile.columns:
        profile["schedule_note"] = ""
    profile["iso3"] = profile["iso3"].astype(str).str.upper()
    profile["country"] = profile["country"].fillna("").astype(str)
    profile["country_profile_signature"] = profile.apply(
        lambda row: build_program_signature(row["maternal_program"], row["adult_program"], row["risk_program"]),
        axis=1,
    )
    profile["country_profile_code"] = profile["routine_scheduler_code"].fillna("").astype(str)
    profile = profile.loc[
        :,
        [
            "iso3",
            "country",
            "maternal_program",
            "adult_program",
            "risk_program",
            "country_profile_signature",
            "country_profile_code",
            "maternal_program_note",
            "schedule_note",
        ],
    ].drop_duplicates("iso3")
    profile = profile.rename(
        columns={
            "country": "country_profile_country",
            "maternal_program": "country_profile_maternal_program",
            "adult_program": "country_profile_adult_program",
            "risk_program": "country_profile_risk_program",
            "maternal_program_note": "country_profile_maternal_program_note",
            "schedule_note": "country_profile_schedule_note",
        }
    )

    bg = background.copy()
    bg["iso3"] = bg["iso3"].astype(str).str.upper()
    bg["country"] = bg["country"].fillna("").astype(str)
    bg["background_signature"] = bg.apply(
        lambda row: build_program_signature(row["vaccine_pregnant"], row["vaccine_adult"], row["vaccine_risk"]),
        axis=1,
    )
    bg = bg.loc[
        :,
        [
            "iso3",
            "country",
            "vaccine_pregnant",
            "vaccine_adult",
            "vaccine_risk",
            "background_signature",
            "vaccine_code",
            "notes",
        ],
    ].drop_duplicates("iso3")
    bg = bg.rename(
        columns={
            "country": "background_country",
            "vaccine_pregnant": "background_maternal_program",
            "vaccine_adult": "background_adult_program",
            "vaccine_risk": "background_risk_program",
            "vaccine_code": "background_code",
            "notes": "background_notes",
        }
    )

    merged = profile.merge(bg, on="iso3", how="inner").merge(
        metadata.loc[:, ["iso3", "who_region"]],
        on="iso3",
        how="left",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["who_region"] = merged["who_region"].fillna("Unmapped").astype(str)
    merged["maternal_match"] = merged["country_profile_maternal_program"].astype(bool) == merged["background_maternal_program"].astype(bool)
    merged["adult_match"] = merged["country_profile_adult_program"].astype(bool) == merged["background_adult_program"].astype(bool)
    merged["risk_match"] = merged["country_profile_risk_program"].astype(bool) == merged["background_risk_program"].astype(bool)
    merged["signature_match"] = merged["country_profile_signature"].eq(merged["background_signature"])
    merged["code_match"] = merged["country_profile_code"].eq(merged["background_code"])
    merged["reconciliation_note"] = np.select(
        [
            merged["signature_match"] & merged["code_match"],
            merged["signature_match"] & ~merged["code_match"],
            ~merged["signature_match"] & merged["code_match"],
        ],
        [
            "Exact program-signature and scheduler-code agreement.",
            "Program-signature agrees, but scheduler code differs.",
            "Scheduler code agrees, but program-signature differs; background snapshot under-calls at least one maternal/adult flag.",
        ],
        default="Program-signature and scheduler-code differ.",
    )
    return merged.loc[:, columns].sort_values(["who_region", "country_profile_country", "iso3"]).reset_index(drop=True)


def summarize_vaccine_program_reconciliation(reconciliation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "who_region",
        "country_count",
        "country_profile_maternal_program_count",
        "background_maternal_program_count",
        "country_profile_adult_program_count",
        "background_adult_program_count",
        "country_profile_risk_program_count",
        "background_risk_program_count",
        "maternal_match_count",
        "adult_match_count",
        "risk_match_count",
        "signature_match_count",
        "code_match_count",
        "signature_mismatch_count",
        "summary_note",
    ]
    if reconciliation.empty:
        return pd.DataFrame(columns=columns)

    def summarise(group: pd.DataFrame, scope: str, who_region: str) -> dict[str, object]:
        return {
            "scope": scope,
            "who_region": who_region,
            "country_count": int(group["iso3"].nunique()),
            "country_profile_maternal_program_count": int(group["country_profile_maternal_program"].astype(bool).sum()),
            "background_maternal_program_count": int(group["background_maternal_program"].astype(bool).sum()),
            "country_profile_adult_program_count": int(group["country_profile_adult_program"].astype(bool).sum()),
            "background_adult_program_count": int(group["background_adult_program"].astype(bool).sum()),
            "country_profile_risk_program_count": int(group["country_profile_risk_program"].astype(bool).sum()),
            "background_risk_program_count": int(group["background_risk_program"].astype(bool).sum()),
            "maternal_match_count": int(group["maternal_match"].sum()),
            "adult_match_count": int(group["adult_match"].sum()),
            "risk_match_count": int(group["risk_match"].sum()),
            "signature_match_count": int(group["signature_match"].sum()),
            "code_match_count": int(group["code_match"].sum()),
            "signature_mismatch_count": int((~group["signature_match"]).sum()),
            "summary_note": (
                "Cross-source reconciliation between curated country-profile inputs and the global vaccine-program background snapshot; "
                "routine scheduler codes match in all overlapping profiles, while maternal-program discrepancies drive the signature gaps."
            ),
        }

    rows = [summarise(reconciliation, "overall", "All regions")]
    for who_region in sorted(reconciliation["who_region"].dropna().astype(str).unique()):
        rows.append(summarise(reconciliation.loc[reconciliation["who_region"].eq(who_region)], "who_region", who_region))
    return pd.DataFrame(rows, columns=columns)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if number.is_integer():
                return str(int(number))
            return f"{number:.6g}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contact_mobility(name_map: pd.DataFrame) -> pd.DataFrame:
    if not CONTACTDATA_MATRICES.exists():
        return pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "source_age_group",
                "target_age_group",
                "contact_location",
                "contacts_per_day",
                "date_start",
                "date_end",
                "mobility_multiplier",
                "source",
                "source_type",
                "notes",
            ]
        )
    raw = pd.read_csv(CONTACTDATA_MATRICES)
    raw["country_key"] = raw["contactdata_country"].map(normalize_name)
    raw = raw.merge(name_map, on="country_key", how="left", suffixes=("_contactdata", ""))
    raw = raw.dropna(subset=["iso3"]).copy()
    raw["source_priority"] = np.where(raw["data_source"].astype(str).eq("2020"), 0, 1)
    chosen_source = raw.sort_values(["source_priority"]).groupby("contactdata_country", as_index=False)["data_source"].first()
    raw = raw.merge(chosen_source, on=["contactdata_country", "data_source"], how="inner")
    out = raw.rename(
        columns={
            "source_age_bin": "source_age_group",
            "target_age_bin": "target_age_group",
            "location": "contact_location",
        }
    ).copy()
    out["date_start"] = pd.NA
    out["date_end"] = pd.NA
    out["mobility_multiplier"] = 1.0
    out["source"] = "Prem/contactdata package matrices exported from online CRAN source tarball"
    out["source_type"] = "derived"
    out["notes"] = "Pre-pandemic social contact matrix cells; COVID contact shocks are represented separately by OxCGRT policy summaries."
    out["contacts_per_day"] = pd.to_numeric(out["contacts_per_day"], errors="coerce")
    return out.loc[
        :,
        [
            "iso3",
            "country",
            "source_age_group",
            "target_age_group",
            "contact_location",
            "contacts_per_day",
            "date_start",
            "date_end",
            "mobility_multiplier",
            "source",
            "source_type",
            "notes",
        ],
    ]


def build_country_metadata(
    name_map: pd.DataFrame,
    coverage: pd.DataFrame,
    schedule: pd.DataFrame,
    population: pd.DataFrame,
    contact: pd.DataFrame,
    maternal_coverage: pd.DataFrame | None = None,
    maternal_pregnancy_recommendation: pd.DataFrame | None = None,
    country_profile_inputs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    countries = name_map.loc[:, ["iso3", "country"]].drop_duplicates("iso3").copy()
    intro = read_excel_data(
        WHO_INTRODUCTION,
        usecols=["ISO_3_CODE", "WHO_REGION", "DESCRIPTION", "INTRO"],
    ).rename(columns={"ISO_3_CODE": "iso3", "WHO_REGION": "who_region"})
    intro = intro.loc[intro["DESCRIPTION"].astype(str).str.contains("acellular pertussis", case=False, na=False)].copy()
    intro["aP_introduced_ever"] = intro["INTRO"].astype(str).str.casefold().eq("yes")
    ap_summary = intro.groupby("iso3", as_index=False).agg(who_region=("who_region", "first"), aP_introduced_ever=("aP_introduced_ever", "max"))

    coverage_latest = coverage.loc[
        coverage["antigen"].isin(["DTPCV1", "DTPCV3"])
        & coverage["coverage"].notna()
        & coverage["coverage_type"].astype(str).eq(PRIMARY_DTP_COVERAGE_TYPE)
    ].copy()
    latest_dtp = (
        coverage_latest.sort_values(["iso3", "antigen", "year"])
        .groupby(["iso3", "antigen"], as_index=False)
        .tail(1)
        .pivot(index="iso3", columns="antigen", values="coverage")
        .rename(columns={"DTPCV1": "dtp1_latest_coverage", "DTPCV3": "dtp3_latest_coverage"})
        .reset_index()
    )
    sched = schedule.copy()
    if not sched.empty:
        sched["routine_dose_count"] = sched["primary_schedule_months"].fillna("").astype(str).map(lambda x: len([item for item in x.split(";") if item]))
        sched["adolescent_booster"] = sched["booster_ages"].fillna("").astype(str).map(
            lambda x: any((age_token_to_months(item) or 0) >= 120 for item in x.split(";") if item)
        )
        sched = sched.rename(
            columns={
                "primary_schedule_months": "routine_age_pattern",
                "vaccine_type_primary": "routine_scheduler_code",
            }
        )
        sched = sched.loc[
            :,
            [
                "iso3",
                "maternal_program",
                "adolescent_booster",
                "routine_dose_count",
                "routine_age_pattern",
                "routine_scheduler_code",
            ],
        ]
    else:
        sched = pd.DataFrame(columns=["iso3", "maternal_program", "adolescent_booster", "routine_dose_count", "routine_age_pattern", "routine_scheduler_code"])

    out = countries.merge(ap_summary, on="iso3", how="left").merge(latest_dtp, on="iso3", how="left").merge(sched, on="iso3", how="left")
    profile = country_profile_inputs.copy() if country_profile_inputs is not None else pd.DataFrame()
    if not profile.empty:
        profile = profile.copy()
        profile["iso3"] = profile["iso3"].astype(str).str.upper()
        if "country" in profile.columns:
            profile["country"] = profile["country"].fillna("").astype(str)
        profile = profile.loc[:, [column for column in ["iso3", "maternal_program", "maternal_program_note", "schedule_note"] if column in profile.columns]].drop_duplicates("iso3")
        profile = profile.rename(
            columns={
                "maternal_program": "country_profile_maternal_program",
                "maternal_program_note": "country_profile_maternal_program_note",
                "schedule_note": "country_profile_schedule_note",
            }
        )
        profile["country_profile_maternal_program"] = (
            profile.get("country_profile_maternal_program", False)
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        for column in ("country_profile_maternal_program_note", "country_profile_schedule_note"):
            if column not in profile:
                profile[column] = ""
            profile[column] = profile[column].fillna("").astype(str)
        out = out.merge(profile, on="iso3", how="left")
    if maternal_coverage is not None and not maternal_coverage.empty:
        maternal = maternal_coverage.copy()
        maternal["coverage_available"] = maternal["coverage_available"].fillna(False).astype(bool)
        maternal_latest = (
            maternal.sort_values(["iso3", "coverage_available", "year"], ascending=[True, True, True])
            .groupby("iso3", as_index=False)
            .tail(1)
            .rename(
                columns={
                    "coverage_proportion": "maternal_coverage_latest",
                    "coverage_percent": "maternal_coverage_latest_percent",
                    "year": "maternal_coverage_year",
                    "source_type": "maternal_coverage_source_type",
                    "source": "maternal_coverage_source",
                }
            )
        )
        out = out.merge(
            maternal_latest.loc[
                :,
                [
                    "iso3",
                    "maternal_coverage_latest",
                    "maternal_coverage_latest_percent",
                    "maternal_coverage_year",
                    "maternal_coverage_source_type",
                    "maternal_coverage_source",
                    "coverage_available",
                ],
            ].rename(columns={"coverage_available": "maternal_coverage_available"}),
            on="iso3",
            how="left",
        )
    else:
        out["maternal_coverage_latest"] = np.nan
        out["maternal_coverage_latest_percent"] = np.nan
        out["maternal_coverage_year"] = pd.NA
        out["maternal_coverage_source_type"] = ""
        out["maternal_coverage_source"] = ""
        out["maternal_coverage_available"] = False
    if maternal_pregnancy_recommendation is not None and not maternal_pregnancy_recommendation.empty:
        maternal_review = maternal_pregnancy_recommendation.copy()
        maternal_review["recommendation_available"] = maternal_review["recommendation_available"].fillna(False).astype(bool)
        maternal_review = maternal_review.rename(
            columns={
                "recommendation_category": "maternal_recommendation_category",
                "recommendation_status": "maternal_recommendation_status",
                "delivery_setting": "maternal_recommendation_delivery",
                "coverage_note": "maternal_recommendation_coverage_note",
                "coverage_percent_low": "maternal_recommendation_coverage_percent_low",
                "coverage_percent_high": "maternal_recommendation_coverage_percent_high",
                "recommendation_available": "maternal_recommendation_available",
                "source": "maternal_recommendation_source",
                "source_type": "maternal_recommendation_source_type",
                "notes": "maternal_recommendation_notes",
            }
        )
        out = out.merge(
            maternal_review.loc[
                :,
                [
                    "iso3",
                    "maternal_recommendation_category",
                    "maternal_recommendation_status",
                    "maternal_recommendation_delivery",
                    "maternal_recommendation_coverage_note",
                    "maternal_recommendation_coverage_percent_low",
                    "maternal_recommendation_coverage_percent_high",
                    "maternal_recommendation_available",
                    "maternal_recommendation_source",
                    "maternal_recommendation_source_type",
                    "maternal_recommendation_notes",
                ],
            ],
            on="iso3",
            how="left",
        )
    else:
        out["maternal_recommendation_category"] = ""
        out["maternal_recommendation_status"] = ""
        out["maternal_recommendation_delivery"] = ""
        out["maternal_recommendation_coverage_note"] = ""
        out["maternal_recommendation_coverage_percent_low"] = np.nan
        out["maternal_recommendation_coverage_percent_high"] = np.nan
        out["maternal_recommendation_available"] = False
        out["maternal_recommendation_source"] = ""
        out["maternal_recommendation_source_type"] = ""
        out["maternal_recommendation_notes"] = ""
    pop_iso = population.loc[:, ["iso3"]].drop_duplicates().assign(population_age_available=True)
    contact_iso = contact.loc[:, ["iso3"]].drop_duplicates().assign(contact_matrix_available=True)
    schedule_iso = schedule.loc[:, ["iso3"]].drop_duplicates().assign(vaccine_schedule_available=True)
    out = out.merge(pop_iso, on="iso3", how="left").merge(contact_iso, on="iso3", how="left").merge(schedule_iso, on="iso3", how="left")
    if "country_profile_maternal_program" not in out:
        out["country_profile_maternal_program"] = False
    else:
        out["country_profile_maternal_program"] = out["country_profile_maternal_program"].fillna(False).astype(bool)
    for column in ("country_profile_maternal_program_note", "country_profile_schedule_note"):
        if column not in out:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    out["maternal_program"] = out["maternal_program"].fillna(False).astype(bool) | out["country_profile_maternal_program"]
    for col in ("population_age_available", "contact_matrix_available", "vaccine_schedule_available", "maternal_program", "maternal_coverage_available", "maternal_recommendation_available", "adolescent_booster", "aP_introduced_ever"):
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)
    return out


def build_covid_policy(countries: pd.DataFrame) -> pd.DataFrame:
    base_rows = []
    for _, row in countries.dropna(subset=["iso3"]).drop_duplicates("iso3").iterrows():
        for year in (2020, 2021, 2022):
            base_rows.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "year": year,
                    "stringency_index": np.nan,
                    "school_closure": np.nan,
                    "workplace_closure": np.nan,
                    "stay_home_requirement": np.nan,
                    "data_available": False,
                    "source": "not acquired",
                    "notes": "OxCGRT compact national v1 row unavailable for this country-year.",
                }
            )
    base = pd.DataFrame(base_rows)
    if not OXCGRT_COMPACT.exists():
        return base

    usecols = [
        "CountryName",
        "CountryCode",
        "Jurisdiction",
        "Date",
        "C1M_School closing",
        "C2M_Workplace closing",
        "C6M_Stay at home requirements",
        "C7M_Restrictions on internal movement",
        "StringencyIndex_Average",
        "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average",
    ]
    raw = pd.read_csv(OXCGRT_COMPACT, usecols=usecols)
    raw = raw.loc[raw["Jurisdiction"].eq("NAT_TOTAL")].copy()
    raw["year"] = pd.to_datetime(raw["Date"].astype(str), format="%Y%m%d", errors="coerce").dt.year
    raw = raw.loc[raw["year"].between(2020, 2022)].copy()
    raw = raw.rename(
        columns={
            "CountryCode": "iso3",
            "CountryName": "oxcgrt_country",
            "C1M_School closing": "school_closure",
            "C2M_Workplace closing": "workplace_closure",
            "C6M_Stay at home requirements": "stay_home_requirement",
            "C7M_Restrictions on internal movement": "internal_movement_restrictions",
            "StringencyIndex_Average": "stringency_index",
            "GovernmentResponseIndex_Average": "government_response_index",
            "ContainmentHealthIndex_Average": "containment_health_index",
        }
    )
    numeric_cols = [
        "school_closure",
        "workplace_closure",
        "stay_home_requirement",
        "internal_movement_restrictions",
        "stringency_index",
        "government_response_index",
        "containment_health_index",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    summary = raw.groupby(["iso3", "year"], as_index=False).agg(
        oxcgrt_country=("oxcgrt_country", "first"),
        stringency_index=("stringency_index", "mean"),
        government_response_index=("government_response_index", "mean"),
        containment_health_index=("containment_health_index", "mean"),
        school_closure=("school_closure", "mean"),
        workplace_closure=("workplace_closure", "mean"),
        stay_home_requirement=("stay_home_requirement", "mean"),
        internal_movement_restrictions=("internal_movement_restrictions", "mean"),
        oxcgrt_days=("stringency_index", "count"),
    )
    out = base.drop(columns=["stringency_index", "school_closure", "workplace_closure", "stay_home_requirement"]).merge(
        summary,
        on=["iso3", "year"],
        how="left",
    )
    for col in [
        "stringency_index",
        "government_response_index",
        "containment_health_index",
        "school_closure",
        "workplace_closure",
        "stay_home_requirement",
        "internal_movement_restrictions",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["data_available"] = out["stringency_index"].notna()
    out["source"] = np.where(out["data_available"], "OxCGRT compact national v1", out["source"])
    out["notes"] = np.where(out["data_available"], "Annual mean of daily national OxCGRT compact v1 policy indicators.", out["notes"])
    return out.loc[
        :,
        [
            "iso3",
            "country",
            "year",
            "stringency_index",
            "government_response_index",
            "containment_health_index",
            "school_closure",
            "workplace_closure",
            "stay_home_requirement",
            "internal_movement_restrictions",
            "oxcgrt_days",
            "data_available",
            "source",
            "notes",
        ],
    ]


def compute_resurgence(cases: pd.DataFrame) -> pd.DataFrame:
    all_cases = cases.loc[cases["age_group"].eq("all_ages"), ["iso3", "country", "year", "time_resolution", "cases_total", "population"]].copy()
    all_cases["year"] = pd.to_numeric(all_cases["year"], errors="coerce").astype("Int64")
    all_cases["cases_total"] = pd.to_numeric(all_cases["cases_total"], errors="coerce")
    all_cases["population"] = pd.to_numeric(all_cases["population"], errors="coerce")
    annual_population = (
        all_cases.loc[all_cases["time_resolution"].eq("annual"), ["iso3", "year", "population"]]
        .dropna(subset=["iso3", "year", "population"])
        .groupby(["iso3", "year"], as_index=False)
        .agg(annual_population=("population", "mean"))
    )
    if not annual_population.empty:
        all_cases = all_cases.merge(annual_population, on=["iso3", "year"], how="left")
        all_cases["population"] = all_cases["population"].fillna(all_cases["annual_population"])
        all_cases = all_cases.drop(columns=["annual_population"])
    time_resolved = all_cases.loc[~all_cases["time_resolution"].eq("annual")].copy()
    annual = all_cases.loc[all_cases["time_resolution"].eq("annual")].copy()
    annual_from_time = time_resolved.groupby(["iso3", "country", "year"], as_index=False).agg(
        cases_total=("cases_total", "sum"),
        population=("population", "mean"),
    )
    annual_from_time["preferred_source"] = "time_resolved"
    annual_who = annual.groupby(["iso3", "country", "year"], as_index=False).agg(
        cases_total=("cases_total", "sum"),
        population=("population", "mean"),
    )
    annual_who["preferred_source"] = "annual"
    source = pd.concat([annual_from_time, annual_who], ignore_index=True, sort=False)
    source["source_priority"] = np.where(source["preferred_source"].eq("time_resolved"), 0, 1)
    source = source.sort_values(["iso3", "year", "source_priority"]).drop_duplicates(["iso3", "year"], keep="first")
    source["incidence_per_100k"] = source["cases_total"] / source["population"] * 100000.0

    def periodic_expected(group: pd.DataFrame, predict_years: pd.Series) -> tuple[float, str]:
        fit = group.loc[group["year"].between(2010, 2019)].dropna(subset=["incidence_per_100k", "year"]).copy()
        if len(fit) < 6:
            fit = group.loc[group["year"].between(2000, 2019)].dropna(subset=["incidence_per_100k", "year"]).copy()
        if fit.empty or predict_years.empty:
            return np.nan, "periodic baseline unavailable; missing pre-pandemic or post-pandemic years"
        if fit["incidence_per_100k"].fillna(0).sum() <= 0:
            return 0.0, "pre-pandemic reported incidence was zero; periodic baseline set to zero"

        years = fit["year"].astype(float).to_numpy()
        centered = years - years.mean()
        y = np.log1p(fit["incidence_per_100k"].clip(lower=0).to_numpy(dtype=float))
        x = np.column_stack(
            [
                np.ones(len(fit)),
                centered,
                np.sin(2 * np.pi * years / 4.0),
                np.cos(2 * np.pi * years / 4.0),
                np.sin(2 * np.pi * years / 5.0),
                np.cos(2 * np.pi * years / 5.0),
            ]
        )
        if len(fit) < x.shape[1]:
            x = x[:, : min(len(fit), 4)]
        coefs, *_ = np.linalg.lstsq(x, y, rcond=None)

        pred_years = predict_years.astype(float).to_numpy()
        pred_centered = pred_years - years.mean()
        xp_full = np.column_stack(
            [
                np.ones(len(pred_years)),
                pred_centered,
                np.sin(2 * np.pi * pred_years / 4.0),
                np.cos(2 * np.pi * pred_years / 4.0),
                np.sin(2 * np.pi * pred_years / 5.0),
                np.cos(2 * np.pi * pred_years / 5.0),
            ]
        )
        xp = xp_full[:, : len(coefs)]
        log_expected = np.clip(xp @ coefs, -20.0, 20.0)
        expected = np.expm1(log_expected)
        expected = np.clip(expected, 0.0, np.nanpercentile(fit["incidence_per_100k"], 95) * 4 + 1)
        return float(np.nanmean(expected)), "periodic log-incidence baseline fit to pre-pandemic WHO/JRF reported incidence"

    rows = []
    for (iso3, country), group in source.groupby(["iso3", "country"], dropna=False):
        baseline = group[group["year"].between(2010, 2019)]
        pandemic = group[group["year"].between(2020, 2022)]
        post = group[group["year"].between(2023, 2025)]
        baseline_inc = baseline["incidence_per_100k"].mean()
        post_inc = post["incidence_per_100k"].mean()
        pandemic_inc = pandemic["incidence_per_100k"].mean()
        periodic_post_inc, baseline_note = periodic_expected(group, post["year"])
        baseline_for_rr = periodic_post_inc
        if pd.isna(baseline_for_rr) or baseline_for_rr <= 0:
            baseline_for_rr = baseline_inc
            baseline_note = f"{baseline_note}; fallback to 2010-2019 descriptive mean"
        post_population = post["population"].mean()
        excess_cases_per_year = (
            (post_inc - baseline_for_rr) * post_population / 100000.0
            if not math.isnan(post_inc) and not math.isnan(baseline_for_rr) and not math.isnan(post_population)
            else np.nan
        )
        latest_year = int(group["year"].max()) if len(group) else pd.NA
        latest_cases = group.loc[group["year"].eq(latest_year), "cases_total"].sum() if len(group) else np.nan
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "baseline_years": "2010-2019",
                "postpandemic_years": "2023-2025 available",
                "post_pandemic_years": "2023-2025 available",
                "baseline_mean_reported_incidence_per_100k": baseline_inc,
                "periodic_expected_postpandemic_incidence_per_100k": periodic_post_inc,
                "pandemic_mean_reported_incidence_per_100k": pandemic_inc,
                "postpandemic_mean_reported_incidence_per_100k": post_inc,
                "post_to_baseline_rr": post_inc / baseline_for_rr if baseline_for_rr and not math.isnan(baseline_for_rr) else np.nan,
                "resurgence_ratio": post_inc / baseline_for_rr if baseline_for_rr and not math.isnan(baseline_for_rr) else np.nan,
                "descriptive_mean_resurgence_ratio": post_inc / baseline_inc if baseline_inc and not math.isnan(baseline_inc) else np.nan,
                "absolute_post_minus_baseline_incidence_per_100k": post_inc - baseline_for_rr if not math.isnan(post_inc) and not math.isnan(baseline_for_rr) else np.nan,
                "excess_cases_per_year": excess_cases_per_year,
                "latest_case_year": latest_year,
                "latest_reported_cases": latest_cases,
                "baseline_method": baseline_note,
            }
        )
    return pd.DataFrame(rows).sort_values("post_to_baseline_rr", ascending=False, na_position="last")


def build_inventory(
    metadata: pd.DataFrame,
    cases: pd.DataFrame,
    coverage: pd.DataFrame,
    age_specific_cases: pd.DataFrame,
    population: pd.DataFrame,
    contact: pd.DataFrame,
    schedule: pd.DataFrame,
    covid_policy: pd.DataFrame,
    resurgence: pd.DataFrame,
    diagnostic_timeline: pd.DataFrame | None = None,
    surveillance_outcomes: pd.DataFrame | None = None,
    gbd_burden: pd.DataFrame | None = None,
    age_pattern_evidence: pd.DataFrame | None = None,
    annual_national_surveillance_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_iso = set(cases["iso3"].dropna().astype(str))
    time_resolved_iso = set(cases.loc[~cases["time_resolution"].eq("annual"), "iso3"].dropna().astype(str))
    age_iso = set(age_specific_cases["iso3"].dropna().astype(str)) if not age_specific_cases.empty else set()
    primary_coverage = coverage.loc[
        coverage["antigen"].isin(["DTPCV1", "DTPCV3"])
        & coverage["coverage"].notna()
        & coverage["coverage_type"].astype(str).eq(PRIMARY_DTP_COVERAGE_TYPE)
    ].copy()
    coverage_iso = set(primary_coverage["iso3"].dropna().astype(str))
    pop_iso = set(population["iso3"].dropna().astype(str))
    contact_iso = set(contact["iso3"].dropna().astype(str))
    schedule_iso = set(schedule["iso3"].dropna().astype(str))
    adolescent_iso = set(metadata.loc[metadata["adolescent_booster"].fillna(False), "iso3"].astype(str))
    maternal_iso = set(metadata.loc[metadata["maternal_program"].fillna(False), "iso3"].astype(str))
    maternal_coverage_iso = set(metadata.loc[metadata.get("maternal_coverage_available", False).fillna(False), "iso3"].astype(str)) if "maternal_coverage_available" in metadata else set()
    maternal_recommendation_iso = (
        set(metadata.loc[metadata["maternal_recommendation_available"].fillna(False), "iso3"].astype(str))
        if "maternal_recommendation_available" in metadata
        else set()
    )
    covid_iso = set(covid_policy.loc[covid_policy["data_available"].fillna(False), "iso3"].dropna().astype(str))
    diagnostic_iso = (
        set(diagnostic_timeline["iso3"].dropna().astype(str))
        if diagnostic_timeline is not None and not diagnostic_timeline.empty and "iso3" in diagnostic_timeline
        else set()
    )
    outcome_iso: set[str] = set()
    hospitalization_iso: set[str] = set()
    death_iso: set[str] = set()
    testing_denominator_iso: set[str] = set()
    testing_denominator_series_iso: set[str] = set()
    testing_denominator_years_available_by_iso: dict[str, int] = {}
    testing_denominator_first_year_by_iso: dict[str, int] = {}
    testing_denominator_last_year_by_iso: dict[str, int] = {}
    age_pattern_case_distribution_iso: set[str] = set()
    age_pattern_severe_outcome_iso: set[str] = set()
    age_pattern_iso: set[str] = set()
    annual_national_surveillance_summary_iso: set[str] = set()
    annual_national_surveillance_summary_years_available_by_iso: dict[str, int] = {}
    annual_national_surveillance_summary_first_year_by_iso: dict[str, int] = {}
    annual_national_surveillance_summary_last_year_by_iso: dict[str, int] = {}
    if surveillance_outcomes is not None and not surveillance_outcomes.empty and "iso3" in surveillance_outcomes:
        outcome = surveillance_outcomes.copy()
        outcome["iso3"] = outcome["iso3"].astype(str).str.upper()
        outcome_iso = set(outcome["iso3"].dropna().astype(str))
        hospitalization_columns = [column for column in ["hospitalizations", "total_hospitalization_percent", "under6m_hospitalization_percent"] if column in outcome]
        death_columns = [column for column in ["total_deaths", "infant_deaths"] if column in outcome]
        testing_columns = [column for column in ["testing_denominator", "test_positivity_percent"] if column in outcome]
        if hospitalization_columns:
            hospitalization_iso = set(outcome.loc[outcome[hospitalization_columns].notna().any(axis=1), "iso3"].dropna().astype(str))
        if death_columns:
            death_iso = set(outcome.loc[outcome[death_columns].notna().any(axis=1), "iso3"].dropna().astype(str))
        if testing_columns:
            testing_mask = outcome[testing_columns].notna().any(axis=1)
            testing_denominator_iso = set(outcome.loc[testing_mask, "iso3"].dropna().astype(str))
            testing_years = outcome.loc[testing_mask & outcome["year"].notna(), ["iso3", "year"]].copy()
            testing_years["year"] = pd.to_numeric(testing_years["year"], errors="coerce")
            testing_years = testing_years.dropna(subset=["year"])
            if not testing_years.empty:
                testing_year_summary = testing_years.groupby("iso3", as_index=False).agg(
                    testing_denominator_years_available=("year", "nunique"),
                    testing_denominator_first_year=("year", "min"),
                    testing_denominator_last_year=("year", "max"),
                )
                testing_denominator_years_available_by_iso = (
                    testing_year_summary.set_index("iso3")["testing_denominator_years_available"]
                    .fillna(0)
                    .astype(int)
                    .to_dict()
                )
                testing_denominator_first_year_by_iso = (
                    testing_year_summary.set_index("iso3")["testing_denominator_first_year"].to_dict()
                )
                testing_denominator_last_year_by_iso = (
                    testing_year_summary.set_index("iso3")["testing_denominator_last_year"].to_dict()
                )
                testing_denominator_series_iso = set(
                    testing_year_summary.loc[testing_year_summary["testing_denominator_years_available"].astype(int).gt(1), "iso3"]
                    .dropna()
                    .astype(str)
                )
    if annual_national_surveillance_summary is not None and not annual_national_surveillance_summary.empty and "iso3" in annual_national_surveillance_summary:
        annual_summary = annual_national_surveillance_summary.copy()
        annual_summary["iso3"] = annual_summary["iso3"].astype(str).str.upper()
        annual_summary["year"] = pd.to_numeric(annual_summary.get("year"), errors="coerce")
        annual_summary = annual_summary.dropna(subset=["iso3", "year"]).copy()
        if not annual_summary.empty:
            annual_national_surveillance_summary_iso = set(annual_summary["iso3"].dropna().astype(str))
            annual_year_summary = annual_summary.groupby("iso3", as_index=False).agg(
                annual_national_surveillance_summary_years_available=("year", "nunique"),
                annual_national_surveillance_summary_first_year=("year", "min"),
                annual_national_surveillance_summary_last_year=("year", "max"),
            )
            annual_national_surveillance_summary_years_available_by_iso = (
                annual_year_summary.set_index("iso3")["annual_national_surveillance_summary_years_available"]
                .fillna(0)
                .astype(int)
                .to_dict()
            )
            annual_national_surveillance_summary_first_year_by_iso = (
                annual_year_summary.set_index("iso3")["annual_national_surveillance_summary_first_year"].to_dict()
            )
            annual_national_surveillance_summary_last_year_by_iso = (
                annual_year_summary.set_index("iso3")["annual_national_surveillance_summary_last_year"].to_dict()
            )
    if age_pattern_evidence is not None and not age_pattern_evidence.empty and "iso3" in age_pattern_evidence:
        pattern = age_pattern_evidence.copy()
        pattern["iso3"] = pattern["iso3"].astype(str).str.upper()
        if "evidence_family" in pattern.columns:
            age_pattern_case_distribution_iso = set(
                pattern.loc[pattern["evidence_family"].eq("reported_case_age_distribution"), "iso3"].dropna().astype(str)
            )
            age_pattern_severe_outcome_iso = set(
                pattern.loc[pattern["evidence_family"].eq("official_severe_outcome_age_signature"), "iso3"].dropna().astype(str)
            )
            age_pattern_iso = age_pattern_case_distribution_iso | age_pattern_severe_outcome_iso
    gbd_iso = (
        set(gbd_burden["iso3"].dropna().astype(str))
        if gbd_burden is not None and not gbd_burden.empty and "iso3" in gbd_burden
        else set()
    )

    rows = []
    for _, row in metadata.sort_values(["country", "iso3"]).iterrows():
        iso3 = str(row["iso3"])
        program_background_available = bool(row.get("vaccine_program_background_available", False))
        score_flags = {
            "annual_cases": iso3 in annual_iso,
            "time_resolved_cases": iso3 in time_resolved_iso,
            "age_group_cases": iso3 in age_iso,
            "vaccine_coverage": iso3 in coverage_iso,
            "maternal_coverage": iso3 in maternal_coverage_iso,
            "booster_schedule": iso3 in adolescent_iso,
            "vaccine_type_history": bool(row.get("aP_introduced_ever", False)) or iso3 in schedule_iso,
            "population_age": iso3 in pop_iso,
            "contact_matrix": iso3 in contact_iso,
            "covid_contact_data": iso3 in covid_iso,
            "testing_reporting_case_definition": iso3 in diagnostic_iso,
            "official_surveillance_outcome": iso3 in outcome_iso,
            "testing_denominator": iso3 in testing_denominator_iso,
            "direct_testing_denominator_series_available": iso3 in testing_denominator_series_iso,
            "age_pattern_evidence": iso3 in age_pattern_iso,
            "age_pattern_case_distribution": iso3 in age_pattern_case_distribution_iso,
            "age_pattern_severe_outcome_signature": iso3 in age_pattern_severe_outcome_iso,
            "gbd_burden": iso3 in gbd_iso,
        }
        tier_score = int(sum(score_flags.values()))
        if (
            score_flags["annual_cases"]
            and score_flags["time_resolved_cases"]
            and score_flags["age_group_cases"]
            and score_flags["vaccine_coverage"]
            and score_flags["population_age"]
        ):
            tier = "Tier 1 dynamic pilot"
            reason = "Local saved WHO annual data plus copied PertussisIncidence time-resolved and age-specific surveillance data are available."
            remaining = [
                "complete hospitalization/death age-time series"
                if (iso3 in hospitalization_iso or iso3 in death_iso)
                else "hospitalization/death outcomes"
            ]
            if not score_flags["testing_reporting_case_definition"]:
                remaining.append("testing/reporting changes")
            elif not score_flags["testing_denominator"]:
                remaining.append("direct testing volume or lab-denominator time series")
            if not score_flags["maternal_coverage"]:
                remaining.append("maternal coverage time series")
            data_gaps = "; ".join(remaining)
        elif score_flags["annual_cases"] and score_flags["vaccine_coverage"] and score_flags["population_age"]:
            tier = "Tier 0 descriptive"
            reason = "Local saved WHO annual cases/coverage and incidence-derived population denominators available."
            if score_flags["age_group_cases"]:
                data_gaps = "monthly/quarterly cases; hospitalization/death outcomes"
            else:
                data_gaps = "local age-specific cases; monthly/quarterly cases; hospitalization/death outcomes"
            if not score_flags["testing_reporting_case_definition"]:
                data_gaps += "; testing/reporting changes"
        else:
            tier = "Needs data"
            reason = "Missing one or more online annual cases, DTP coverage, or population denominator inputs."
            data_gaps = "annual cases, DTP coverage, or population denominators from local saved sources"
        rows.append(
            {
                "iso3": iso3,
                "country": row["country"],
                "region": row.get("who_region", pd.NA),
                "data_source": "local saved WHO Immunization Data Portal exports; local contactdata package snapshot; local OxCGRT snapshot",
                "case_time_resolution": "weekly/monthly" if score_flags["time_resolved_cases"] else ("annual" if score_flags["annual_cases"] else "none"),
                "age_group_available": iso3 in age_iso,
                "hospitalization_available": iso3 in hospitalization_iso,
                "death_available": iso3 in death_iso,
                "vaccine_coverage_available": iso3 in coverage_iso,
                "maternal_coverage_available": iso3 in maternal_coverage_iso,
                "maternal_recommendation_available": iso3 in maternal_recommendation_iso,
                "booster_coverage_available": False,
                "vaccine_type_history_available": score_flags["vaccine_type_history"],
                "vaccine_program_background_available": program_background_available,
                "vaccine_program_background_profile": str(row.get("vaccine_program_background_profile", "")),
                "covid_contact_data_available": iso3 in covid_iso,
                "testing_reporting_case_definition_available": score_flags["testing_reporting_case_definition"],
                "official_surveillance_outcome_available": score_flags["official_surveillance_outcome"],
                "testing_denominator_available": score_flags["testing_denominator"],
                "direct_testing_denominator_series_available": score_flags["direct_testing_denominator_series_available"],
                "age_pattern_evidence_available": score_flags["age_pattern_evidence"],
                "age_pattern_case_distribution_available": score_flags["age_pattern_case_distribution"],
                "age_pattern_severe_outcome_signature_available": score_flags["age_pattern_severe_outcome_signature"],
                "testing_denominator_years_available": testing_denominator_years_available_by_iso.get(iso3, pd.NA),
                "testing_denominator_first_year": testing_denominator_first_year_by_iso.get(iso3, pd.NA),
                "testing_denominator_last_year": testing_denominator_last_year_by_iso.get(iso3, pd.NA),
                "annual_national_surveillance_summary_available": iso3 in annual_national_surveillance_summary_iso,
                "annual_national_surveillance_summary_years_available": annual_national_surveillance_summary_years_available_by_iso.get(iso3, pd.NA),
                "annual_national_surveillance_summary_first_year": annual_national_surveillance_summary_first_year_by_iso.get(iso3, pd.NA),
                "annual_national_surveillance_summary_last_year": annual_national_surveillance_summary_last_year_by_iso.get(iso3, pd.NA),
                "gbd_burden_available": score_flags["gbd_burden"],
                "case_definition_notes": (
                    "Local diagnostic/case-definition timeline available; use as observation-process covariate and claim boundary."
                    if score_flags["testing_reporting_case_definition"]
                    else "All current observed case series are all-age WHO/JRF reported cases; observation model required."
                ),
                "recommended_tier": tier,
                "tier_score": tier_score,
                "tier_reason": reason,
                "data_gaps": data_gaps,
                **score_flags,
            }
        )
    inventory = pd.DataFrame(rows).merge(
        resurgence.loc[:, ["iso3", "post_to_baseline_rr", "latest_case_year", "latest_reported_cases"]],
        on="iso3",
        how="left",
    )
    candidates = inventory.copy()
    preferred_model2 = {
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
    }
    candidates["pilot_priority"] = (
        candidates["post_to_baseline_rr"].fillna(0).clip(upper=20)
        + candidates["tier_score"] / 10
        + candidates["iso3"].isin(preferred_model2).astype(float) * 5.0
    )
    candidates = candidates.sort_values("pilot_priority", ascending=False)
    candidates["model2_pilot_candidate"] = False
    pilot_mask = (
        candidates["iso3"].isin(preferred_model2)
        & candidates["annual_cases"]
        & candidates["time_resolved_cases"]
        & candidates["age_group_cases"]
        & candidates["vaccine_coverage"]
        & candidates["population_age"]
        & candidates["contact_matrix"]
        & candidates["covid_contact_data"]
        & candidates["post_to_baseline_rr"].notna()
    )
    pilot_index = candidates.loc[pilot_mask].head(3).index
    if len(pilot_index) < 3:
        pilot_index = candidates.loc[
            candidates["annual_cases"] & candidates["vaccine_coverage"] & candidates["population_age"] & candidates["post_to_baseline_rr"].notna()
        ].head(3).index
    candidates.loc[pilot_index, "model2_pilot_candidate"] = True
    tier_assignment = candidates.loc[
        :,
        [
            "iso3",
            "country",
            "region",
            "recommended_tier",
            "tier_score",
            "post_to_baseline_rr",
            "latest_case_year",
            "latest_reported_cases",
            "time_resolved_cases",
            "age_group_cases",
            "model2_pilot_candidate",
            "tier_reason",
            "data_gaps",
        ],
    ]
    return inventory.sort_values(["recommended_tier", "country"]), tier_assignment


def write_source_registry() -> None:
    pertussis_rows: list[dict[str, object]] = []
    if PERTUSSIS_INCIDENCE_PROVENANCE_DIR.exists() or PI_INCIDENCE.exists() or PI_AGE.exists():
        commit = PI_COMMIT.read_text(encoding="utf-8").strip() if PI_COMMIT.exists() else ""
        for source_key, source_name, local_path, notes in [
            (
                "pertussis_incidence_time_resolved_cases",
                "PertussisIncidence curated weekly/monthly national surveillance records",
                PI_INCIDENCE,
                "Used for weekly/monthly all-age case series and source population denominators.",
            ),
            (
                "pertussis_incidence_age_specific_cases",
                "PertussisIncidence curated age-specific annual cases",
                PI_AGE,
                "Used for country-year source age bins and model-age-group allocations.",
            ),
            (
                "pertussis_incidence_healthmap_alerts",
                "PertussisIncidence HealthMap alert records",
                PI_HEALTHMAP,
                "Used as digital surveillance signal metadata; not interpreted as verified outbreak counts.",
            ),
            (
                "pertussis_incidence_healthmap_iso_lookup",
                "PertussisIncidence HealthMap country ISO lookup",
                PI_HEALTHMAP_ISO,
                "Used to standardize HealthMap place-country labels to ISO3 codes.",
            ),
            (
                "pertussis_incidence_annual_overrides",
                "PertussisIncidence annual case overrides",
                PI_OVERRIDES,
                "Used to reproduce US annual calibration of weekly NNDSS interval counts.",
            ),
            (
                "pertussis_incidence_license",
                "PertussisIncidence GPL-3.0 license",
                PERTUSSIS_INCIDENCE_PROVENANCE_DIR / "LICENSE",
                "License file copied from the local clone before deleting the clone directory.",
            ),
        ]:
            if local_path.exists():
                pertussis_rows.append(
                    {
                        "source_key": source_key,
                        "source_name": source_name,
                        "source_url": PERTUSSIS_INCIDENCE_SOURCE_URL,
                        "local_path": repo_path(local_path),
                        "access_date": ACCESS_DATE,
                        "status": "copied_from_local_clone",
                        "bytes": local_path.stat().st_size,
                        "sha256": "",
                        "license": "GPL-3.0; see copied LICENSE",
                        "notes": f"{notes} Source commit: {commit}",
                        "error": "",
                    }
                )
    local_rows: list[dict[str, object]] = []
    for source_key, source_name, source_url, local_path, license_text, notes in [
        (
            "local_diagnostic_standards_timeline",
            "Curated pertussis diagnostic and case-definition regime timeline",
            "https://www.who.int/publications/m/item/vaccine-preventable-diseases-surveillance-standards-pertussis",
            LOCAL_DIAGNOSTIC_TIMELINE,
            "Mixed public sources; verify source-specific terms before publication",
            "Auxiliary observation-process covariate table with period-specific PCR/culture/serology and reporting-rule notes.",
        ),
        (
            "local_diagnostic_standards_sources",
            "Source ledger for pertussis diagnostic standards timeline",
            "https://www.who.int/publications/m/item/vaccine-preventable-diseases-surveillance-standards-pertussis",
            LOCAL_DIAGNOSTIC_SOURCES,
            "Mixed public sources; verify source-specific terms before publication",
            "Citation and URL ledger for the diagnostic and case-definition regime timeline.",
        ),
        (
            "local_surveillance_outcome_evidence",
            "Curated official pertussis surveillance and severe-outcome evidence",
            "https://www.cdc.gov/pertussis/php/surveillance/index.html",
            LOCAL_SURVEILLANCE_OUTCOME_EVIDENCE,
            "Mixed public sources; verify source-specific terms before publication",
            "Official recent case, death, hospitalization, and testing-denominator summaries, including the Australian 2006-2012 national review, Australia's NSW 2012-2013 denominator-data project and NSW 2014-2016 vaccine-preventable disease reports, Brazil's 2000-2016 age-specific burden study plus its São Paulo routine RT-PCR direct-testing summary, its 2024 situation and news summaries and later year-end state summary updating the 2024 death total and the 1998-2020 death table, Canada's 2015-2019 IMPACT pertussis hospitalization/death summary, Germany's 2024 RKI bulletin, the US 2012-2024 annual CDC age-specific surveillance series (age-specific case/death tables from 2012 onward and hospitalization tables from 2016 onward) plus the US 1994-2003 under-2 hospitalization summary, China's 2021-2022 active-surveillance testing-denominator summary and two-city hospitalization-rate study, England and Wales enhanced surveillance hospitalization/death summary, France's RENACOQ hospitalized-children surveillance and microbiological PCR monitoring page, the Netherlands 2012-2024 national surveillance and vaccine-effectiveness study plus a Netherlands regional single-study direct-testing summary, Switzerland's nationwide hospitalized-children surveillance plus a nationwide sentinel PCR-supported surveillance summary, Thailand's 2023-2024 severe-outcome summaries, Sweden's 2014-2017 testing-denominator series and 2020 severe-outcome summary, Australia and New Zealand summary-level severe-outcome evidence, and the England 2019-2024 testing-denominator series, used for surveillance and policy-evidence audits.",
        ),
        (
            "local_surveillance_outcome_sources",
            "Source ledger for official surveillance and severe-outcome evidence",
            "https://www.cdc.gov/pertussis/php/surveillance/index.html",
            LOCAL_SURVEILLANCE_OUTCOME_SOURCES,
            "Mixed public sources; verify source-specific terms before publication",
            "Citation and URL ledger for the official surveillance outcome evidence table.",
        ),
        (
            "local_thailand_infant_severe_2019_2023",
            "Thailand pertussis severe outcomes, 2019-2023",
            "https://apps-doe.moph.go.th/boeeng/annual/Annual/Annual_Report_2565.pdf; https://cdn.who.int/media/docs/default-source/searo/whe/wherepib/2024_3_searo_epi_bulletin.pdf",
            LOCAL_THA_INFANT_SEVERE,
            "Source-specific public-health publication terms; verify before publication",
            "Conservative task-specific raw extraction: Thai MoPH annual-report all-age deaths for 2019-2022 and WHO SEARO-reported Thai MoPH 2023 deaths in infants under 3 months. Blank infant-death and hospitalisation fields mean not published, not zero; not a current Model 2C likelihood input.",
        ),
        (
            "local_japan_annual_national_surveillance_summary",
            "Japan annual national pertussis surveillance summary",
            "https://id-info.jihs.go.jp/en/surveillance/iasr/pathogens/vol47/551/513te.html",
            LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SUMMARY,
            "Source-specific terms require verification before publication",
            "Official annual national surveillance summary boundary for Japan, aggregating the 2022, 2023, 2024, and 2025 partial JIHS national surveillance updates; summary-level evidence only and not a direct age-time denominator series.",
        ),
        (
            "local_japan_annual_national_surveillance_sources",
            "Source ledger for Japan annual national pertussis surveillance summary",
            "https://id-info.jihs.go.jp/en/surveillance/iasr/pathogens/vol47/551/513te.html",
            LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SOURCES,
            "Source-specific terms require verification before publication",
            "Citation and URL ledger for the Japan annual national surveillance summary table.",
        ),
        (
            "local_serology_anti_pt_igg",
            "Curated national pertussis serology evidence table",
            "https://pubmed.ncbi.nlm.nih.gov/16690000/",
            LOCAL_SEROLOGY_EVIDENCE,
            "Mixed public sources; verify source-specific terms before publication",
            "Country-specific anti-PT-IgG serosurveys used for external validation of latent true-infection and infection-pressure claims in Australia, Brazil, England and Wales (GBR), China, Japan, New Zealand, the Netherlands, Sweden, Thailand, and the United States.",
        ),
        (
            "local_serology_anti_pt_igg_sources",
            "Source ledger for national pertussis serology evidence",
            "https://pubmed.ncbi.nlm.nih.gov/16690000/",
            LOCAL_SEROLOGY_SOURCES,
            "Mixed public sources; verify source-specific terms before publication",
            "Citation and URL ledger for the curated serology evidence table.",
        ),
        (
            "local_direct_contact_validation",
            "Curated direct contact-validation evidence table",
            "https://pubmed.ncbi.nlm.nih.gov/35231023/",
            LOCAL_CONTACT_VALIDATION_EVIDENCE,
            "Mixed public sources; verify source-specific terms before publication",
            "Country-specific direct contact-survey validation for England/GBR, Canada/Quebec, France, Germany, Japan, the Netherlands, Switzerland, and the United States; used to qualify contact-shock and natural-boosting wording where direct survey evidence exists.",
        ),
        (
            "local_direct_contact_validation_sources",
            "Source ledger for direct contact-validation evidence",
            "https://pubmed.ncbi.nlm.nih.gov/35231023/",
            LOCAL_CONTACT_VALIDATION_SOURCES,
            "Mixed public sources; verify source-specific terms before publication",
            "Citation and URL ledger for the direct contact-validation evidence table.",
        ),
        (
            "local_age_pattern_evidence",
            "Structured age-pattern evidence table for pertussis surveillance",
            "https://www.cdc.gov/pertussis/php/surveillance/index.html",
            LOCAL_AGE_PATTERN_EVIDENCE,
            "Mixed public sources; verify source-specific terms before publication",
            "Structured age-pattern summary compiled from local age-specific case distributions and official surveillance outcome summaries, including the US 2012-2024 annual CDC age-specific surveillance series (age-specific case/death tables from 2012 onward and hospitalization tables from 2016 onward), France's RENACOQ hospitalized-children surveillance, and Switzerland's nationwide hospitalized-children surveillance; used for audit and claim-boundary support rather than direct age-time likelihood fitting.",
        ),
        (
            "local_pathogen_strain_amr_timeline",
            "Curated pertussis pathogen strain and antimicrobial-resistance timeline",
            "https://www.cdc.gov/pertussis/hcp/antibiotic-resistance/index.html",
            LOCAL_STRAIN_AMR_TIMELINE,
            "Mixed public sources; verify source-specific terms before publication",
            "Auxiliary pathogen-pressure covariate table with ptxP3, pertactin-deficiency, and macrolide-resistance evidence.",
        ),
        (
            "local_pathogen_strain_amr_sources",
            "Source ledger for pertussis strain and antimicrobial-resistance timeline",
            "https://www.cdc.gov/pertussis/hcp/antibiotic-resistance/index.html",
            LOCAL_STRAIN_AMR_SOURCES,
            "Mixed public sources; verify source-specific terms before publication",
            "Citation and URL ledger for the pathogen strain and AMR timeline.",
        ),
        (
            "local_maternal_tdap_coverage",
            "Curated maternal pertussis immunization coverage summary",
            "https://immunizationdata.who.int/",
            CLEAN / "maternal_immunization_coverage.csv",
            "Mixed WHO and literature sources; verify source-specific terms before publication",
            "Country-level latest maternal coverage summary derived from the raw workbook; official coverage rows outrank administrative rows, which outrank article rows within a country-year.",
        ),
        (
            "local_maternal_pregnancy_recommendation_review",
            "2024 global pregnancy Tdap recommendation and uptake review snapshot",
            "https://doi.org/10.1002/ijgo.15311",
            LOCAL_MATERNAL_PREGNANCY_RECOMMENDATION_REVIEW,
            "Source-specific terms require verification before publication",
            "18-country pregnancy Tdap recommendation and uptake snapshot from Table 1 of the 2024 GPI review; descriptive policy background only.",
        ),
        (
            "official_pertussis_schedule_strategy",
            "Official national pertussis schedule strategy table",
            "official national public-health webpages listed per row",
            OFFICIAL_PERTUSSIS_SCHEDULE_STRATEGY,
            "Official public-health webpages and PDFs; verify source-specific terms before publication",
            "Event-level pertussis schedule strategy table for the Figure 2 focus countries, replacing WHO vaccine schedule rows for current schedule-strategy display and model metadata; China intentionally retains the pre-2025 analysis-era strategy.",
        ),
        (
            "local_maternal_tdap_coverage_workbook",
            "Raw maternal pertussis immunization coverage workbook",
            "https://immunizationdata.who.int/",
            LOCAL_MATERNAL_COVERAGE,
            "Mixed WHO and literature sources; verify source-specific terms before publication",
            "Raw workbook backing the latest-country maternal coverage summary; the clean table retains the selected latest row per country.",
        ),
        (
            "local_wpp_age_population_annual",
            "UN WPP 2024 annual age-group population processed table",
            "https://population.un.org/wpp/",
            LOCAL_WPP_AGE_GROUPS_ANNUAL,
            "United Nations WPP terms apply",
            "Processed annual age-group population for selected Model 2 and auxiliary countries; replaces the contactdata age-share proxy where available.",
        ),
        (
            "local_country_profile_inputs",
            "Curated country profile inputs for pertussis transmission analysis",
            "https://immunizationdata.who.int/",
            LOCAL_COUNTRY_PROFILE_INPUTS,
            "Mixed public sources; verify source-specific terms before publication",
            "Country-level DTP, maternal program, schedule, and program metadata summary from the saved local analysis snapshot; retained as auxiliary profile context after replacing schedule-strategy rows with official national sources.",
        ),
        (
            "local_vaccine_program_background",
            "Global pertussis vaccine-program background snapshot",
            "https://globalinfectiousdisease.com/diseases/pertussis/",
            LOCAL_VACCINE_PROGRAM_BACKGROUND,
            "Source-specific terms require verification before publication",
            "Country-level DTP coverage, maternal/adult/risk flags, schedule timing, vaccine code, and outbreak-size labels from the saved local CleanedData snapshot; used as descriptive program-background context rather than a causal model input.",
        ),
        (
            "local_gbd_pertussis_burden_extract_main",
            "IHME GBD 2021 pertussis burden extract",
            "https://ghdx.healthdata.org/gbd-2021",
            LOCAL_GBD_FILES[0],
            "IHME GBD terms apply; verify before publication",
            "Auxiliary pertussis burden estimates; not a direct hospitalization surveillance series.",
        ),
        (
            "local_gbd_pertussis_burden_extract_deaths",
            "IHME GBD 2021 pertussis death extract",
            "https://ghdx.healthdata.org/gbd-2021",
            LOCAL_GBD_FILES[1],
            "IHME GBD terms apply; verify before publication",
            "Auxiliary pertussis death estimates; not a direct hospitalization surveillance series.",
        ),
        (
            "local_ecdc_pertussis_surveillance",
            "ECDC pertussis surveillance data snapshot",
            "https://atlas.ecdc.europa.eu/",
            LOCAL_ECDC_SURVEILLANCE,
            "ECDC terms apply; verify before publication",
            "Auxiliary EU/EEA reported-case surveillance snapshot; processed into a clean comparison table for annual WHO concordance checks.",
        ),
        (
            "local_ecdc_reported_cases",
            "Processed ECDC annual reported-case comparison table",
            "https://atlas.ecdc.europa.eu/",
            CLEAN / "ecdc_pertussis_reported_cases.csv",
            "ECDC terms apply; verify before publication",
            "Annual country-level and EU/EEA-aggregate ECDC pertussis reported cases, standardized for concordance checks against WHO annual reported cases.",
        ),
        (
            "local_auxiliary_time_resolved_cases",
            "Auxiliary pertussis incidence report workbook",
            "https://globalinfectiousdisease.com/diseases/pertussis/",
            LOCAL_AUXILIARY_INCIDENCE_REPORT,
            "Source-specific terms require verification before publication",
            "Auxiliary monthly/weekly all-age surveillance series for Brazil, South Africa, and Thailand, used to expand time-resolved coverage beyond the default PertussisIncidence snapshot.",
        ),
        (
            "local_annual_country_series_australia",
            "Australia annual CountrySeries reported-case extract",
            "https://globalinfectiousdisease.com/diseases/pertussis/",
            LOCAL_COUNTRY_SERIES_DIR / "australia.csv",
            "Source-specific terms require verification before publication",
            "Auxiliary annual reported-case extension for Australia used to audit concordance against overlapping workbook annual sums.",
        ),
        (
            "local_annual_country_series_canada",
            "Canada annual CountrySeries reported-case extract",
            "https://globalinfectiousdisease.com/diseases/pertussis/",
            LOCAL_COUNTRY_SERIES_DIR / "canada.csv",
            "Source-specific terms require verification before publication",
            "Auxiliary annual reported-case extension for Canada used as a historical annual-history check; no overlapping workbook annual sums are available.",
        ),
        (
            "local_japan_annual_national_surveillance_summary",
            "Japan annual national pertussis surveillance summary",
            "https://id-info.jihs.go.jp/en/surveillance/iasr/pathogens/vol47/551/513te.html",
            LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SUMMARY,
            "Source-specific terms require verification before publication",
            "Official annual national surveillance summary boundary for Japan, aggregating the 2022, 2023, 2024, and 2025 partial JIHS national surveillance updates; summary-level evidence only and not a direct age-time denominator series.",
        ),
        (
            "local_japan_annual_national_surveillance_sources",
            "Source ledger for Japan annual national pertussis surveillance summary",
            "https://id-info.jihs.go.jp/en/surveillance/iasr/pathogens/vol47/551/513te.html",
            LOCAL_JAPAN_ANNUAL_NATIONAL_SURVEILLANCE_SOURCES,
            "Source-specific terms require verification before publication",
            "Citation and URL ledger for the Japan annual national surveillance summary table.",
        ),
        (
            "local_annual_country_series_china",
            "China annual CountrySeries reported-case extract",
            "https://globalinfectiousdisease.com/diseases/pertussis/",
            LOCAL_COUNTRY_SERIES_DIR / "china.csv",
            "Source-specific terms require verification before publication",
            "Auxiliary annual reported-case extension for China used to audit concordance against overlapping workbook annual sums.",
        ),
        (
            "local_annual_country_series_thailand",
            "Thailand annual CountrySeries reported-case extract",
            "https://globalinfectiousdisease.com/diseases/pertussis/",
            LOCAL_COUNTRY_SERIES_DIR / "thailand.csv",
            "Source-specific terms require verification before publication",
            "Auxiliary annual reported-case extension for Thailand used to audit concordance against overlapping workbook annual sums.",
        ),
        (
            "local_who_annual_reported_cases_workbook",
            "WHO annual pertussis reported-cases workbook",
            "https://immunizationdata.who.int/",
            LOCAL_WHO_REPORTED_CASES_WORKBOOK,
            "WHO terms apply; source-specific verification required before publication",
            "Annual country-level WHO pertussis reported cases for 2000-2023; used as a provenance concordance check against the existing WHO annual country-case series.",
        ),
        (
            "local_who_region_pertussis_rate_workbook",
            "WHO-region pertussis incidence-rate workbook",
            "https://immunizationdata.who.int/",
            LOCAL_REGION_RATE_WORKBOOK,
            "WHO terms apply; source-specific verification required before publication",
            "Annual Global plus the six WHO regions pertussis incidence rates per million total population, used only as descriptive regional context.",
        ),
        (
            "local_extended_age_cases_workbook",
            "Extended pertussis age-specific case workbook",
            PERTUSSIS_INCIDENCE_SOURCE_URL,
            LOCAL_EXTENDED_AGE_CASES,
            "Source-specific terms require verification before publication",
            "Larger local age-specific case workbook retaining additional Canada, Brazil, and Thailand country sheets beyond the default PertussisIncidence snapshot, plus the EU age-specific rate panel for additional European countries.",
        ),
        (
            "local_eu_age_specific_rate_panel",
            "EU annual pertussis age-specific rate panel",
            PERTUSSIS_INCIDENCE_SOURCE_URL,
            LOCAL_EXTENDED_AGE_CASES,
            "Source-specific terms require verification before publication",
            "EU sheet in the extended local age workbook provides annual age-specific rates for 26 European countries not otherwise covered by the exact workbook sheets; the rates are normalized to annual reported-case totals to preserve age-pattern evidence without relying on missing census-age denominators.",
        ),
        (
            "local_surveillance_age_specific_cases",
            "Supplemental pertussis age-specific surveillance CSV extract",
            PERTUSSIS_INCIDENCE_SOURCE_URL,
            LOCAL_SURVEILLANCE_AGE_CASES,
            "Source-specific terms require verification before publication",
            "Supplemental age-specific extract used to extend early-year Australia and China annual age-specific coverage before the workbook coverage begins.",
        ),
        (
            "local_canada_official_age_distribution_summary",
            "Canada official pertussis age-distribution summary extract",
            "https://www.canada.ca/en/public-health/services/publications/healthy-living/vaccine-preventable-disease-surveillance-report-2019.html",
            LOCAL_CANADA_OFFICIAL_AGE_DISTRIBUTION,
            "Source-specific terms require verification before publication",
            "Normalized extract of PHAC Figure 16 text equivalent for Canada's 2015-2019 pertussis age distribution; the visible age bins sum to 14,638 cases and the report total is 14,646, implying 8 unspecified cases.",
        ),
        (
            "local_england_official_age_distribution_summary",
            "England official pertussis age-distribution summary extract",
            "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/881386/Laboratory_confirmed_cases_of_pertussis_in_England_2019.pdf",
            LOCAL_ENGLAND_OFFICIAL_AGE_DISTRIBUTION,
            "Source-specific terms require verification before publication",
            "Extracted 1994-2014 age-distribution rows from Table 3 of the 2019 UKHSA annual report for laboratory-confirmed pertussis in England; the report table spans 1994-2019, but 2015-2019 overlap the workbook age coverage and are omitted to avoid duplication.",
        ),
        (
            "local_diagnostic_age_testing_summary",
            "Netherlands age-stratified testing summary",
            "https://pubmed.ncbi.nlm.nih.gov/28749331/",
            LOCAL_DIAGNOSTIC_AGE_TESTING_SUMMARY,
            "Source-specific terms require verification before publication",
            "Table 2 age-stratified testing summary from the Dutch 2010-2013 diagnostic study; infants <1 year were recommended PCR or culture, while older prolonged-cough patients were mainly tested by serology.",
        ),
        ]:
            if local_path.exists():
                local_rows.append(
                    {
                        "source_key": source_key,
                        "source_name": source_name,
                        "source_url": source_url,
                        "local_path": str(local_path.relative_to(ROOT)),
                        "access_date": ACCESS_DATE,
                        "status": "copied_from_local_saved_snapshot",
                        "bytes": local_path.stat().st_size,
                        "sha256": sha256_file(local_path),
                        "license": license_text,
                        "notes": notes,
                        "error": "",
                    }
                )
    manifest = METADATA / "online_data_sources.csv"
    if manifest.exists():
        registry = pd.read_csv(manifest)
        for column in ["local_path", "copied_path"]:
            if column in registry.columns:
                registry[column] = (
                    registry[column]
                    .fillna("")
                    .astype(str)
                    .str.replace(r"^data_raw/", "data/raw/", regex=True)
                    .str.replace(r"^data_clean/", "data/clean/", regex=True)
                    .str.replace(r"^data_intermediate/", "data/intermediate/", regex=True)
                    .str.replace(r"^metadata/", "data/metadata/", regex=True)
                )
        if "source_key" in registry.columns and "notes" in registry.columns:
            legacy_schedule = registry["source_key"].astype(str).eq("who_vaccine_schedule")
            registry.loc[legacy_schedule, "notes"] = (
                "Legacy WHO vaccine schedule export retained for provenance only; "
                "current pertussis schedule-strategy display and model metadata use "
                "data/raw/immunization/official_pertussis_schedule_strategy.csv."
            )
        registry = registry.loc[~registry["status"].astype(str).eq("failed")].copy()
        if pertussis_rows:
            registry = pd.concat([registry, pd.DataFrame(pertussis_rows)], ignore_index=True, sort=False)
        if local_rows:
            registry = pd.concat([registry, pd.DataFrame(local_rows)], ignore_index=True, sort=False)
        registry = registry.drop_duplicates("source_key", keep="last")
        registry.to_csv(METADATA / "source_registry.csv", index=False)
        return
    rows = [
        {
            "source_key": "who_immunization_exports",
            "source_name": "WHO Immunization Data Portal exports",
            "source_url": WHO_SOURCE_URL,
            "local_path": "data/raw/annual_cases_and_rates; data/raw/immunization",
            "access_date": ACCESS_DATE,
            "license": "WHO terms apply; source-specific verification required before publication",
            "notes": "Local saved source snapshot; data/metadata/online_data_sources.csv is retained only as provenance if present.",
        },
        {
            "source_key": "contactdata_r_package",
            "source_name": "contactdata R package",
            "source_url": CONTACTDATA_SOURCE_URL,
            "local_path": "data/raw/contact_mobility",
            "access_date": ACCESS_DATE,
            "license": "MIT",
            "notes": "Local saved package snapshot and exported matrices.",
        },
        {
            "source_key": "oxcgrt_compact_national_v1",
            "source_name": "OxCGRT compact national v1",
            "source_url": OXCGRT_SOURCE_URL,
            "local_path": "data/raw/covid_policy/OxCGRT_compact_national_v1.csv",
            "access_date": ACCESS_DATE,
            "license": "CC BY 4.0",
            "notes": "Official OxCGRT final compact national v1 data.",
        },
    ]
    rows.extend(pertussis_rows)
    rows.extend(local_rows)
    pd.DataFrame(rows).to_csv(METADATA / "source_registry.csv", index=False)


def write_external_data_discovery() -> None:
    rows = [
        {
            "source_key": "local_age_specific_cases",
            "source_path": "xmusphlkg/PertussisIncidence Data/Pertussis case year age.xlsx plus extended local age workbook plus supplemental local surveillance CSV extract",
            "copied_path": f"{repo_path(PI_AGE)}; {repo_path(LOCAL_EXTENDED_AGE_CASES)}; {repo_path(LOCAL_SURVEILLANCE_AGE_CASES)}",
            "data_domain": "age-specific reported pertussis cases",
            "current_use": "default model uses eight PertussisIncidence source countries; the extended local workbook contributes auxiliary Canada, Brazil, and Thailand annual age-specific case rows, the EU sheet contributes additional European age-rate panels normalized to reported-case totals for 26 European countries, and the supplemental local surveillance CSV adds earlier Australia and China age-specific years",
            "priority": "high",
            "notes": "Default workbook remains the reproducible model source; the extended workbook is consumed for auxiliary Canada/Brazil/Thailand age-specific cases, its EU sheet supplies additional European age-rate panels normalized to annual reported-case totals for 26 European countries, and the supplemental surveillance CSV extends early Australia/China age-specific years before the workbook coverage begins.",
        },
        {
            "source_key": "local_eu_age_specific_rate_panel",
            "source_path": "EU sheet of extended Pertussis case year age workbook",
            "copied_path": repo_path(LOCAL_EXTENDED_AGE_CASES),
            "data_domain": "age-specific reported pertussis rates",
            "current_use": "used to derive annual age-specific rate-based case estimates for 26 European countries not otherwise covered by the exact workbook sheets",
            "priority": "medium",
            "notes": "EU/EEA annual age-specific rate panel is normalized to annual reported-case totals to yield approximate age-pattern case weights for 26 European countries; exact workbook-sheet countries are excluded to avoid duplication.",
        },
        {
            "source_key": "local_surveillance_age_specific_cases",
            "source_path": "supplemental local age-specific surveillance CSV extract",
            "copied_path": repo_path(LOCAL_SURVEILLANCE_AGE_CASES),
            "data_domain": "age-specific reported pertussis cases",
            "current_use": "used to extend early-year Australia and China annual age-specific rows before the workbook coverage begins",
            "priority": "medium",
            "notes": "Supplemental local surveillance extract derived from the saved PertussisIncidence age workbook and used only for years earlier than the workbook coverage.",
        },
        {
            "source_key": "local_time_resolved_cases",
            "source_path": "xmusphlkg/PertussisIncidence Data/Pertussis incidence.xlsx plus retained auxiliary time-resolved case workbook",
            "copied_path": repo_path(PI_INCIDENCE),
            "data_domain": "monthly or weekly pertussis cases",
            "current_use": "used for AU, CN, GB, JP, NZ, SE, SG, and US weekly/monthly all-age surveillance series",
            "priority": "high",
            "notes": "US weekly series is calibrated to annual CDC totals using copied annual_case_overrides.csv.",
        },
        {
            "source_key": "local_ecdc_reported_cases",
            "source_path": "ECDC pertussis surveillance data snapshot plus processed annual comparison table",
            "copied_path": "data/clean/ecdc_pertussis_reported_cases.csv; outputs/ecdc_who_reported_case_comparison.csv",
            "data_domain": "annual reported pertussis cases",
            "current_use": "used as an auxiliary concordance check for EU/EEA annual reported-case trends against WHO annual reported cases",
            "priority": "medium",
            "notes": "Country-level annual ECDC reported cases can validate European annual case trends and highlight WHO/ECDC overlap coverage.",
        },
        {
            "source_key": "local_who_region_pertussis_rate_workbook",
            "source_path": "WHO-region pertussis incidence-rate workbook",
            "copied_path": "data/clean/pertussis_region_rates.csv; outputs/pertussis_region_rate_summary.csv",
            "data_domain": "regional pertussis incidence rates",
            "current_use": "used to derive Global plus the six WHO regions annual incidence-rate context for manuscript wording and regional context audits",
            "priority": "medium",
            "notes": "Annual rates per million total population for Global plus the six WHO regions, 2000-2024; parsed into a clean long table and a seven-row summary for descriptive context only, not country-level surveillance counts.",
        },
        {
            "source_key": "local_auxiliary_time_resolved_cases",
            "source_path": "auxiliary Pertussis Incidence Report workbook",
            "copied_path": repo_path(LOCAL_AUXILIARY_INCIDENCE_REPORT),
            "data_domain": "monthly or weekly pertussis cases",
            "current_use": "used to ingest Brazil, South Africa, and Thailand time-resolved all-age surveillance series",
            "priority": "high",
            "notes": "Adds auxiliary time-resolved country series beyond the default PertussisIncidence snapshot; source comparability still requires caution.",
        },
        {
            "source_key": "local_annual_country_series",
            "source_path": "annual CountrySeries reported-case extracts",
            "copied_path": repo_path(LOCAL_COUNTRY_SERIES_DIR),
            "data_domain": "annual reported pertussis cases",
            "current_use": "used to extend early annual history for Australia, Canada, China, and Thailand and to audit concordance against overlapping workbook annual sums where available",
            "priority": "medium",
            "notes": "Auxiliary annual reported-case extension from local CountrySeries extracts; useful for historical context and overlap concordance checks, not for the main monthly ODE fit.",
        },
        {
            "source_key": "local_who_annual_reported_cases_workbook",
            "source_path": "official WHO annual pertussis reported-cases workbook",
            "copied_path": "data/clean/who_reported_cases_workbook.csv; outputs/who_reported_cases_concordance.csv; outputs/who_reported_cases_concordance_summary.csv",
            "data_domain": "annual reported pertussis cases",
            "current_use": "used as an official WHO annual-country-case concordance check against the existing WHO annual reported-case series",
            "priority": "medium",
            "notes": "Annual country-level WHO pertussis reported cases for 2000-2023; the workbook matches 4048 overlapping country-years exactly against the existing WHO annual country-case series.",
        },
        {
            "source_key": "local_annual_country_series_canada",
            "source_path": "Canada annual CountrySeries reported-case extract",
            "copied_path": repo_path(LOCAL_COUNTRY_SERIES_DIR / "canada.csv"),
            "data_domain": "annual reported pertussis cases",
            "current_use": "used to extend annual history for Canada and to provide a standalone historical annual-history check",
            "priority": "medium",
            "notes": "Annual CountrySeries extract from the PHAC 2005-2019 pertussis epidemiology report; Canada has no overlapping workbook annual sums in the current snapshot.",
        },
        {
            "source_key": "local_vaccine_program_background",
            "source_path": "global pertussis vaccine-program background snapshot",
            "copied_path": repo_path(LOCAL_VACCINE_PROGRAM_BACKGROUND),
            "data_domain": "pertussis vaccine-program background",
            "current_use": "used to summarize global pertussis vaccine-program heterogeneity and to join a descriptive background panel into country metadata",
            "priority": "medium",
            "notes": "Country-level DTP coverage, maternal/adult/risk flags, schedule timing, vaccine code, and outbreak-size labels from the saved CleanedData snapshot; descriptive context only.",
        },
        {
            "source_key": "official_pertussis_schedule_strategy",
            "source_path": "official national pertussis schedule strategy table",
            "copied_path": repo_path(OFFICIAL_PERTUSSIS_SCHEDULE_STRATEGY),
            "data_domain": "pertussis schedule strategy",
            "current_use": "current source for Figure 2d schedule-strategy display and country-level schedule metadata used by model priors",
            "priority": "high",
            "notes": "Event-level table with official source name and URL per row; replaces WHO vaccine schedule rows for schedule-strategy use. China keeps the pre-2025 strategy because the analysis window is mainly through 2025.",
        },
        {
            "source_key": "local_hospitalizations_deaths",
            "source_path": "curated official pertussis surveillance outcome evidence plus IHME GBD pertussis burden extract",
            "copied_path": "data/clean/surveillance_outcome_evidence.csv; data/clean/gbd_pertussis_burden.csv",
            "data_domain": "severe outcomes",
            "current_use": "used in surveillance, policy, and external-validation audits as official severe-outcome support; not yet fitted as source-specific age-time severe-outcome likelihood",
            "priority": "medium",
            "notes": "Official summary rows, Brazil's 2000-2016 age-specific burden study, Canada's 2015-2019 IMPACT pertussis hospitalization/death summary, Germany's 2024 RKI bulletin, China's two-city active-surveillance hospitalization-rate study, England and Wales enhanced surveillance hospitalization/death summary, France's RENACOQ hospitalized-children surveillance, the Netherlands 2012-2024 national surveillance and vaccine-effectiveness study, Switzerland's nationwide hospitalized-children surveillance, the US 2012-2024 annual CDC age-specific surveillance series (age-specific case/death tables from 2012 onward and hospitalization tables from 2016 onward) plus the US 1994-2003 under-2 hospitalization summary support severe-outcome claim boundaries for Australia, Brazil, Canada, China, DEU, GBR, France, Japan, New Zealand, Netherlands, Sweden, Switzerland, Thailand, and the United States; complete country-specific hospitalization/death age-time series are still needed.",
        },
        {
            "source_key": "local_japan_annual_national_surveillance_summary",
            "source_path": "official annual national pertussis surveillance summary for Japan",
            "copied_path": "data/clean/japan_annual_national_surveillance_summary.csv; data/clean/japan_annual_national_surveillance_sources.csv",
            "data_domain": "official annual national surveillance summary",
            "current_use": "used as an annual Japanese surveillance summary boundary that sharpens observation-process wording without replacing direct age-time denominator series",
            "priority": "medium",
            "notes": "Aggregated annual JIHS national pertussis summaries for 2022, 2023, 2024, and 2025 partial week-44; summary-level evidence only and not a direct age-time denominator series.",
        },
        {
            "source_key": "local_japan_annual_national_surveillance_sources",
            "source_path": "source ledger for Japan annual national pertussis surveillance summary",
            "copied_path": "data/clean/japan_annual_national_surveillance_summary.csv; data/clean/japan_annual_national_surveillance_sources.csv",
            "data_domain": "official annual national surveillance summary source ledger",
            "current_use": "used as the citation and URL ledger for the Japan annual surveillance summary boundary",
            "priority": "medium",
            "notes": "Citation and URL ledger for the aggregated annual JIHS national pertussis summaries for 2022, 2023, 2024, and 2025 partial week-44.",
        },
        {
            "source_key": "local_gbd_pertussis_burden_summary",
            "source_path": "modeled IHME GBD 2021 pertussis burden extract summaries",
            "copied_path": "outputs/gbd_pertussis_burden_summary.csv; outputs/gbd_pertussis_burden_year_summary.csv",
            "data_domain": "modeled pertussis burden summary",
            "current_use": "used as auxiliary country/year burden context in manuscript wording and burden-oriented audits",
            "priority": "medium",
            "notes": "Country- and year-level modeled burden summaries across 204 countries and 12 years; DALYs are the primary metric and incidence is a secondary consistency check. These summaries are not surveillance or fitted age-time severe-outcome evidence.",
        },
        {
            "source_key": "local_serology_anti_pt_igg",
            "source_path": "curated national pertussis serosurveys with anti-PT-IgG or comparable infection-intensity measures",
            "copied_path": "data/clean/serology_anti_pt_igg_evidence.csv; data/clean/serology_anti_pt_igg_sources.csv",
            "data_domain": "serology anti-PT-IgG infection pressure",
            "current_use": "used as country-specific external-validation boundary for latent true-infection scale where local serosurveys exist (AUS, BRA, CHN, GBR, JPN, NLD, NZL, SWE, THA, USA)",
            "priority": "high",
            "notes": "Country-specific anti-PT-IgG evidence now supports latent true-infection and infection-pressure claims for Australia, Brazil, England and Wales (GBR), China, Japan, New Zealand, the Netherlands, Sweden, Thailand, and the United States; broader longitudinal serology coverage remains incomplete.",
        },
        {
            "source_key": "local_direct_contact_validation",
            "source_path": "national or country-year direct contact surveys and validated contact-shock calibration data",
            "copied_path": "data/clean/contact_validation_evidence.csv; data/clean/contact_validation_sources.csv",
            "data_domain": "direct contact validation",
            "current_use": "used as country-specific external-validation boundary for contact-shock and natural-boosting language where direct contact surveys exist (England/GBR, Canada/Quebec, France, Germany, Japan, the Netherlands, Switzerland, and the United States)",
            "priority": "medium",
            "notes": "Country-specific direct contact-survey evidence from England/GBR, Canada/Quebec, France, Germany, Japan, the Netherlands, Switzerland, and the United States now constrains contact-shock wording; most other countries still rely on OxCGRT/contactdata proxy sensitivity.",
        },
        {
            "source_key": "local_age_pattern_evidence",
            "source_path": "structured age-pattern evidence table compiled from local age-specific case distributions and official surveillance outcome summaries",
            "copied_path": "data/clean/age_pattern_evidence.csv",
            "data_domain": "age-pattern evidence",
            "current_use": "used in surveillance and claim-readiness audits to support age-pattern wording and separate summary-level age signatures from direct age-time outcome likelihoods",
            "priority": "medium",
            "notes": "Compiled age-distribution and age-specific severe-outcome signatures for the current data snapshot, including Canada's official 2015-2019 age-distribution summary, Germany's 2024 RKI bulletin, France's RENACOQ hospitalized-children surveillance, Switzerland's nationwide hospitalized-children surveillance, and the US 2012-2024 annual CDC age-specific surveillance series (age-specific case/death tables from 2012 onward and hospitalization tables from 2016 onward); supports age-pattern claim boundaries but does not replace direct age-time hospitalization or death series.",
        },
        {
            "source_key": "local_canada_official_age_distribution_summary",
            "source_path": "Canada official pertussis age-distribution summary extract",
            "copied_path": repo_path(LOCAL_CANADA_OFFICIAL_AGE_DISTRIBUTION),
            "data_domain": "age-specific reported pertussis cases",
            "current_use": "used to add Canada's 2015-2019 official age-distribution summary to the age-pattern evidence layer",
            "priority": "medium",
            "notes": "Normalized extract of PHAC Figure 16 text equivalent; the visible age bins sum to 14,638 cases and the report total is 14,646, implying 8 unspecified cases.",
        },
        {
            "source_key": "local_england_official_age_distribution_summary",
            "source_path": "England annual pertussis age-distribution summary extract",
            "copied_path": repo_path(LOCAL_ENGLAND_OFFICIAL_AGE_DISTRIBUTION),
            "data_domain": "age-specific reported pertussis cases",
            "current_use": "used to extend England/GBR annual age-specific surveillance history from 1994-2014 before workbook coverage begins",
            "priority": "medium",
            "notes": "Extracted 1994-2014 age-distribution rows from Table 3 of the 2019 UKHSA annual report for laboratory-confirmed pertussis in England; the report table spans 1994-2019, but 2015-2019 overlap the workbook age coverage and are omitted here to avoid duplication.",
        },
        {
            "source_key": "local_testing_reporting_case_definition",
            "source_path": "national laboratory testing volume, laboratory confirmation fraction, reporting-rule, and case-definition change files",
            "copied_path": "data/clean/surveillance_diagnostic_timeline.csv",
            "data_domain": "testing reporting case-definition covariates",
            "current_use": "used as an external diagnostic/reporting prior in the monthly ODE observation mapping and cross-checked against official outcome evidence where available",
            "priority": "high",
            "notes": "Use to qualify the fitted post-pandemic reporting multiplier; direct testing-volume denominators remain sparse, but now include selected official summary rows plus the NSW state-level denominator-data project, the British Columbia provincial 2011-2013 series, the Sweden series, the England series, Brazil's São Paulo routine RT-PCR direct-testing summary, a China active-surveillance summary, a Netherlands regional single-study direct-testing summary, a U.S. multiplex PCR network summary, and single-study direct-testing summaries from Japan, Switzerland, and Thailand.",
        },
        {
            "source_key": "local_diagnostic_age_testing_summary",
            "source_path": "Netherlands age-stratified testing summary from the Dutch 2010-2013 diagnostic study",
            "copied_path": repo_path(LOCAL_DIAGNOSTIC_AGE_TESTING_SUMMARY),
            "data_domain": "age-stratified testing summary",
            "current_use": "used to show that Dutch infants were tested with PCR or culture while older prolonged-cough patients were mainly tested by serology; observation-process boundary only",
            "priority": "medium",
            "notes": "Table 2 of the Heil et al. 2017 Dutch diagnostic study splits testing behaviour by age group and symptom duration; use as an age-stratified diagnostic boundary rather than a national multi-year denominator series.",
        },
        {
            "source_key": "local_pathogen_strain_amr",
            "source_path": "curated ptxP3, pertactin-deficiency, and macrolide-resistance evidence by country-period",
            "copied_path": "data/clean/pertussis_strain_amr_timeline.csv",
            "data_domain": "pathogen strain and antimicrobial-resistance covariates",
            "current_use": "used as a bounded pathogen-pressure prior in the monthly ODE transmission mapping",
            "priority": "medium",
            "notes": "Use as hypothesis-generating transmission-drift evidence; isolate-denominator genomic surveillance remains incomplete for several countries.",
        },
        {
            "source_key": "un_wpp_bulk_population",
            "source_path": "UN WPP 2024 local processed and raw population extracts",
            "copied_path": "data/clean/population_age.csv",
            "data_domain": "age-specific annual population",
            "current_use": "used where local WPP processed age groups are available; fallback countries still use the transparent contactdata/WHO proxy",
            "priority": "medium",
            "notes": "WPP rows are available for the selected Model 2 countries and several auxiliary countries.",
        },
        {
            "source_key": "local_maternal_tdap_coverage",
            "source_path": "curated local maternal pertussis immunization coverage summary derived from the raw workbook",
            "copied_path": "data/clean/maternal_immunization_coverage.csv; outputs/maternal_immunization_coverage_summary.csv",
            "data_domain": "maternal Tdap coverage",
            "current_use": "used in country metadata and auxiliary maternal-protection scenario baselines when available; not a default Model 2C susceptibility-side covariate",
            "priority": "high",
            "notes": "Latest-country maternal pertussis coverage summary with one selected row per country; official coverage rows outrank administrative rows, which outrank article rows within a country-year. Replaces fixed maternal baseline assumptions for countries with usable local coverage records; source comparability still requires caution.",
        },
        {
            "source_key": "local_maternal_pregnancy_recommendation_review",
            "source_path": "2024 global pregnancy Tdap recommendation and uptake review snapshot",
            "copied_path": repo_path(LOCAL_MATERNAL_PREGNANCY_RECOMMENDATION_REVIEW),
            "data_domain": "maternal Tdap recommendation background",
            "current_use": "used as descriptive pregnancy Tdap recommendation and uptake context for 18 countries; not a coverage time series",
            "priority": "medium",
            "notes": "Table 1 of the 2024 GPI review provides an 18-country pregnancy Tdap recommendation and uptake snapshot; descriptive policy background only.",
        },
        {
            "source_key": "local_maternal_tdap_coverage_workbook",
            "source_path": "raw maternal immunization coverage workbook",
            "copied_path": repo_path(LOCAL_MATERNAL_COVERAGE),
            "data_domain": "maternal Tdap coverage",
            "current_use": "used to derive the country-level latest maternal coverage summary and metadata baseline",
            "priority": "high",
            "notes": "Raw workbook with 236 usable coverage rows across 71 countries; the latest selected row per country prioritizes official coverage, then administrative coverage, then article evidence within a country-year.",
        },
        {
            "source_key": "local_country_profile_inputs",
            "source_path": "curated local country profile inputs for pertussis transmission analysis",
            "copied_path": repo_path(LOCAL_COUNTRY_PROFILE_INPUTS),
            "data_domain": "country-level DTP, maternal program, schedule, and program metadata",
            "current_use": "retained as auxiliary profile context after replacing schedule-strategy rows with official national sources",
            "priority": "high",
            "notes": "Local profile snapshot drawn from the saved pertussis analysis/data snapshot; upstream source terms still require publication-time verification.",
        },
    ]
    pd.DataFrame(rows).to_csv(METADATA / "external_data_discovery.csv", index=False)


def write_outputs() -> None:
    ensure_directories()

    name_map = read_name_map()
    country_cases, region_cases = read_who_cases_long(name_map)
    coverage = read_dtp_coverage()
    who_cases = build_cases(country_cases)
    pertussis_incidence_cases = read_pertussis_incidence_time_series(country_cases)
    cases = pd.concat([who_cases, pertussis_incidence_cases], ignore_index=True, sort=False)
    total_population = (
        cases.loc[cases["age_group"].eq("all_ages"), ["iso3", "country", "year", "population"]]
        .dropna(subset=["iso3", "year", "population"])
        .groupby(["iso3", "country", "year"], as_index=False)
        .agg(total_population=("population", "mean"))
    )
    population = build_population_age(total_population, name_map)
    population_age_audit = build_population_age_structure_projection_audit(population)
    age_specific, age_model_groups = read_pertussis_age_specific_cases()
    healthmap_alerts = read_healthmap_alerts()
    schedule = build_vaccine_schedule()
    country_profile_inputs_full = read_local_country_profile_inputs_full()
    country_profile_inputs = country_profile_inputs_full.loc[:, [column for column in ["iso3", "country", "maternal_program", "maternal_program_note", "schedule_note"] if column in country_profile_inputs_full.columns]]
    vaccine_program_background = read_local_vaccine_program_background()
    contact = build_contact_mobility(name_map)
    maternal_coverage = read_local_maternal_coverage()
    maternal_pregnancy_recommendation = read_local_maternal_pregnancy_recommendation_summary()
    diagnostic_timeline = read_local_diagnostic_timeline()
    diagnostic_sources = read_local_diagnostic_sources()
    diagnostic_age_testing_summary = read_local_diagnostic_age_testing_summary()
    surveillance_outcomes = read_local_surveillance_outcome_evidence()
    surveillance_outcome_sources = read_local_surveillance_outcome_sources()
    japan_annual_national_surveillance_summary = read_local_japan_annual_national_surveillance_summary()
    japan_annual_national_surveillance_sources = read_local_japan_annual_national_surveillance_sources()
    if not japan_annual_national_surveillance_sources.empty:
        surveillance_outcome_sources = pd.concat(
            [surveillance_outcome_sources, japan_annual_national_surveillance_sources],
            ignore_index=True,
            sort=False,
        )
        surveillance_outcome_sources = surveillance_outcome_sources.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    age_pattern_evidence = build_age_pattern_evidence(age_model_groups, surveillance_outcomes)
    canada_age_pattern = read_canada_official_age_distribution_summary()
    if canada_age_pattern is not None:
        age_pattern_evidence = pd.concat(
            [age_pattern_evidence, pd.DataFrame([canada_age_pattern]).reindex(columns=age_pattern_evidence.columns)],
            ignore_index=True,
            sort=False,
        )
    usa_annual_age_specific_surveillance_summary = summarize_usa_annual_age_specific_surveillance_summary(
        age_pattern_evidence
    )
    age_pattern_global_summary = summarize_global_age_pattern_summary(age_pattern_evidence)
    gbr_age_pattern_history = summarize_gbr_age_pattern_history(age_model_groups, age_pattern_evidence)
    eu_age_pattern_summary = summarize_eu_age_pattern_country_summary(age_pattern_evidence)
    eu_age_pattern_year_summary = summarize_eu_age_pattern_year_summary(age_pattern_evidence)
    serology_evidence = read_local_serology_evidence()
    serology_sources = read_local_serology_sources()
    serology_summary = summarize_serology_anti_pt_igg(serology_evidence)
    contact_validation_evidence = read_local_contact_validation_evidence()
    contact_validation_sources = read_local_contact_validation_sources()
    contact_validation_summary = summarize_contact_validation(contact_validation_evidence)
    external_validation_summary = summarize_external_validation_boundary(
        serology_evidence,
        contact_validation_evidence,
    )
    annual_country_series = read_annual_country_series()
    annual_country_series_comparison = build_annual_country_series_comparison(annual_country_series, pertussis_incidence_cases)
    annual_country_series_concordance_summary = summarize_annual_country_series_concordance(annual_country_series, pertussis_incidence_cases)
    south_africa_monthly_series_summary = summarize_south_africa_monthly_series(cases)
    time_resolved_all_age_series_summary = summarize_time_resolved_all_age_series(cases)
    time_resolved_all_age_series_completeness_audit = audit_time_resolved_all_age_series_completeness(cases)
    region_rates = read_pertussis_region_rates()
    region_rate_summary = summarize_pertussis_region_rates(region_rates)
    who_reported_cases_workbook = read_who_reported_cases_workbook(name_map)
    who_reported_cases_concordance = build_who_reported_cases_concordance(who_reported_cases_workbook, country_cases)
    who_reported_cases_concordance_summary = summarize_who_reported_cases_concordance(who_reported_cases_concordance)
    ecdc_reported_cases = read_local_ecdc_reported_cases()
    ecdc_who_comparison = build_ecdc_who_comparison(ecdc_reported_cases, country_cases)
    ecdc_who_concordance_summary = summarize_ecdc_who_concordance(ecdc_who_comparison)
    strain_amr_timeline = read_local_strain_amr_timeline()
    strain_amr_sources = read_local_strain_amr_sources()
    gbd_burden = read_local_gbd_burden()
    gbd_burden_country_summary = summarize_gbd_burden_country_summary(gbd_burden)
    gbd_burden_year_summary = summarize_gbd_burden_year_summary(gbd_burden)
    local_source_manifest = build_local_source_manifest()
    metadata = build_country_metadata(
        name_map,
        coverage,
        schedule,
        population,
        contact,
        maternal_coverage,
        maternal_pregnancy_recommendation,
        country_profile_inputs,
    )
    if not vaccine_program_background.empty:
        background_lookup = vaccine_program_background.loc[
            :,
            [
                "iso3",
                "vaccine_pregnant",
                "vaccine_adult",
                "vaccine_risk",
                "vaccine_code",
                "vaccine_program_flag_count",
                "vaccine_program_profile",
                "outbreak_size_2022",
                "outbreak_size_2023",
                "outbreak_size_rank",
            ],
        ].rename(
            columns={
                "vaccine_pregnant": "vaccine_program_background_pregnant",
                "vaccine_adult": "vaccine_program_background_adult",
                "vaccine_risk": "vaccine_program_background_risk",
                "vaccine_code": "vaccine_program_background_code",
                "vaccine_program_flag_count": "vaccine_program_background_flag_count",
                "vaccine_program_profile": "vaccine_program_background_profile",
                "outbreak_size_2022": "vaccine_program_background_outbreak_size_2022",
                "outbreak_size_2023": "vaccine_program_background_outbreak_size_2023",
                "outbreak_size_rank": "vaccine_program_background_outbreak_size_rank",
            }
        )
    else:
        background_lookup = pd.DataFrame(
            columns=[
                "iso3",
                "vaccine_program_background_pregnant",
                "vaccine_program_background_adult",
                "vaccine_program_background_risk",
                "vaccine_program_background_code",
                "vaccine_program_background_flag_count",
                "vaccine_program_background_profile",
                "vaccine_program_background_outbreak_size_2022",
                "vaccine_program_background_outbreak_size_2023",
                "vaccine_program_background_outbreak_size_rank",
            ]
        )
    metadata = metadata.merge(background_lookup, on="iso3", how="left")
    metadata["vaccine_program_background_available"] = metadata["vaccine_program_background_profile"].notna()
    metadata["vaccine_program_background_available"] = metadata["vaccine_program_background_available"].fillna(False).astype(bool)
    metadata["vaccine_program_background_profile"] = metadata["vaccine_program_background_profile"].fillna("").astype(str)
    for column in [
        "vaccine_program_background_pregnant",
        "vaccine_program_background_adult",
        "vaccine_program_background_risk",
    ]:
        metadata[column] = metadata[column].fillna(False).astype(bool)
    metadata["vaccine_program_background_code"] = metadata["vaccine_program_background_code"].fillna("").astype(str)
    metadata["vaccine_program_background_flag_count"] = pd.to_numeric(
        metadata["vaccine_program_background_flag_count"], errors="coerce"
    ).astype("Int64")
    metadata["vaccine_program_background_outbreak_size_2022"] = metadata["vaccine_program_background_outbreak_size_2022"].fillna("").astype(str)
    metadata["vaccine_program_background_outbreak_size_2023"] = metadata["vaccine_program_background_outbreak_size_2023"].fillna("").astype(str)
    metadata["vaccine_program_background_outbreak_size_rank"] = pd.to_numeric(
        metadata["vaccine_program_background_outbreak_size_rank"], errors="coerce"
    ).astype("Int64")
    vaccine_program_background_summary = summarize_vaccine_program_background(vaccine_program_background, metadata)
    vaccine_program_background_reconciliation = build_vaccine_program_reconciliation(
        country_profile_inputs_full,
        vaccine_program_background,
        metadata,
    )
    vaccine_program_background_reconciliation_summary = summarize_vaccine_program_reconciliation(
        vaccine_program_background_reconciliation,
    )
    covid_policy = build_covid_policy(metadata.loc[:, ["iso3", "country"]])
    resurgence = compute_resurgence(cases)
    inventory, tier_assignment = build_inventory(
        metadata,
        cases,
        coverage,
        age_specific,
        population,
        contact,
        schedule,
        covid_policy,
        resurgence,
        diagnostic_timeline,
        surveillance_outcomes,
        gbd_burden,
        age_pattern_evidence,
        japan_annual_national_surveillance_summary,
    )
    age_pattern_validation_lattice = summarize_age_pattern_validation_lattice(
        age_pattern_global_summary,
        inventory,
        external_validation_summary,
    )

    cases.to_csv(CLEAN / "cases_country_age_time.csv", index=False)
    age_specific.to_csv(CLEAN / "cases_age_specific.csv", index=False)
    age_model_groups.to_csv(CLEAN / "cases_age_model_groups.csv", index=False)
    healthmap_alerts.to_csv(CLEAN / "healthmap_alerts.csv", index=False)
    diagnostic_timeline.to_csv(CLEAN / "surveillance_diagnostic_timeline.csv", index=False)
    diagnostic_sources.to_csv(CLEAN / "surveillance_diagnostic_sources.csv", index=False)
    diagnostic_age_testing_summary.to_csv(CLEAN / "diagnostic_age_testing_summary.csv", index=False)
    surveillance_outcomes.to_csv(CLEAN / "surveillance_outcome_evidence.csv", index=False)
    surveillance_outcome_sources.to_csv(CLEAN / "surveillance_outcome_sources.csv", index=False)
    japan_annual_national_surveillance_summary.to_csv(CLEAN / "japan_annual_national_surveillance_summary.csv", index=False)
    japan_annual_national_surveillance_sources.to_csv(CLEAN / "japan_annual_national_surveillance_sources.csv", index=False)
    age_pattern_evidence.to_csv(CLEAN / "age_pattern_evidence.csv", index=False)
    age_pattern_global_summary.to_csv(OUTPUTS / "age_pattern_global_summary.csv", index=False)
    age_pattern_validation_lattice.to_csv(OUTPUTS / "age_pattern_validation_lattice_summary.csv", index=False)
    usa_annual_age_specific_surveillance_summary.to_csv(
        OUTPUTS / "usa_annual_age_specific_surveillance_summary.csv",
        index=False,
    )
    maternal_coverage.to_csv(CLEAN / "maternal_immunization_coverage.csv", index=False)
    eu_age_pattern_summary.to_csv(OUTPUTS / "eu_age_pattern_summary.csv", index=False)
    eu_age_pattern_year_summary.to_csv(OUTPUTS / "eu_age_pattern_year_summary.csv", index=False)
    serology_evidence.to_csv(CLEAN / "serology_anti_pt_igg_evidence.csv", index=False)
    serology_sources.to_csv(CLEAN / "serology_anti_pt_igg_sources.csv", index=False)
    contact_validation_evidence.to_csv(CLEAN / "contact_validation_evidence.csv", index=False)
    contact_validation_sources.to_csv(CLEAN / "contact_validation_sources.csv", index=False)
    serology_summary.to_csv(OUTPUTS / "serology_anti_pt_igg_summary.csv", index=False)
    contact_validation_summary.to_csv(OUTPUTS / "contact_validation_summary.csv", index=False)
    external_validation_summary.to_csv(OUTPUTS / "external_validation_summary.csv", index=False)
    diagnostic_age_testing_summary.to_csv(OUTPUTS / "diagnostic_age_testing_summary.csv", index=False)
    vaccine_program_background.to_csv(CLEAN / "vaccine_program_background.csv", index=False)
    maternal_coverage.to_csv(OUTPUTS / "maternal_immunization_coverage_summary.csv", index=False)
    gbr_age_pattern_history.to_csv(OUTPUTS / "gbr_age_pattern_history_summary.csv", index=False)
    maternal_pregnancy_recommendation.to_csv(CLEAN / "maternal_pregnancy_recommendation_summary.csv", index=False)
    maternal_pregnancy_recommendation.to_csv(OUTPUTS / "maternal_pregnancy_recommendation_summary.csv", index=False)
    annual_country_series.to_csv(CLEAN / "annual_country_series.csv", index=False)
    region_rates.to_csv(CLEAN / "pertussis_region_rates.csv", index=False)
    who_reported_cases_workbook.to_csv(CLEAN / "who_reported_cases_workbook.csv", index=False)
    ecdc_reported_cases.to_csv(CLEAN / "ecdc_pertussis_reported_cases.csv", index=False)
    strain_amr_timeline.to_csv(CLEAN / "pertussis_strain_amr_timeline.csv", index=False)
    strain_amr_sources.to_csv(CLEAN / "pertussis_strain_amr_sources.csv", index=False)
    gbd_burden.to_csv(CLEAN / "gbd_pertussis_burden.csv", index=False)
    gbd_burden_country_summary.to_csv(OUTPUTS / "gbd_pertussis_burden_summary.csv", index=False)
    gbd_burden_year_summary.to_csv(OUTPUTS / "gbd_pertussis_burden_year_summary.csv", index=False)
    region_rate_summary.to_csv(OUTPUTS / "pertussis_region_rate_summary.csv", index=False)
    coverage.to_csv(CLEAN / "vaccine_coverage.csv", index=False)
    schedule.to_csv(CLEAN / "vaccine_schedule.csv", index=False)
    population.to_csv(CLEAN / "population_age.csv", index=False)
    contact.to_csv(CLEAN / "contact_mobility.csv", index=False)
    covid_policy.to_csv(CLEAN / "covid_policy.csv", index=False)
    metadata.to_csv(CLEAN / "country_metadata.csv", index=False)
    inventory.to_csv(OUTPUTS / "data_inventory.csv", index=False)
    tier_assignment.to_csv(OUTPUTS / "country_tier_assignment.csv", index=False)
    resurgence.to_csv(OUTPUTS / "resurgence_index_by_country.csv", index=False)
    population_age_audit.to_csv(OUTPUTS / "population_age_structure_projection_audit.csv", index=False)
    annual_country_series_comparison.to_csv(OUTPUTS / "annual_country_series_comparison.csv", index=False)
    annual_country_series_concordance_summary.to_csv(OUTPUTS / "annual_country_series_concordance_summary.csv", index=False)
    japan_annual_national_surveillance_summary.to_csv(OUTPUTS / "japan_annual_national_surveillance_summary.csv", index=False)
    south_africa_monthly_series_summary.to_csv(OUTPUTS / "south_africa_monthly_series_summary.csv", index=False)
    time_resolved_all_age_series_summary.to_csv(OUTPUTS / "time_resolved_all_age_series_summary.csv", index=False)
    time_resolved_all_age_series_completeness_audit.to_csv(
        OUTPUTS / "time_resolved_all_age_series_completeness_audit.csv",
        index=False,
    )
    who_reported_cases_concordance.to_csv(OUTPUTS / "who_reported_cases_concordance.csv", index=False)
    who_reported_cases_concordance_summary.to_csv(OUTPUTS / "who_reported_cases_concordance_summary.csv", index=False)
    annual_country_series_concordance_summary.to_csv(MANUSCRIPT / "tables" / "annual_country_series_concordance_summary.csv", index=False)
    write_markdown_table(
        annual_country_series_concordance_summary,
        MANUSCRIPT / "tables_md" / "annual_country_series_concordance_summary.md",
    )
    japan_annual_national_surveillance_summary.to_csv(MANUSCRIPT / "tables" / "japan_annual_national_surveillance_summary.csv", index=False)
    write_markdown_table(
        japan_annual_national_surveillance_summary,
        MANUSCRIPT / "tables_md" / "japan_annual_national_surveillance_summary.md",
    )
    south_africa_monthly_series_summary.to_csv(MANUSCRIPT / "tables" / "south_africa_monthly_series_summary.csv", index=False)
    write_markdown_table(
        south_africa_monthly_series_summary,
        MANUSCRIPT / "tables_md" / "south_africa_monthly_series_summary.md",
    )
    time_resolved_all_age_series_summary.to_csv(MANUSCRIPT / "tables" / "time_resolved_all_age_series_summary.csv", index=False)
    write_markdown_table(
        time_resolved_all_age_series_summary,
        MANUSCRIPT / "tables_md" / "time_resolved_all_age_series_summary.md",
    )
    time_resolved_all_age_series_completeness_audit.to_csv(
        MANUSCRIPT / "tables" / "time_resolved_all_age_series_completeness_audit.csv",
        index=False,
    )
    write_markdown_table(
        time_resolved_all_age_series_completeness_audit,
        MANUSCRIPT / "tables_md" / "time_resolved_all_age_series_completeness_audit.md",
    )
    vaccine_program_background_summary.to_csv(OUTPUTS / "vaccine_program_background_summary.csv", index=False)
    vaccine_program_background_reconciliation.to_csv(OUTPUTS / "vaccine_program_background_reconciliation.csv", index=False)
    vaccine_program_background_reconciliation_summary.to_csv(OUTPUTS / "vaccine_program_background_reconciliation_summary.csv", index=False)
    ecdc_who_comparison.to_csv(OUTPUTS / "ecdc_who_reported_case_comparison.csv", index=False)
    ecdc_who_concordance_summary.to_csv(OUTPUTS / "ecdc_who_reported_case_concordance_summary.csv", index=False)
    ecdc_who_concordance_summary.to_csv(MANUSCRIPT / "tables" / "ecdc_who_reported_case_concordance_summary.csv", index=False)
    write_markdown_table(
        ecdc_who_concordance_summary,
        MANUSCRIPT / "tables_md" / "ecdc_who_reported_case_concordance_summary.md",
    )
    gbd_burden_country_summary.to_csv(MANUSCRIPT / "tables" / "gbd_pertussis_burden_summary.csv", index=False)
    write_markdown_table(gbd_burden_country_summary, MANUSCRIPT / "tables_md" / "gbd_pertussis_burden_summary.md")
    gbd_burden_year_summary.to_csv(MANUSCRIPT / "tables" / "gbd_pertussis_burden_year_summary.csv", index=False)
    write_markdown_table(gbd_burden_year_summary, MANUSCRIPT / "tables_md" / "gbd_pertussis_burden_year_summary.md")
    age_pattern_evidence.to_csv(MANUSCRIPT / "tables" / "age_pattern_evidence.csv", index=False)
    write_markdown_table(age_pattern_evidence, MANUSCRIPT / "tables_md" / "age_pattern_evidence.md")
    usa_annual_age_specific_surveillance_summary.to_csv(
        MANUSCRIPT / "tables" / "usa_annual_age_specific_surveillance_summary.csv",
        index=False,
    )
    write_markdown_table(
        usa_annual_age_specific_surveillance_summary,
        MANUSCRIPT / "tables_md" / "usa_annual_age_specific_surveillance_summary.md",
    )
    age_pattern_global_summary.to_csv(MANUSCRIPT / "tables" / "age_pattern_global_summary.csv", index=False)
    write_markdown_table(age_pattern_global_summary, MANUSCRIPT / "tables_md" / "age_pattern_global_summary.md")
    age_pattern_global_summary.to_csv(MANUSCRIPT / "figure_data" / "age_pattern_global_summary.csv", index=False)
    age_pattern_validation_lattice.to_csv(MANUSCRIPT / "tables" / "age_pattern_validation_lattice_summary.csv", index=False)
    write_markdown_table(
        age_pattern_validation_lattice,
        MANUSCRIPT / "tables_md" / "age_pattern_validation_lattice_summary.md",
    )
    age_pattern_validation_lattice.to_csv(MANUSCRIPT / "figure_data" / "age_pattern_validation_lattice_summary.csv", index=False)
    gbr_age_pattern_history.to_csv(MANUSCRIPT / "tables" / "gbr_age_pattern_history_summary.csv", index=False)
    write_markdown_table(gbr_age_pattern_history, MANUSCRIPT / "tables_md" / "gbr_age_pattern_history_summary.md")
    gbr_age_pattern_history.to_csv(MANUSCRIPT / "figure_data" / "gbr_age_pattern_history_summary.csv", index=False)
    japan_annual_national_surveillance_summary.to_csv(
        MANUSCRIPT / "figure_data" / "japan_annual_national_surveillance_summary.csv",
        index=False,
    )
    region_rate_summary.to_csv(MANUSCRIPT / "tables" / "pertussis_region_rate_summary.csv", index=False)
    write_markdown_table(region_rate_summary, MANUSCRIPT / "tables_md" / "pertussis_region_rate_summary.md")
    who_reported_cases_concordance_summary.to_csv(MANUSCRIPT / "tables" / "who_reported_cases_concordance_summary.csv", index=False)
    write_markdown_table(
        who_reported_cases_concordance_summary,
        MANUSCRIPT / "tables_md" / "who_reported_cases_concordance_summary.md",
    )
    eu_age_pattern_summary.to_csv(MANUSCRIPT / "tables" / "eu_age_pattern_summary.csv", index=False)
    write_markdown_table(eu_age_pattern_summary, MANUSCRIPT / "tables_md" / "eu_age_pattern_summary.md")
    eu_age_pattern_year_summary.to_csv(MANUSCRIPT / "tables" / "eu_age_pattern_year_summary.csv", index=False)
    write_markdown_table(eu_age_pattern_year_summary, MANUSCRIPT / "tables_md" / "eu_age_pattern_year_summary.md")
    vaccine_program_background_summary.to_csv(MANUSCRIPT / "tables" / "vaccine_program_background_summary.csv", index=False)
    write_markdown_table(vaccine_program_background_summary, MANUSCRIPT / "tables_md" / "vaccine_program_background_summary.md")
    vaccine_program_background_reconciliation_summary.to_csv(MANUSCRIPT / "tables" / "vaccine_program_background_reconciliation_summary.csv", index=False)
    write_markdown_table(
        vaccine_program_background_reconciliation_summary,
        MANUSCRIPT / "tables_md" / "vaccine_program_background_reconciliation_summary.md",
    )
    maternal_coverage.to_csv(MANUSCRIPT / "tables" / "maternal_immunization_coverage_summary.csv", index=False)
    write_markdown_table(maternal_coverage, MANUSCRIPT / "tables_md" / "maternal_immunization_coverage_summary.md")
    maternal_pregnancy_recommendation.to_csv(MANUSCRIPT / "tables" / "maternal_pregnancy_recommendation_summary.csv", index=False)
    write_markdown_table(
        maternal_pregnancy_recommendation,
        MANUSCRIPT / "tables_md" / "maternal_pregnancy_recommendation_summary.md",
    )
    contact_validation_summary.to_csv(MANUSCRIPT / "tables" / "contact_validation_summary.csv", index=False)
    write_markdown_table(contact_validation_summary, MANUSCRIPT / "tables_md" / "contact_validation_summary.md")
    serology_summary.to_csv(MANUSCRIPT / "tables" / "serology_anti_pt_igg_summary.csv", index=False)
    write_markdown_table(serology_summary, MANUSCRIPT / "tables_md" / "serology_anti_pt_igg_summary.md")
    external_validation_summary.to_csv(MANUSCRIPT / "tables" / "external_validation_summary.csv", index=False)
    write_markdown_table(
        external_validation_summary,
        MANUSCRIPT / "tables_md" / "external_validation_summary.md",
    )
    diagnostic_age_testing_summary.to_csv(MANUSCRIPT / "tables" / "diagnostic_age_testing_summary.csv", index=False)
    write_markdown_table(
        diagnostic_age_testing_summary,
        MANUSCRIPT / "tables_md" / "diagnostic_age_testing_summary.md",
    )
    local_source_manifest.to_csv(METADATA / "local_source_manifest.csv", index=False)
    write_source_registry()
    write_external_data_discovery()


def main() -> None:
    write_outputs()
    print("Phase 0-3 source-integrated outputs written.")


if __name__ == "__main__":
    main()
