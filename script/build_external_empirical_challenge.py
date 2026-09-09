#!/usr/bin/env python3
"""Build an illustrative ecological triangulation of linked surveillance.

The four purposively selected 2024 resurgence episodes are interpreted with a
retrospective deterministic rule. The analysis asks whether the observed
ecological streams are inconsistent with a pure ascertainment-only explanation
if stream definitions and coverage are stable. It is neither external
validation of the simulated classifier nor an estimate of historical mechanism
shares, sensitivity, specificity or prospective accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2

from build_tiered_stream_evidence import load_exact_testing


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EXTRACTS = (
    ROOT
    / "data"
    / "raw"
    / "external_empirical_validation"
    / "official_country_year_extracts.csv"
)
UKHSA_WORKBOOK = (
    ROOT
    / "data"
    / "raw"
    / "external_empirical_validation"
    / "ukhsa_pertussis_england_2024_supplementary_tables.xlsx"
)
SEVERE_EVIDENCE = (
    ROOT / "data" / "raw" / "severe_outcomes" / "severe_outcome_age_time_evidence.csv"
)
CHINA_ANNUAL = (
    ROOT / "data" / "raw" / "time_resolved_cases" / "annual_country_series" / "china.csv"
)
POPULATION_AGE = ROOT / "data" / "clean" / "population_age.csv"

STREAM_OUTPUT = ROOT / "outputs" / "external_empirical_challenge_streams.csv"
COMPARISON_OUTPUT = ROOT / "outputs" / "external_empirical_challenge_comparisons.csv"
INTERPRETATION_OUTPUT = ROOT / "outputs" / "external_empirical_challenge_summary.csv"
MANUSCRIPT_FIGURE_DATA = (
    ROOT / "manuscript" / "figure_data" / "figure_8_external_empirical_challenge.csv"
)
MANUSCRIPT_TABLE = (
    ROOT / "manuscript" / "tables" / "external_empirical_challenge_summary.csv"
)
MANUSCRIPT_MARKDOWN = (
    ROOT / "manuscript" / "tables_md" / "external_empirical_challenge_summary.md"
)
MANUSCRIPT_COMPARISON_MARKDOWN = (
    ROOT
    / "manuscript"
    / "tables_md"
    / "external_empirical_challenge_comparisons.md"
)

YEARS = (2019, 2023, 2024)

# Population rates are restricted to streams whose archived geographic scope
# can be paired with the available national World Population Prospects
# denominator. France's network streams and England-only streams are therefore
# excluded rather than divided by national populations with different scopes.
RATE_DENOMINATOR_RULES = {
    ("BRA", "confirmed_cases"): "all ages",
    ("BRA", "all_age_admissions"): "all ages",
    ("BRA", "infant_admissions"): "younger than one year",
    ("BRA", "all_age_deaths"): "all ages",
    ("CHN", "reported_cases"): "all ages",
    ("CHN", "all_age_deaths"): "all ages",
}


def _integer_count(value: float) -> int:
    count = int(round(float(value)))
    if count < 0 or not np.isclose(float(value), count, rtol=0, atol=1e-9):
        raise ValueError(f"Expected an integer count, received {value}")
    return count


def _poisson_count_interval(value: float, alpha: float = 0.05) -> tuple[float, float]:
    """Garwood interval for a count under a Poisson sampling model."""

    if pd.isna(value):
        return np.nan, np.nan
    count = _integer_count(value)
    lower = 0.0 if count == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * count)
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (count + 1))
    return float(lower), float(upper)


def _conditional_poisson_ratio_interval(
    numerator: float,
    denominator: float,
    *,
    numerator_exposure: float = 1.0,
    denominator_exposure: float = 1.0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Exact conditional interval for a ratio of two independent Poisson rates."""

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or numerator_exposure <= 0
        or denominator_exposure <= 0
    ):
        return np.nan, np.nan
    count_1 = _integer_count(numerator)
    count_0 = _integer_count(denominator)
    if count_0 == 0:
        # The point ratio and its upper confidence limit are unbounded. Retain
        # the zero-to-positive change rather than reporting a misleading ratio.
        return np.nan, np.nan
    p_lower = 0.0 if count_1 == 0 else beta.ppf(alpha / 2, count_1, count_0 + 1)
    p_upper = beta.ppf(1 - alpha / 2, count_1 + 1, count_0)
    exposure_ratio = denominator_exposure / numerator_exposure
    lower = exposure_ratio * p_lower / (1 - p_lower) if p_lower < 1 else np.inf
    upper = exposure_ratio * p_upper / (1 - p_upper) if p_upper < 1 else np.inf
    return float(lower), float(upper)


def _binomial_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson interval, returned on the percentage scale."""

    if trials <= 0 or successes < 0 or successes > trials:
        return np.nan, np.nan
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta.ppf(
        1 - alpha / 2, successes + 1, trials - successes
    )
    return 100 * float(lower), 100 * float(upper)


def _row(
    *,
    iso3: str,
    country: str,
    year: int,
    pathway: str,
    metric: str,
    display_metric: str,
    value: float,
    unit: str,
    geographic_scope: str,
    source_id: str,
    source_url: str,
    extraction_date: str,
    notes: str,
) -> dict[str, object]:
    return {
        "iso3": iso3,
        "country": country,
        "year": int(year),
        "pathway": pathway,
        "metric": metric,
        "display_metric": display_metric,
        "value": float(value),
        "unit": unit,
        "geographic_scope": geographic_scope,
        "source_id": source_id,
        "source_url": source_url,
        "extraction_date": extraction_date,
        "notes": notes,
    }


def _exact_testing_rows(iso3: str) -> list[dict[str, object]]:
    testing = load_exact_testing()
    testing = testing.loc[testing.iso3.eq(iso3) & testing.year.isin(YEARS)]
    rows: list[dict[str, object]] = []
    specs = (
        [("submitted_samples", "submitted_samples", "Submitted samples", "count"),
         ("eligible_tests", "eligible_samples", "Eligible samples", "count"),
         ("positive_tests", "positive_samples", "Positive samples", "count"),
         ("positivity_percent", "test_positivity", "Test positivity", "percent")]
        if iso3 == "GBR" else
        [("eligible_tests", "pcr_tests", "PCR tests", "count"),
         ("positive_tests", "positive_pcr_tests", "Positive PCR tests", "count"),
         ("positivity_percent", "pcr_positivity", "PCR positivity", "percent")]
    )
    for _, source in testing.iterrows():
        for field, metric, display, unit in specs:
            row = _row(
                iso3=iso3, country=source.country, year=int(source.year),
                pathway="testing", metric=metric, display_metric=display,
                value=float(source[field]), unit=unit,
                geographic_scope=source.geographic_scope,
                source_id=source.source_id, source_url=source.source_url,
                extraction_date=source.extraction_date, notes=source.notes,
            )
            row.update(source_publication_date=source.source_publication_date,
                       data_vintage=source.data_vintage, count_basis=source.count_basis)
            rows.append(row)
    return rows


def read_england() -> list[dict[str, object]]:
    deaths = pd.read_excel(UKHSA_WORKBOOK, sheet_name="Table_5", header=2)
    testing = load_exact_testing()
    testing = testing.loc[testing.iso3.eq("GBR") & testing.year.isin(YEARS)]
    rows = _exact_testing_rows("GBR")
    for _, source in testing.iterrows():
        year = int(source.year)
        death_row = deaths.loc[pd.to_numeric(deaths["Year"], errors="coerce").eq(year)]
        if len(death_row) != 1:
            raise RuntimeError(f"Incomplete England deaths source for {year}")
        cases = _row(
            iso3="GBR", country="England", year=year, pathway="notification",
            metric="laboratory_confirmed_cases", display_metric="Laboratory-confirmed cases",
            value=source.confirmed_cases, unit="count",
            geographic_scope="England enhanced laboratory surveillance",
            source_id=source.confirmed_cases_source_id,
            source_url=source.confirmed_cases_source_url,
            extraction_date=source.extraction_date,
            notes="Distinct cases, each represented once; culture, PCR, serology or oral fluid. Cases are not the positivity numerator.",
        )
        cases["source_publication_date"] = source.confirmed_cases_source_publication_date
        rows.append(cases)
        rows.append(_row(
            iso3="GBR", country="England", year=year, pathway="severe",
            metric="infant_deaths", display_metric="Infant deaths",
            value=float(death_row.iloc[0]["Deaths"]), unit="count",
            geographic_scope="England enhanced surveillance",
            source_id="UKHSA_ENGLAND_2024_XLSX",
            source_url="https://www.gov.uk/government/publications/pertussis-laboratory-confirmed-cases-reported-in-england-2024",
            extraction_date="2026-08-06",
            notes="Deaths in infants younger than one year, assigned to confirmation year.",
        ))
    return rows


def read_france(official: pd.DataFrame) -> list[dict[str, object]]:
    rows = _exact_testing_rows("FRA")
    selected = official.loc[official["iso3"].eq("FRA")]
    for _, source in selected.iterrows():
        metric = str(source["metric"])
        display = {"infant_hospitalizations": "Infant hospitalizations",
                   "infant_deaths": "Infant deaths"}[metric]
        rows.append(_row(
            iso3="FRA", country="France", year=int(source["year"]), pathway="severe",
            metric=metric, display_metric=display, value=float(source["value"]),
            unit=str(source["unit"]), geographic_scope=str(source["geographic_scope"]),
            source_id=str(source["source_id"]), source_url=str(source["source_url"]),
            extraction_date=str(source["extraction_date"]), notes=str(source["notes"]),
        ))
    return rows


def read_brazil(official: pd.DataFrame, severe: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    official_cases = official.loc[official["iso3"].eq("BRA")]
    historical_cases = severe.loc[
        severe["iso3"].eq("BRA")
        & severe["outcome_type"].eq("reported_cases")
        & severe["age_group"].eq("all_ages")
        & severe["year"].eq(2019)
    ]
    case_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "year": historical_cases["year"],
                    "value": historical_cases["count"],
                    "source_id": historical_cases["source_id"],
                    "source_url": historical_cases["source_url"],
                    "extraction_date": historical_cases["extraction_date"],
                    "geographic_scope": historical_cases["geographic_scope"],
                    "notes": historical_cases["notes"],
                }
            ),
            official_cases[
                [
                    "year",
                    "value",
                    "source_id",
                    "source_url",
                    "extraction_date",
                    "geographic_scope",
                    "notes",
                ]
            ],
        ],
        ignore_index=True,
    )
    for _, source in case_rows.iterrows():
        rows.append(
            _row(
                iso3="BRA",
                country="Brazil",
                year=int(source["year"]),
                pathway="notification",
                metric="confirmed_cases",
                display_metric="Confirmed cases",
                value=float(source["value"]),
                unit="count",
                geographic_scope=str(source["geographic_scope"]),
                source_id=str(source["source_id"]),
                source_url=str(source["source_url"]),
                extraction_date=str(source["extraction_date"]),
                notes=str(source["notes"]),
            )
        )

    selections = {
        "all_age_admissions": ("hospitalizations", "all_ages", "All-age admissions"),
        "infant_admissions": ("hospitalizations", "<1 year", "Infant admissions"),
        "all_age_deaths": ("deaths", "all_ages", "All-age deaths"),
    }
    for metric, (outcome, age, display) in selections.items():
        chosen = severe.loc[
            severe["iso3"].eq("BRA")
            & severe["outcome_type"].eq(outcome)
            & severe["age_group"].eq(age)
            & severe["year"].isin(YEARS)
        ].copy()
        if metric == "all_age_deaths":
            # DATASUS omits zero-valued rows; the 2023 zero is restored only
            # after verifying complete annual coverage around that year.
            present_years = set(chosen["year"].astype(int))
            if 2023 not in present_years:
                template = chosen.loc[chosen["year"].eq(2024)].iloc[0].copy()
                template["year"] = 2023
                template["count"] = 0
                template["notes"] = (
                    "Zero restored from an omitted zero-valued DATASUS row; "
                    "the annual extraction covered 2023."
                )
                chosen = pd.concat([chosen, template.to_frame().T], ignore_index=True)
        for _, source in chosen.iterrows():
            rows.append(
                _row(
                    iso3="BRA",
                    country="Brazil",
                    year=int(source["year"]),
                    pathway="severe",
                    metric=metric,
                    display_metric=display,
                    value=float(source["count"]),
                    unit="count",
                    geographic_scope=str(source["geographic_scope"]),
                    source_id=str(source["source_id"]),
                    source_url=str(source["source_url"]),
                    extraction_date=str(source["extraction_date"]),
                    notes=str(source["notes"]),
                )
            )
    return rows


def read_china(official: pd.DataFrame) -> list[dict[str, object]]:
    historical = pd.read_csv(CHINA_ANNUAL)
    row_2019 = historical.loc[historical["year"].eq(2019)]
    if len(row_2019) != 1:
        raise RuntimeError("Expected one China 2019 annual case row")
    rows = [
        _row(
            iso3="CHN",
            country="China",
            year=2019,
            pathway="notification",
            metric="reported_cases",
            display_metric="Reported cases",
            value=float(row_2019.iloc[0]["cases"]),
            unit="count",
            geographic_scope="China national notifiable-disease surveillance",
            source_id="CHN_ANNUAL_COUNTRY_SERIES_2025_12_02",
            source_url=(
                "https://www.ndcpa.gov.cn/jbkzzx/c100016/common/content/"
                "content_1998217472895258624.html"
            ),
            extraction_date="2026-08-06",
            notes="Archived annual-country series used only as the 2019 comparator.",
        )
    ]
    for _, source in official.loc[official["iso3"].eq("CHN")].iterrows():
        metric = str(source["metric"])
        rows.append(
            _row(
                iso3="CHN",
                country="China",
                year=int(source["year"]),
                pathway="notification" if metric == "reported_cases" else "severe",
                metric=metric,
                display_metric=(
                    "Reported cases" if metric == "reported_cases" else "All-age deaths"
                ),
                value=float(source["value"]),
                unit=str(source["unit"]),
                geographic_scope=str(source["geographic_scope"]),
                source_id=str(source["source_id"]),
                source_url=str(source["source_url"]),
                extraction_date=str(source["extraction_date"]),
                notes=str(source["notes"]),
            )
        )
    return rows


def build_comparisons(streams: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["iso3", "country", "pathway", "metric", "display_metric", "unit"]
    for key, group in streams.groupby(keys, sort=False, dropna=False):
        values = group.set_index("year")["value"].to_dict()
        value_2024 = values.get(2024, np.nan)
        value_2023 = values.get(2023, np.nan)
        value_2019 = values.get(2019, np.nan)

        def ratio(numerator: float, denominator: float) -> float:
            if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
                return np.nan
            return float(numerator / denominator)

        record = {
                "iso3": key[0],
                "country": key[1],
                "pathway": key[2],
                "metric": key[3],
                "display_metric": key[4],
                "unit": key[5],
                "value_2019": value_2019,
                "value_2023": value_2023,
                "value_2024": value_2024,
                "ratio_2024_vs_2019": ratio(value_2024, value_2019),
                "ratio_2024_vs_2023": ratio(value_2024, value_2023),
                "absolute_change_2024_vs_2023": (
                    value_2024 - value_2023
                    if pd.notna(value_2024) and pd.notna(value_2023)
                    else np.nan
                ),
                "percentage_point_change_2024_vs_2023": (
                    value_2024 - value_2023
                    if key[5] == "percent"
                    and pd.notna(value_2024)
                    and pd.notna(value_2023)
                    else np.nan
                ),
                "zero_2023_to_positive_2024": bool(
                    pd.notna(value_2023)
                    and pd.notna(value_2024)
                    and value_2023 == 0
                    and value_2024 > 0
                ),
                "source_ids": ";".join(sorted(set(group["source_id"].astype(str)))),
                "source_urls": ";".join(sorted(set(group["source_url"].astype(str)))),
                "scope_note": "; ".join(sorted(set(group["geographic_scope"].astype(str)))),
            }
        if key[5] == "count":
            for year, value in [(2019, value_2019), (2023, value_2023), (2024, value_2024)]:
                lower, upper = _poisson_count_interval(value)
                record[f"value_{year}_poisson_lower_95"] = lower
                record[f"value_{year}_poisson_upper_95"] = upper
            for comparator_year, comparator_value in [(2019, value_2019), (2023, value_2023)]:
                lower, upper = _conditional_poisson_ratio_interval(
                    value_2024, comparator_value
                )
                record[f"ratio_2024_vs_{comparator_year}_lower_95"] = lower
                record[f"ratio_2024_vs_{comparator_year}_upper_95"] = upper
        records.append(record)
    comparisons = pd.DataFrame.from_records(records)
    comparisons = add_population_rates(comparisons)
    comparisons = add_positivity_intervals(comparisons)
    return comparisons


def _population_denominator(
    population: pd.DataFrame,
    iso3: str,
    year: int,
    denominator_type: str,
) -> float:
    selected = population.loc[
        population["iso3"].eq(iso3) & population["year"].eq(year)
    ]
    if selected.empty:
        return np.nan
    if denominator_type == "all ages":
        values = selected["total_population"].dropna().unique()
        if len(values) != 1:
            raise RuntimeError(f"Expected one total population for {iso3} {year}")
        return float(values[0])
    if denominator_type == "younger than one year":
        return float(
            selected.loc[
                selected["age_group"].isin(["infant_0_2m", "infant_3_11m"]),
                "population",
            ].sum()
        )
    raise ValueError(f"Unknown population denominator type: {denominator_type}")


def add_population_rates(comparisons: pd.DataFrame) -> pd.DataFrame:
    population = pd.read_csv(POPULATION_AGE)
    enriched = comparisons.copy()
    enriched["population_rate_available"] = False
    enriched["population_denominator_type"] = pd.NA
    enriched["population_rate_source"] = pd.NA
    enriched["population_rate_boundary"] = (
        "Not estimable with the archived scope-matched national population denominator."
    )
    for year in YEARS:
        for suffix in [
            f"population_denominator_{year}",
            f"rate_per_100000_{year}",
            f"rate_per_100000_{year}_lower_95",
            f"rate_per_100000_{year}_upper_95",
        ]:
            enriched[suffix] = np.nan
    for comparator_year in [2019, 2023]:
        for suffix in [
            f"rate_ratio_2024_vs_{comparator_year}",
            f"rate_ratio_2024_vs_{comparator_year}_lower_95",
            f"rate_ratio_2024_vs_{comparator_year}_upper_95",
        ]:
            enriched[suffix] = np.nan

    for index, row in enriched.iterrows():
        denominator_type = RATE_DENOMINATOR_RULES.get((row["iso3"], row["metric"]))
        if denominator_type is None or row["unit"] != "count":
            continue
        enriched.at[index, "population_rate_available"] = True
        enriched.at[index, "population_denominator_type"] = denominator_type
        enriched.at[index, "population_rate_source"] = "UN World Population Prospects 2024"
        enriched.at[index, "population_rate_boundary"] = (
            "Crude population rate; surveillance coverage, coding and case definitions "
            "remain those of the source system."
        )
        denominators: dict[int, float] = {}
        for year in YEARS:
            value = row.get(f"value_{year}", np.nan)
            denominator = _population_denominator(
                population, str(row["iso3"]), year, denominator_type
            )
            denominators[year] = denominator
            enriched.at[index, f"population_denominator_{year}"] = denominator
            if pd.isna(value) or pd.isna(denominator) or denominator <= 0:
                continue
            lower, upper = _poisson_count_interval(value)
            enriched.at[index, f"rate_per_100000_{year}"] = 100000 * value / denominator
            enriched.at[index, f"rate_per_100000_{year}_lower_95"] = 100000 * lower / denominator
            enriched.at[index, f"rate_per_100000_{year}_upper_95"] = 100000 * upper / denominator
        for comparator_year in [2019, 2023]:
            value_2024 = row.get("value_2024", np.nan)
            comparator_value = row.get(f"value_{comparator_year}", np.nan)
            exposure_2024 = denominators.get(2024, np.nan)
            comparator_exposure = denominators.get(comparator_year, np.nan)
            if (
                pd.isna(value_2024)
                or pd.isna(comparator_value)
                or pd.isna(exposure_2024)
                or pd.isna(comparator_exposure)
                or comparator_value <= 0
            ):
                continue
            rate_ratio = (value_2024 / exposure_2024) / (
                comparator_value / comparator_exposure
            )
            lower, upper = _conditional_poisson_ratio_interval(
                value_2024,
                comparator_value,
                numerator_exposure=exposure_2024,
                denominator_exposure=comparator_exposure,
            )
            enriched.at[index, f"rate_ratio_2024_vs_{comparator_year}"] = rate_ratio
            enriched.at[index, f"rate_ratio_2024_vs_{comparator_year}_lower_95"] = lower
            enriched.at[index, f"rate_ratio_2024_vs_{comparator_year}_upper_95"] = upper
    return enriched


def add_positivity_intervals(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Use source integer successes/trials, never rounded percentages or case reports."""
    enriched = comparisons.copy()
    for year in YEARS:
        for field in ["positivity_trials", "positivity_positive_count"]:
            enriched[f"{field}_{year}"] = np.nan
        enriched[f"value_{year}_binomial_lower_95"] = np.nan
        enriched[f"value_{year}_binomial_upper_95"] = np.nan
    enriched["positivity_interval_method"] = pd.Series(dtype=object)
    counts = {
        ("FRA", "pcr_positivity"): ("pcr_tests", "positive_pcr_tests"),
        ("GBR", "test_positivity"): ("eligible_samples", "positive_samples"),
    }
    for index, row in enriched.iterrows():
        count_metrics = counts.get((row["iso3"], row["metric"]))
        if count_metrics is None:
            continue
        selected = []
        for metric in count_metrics:
            source = enriched.loc[enriched.iso3.eq(row.iso3) & enriched.metric.eq(metric)]
            if len(source) != 1:
                raise RuntimeError(f"Missing exact positivity count {row.iso3} {metric}")
            if source.iloc[0].scope_note != row.scope_note:
                raise ValueError("Positivity numerator and denominator must share the same scope")
            selected.append(source.iloc[0])
        denominator, numerator = selected
        for year in YEARS:
            if pd.isna(row[f"value_{year}"]):
                continue
            trials = _integer_count(denominator[f"value_{year}"])
            successes = _integer_count(numerator[f"value_{year}"])
            if not np.isclose(100 * successes / trials, row[f"value_{year}"], rtol=1e-12):
                raise ValueError("Positivity must equal exact positive tests / eligible tests")
            lower, upper = _binomial_interval(successes, trials)
            enriched.at[index, f"positivity_trials_{year}"] = trials
            enriched.at[index, f"positivity_positive_count_{year}"] = successes
            enriched.at[index, f"value_{year}_binomial_lower_95"] = lower
            enriched.at[index, f"value_{year}_binomial_upper_95"] = upper
        enriched.at[index, "positivity_interval_method"] = "Clopper-Pearson from exact source counts; independent-binomial sampling model"
    return enriched


def _metric(comparisons: pd.DataFrame, iso3: str, metric: str) -> pd.Series | None:
    selected = comparisons.loc[
        comparisons["iso3"].eq(iso3) & comparisons["metric"].eq(metric)
    ]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise RuntimeError(f"Expected one comparison row for {iso3} {metric}")
    return selected.iloc[0]


def _rose(row: pd.Series | None) -> bool:
    if row is None:
        return False
    ratio = row["ratio_2024_vs_2023"]
    return bool((pd.notna(ratio) and float(ratio) > 1) or row["zero_2023_to_positive_2024"])


def build_interpretations(comparisons: pd.DataFrame) -> pd.DataFrame:
    country_specs = {
        "FRA": {
            "testing_metric": "pcr_tests",
            "positivity_metric": "pcr_positivity",
            "hospital_metric": "infant_hospitalizations",
            "death_metric": "infant_deaths",
            "interpretation": "burden-linked with concurrent testing expansion",
            "scope": "network testing, positivity and infant severe outcomes",
            "boundary": "3Labos and RENACOQ are networks rather than an individual-level linked national cohort.",
        },
        "GBR": {
            "testing_metric": "eligible_samples",
            "positivity_metric": "test_positivity",
            "hospital_metric": None,
            "death_metric": "infant_deaths",
            "interpretation": "burden-linked with concurrent testing expansion",
            "scope": "testing, positivity and infant mortality",
            "boundary": "Eligible serum, oral-fluid and PCR samples form the positivity denominator; submitted samples, positive samples and distinct confirmed cases are separate counts. Sampling composition and assay performance may change.",
        },
        "BRA": {
            "testing_metric": None,
            "positivity_metric": None,
            "hospital_metric": "all_age_admissions",
            "death_metric": "all_age_deaths",
            "interpretation": "burden-linked; ascertainment contribution unresolved",
            "scope": "notifications, public-system admissions and mortality",
            "boundary": "SINAN, SIH/SUS and SIM are separate systems and SIH/SUS excludes private admissions.",
        },
        "CHN": {
            "testing_metric": None,
            "positivity_metric": None,
            "hospital_metric": None,
            "death_metric": "all_age_deaths",
            "interpretation": "within-system severity descriptor; no independent burden triangulation",
            "scope": "all-age notifications and deaths from the same reporting channel",
            "boundary": "Cases and deaths share the national notifiable-disease system; neither an age-specific severe-outcome series nor a comparable testing denominator was located.",
        },
    }
    records: list[dict[str, object]] = []
    for iso3, spec in country_specs.items():
        country = str(comparisons.loc[comparisons["iso3"].eq(iso3), "country"].iloc[0])
        testing = _metric(comparisons, iso3, spec["testing_metric"]) if spec["testing_metric"] else None
        positivity = (
            _metric(comparisons, iso3, spec["positivity_metric"])
            if spec["positivity_metric"]
            else None
        )
        hospital = (
            _metric(comparisons, iso3, spec["hospital_metric"])
            if spec["hospital_metric"]
            else None
        )
        death = _metric(comparisons, iso3, spec["death_metric"]) if spec["death_metric"] else None
        testing_rise = _rose(testing)
        positivity_rise = _rose(positivity)
        hospital_rise = _rose(hospital)
        death_rise = _rose(death)
        # China's all-age deaths share the same notification system as its case
        # counts and therefore do not constitute an independent burden-linked
        # stream. Other retained severe-outcome rows are cross-system or
        # age-specific burden-proximal observations.
        burden_signal = positivity_rise or hospital_rise or (death_rise and iso3 != "CHN")
        records.append(
            {
                "iso3": iso3,
                "country": country,
                "testing_volume_available": testing is not None,
                "positivity_available": positivity is not None,
                "hospitalization_available": hospital is not None,
                "death_available": death is not None,
                "testing_volume_increased": testing_rise if testing is not None else pd.NA,
                "positivity_increased": positivity_rise if positivity is not None else pd.NA,
                "hospitalizations_increased": hospital_rise if hospital is not None else pd.NA,
                "deaths_increased": death_rise if death is not None else pd.NA,
                "burden_linked_signal": burden_signal,
                "inconsistent_with_pure_ascertainment_only_under_stable_stream_definitions": burden_signal,
                "ecological_evidence_scope": spec["scope"],
                "rule_based_interpretation": spec["interpretation"],
                "event_selection_basis": (
                    "Flow: identify an authority-described 2024 resurgence; search official "
                    "sources for a 2023–24 auxiliary comparison; retain episodes with at least "
                    "one retrievable stream and preserve source scope. The eligible universe "
                    "and exclusion counts were not enumerated."
                ),
                "rule_timing": (
                    "Retrospective rule specified for this analysis; not prospectively "
                    "registered and not documented as fixed before source inspection."
                ),
                "negative_control_status": (
                    "No negative-control year or disease was included; 2019 is a historical "
                    "comparator, not a negative control."
                ),
                "detection_change_comparator_status": (
                    "No event known to be dominated by a detection-policy change was included."
                ),
                "interpretation_boundary": spec["boundary"],
            }
        )
    return pd.DataFrame.from_records(records)


def markdown_table(interpretations: pd.DataFrame) -> str:
    display = interpretations[
        [
            "country",
            "ecological_evidence_scope",
            "rule_based_interpretation",
            "event_selection_basis",
            "rule_timing",
            "negative_control_status",
            "detection_change_comparator_status",
            "interpretation_boundary",
        ]
    ].copy()
    display.columns = [
        "Setting",
        "Ecological streams",
        "Rule-based interpretation",
        "Event selection",
        "Rule timing",
        "Negative controls",
        "Detection-change comparator",
        "Boundary",
    ]
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[column]).replace("|", "\\|") for column in display.columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs(value) >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def markdown_comparisons(comparisons: pd.DataFrame) -> str:
    lines = [
        "| Setting | Observed stream | 2023 | 2024 | 2024/2023 change (95% conditional interval) | Population-rate context | Scope boundary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in comparisons.sort_values(["country", "pathway", "metric"]).iterrows():
        if row["unit"] == "percent":
            value_2023 = (
                f"{_format_number(row['value_2023'])}% "
                f"({_format_number(row['value_2023_binomial_lower_95'])}–"
                f"{_format_number(row['value_2023_binomial_upper_95'])})"
            )
            value_2024 = (
                f"{_format_number(row['value_2024'])}% "
                f"({_format_number(row['value_2024_binomial_lower_95'])}–"
                f"{_format_number(row['value_2024_binomial_upper_95'])})"
            )
            change = (
                f"{_format_number(row['percentage_point_change_2024_vs_2023'])} percentage points; "
                "Clopper–Pearson intervals use exact positive and eligible test counts"
            )
            rate_context = "Not applicable to a positivity proportion."
        else:
            value_2023 = (
                f"{_format_number(row['value_2023'])} "
                f"({_format_number(row['value_2023_poisson_lower_95'])}–"
                f"{_format_number(row['value_2023_poisson_upper_95'])})"
            )
            value_2024 = (
                f"{_format_number(row['value_2024'])} "
                f"({_format_number(row['value_2024_poisson_lower_95'])}–"
                f"{_format_number(row['value_2024_poisson_upper_95'])})"
            )
            if bool(row["zero_2023_to_positive_2024"]):
                change = f"0→{_format_number(row['value_2024'])}; ratio undefined"
            elif pd.notna(row["ratio_2024_vs_2023"]):
                change = (
                    f"{_format_number(row['ratio_2024_vs_2023'])}× "
                    f"({_format_number(row['ratio_2024_vs_2023_lower_95'])}–"
                    f"{_format_number(row['ratio_2024_vs_2023_upper_95'])})"
                )
            else:
                change = "Not estimable"
            if bool(row["population_rate_available"]) and pd.notna(
                row["rate_per_100000_2023"]
            ):
                rate_context = (
                    f"{_format_number(row['rate_per_100000_2023'])}→"
                    f"{_format_number(row['rate_per_100000_2024'])} per 100,000; "
                )
                if pd.notna(row["rate_ratio_2024_vs_2023"]):
                    rate_context += (
                        f"rate ratio {_format_number(row['rate_ratio_2024_vs_2023'])} "
                        f"({_format_number(row['rate_ratio_2024_vs_2023_lower_95'])}–"
                        f"{_format_number(row['rate_ratio_2024_vs_2023_upper_95'])})"
                    )
                else:
                    rate_context += "rate ratio undefined"
            else:
                rate_context = str(row["population_rate_boundary"])
        fields = [
            row["country"],
            row["display_metric"],
            value_2023,
            value_2024,
            change,
            rate_context,
            row["scope_note"],
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in fields) + " |")
    return "\n".join(lines) + "\n"


def validate(
    streams: pd.DataFrame,
    comparisons: pd.DataFrame,
    interpretations: pd.DataFrame,
) -> None:
    if streams.duplicated(["iso3", "year", "metric"]).any():
        raise RuntimeError("Duplicate country-year-metric rows in empirical streams")
    expected = {
        ("FRA", "pcr_tests"): (13852.0, 188539.0),
        ("FRA", "positive_pcr_tests"): (518.0, 38847.0),
        ("FRA", "infant_hospitalizations"): (41.0, 537.0),
        ("GBR", "submitted_samples"): (9268.0, 36472.0),
        ("GBR", "eligible_samples"): (9252.0, 36274.0),
        ("GBR", "positive_samples"): (767.0, 13519.0),
        ("GBR", "laboratory_confirmed_cases"): (857.0, 14879.0),
        ("GBR", "infant_deaths"): (1.0, 11.0),
        ("BRA", "confirmed_cases"): (216.0, 7440.0),
        ("BRA", "all_age_admissions"): (238.0, 730.0),
        ("CHN", "reported_cases"): (41124.0, 476690.0),
        ("CHN", "all_age_deaths"): (5.0, 30.0),
    }
    for (iso3, metric), values in expected.items():
        row = _metric(comparisons, iso3, metric)
        if row is None:
            raise RuntimeError(f"Missing expected comparison {iso3} {metric}")
        observed = (float(row["value_2023"]), float(row["value_2024"]))
        if observed != values:
            raise RuntimeError(f"Unexpected {iso3} {metric}: {observed} != {values}")
    rule_column = (
        "inconsistent_with_pure_ascertainment_only_under_stable_stream_definitions"
    )
    expected_rule = {"FRA": True, "GBR": True, "BRA": True, "CHN": False}
    observed_rule = interpretations.set_index("iso3")[rule_column].to_dict()
    if observed_rule != expected_rule:
        raise RuntimeError(f"Unexpected bounded interpretations: {observed_rule}")
    if interpretations["negative_control_status"].str.startswith("No ").sum() != 4:
        raise RuntimeError("Negative-control absence must be explicit for every episode")


def main() -> None:
    official = pd.read_csv(OFFICIAL_EXTRACTS)
    severe = pd.read_csv(SEVERE_EVIDENCE)
    severe["year"] = pd.to_numeric(severe["year"], errors="raise").astype(int)
    rows = []
    rows.extend(read_france(official))
    rows.extend(read_england())
    rows.extend(read_brazil(official, severe))
    rows.extend(read_china(official))
    streams = pd.DataFrame.from_records(rows).sort_values(
        ["country", "pathway", "metric", "year"]
    )
    comparisons = build_comparisons(streams).sort_values(
        ["country", "pathway", "metric"]
    )
    interpretations = build_interpretations(comparisons)
    validate(streams, comparisons, interpretations)

    for path in [
        STREAM_OUTPUT,
        COMPARISON_OUTPUT,
        INTERPRETATION_OUTPUT,
        MANUSCRIPT_FIGURE_DATA,
        MANUSCRIPT_TABLE,
        MANUSCRIPT_MARKDOWN,
        MANUSCRIPT_COMPARISON_MARKDOWN,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    streams.to_csv(STREAM_OUTPUT, index=False)
    comparisons.to_csv(COMPARISON_OUTPUT, index=False)
    interpretations.to_csv(INTERPRETATION_OUTPUT, index=False)
    comparisons.to_csv(MANUSCRIPT_FIGURE_DATA, index=False)
    interpretations.to_csv(MANUSCRIPT_TABLE, index=False)
    MANUSCRIPT_MARKDOWN.write_text(markdown_table(interpretations), encoding="utf-8")
    MANUSCRIPT_COMPARISON_MARKDOWN.write_text(
        markdown_comparisons(comparisons), encoding="utf-8"
    )
    print(
        f"Wrote {len(streams)} empirical stream rows, {len(comparisons)} comparisons "
        f"and {len(interpretations)} rule-based ecological interpretations."
    )


if __name__ == "__main__":
    main()
