#!/usr/bin/env python3
"""Build comparator-sensitivity results for the four empirical episodes.

The analysis is descriptive. It compares each 2024 value with the adjacent
2023 value, the last pre-pandemic year (2019), the arithmetic mean of a
prespecified same-scope pre-pandemic window, and the maximum in that window.
Aggregate pre-pandemic comparators are emitted only when at least three annual
observations are available. Missing comparators remain explicit rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_tiered_stream_evidence import load_exact_testing


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_STREAMS = ROOT / "outputs" / "external_empirical_challenge_streams.csv"
SEVERE = ROOT / "data" / "raw" / "severe_outcomes" / "severe_outcome_age_time_evidence.csv"
SURVEILLANCE = ROOT / "data" / "raw" / "severe_outcomes" / "surveillance_outcome_evidence.csv"
CHINA_ANNUAL = ROOT / "data" / "raw" / "time_resolved_cases" / "annual_country_series" / "china.csv"

OUTPUT = ROOT / "outputs" / "empirical_baseline_sensitivity.csv"
FIGURE_DATA = ROOT / "manuscript" / "figure_data" / "supplement_empirical_baseline_sensitivity.csv"
TABLE = ROOT / "manuscript" / "tables" / "empirical_baseline_sensitivity.csv"
MARKDOWN = ROOT / "manuscript" / "tables_md" / "empirical_baseline_sensitivity.md"

COMPARATOR_ORDER = {
    "adjacent_2023": 0,
    "last_prepandemic_2019": 1,
    "prepandemic_mean": 2,
    "prepandemic_maximum": 3,
}


def _add_series(
    records: list[dict[str, object]],
    *,
    iso3: str,
    country: str,
    pathway: str,
    metric: str,
    display_metric: str,
    unit: str,
    values: dict[int, float],
    pre_start: int,
    pre_end: int = 2019,
    source_scope: str,
    source_ids: str,
    evidence_role: str,
    comparison_note: str,
) -> None:
    if 2024 not in values or not np.isfinite(values[2024]):
        raise ValueError(f"Missing finite 2024 value for {iso3} {metric}")
    value_2024 = float(values[2024])
    pre_values = {
        year: float(value)
        for year, value in values.items()
        if pre_start <= year <= pre_end and np.isfinite(value)
    }
    comparator_specs: list[tuple[str, str, int, int, list[float]]] = [
        ("adjacent_2023", "2023", 2023, 2023, [values[2023]] if 2023 in values else []),
        (
            "last_prepandemic_2019",
            "2019",
            2019,
            2019,
            [values[2019]] if 2019 in values else [],
        ),
    ]
    if len(pre_values) >= 3:
        comparator_specs.extend(
            [
                (
                    "prepandemic_mean",
                    f"{pre_start}–{pre_end} mean",
                    pre_start,
                    pre_end,
                    list(pre_values.values()),
                ),
                (
                    "prepandemic_maximum",
                    f"{pre_start}–{pre_end} maximum",
                    pre_start,
                    pre_end,
                    [max(pre_values.values())],
                ),
            ]
        )

    for comparator_type, label, start, end, comparator_values in comparator_specs:
        if comparator_type == "prepandemic_mean" and comparator_values:
            baseline = float(np.mean(comparator_values))
            n_years = len(comparator_values)
        elif comparator_values:
            baseline = float(comparator_values[0])
            n_years = len(pre_values) if comparator_type == "prepandemic_maximum" else 1
        else:
            baseline = np.nan
            n_years = 0
        ratio = value_2024 / baseline if np.isfinite(baseline) and baseline > 0 else np.nan
        status = "available" if np.isfinite(ratio) else "unavailable"
        records.append(
            {
                "iso3": iso3,
                "country": country,
                "pathway": pathway,
                "metric": metric,
                "display_metric": display_metric,
                "unit": unit,
                "value_2024": value_2024,
                "comparator_type": comparator_type,
                "comparator_label": label,
                "baseline_start_year": start,
                "baseline_end_year": end,
                "n_baseline_years": n_years,
                "baseline_value": baseline,
                "ratio_2024_to_baseline": ratio,
                "comparison_status": status,
                "source_scope": source_scope,
                "source_ids": source_ids,
                "evidence_role": evidence_role,
                "comparison_note": comparison_note,
            }
        )


def _values(frame: pd.DataFrame, year: str, value: str) -> dict[int, float]:
    selected = frame[[year, value]].dropna().copy()
    selected[year] = selected[year].astype(int)
    if selected[year].duplicated().any():
        raise ValueError(f"Duplicate years in {value}: {selected.loc[selected[year].duplicated(), year].tolist()}")
    return dict(zip(selected[year], selected[value].astype(float), strict=True))


def _fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|")


def _write_markdown(frame: pd.DataFrame) -> None:
    columns = [
        "Country",
        "Stream",
        "2024 value",
        "Comparator",
        "Baseline value",
        "2024/baseline ratio",
        "Status",
        "Evidence role",
    ]
    display = frame.rename(
        columns={
            "country": "Country",
            "display_metric": "Stream",
            "value_2024": "2024 value",
            "comparator_label": "Comparator",
            "baseline_value": "Baseline value",
            "ratio_2024_to_baseline": "2024/baseline ratio",
            "comparison_status": "Status",
            "evidence_role": "Evidence role",
        }
    )[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    testing = load_exact_testing().rename(columns={
        "eligible_tests": "testing_denominator",
        "positivity_percent": "test_positivity_percent",
        "source_id": "direct_testing_series_source_ids",
    })
    external = pd.read_csv(EXTERNAL_STREAMS)
    severe = pd.read_csv(SEVERE)
    surveillance = pd.read_csv(SURVEILLANCE)
    china = pd.read_csv(CHINA_ANNUAL)
    records: list[dict[str, object]] = []

    for iso3, country, pre_start, scope in [
        ("FRA", "France", 2017, "France 3Labos PCR network"),
        ("GBR", "England", 2019, "England UKHSA positivity-eligible sample surveillance"),
    ]:
        selected = testing.loc[testing["iso3"].eq(iso3)]
        for metric, label, unit in [
            ("testing_denominator", "Tests eligible for positivity", "count"),
            ("test_positivity_percent", "Test positivity", "percent"),
        ]:
            _add_series(
                records,
                iso3=iso3,
                country=country,
                pathway="testing",
                metric=metric,
                display_metric=label,
                unit=unit,
                values=_values(selected, "year", metric),
                pre_start=pre_start,
                source_scope=scope,
                source_ids=";".join(sorted(selected["direct_testing_series_source_ids"].dropna().astype(str).unique())),
                evidence_role="same-system testing–positivity decomposition",
                comparison_note=(
                    "Aggregate pre-pandemic comparators require at least three annual observations; "
                    "England testing therefore has only 2019 and adjacent-year comparisons. "
                    "Eligible tests and exact source positives define positivity; submissions and confirmed cases are separate units."
                ),
            )

    france = external.loc[external["iso3"].eq("FRA") & external["metric"].isin(
        ["infant_hospitalizations", "infant_deaths"]
    )]
    for metric, label in [
        ("infant_hospitalizations", "Infant hospitalisations"),
        ("infant_deaths", "Infant deaths"),
    ]:
        chosen = france.loc[france["metric"].eq(metric)]
        _add_series(
            records,
            iso3="FRA",
            country="France",
            pathway="severe",
            metric=metric,
            display_metric=label,
            unit="count",
            values=_values(chosen, "year", "value"),
            pre_start=2019,
            source_scope=str(chosen["geographic_scope"].iloc[0]),
            source_ids=";".join(sorted(chosen["source_id"].astype(str).unique())),
            evidence_role="burden-proximal adjacent-year descriptor",
            comparison_note="Only 2023 and 2024 annual values were structured in this extraction. The official source displays longer history; compatibility across the 2016 RENACOQ definition change has not been established here.",
        )

    england_external = external.loc[
        external["iso3"].eq("GBR") & external["metric"].eq("infant_deaths")
    ]
    england_history = surveillance.loc[
        surveillance["iso3"].eq("GBR")
        & surveillance["source_ids"].eq("UKHSA_ENGLAND_2024_XLSX")
        & surveillance["year"].between(2001, 2019)
    ]
    england_values = _values(england_history, "year", "infant_deaths")
    england_values.update(_values(england_external, "year", "value"))
    _add_series(
        records,
        iso3="GBR",
        country="England",
        pathway="severe",
        metric="infant_deaths",
        display_metric="Infant deaths",
        unit="count",
        values=england_values,
        pre_start=2001,
        source_scope="England, UKHSA enhanced surveillance/HES/ONS compilation",
        source_ids="UKHSA_ENGLAND_2024_XLSX;UKHSA_ENGLAND_2024_ANNUAL",
        evidence_role="burden-proximal long-series descriptor",
        comparison_note="The long same-scope series supports mean and maximum comparators; counts remain sparse.",
    )

    brazil_specs = [
        ("hospitalizations", "<1 year", "infant_hospitalizations", "Infant hospitalisations"),
        ("deaths", "<1 year", "infant_deaths", "Infant deaths"),
        ("reported_cases", "<1 Ano", "infant_reported_cases", "Infant reported cases"),
    ]
    for outcome, age_group, metric, label in brazil_specs:
        chosen = severe.loc[
            severe["iso3"].eq("BRA")
            & severe["outcome_type"].eq(outcome)
            & severe["age_group"].eq(age_group)
        ]
        _add_series(
            records,
            iso3="BRA",
            country="Brazil",
            pathway="notification" if outcome == "reported_cases" else "severe",
            metric=metric,
            display_metric=label,
            unit="count",
            values=_values(chosen, "year", "count"),
            pre_start=2010,
            source_scope=str(chosen["geographic_scope"].iloc[0]),
            source_ids=";".join(sorted(chosen["source_id"].astype(str).unique())),
            evidence_role=(
                "notification descriptor" if outcome == "reported_cases" else "burden-proximal long-series descriptor"
            ),
            comparison_note=(
                "The 2010–19 mean and maximum quantify baseline sensitivity; absent death rows are not imputed as zero."
            ),
        )

    china_external = external.loc[external["iso3"].eq("CHN")]
    china_cases = _values(china, "year", "cases")
    china_cases.update(
        _values(china_external.loc[china_external["metric"].eq("reported_cases")], "year", "value")
    )
    _add_series(
        records,
        iso3="CHN",
        country="China",
        pathway="notification",
        metric="reported_cases",
        display_metric="Reported cases",
        unit="count",
        values=china_cases,
        pre_start=2015,
        source_scope="China national notifiable-disease surveillance",
        source_ids="CHN_ANNUAL_COUNTRY_SERIES_2025_12_02;CHN_NDCPA_2023_ANNUAL;CHINA_NDCPA_2024_ANNUAL",
        evidence_role="within-notification-system descriptor",
        comparison_note="Ratios describe notification magnitude and do not identify burden independently of ascertainment.",
    )
    china_deaths = china_external.loc[china_external["metric"].eq("all_age_deaths")]
    _add_series(
        records,
        iso3="CHN",
        country="China",
        pathway="severe",
        metric="all_age_deaths",
        display_metric="All-age reported deaths",
        unit="count",
        values=_values(china_deaths, "year", "value"),
        pre_start=2019,
        source_scope="China national notifiable-disease surveillance",
        source_ids="CHN_NDCPA_2023_ANNUAL;CHINA_NDCPA_2024_ANNUAL",
        evidence_role="within-notification-system severity descriptor",
        comparison_note="Only 2023 and 2024 were archived and deaths share the notification channel with cases.",
    )

    frame = pd.DataFrame.from_records(records)
    country_order = {"France": 0, "England": 1, "Brazil": 2, "China": 3}
    frame["_country_order"] = frame["country"].map(country_order)
    frame["_comparator_order"] = frame["comparator_type"].map(COMPARATOR_ORDER)
    frame = frame.sort_values(
        ["_country_order", "pathway", "display_metric", "_comparator_order"]
    ).drop(columns=["_country_order", "_comparator_order"])

    if not frame.loc[frame["comparator_type"].eq("prepandemic_mean"), "n_baseline_years"].ge(3).all():
        raise RuntimeError("Aggregate baselines require at least three annual observations")
    if frame.loc[frame["comparison_status"].eq("available"), "ratio_2024_to_baseline"].le(0).any():
        raise RuntimeError("Available ratios must be positive")
    france_severe = frame.loc[frame["iso3"].eq("FRA") & frame["pathway"].eq("severe")]
    if set(france_severe["comparator_type"]) != {"adjacent_2023", "last_prepandemic_2019"}:
        raise RuntimeError("France severe outcomes must remain limited to observed comparator availability")

    for path in [OUTPUT, FIGURE_DATA, TABLE]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    _write_markdown(frame)
    print(
        f"Wrote {len(frame)} comparator rows; "
        f"{frame['comparison_status'].eq('available').sum()} have estimable ratios."
    )


if __name__ == "__main__":
    main()
