#!/usr/bin/env python3
"""Build the descriptive real-world triangulation used in the NC revision.

This analysis is deliberately narrower than external validation. It quantifies
retrospectively specified 2023-to-2024 changes in observed testing and severe-outcome streams
where the local evidence ledger contains comparable annual rows. The resulting
contrasts show whether auxiliary streams move as simple rescalings of routine
notifications; they do not estimate transmission or validate the simulated
classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TESTING_PATH = ROOT / "outputs" / "direct_testing_denominator_evidence.csv"
SEVERE_PATH = ROOT / "outputs" / "severe_outcome_age_time_evidence.csv"
OUTPUT_PATH = ROOT / "outputs" / "nc_real_world_triangulation.csv"
MANUSCRIPT_TABLE_PATH = ROOT / "manuscript" / "tables" / "nc_real_world_triangulation.csv"
MANUSCRIPT_MARKDOWN_PATH = (
    ROOT / "manuscript" / "tables_md" / "nc_real_world_triangulation.md"
)


@dataclass(frozen=True)
class Contrast:
    setting: str
    iso3: str
    geographic_scope: str
    stream: str
    metric: str
    unit: str
    baseline_year: int
    comparison_year: int
    baseline_value: float
    comparison_value: float
    source_ids: str
    interpretation_boundary: str


def require_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def testing_contrast(
    testing: pd.DataFrame,
    *,
    iso3: str,
    setting: str,
    geographic_scope: str,
    metric: str,
    column: str,
    unit: str,
    boundary: str,
) -> Contrast:
    baseline = require_one(testing, iso3=iso3, year=2023)
    comparison = require_one(testing, iso3=iso3, year=2024)
    source_ids = "; ".join(
        sorted(
            {
                str(baseline["direct_testing_series_source_ids"]),
                str(comparison["direct_testing_series_source_ids"]),
            }
        )
    )
    return Contrast(
        setting=setting,
        iso3=iso3,
        geographic_scope=geographic_scope,
        stream="testing",
        metric=metric,
        unit=unit,
        baseline_year=2023,
        comparison_year=2024,
        baseline_value=float(baseline[column]),
        comparison_value=float(comparison[column]),
        source_ids=source_ids,
        interpretation_boundary=boundary,
    )


def severe_contrast(
    severe: pd.DataFrame,
    *,
    iso3: str,
    setting: str,
    geographic_scope: str,
    stream: str,
    outcome_type: str,
    age_group: str,
    metric: str,
    boundary: str,
) -> Contrast:
    baseline = require_one(
        severe,
        iso3=iso3,
        outcome_type=outcome_type,
        age_group=age_group,
        year=2023,
    )
    comparison = require_one(
        severe,
        iso3=iso3,
        outcome_type=outcome_type,
        age_group=age_group,
        year=2024,
    )
    if str(baseline["geographic_scope"]) != str(comparison["geographic_scope"]):
        raise RuntimeError(f"Scope changed between years for {iso3} {metric}")
    return Contrast(
        setting=setting,
        iso3=iso3,
        geographic_scope=geographic_scope,
        stream=stream,
        metric=metric,
        unit="count",
        baseline_year=2023,
        comparison_year=2024,
        baseline_value=float(baseline["count"]),
        comparison_value=float(comparison["count"]),
        source_ids=str(comparison["source_id"]),
        interpretation_boundary=boundary,
    )


def to_frame(contrasts: list[Contrast]) -> pd.DataFrame:
    frame = pd.DataFrame([contrast.__dict__ for contrast in contrasts])
    frame["absolute_change"] = frame["comparison_value"] - frame["baseline_value"]
    frame["ratio_2024_vs_2023"] = np.where(
        frame["baseline_value"] > 0,
        frame["comparison_value"] / frame["baseline_value"],
        np.nan,
    )
    frame["percent_change"] = np.where(
        frame["baseline_value"] > 0,
        100 * frame["absolute_change"] / frame["baseline_value"],
        np.nan,
    )
    frame["percentage_point_change"] = np.where(
        frame["unit"] == "percent",
        frame["absolute_change"],
        np.nan,
    )
    columns = [
        "setting",
        "iso3",
        "geographic_scope",
        "stream",
        "metric",
        "unit",
        "baseline_year",
        "baseline_value",
        "comparison_year",
        "comparison_value",
        "absolute_change",
        "ratio_2024_vs_2023",
        "percent_change",
        "percentage_point_change",
        "source_ids",
        "interpretation_boundary",
    ]
    return frame.loc[:, columns]


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame[
        [
            "setting",
            "geographic_scope",
            "metric",
            "baseline_value",
            "comparison_value",
            "ratio_2024_vs_2023",
            "percentage_point_change",
            "interpretation_boundary",
        ]
    ].copy()
    display.columns = [
        "Setting",
        "Geographic scope",
        "Observed stream",
        "2023",
        "2024",
        "2024/2023 ratio",
        "Percentage-point change",
        "Boundary",
    ]
    for column in ["2023", "2024"]:
        display[column] = display[column].map(lambda value: f"{value:g}")
    display["2024/2023 ratio"] = display["2024/2023 ratio"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )
    display["Percentage-point change"] = display["Percentage-point change"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    testing = pd.read_csv(TESTING_PATH)
    severe = pd.read_csv(SEVERE_PATH)
    testing["year"] = pd.to_numeric(testing["year"], errors="raise").astype(int)
    severe["year"] = pd.to_numeric(severe["year"], errors="raise").astype(int)

    testing_boundary = (
        "Testing counts and positivity describe the observation process; "
        "positivity uses exact positive/eligible tests, while submitted samples and "
        "distinct confirmed cases are separate units; these counts do not estimate infection incidence."
    )
    england_death_boundary = (
        "Testing covers England whereas mortality covers England and Wales; "
        "the paired display is descriptive, not a linked validation cohort or negative control."
    )
    brazil_case_boundary = (
        "Notifications are from SINAN and admissions from the public SIH/SUS "
        "system; neither series is a census of all infections or admissions."
    )

    contrasts = [
        testing_contrast(
            testing,
            iso3="FRA",
            setting="France",
            geographic_scope="national 3Labos network",
            metric="PCR tests",
            column="testing_denominator",
            unit="count",
            boundary=testing_boundary,
        ),
        testing_contrast(
            testing,
            iso3="FRA",
            setting="France",
            geographic_scope="national 3Labos network",
            metric="PCR positivity",
            column="test_positivity_percent",
            unit="percent",
            boundary=testing_boundary,
        ),
        testing_contrast(
            testing,
            iso3="GBR",
            setting="England",
            geographic_scope="England laboratory surveillance",
            metric="submitted samples",
            column="submitted_samples",
            unit="count",
            boundary=testing_boundary,
        ),
        testing_contrast(
            testing,
            iso3="GBR",
            setting="England",
            geographic_scope="England laboratory surveillance",
            metric="test positivity",
            column="test_positivity_percent",
            unit="percent",
            boundary=testing_boundary,
        ),
        severe_contrast(
            severe,
            iso3="GBR",
            setting="England and Wales",
            geographic_scope="England and Wales mortality",
            stream="severe outcome",
            outcome_type="deaths",
            age_group="<1 year",
            metric="infant deaths",
            boundary=england_death_boundary,
        ),
        severe_contrast(
            severe,
            iso3="GBR",
            setting="England and Wales",
            geographic_scope="England and Wales mortality",
            stream="severe outcome",
            outcome_type="deaths",
            age_group="all_ages",
            metric="all-age deaths",
            boundary=england_death_boundary,
        ),
        severe_contrast(
            severe,
            iso3="BRA",
            setting="Brazil",
            geographic_scope="Brazil national notifications",
            stream="notification",
            outcome_type="reported_cases",
            age_group="all_ages",
            metric="notified cases",
            boundary=brazil_case_boundary,
        ),
        severe_contrast(
            severe,
            iso3="BRA",
            setting="Brazil",
            geographic_scope="Brazil national SIH/SUS admissions",
            stream="severe outcome",
            outcome_type="hospitalizations",
            age_group="all_ages",
            metric="all-age admissions",
            boundary=brazil_case_boundary,
        ),
        severe_contrast(
            severe,
            iso3="BRA",
            setting="Brazil",
            geographic_scope="Brazil national SIH/SUS admissions",
            stream="severe outcome",
            outcome_type="hospitalizations",
            age_group="<1 year",
            metric="infant admissions",
            boundary=brazil_case_boundary,
        ),
    ]
    frame = to_frame(contrasts)

    expected = {
        ("France", "PCR tests"): (13852.0, 188539.0),
        ("England", "submitted samples"): (9268.0, 36472.0),
        ("Brazil", "notified cases"): (222.0, 7204.0),
        ("Brazil", "all-age admissions"): (238.0, 730.0),
        ("Brazil", "infant admissions"): (193.0, 545.0),
    }
    for key, values in expected.items():
        row = frame.loc[
            (frame["setting"] == key[0]) & (frame["metric"] == key[1])
        ].iloc[0]
        observed = (row["baseline_value"], row["comparison_value"])
        if observed != values:
            raise RuntimeError(f"Unexpected values for {key}: {observed} != {values}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    frame.to_csv(MANUSCRIPT_TABLE_PATH, index=False)
    MANUSCRIPT_MARKDOWN_PATH.write_text(markdown_table(frame), encoding="utf-8")

    print(
        "Wrote "
        f"{len(frame)} contrasts to {OUTPUT_PATH.relative_to(ROOT)} and manuscript tables"
    )


if __name__ == "__main__":
    main()
