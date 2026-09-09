#!/usr/bin/env python3
"""Convert assumed baseline infant-event rates to illustrative fixed catchments.

The frontier input is annual baseline intensity b for infants younger than one
under a three-year follow-up with a known baseline. Observed epidemic-year rates
are context only. Arithmetic catchments do not define minimum service standards
or justify trading follow-up duration against catchment population.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
# Preserve the published table filename; record this scientific revision separately.
DATE_TAG = "2026-08-13"
REVISION_DATE = "2026-09-08"
FOLLOWUP_YEARS = 3
sys.path.insert(0, str(ROOT / "script" / "src"))
from pertussis_global_dynamics.bayes.model2c_surveillance_design import SURVEILLANCE_DESIGN_VERSION

FIGURE_FRONTIER = ROOT / "manuscript" / "tables" / "figure_7_severe_resource_frontier.csv"
SEVERE_SUMMARY = ROOT / "manuscript" / "tables" / "severe_outcome_age_time_summary.csv"
SEVERE_EVIDENCE = ROOT / "manuscript" / "tables" / "severe_outcome_age_time_evidence.csv"
GBD_SUMMARY = ROOT / "manuscript" / "tables" / "gbd_pertussis_burden_summary.csv"
GBD_YEAR_SUMMARY = ROOT / "manuscript" / "tables" / "gbd_pertussis_burden_year_summary.csv"
POPULATION_AGE = ROOT / "data" / "clean" / "population_age.csv"
EXTERNAL_CHALLENGE = ROOT / "outputs" / "external_empirical_challenge_comparisons.csv"
TRIANGULATION = ROOT / "manuscript" / "tables" / "nc_real_world_triangulation.csv"
MANUSCRIPT = ROOT / "manuscript" / "manuscript.md"
SUPPLEMENT = ROOT / "manuscript" / "appendix" / "supplementary_appendix.md"

OUTPUT_CSV = ROOT / "outputs" / f"severe_event_frontier_lookup_{DATE_TAG}.csv"
OUTPUT_MD = ROOT / "outputs" / f"severe_event_frontier_lookup_{DATE_TAG}.md"
TABLE_CSV = ROOT / "manuscript" / "tables" / f"severe_event_frontier_lookup_{DATE_TAG}.csv"
REPORT_MD = ROOT / "manuscript" / "reports" / f"severe_event_frontier_lookup_{DATE_TAG}.md"

CATCHMENT_POPULATIONS = [100_000, 500_000, 1_000_000]
INFANT_AGE_GROUPS = {"<1 year", "<1 Ano", "0 years", "0  years", "00-00"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def as_int(value: object) -> int | None:
    value_float = as_float(value)
    if math.isnan(value_float):
        return None
    return int(value_float)


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(v for v in values if math.isfinite(v))
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def format_number(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def load_population() -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, str]]:
    infant = defaultdict(float)
    total = {}
    countries = {}
    for row in read_csv(POPULATION_AGE):
        iso3 = row["iso3"]
        year = as_int(row["year"])
        if year is None:
            continue
        countries[iso3] = row["country"]
        total[(iso3, year)] = as_float(row["total_population"])
        if row["age_group"] in {"infant_0_2m", "infant_3_11m"}:
            infant[(iso3, year)] += as_float(row["population"], 0.0)
    population = {}
    for key, infant_population in infant.items():
        total_population = total.get(key, math.nan)
        population[key] = {
            "infant_population": infant_population,
            "total_population": total_population,
            "under1_share": infant_population / total_population
            if total_population and math.isfinite(total_population)
            else math.nan,
        }
    return population, countries


def load_frontier_auc(path: Path = FIGURE_FRONTIER) -> dict[tuple[float, float], dict[str, object]]:
    """Read unrounded current infant-frontier results, rejecting stale estimands."""
    frontier = {}
    for row in read_csv(path):
        if row.get("surveillance_design_version") != SURVEILLANCE_DESIGN_VERSION:
            raise ValueError("Rebuild the infant severe frontier before translating its baseline intensity")
        if row.get("package_id") != "notifications+severe":
            raise ValueError("Expected the notification-plus-infant-severe package")
        if (as_int(row.get("followup_years")) != FOLLOWUP_YEARS
                or row.get("baseline_uncertainty") != "known baseline expectation"
                or "baseline annual expected infant events" not in row.get("event_scale", "")):
            raise ValueError("Lookup requires the three-year, known-baseline infant event design")
        events = as_float(row["severe_baseline_events_per_country_year"])
        drift = as_float(row["severe_annual_log_rate_drift_sd"])
        auc, lower, upper = (as_float(row[column]) for column in ["auc", "auc_lower_95", "auc_upper_95"])
        if not (events > 0 and drift >= 0 and 0 <= auc <= 1 and 0 <= lower <= upper <= 1):
            raise ValueError("Invalid event intensity, drift or AUC interval")
        key = (events, drift)
        if key in frontier:
            raise ValueError("Duplicate event-intensity/drift cell")
        for field, expected in [("meets_auc_0_80", auc >= 0.80),
                                ("meets_conservative_auc_0_80", lower >= 0.80)]:
            if str(row[field]).lower() != str(expected).lower():
                raise ValueError("Frontier pass status must use unrounded AUC and confidence limits")
        frontier[key] = dict(auc=auc, auc_lower_95=lower, auc_upper_95=upper,
                             meets_auc_0_80=auc >= 0.80,
                             meets_conservative_auc_0_80=lower >= 0.80,
                             surveillance_design_version=row["surveillance_design_version"])
    if not frontier:
        raise ValueError("Empty infant severe frontier")
    return frontier


def frontier_thresholds(frontier: dict[tuple[float, float], dict[str, object]]) -> list[dict[str, object]]:
    """First passing *tested* b for each drift, without interpolation/extrapolation."""
    rows = []
    for drift in sorted({key[1] for key in frontier}):
        cells = sorted((b, cell) for (b, d), cell in frontier.items() if d == drift)
        row = {"annual_log_rate_drift_sd": drift, "followup_years": FOLLOWUP_YEARS,
               "baseline_assumption": "known baseline intensity", "age_scope": "younger than 1 year"}
        for label, field in [("point", "auc"), ("conservative", "auc_lower_95")]:
            passing = [b for b, cell in cells if as_float(cell[field]) >= 0.80]
            row[f"first_tested_baseline_intensity_{label}_auc_0_80"] = min(passing) if passing else math.nan
            row[f"{label}_threshold_status"] = "reached_in_tested_grid" if passing else "not_reached_in_tested_grid"
        rows.append(row)
    return rows


def observed_infant_rates(
    population: dict[tuple[str, int], dict[str, float]]
) -> list[dict[str, object]]:
    rates = []
    for row in read_csv(SEVERE_EVIDENCE):
        if row["outcome_type"] not in {"hospitalizations", "deaths"}:
            continue
        if row["age_group"] not in INFANT_AGE_GROUPS:
            continue
        # National populations must not be assigned to England-only, network,
        # or other subnational numerators.
        allowed_scopes = {
            "BRA": {"Brazil national resident deaths", "Brazil national SIH/SUS admissions"},
            "JPN": {"Japan national vital statistics"},
            "USA": {"national"},
        }
        if row["geographic_scope"] not in allowed_scopes.get(row["iso3"], set()):
            continue
        year = as_int(row["year"])
        count = as_float(row["count"])
        if year is None or not math.isfinite(count):
            continue
        pop = population.get((row["iso3"], year))
        if not pop or not math.isfinite(pop["infant_population"]) or pop["infant_population"] <= 0:
            continue
        rate = count / pop["infant_population"] * 100_000
        rates.append(
            {
                "scenario_group": "observed_rate_context",
                "rate_role": "observed country-year context; not a model baseline rate",
                "scenario_id": f"{row['iso3']}_{year}_{row['outcome_type']}_{row['age_group']}".replace(" ", "_"),
                "scenario_label": (
                    f"{row['country']} {year} infant {row['outcome_type']}"
                ),
                "iso3": row["iso3"],
                "country": row["country"],
                "year": year,
                "outcome_type": row["outcome_type"],
                "event_rate_per_100k_infant_years": rate,
                "source_count": count,
                "infant_population_denominator": pop["infant_population"],
                "under1_share": pop["under1_share"],
                "source_id": row["source_id"],
                "geographic_scope": row["geographic_scope"],
                "rate_source": (
                    "severe_outcome_age_time_evidence.csv count divided by "
                    "data/clean/population_age.csv <1-year population"
                ),
                "interpretation_boundary": (
                    "Observed source-system rate for context only; not a baseline-rate estimate, "
                    "not an external validation rate and not converted into a frontier catchment. "
                    "Surveillance coverage and admission/coding practice remain source-specific."
                ),
            }
        )
    return rates


def scenario_rates(observed: list[dict[str, object]]) -> list[dict[str, object]]:
    brazil_hospitalization_rates = [
        as_float(row["event_rate_per_100k_infant_years"])
        for row in observed
        if row["iso3"] == "BRA" and row["outcome_type"] == "hospitalizations"
    ]
    percentiles = {
        "low_event_rate": 0.10,
        "medium_event_rate": 0.50,
        "high_event_rate": 0.90,
    }
    labels = {
        "low_event_rate": "Assumed low baseline infant admission rate",
        "medium_event_rate": "Assumed medium baseline infant admission rate",
        "high_event_rate": "Assumed high baseline infant admission rate",
    }
    rows = []
    for scenario_id, percentile in percentiles.items():
        rows.append(
            {
                "scenario_group": "assumed_baseline_rate",
                "rate_role": "hypothetical annual baseline rate, numerically anchored to historical quantiles",
                "scenario_id": scenario_id,
                "scenario_label": labels[scenario_id],
                "iso3": "",
                "country": "",
                "year": "",
                "outcome_type": "hospitalizations",
                "event_rate_per_100k_infant_years": quantile(brazil_hospitalization_rates, percentile),
                "source_count": "",
                "infant_population_denominator": "",
                "under1_share": "",
                "source_id": "BRA_SIH_TABNET_COQUELUCHE_HOSPITALIZATIONS_2010_2024",
                "geographic_scope": "Brazil national SIH/SUS admissions, 2010-2024",
                "rate_source": (
                    f"{int(percentile * 100)}th percentile of Brazil 2010-2024 annual "
                    "infant SIH/SUS pertussis admission rates"
                ),
                "interpretation_boundary": (
                    "Assumed baseline rate for arithmetic illustration only; historical quantiles "
                    "do not estimate this model baseline. Three annual follow-up observations and "
                    "a known baseline are assumed; not an external validation or minimum service standard."
                ),
            }
        )
    return rows


def selected_observed_examples(observed: list[dict[str, object]]) -> list[dict[str, object]]:
    keep = {
        ("BRA", 2019, "hospitalizations"),
        ("BRA", 2023, "hospitalizations"),
        ("BRA", 2024, "hospitalizations"),
        ("BRA", 2024, "deaths"),
        ("JPN", 2024, "deaths"),
        ("USA", 2019, "deaths"),
        ("USA", 2024, "deaths"),
        ("USA", 2025, "deaths"),
    }
    examples = [
        row
        for row in observed
        if (row["iso3"], row["year"], row["outcome_type"]) in keep
    ]
    examples.sort(key=lambda row: (str(row["iso3"]), str(row["outcome_type"]), int(row["year"])))
    return examples


def build_lookup_rows() -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]]]:
    population, _ = load_population()
    frontier_auc = load_frontier_auc()
    observed = observed_infant_rates(population)
    selected_examples = selected_observed_examples(observed)
    rate_rows = scenario_rates(observed) + selected_examples

    under1_shares_2024 = [
        pop["under1_share"]
        for (iso3, year), pop in population.items()
        if year == 2024 and math.isfinite(pop["under1_share"]) and pop["under1_share"] > 0
    ]
    share_summary = {
        "under1_share_q25_2024": quantile(under1_shares_2024, 0.25),
        "under1_share_median_2024": quantile(under1_shares_2024, 0.50),
        "under1_share_q75_2024": quantile(under1_shares_2024, 0.75),
    }

    lookup_rows = []
    threshold_rows = frontier_thresholds(frontier_auc)
    targets = sorted({key[0] for key in frontier_auc})
    for scenario in rate_rows:
        rate = as_float(scenario["event_rate_per_100k_infant_years"])
        if not math.isfinite(rate) or rate <= 0:
            continue
        if scenario["scenario_group"] == "observed_rate_context":
            # Epidemic-year observations never inherit an AUC, target b, or
            # catchment size by substituting their observed rate for r0.
            lookup_rows.append({**scenario, "revision_date": REVISION_DATE,
                                "lookup_role": "context_only_no_frontier_mapping"})
            continue
        assumed_share = share_summary["under1_share_median_2024"]
        for target in targets:
            infant_population = target * 100_000 / rate
            row = {
                **scenario,
                "revision_date": REVISION_DATE,
                "lookup_role": "conditional_baseline_rate_arithmetic",
                "target_baseline_expected_infant_events_per_year": target,
                "assumed_baseline_rate_per_100k_infant_years": rate,
                "hypothetical_infant_population": infant_population,
                "hypothetical_total_population_at_assumed_infant_share": infant_population / assumed_share,
                "assumed_under1_population_share": assumed_share,
                "infant_share_reference_year": 2024,
                "hypothetical_total_population_at_q25_infant_share": infant_population / share_summary["under1_share_q25_2024"],
                "hypothetical_total_population_at_q75_infant_share": infant_population / share_summary["under1_share_q75_2024"],
                "age_scope": "younger than 1 year",
                "followup_years": FOLLOWUP_YEARS,
                "baseline_assumption": "known annual baseline intensity; three distinct annual follow-up measurements",
                "epidemic_event_expectation": "b multiplied by year-specific infant symptomatic burden ratio and rate drift",
                "population_assumption": "illustrative fixed population and under-1 share across follow-up",
                "formula": "b = r0 * N_infant / 100000; N_infant = b * 100000 / r0; N_total = N_infant / s",
                "observation_duration_boundary": "No equivalence between extra follow-up years and a larger catchment is established",
                "operational_minimum_status": "not estimated or validated",
                "surveillance_design_version": SURVEILLANCE_DESIGN_VERSION,
            }
            for catchment in CATCHMENT_POPULATIONS:
                row[f"baseline_expected_events_per_year_in_{catchment}_catchment"] = rate / 100_000 * catchment * assumed_share
            for thresholds in threshold_rows:
                drift = thresholds["annual_log_rate_drift_sd"]
                auc = frontier_auc.get((float(target), drift), {})
                row[f"figure_frontier_auc_at_drift_{drift:g}"] = auc.get("auc", "")
                row[f"figure_frontier_lower95_at_drift_{drift:g}"] = auc.get("auc_lower_95", "")
                for criterion in ["point", "conservative"]:
                    value = thresholds[f"first_tested_baseline_intensity_{criterion}_auc_0_80"]
                    row[f"first_tested_b_{criterion}_at_drift_{drift:g}"] = value
                    row[f"{criterion}_threshold_status_at_drift_{drift:g}"] = thresholds[f"{criterion}_threshold_status"]
                row[f"figure_frontier_meets_auc_0_80_at_drift_{drift:g}"] = auc.get("meets_auc_0_80", "")
                row[f"figure_frontier_meets_conservative_auc_0_80_at_drift_{drift:g}"] = auc.get("meets_conservative_auc_0_80", "")
            lookup_rows.append(row)
    return lookup_rows, share_summary, rate_rows


def gbd_context() -> dict[str, object]:
    year_rows = read_csv(GBD_YEAR_SUMMARY)
    latest = max(year_rows, key=lambda row: int(row["year"]))
    country_rows = read_csv(GBD_SUMMARY)
    country_daly_shares = [
        as_float(row["dalys_under1_share_allages_mean"])
        for row in country_rows
        if row.get("scope") == "country"
    ]
    country_incidence_shares = [
        as_float(row["incidence_under1_share_allages_mean"])
        for row in country_rows
        if row.get("scope") == "country"
    ]
    return {
        "latest_gbd_year": latest["year"],
        "weighted_dalys_under1_share_latest": as_float(
            latest["weighted_dalys_under1_share_allages"]
        ),
        "weighted_incidence_under1_share_latest": as_float(
            latest["weighted_incidence_under1_share_allages"]
        ),
        "median_country_dalys_under1_share": statistics.median(country_daly_shares),
        "median_country_incidence_under1_share": statistics.median(country_incidence_shares),
        "gbd_note": (
            "GBD burden is used here as infant-concentration context only; it is not used "
            "as an observable severe-event rate."
        ),
    }


def extract_context_lines() -> list[str]:
    contexts = []
    for path, label in [(MANUSCRIPT, "manuscript"), (SUPPLEMENT, "supplement")]:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            lower = line.lower()
            if (
                "severe-event frontier" in lower
                or "expected observable infant event" in lower
                or "severe_baseline_events" in lower
                or "one-parameter monitoring precision frontiers" in lower
            ):
                contexts.append(f"{label}:{i}: {line.strip()}")
    return contexts[:16]


def markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = format_number(value)
            cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(
    rows: list[dict[str, object]],
    share_summary: dict[str, float],
    rate_rows: list[dict[str, object]],
) -> None:
    gbd = gbd_context()
    thresholds = frontier_thresholds(load_frontier_auc())
    scenarios = [row for row in rate_rows if row["scenario_group"] == "assumed_baseline_rate"]
    lookup = [row for row in rows if row["lookup_role"] == "conditional_baseline_rate_arithmetic"]
    context = [row for row in rows if row["lookup_role"] == "context_only_no_frontier_mapping"]
    report = f"""# Conditional baseline infant-event arithmetic (revised {REVISION_DATE})

The legacy filename is retained for Supplementary Table 22. This revision uses the current infant (<1 year) frontier, `{SURVEILLANCE_DESIGN_VERSION}`. It does not define a minimum catchment, minimum service standard, or validated surveillance duration.

## Estimand and arithmetic

The model input `b` is the expected **annual baseline** number of infant severe events per fixed catchment. It is not the expected event count in an observed epidemic year or the total number accumulated during follow-up. For an explicitly assumed baseline event rate `r0` per 100,000 infant-years and an illustrative fixed infant population:

`b = r0 * N_infant / 100000`

`N_infant = b * 100000 / r0; N_total = N_infant / s`

Generic arithmetic uses the 2024 median under-1 population share `s={share_summary['under1_share_median_2024']:.6f}` from the available population table, with q25/q75 sensitivity columns. This share is an assumed fixed demographic conversion, not an estimate of every prospective sentinel's population composition.

The classifier uses **three distinct annual observations in 2023–25 and a known baseline intensity**. Epidemic-year expectations vary with the infant symptomatic-burden ratio and severe-rate drift. Pooling sites or observing for more years does not automatically reproduce those three measurements, their uncertainty or the fixed-baseline assumption. The lookup therefore supplies no required-sentinel-years column.

## Assumed baseline-rate scenarios

The low, medium and high numeric rates are anchored to the 10th, 50th and 90th percentiles of Brazil's 2010–24 SIH/SUS infant admission rates. Treating these values as `r0` is an explicit scenario assumption. Those historical rates, which include epidemic and trough years, are not estimates of the simulation's pre-event baseline or calibrated event-conversion probabilities. SIH/SUS excludes private admissions.

{markdown_table(scenarios, ['scenario_label', 'event_rate_per_100k_infant_years', 'rate_source'])}

GBD remains infant-burden context only. In the latest GBD year ({gbd['latest_gbd_year']}), its weighted under-1 DALY share was {gbd['weighted_dalys_under1_share_latest']:.3f}; it is not an observable event rate in this calculation.

## Current tested frontier thresholds

Thresholds below are the first **tested** baseline intensities satisfying the point AUC or lower-95%-limit criterion of 0.80 at each drift. They are calculated from unrounded current results without interpolation, and an absent crossing remains `not_reached_in_tested_grid`.

{markdown_table(thresholds, ['annual_log_rate_drift_sd', 'first_tested_baseline_intensity_point_auc_0_80', 'point_threshold_status', 'first_tested_baseline_intensity_conservative_auc_0_80', 'conservative_threshold_status'])}

## Conditional population arithmetic

These hypothetical populations achieve the specified baseline intensity only under the chosen `r0`, fixed coverage and demographic share. The attached AUCs still come from the conditional three-year simulation, not from the empirical rate source.

{markdown_table(lookup, ['scenario_label', 'target_baseline_expected_infant_events_per_year', 'assumed_baseline_rate_per_100k_infant_years', 'hypothetical_infant_population', 'hypothetical_total_population_at_assumed_infant_share'])}

## Observed rates: context only

The observed country-year rows below retain their actual endpoint, geography and population denominator. They are not mapped to `b`, an AUC, or a catchment size. In particular, a 2024 outbreak-year hospitalisation rate does not identify the baseline event rate needed by the frontier. Infant mortality is a different and sparse endpoint.

{markdown_table(context, ['scenario_label', 'geographic_scope', 'source_count', 'event_rate_per_100k_infant_years', 'source_id'])}

Only source scopes compatible with the available national infant populations are converted to contextual rates. France's RENACOQ network and England/England-and-Wales records are excluded from national-population division here. Source-specific testing, care access, attribution, referral and catchment coverage may change, and baseline-rate estimation uncertainty is not included. Operational sizing requires separate validation of these conditions and cannot be inferred from this arithmetic alone.
"""
    for path in [OUTPUT_MD, REPORT_MD]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")


def main() -> None:
    # Explicitly load all requested inputs so missing required evidence fails early.
    for path in [
        FIGURE_FRONTIER,
        SEVERE_SUMMARY,
        SEVERE_EVIDENCE,
        GBD_SUMMARY,
        GBD_YEAR_SUMMARY,
        MANUSCRIPT,
        SUPPLEMENT,
        EXTERNAL_CHALLENGE,
        TRIANGULATION,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    rows, share_summary, rate_rows = build_lookup_rows()
    write_csv(OUTPUT_CSV, rows)
    write_csv(TABLE_CSV, rows)
    write_report(rows, share_summary, rate_rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {TABLE_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
