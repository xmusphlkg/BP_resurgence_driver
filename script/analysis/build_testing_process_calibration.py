#!/usr/bin/env python3
"""Summarise archived pertussis testing evidence for a qualitative crosswalk.

This script is a submission-facing comparability aid. It separates tests
performed, approximate positives/positivity, assay mix, denominator scope and
tested-population notes, then contrasts those empirical ranges with the already
archived simulated testing-index precision and cross-loading frontiers. Observed
testing-volume changes are not treated as measurement-error SD estimates.

The output is not classifier validation and does not estimate empirical
sensitivity, specificity, burden fraction or ascertainment fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "manuscript" / "tables"
REPORTS = ROOT / "manuscript" / "reports"

DIRECT_EVIDENCE = OUTPUTS / "direct_testing_denominator_evidence.csv"
DIRECT_SUMMARY = TABLES / "direct_testing_denominator_evidence_summary.csv"
AGE_TESTING = TABLES / "diagnostic_age_testing_summary.csv"
DIAGNOSTIC_TIMELINE = ROOT / "data" / "clean" / "surveillance_diagnostic_timeline.csv"
DIAGNOSTIC_SOURCES = ROOT / "data" / "clean" / "surveillance_diagnostic_sources.csv"
OUTCOME_SOURCES = ROOT / "data" / "clean" / "surveillance_outcome_sources.csv"
SOURCE_REGISTRY = ROOT / "data" / "metadata" / "source_registry.csv"
PRECISION_FRONTIER = TABLES / "surveillance_precision_frontier.csv"
CONTAMINATION_FRONTIER = TABLES / "nc_measurement_contamination_frontier.csv"

COUNTRY_INVENTORY_OUTPUT = OUTPUTS / (
    "testing_process_calibration_2026-08-13.country_source_inventory.csv"
)
COUNTRY_MAPPING_OUTPUT = OUTPUTS / (
    "testing_process_calibration_2026-08-13.country_parameter_mapping.csv"
)
FRONTIER_CROSSWALK_OUTPUT = OUTPUTS / (
    "testing_process_calibration_2026-08-13.frontier_crosswalk.csv"
)
OUTPUT_REPORT = OUTPUTS / "testing_process_calibration_2026-08-13.md"
MANUSCRIPT_TABLE = TABLES / "testing_process_calibration_summary.csv"
MANUSCRIPT_REPORT = REPORTS / "testing_process_calibration_2026-08-13.md"


@dataclass(frozen=True)
class ProcessMetadata:
    assay_mix: str
    denominator_scope: str
    denominator_scope_class: str
    eligibility_or_population: str
    cross_loading_fraction: float
    cross_loading_rationale: str


PROCESS_METADATA: dict[str, ProcessMetadata] = {
    "FRA": ProcessMetadata(
        "PCR-only 3Labos stream with same-stream positivity",
        "National 3Labos laboratory network; not an all-laboratory census",
        "network",
        "PCR tests submitted to the 3Labos surveillance network",
        0.25,
        "Same-stream PCR denominator and positivity; network scope remains a limitation.",
    ),
    "GBR": ProcessMetadata(
        "Mixed serum, oral-fluid and PCR submitted samples",
        "England laboratory surveillance; retained under GBR for model alignment",
        "england_only",
        "Submitted diagnostic samples in enhanced laboratory surveillance",
        0.50,
        "Denominator and positivity are available, but assay pathways are mixed.",
    ),
    "USA": ProcessMetadata(
        "Commercial Bordetella test orders; assay mix not separated in archived extract",
        "Large commercial laboratory series; not a national all-laboratory census",
        "commercial_laboratory",
        "Commercial laboratory test orders for Bordetella species",
        0.50,
        "Repeated volume series exists but positivity and assay-specific splits are absent.",
    ),
    "SWE": ProcessMetadata(
        "B. pertussis samples analysed; assay mix not separated in archived extract",
        "Official national laboratory-reporting context",
        "national_laboratory",
        "Samples analysed for B. pertussis",
        0.50,
        "Repeated denominator exists but positivity and assay-specific splits are absent.",
    ),
    "THA": ProcessMetadata(
        "RT-PCR pediatric hospital snapshot plus national outbreak laboratory submissions",
        "Mixed hospital, outbreak and national snapshots",
        "scope_changed_snapshot",
        "Children with persistent cough in 2019; suspected outbreak submissions in 2023",
        0.75,
        "Two rows are useful bounds but denominator scope and eligibility differ.",
    ),
    "AUS": ProcessMetadata(
        "NSW laboratory testing denominator; assay mix not separated in archived extract",
        "New South Wales laboratory project; historical only for current model window",
        "provincial_historical",
        "Pertussis tests reported by participating NSW laboratories",
        0.75,
        "Provincial and historical denominator; useful as a boundary, not a current module.",
    ),
    "CAN": ProcessMetadata(
        "British Columbia pertussis test requests, mainly PCR/culture context",
        "British Columbia provincial surveillance; historical only",
        "provincial_historical",
        "Monthly pertussis test requests during a provincial outbreak period",
        0.75,
        "Provincial denominator and incomplete positivity; useful as a boundary.",
    ),
    "CHN": ProcessMetadata(
        "Culture/PCR active surveillance positivity snapshot",
        "Three sentinel hospitals in Yiwu and Yongcheng",
        "sentinel_snapshot",
        "Active surveillance among medically attended/surveilled cases",
        0.75,
        "Single active-surveillance snapshot with sentinel scope.",
    ),
    "BRA": ProcessMetadata(
        "Routine RT-PCR/culture reference-laboratory summary",
        "Sao Paulo state reference-laboratory study",
        "regional_single_study",
        "Suspected-case/contact samples submitted for routine diagnosis",
        0.75,
        "Single regional laboratory study, not a repeated national denominator.",
    ),
    "JPN": ProcessMetadata(
        "Multitarget PCR laboratory-surveillance summary",
        "National multi-site laboratory-surveillance study",
        "single_study",
        "Clinically diagnosed patients tested by PCR",
        0.75,
        "Single historical laboratory study, not a repeated national denominator.",
    ),
    "NLD": ProcessMetadata(
        "Age-stratified PCR/culture versus serology testing summary",
        "One Dutch region",
        "regional_age_stratified",
        "Infants recommended PCR/culture; older prolonged-cough patients mainly serology",
        0.50,
        "Assay/tested-population split is explicit, but evidence is regional and historical.",
    ),
    "CHE": ProcessMetadata(
        "Sentinel PCR-supported surveillance summary",
        "Swiss Sentinel Surveillance Network",
        "sentinel_single_study",
        "PCR was offered for sentinel-reported suspected cases",
        0.75,
        "Single sentinel-era denominator summary, not a repeated current series.",
    ),
}

SOURCE_ALIASES = {
    "FRA_RENACOQ_2025_DATA": "FRA_SPF_2025_RENACOQ_DATA",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _evidence_tier(support_class: str) -> str:
    support_class = str(support_class)
    if support_class == "multi_year_model_window_series":
        return "direct_repeated_model_window"
    if support_class == "multi_year_historical_only_series":
        return "historical_repeated_boundary"
    if support_class == "multi_year_direct_testing_denominator_series_support":
        return "direct_repeated_denominator_series"
    if support_class == "single_year_model_window_summary":
        return "single_model_window_anchor"
    if support_class == "single_year_historical_summary":
        return "single_historical_boundary"
    if support_class == "single_year_direct_testing_denominator_summary":
        return "single_year_denominator_summary"
    return "unclassified"


def build_source_lookup(
    diagnostic_sources: pd.DataFrame,
    outcome_sources: pd.DataFrame,
    source_registry: pd.DataFrame,
) -> pd.DataFrame:
    ledgers = []
    for source in (diagnostic_sources, outcome_sources):
        ledgers.append(
            source.loc[:, ["source_id", "citation", "source_type", "url"]].assign(
                provenance_table="curated_source_ledger"
            )
        )

    registry = source_registry.rename(
        columns={
            "source_key": "source_id",
            "source_name": "citation",
            "source_url": "url",
        }
    )
    registry = registry.loc[:, ["source_id", "citation", "url"]].assign(
        source_type="source_registry", provenance_table="source_registry"
    )

    lookup = pd.concat([*ledgers, registry], ignore_index=True)
    lookup["source_id"] = lookup["source_id"].astype(str)
    return lookup.drop_duplicates(["source_id"], keep="first")


def _resolve_source_ids(source_ids: str, source_lookup: pd.DataFrame) -> dict[str, str]:
    resolved_ids = []
    citations = []
    urls = []
    types = []
    for raw_id in str(source_ids).split(";"):
        source_id = raw_id.strip()
        if not source_id:
            continue
        resolved_id = SOURCE_ALIASES.get(source_id, source_id)
        row = source_lookup.loc[source_lookup["source_id"].eq(resolved_id)]
        resolved_ids.append(resolved_id)
        if row.empty:
            citations.append("unresolved")
            urls.append("unresolved")
            types.append("unresolved")
            continue
        first = row.iloc[0]
        citations.append(str(first["citation"]))
        urls.append(str(first["url"]))
        types.append(str(first["source_type"]))
    return {
        "resolved_source_ids": "; ".join(resolved_ids),
        "source_citations": " || ".join(citations),
        "source_urls": " || ".join(urls),
        "resolved_source_types": "; ".join(types),
    }


def build_country_inventory(direct: pd.DataFrame, source_lookup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in direct.iterrows():
        iso3 = str(row["iso3"])
        meta = PROCESS_METADATA.get(
            iso3,
            ProcessMetadata(
                "not classified",
                "not classified",
                "not_classified",
                "not classified",
                0.75,
                "No manual process metadata was assigned.",
            ),
        )
        positivity = pd.to_numeric(row.get("test_positivity_percent"), errors="coerce")
        tests = pd.to_numeric(row.get("testing_denominator"), errors="coerce")
        positives = np.nan
        if pd.notna(positivity) and pd.notna(tests):
            positives = round(float(tests) * float(positivity) / 100.0)
        source_meta = _resolve_source_ids(
            str(row["direct_testing_series_source_ids"]), source_lookup
        )
        rows.append(
            {
                "iso3": iso3,
                "country": row["country"],
                "year": int(row["year"]),
                "tests_performed": tests,
                "positive_tests_approx": positives,
                "positivity_percent": positivity,
                "assay_mix": meta.assay_mix,
                "denominator_scope": meta.denominator_scope,
                "denominator_scope_class": meta.denominator_scope_class,
                "eligibility_or_tested_population": meta.eligibility_or_population,
                "evidence_tier": _evidence_tier(row["direct_testing_series_support_level"]),
                "support_level": row["direct_testing_series_support_level"],
                "source_ids": row["direct_testing_series_source_ids"],
                "resolved_source_ids": source_meta["resolved_source_ids"],
                "source_citations": source_meta["source_citations"],
                "source_urls": source_meta["source_urls"],
                "source_type": row["direct_testing_series_source_type"],
                "resolved_source_types": source_meta["resolved_source_types"],
                "notes": row["direct_testing_series_notes"],
            }
        )
    return pd.DataFrame(rows).sort_values(["iso3", "year", "source_ids"])


def _country_fold_metrics(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values("year")
    tests = pd.to_numeric(group["tests_performed"], errors="coerce")
    positivity = pd.to_numeric(group["positivity_percent"], errors="coerce")
    years = pd.to_numeric(group["year"], errors="coerce")

    max_adjacent_fold = np.nan
    max_abs_annual_log_fold = np.nan
    latest_previous_fold = np.nan
    if len(group) >= 2 and tests.notna().all():
        folds: list[float] = []
        annual_logs: list[float] = []
        for i in range(1, len(group)):
            previous = float(tests.iloc[i - 1])
            current = float(tests.iloc[i])
            gap = max(int(years.iloc[i] - years.iloc[i - 1]), 1)
            fold = _safe_ratio(current, previous)
            if pd.notna(fold):
                folds.append(max(fold, 1.0 / fold))
                annual_logs.append(abs(np.log(fold)) / gap)
        if folds:
            max_adjacent_fold = float(np.max(folds))
            max_abs_annual_log_fold = float(np.max(annual_logs))
            latest_previous_fold = _safe_ratio(float(tests.iloc[-1]), float(tests.iloc[-2]))

    latest_previous_positivity_pp = np.nan
    if len(group) >= 2 and positivity.notna().iloc[-1] and positivity.notna().iloc[-2]:
        latest_previous_positivity_pp = float(positivity.iloc[-1] - positivity.iloc[-2])

    return {
        "testing_volume_overall_fold_min_to_max": _safe_ratio(tests.max(), tests.min()),
        "max_adjacent_observed_testing_fold_change": max_adjacent_fold,
        "max_abs_annual_log_testing_change": max_abs_annual_log_fold,
        "latest_previous_testing_fold_change": latest_previous_fold,
        "positivity_percent_min": float(positivity.min()) if positivity.notna().any() else np.nan,
        "positivity_percent_max": float(positivity.max()) if positivity.notna().any() else np.nan,
        "positivity_pp_range": (
            float(positivity.max() - positivity.min()) if positivity.notna().any() else np.nan
        ),
        "latest_previous_positivity_pp_change": latest_previous_positivity_pp,
    }


def build_country_mapping(
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    precision_frontier: pd.DataFrame,
    contamination_frontier: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contamination_testing = contamination_frontier.loc[
        contamination_frontier["contamination_regime"].eq("testing_only")
        & contamination_frontier["package_id"].eq("notifications+testing")
    ].copy()

    mapping_rows: list[dict[str, object]] = []
    crosswalk_rows: list[dict[str, object]] = []
    for iso3, group in inventory.groupby("iso3", sort=True):
        meta = PROCESS_METADATA[iso3]
        country = str(group["country"].iloc[0])
        support = summary.loc[summary["iso3"].eq(iso3)]
        support_row = support.iloc[0] if not support.empty else pd.Series(dtype=object)
        metrics = _country_fold_metrics(group)

        contamination_row = contamination_testing.loc[
            pd.to_numeric(
                contamination_testing["testing_cross_loading_fraction"], errors="coerce"
            ).eq(meta.cross_loading_fraction)
        ]

        age_rows = int(group["eligibility_or_tested_population"].str.contains("Infants").sum())
        has_positivity = bool(group["positivity_percent"].notna().any())
        has_positive_tests = bool(group["positive_tests_approx"].notna().any())

        row = {
            "iso3": iso3,
            "country": country,
            "testing_denominator_years": int(group["year"].nunique()),
            "first_testing_year": int(group["year"].min()),
            "last_testing_year": int(group["year"].max()),
            "evidence_tier": _evidence_tier(
                support_row.get("direct_testing_series_support_class", "")
            ),
            "tests_performed_available": True,
            "positive_tests_available_or_reconstructable": has_positive_tests,
            "positivity_available": has_positivity,
            "assay_mix": meta.assay_mix,
            "denominator_scope_class": meta.denominator_scope_class,
            "eligibility_or_tested_population": meta.eligibility_or_population,
            "age_testing_population_split_available": iso3 == "NLD" or age_rows > 0,
            "testing_volume_overall_fold_min_to_max": metrics[
                "testing_volume_overall_fold_min_to_max"
            ],
            "max_adjacent_observed_testing_fold_change": metrics[
                "max_adjacent_observed_testing_fold_change"
            ],
            "latest_previous_testing_fold_change": metrics[
                "latest_previous_testing_fold_change"
            ],
            "positivity_percent_min": metrics["positivity_percent_min"],
            "positivity_percent_max": metrics["positivity_percent_max"],
            "positivity_pp_range": metrics["positivity_pp_range"],
            "latest_previous_positivity_pp_change": metrics[
                "latest_previous_positivity_pp_change"
            ],
            "observed_max_abs_annual_log_testing_change": metrics[
                "max_abs_annual_log_testing_change"
            ],
            "testing_volume_change_interpretation": (
                "observed_signal_or_process_change_not_measurement_error_sd"
            ),
            "mapped_testing_cross_loading_fraction": meta.cross_loading_fraction,
            "testing_cross_loading_rationale": meta.cross_loading_rationale,
            "notifications_plus_testing_auc_at_mapped_testing_cross_loading": (
                float(contamination_row.iloc[0]["grouped_cv_auc"])
                if not contamination_row.empty
                else np.nan
            ),
            "scope_or_eligibility_warning": (
                meta.denominator_scope_class
                in {
                    "scope_changed_snapshot",
                    "provincial_historical",
                    "sentinel_snapshot",
                    "regional_single_study",
                    "single_study",
                    "sentinel_single_study",
                }
            ),
            "source_ids": "; ".join(sorted(group["source_ids"].astype(str).unique())),
        }
        row["module_readiness"] = _module_readiness(row)
        mapping_rows.append(row)

        crosswalk_rows.append(
            {
                "iso3": iso3,
                "country": country,
                "crosswalk_axis": "testing_cross_loading",
                "observed_proxy": meta.cross_loading_fraction,
                "mapped_grid_value": meta.cross_loading_fraction,
                "package_id": "notifications+testing",
                "auc": row[
                    "notifications_plus_testing_auc_at_mapped_testing_cross_loading"
                ],
                "auc_lower_95": (
                    float(contamination_row.iloc[0]["grouped_cv_auc_lower_95"])
                    if not contamination_row.empty
                    else np.nan
                ),
                "interpretation": meta.cross_loading_rationale,
            }
        )

    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["module_readiness", "iso3"], ascending=[True, True]
    )
    crosswalk = pd.DataFrame(crosswalk_rows).sort_values(["iso3", "crosswalk_axis"])
    return mapping, crosswalk


def _module_readiness(row: dict[str, object]) -> str:
    if (
        row["evidence_tier"] == "direct_repeated_model_window"
        and row["positivity_available"]
        and not row["scope_or_eligibility_warning"]
    ):
        return "candidate_supplementary_lookup_ready"
    if (
        row["evidence_tier"] == "direct_repeated_model_window"
        and not row["scope_or_eligibility_warning"]
    ):
        return "denominator_ready_needs_positivity_or_assay_split"
    if row["evidence_tier"] in {"historical_repeated_boundary", "single_model_window_anchor"}:
        return "boundary_evidence_needs_new_repeated_current_data"
    return "single_study_boundary_needs_new_data"


def build_manuscript_summary(mapping: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "testing_denominator_years",
        "evidence_tier",
        "positivity_available",
        "assay_mix",
        "denominator_scope_class",
        "testing_volume_overall_fold_min_to_max",
        "max_adjacent_observed_testing_fold_change",
        "positivity_pp_range",
        "testing_volume_change_interpretation",
        "mapped_testing_cross_loading_fraction",
        "module_readiness",
    ]
    return mapping.loc[:, columns].sort_values(["module_readiness", "country"])


def _fmt_number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    data = df.loc[:, columns]
    if limit is not None:
        data = data.head(limit)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in data.iterrows():
        rows.append("| " + " | ".join(_fmt_number(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *rows])


def build_report(
    inventory: pd.DataFrame,
    mapping: pd.DataFrame,
    age_testing: pd.DataFrame,
) -> str:
    repeated = mapping.loc[mapping["evidence_tier"].eq("direct_repeated_model_window")]
    positivity = mapping.loc[mapping["positivity_available"]]
    ready = mapping.loc[mapping["module_readiness"].eq("candidate_supplementary_lookup_ready")]
    needs_new = mapping.loc[mapping["module_readiness"].str.contains("needs_new")]

    max_fold = mapping["testing_volume_overall_fold_min_to_max"].max()
    max_pp = mapping["positivity_pp_range"].max()
    largest_volume_changes = mapping.sort_values(
        "max_adjacent_observed_testing_fold_change", ascending=False
    ).head(3)

    lines = [
        "# Qualitative testing-process crosswalk, 2026-08-13",
        "",
        "## Scope",
        "",
        (
            "This audit separates the archived testing-process evidence into tests "
            "performed, approximate positive tests, positivity, assay mix, denominator "
            "scope and eligibility/tested-population notes. It is a qualitative "
            "crosswalk between observed testing systems and the assumptions behind "
            "the simulated testing index, not a calibration of observed annual "
            "volume changes to measurement-error SD. It is not external classifier "
            "validation and does not estimate empirical burden or ascertainment "
            "contributions."
        ),
        "",
        "## Evidence inventory",
        "",
        (
            f"Countries or settings with direct testing-denominator rows: "
            f"{mapping.shape[0]}. Repeated model-window denominator series: "
            f"{repeated.shape[0]} ({', '.join(repeated['iso3'])}). Positivity is "
            f"available or reconstructable in {positivity.shape[0]} settings "
            f"({', '.join(positivity['iso3'])})."
        ),
        (
            f"Observed testing-volume min-to-max fold changes reach {_fmt_number(max_fold)}x "
            f"across archived series. Positivity ranges reach {_fmt_number(max_pp)} "
            "percentage points among settings with positivity."
        ),
        "",
        _table(
            mapping,
            [
                "iso3",
                "testing_denominator_years",
                "evidence_tier",
                "positivity_available",
                "testing_volume_overall_fold_min_to_max",
                "positivity_pp_range",
                "module_readiness",
            ],
        ),
        "",
        "## Qualitative testing-process crosswalk",
        "",
        (
            "Observed annual testing-denominator changes are reported as temporal "
            "system descriptors. They may reflect true changes in diagnostic "
            "opportunity, eligibility, care seeking, laboratory coverage or assay "
            "mix; they are not treated as estimates of testing-index measurement "
            "error. The simulated precision frontier remains a separate benchmark "
            "for asking how much additive testing-index error the design can tolerate."
        ),
        "",
        _table(
            largest_volume_changes,
            [
                "iso3",
                "testing_volume_overall_fold_min_to_max",
                "max_adjacent_observed_testing_fold_change",
                "positivity_pp_range",
                "testing_volume_change_interpretation",
            ],
        ),
        "",
        (
            "The cross-loading mapping is qualitative: 0.25 for same-stream PCR "
            "denominator and positivity with stable scope, 0.50 for mixed assays "
            "or missing assay/positivity splits, and 0.75 for scope-changing, "
            "sentinel, provincial or single-study boundary evidence."
        ),
        "",
        _table(
            mapping,
            [
                "iso3",
                "denominator_scope_class",
                "mapped_testing_cross_loading_fraction",
                "notifications_plus_testing_auc_at_mapped_testing_cross_loading",
                "testing_cross_loading_rationale",
            ],
        ),
        "",
        "## Age, assay and tested-population split",
        "",
        (
            "The only archived explicit age-by-assay testing split is the Dutch "
            "2010-2013 regional summary: infants younger than one year were "
            "recommended PCR or culture, but 44% received serology; older patients "
            "with more than three weeks of cough were predominantly tested by "
            "serology (94%). This is valuable design evidence, but it is regional "
            "and historical rather than a current denominator comparability stream."
        ),
        "",
        _table(
            age_testing,
            [
                "iso3",
                "age_group",
                "recommended_test",
                "total_tests",
                "serology_percent",
                "source_ids",
            ],
        ),
        "",
        "## Submission use",
        "",
        (
            f"Ready for a supplementary lookup table: {', '.join(ready['iso3']) or 'none'}. "
            "These settings have repeated model-window denominators and positivity "
            "with a defensible process scope."
        ),
        (
            f"Need new data before being presented as calibrated testing modules: "
            f"{', '.join(needs_new['iso3']) or 'none'}. The missing elements are "
            "usually current repeated denominators, assay-specific splits, positivity "
            "or stable eligibility/catchment definitions."
        ),
        "",
        "Suggested manuscript wording:",
        "",
        (
            "Archived official and peer-reviewed testing data show that real "
            "ascertainment modules can be parameterized as tests performed, positives, "
            "positivity, assay mix, denominator scope and eligibility rather than as "
            "a single abstract index. Across the archived repeated model-window series, "
            "testing-denominator changes range from stable laboratory volumes to large "
            "resurgence-linked expansions, while positivity is available for France, "
            "England and Thailand. These data support a supplementary comparability "
            "lookup for prospective testing modules, but they do not calibrate "
            "measurement-error SD, validate the classifier or decompose burden and "
            "ascertainment without prospectively specified, assay-specific denominator "
            "streams."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    direct = _read_csv(DIRECT_EVIDENCE)
    summary = _read_csv(DIRECT_SUMMARY)
    age_testing = _read_csv(AGE_TESTING)
    _ = _read_csv(DIAGNOSTIC_TIMELINE)
    diagnostic_sources = _read_csv(DIAGNOSTIC_SOURCES)
    outcome_sources = _read_csv(OUTCOME_SOURCES)
    source_registry = _read_csv(SOURCE_REGISTRY)
    precision_frontier = _read_csv(PRECISION_FRONTIER)
    contamination_frontier = _read_csv(CONTAMINATION_FRONTIER)

    source_lookup = build_source_lookup(diagnostic_sources, outcome_sources, source_registry)
    inventory = build_country_inventory(direct, source_lookup)
    mapping, crosswalk = build_country_mapping(
        inventory, summary, precision_frontier, contamination_frontier
    )
    manuscript_summary = build_manuscript_summary(mapping)
    report = build_report(inventory, mapping, age_testing)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(COUNTRY_INVENTORY_OUTPUT, index=False)
    mapping.to_csv(COUNTRY_MAPPING_OUTPUT, index=False)
    crosswalk.to_csv(FRONTIER_CROSSWALK_OUTPUT, index=False)
    manuscript_summary.to_csv(MANUSCRIPT_TABLE, index=False)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    MANUSCRIPT_REPORT.write_text(report, encoding="utf-8")

    print(f"Wrote {COUNTRY_INVENTORY_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {COUNTRY_MAPPING_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {FRONTIER_CROSSWALK_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {MANUSCRIPT_TABLE.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_REPORT.relative_to(ROOT)}")
    print(f"Wrote {MANUSCRIPT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
