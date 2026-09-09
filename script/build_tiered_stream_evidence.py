#!/usr/bin/env python3
"""Build observed testing-stream evidence for the burden-ascertainment analysis.

Positive tests decompose as eligible tests x test positivity. Notifications and
distinct confirmed cases are separate units. A volume-only increase under stable
sampling and assay performance leaves expected positivity unchanged; changes in
eligibility or sample composition can move positivity without a burden change.
Decomposition is allowed only when tests and positives share a sampling frame.
Unlinked US streams and the two
non-comparable Thai snapshots are retained as measurement-boundary examples,
not as longitudinal positivity decompositions.

Annual severe-outcome counts come from the official country-year extracts.
The surveillance-outcome evidence file is used only for testing denominators and
positivity: its France 2024 row stores 2013-2024 cumulative RENACOQ totals in the
annual count columns, so it must not be read as an annual severe-outcome series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURVEILLANCE_EVIDENCE = (
    ROOT / "data" / "raw" / "severe_outcomes" / "surveillance_outcome_evidence.csv"
)
EXACT_TESTING = ROOT / "data" / "raw" / "external_empirical_validation" / "testing_count_extracts.csv"
OFFICIAL_EXTRACTS = (
    ROOT
    / "data"
    / "raw"
    / "external_empirical_validation"
    / "official_country_year_extracts.csv"
)
SEVERE_AGE_TIME = (
    ROOT / "data" / "raw" / "severe_outcomes" / "severe_outcome_age_time_evidence.csv"
)
RESURGENCE_INDEX = ROOT / "outputs" / "resurgence_index_by_country.csv"

TESTING_OUTPUT = ROOT / "outputs" / "tiered_testing_decomposition.csv"
TESTING_MD = ROOT / "manuscript" / "tables_md" / "tiered_testing_decomposition.md"
FIGURE_DATA = ROOT / "manuscript" / "figure_data" / "figure_2_testing_decomposition.csv"

# Testing-stream series. Each entry fixes the scope filter so a country's rows stay on one
# measurement system across years.
TESTING_STREAMS = {
    "FRA": dict(
        country="France",
        scope_contains="3Labos",
        system="3Labos sentinel PCR laboratory network",
        # The 2024 row is scope-labelled RENACOQ because it also carries hospital
        # data, but its testing_denominator/positivity are the 3Labos Figure 12
        # series (188,539 tests at 20.6%) from the same official report.
        scope_override_years={2024: "national RENACOQ network"},
        positivity_source="reported",
        comparison_status="same_system_longitudinal",
        comparison_note="Repeated 3Labos tests and positive-test proportions.",
    ),
    "GBR": dict(
        country="England",
        scope_contains="England",
        system="UKHSA laboratory-confirmed testing (serum, oral fluid, PCR)",
        scope_exact="England",
        positivity_source="reported",
        comparison_status="same_system_longitudinal",
        comparison_note="Repeated UKHSA tests and positive-test proportions.",
    ),
    "USA": dict(
        country="United States",
        scope_contains="commercial",
        system="Commercial laboratory network (tests) with NNDSS notifications",
        positivity_source="unavailable_unlinked",
        comparison_status="unlinked_cross_system",
        comparison_note="Commercial-laboratory orders and national notifications are unlinked; their ratio is not test positivity.",
    ),
    "THA": dict(
        country="Thailand",
        scope_allowed=(
            "Queen Sirikit National Institute of Child Health, Bangkok",
            "national",
        ),
        system="Outbreak-investigation laboratory submissions",
        positivity_source="reported",
        comparison_status="noncomparable_snapshots",
        comparison_note="The 2019 Bangkok referral-hospital and 2023 national outbreak-investigation samples differ in geography and eligibility.",
    ),
}

SEVERE_STREAMS = ["infant_hospitalizations", "infant_deaths", "all_age_deaths"]


def load_exact_testing(path: Path = EXACT_TESTING) -> pd.DataFrame:
    """Validated empirical count units, separate from frozen calibration inputs."""
    frame = pd.read_csv(path)
    if frame.duplicated(["iso3", "year"]).any():
        raise ValueError("Duplicate exact testing country-years")
    for column in ["eligible_tests", "positive_tests", "negative_tests"]:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Exact {column} must be non-negative integers")
        frame[column] = values.astype(int)
    if frame.eligible_tests.le(0).any() or not (
        frame.positive_tests + frame.negative_tests == frame.eligible_tests
    ).all():
        raise ValueError("Positive plus negative tests must equal the eligible denominator")
    if (frame.submitted_samples.dropna() < frame.loc[frame.submitted_samples.notna(), "eligible_tests"]).any():
        raise ValueError("Submitted samples cannot be fewer than eligible samples")
    frame["positivity_percent"] = 100 * frame.positive_tests / frame.eligible_tests
    return frame


def correct_empirical_testing_series(series: pd.DataFrame) -> pd.DataFrame:
    """Overlay audited counts without recomputing the frozen detection-prior fields."""
    frame = series.copy()
    if "calibration_testing_denominator" not in frame:
        frame["calibration_testing_denominator"] = frame["testing_denominator"]
    if "calibration_test_positivity_percent" not in frame:
        frame["calibration_test_positivity_percent"] = frame["test_positivity_percent"]
    exact = load_exact_testing().set_index(["iso3", "year"])
    for column in exact.columns:
        if column not in frame:
            frame[column] = (
                np.nan if pd.api.types.is_numeric_dtype(exact[column])
                else pd.Series(index=frame.index, dtype=object)
            )
    for index, row in frame.iterrows():
        key = (row.iso3, int(row.year))
        if key not in exact.index:
            continue
        source = exact.loc[key]
        for column in source.index:
            frame.at[index, column] = source[column]
        frame.at[index, "testing_denominator"] = source.eligible_tests
        frame.at[index, "test_positivity_percent"] = source.positivity_percent
        frame.at[index, "direct_testing_series_source_ids"] = source.source_id
        frame.at[index, "direct_testing_series_notes"] = source.notes
    frame["calibration_boundary"] = (
        "Detection-prior factors retain the archived calibration denominator; "
        "exact eligible/positive counts revise empirical displays only."
    )
    return frame


def _load_surveillance() -> pd.DataFrame:
    d = pd.read_csv(SURVEILLANCE_EVIDENCE)
    for c in [
        "case_count",
        "testing_denominator",
        "test_positivity_percent",
        "infant_case_count",
        "infant_deaths",
        "total_deaths",
        "hospitalizations",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def build_testing_streams(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exact = load_exact_testing()
    for iso, cfg in TESTING_STREAMS.items():
        if iso in {"FRA", "GBR"}:
            for _, source in exact.loc[exact.iso3.eq(iso)].iterrows():
                rows.append(dict(
                    **source.to_dict(),
                    system=cfg["system"],
                    tests=source.eligible_tests,
                    positivity_source="exact_source_integers",
                    comparison_status=cfg["comparison_status"],
                    comparison_note=("Same-network positive tests = eligible tests × positivity; "
                                     "burden interpretation requires stable sampling composition and assay performance."),
                    notifications=np.nan,
                ))
            continue
        g = d[d.iso3 == iso].copy()
        if cfg.get("scope_exact"):
            keep = g.geographic_scope == cfg["scope_exact"]
        elif cfg.get("scope_allowed"):
            keep = g.geographic_scope.isin(cfg["scope_allowed"])
        else:
            keep = g.geographic_scope.str.contains(
                cfg["scope_contains"], na=False, case=False
            )
        for yr, scope in cfg.get("scope_override_years", {}).items():
            keep |= (g.year == yr) & (g.geographic_scope == scope)
        g = g[keep]

        tests = (
            g[g.testing_denominator.notna()]
            .groupby("year")["testing_denominator"]
            .first()
        )

        if cfg["positivity_source"] == "reported":
            pos = (
                g[g.test_positivity_percent.notna()]
                .groupby("year")["test_positivity_percent"]
                .first()
            )
        else:
            # National notifications cannot supply a commercial-laboratory
            # positivity numerator. Keep both observed series with distinct units.
            nat = d[(d.iso3 == iso) & (d.geographic_scope == "national")]
            cases = nat[nat.case_count.notna()].groupby("year")["case_count"].first()
            pos = pd.Series(np.nan, index=tests.index)

        for yr in sorted(set(tests.index) & set(pos.index)):
            rows.append(
                dict(
                    iso3=iso,
                    country=cfg["country"],
                    year=int(yr),
                    system=cfg["system"],
                    tests=float(tests.loc[yr]),
                    positivity_percent=float(pos.loc[yr]),
                    positivity_source=cfg["positivity_source"],
                    notifications=float(cases.loc[yr]) if iso == "USA" and yr in cases.index else np.nan,
                    positive_tests=np.nan,
                    geographic_scope=str(
                        g.loc[g.year.eq(yr), "geographic_scope"].iloc[0]
                    ),
                    comparison_status=cfg["comparison_status"],
                    comparison_note=cfg["comparison_note"],
                )
            )

    t = pd.DataFrame(rows).sort_values(["iso3", "year"]).reset_index(drop=True)

    # Decompose each year against the pre-pandemic 2019 reference.
    out = []
    for iso, g in t.groupby("iso3"):
        g = g.sort_values("year").copy()
        ref = g[g.year == 2019]
        if ref.empty:
            ref = g.iloc[[0]]
        r = ref.iloc[0]
        g["ref_year"] = int(r.year)
        g["tests_fold_vs_ref"] = g.tests / r.tests
        g["positivity_fold_vs_ref"] = g.positivity_percent / r.positivity_percent
        g["positive_tests_fold_vs_ref"] = g.tests_fold_vs_ref * g.positivity_fold_vs_ref
        # Algebraic shares of positive-test growth, not causal attribution.
        lt = np.log(g.tests_fold_vs_ref)
        lp = np.log(g.positivity_fold_vs_ref)
        tot = lt + lp
        g["log_share_testing"] = np.where(np.abs(tot) > 1e-9, lt / tot, np.nan)
        g["log_share_positivity"] = np.where(np.abs(tot) > 1e-9, lp / tot, np.nan)
        g["positivity_exceeds_prepandemic"] = g.positivity_percent > r.positivity_percent
        if r.comparison_status != "same_system_longitudinal":
            # Preserve raw observations for the boundary panels, but never emit
            # a longitudinal fold or classification from incomparable streams.
            for column in (
                "tests_fold_vs_ref",
                "positivity_fold_vs_ref",
                "positive_tests_fold_vs_ref",
                "log_share_testing",
                "log_share_positivity",
            ):
                g[column] = np.nan
            g["positivity_exceeds_prepandemic"] = False
        out.append(g)
    return pd.concat(out, ignore_index=True)


def classify(t1: pd.DataFrame) -> pd.DataFrame:
    """Classify each Tier 1 setting's latest resurgence year."""
    rows = []
    for iso, g in t1.groupby("iso3"):
        g = g.sort_values("year")
        if g.comparison_status.iloc[0] != "same_system_longitudinal":
            continue
        latest = g.iloc[-1]
        if latest.year == latest.ref_year:
            continue
        # Descriptive joint direction; changing selection or assays can also
        # raise positivity, so this is not an unconditional mechanism classifier.
        if latest.positivity_fold_vs_ref > 1 and latest.tests_fold_vs_ref > 1:
            call = "positivity_increase_with_testing_expansion"
        elif latest.positivity_fold_vs_ref < 1 and latest.tests_fold_vs_ref > 1:
            call = "testing_expansion_with_lower_positivity"
        else:
            call = "mixed_or_unresolved"
        rows.append(
            dict(
                iso3=iso,
                country=latest.country,
                system=latest.system,
                ref_year=int(latest.ref_year),
                year=int(latest.year),
                tests_fold=latest.tests_fold_vs_ref,
                positivity_fold=latest.positivity_fold_vs_ref,
                positivity_percent=latest.positivity_percent,
                positive_tests_fold=latest.positive_tests_fold_vs_ref,
                log_share_testing=latest.log_share_testing,
                classification=call,
                positivity_source=latest.positivity_source,
            )
        )
    return pd.DataFrame(rows).sort_values("iso3").reset_index(drop=True)


def build_severe_streams() -> pd.DataFrame:
    """Annual severe-outcome streams from official extracts and age-time evidence."""
    # Start with the 10-row official country-year extracts.
    e = pd.read_csv(OFFICIAL_EXTRACTS)
    e["value"] = pd.to_numeric(e.value, errors="coerce")
    sev = e[e.metric.isin(SEVERE_STREAMS)]
    rows = []
    for (iso, metric), g in sev.groupby(["iso3", "metric"]):
        g = g.sort_values("year")
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[-1]
        rows.append(
            dict(
                iso3=iso,
                country=a.country,
                metric=metric,
                year_from=int(a.year),
                year_to=int(b.year),
                value_from=a.value,
                value_to=b.value,
                fold=(b.value / a.value) if a.value > 0 else np.nan,
                geographic_scope=b.geographic_scope,
                source="official_extracts",
            )
        )

    # Add aggregated infant streams from the age-time evidence (BRA, GBR, JPN, USA).
    INFANT_AGES = ["0 years", "0  years", "<1 year", "<1 Ano", "00-00"]
    at = pd.read_csv(SEVERE_AGE_TIME)
    at["count"] = pd.to_numeric(at["count"], errors="coerce")
    inf = at[at.age_group.isin(INFANT_AGES)].copy()

    for (iso, outcome), g in inf.groupby(["iso3", "outcome_type"]):
        ann = g.groupby("year")["count"].sum().dropna()
        if len(ann) < 2:
            continue
        yrs = sorted(ann.index)
        a_yr, b_yr = yrs[0], yrs[-1]
        a_val, b_val = ann.loc[a_yr], ann.loc[b_yr]

        # Map outcome_type to metric name.
        if outcome == "deaths":
            metric = "infant_deaths"
        elif outcome == "hospitalizations":
            metric = "infant_hospitalizations"
        elif outcome == "reported_cases":
            metric = "infant_reported_cases"
        else:
            continue

        # Skip if already covered by official_extracts (to avoid duplication).
        if any((r["iso3"] == iso and r["metric"] == metric) for r in rows):
            continue

        country = g.country.iloc[0]
        scope = g.geographic_scope.iloc[0]
        rows.append(
            dict(
                iso3=iso,
                country=country,
                metric=metric,
                year_from=int(a_yr),
                year_to=int(b_yr),
                value_from=float(a_val),
                value_to=float(b_val),
                fold=(b_val / a_val) if a_val > 0 else np.nan,
                geographic_scope=scope,
                source="age_time_evidence",
            )
        )

    return pd.DataFrame(rows).sort_values(["iso3", "metric"]).reset_index(drop=True)


def _write_md(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            return f"{v:.4g}"
        return "" if pd.isna(v) else str(v)

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    d = _load_surveillance()
    t1 = build_testing_streams(d)
    calls = classify(t1)
    t2 = build_severe_streams()

    TESTING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    t1.to_csv(TESTING_OUTPUT, index=False)
    FIGURE_DATA.parent.mkdir(parents=True, exist_ok=True)
    t1.to_csv(FIGURE_DATA, index=False)
    _write_md(t1, TESTING_MD)

    print("=== Observed testing streams ===")
    for _, r in t1.iterrows():
        print(
            f"  {r.iso3} {r.year}  tests={r.tests:>10,.0f}  pos={r.positivity_percent:>6.2f}%"
            f"  vs{r.ref_year}: xT={r.tests_fold_vs_ref:>6.2f} xP={r.positivity_fold_vs_ref:>6.2f}"
        )
    print("\n=== Classification (latest year vs pre-pandemic reference) ===")
    for _, r in calls.iterrows():
        print(
            f"  {r.iso3} {r.year} vs {r.ref_year}: xTests={r.tests_fold:.2f} "
            f"xPos={r.positivity_fold:.2f} pos={r.positivity_percent:.2f}% -> {r.classification}"
        )
    print("\n=== Observed severe-outcome streams ===")
    for _, r in t2.iterrows():
        f = f"{r.fold:.2f}" if pd.notna(r.fold) else "n/a"
        print(
            f"  {r.iso3} {r.metric:25s}: {r.value_from:>6,.0f} -> {r.value_to:>6,.0f} "
            f"({int(r.year_from)}->{int(r.year_to)}) fold={f:>6s} src={r.source}"
        )
if __name__ == "__main__":
    main()
