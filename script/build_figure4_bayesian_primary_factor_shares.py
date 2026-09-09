#!/usr/bin/env python3
"""Build Figure 4 component split primary mechanism shares from Bayesian Model 2B only.

The main manuscript attribution is intentionally single-route: Figure 4 is
driven by the Bayesian transmission layer over true infections in 2023-2025.
Testing/reporting is not a primary true-infection mechanism here; it is handled
as a secondary reported-case attribution and in the observation-process audit.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURE_DATA = ROOT / "manuscript" / "figure_data"
MODEL2B_FACTOR_SUMMARY_CANDIDATES = [
    OUTPUTS / "model2b_bayes_factor_weight_summary.csv",
]
MODEL2B_OUTCOME_TARGET = "true_infections_2023_2025"
MODEL2B_OUTPUT_VERSION = "model2b_v2_20260616_layered_endemic_background_attribution"

PILOT_ISO3 = ["AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"]

MODEL2B_FACTOR_DEFS = [
    (
        "low_circulation_missed_boosting",
        "Low circulation / missed natural boosting",
        "Model-implied interventional attribution for low circulation and missed natural/exposure boosting.",
    ),
    (
        "routine_immunization_disruption",
        "Routine immunization disruption",
        "Model-implied interventional attribution for DTP coverage disruption.",
    ),
    (
        "vaccine_waning_booster_gap",
        "Waning/booster gap",
        "Model-implied interventional attribution for waning-derived susceptibility and booster gaps.",
    ),
    (
        "birth_cohort_demography",
        "Birth-cohort demography",
        "Model-implied interventional attribution for births, aging and age-structure change.",
    ),
]

FIELDNAMES = [
    "iso3",
    "country",
    "pilot_rank",
    "factor_id",
    "factor_label",
    "factor_order",
    "model_pressure_units",
    "share_percent",
    "share_percent_lower_95",
    "share_percent_upper_95",
    "share_percent_type",
    "signed_share_percent",
    "signed_share_percent_lower_95",
    "signed_share_percent_upper_95",
    "dominant_probability",
    "quantification_layer",
    "attribution_status",
    "model2b_output_version",
    "source_note",
]


def read_first_available(paths: list[Path]) -> list[dict[str, str]]:
    for path in paths:
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
    candidates = ", ".join(str(path.relative_to(ROOT)) for path in paths)
    raise SystemExit(
        "Required Bayesian Model 2B factor summary is missing. "
        f"Checked: {candidates}. Run `make model2b-nodata` to regenerate model outputs, "
        "then rebuild Figure 4 from the complete output summary."
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def min_required_draws() -> int:
    raw = os.environ.get("MODEL2B_MIN_MANUSCRIPT_DRAWS", "2000")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2000


def require_complete_model2b_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    true_rows = [
        row
        for row in rows
        if row.get("iso3") in PILOT_ISO3 and row.get("outcome_target") == MODEL2B_OUTCOME_TARGET
    ]
    by_iso_factor = {(row.get("iso3", ""), row.get("factor_id", "")): row for row in true_rows}
    missing = [
        f"{iso3}:{factor_id}"
        for iso3 in PILOT_ISO3
        for factor_id, *_ in MODEL2B_FACTOR_DEFS
        if (iso3, factor_id) not in by_iso_factor
    ]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise SystemExit(
            "Bayesian Model 2B factor summary is incomplete for Figure 4 primary attribution: "
            f"{preview}{suffix}. Re-run Model 2B for all nine dynamic bridge countries."
        )
    missing_version = [
        f"{row.get('iso3')}:{row.get('factor_id')}"
        for row in true_rows
        if row.get("model2b_output_version") != MODEL2B_OUTPUT_VERSION
    ]
    if missing_version:
        preview = ", ".join(missing_version[:12])
        suffix = "..." if len(missing_version) > 12 else ""
        raise SystemExit(
            "Bayesian Model 2B factor summary is not from the current manuscript model version: "
            f"{preview}{suffix}. Re-run attribution from current posterior draws."
        )
    min_draws = min_required_draws()
    too_few_draws = [
        f"{iso3}:{factor_id} n={to_float(row.get('n_posterior_draws'), 0):.0f}"
        for (iso3, factor_id), row in by_iso_factor.items()
        if to_float(row.get("n_posterior_draws"), 0) < min_draws
    ]
    if too_few_draws:
        preview = ", ".join(too_few_draws[:12])
        suffix = "..." if len(too_few_draws) > 12 else ""
        raise SystemExit(
            f"Bayesian Model 2B factor summary has fewer than {min_draws} posterior draws for Figure 4: "
            f"{preview}{suffix}. Re-run the complete default Model 2B pipeline."
        )
    required_share_columns = [
        "share_percent_median",
        "share_percent_lower_95",
        "share_percent_upper_95",
        "share_percent_basis",
        "signed_share_percent_median",
        "signed_share_percent_lower_95",
        "signed_share_percent_upper_95",
        "attribution_layer",
    ]
    missing_share_values = [
        f"{iso3}:{factor_id}:{column}"
        for (iso3, factor_id), row in by_iso_factor.items()
        for column in required_share_columns
        if (column == "attribution_layer" and row.get(column) != "primary_susceptibility")
        or (
            column == "share_percent_basis"
            and row.get(column) != "positive_normalized_primary_susceptibility_share"
        )
        or (
            column not in {"attribution_layer", "share_percent_basis"}
            and not math.isfinite(to_float(row.get(column)))
        )
    ]
    if missing_share_values:
        preview = ", ".join(missing_share_values[:12])
        suffix = "..." if len(missing_share_values) > 12 else ""
        raise SystemExit(
            "Bayesian Model 2B factor summary is missing share columns required for Figure 4B: "
            f"{preview}{suffix}. Re-run attribution from current posterior draws."
        )
    return by_iso_factor


def build_rows(model2b_summary: list[dict[str, str]]) -> list[dict[str, object]]:
    by_iso_factor = require_complete_model2b_summary(model2b_summary)
    rows: list[dict[str, object]] = []
    for pilot_rank, iso3 in enumerate(PILOT_ISO3, start=1):
        country = by_iso_factor[(iso3, MODEL2B_FACTOR_DEFS[0][0])].get("country", iso3)
        for factor_order, (factor_id, factor_label, note) in enumerate(MODEL2B_FACTOR_DEFS, start=1):
            source = by_iso_factor[(iso3, factor_id)]
            units = to_float(source.get("contribution_median"))
            share = to_float(source.get("share_percent_median"))
            lower = to_float(source.get("share_percent_lower_95"))
            upper = to_float(source.get("share_percent_upper_95"))
            signed_share = to_float(source.get("signed_share_percent_median"))
            signed_lower = to_float(source.get("signed_share_percent_lower_95"))
            signed_upper = to_float(source.get("signed_share_percent_upper_95"))
            dominant_probability = to_float(source.get("dominant_probability"))
            source_note = (
                note
                + f" Figure 4B displays the bounded positive-normalized primary susceptibility share median {share:.1f}%"
                + (f" (95% CrI {lower:.1f}-{upper:.1f}%)." if math.isfinite(lower) and math.isfinite(upper) else ".")
                + (
                    f" Within-layer largest-positive draw probability {dominant_probability:.2f}."
                    if math.isfinite(dominant_probability)
                    else ""
                )
                + (
                    f" Signed denominator share median {signed_share:.1f}% is retained as a diagnostic."
                    if math.isfinite(signed_share)
                    else ""
                )
                + " Display shares are normalized among positive primary susceptibility contributions; "
                + "transmission-context and reporting factors are audited outside this primary panel."
            )
            rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "pilot_rank": pilot_rank,
                    "factor_id": factor_id,
                    "factor_label": factor_label,
                    "factor_order": factor_order,
                    "model_pressure_units": f"{units:.10g}" if math.isfinite(units) else "",
                    "share_percent": f"{share:.10g}" if math.isfinite(share) else "",
                    "share_percent_lower_95": f"{lower:.10g}" if math.isfinite(lower) else "",
                    "share_percent_upper_95": f"{upper:.10g}" if math.isfinite(upper) else "",
                    "share_percent_type": "positive_normalized_primary_susceptibility_share",
                    "signed_share_percent": f"{signed_share:.10g}" if math.isfinite(signed_share) else "",
                    "signed_share_percent_lower_95": f"{signed_lower:.10g}" if math.isfinite(signed_lower) else "",
                    "signed_share_percent_upper_95": f"{signed_upper:.10g}" if math.isfinite(signed_upper) else "",
                    "dominant_probability": f"{dominant_probability:.10g}" if math.isfinite(dominant_probability) else "",
                    "quantification_layer": "primary_susceptibility",
                    "attribution_status": "posterior_primary",
                    "model2b_output_version": MODEL2B_OUTPUT_VERSION,
                    "source_note": source_note,
                }
            )
    return rows


def main() -> None:
    rows = build_rows(read_first_available(MODEL2B_FACTOR_SUMMARY_CANDIDATES))
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DATA / "figure_4_bayesian_primary_factor_shares.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT)} from Bayesian Model 2B transmission attribution")


if __name__ == "__main__":
    main()
