#!/usr/bin/env python3
"""Build a read-only sampling and artifact audit for Model 2C structural refits.

The audit never starts a fit or replays a scenario.  It records absent and
partial tagged outputs explicitly so it is safe to run while the isolated
structural-refit suite is still being populated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "manuscript" / "reports" / "model2c_structural_refit_sampling_diagnostics.md"
DIAGNOSTIC_CSV = "model2c_structural_refit_sampling_diagnostics.csv"

DEFAULT_TAGS = (
    "structural_full",
    "structural_no_exposure_interruption",
    "structural_no_post2023_waning",
    "structural_no_reporting_shift",
    "structural_flexible_seeding",
)
EXPECTED_ISO3 = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
ARTIFACT_TEMPLATES = {
    "fit_summary": "model2c_fit_summary_{tag}.csv",
    "posterior_draws": "model2c_posterior_draws_{tag}.csv",
    "posterior_predictive": "model2c_posterior_predictive_{tag}.csv",
    "observation_process_audit": "model2c_observation_process_audit_{tag}.csv",
    "stan_data_metadata": "model2c_stan_data_metadata_{tag}.csv",
    "country_selection_audit": "model2c_country_selection_audit_{tag}.csv",
}


def _csv_has_data(path: Path) -> bool:
    """Return whether a CSV exists and contains a nonblank record after its header."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if not handle.readline().strip():
                return False
            return any(line.strip() for line in handle)
    except OSError:
        return False


def _read_fit_summary(path: Path) -> tuple[pd.DataFrame, str]:
    required = {"iso3", "max_rhat", "min_ess_bulk"}
    if not _csv_has_data(path):
        return pd.DataFrame(columns=sorted(required)), "missing_or_empty"
    try:
        frame = pd.read_csv(path, usecols=lambda name: name in required | {"country"})
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=sorted(required)), "unreadable_or_invalid"
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=sorted(required)), "missing_required_columns:" + ";".join(sorted(missing))
    frame = frame.copy()
    frame["iso3"] = frame["iso3"].astype(str).str.strip()
    frame["max_rhat"] = pd.to_numeric(frame["max_rhat"], errors="coerce")
    frame["min_ess_bulk"] = pd.to_numeric(frame["min_ess_bulk"], errors="coerce")
    return frame, "readable"


def build_diagnostics(
    outputs_dir: Path = OUTPUTS,
    tags: Iterable[str] = DEFAULT_TAGS,
) -> pd.DataFrame:
    """Return one audit row per requested tag and expected country."""
    expected = set(EXPECTED_ISO3)
    records: list[dict[str, object]] = []
    for tag in tags:
        fit_path = outputs_dir / ARTIFACT_TEMPLATES["fit_summary"].format(tag=tag)
        fit, fit_read_status = _read_fit_summary(fit_path)
        observed = set(fit["iso3"].dropna().astype(str)) if "iso3" in fit else set()
        counts = fit["iso3"].value_counts() if "iso3" in fit else pd.Series(dtype=int)
        fit_complete = bool(
            fit_read_status == "readable"
            and observed == expected
            and len(fit) == len(EXPECTED_ISO3)
            and counts.eq(1).all()
        )
        artifact_paths = {
            name: outputs_dir / template.format(tag=tag)
            for name, template in ARTIFACT_TEMPLATES.items()
        }
        artifact_nonempty = {name: _csv_has_data(path) for name, path in artifact_paths.items()}
        all_artifacts_nonempty = all(artifact_nonempty.values())
        missing_iso3 = ";".join(sorted(expected - observed))
        unexpected_iso3 = ";".join(sorted(observed - expected))

        for iso3 in EXPECTED_ISO3:
            country_rows = fit.loc[fit["iso3"].eq(iso3)] if "iso3" in fit else pd.DataFrame()
            row_present_once = len(country_rows) == 1
            row = country_rows.iloc[0] if row_present_once else None
            max_rhat = float(row["max_rhat"]) if row is not None and pd.notna(row["max_rhat"]) else np.nan
            min_ess = float(row["min_ess_bulk"]) if row is not None and pd.notna(row["min_ess_bulk"]) else np.nan
            passes_rhat = bool(row_present_once and np.isfinite(max_rhat) and max_rhat < 1.01)
            passes_ess = bool(row_present_once and np.isfinite(min_ess) and min_ess >= 400.0)
            record: dict[str, object] = {
                "tag": tag,
                "iso3": iso3,
                "country": row.get("country", "") if row is not None else "",
                "fit_summary_read_status": fit_read_status,
                "fit_summary_row_present_once": row_present_once,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_ess,
                "passes_core_rhat_lt_1_01": passes_rhat,
                "passes_core_ess_ge_400": passes_ess,
                "passes_core_sampling_gate": passes_rhat and passes_ess,
                "fit_summary_country_rows": int(len(fit)),
                "fit_summary_unique_countries": int(len(observed)),
                "fit_summary_nine_countries_complete": fit_complete,
                "missing_expected_iso3": missing_iso3,
                "unexpected_iso3": unexpected_iso3,
            }
            for name, path in artifact_paths.items():
                record[f"{name}_path"] = str(path)
                record[f"{name}_nonempty"] = artifact_nonempty[name]
            record["all_required_artifacts_nonempty"] = all_artifacts_nonempty
            record["tag_complete"] = fit_complete and all_artifacts_nonempty
            records.append(record)
    return pd.DataFrame.from_records(records)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_report(diagnostics: pd.DataFrame, path: Path = REPORT) -> None:
    summary_rows = []
    for tag, group in diagnostics.groupby("tag", sort=False):
        first = group.iloc[0]
        failed = group.loc[~group["passes_core_sampling_gate"], "iso3"].tolist()
        missing_artifacts = [
            name for name in ARTIFACT_TEMPLATES if not bool(first[f"{name}_nonempty"])
        ]
        summary_rows.append(
            {
                "tag": tag,
                "fit summary": first["fit_summary_read_status"],
                "nine countries complete": bool(first["fit_summary_nine_countries_complete"]),
                "countries passing core gate": int(group["passes_core_sampling_gate"].sum()),
                "failed/missing countries": ";".join(failed),
                "missing/empty artifacts": ";".join(missing_artifacts),
                "tag complete": bool(first["tag_complete"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    detail = diagnostics.loc[
        :,
        ["tag", "iso3", "max_rhat", "min_ess_bulk", "passes_core_sampling_gate"],
    ]
    text = "\n".join(
        [
            "# Model 2C structural-refit sampling diagnostics",
            "",
            "This is a read-only audit of isolated tagged outputs. The core sampling gate is evaluated per country as maximum core R-hat <1.01 and minimum core bulk ESS >=400. A tag is complete only when the fit summary contains exactly one row for each of the nine expected countries and all six required CSV artifacts contain at least one data record.",
            "",
            "## Tag-level completeness",
            "",
            _markdown_table(summary),
            "",
            "## Country-level core gate",
            "",
            _markdown_table(detail),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_tags(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_TAGS
    tags = tuple(part.strip() for value in values for part in value.split(",") if part.strip())
    if not tags:
        raise ValueError("--tags must contain at least one nonblank tag")
    return tuple(dict.fromkeys(tags))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", help="Optional space- or comma-separated structural output tags.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS, help=argparse.SUPPRESS)
    parser.add_argument("--report-path", type=Path, default=REPORT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    tags = _parse_tags(args.tags)
    diagnostics = build_diagnostics(args.outputs_dir, tags)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outputs_dir / DIAGNOSTIC_CSV
    diagnostics.to_csv(csv_path, index=False)
    write_report(diagnostics, args.report_path)
    complete = int(diagnostics.groupby("tag", sort=False)["tag_complete"].first().sum())
    print(f"Wrote {len(diagnostics)} rows for {len(tags)} tags; {complete} tags complete.")


if __name__ == "__main__":
    main()
