"""Promote one isolated tagged Model 2C country refit into primary outputs.

The script is intentionally narrow: it replaces one ISO3 country's rows in the
manuscript-default Model 2C CSVs only after the tagged fit satisfies the
registered primary sampling gate. It does not rerun scenario replay or
manuscript-table generation; run those downstream scripts after promotion.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

PRIMARY_TABLES = (
    "model2c_fit_summary.csv",
    "model2c_posterior_draws.csv",
    "model2c_posterior_predictive.csv",
    "model2c_observation_process_audit.csv",
    "model2c_stan_data_metadata.csv",
    "model2c_hmc_diagnostics.csv",
)
COUNTRY_ORDER = ("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
SORT_COLUMNS = {
    "model2c_fit_summary.csv": ["_iso_order"],
    "model2c_posterior_draws.csv": ["_iso_order", "draw", "chain", "iteration"],
    "model2c_posterior_predictive.csv": ["_iso_order", "draw", "year", "month", "model_age_group"],
    "model2c_observation_process_audit.csv": ["_iso_order"],
    "model2c_stan_data_metadata.csv": ["_iso_order"],
    "model2c_hmc_diagnostics.csv": ["_iso_order", "chain"],
}


def boolish(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass"}


def tagged_path(stem: str, tag: str) -> Path:
    return OUTPUTS / f"{stem.removesuffix('.csv')}_{tag}.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Required file is missing: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def normalize_promoted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    promoted = frame.copy()
    if "model2c_output_tag" in promoted.columns:
        promoted["model2c_output_tag"] = "main"
    return promoted


def assert_tagged_gate(tag: str, iso3: str) -> pd.Series:
    fit = read_csv(tagged_path("model2c_fit_summary.csv", tag))
    rows = fit.loc[fit["iso3"].astype(str).str.upper().eq(iso3)]
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one tagged fit row for {iso3}, found {len(rows)}.")
    row = rows.iloc[0]
    max_rhat = pd.to_numeric(pd.Series([row.get("max_rhat")]), errors="coerce").iloc[0]
    min_ess = pd.to_numeric(pd.Series([row.get("min_ess_bulk")]), errors="coerce").iloc[0]
    hmc_pass = boolish(row.get("registered_hmc_gate_pass", False))
    if not (np.isfinite(max_rhat) and max_rhat < 1.01 and np.isfinite(min_ess) and min_ess >= 400 and hmc_pass):
        raise SystemExit(
            f"Tagged fit {tag}/{iso3} does not pass the registered gate "
            f"(max_rhat={max_rhat:.5g}, min_ess_bulk={min_ess:.5g}, hmc_pass={hmc_pass})."
        )
    return row


def sort_promoted(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if "iso3" not in frame.columns:
        return frame
    order = {iso3: index for index, iso3 in enumerate(COUNTRY_ORDER)}
    out = frame.copy()
    out["_iso_order"] = out["iso3"].astype(str).str.upper().map(order).fillna(len(order)).astype(int)
    sort_columns = [column for column in SORT_COLUMNS.get(table_name, ["_iso_order"]) if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns, kind="mergesort")
    return out.drop(columns=["_iso_order"])


def promote_table(table_name: str, tag: str, iso3: str, dry_run: bool) -> dict[str, object]:
    source = read_csv(tagged_path(table_name, tag))
    source_rows = source.loc[source["iso3"].astype(str).str.upper().eq(iso3)]
    if source_rows.empty:
        raise SystemExit(f"Tagged table {table_name} has no rows for {iso3}.")
    source_rows = normalize_promoted_rows(source_rows)

    destinations = [OUTPUTS / table_name, OUTPUTS / table_name.replace(".csv", "_full.csv")]
    written: list[str] = []
    before_rows: dict[str, int] = {}
    after_rows: dict[str, int] = {}
    replaced_rows: dict[str, int] = {}
    for destination in destinations:
        if not destination.exists():
            continue
        primary = read_csv(destination)
        if "iso3" not in primary.columns:
            raise SystemExit(f"Destination table has no iso3 column: {destination.relative_to(ROOT)}")
        before_rows[str(destination.relative_to(ROOT))] = len(primary)
        old_country = primary["iso3"].astype(str).str.upper().eq(iso3)
        replaced_rows[str(destination.relative_to(ROOT))] = int(old_country.sum())
        columns = list(dict.fromkeys([*primary.columns, *source_rows.columns]))
        promoted = pd.concat(
            [
                primary.loc[~old_country].reindex(columns=columns),
                source_rows.reindex(columns=columns),
            ],
            ignore_index=True,
            sort=False,
        )
        promoted = sort_promoted(promoted, table_name)
        after_rows[str(destination.relative_to(ROOT))] = len(promoted)
        if not dry_run:
            promoted.to_csv(destination, index=False)
        written.append(str(destination.relative_to(ROOT)))
    if not written:
        raise SystemExit(f"No primary destination files found for {table_name}.")
    return {
        "table_name": table_name,
        "source_file": str(tagged_path(table_name, tag).relative_to(ROOT)),
        "source_rows": len(source_rows),
        "destinations": ";".join(written),
        "destination_rows_before": before_rows,
        "destination_rows_after": after_rows,
        "replaced_rows": replaced_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Tagged refit suffix, e.g. jpn_rhat_fix_seed20260813.")
    parser.add_argument("--iso3", required=True, help="Country to promote.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tag = args.tag.strip().lower().replace("-", "_")
    iso3 = args.iso3.strip().upper()
    gate_row = assert_tagged_gate(tag, iso3)
    audit_rows = []
    for table_name in PRIMARY_TABLES:
        audit_rows.append(promote_table(table_name, tag, iso3, args.dry_run))

    audit = pd.DataFrame(audit_rows)
    audit.insert(0, "promoted_iso3", iso3)
    audit.insert(0, "promoted_tag", tag)
    audit.insert(0, "promoted_at_utc", datetime.now(UTC).isoformat(timespec="seconds"))
    audit["tagged_max_rhat"] = gate_row.get("max_rhat")
    audit["tagged_min_ess_bulk"] = gate_row.get("min_ess_bulk")
    audit["tagged_registered_hmc_gate_pass"] = gate_row.get("registered_hmc_gate_pass")
    audit_path = OUTPUTS / "model2c_country_refit_promotion_audit.csv"
    if not args.dry_run:
        audit.to_csv(audit_path, index=False)
    print(
        f"{'Would promote' if args.dry_run else 'Promoted'} {iso3} from {tag}; "
        f"gate max_rhat={float(gate_row['max_rhat']):.5g}, min_ess_bulk={float(gate_row['min_ess_bulk']):.5g}."
    )


if __name__ == "__main__":
    main()
