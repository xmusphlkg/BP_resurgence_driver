"""Promote one isolated tagged Model 2C refit into a sensitivity tag.

The script replaces one ISO3 country's rows in an existing sensitivity-output
tag only after the isolated source refit passes the registered HMC gate. It is
for repairing sensitivity diagnostics and never writes manuscript-default Model
2C outputs.
"""

from __future__ import annotations

import argparse
import fcntl
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

SENSITIVITY_TABLES = (
    "model2c_fit_summary.csv",
    "model2c_posterior_draws.csv",
    "model2c_posterior_predictive.csv",
    "model2c_observation_process_audit.csv",
    "model2c_stan_data_metadata.csv",
    "model2c_hmc_diagnostics.csv",
    "model2c_country_selection_audit.csv",
    "model2c_counterfactual_scenario_summary.csv",
    "model2c_counterfactual_scenario_monthly.csv",
)
OPTIONAL_CREATE_IF_MISSING = {
    "model2c_hmc_diagnostics.csv",
}
SORT_COLUMNS = {
    "model2c_fit_summary.csv": ["iso3"],
    "model2c_posterior_draws.csv": ["iso3", "draw", "chain", "iteration"],
    "model2c_posterior_predictive.csv": ["iso3", "draw", "year", "month", "model_age_group"],
    "model2c_observation_process_audit.csv": ["iso3"],
    "model2c_stan_data_metadata.csv": ["iso3"],
    "model2c_hmc_diagnostics.csv": ["iso3", "chain"],
    "model2c_country_selection_audit.csv": ["iso3"],
    "model2c_counterfactual_scenario_summary.csv": [
        "iso3",
        "scenario_id",
        "period_start",
        "period_end",
        "outcome_target",
    ],
    "model2c_counterfactual_scenario_monthly.csv": [
        "iso3",
        "scenario_id",
        "year",
        "month",
        "outcome_target",
    ],
}


def boolish(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def tagged_path(stem: str, tag: str) -> Path:
    return OUTPUTS / f"{stem.removesuffix('.csv')}_{tag}.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Required file is missing: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def assert_source_gate(source_tag: str, iso3: str) -> pd.Series:
    fit = read_csv(tagged_path("model2c_fit_summary.csv", source_tag))
    rows = fit.loc[fit["iso3"].astype(str).str.upper().eq(iso3)]
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one source fit row for {iso3}, found {len(rows)}.")
    row = rows.iloc[0]
    max_rhat = pd.to_numeric(pd.Series([row.get("max_rhat")]), errors="coerce").iloc[0]
    min_ess = pd.to_numeric(pd.Series([row.get("min_ess_bulk")]), errors="coerce").iloc[0]
    hmc_pass = boolish(row.get("registered_hmc_gate_pass", False))
    if not (np.isfinite(max_rhat) and max_rhat < 1.01 and np.isfinite(min_ess) and min_ess >= 400 and hmc_pass):
        raise SystemExit(
            f"Source refit {source_tag}/{iso3} does not pass the registered gate "
            f"(max_rhat={max_rhat:.5g}, min_ess_bulk={min_ess:.5g}, hmc_pass={hmc_pass})."
        )
    return row


def normalized_source_rows(source_rows: pd.DataFrame, target_tag: str) -> pd.DataFrame:
    promoted = source_rows.copy()
    if "model2c_output_tag" in promoted.columns:
        promoted["model2c_output_tag"] = target_tag
    return promoted


def sorted_rows(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    sort_columns = [column for column in SORT_COLUMNS.get(table_name, ["iso3"]) if column in frame.columns]
    if not sort_columns:
        return frame
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def promote_table(table_name: str, source_tag: str, target_tag: str, iso3: str, dry_run: bool) -> dict[str, object]:
    source = read_csv(tagged_path(table_name, source_tag))
    target_path = tagged_path(table_name, target_tag)
    if target_path.exists():
        target = read_csv(target_path)
    elif table_name in OPTIONAL_CREATE_IF_MISSING:
        target = pd.DataFrame(columns=source.columns)
    else:
        target = read_csv(target_path)
    if "iso3" not in source.columns or "iso3" not in target.columns:
        raise SystemExit(f"Both source and target tables must contain iso3: {table_name}")
    source_rows = source.loc[source["iso3"].astype(str).str.upper().eq(iso3)]
    if source_rows.empty:
        raise SystemExit(f"Source table {table_name} has no rows for {iso3}.")
    old_country = target["iso3"].astype(str).str.upper().eq(iso3)
    if not old_country.any() and table_name not in OPTIONAL_CREATE_IF_MISSING:
        raise SystemExit(f"Target table {target_path.relative_to(ROOT)} has no rows for {iso3}.")
    source_rows = normalized_source_rows(source_rows, target_tag)
    columns = list(dict.fromkeys([*target.columns, *source_rows.columns]))
    promoted = pd.concat(
        [
            target.loc[~old_country].reindex(columns=columns),
            source_rows.reindex(columns=columns),
        ],
        ignore_index=True,
        sort=False,
    )
    promoted = sorted_rows(promoted, table_name)
    if not dry_run:
        promoted.to_csv(target_path, index=False)
    return {
        "table_name": table_name,
        "source_file": str(tagged_path(table_name, source_tag).relative_to(ROOT)),
        "target_file": str(target_path.relative_to(ROOT)),
        "source_rows": len(source_rows),
        "target_rows_before": len(target),
        "target_rows_after": len(promoted),
        "replaced_rows": int(old_country.sum()),
    }


def append_audit(rows: list[dict[str, object]], dry_run: bool) -> None:
    audit = pd.DataFrame(rows)
    path = OUTPUTS / "model2c_sensitivity_refit_promotion_audit.csv"
    if dry_run:
        return
    if path.exists():
        previous = pd.read_csv(path)
        audit = pd.concat([previous, audit], ignore_index=True, sort=False)
    audit.to_csv(path, index=False)


class TargetLock:
    def __init__(self, target_tag: str) -> None:
        self.path = OUTPUTS / f".model2c_promote_{target_tag}.lock"
        self.handle = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--iso3", required=True)
    parser.add_argument(
        "--tables",
        default="",
        help="Optional comma-separated table filenames from the registered promotion table set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_tag = args.source_tag.strip().lower().replace("-", "_")
    target_tag = args.target_tag.strip().lower().replace("-", "_")
    iso3 = args.iso3.strip().upper()
    table_names = SENSITIVITY_TABLES
    if args.tables:
        requested = {value.strip() for value in args.tables.split(",") if value.strip()}
        unknown = sorted(requested.difference(SENSITIVITY_TABLES))
        if unknown:
            raise SystemExit("Unknown promotion table(s): " + ", ".join(unknown))
        table_names = tuple(table for table in SENSITIVITY_TABLES if table in requested)
    with TargetLock(target_tag):
        gate_row = assert_source_gate(source_tag, iso3)

        audit_rows: list[dict[str, object]] = []
        for table_name in table_names:
            row = promote_table(table_name, source_tag, target_tag, iso3, args.dry_run)
            row.update(
                {
                    "promoted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "source_tag": source_tag,
                    "target_tag": target_tag,
                    "promoted_iso3": iso3,
                    "source_max_rhat": gate_row.get("max_rhat"),
                    "source_min_ess_bulk": gate_row.get("min_ess_bulk"),
                    "source_registered_hmc_gate_pass": gate_row.get("registered_hmc_gate_pass"),
                }
            )
            audit_rows.append(row)
        append_audit(audit_rows, args.dry_run)
    print(
        f"{'Would promote' if args.dry_run else 'Promoted'} {iso3} from {source_tag} into {target_tag}; "
        f"gate max_rhat={float(gate_row['max_rhat']):.5g}, min_ess_bulk={float(gate_row['min_ess_bulk']):.5g}."
    )


if __name__ == "__main__":
    main()
