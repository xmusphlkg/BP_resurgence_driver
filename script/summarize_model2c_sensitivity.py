#!/usr/bin/env python3
"""Compare isolated Model 2C sensitivity outputs with the primary fit.

The summariser is deliberately read-only with respect to posterior, scenario,
and observation-process files.  It may be rerun while fits are in progress:
missing, incomplete, or temporarily unreadable tagged outputs are represented
in the status table rather than treated as evidence of robustness.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from run_model2c_sensitivity import SensitivitySpec, sensitivity_specs


ROOT = Path(__file__).resolve().parents[1]

SCENARIO_KEYS = ["iso3", "country", "scenario_id", "period_start", "period_end", "outcome_target"]
SCENARIO_METRICS = [
    "scenario_to_full_ratio_median",
    "full_minus_scenario_median",
]
OBSERVATION_METRICS = [
    "post_reporting_multiplier_median",
    "post_reporting_peak_multiplier_median",
    "post_transmission_multiplier_median",
    "excess_epidemic_pressure_scale_median",
    "outbreak_ignition_pulse_gain_median",
    "post_reporting_trend_per_year_median",
]


@dataclass(frozen=True)
class ReadResult:
    path: str
    state: str
    frame: pd.DataFrame | None
    detail: str = ""


def _read_csv_safely(path: Path) -> ReadResult:
    if not path.exists():
        return ReadResult(str(path), "missing", None)
    try:
        if path.stat().st_size == 0:
            return ReadResult(str(path), "empty", None, "zero-byte file")
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return ReadResult(str(path), "unreadable_or_running", None, f"{type(exc).__name__}: {exc}")
    if frame.empty:
        return ReadResult(str(path), "empty", frame, "header present but no rows")
    if "iso3" not in frame.columns:
        return ReadResult(str(path), "invalid_schema", frame, "required iso3 column absent")
    return ReadResult(str(path), "ready", frame)


def _first_existing(outputs: Path, names: Iterable[str]) -> Path:
    paths = [outputs / name for name in names]
    return next((path for path in paths if path.exists()), paths[0])


def _primary_paths(outputs: Path) -> dict[str, Path]:
    return {
        "fit": _first_existing(outputs, ["model2c_fit_summary_full.csv", "model2c_fit_summary.csv"]),
        "scenario": _first_existing(
            outputs,
            ["model2c_counterfactual_scenario_summary_full.csv", "model2c_counterfactual_scenario_summary.csv"],
        ),
        "observation": _first_existing(
            outputs,
            ["model2c_observation_process_audit_full.csv", "model2c_observation_process_audit.csv"],
        ),
    }


def _tagged_paths(outputs: Path, tag: str) -> dict[str, Path]:
    return {
        "fit": outputs / f"model2c_fit_summary_{tag}.csv",
        "scenario": outputs / f"model2c_counterfactual_scenario_summary_{tag}.csv",
        "observation": outputs / f"model2c_observation_process_audit_{tag}.csv",
    }


def _countries(result: ReadResult) -> set[str]:
    if result.frame is None or "iso3" not in result.frame:
        return set()
    return set(result.frame["iso3"].dropna().astype(str))


def _run_status(spec: SensitivitySpec, reads: dict[str, ReadResult]) -> str:
    expected = set(spec.countries)
    fit = _countries(reads["fit"])
    scenario = _countries(reads["scenario"])
    observation = _countries(reads["observation"])
    states = {result.state for result in reads.values()}
    if "unreadable_or_running" in states or "empty" in states:
        return "partial_or_running"
    if not fit and not scenario and not observation:
        return "not_started"
    if not expected.issubset(fit):
        return "partial_fit"
    if not expected.issubset(scenario):
        return "fit_complete_replay_pending"
    if not expected.issubset(observation):
        return "fit_and_replay_complete_observation_pending"
    return "complete"


def _direction(relative_change_pct: float) -> str:
    if not math.isfinite(relative_change_pct):
        return "not_classifiable"
    if abs(relative_change_pct) < 1.0:
        return "stable"
    return "increased" if relative_change_pct > 0 else "decreased"


def _magnitude(relative_change_pct: float) -> str:
    if not math.isfinite(relative_change_pct):
        return "not_classifiable"
    value = abs(relative_change_pct)
    if value < 1.0:
        return "stable_lt_1pct"
    if value < 5.0:
        return "negligible_1_to_lt_5pct"
    if value < 20.0:
        return "modest_5_to_lt_20pct"
    if value < 50.0:
        return "material_20_to_lt_50pct"
    return "large_ge_50pct"


def _dependence(relative_change_pct: float) -> str:
    if not math.isfinite(relative_change_pct):
        return "not_classifiable"
    value = abs(relative_change_pct)
    if value < 5.0:
        return "low"
    if value < 20.0:
        return "limited"
    if value < 50.0:
        return "moderate"
    return "strong"


def _comparison_row(
    *,
    spec: SensitivitySpec,
    source: str,
    metric: str,
    primary_value: object,
    sensitivity_value: object,
    context: dict[str, object],
) -> dict[str, object] | None:
    primary = pd.to_numeric(pd.Series([primary_value]), errors="coerce").iloc[0]
    sensitivity = pd.to_numeric(pd.Series([sensitivity_value]), errors="coerce").iloc[0]
    if pd.isna(primary) or pd.isna(sensitivity):
        return None
    primary = float(primary)
    sensitivity = float(sensitivity)
    difference = sensitivity - primary
    if primary == 0:
        relative = 0.0 if sensitivity == 0 else math.nan
    else:
        relative = 100.0 * difference / abs(primary)
    return {
        "tag": spec.tag,
        "suite": spec.suite,
        "source": source,
        "metric": metric,
        **context,
        "primary_value": primary,
        "sensitivity_value": sensitivity,
        "absolute_difference": difference,
        "relative_change_pct": relative,
        "direction": (
            "stable"
            if primary == 0 and sensitivity == 0
            else ("increased_from_zero" if primary == 0 and sensitivity > 0 else
                  "decreased_from_zero" if primary == 0 and sensitivity < 0 else _direction(relative))
        ),
        "magnitude_class": _magnitude(relative),
        "prior_dependence_class": _dependence(relative),
    }


def _compare_frames(
    primary: ReadResult,
    tagged: ReadResult,
    *,
    spec: SensitivitySpec,
    source: str,
    keys: list[str],
    metrics: list[str],
) -> list[dict[str, object]]:
    if primary.state != "ready" or tagged.state != "ready" or primary.frame is None or tagged.frame is None:
        return []
    available_keys = [key for key in keys if key in primary.frame.columns and key in tagged.frame.columns]
    available_metrics = [metric for metric in metrics if metric in primary.frame.columns and metric in tagged.frame.columns]
    if "iso3" not in available_keys or not available_metrics:
        return []
    primary_columns = available_keys + available_metrics
    tagged_columns = available_keys + available_metrics
    merged = tagged.frame[tagged_columns].merge(
        primary.frame[primary_columns],
        on=available_keys,
        how="inner",
        suffixes=("_sensitivity", "_primary"),
    )
    rows: list[dict[str, object]] = []
    for record in merged.to_dict(orient="records"):
        context = {key: record.get(key, "") for key in SCENARIO_KEYS}
        for metric in available_metrics:
            row = _comparison_row(
                spec=spec,
                source=source,
                metric=metric,
                primary_value=record[f"{metric}_primary"],
                sensitivity_value=record[f"{metric}_sensitivity"],
                context=context,
            )
            if row is not None:
                rows.append(row)
    return rows


def _status_row(spec: SensitivitySpec, reads: dict[str, ReadResult]) -> dict[str, object]:
    expected = set(spec.countries)
    row: dict[str, object] = {
        "tag": spec.tag,
        "suite": spec.suite,
        "expected_countries": len(expected),
        "run_status": _run_status(spec, reads),
        "description": spec.description,
    }
    for source, result in reads.items():
        found = _countries(result)
        row[f"{source}_file_state"] = result.state
        row[f"{source}_countries_found"] = len(found & expected)
        row[f"{source}_countries_missing"] = ",".join(sorted(expected - found))
        row[f"{source}_read_detail"] = result.detail
    fit = reads["fit"].frame
    if fit is not None:
        fit_expected = fit.loc[fit["iso3"].astype(str).isin(expected)]
        def numeric_extreme(column: str, operation: str) -> float:
            if column not in fit_expected:
                return math.nan
            values = pd.to_numeric(fit_expected[column], errors="coerce")
            return float(values.max() if operation == "max" else values.min())

        row["max_rhat"] = numeric_extreme("max_rhat", "max")
        row["min_ess_bulk"] = numeric_extreme("min_ess_bulk", "min")
        row["posterior_draws_min"] = numeric_extreme("posterior_draws", "min")
    else:
        row.update({"max_rhat": math.nan, "min_ess_bulk": math.nan, "posterior_draws_min": math.nan})
    row["diagnostic_status"] = (
        "pass"
        if pd.notna(row["max_rhat"])
        and pd.notna(row["min_ess_bulk"])
        and float(row["max_rhat"]) < 1.01
        and float(row["min_ess_bulk"]) >= 400
        else "fail_or_unavailable"
    )
    return row


def _suite_summary(detail: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "suite", "metric", "n_tags_compared", "n_countries_compared", "n_comparisons",
        "median_absolute_relative_change_pct", "max_absolute_relative_change_pct",
        "direction_consistency", "prior_dependence_class", "complete_tags", "expected_tags",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    finite = detail.loc[pd.to_numeric(detail["relative_change_pct"], errors="coerce").notna()].copy()
    finite["absolute_relative_change_pct"] = finite["relative_change_pct"].abs()
    records: list[dict[str, object]] = []
    expected_by_suite = status.groupby("suite")["tag"].nunique().to_dict()
    complete_by_suite = status.loc[status["run_status"] == "complete"].groupby("suite")["tag"].nunique().to_dict()
    for (suite, metric), group in finite.groupby(["suite", "metric"], sort=True):
        directions = set(group.loc[group["direction"] != "stable", "direction"])
        maximum = float(group["absolute_relative_change_pct"].max())
        records.append(
            {
                "suite": suite,
                "metric": metric,
                "n_tags_compared": group["tag"].nunique(),
                "n_countries_compared": group["iso3"].nunique(),
                "n_comparisons": len(group),
                "median_absolute_relative_change_pct": float(group["absolute_relative_change_pct"].median()),
                "max_absolute_relative_change_pct": maximum,
                "direction_consistency": "stable" if not directions else (next(iter(directions)) if len(directions) == 1 else "mixed"),
                "prior_dependence_class": _dependence(maximum),
                "complete_tags": int(complete_by_suite.get(suite, 0)),
                "expected_tags": int(expected_by_suite.get(suite, 0)),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No comparable completed outputs are available._\n"
    shown = frame.loc[:, [column for column in columns if column in frame]].copy()
    shown = shown.fillna("")
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    lines = [header, rule]
    for values in shown.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarise(outputs: Path, specs: list[SensitivitySpec] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = sensitivity_specs() if specs is None else specs
    primary = {name: _read_csv_safely(path) for name, path in _primary_paths(outputs).items()}
    if any(result.state != "ready" for result in primary.values()):
        states = ", ".join(f"{name}={result.state}" for name, result in primary.items())
        raise SystemExit(f"Primary Model 2C outputs are not all readable: {states}")

    status_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for spec in specs:
        tagged = {name: _read_csv_safely(path) for name, path in _tagged_paths(outputs, spec.tag).items()}
        status_rows.append(_status_row(spec, tagged))
        detail_rows.extend(
            _compare_frames(
                primary["scenario"], tagged["scenario"], spec=spec, source="scenario",
                keys=SCENARIO_KEYS, metrics=SCENARIO_METRICS,
            )
        )
        detail_rows.extend(
            _compare_frames(
                primary["observation"], tagged["observation"], spec=spec, source="observation",
                keys=["iso3", "country"], metrics=OBSERVATION_METRICS,
            )
        )
    status = pd.DataFrame.from_records(status_rows)
    detail = pd.DataFrame.from_records(detail_rows)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                "tag", "suite", "source", "metric", *SCENARIO_KEYS,
                "primary_value", "sensitivity_value", "absolute_difference", "relative_change_pct",
                "direction", "magnitude_class", "prior_dependence_class",
            ]
        )
    summary = _suite_summary(detail, status)
    return status, detail, summary


def write_outputs(outputs: Path, status: pd.DataFrame, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    _write_atomic(status, outputs / "model2c_sensitivity_comparison_status.csv")
    _write_atomic(detail, outputs / "model2c_sensitivity_comparison.csv")
    _write_atomic(summary, outputs / "model2c_sensitivity_comparison_summary.csv")
    markdown = [
        "# Model 2C sensitivity comparison\n",
        "Tagged sensitivity outputs are compared with the primary full-contact fit. Missing or partial runs are not interpreted as robustness evidence. Diagnostic pass requires max R-hat < 1.01 and minimum bulk ESS >= 400.\n",
        "## Run and diagnostic status\n",
        _markdown_table(
            status,
            ["tag", "suite", "run_status", "fit_countries_found", "scenario_countries_found", "observation_countries_found", "max_rhat", "min_ess_bulk", "diagnostic_status"],
        ),
        "\n## Comparison summary\n",
        _markdown_table(
            summary,
            ["suite", "metric", "n_tags_compared", "n_countries_compared", "n_comparisons", "median_absolute_relative_change_pct", "max_absolute_relative_change_pct", "direction_consistency", "prior_dependence_class", "complete_tags", "expected_tags"],
        ),
        "\n## Interpretation boundary\n",
        "The dependence classification is diagnostic: low (<5%), limited (5-<20%), moderate (20-<50%), or strong (>=50%) maximum relative change from the primary estimate. It does not establish model adequacy, causal identification, or external validity.\n",
    ]
    path = outputs / "model2c_sensitivity_comparison.md"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(markdown), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--only-tags", default="", help="Comma-separated tag subset; default is all registered tags.")
    args = parser.parse_args(argv)
    specs = sensitivity_specs()
    if args.only_tags:
        requested = {value.strip().lower().replace("-", "_") for value in args.only_tags.split(",") if value.strip()}
        specs = [spec for spec in specs if spec.tag in requested]
        if not specs:
            raise SystemExit("No registered sensitivity tags matched --only-tags.")
    status, detail, summary = summarise(args.outputs_dir, specs)
    write_outputs(args.outputs_dir, status, detail, summary)
    print(
        f"Wrote sensitivity comparison for {len(status)} tag(s): "
        f"{sum(status['run_status'] == 'complete')} complete; {len(detail)} comparable estimates."
    )


if __name__ == "__main__":
    main()
