#!/usr/bin/env python3
"""Summarize MCMC diagnostics for completed, isolated Model 2C recovery fits.

This script never launches Stan and never changes fit artifacts.  A completion
marker records workflow completion only; convergence is assessed separately
from all 27 recovery parameters using the prespecified R-hat/ESS gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "script" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pertussis_global_dynamics.bayes.model2c_recovery import (  # noqa: E402
    RECOVERY_PARAMETERS,
    assert_isolated_recovery_dir,
    safe_recovery_tag,
)


OUTPUTS = ROOT / "outputs"
RHAT_THRESHOLD = 1.01
ESS_BULK_THRESHOLD = 400.0
EBFMI_THRESHOLD = 0.3
FIT_OUTPUT = "sampling_diagnostics_by_fit.csv"
GROUP_OUTPUT = "sampling_diagnostics_by_country_scenario.csv"
GATE_OUTPUT = "synthetic_recovery_gate_audit.csv"
REPORT_OUTPUT = "sampling_diagnostics_report.md"


def _marker_state(path: Path, run_id: str) -> tuple[bool, bool, str]:
    """Return marker presence, validity, and a concise parsing error."""
    if not path.exists():
        return False, False, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return True, False, f"invalid completion marker: {exc}"
    valid = payload.get("run_id") == run_id and payload.get("status") == "fit_complete"
    error = "" if valid else "completion marker run_id/status mismatch"
    return True, valid, error


def _summary_metrics(path: Path) -> tuple[int, float, float, str]:
    """Read only core-parameter rows from one CmdStan summary CSV."""
    try:
        frame = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return 0, np.nan, np.nan, f"unreadable fit summary: {exc}"
    required_columns = {"R_hat", "ESS_bulk"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        return 0, np.nan, np.nan, f"missing columns: {', '.join(missing_columns)}"

    present: list[str] = []
    rhats: list[float] = []
    esses: list[float] = []
    for parameter in RECOVERY_PARAMETERS:
        if parameter not in frame.index:
            continue
        row: Any = frame.loc[parameter]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rhat = pd.to_numeric(pd.Series([row["R_hat"]]), errors="coerce").iloc[0]
        ess = pd.to_numeric(pd.Series([row["ESS_bulk"]]), errors="coerce").iloc[0]
        if not np.isfinite(rhat) or not np.isfinite(ess):
            continue
        present.append(parameter)
        rhats.append(float(rhat))
        esses.append(float(ess))
    missing = [parameter for parameter in RECOVERY_PARAMETERS if parameter not in present]
    error = "" if not missing else f"missing/non-finite core parameters: {', '.join(missing)}"
    return len(present), max(rhats, default=np.nan), min(esses, default=np.nan), error


def _hmc_metrics(path: Path) -> tuple[int, int, float, str]:
    if not path.exists():
        return 0, 0, np.nan, "HMC diagnostic file is missing"
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return 0, 0, np.nan, f"unreadable HMC diagnostics: {exc}"
    required = {"divergent_transitions", "treedepth_saturations", "minimum_ebfmi"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return 0, 0, np.nan, f"HMC diagnostics missing columns: {', '.join(missing)}"
    return (
        int(pd.to_numeric(frame["divergent_transitions"]).sum()),
        int(pd.to_numeric(frame["treedepth_saturations"]).sum()),
        float(pd.to_numeric(frame["minimum_ebfmi"]).min()),
        "",
    )


def summarize_experiment(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-fit and grouped diagnostic tables from existing artifacts."""
    manifest_path = experiment_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing recovery manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"run_id", "iso3", "scenario_id", "replicate"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Recovery manifest missing columns: {', '.join(missing)}")

    summary_dir = experiment_dir / "fit_summaries"
    rows: list[dict[str, Any]] = []
    for item in manifest.sort_values(["iso3", "scenario_id", "replicate"]).to_dict(orient="records"):
        run_id = str(item["run_id"])
        summary_path = summary_dir / f"{run_id}.csv"
        marker_path = summary_dir / f"{run_id}.complete.json"
        hmc_path = summary_dir / f"{run_id}.hmc.csv"
        marker_present, fit_complete, marker_error = _marker_state(marker_path, run_id)
        n_present, max_rhat, min_ess, summary_error = (0, np.nan, np.nan, "")
        # A marker is written atomically after the summary.  Do not inspect an
        # unmarked summary that may still be in the process of being written.
        if fit_complete:
            if summary_path.exists():
                n_present, max_rhat, min_ess, summary_error = _summary_metrics(summary_path)
            else:
                summary_error = "completion marker exists but fit summary is missing"
        divergent, depth_saturations, minimum_ebfmi, hmc_error = _hmc_metrics(hmc_path)
        diagnostics_evaluable = bool(
            fit_complete and n_present == len(RECOVERY_PARAMETERS) and not hmc_error
        )
        gate_pass = bool(
            diagnostics_evaluable
            and max_rhat < RHAT_THRESHOLD
            and min_ess >= ESS_BULK_THRESHOLD
            and divergent == 0
            and depth_saturations == 0
            and minimum_ebfmi >= EBFMI_THRESHOLD
        )
        rows.append(
            {
                "run_id": run_id,
                "iso3": str(item["iso3"]),
                "scenario_id": str(item["scenario_id"]),
                "replicate": int(item["replicate"]),
                "completion_marker_present": marker_present,
                "fit_complete": fit_complete,
                "fit_summary_present": summary_path.exists(),
                "n_core_parameters_expected": len(RECOVERY_PARAMETERS),
                "n_core_parameters_present": n_present,
                "diagnostics_evaluable": diagnostics_evaluable,
                "max_core_rhat": max_rhat,
                "min_core_ess_bulk": min_ess,
                "total_divergent_transitions": divergent,
                "total_treedepth_saturations": depth_saturations,
                "minimum_ebfmi": minimum_ebfmi,
                "diagnostic_gate_pass": gate_pass,
                "diagnostic_gate_rule": "R-hat <1.01; bulk ESS >=400; zero divergence; zero tree-depth saturation; E-BFMI >=0.3",
                "artifact_error": "; ".join(error for error in (marker_error, summary_error, hmc_error) if error),
            }
        )
    by_fit = pd.DataFrame(rows)
    grouped = aggregate_diagnostics(by_fit)
    return by_fit, grouped


def aggregate_diagnostics(by_fit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completion and convergence separately by country/scenario."""
    keys_by_level: tuple[tuple[str, list[str]], ...] = (
        ("country_scenario", ["iso3", "scenario_id"]),
        ("country", ["iso3"]),
        ("truth_scenario", ["scenario_id"]),
        ("overall", []),
    )
    records: list[dict[str, Any]] = []
    for level, keys in keys_by_level:
        groups = [((), by_fit)] if not keys else by_fit.groupby(keys, sort=True, dropna=False)
        for labels, group in groups:
            labels = labels if isinstance(labels, tuple) else (labels,)
            label_map = dict(zip(keys, labels))
            evaluable = group.loc[group["diagnostics_evaluable"]]
            records.append(
                {
                    "aggregation_level": level,
                    "iso3": label_map.get("iso3", "ALL"),
                    "scenario_id": label_map.get("scenario_id", "ALL"),
                    "n_expected_fits": int(len(group)),
                    "n_fit_complete": int(group["fit_complete"].sum()),
                    "n_diagnostics_evaluable": int(group["diagnostics_evaluable"].sum()),
                    "n_diagnostic_gate_pass": int(group["diagnostic_gate_pass"].sum()),
                    "diagnostic_gate_pass_rate_among_evaluable": (
                        float(evaluable["diagnostic_gate_pass"].mean()) if len(evaluable) else np.nan
                    ),
                    "worst_max_core_rhat_among_evaluable": (
                        float(evaluable["max_core_rhat"].max()) if len(evaluable) else np.nan
                    ),
                    "worst_min_core_ess_bulk_among_evaluable": (
                        float(evaluable["min_core_ess_bulk"].min()) if len(evaluable) else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def synthetic_recovery_gate_audit(by_fit: pd.DataFrame) -> pd.DataFrame:
    """State whether the long pilot authorizes expansion to the full recovery set."""

    n_expected = int(len(by_fit))
    n_complete = int(by_fit["fit_complete"].sum())
    n_evaluable = int(by_fit["diagnostics_evaluable"].sum())
    n_pass = int(by_fit["diagnostic_gate_pass"].sum())
    pass_rate_all_planned = float(n_pass / n_expected) if n_expected else np.nan
    pass_rate_evaluable = float(n_pass / n_evaluable) if n_evaluable else np.nan
    expansion_authorized = bool(n_expected > 0 and pass_rate_all_planned >= 0.90)
    if expansion_authorized:
        status_label = "pilot_gate_pass_expand_to_180"
        action = "Full long-sampling recovery may proceed under the registered gate."
    elif n_evaluable < n_expected:
        status_label = "not_ready_missing_or_incomplete_hmc_diagnostics"
        action = "Finish or recover the missing HMC diagnostics, then recompute the gate before expansion."
    else:
        status_label = "pilot_gate_failed_do_not_expand"
        action = "Do not expand to 180 fits; retain recovery results as diagnostics only."
    return pd.DataFrame(
        [
            {
                "planned_pilot_fits": n_expected,
                "workflow_complete_fits": n_complete,
                "diagnostically_evaluable_fits": n_evaluable,
                "gate_passing_fits": n_pass,
                "pilot_pass_rate_all_planned": pass_rate_all_planned,
                "pilot_pass_rate_among_evaluable": pass_rate_evaluable,
                "required_pilot_pass_rate_for_expansion": 0.90,
                "expansion_to_180_authorized": expansion_authorized,
                "gate_status": status_label,
                "registered_gate_rule": (
                    "R-hat <1.01; bulk ESS >=400; zero divergence; "
                    "zero tree-depth saturation; E-BFMI >=0.3"
                ),
                "required_action": action,
            }
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    if frame.empty:
        return "(none)"

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_outputs(experiment_dir: Path, by_fit: pd.DataFrame, grouped: pd.DataFrame) -> None:
    by_fit.to_csv(experiment_dir / FIT_OUTPUT, index=False)
    grouped.to_csv(experiment_dir / GROUP_OUTPUT, index=False)
    gate = synthetic_recovery_gate_audit(by_fit)
    gate.to_csv(experiment_dir / GATE_OUTPUT, index=False)
    overall = grouped.loc[grouped["aggregation_level"].eq("overall")].iloc[0]
    gate_row = gate.iloc[0]
    country_scenario = grouped.loc[grouped["aggregation_level"].eq("country_scenario")]
    failures = by_fit.loc[by_fit["diagnostics_evaluable"] & ~by_fit["diagnostic_gate_pass"]]
    incomplete = by_fit.loc[~by_fit["fit_complete"]]
    lines = [
        "# Model 2C recovery sampling diagnostics",
        "",
        f"Recovery tag: `{experiment_dir.name}`.",
        "",
        "A `*.complete.json` marker means only that the fitting workflow finished and wrote its artifacts; it is not evidence of convergence. The independent diagnostic gate requires all 27 core parameters (`log_beta` through `init_natural_recent_fraction`) to have finite diagnostics, maximum R-hat < 1.01, minimum bulk ESS >= 400, zero divergences, zero maximum-tree-depth saturations and E-BFMI >= 0.3 in every chain.",
        "",
        "## Overall status",
        "",
        f"Expected fits: {int(overall['n_expected_fits'])}; workflow-complete fits: {int(overall['n_fit_complete'])}; diagnostically evaluable fits: {int(overall['n_diagnostics_evaluable'])}; gate-passing fits: {int(overall['n_diagnostic_gate_pass'])}.",
        "",
        "## Expansion gate",
        "",
        f"Status: `{gate_row['gate_status']}`. Expansion to all 180 long-sampling fits authorized: {bool(gate_row['expansion_to_180_authorized'])}. Required action: {gate_row['required_action']}",
        "",
        "## By country and truth scenario",
        "",
        _markdown_table(country_scenario),
        "",
        "## Diagnostic failures",
        "",
    ]
    if failures.empty:
        lines.append("No completed, evaluable fit failed the diagnostic gate.")
    else:
        lines.append(_markdown_table(failures[[
            "run_id", "max_core_rhat", "min_core_ess_bulk",
            "total_divergent_transitions", "total_treedepth_saturations", "minimum_ebfmi",
        ]]))
    lines += ["", "## Incomplete or invalid artifacts", ""]
    if incomplete.empty:
        lines.append("All expected fits have valid completion markers.")
    else:
        lines.append(_markdown_table(incomplete[["run_id", "completion_marker_present", "fit_summary_present", "artifact_error"]]))
    lines += [
        "",
        "These diagnostics qualify recovery estimates; they do not turn simulation recovery into external validation.",
    ]
    (experiment_dir / REPORT_OUTPUT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Existing isolated recovery experiment tag.")
    args = parser.parse_args(argv)
    try:
        args.tag = safe_recovery_tag(args.tag)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_dir = OUTPUTS / "model2c_recovery" / args.tag
    assert_isolated_recovery_dir(experiment_dir, OUTPUTS)
    if not experiment_dir.is_dir():
        raise SystemExit(f"Missing recovery experiment directory: {experiment_dir}")
    by_fit, grouped = summarize_experiment(experiment_dir)
    write_outputs(experiment_dir, by_fit, grouped)
    print(
        f"Wrote {len(by_fit)} per-fit rows and {len(grouped)} grouped rows to "
        f"{experiment_dir.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
