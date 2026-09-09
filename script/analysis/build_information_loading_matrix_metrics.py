#!/usr/bin/env python3
"""Build analytical loading-matrix metrics for the surveillance frontier.

The lightweight surrogate represents optional testing and severe-outcome
streams with

    A = [[gamma_T, 1],
         [1, gamma_S]].

This script does not use fitted model outputs. It records the determinant,
rank, singular values, condition number and Fisher-information eigenvalues for
the same cross-loading grid used in the manuscript frontier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "manuscript" / "tables"
TABLES_MD = ROOT / "manuscript" / "tables_md"
FIGURE_DATA = ROOT / "manuscript" / "figure_data"

LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
ANALYSIS_VERSION = "information_loading_matrix_metrics_v1_2026-08-14"


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return "inf"
        return f"{float(value):.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(rows)


def build_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gamma_t in LEVELS:
        for gamma_s in LEVELS:
            loading = np.array([[gamma_t, 1.0], [1.0, gamma_s]], dtype=float)
            determinant = float(np.linalg.det(loading))
            singular_values = np.linalg.svd(loading, compute_uv=False)
            singular_min = float(np.min(singular_values))
            singular_max = float(np.max(singular_values))
            condition_number = (
                singular_max / singular_min if singular_min > 1e-12 else np.inf
            )
            fisher = loading.T @ loading
            eigenvalues = np.linalg.eigvalsh(fisher)
            fisher_min = float(np.min(eigenvalues))
            fisher_max = float(np.max(eigenvalues))
            rows.append(
                {
                    "testing_cross_loading_fraction": gamma_t,
                    "severe_cross_loading_fraction": gamma_s,
                    "loading_matrix": (
                        f"[[{gamma_t:.2f},1.00],[1.00,{gamma_s:.2f}]]"
                    ),
                    "determinant": determinant,
                    "absolute_determinant": abs(determinant),
                    "rank": int(np.linalg.matrix_rank(loading, tol=1e-10)),
                    "minimum_singular_value": singular_min,
                    "maximum_singular_value": singular_max,
                    "condition_number": condition_number,
                    "inverse_condition_index": (
                        0.0 if np.isinf(condition_number) else 1.0 / condition_number
                    ),
                    "fisher_minimum_eigenvalue_unit_noise": fisher_min,
                    "fisher_maximum_eigenvalue_unit_noise": fisher_max,
                    "interpretation": (
                        "rank_loss"
                        if singular_min <= 1e-12
                        else "weakly_conditioned"
                        if condition_number >= 5.0
                        else "well_conditioned"
                    ),
                    "analysis_version": ANALYSIS_VERSION,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    metrics = build_metrics()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    TABLES_MD.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)

    for path in (
        OUTPUTS / "information_loading_matrix_metrics.csv",
        TABLES / "information_loading_matrix_metrics.csv",
        FIGURE_DATA / "information_loading_matrix_metrics.csv",
    ):
        metrics.to_csv(path, index=False)

    display = metrics.loc[
        :,
        [
            "testing_cross_loading_fraction",
            "severe_cross_loading_fraction",
            "determinant",
            "rank",
            "condition_number",
            "fisher_minimum_eigenvalue_unit_noise",
            "interpretation",
        ],
    ].copy()
    display.columns = [
        "Testing cross-loading",
        "Severe cross-loading",
        "det(A)",
        "Rank",
        "Condition number",
        "Minimum eigenvalue of A^T A",
        "Interpretation",
    ]
    (TABLES_MD / "information_loading_matrix_metrics.md").write_text(
        _markdown_table(display),
        encoding="utf-8",
    )
    print(
        "Wrote information-loading matrix metrics to "
        "manuscript/tables/information_loading_matrix_metrics.csv"
    )


if __name__ == "__main__":
    main()
