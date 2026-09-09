from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_model2c_temporal_validation.py"
SPEC = importlib.util.spec_from_file_location("temporal_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_draws() -> pd.DataFrame:
    rows = []
    for month, observed in [(1, 10), (2, 20)]:
        for draw, predicted in enumerate([observed - 2, observed, observed + 2], start=1):
            rows.append(
                {
                    "iso3": "TST", "country": "Testland", "draw": draw,
                    "year": 2023, "month": month, "observed_cases": observed,
                    "reported_cases_predictive": predicted, "fit_role": "temporal_check",
                }
            )
    return pd.DataFrame(rows)


def test_collapse_draws_scores_exact_median_prediction() -> None:
    monthly = MODULE.collapse_draws(synthetic_draws(), 2023, 2023)
    assert len(monthly) == 2
    assert monthly["absolute_error"].eq(0).all()
    assert monthly["covered_50"].all()
    assert monthly["covered_80"].all()
    assert monthly["covered_95"].all()
    assert monthly["horizon_band"].eq("months_01_12").all()


def test_summary_and_peak_timing_have_expected_scopes() -> None:
    monthly = MODULE.collapse_draws(synthetic_draws(), 2023, 2023)
    summary = MODULE.validation_summary(monthly)
    assert {"country_all_horizons", "pooled_all_horizons"}.issubset(set(summary["scope"]))
    peaks = MODULE.peak_timing_summary(monthly)
    assert len(peaks) == 1
    assert peaks.loc[0, "absolute_peak_timing_error_months"] == 0


def test_fit_likelihood_rows_are_excluded() -> None:
    draws = synthetic_draws()
    duplicate = draws.iloc[[0]].copy()
    duplicate["fit_role"] = "fit_likelihood"
    duplicate["observed_cases"] = 999
    monthly = MODULE.collapse_draws(pd.concat([draws, duplicate], ignore_index=True), 2023, 2023)
    assert monthly["observed_cases"].max() == 20
