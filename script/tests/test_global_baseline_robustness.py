from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "script" / "run_global_baseline_robustness.py"
SPEC = importlib.util.spec_from_file_location("global_baseline_robustness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _annual_series(iso3: str = "TST", source: str = "annual", missing_year: int | None = None) -> pd.DataFrame:
    years = list(range(2010, 2026))
    rows = []
    for year in years:
        if year == missing_year:
            continue
        incidence = 1.0 + 0.08 * (year - 2010) + 0.25 * np.sin(2 * np.pi * year / 4)
        if year >= 2023:
            incidence *= 3
        rows.append(
            {
                "iso3": iso3,
                "country": "Testland",
                "year": year,
                "cases_total": incidence * 10,
                "population": 1_000_000,
                "preferred_source": source,
                "incidence_per_100k": incidence,
            }
        )
    return pd.DataFrame(rows)


def test_periodic_bootstrap_is_reproducible_and_ordered() -> None:
    annual = _annual_series()
    first = MODULE.fit_periodic_baseline(annual, [2023, 2024, 2025], n_bootstrap=500, seed=17)
    second = MODULE.fit_periodic_baseline(annual, [2023, 2024, 2025], n_bootstrap=500, seed=17)
    assert first == second
    assert first["periodic_bootstrap_status"] == "hc2_rademacher_wild_residual_bootstrap"
    assert first["periodic_expected_lcl"] <= first["periodic_expected_ucl"]
    assert first["periodic_bootstrap_successes"] == 500


def test_panel_definitions_are_completeness_and_source_based() -> None:
    complete = _annual_series("AAA")
    incomplete = _annual_series("BBB", missing_year=2014)
    switched = _annual_series("CCC")
    switched.loc[switched["year"].ge(2023), "preferred_source"] = "time_resolved"
    result = MODULE.analyse_configuration(
        pd.concat([complete, incomplete, switched], ignore_index=True),
        requested_cutoff=2025,
        provisional_policy="include",
        n_bootstrap=200,
        seed=10,
    ).set_index("iso3")
    assert bool(result.loc["AAA", "balanced_panel"])
    assert bool(result.loc["AAA", "stable_reporter_panel"])
    assert not bool(result.loc["BBB", "balanced_panel"])
    assert bool(result.loc["CCC", "balanced_panel"])
    assert not bool(result.loc["CCC", "stable_reporter_panel"])


def test_low_baseline_safeguard_retains_raw_and_suppresses_reportable_ratio() -> None:
    annual = _annual_series()
    annual.loc[annual["year"].le(2019), ["incidence_per_100k", "cases_total"]] = [0.01, 0.1]
    result = MODULE.analyse_configuration(
        annual,
        requested_cutoff=2025,
        provisional_policy="include",
        n_bootstrap=200,
        seed=11,
    ).iloc[0]
    assert bool(result["low_periodic_baseline_below_0_1_per_100k"])
    assert np.isfinite(result["periodic_resurgence_ratio"])
    assert pd.isna(result["periodic_ratio_reportable_above_0_1"])
    assert result["periodic_ratio_with_0_1_floor"] < result["periodic_resurgence_ratio"]


def test_excluding_cutoff_changes_effective_post_window() -> None:
    annual = _annual_series()
    include = MODULE.analyse_configuration(
        annual, requested_cutoff=2025, provisional_policy="include", n_bootstrap=200, seed=3
    ).iloc[0]
    exclude = MODULE.analyse_configuration(
        annual, requested_cutoff=2025, provisional_policy="exclude_cutoff", n_bootstrap=200, seed=3
    ).iloc[0]
    assert include["effective_post_end_year"] == 2025
    assert exclude["effective_post_end_year"] == 2024
    assert include["post_observed_years"] == 3
    assert exclude["post_observed_years"] == 2


def test_periodic_prediction_uses_only_observed_post_years() -> None:
    annual = _annual_series()
    annual = annual.loc[~annual["year"].eq(2023)].copy()
    result = MODULE.analyse_configuration(
        annual, requested_cutoff=2025, provisional_policy="include", n_bootstrap=200, seed=5
    ).iloc[0]
    direct = MODULE.fit_periodic_baseline(annual, [2024, 2025], n_bootstrap=200, seed=5)
    assert np.isclose(
        result["periodic_expected_incidence_per_100k"], direct["periodic_expected_incidence_per_100k"]
    )
