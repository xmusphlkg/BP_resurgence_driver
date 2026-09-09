from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pertussis_global_dynamics import model2c
from pertussis_global_dynamics.bayes.model2c_structural import (
    STRUCTURAL_FLAG_DEFAULTS,
    augment_structural_data,
    derive_structural_stan,
)
from pertussis_global_dynamics.bayes.model2c_scenarios import model2c_observation_audit
from run_model2c_structural_refits import structural_specs
from summarize_model2c_structural_refits import compare_with_control, summarize_prediction_file


ROOT = Path(__file__).resolve().parents[2]


def test_structural_specs_are_isolated_and_cover_required_contrasts() -> None:
    specs = structural_specs()
    assert {spec.tag for spec in specs} == {
        "structural_full",
        "structural_no_exposure_interruption",
        "structural_no_post2023_waning",
        "structural_no_reporting_shift",
        "structural_flexible_seeding",
    }
    assert all(spec.tag.startswith("structural_") for spec in specs)
    assert not any("MODEL2C_OUTPUT_TAG" in spec.flags for spec in specs)


def test_augment_structural_data_defaults_and_binary_validation() -> None:
    data = augment_structural_data({"N_months": 2}, {"use_reporting_shift": 0})
    assert data["N_months"] == 2
    assert data["use_reporting_shift"] == 0
    assert data["use_exposure_interruption"] == STRUCTURAL_FLAG_DEFAULTS["use_exposure_interruption"]
    with pytest.raises(ValueError, match="binary"):
        augment_structural_data({}, {"use_reporting_shift": 2})
    with pytest.raises(ValueError, match="Unknown"):
        augment_structural_data({}, {"not_a_flag": 0})


def test_derived_stan_contains_each_switch_and_flexible_basis_without_changing_base() -> None:
    base_path = ROOT / "script/stan/model2c_transmission_scenario_replay.stan"
    base = base_path.read_text(encoding="utf-8")
    derived = derive_structural_stan(base)
    assert "use_exposure_interruption" in derived
    assert "use_post_2023_waning" in derived
    assert "use_reporting_shift" in derived
    assert "use_flexible_outbreak_seeding" in derived
    assert "outbreak_seed_basis_gain" in derived
    assert "basis_center = 6.0 + 12.0 * (k - 1)" in derived
    assert "use_exposure_interruption" not in base


def test_predictive_summary_separates_fit_and_heldout_and_compares_control(tmp_path: Path) -> None:
    rows = []
    for draw, offset in [(1, 0.0), (2, 2.0)]:
        for year, role, observed in [(2025, "fit_likelihood", 10.0), (2026, "posterior_predictive_check", 20.0)]:
            rows.append(
                {
                    "iso3": "AUS", "country": "Australia", "draw": draw,
                    "year": year, "month": 1, "observed_cases": observed,
                    "reported_cases_predicted": observed + offset,
                    "reported_cases_predictive": observed + offset,
                    "fit_role": role,
                }
            )
    path = tmp_path / "ppc.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    control = summarize_prediction_file(path, "structural_full", "full")
    alternative = control.copy()
    alternative["tag"] = "structural_no_reporting_shift"
    alternative["structural_model"] = "no_reporting_shift"
    alternative["rmse"] *= 2.0
    summary = pd.concat([control, alternative], ignore_index=True)
    comparison = compare_with_control(summary)
    assert set(control["fit_role"]) == {"fit_likelihood", "posterior_predictive_check"}
    rmse = comparison.loc[comparison["metric"].eq("rmse")]
    assert len(rmse) == 2
    assert rmse["relative_change_pct"].eq(100.0).all()


def test_no_reporting_shift_audit_reports_effective_fixed_values() -> None:
    draws = pd.DataFrame(
        {
            "log_post_reporting_multiplier": [0.7, -0.5],
            "post_reporting_peak_gain": [1.0, 2.0],
            "post_reporting_trend": [0.4, -0.2],
        }
    )
    audit = model2c_observation_audit(
        {"use_reporting_shift": 0},
        {"iso3": "AUS", "country": "Australia"},
        draws,
    ).iloc[0]
    assert audit["post_reporting_multiplier_median"] == 1.0
    assert audit["post_reporting_peak_multiplier_median"] == 1.0
    assert audit["post_reporting_trend_per_year_median"] == 0.0


def test_draws_dataframe_requests_vector_base_name_and_keeps_registered_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    class FakeFit:
        def draws_pd(self, *, vars: list[str]) -> pd.DataFrame:
            requested.extend(vars)
            return pd.DataFrame(
                {
                    "chain__": [1],
                    "iter__": [1],
                    "draw__": [1],
                    "log_beta": [-3.0],
                    "outbreak_seed_basis_gain[1]": [0.01],
                    "outbreak_seed_basis_gain[2]": [0.02],
                }
            )

    monkeypatch.setattr(
        model2c,
        "MODEL2C_STAN_DRAW_COLUMNS",
        ["log_beta", "outbreak_seed_basis_gain[1]", "outbreak_seed_basis_gain[2]"],
    )
    draws = model2c._draws_dataframe(FakeFit(), {"iso3": "AUS", "country": "Australia"})
    assert "outbreak_seed_basis_gain" in requested
    assert "outbreak_seed_basis_gain[1]" not in requested
    assert draws["outbreak_seed_basis_gain[1]"].iloc[0] == pytest.approx(0.01)
    assert draws["outbreak_seed_basis_gain[2]"].iloc[0] == pytest.approx(0.02)
