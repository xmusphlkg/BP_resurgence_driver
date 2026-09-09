from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pertussis_global_dynamics.bayes.model2c_age_composition import (
    DATA_BLOCK_OLD,
    LIKELIHOOD_OLD,
    augment_age_composition_data,
    derive_age_composition_stan,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_weight_module():
    path = ROOT / "script" / "build_model2c_likelihood_weight_diagnostic.py"
    spec = importlib.util.spec_from_file_location("likelihood_weight_diagnostic", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_augment_age_composition_builds_one_based_ragged_groups() -> None:
    data = {
        "N_age_obs": 4,
        "age_obs_year_idx": [2, 1, 2, 1],
        "age_obs_age_idx": [2, 2, 1, 1],
        "y_age": [7, 3, 5, 2],
        "age_obs_weight": [0.8, 0.6, 1.0, 1.0],
    }
    result = augment_age_composition_data(data)
    assert result["N_age_comp_years"] == 2
    assert result["age_comp_start"] == [1, 3]
    assert result["age_comp_len"] == [2, 2]
    assert result["age_comp_age_idx"] == [1, 2, 1, 2]
    assert result["y_age_comp"] == [2, 3, 5, 7]
    assert result["age_comp_weight"] == pytest.approx([1.0, 1.0])


def test_augment_age_composition_rejects_duplicate_age_cells() -> None:
    data = {
        "N_age_obs": 2,
        "age_obs_year_idx": [1, 1],
        "age_obs_age_idx": [1, 1],
        "y_age": [2, 3],
        "age_obs_weight": [1.0, 1.0],
    }
    with pytest.raises(ValueError, match="Duplicate age cells"):
        augment_age_composition_data(data)


def test_stan_derivation_replaces_only_age_likelihood() -> None:
    base = f"data {{\n{DATA_BLOCK_OLD}}}\nmodel {{\nmonthly marker\n{LIKELIHOOD_OLD}}}\n"
    result = derive_age_composition_stan(base)
    assert "dirichlet_multinomial_lpmf" in result
    assert "monthly marker" in result
    assert "N_age_obs" not in result
    assert "N_age_comp_years" in result


def test_weight_diagnostic_reports_kish_metrics_and_nonidentifiability() -> None:
    module = _load_weight_module()
    data = {
        "month_obs_weight": [1.0, 3.0],
        "month_obs_idx": [1, 2],
        "y_month": [10, 20],
        "year_index": [1, 1],
        "age_obs_weight": [1.0, 1.0],
        "age_obs_year_idx": [1, 1],
        "y_age": [12, 18],
    }
    row = module.diagnose_likelihood_weights(data, "TST", "test.json")
    assert row["monthly_weight_mean_implied_temperature"] == pytest.approx(2.0)
    assert row["monthly_weight_kish_ess"] == pytest.approx(1.6)
    assert row["age_to_monthly_case_total_ratio_shared_years"] == pytest.approx(1.0)
    assert row["learning_rate_calibrated"] is False
