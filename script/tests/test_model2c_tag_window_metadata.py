from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "normalize_model2c_tag_window_metadata.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("tag_windows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalize_uses_stan_indices_as_authority(tmp_path: Path) -> None:
    data = {
        "year": [2022, 2022, 2023, 2023, 2026],
        "month_obs_idx": [1, 2],
        "month_ppc_obs_idx": [3, 4, 5],
        "age_obs_year_idx": [],
    }
    (tmp_path / "model2c_stan_data_TST_temporal.json").write_text(json.dumps(data))
    frame = pd.DataFrame([{"iso3": "TST", "model2c_fit_end_year": 2025}])
    result = MODULE.normalize_frame(frame, "temporal", tmp_path)
    row = result.iloc[0]
    assert row["model2c_fit_end_year"] == 2022
    assert row["model2c_temporal_check_start_year"] == 2023
    assert row["model2c_temporal_check_end_year"] == 2026
    assert row["model2c_temporal_check_n_months"] == 3
    assert row["model2c_2026_ppc_observed_n_months"] == 1
