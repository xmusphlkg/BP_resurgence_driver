#!/usr/bin/env python3
"""Compare the conditional annual age-composition fit with the primary fit."""

from __future__ import annotations

from pathlib import Path

from run_model2c_sensitivity import SensitivitySpec
from summarize_model2c_sensitivity import summarise


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    spec = SensitivitySpec(
        tag="age_composition_dm",
        suite="likelihood_structure",
        countries=("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA"),
        env={},
        description=(
            "Monthly all-age negative-binomial likelihood plus annual age composition "
            "conditional on the observed annual total via a Dirichlet-multinomial term."
        ),
    )
    detail, status, summary = summarise(OUTPUTS, [spec])
    detail.to_csv(OUTPUTS / "model2c_age_composition_comparison.csv", index=False)
    status.to_csv(OUTPUTS / "model2c_age_composition_comparison_status.csv", index=False)
    summary.to_csv(OUTPUTS / "model2c_age_composition_comparison_summary.csv", index=False)
    print("Wrote conditional age-composition comparison outputs.")


if __name__ == "__main__":
    main()
