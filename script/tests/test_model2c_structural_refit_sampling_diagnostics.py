from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_model2c_structural_refit_sampling_diagnostics as diagnostics


def _write_tag(outputs: Path, tag: str, *, omit_iso3: str | None = None) -> None:
    rows = []
    for index, iso3 in enumerate(diagnostics.EXPECTED_ISO3):
        if iso3 == omit_iso3:
            continue
        rows.append(
            {
                "iso3": iso3,
                "country": iso3,
                "max_rhat": 1.009 if index else 1.01,
                "min_ess_bulk": 400.0 if index != 1 else 399.9,
            }
        )
    for name, template in diagnostics.ARTIFACT_TEMPLATES.items():
        path = outputs / template.format(tag=tag)
        if name == "fit_summary":
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            pd.DataFrame({"iso3": ["AUS"], "value": [1]}).to_csv(path, index=False)


def test_diagnostics_apply_strict_rhat_and_inclusive_ess_gate(tmp_path: Path) -> None:
    _write_tag(tmp_path, "structural_test")
    result = diagnostics.build_diagnostics(tmp_path, ["structural_test"])
    assert len(result) == 9
    assert result["fit_summary_nine_countries_complete"].all()
    assert result["all_required_artifacts_nonempty"].all()
    assert not result.loc[result["iso3"].eq("AUS"), "passes_core_sampling_gate"].item()
    assert not result.loc[result["iso3"].eq("BRA"), "passes_core_sampling_gate"].item()
    assert result.loc[result["iso3"].eq("CHN"), "passes_core_sampling_gate"].item()


def test_missing_tag_and_country_are_recorded_without_error(tmp_path: Path) -> None:
    _write_tag(tmp_path, "partial", omit_iso3="JPN")
    result = diagnostics.build_diagnostics(tmp_path, ["partial", "absent"])
    partial = result.loc[result["tag"].eq("partial")]
    absent = result.loc[result["tag"].eq("absent")]
    assert not partial["fit_summary_nine_countries_complete"].any()
    assert partial["missing_expected_iso3"].eq("JPN").all()
    assert not partial.loc[partial["iso3"].eq("JPN"), "fit_summary_row_present_once"].item()
    assert absent["fit_summary_read_status"].eq("missing_or_empty").all()
    assert absent["missing_expected_iso3"].eq(";".join(diagnostics.EXPECTED_ISO3)).all()
    assert not absent["tag_complete"].any()


def test_header_only_artifact_is_not_nonempty_and_cli_accepts_comma_tags(tmp_path: Path) -> None:
    _write_tag(tmp_path, "one")
    _write_tag(tmp_path, "two")
    (tmp_path / "model2c_posterior_draws_two.csv").write_text("iso3,value\n", encoding="utf-8")
    report = tmp_path / "report.md"
    diagnostics.main(
        [
            "--tags", "one,two", "missing",
            "--outputs-dir", str(tmp_path),
            "--report-path", str(report),
        ]
    )
    result = pd.read_csv(tmp_path / diagnostics.DIAGNOSTIC_CSV)
    assert len(result) == 27
    two = result.loc[result["tag"].eq("two")]
    assert not two["posterior_draws_nonempty"].any()
    assert not two["tag_complete"].any()
    assert "missing/empty artifacts" in report.read_text(encoding="utf-8")
