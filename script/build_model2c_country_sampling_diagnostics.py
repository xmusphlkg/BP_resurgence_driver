"""Summarise Singapore inclusion and the longer Japan Model 2C refit.

This script is deliberately separate from the submission-audit generator.  It
only reads isolated tagged outputs and never replaces the primary nine-country
posterior or counterfactual files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_project_src_to_path


add_project_src_to_path()

from pertussis_global_dynamics.model2c import MODEL2C_STAN_PARAMETER_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "manuscript" / "reports"

SGP_TAG = "singapore_sensitivity"
JPN_TAG = "jpn_long_refit"
JPN_RAW_DIR = OUTPUTS / "logs" / "model2c_country_diagnostics_20260718" / "jpn_long_refit_cmdstan"


def _read(name: str) -> pd.DataFrame:
    path = OUTPUTS / name
    if not path.exists():
        raise SystemExit(f"Required completed output is missing: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def _reported_cases(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["outcome_target"].eq("reported_cases")].copy()


def _pct_reduction(frame: pd.DataFrame) -> pd.Series:
    return 100.0 * (1.0 - frame["scenario_to_full_ratio_median"])


def build_fit_comparison() -> pd.DataFrame:
    primary = _read("model2c_fit_summary.csv")
    singapore = _read(f"model2c_fit_summary_{SGP_TAG}.csv")
    japan_long = _read(f"model2c_fit_summary_{JPN_TAG}.csv")
    frames = [
        primary.loc[primary["iso3"].eq("JPN")].copy().assign(run="primary_japan"),
        japan_long.copy().assign(run="japan_long_refit"),
        singapore.copy().assign(run="singapore_sensitivity"),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "run",
        "iso3",
        "country",
        "model2c_output_tag",
        "chains",
        "iter_warmup",
        "iter_sampling",
        "posterior_draws",
        "max_rhat",
        "min_ess_bulk",
        "max_rhat_all_reported_quantities",
        "min_ess_bulk_all_reported_quantities",
    ]
    result = combined.loc[:, columns].copy()
    result["passes_core_rhat_lt_1_01"] = result["max_rhat"].lt(1.01)
    result["passes_core_ess_ge_400"] = result["min_ess_bulk"].ge(400)
    result["passes_core_sampling_gate"] = (
        result["passes_core_rhat_lt_1_01"] & result["passes_core_ess_ge_400"]
    )
    return result


def build_panel_comparison() -> pd.DataFrame:
    primary = _reported_cases(_read("model2c_counterfactual_scenario_summary.csv"))
    singapore = _reported_cases(_read(f"model2c_counterfactual_scenario_summary_{SGP_TAG}.csv"))
    scenario_columns = ["scenario_id", "scenario_label"]
    nine = primary.groupby(scenario_columns, as_index=False).agg(
        primary_9_full_total_median=("full_total_median", "sum"),
        primary_9_scenario_total_median=("scenario_total_median", "sum"),
    )
    sgp = singapore.loc[:, scenario_columns + ["full_total_median", "scenario_total_median"]].rename(
        columns={
            "full_total_median": "singapore_full_total_median",
            "scenario_total_median": "singapore_scenario_total_median",
        }
    )
    result = nine.merge(sgp, on=scenario_columns, how="inner", validate="one_to_one")
    result["expanded_10_full_total_median"] = (
        result["primary_9_full_total_median"] + result["singapore_full_total_median"]
    )
    result["expanded_10_scenario_total_median"] = (
        result["primary_9_scenario_total_median"] + result["singapore_scenario_total_median"]
    )
    result["primary_9_descriptive_reduction_percent"] = 100.0 * (
        1.0 - result["primary_9_scenario_total_median"] / result["primary_9_full_total_median"]
    )
    result["expanded_10_descriptive_reduction_percent"] = 100.0 * (
        1.0 - result["expanded_10_scenario_total_median"] / result["expanded_10_full_total_median"]
    )
    result["change_after_singapore_percentage_points"] = (
        result["expanded_10_descriptive_reduction_percent"]
        - result["primary_9_descriptive_reduction_percent"]
    )
    return result


def build_low_fit_exclusion_comparison() -> pd.DataFrame:
    fit = _read("model2c_fit_summary.csv")
    flagged = fit.loc[fit["max_rhat"].ge(1.01) | fit["min_ess_bulk"].lt(400), "iso3"].astype(str)
    flagged_iso3 = sorted(flagged.unique().tolist())
    scenarios = _reported_cases(_read("model2c_counterfactual_scenario_summary.csv"))
    retained = scenarios.loc[~scenarios["iso3"].isin(flagged_iso3)].copy()
    keys = ["scenario_id", "scenario_label"]
    all_nine = scenarios.groupby(keys, as_index=False).agg(
        all_9_full_total_median=("full_total_median", "sum"),
        all_9_scenario_total_median=("scenario_total_median", "sum"),
    )
    restricted = retained.groupby(keys, as_index=False).agg(
        retained_full_total_median=("full_total_median", "sum"),
        retained_scenario_total_median=("scenario_total_median", "sum"),
    )
    result = all_nine.merge(restricted, on=keys, how="inner", validate="one_to_one")
    result["all_9_descriptive_reduction_percent"] = 100.0 * (
        1.0 - result["all_9_scenario_total_median"] / result["all_9_full_total_median"]
    )
    result["low_fit_excluded_descriptive_reduction_percent"] = 100.0 * (
        1.0 - result["retained_scenario_total_median"] / result["retained_full_total_median"]
    )
    result["change_after_low_fit_exclusion_percentage_points"] = (
        result["low_fit_excluded_descriptive_reduction_percent"]
        - result["all_9_descriptive_reduction_percent"]
    )
    result["diagnostic_rule"] = "exclude max core R-hat >=1.01 or min core bulk ESS <400"
    result["excluded_iso3"] = ";".join(flagged_iso3)
    result["n_primary_countries"] = int(scenarios["iso3"].nunique())
    result["n_retained_countries"] = int(retained["iso3"].nunique())
    return result


def build_japan_scenario_comparison() -> pd.DataFrame:
    primary = _reported_cases(_read("model2c_counterfactual_scenario_summary.csv"))
    primary = primary.loc[primary["iso3"].eq("JPN")].copy()
    long = _reported_cases(_read(f"model2c_counterfactual_scenario_summary_{JPN_TAG}.csv"))
    keys = ["scenario_id", "scenario_label"]
    keep = keys + ["full_total_median", "scenario_total_median", "scenario_to_full_ratio_median"]
    result = primary.loc[:, keep].merge(
        long.loc[:, keep], on=keys, suffixes=("_primary", "_long_refit"), validate="one_to_one"
    )
    result["reduction_percent_primary"] = _pct_reduction(
        result.rename(columns={"scenario_to_full_ratio_median_primary": "scenario_to_full_ratio_median"})
    )
    result["reduction_percent_long_refit"] = _pct_reduction(
        result.rename(columns={"scenario_to_full_ratio_median_long_refit": "scenario_to_full_ratio_median"})
    )
    result["reduction_change_percentage_points"] = (
        result["reduction_percent_long_refit"] - result["reduction_percent_primary"]
    )
    result["full_total_change_percent"] = 100.0 * (
        result["full_total_median_long_refit"] / result["full_total_median_primary"] - 1.0
    )
    return result


def _ebfmi(energy: np.ndarray) -> float:
    energy = np.asarray(energy, dtype=float)
    if energy.size < 2 or not np.isfinite(energy).all():
        return float("nan")
    variance = float(np.var(energy, ddof=1))
    if variance <= 0:
        return float("nan")
    return float(np.mean(np.diff(energy) ** 2) / variance)


def build_japan_hmc_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    from cmdstanpy import from_csv

    csv_files = sorted(JPN_RAW_DIR.glob("*.csv"))
    if len(csv_files) != 4:
        raise SystemExit(f"Expected four completed Japan CmdStan CSV files in {JPN_RAW_DIR.relative_to(ROOT)}")
    fit = from_csv([str(path) for path in csv_files])
    method = fit.method_variables()
    energy = np.asarray(method["energy__"])
    divergent = np.asarray(method["divergent__"])
    treedepth = np.asarray(method["treedepth__"])
    if energy.ndim == 1:
        energy = energy[:, None]
        divergent = divergent[:, None]
        treedepth = treedepth[:, None]
    chain_rows = []
    for chain_index in range(energy.shape[1]):
        chain_rows.append({
            "chain": chain_index + 1,
            "post_warmup_draws": int(energy.shape[0]),
            "divergent_transitions": int(np.sum(divergent[:, chain_index])),
            "max_treedepth_observed": int(np.max(treedepth[:, chain_index])),
            "treedepth_15_saturations": int(np.sum(treedepth[:, chain_index] >= 15)),
            "ebfmi": _ebfmi(energy[:, chain_index]),
            "passes_ebfmi_ge_0_3": bool(_ebfmi(energy[:, chain_index]) >= 0.3),
        })
    summary = fit.summary()
    core = summary.loc[summary.index.intersection(MODEL2C_STAN_PARAMETER_COLUMNS)].copy()
    if core.empty:
        raise SystemExit("No core sampled parameters were found in the completed Japan CmdStan summary.")
    relative_mcse = core["MCSE"] / core["StdDev"].replace(0.0, np.nan)
    overall = pd.DataFrame([{
        "iso3": "JPN",
        "tag": JPN_TAG,
        "n_chains": int(energy.shape[1]),
        "post_warmup_draws_per_chain": int(energy.shape[0]),
        "total_divergent_transitions": int(np.sum(divergent)),
        "total_treedepth_15_saturations": int(np.sum(treedepth >= 15)),
        "minimum_ebfmi": min(row["ebfmi"] for row in chain_rows),
        "max_core_rhat": float(core["R_hat"].dropna().max()),
        "min_core_ess_bulk": float(core["ESS_bulk"].dropna().min()),
        "min_core_ess_tail": float(core["ESS_tail"].dropna().min()),
        "max_core_mcse": float(core["MCSE"].dropna().max()),
        "max_core_relative_mcse": float(relative_mcse.dropna().max()),
        "passes_hmc_gate": bool(
            np.sum(divergent) == 0
            and np.sum(treedepth >= 15) == 0
            and min(row["ebfmi"] for row in chain_rows) >= 0.3
            and core["R_hat"].dropna().max() < 1.01
            and core["ESS_bulk"].dropna().min() >= 400
            and core["ESS_tail"].dropna().min() >= 400
        ),
    }])
    return pd.DataFrame(chain_rows), overall


def _markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)
    columns = [str(column) for column in display.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_report(
    fit: pd.DataFrame,
    panel: pd.DataFrame,
    low_fit: pd.DataFrame,
    japan: pd.DataFrame,
    hmc_chains: pd.DataFrame,
    hmc_overall: pd.DataFrame,
) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "model2c_country_inclusion_sampling_diagnostics.md"
    selected = [
        "no_missed_exposure_boosting",
        "no_routine_immunization_disruption",
        "low_vaccine_waning",
        "restored_booster_reach",
        "fixed_birth_cohort_demography",
        "no_reporting_testing_shift",
    ]
    panel_display = panel.loc[panel["scenario_id"].isin(selected), [
        "scenario_label",
        "primary_9_descriptive_reduction_percent",
        "expanded_10_descriptive_reduction_percent",
        "change_after_singapore_percentage_points",
    ]]
    japan_display = japan.loc[japan["scenario_id"].isin(selected), [
        "scenario_label",
        "reduction_percent_primary",
        "reduction_percent_long_refit",
        "reduction_change_percentage_points",
        "full_total_change_percent",
    ]]
    low_fit_display = low_fit.loc[low_fit["scenario_id"].isin(selected), [
        "scenario_label",
        "all_9_descriptive_reduction_percent",
        "low_fit_excluded_descriptive_reduction_percent",
        "change_after_low_fit_exclusion_percentage_points",
    ]]
    sgp_fit = fit.loc[fit["run"].eq("singapore_sensitivity")].iloc[0]
    jpn_primary = fit.loc[fit["run"].eq("primary_japan")].iloc[0]
    jpn_long = fit.loc[fit["run"].eq("japan_long_refit")].iloc[0]
    body = f"""# Model 2C country-inclusion and sampling diagnostics

Generated from isolated tagged outputs. Neither tagged run replaces or merges
with the primary nine-country posterior.

## Sampling diagnostics

{_markdown_table(fit)}

The longer Japan fit changed maximum core-parameter R-hat from
{jpn_primary['max_rhat']:.4f} to {jpn_long['max_rhat']:.4f} and minimum bulk ESS
from {jpn_primary['min_ess_bulk']:.1f} to {jpn_long['min_ess_bulk']:.1f}.
Singapore's tagged fit has maximum core R-hat {sgp_fit['max_rhat']:.4f} and
minimum bulk ESS {sgp_fit['min_ess_bulk']:.1f}. The prespecified diagnostic gate
used here is R-hat <1.01 and bulk ESS >=400 for core sampled parameters.

## Descriptive nine-versus-ten-country comparison

{_markdown_table(panel_display)}

These are sums of country-specific posterior medians, not draws from a joint
multi-country posterior; no uncertainty interval is attached to the summed
contrast. Because countries are fitted independently, adding Singapore does not
change any of the original nine country posteriors. It tests panel inclusion and
descriptive aggregation only.

## Prespecified low-fit-country exclusion sensitivity

Diagnostic rule: exclude a country if maximum core-parameter R-hat is >=1.01
or minimum core-parameter bulk ESS is <400. Flagged countries:
{low_fit['excluded_iso3'].iloc[0]}. Retained countries:
{int(low_fit['n_retained_countries'].iloc[0])} of
{int(low_fit['n_primary_countries'].iloc[0])}.

{_markdown_table(low_fit_display)}

This is a diagnostic exclusion and recomputation of descriptive sums from the
existing country-specific posteriors; it is not a refit or a joint-posterior
leave-one-out analysis.

## Japan primary-versus-long-refit counterfactual comparison

{_markdown_table(japan_display)}

Differences reflect Monte Carlo/refit variation under the same model structure.
They do not constitute temporal validation or structural model comparison.

## Japan long-refit HMC diagnostics

{_markdown_table(hmc_chains)}

{_markdown_table(hmc_overall)}

The HMC gate requires zero divergent transitions, zero depth-15 saturations,
E-BFMI >=0.3 in every chain, core R-hat <1.01, and minimum core bulk and tail
ESS >=400. MCSE is reported both on the parameter scale and relative to the
posterior standard deviation.
"""
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    fit = build_fit_comparison()
    panel = build_panel_comparison()
    low_fit = build_low_fit_exclusion_comparison()
    japan = build_japan_scenario_comparison()
    hmc_chains, hmc_overall = build_japan_hmc_diagnostics()
    fit.to_csv(OUTPUTS / "model2c_country_sampling_fit_comparison.csv", index=False)
    panel.to_csv(OUTPUTS / "model2c_country_inclusion_9_vs_10_comparison.csv", index=False)
    low_fit.to_csv(OUTPUTS / "model2c_low_fit_country_exclusion_comparison.csv", index=False)
    japan.to_csv(OUTPUTS / "model2c_japan_long_refit_scenario_comparison.csv", index=False)
    hmc_chains.to_csv(OUTPUTS / "model2c_japan_long_refit_hmc_by_chain.csv", index=False)
    hmc_overall.to_csv(OUTPUTS / "model2c_japan_long_refit_hmc_summary.csv", index=False)
    report = write_report(fit, panel, low_fit, japan, hmc_chains, hmc_overall)
    print(f"Wrote country and sampling diagnostics to {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
