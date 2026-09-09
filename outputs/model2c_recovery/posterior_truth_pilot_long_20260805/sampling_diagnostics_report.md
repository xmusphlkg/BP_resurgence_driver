# Model 2C recovery sampling diagnostics

Recovery tag: `posterior_truth_pilot_long_20260805`.

A `*.complete.json` marker means only that the fitting workflow finished and wrote its artifacts; it is not evidence of convergence. The independent diagnostic gate requires all 27 core parameters (`log_beta` through `init_natural_recent_fraction`) to have finite diagnostics, maximum R-hat < 1.01, minimum bulk ESS >= 400, zero divergences, zero maximum-tree-depth saturations and E-BFMI >= 0.3 in every chain.

## Overall status

Expected fits: 36; workflow-complete fits: 7; diagnostically evaluable fits: 7; gate-passing fits: 6.

## Expansion gate

Status: `not_ready_missing_or_incomplete_hmc_diagnostics`. Expansion to all 180 long-sampling fits authorized: False. Required action: Finish or recover the missing HMC diagnostics, then recompute the gate before expansion.

## By country and truth scenario

| aggregation_level | iso3 | scenario_id | n_expected_fits | n_fit_complete | n_diagnostics_evaluable | n_diagnostic_gate_pass | diagnostic_gate_pass_rate_among_evaluable | worst_max_core_rhat_among_evaluable | worst_min_core_ess_bulk_among_evaluable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| country_scenario | BRA | multi_mechanism | 3 | 2 | 2 | 1 | 0.5 | 1.01256 | 449.413 |
| country_scenario | BRA | null_both | 3 | 1 | 1 | 1 | 1 | 1.00306 | 1718.55 |
| country_scenario | BRA | reporting_only | 3 | 2 | 2 | 2 | 1 | 1.00459 | 1387.93 |
| country_scenario | BRA | transmission_only | 3 | 2 | 2 | 2 | 1 | 1.00422 | 940.957 |
| country_scenario | CHN | multi_mechanism | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | CHN | null_both | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | CHN | reporting_only | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | CHN | transmission_only | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | GBR | multi_mechanism | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | GBR | null_both | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | GBR | reporting_only | 3 | 0 | 0 | 0 | NA | NA | NA |
| country_scenario | GBR | transmission_only | 3 | 0 | 0 | 0 | NA | NA | NA |

## Diagnostic failures

| run_id | max_core_rhat | min_core_ess_bulk | total_divergent_transitions | total_treedepth_saturations | minimum_ebfmi |
| --- | --- | --- | --- | --- | --- |
| BRA_multi_mechanism_r001 | 1.01256 | 449.413 | 0 | 0 | 0.830968 |

## Incomplete or invalid artifacts

| run_id | completion_marker_present | fit_summary_present | artifact_error |
| --- | --- | --- | --- |
| BRA_multi_mechanism_r003 | False | False | HMC diagnostic file is missing |
| BRA_null_both_r002 | False | False | HMC diagnostic file is missing |
| BRA_null_both_r003 | False | False | HMC diagnostic file is missing |
| BRA_reporting_only_r003 | False | False | HMC diagnostic file is missing |
| BRA_transmission_only_r003 | False | False | HMC diagnostic file is missing |
| CHN_multi_mechanism_r001 | False | False | HMC diagnostic file is missing |
| CHN_multi_mechanism_r002 | False | False | HMC diagnostic file is missing |
| CHN_multi_mechanism_r003 | False | False | HMC diagnostic file is missing |
| CHN_null_both_r001 | False | False | HMC diagnostic file is missing |
| CHN_null_both_r002 | False | False | HMC diagnostic file is missing |
| CHN_null_both_r003 | False | False | HMC diagnostic file is missing |
| CHN_reporting_only_r001 | False | False | HMC diagnostic file is missing |
| CHN_reporting_only_r002 | False | False | HMC diagnostic file is missing |
| CHN_reporting_only_r003 | False | False | HMC diagnostic file is missing |
| CHN_transmission_only_r001 | False | False | HMC diagnostic file is missing |
| CHN_transmission_only_r002 | False | False | HMC diagnostic file is missing |
| CHN_transmission_only_r003 | False | False | HMC diagnostic file is missing |
| GBR_multi_mechanism_r001 | False | False | HMC diagnostic file is missing |
| GBR_multi_mechanism_r002 | False | False | HMC diagnostic file is missing |
| GBR_multi_mechanism_r003 | False | False | HMC diagnostic file is missing |
| GBR_null_both_r001 | False | False | HMC diagnostic file is missing |
| GBR_null_both_r002 | False | False | HMC diagnostic file is missing |
| GBR_null_both_r003 | False | False | HMC diagnostic file is missing |
| GBR_reporting_only_r001 | False | False | HMC diagnostic file is missing |
| GBR_reporting_only_r002 | False | False | HMC diagnostic file is missing |
| GBR_reporting_only_r003 | False | False | HMC diagnostic file is missing |
| GBR_transmission_only_r001 | False | False | HMC diagnostic file is missing |
| GBR_transmission_only_r002 | False | False | HMC diagnostic file is missing |
| GBR_transmission_only_r003 | False | False | HMC diagnostic file is missing |

These diagnostics qualify recovery estimates; they do not turn simulation recovery into external validation.
