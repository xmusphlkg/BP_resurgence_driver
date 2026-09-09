# Model 2C surveillance-design results

Analysis version: `model2c_surveillance_design_v5_20260726`.

The experiment uses paired complete and one-process-neutralised trajectories,
group-held-out posterior draws, and the 16 prespecified monitoring packages.
AUC measures structural distinguishability conditional on the mechanism library;
it does not identify the historical cause of a national resurgence.

## Main results

| Mechanism contrast | Current package AUC | Complete package AUC | Largest average marginal gain | Recommendation |
| --- | ---: | ---: | --- | --- |
| missed_exposure_boosting | 0.596 | 0.653 | serology (+0.056) | notifications+severe+serology (no_package_met_success_rule) |
| routine_immunization_disruption | 0.497 | 0.520 | serology (+0.022) | notifications+severe+serology (no_package_met_success_rule) |
| vaccine_waning | 0.900 | 0.915 | age (+0.053) | notifications (success_rule_met) |
| post_2023_transmission_context | 1.000 | 1.000 | severe (+0.000) | notifications (success_rule_met) |
| reporting_testing_shift | 0.577 | 0.657 | testing (+0.071) | notifications+age+testing+severe+serology (no_package_met_success_rule) |
| transmission_vs_reporting | 0.537 | 0.999 | severe (+0.209) | notifications+severe (success_rule_met) |

## Interpretation rule

The prespecified operational success rule requires pooled AUC at least 0.80
and country-specific AUC at least 0.70 in six of nine backgrounds. A package
that fails this rule should not be used to rank that mechanism from routine
surveillance alone.

## Reproducibility

- `latent_surveillance_signals.csv` contains the deterministic model signals.
- `observation_features_*.csv` contains the simulated monitoring observations.
- `package_discrimination.csv` and `country_discrimination.csv` contain held-out results.
- `stream_shapley_gains.csv` contains average marginal AUC gains.
- `minimum_package_recommendations.csv` applies the prespecified success rule.
