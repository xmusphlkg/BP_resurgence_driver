# Model 2C simulation-based recovery

Tagged experiment directory: `outputs/model2c_recovery/posterior_truth_nine_country_r5`.
Prepared runs: 180; completed refits: 180.

## Protocol

For each country and replicate, one fitted posterior draw supplies the truth. The same truth draw is replayed under four prespecified structures: both reporting and transmission mechanisms, reporting only, transmission only, and neither. Monthly all-age and annual age-group counts are independently generated with the model's NB2 observation distributions, then refitted with the unchanged Model 2C Stan likelihood.

Primary diagnostics are 95% interval coverage, bias/RMSE, direction recovery for log-scale shift parameters, and active/inactive mechanism discrimination using a ±log(1.05) region of practical equivalence. Posterior mechanism activity probabilities ≥0.80 are classified active, ≤0.20 inactive, and intermediate values ambiguous.

This is a recoverability diagnostic conditional on the observed covariate histories and posterior-draw truth distribution; it is not external validation and does not calibrate the composite likelihood learning rate.

## Aggregate results

| metric_family | estimand | n_runs | coverage_95 | mean_bias | rmse | direction_accuracy | discrimination_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parameter_recovery | contact_sensitivity | 180 | 0.677778 | -0.177661 | 0.336324 | 1 | NA |
| parameter_recovery | endemic_reseed_pressure | 180 | 0.65 | 0.0080831 | 0.0125824 | 1 | NA |
| parameter_recovery | epidemic_feedback_gain | 180 | 0.905556 | 1.62531 | 3.44221 | 1 | NA |
| parameter_recovery | epidemic_pressure_retention | 180 | 0.861111 | -0.0372866 | 0.164754 | 1 | NA |
| parameter_recovery | init_natural_recent_fraction | 180 | 0.894444 | -0.0315435 | 0.2071 | 1 | NA |
| parameter_recovery | init_natural_share | 180 | 0.622222 | -0.0540218 | 0.128929 | 1 | NA |
| parameter_recovery | init_vaccine_recent_fraction | 180 | 0.85 | -0.0371564 | 0.18627 | 1 | NA |
| parameter_recovery | log_beta | 180 | 0.783333 | -0.320488 | 0.698652 | 1 | NA |
| parameter_recovery | log_post_reporting_multiplier | 180 | 0.905556 | 0.040048 | 0.228881 | 0.905556 | NA |
| parameter_recovery | log_post_transmission_multiplier | 180 | 0.883333 | -0.0239619 | 0.176268 | 0.95 | NA |
| parameter_recovery | log_reporting_rate | 180 | 0.85 | -0.27353 | 0.605159 | 1 | NA |
| parameter_recovery | natural_boosting_efficiency | 180 | 0.844444 | 0.0345685 | 0.165373 | 1 | NA |
| parameter_recovery | nb_phi_age | 180 | 0.833333 | 0.243822 | 0.574936 | 1 | NA |
| parameter_recovery | nb_phi_month | 180 | 0.611111 | 2.03663 | 4.9474 | 1 | NA |
| parameter_recovery | outbreak_seeding_gain | 180 | 0.433333 | 0.000827363 | 0.0299602 | 1 | NA |
| parameter_recovery | pathogen_pressure_scale | 180 | 0.455556 | 0.485193 | 0.733344 | 1 | NA |
| parameter_recovery | post_reporting_peak_gain | 180 | 0.344444 | 0.0226632 | 0.404002 | 1 | NA |
| parameter_recovery | post_reporting_trend | 180 | 0.85 | -0.0519103 | 0.171292 | 1 | NA |
| parameter_recovery | rebound_asymmetry | 180 | 0.85 | 0.0860847 | 0.348537 | 1 | NA |
| parameter_recovery | rebound_peak_month | 180 | 0.644444 | 0.0680282 | 1.63458 | 1 | NA |
| parameter_recovery | rebound_release_gain | 180 | 0.466667 | 0.424686 | 1.01578 | 1 | NA |
| parameter_recovery | rebound_width | 180 | 0.722222 | -0.228515 | 1.22788 | 1 | NA |
| parameter_recovery | seasonal_strength | 180 | 0.855556 | 0.0585692 | 0.157812 | 1 | NA |
| parameter_recovery | secondary_rebound_delay | 180 | 0.794444 | 1.07891 | 3.83816 | 1 | NA |
| parameter_recovery | secondary_rebound_release_gain | 180 | 0.45 | 0.411822 | 0.585715 | 1 | NA |
| parameter_recovery | secondary_rebound_width | 180 | 0.9 | -0.274761 | 2.26693 | 1 | NA |
| parameter_recovery | waning_rate | 180 | 0.866667 | -0.020225 | 0.0546311 | 1 | NA |
| mechanism_discrimination | post_reporting_shift | 180 | NA | NA | NA | NA | 0.261111 |
| mechanism_discrimination | post_transmission_shift | 180 | NA | NA | NA | NA | 0.277778 |
