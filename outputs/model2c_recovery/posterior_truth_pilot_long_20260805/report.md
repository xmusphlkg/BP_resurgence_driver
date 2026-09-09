# Model 2C simulation-based recovery

Tagged experiment directory: `outputs/model2c_recovery/posterior_truth_pilot_long_20260805`.
Prepared runs: 36; completed refits: 7.

## Protocol

For each country and replicate, one fitted posterior draw supplies the truth. The same truth draw is replayed under four prespecified structures: both reporting and transmission mechanisms, reporting only, transmission only, and neither. Monthly all-age and annual age-group counts are independently generated with the model's NB2 observation distributions, then refitted with the unchanged Model 2C Stan likelihood.

Primary diagnostics are 95% interval coverage, bias/RMSE, direction recovery for log-scale shift parameters, and active/inactive mechanism discrimination using a ±log(1.05) region of practical equivalence. Posterior mechanism activity probabilities ≥0.80 are classified active, ≤0.20 inactive, and intermediate values ambiguous.

This is a recoverability diagnostic conditional on the observed covariate histories and posterior-draw truth distribution; it is not external validation and does not calibrate the composite likelihood learning rate.

## Aggregate results

| metric_family | estimand | n_runs | coverage_95 | mean_bias | rmse | direction_accuracy | discrimination_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parameter_recovery | contact_sensitivity | 7 | 0.571429 | -0.408252 | 0.456617 | 1 | NA |
| parameter_recovery | endemic_reseed_pressure | 7 | 0.428571 | 0.00310147 | 0.00943402 | 1 | NA |
| parameter_recovery | epidemic_feedback_gain | 7 | 1 | 0.910514 | 2.11134 | 1 | NA |
| parameter_recovery | epidemic_pressure_retention | 7 | 1 | -0.0391393 | 0.108434 | 1 | NA |
| parameter_recovery | init_natural_recent_fraction | 7 | 1 | 0.168809 | 0.1744 | 1 | NA |
| parameter_recovery | init_natural_share | 7 | 0.571429 | 0.0867317 | 0.0924755 | 1 | NA |
| parameter_recovery | init_vaccine_recent_fraction | 7 | 1 | -0.0338471 | 0.174945 | 1 | NA |
| parameter_recovery | log_beta | 7 | 0.857143 | -0.317768 | 0.563566 | 1 | NA |
| parameter_recovery | log_post_reporting_multiplier | 7 | 0.857143 | 0.0639669 | 0.203924 | 1 | NA |
| parameter_recovery | log_post_transmission_multiplier | 7 | 0.428571 | 0.0721762 | 0.235619 | 1 | NA |
| parameter_recovery | log_reporting_rate | 7 | 1 | 0.0596519 | 0.191924 | 1 | NA |
| parameter_recovery | natural_boosting_efficiency | 7 | 0.714286 | -0.105782 | 0.147015 | 1 | NA |
| parameter_recovery | nb_phi_age | 7 | 0.857143 | 0.198839 | 0.304186 | 1 | NA |
| parameter_recovery | nb_phi_month | 7 | 0.714286 | 1.54652 | 2.0564 | 1 | NA |
| parameter_recovery | outbreak_seeding_gain | 7 | 0.285714 | -0.0217475 | 0.0438347 | 1 | NA |
| parameter_recovery | pathogen_pressure_scale | 7 | 0.571429 | 0.0479222 | 0.261891 | 1 | NA |
| parameter_recovery | post_reporting_peak_gain | 7 | 0.285714 | -0.0367482 | 0.611887 | 1 | NA |
| parameter_recovery | post_reporting_trend | 7 | 1 | -0.0842682 | 0.0986435 | 1 | NA |
| parameter_recovery | rebound_asymmetry | 7 | 1 | 0.0632346 | 0.239044 | 1 | NA |
| parameter_recovery | rebound_peak_month | 7 | 0.571429 | -0.634065 | 0.95018 | 1 | NA |
| parameter_recovery | rebound_release_gain | 7 | 0.571429 | 0.130615 | 1.04508 | 1 | NA |
| parameter_recovery | rebound_width | 7 | 0.714286 | -0.499059 | 0.826396 | 1 | NA |
| parameter_recovery | seasonal_strength | 7 | 0.857143 | 0.0450923 | 0.165259 | 1 | NA |
| parameter_recovery | secondary_rebound_delay | 7 | 1 | -1.26657 | 3.58584 | 1 | NA |
| parameter_recovery | secondary_rebound_release_gain | 7 | 0.428571 | 0.406562 | 0.454405 | 1 | NA |
| parameter_recovery | secondary_rebound_width | 7 | 1 | -0.322646 | 1.32539 | 1 | NA |
| parameter_recovery | waning_rate | 7 | 0.857143 | -0.0800434 | 0.0824777 | 1 | NA |
| mechanism_discrimination | post_reporting_shift | 7 | NA | NA | NA | NA | 0.428571 |
| mechanism_discrimination | post_transmission_shift | 7 | NA | NA | NA | NA | 0.428571 |
