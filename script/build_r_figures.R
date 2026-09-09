#!/usr/bin/env Rscript

figure_scripts <- c(
  "figure_1_global_trends.R",
  "figure_2_schedule_clusters.R",
  "figure_2_process_linked_surveillance.R",
  "figure_4_model2c_scenario_replay.R",
  "figure_5_ambiguity_partial_matching.R",
  "figure_5_model2c_observation_boundary.R",
  "figure_6_recovery_decisions.R",
  "figure_7_boundaries.R",
  "figure_4_empirical_comparability.R",
  # Named-country supplement for the main surveillance-design result.
  "supplement_named_country_auc.R",
  "supplement_time_resolved_all_age_series.R",
  "supplement_south_africa_monthly_series.R",
  "supplement_age_pattern_global_summary.R",
  "supplement_age_pattern_validation_lattice.R",
  "supplement_direct_testing_denominator_evidence.R",
  "supplement_reporting_practice_timeline.R",
  "supplement_official_surveillance_boundary_summary.R",
  "supplement_ecdc_who_concordance.R",
  "supplement_usa_annual_age_specific_surveillance_summary.R",
  "supplement_japan_annual_national_surveillance_summary.R",
  "supplement_gbr_age_pattern_history.R",
  "supplement_nld_age_stratified_testing.R",
  "supplement_model2c_monthly_fit_diagnostics.R",
  "supplement_model2c_ppc_diagnostics.R",
  "supplement_model2c_parameter_posteriors.R",
  "supplement_model2c_observation_audit.R",
  "supplement_model2c_contact_sensitivity_boundary.R",
  "supplement_model2c_literature_scenario_evidence.R",
  "supplement_model2c_external_validation_lattice.R",
  "supplement_empirical_baseline_sensitivity.R",
  "supplement_global_notification_context.R"
)

script_dir <- file.path("script", "figures")
rscript <- Sys.getenv("RSCRIPT", unset = "Rscript")

for (script in figure_scripts) {
  path <- file.path(script_dir, script)
  if (!file.exists(path)) {
    stop("Missing R figure script: ", path, call. = FALSE)
  }
  message("Running ", path)
  status <- system2(rscript, path)
  if (!identical(status, 0L)) {
    stop("Figure script failed: ", path, call. = FALSE)
  }
}
