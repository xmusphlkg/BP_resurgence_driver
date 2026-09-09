#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

as_logical_flag <- function(x) {
  str_to_lower(as.character(x)) %in% c("true", "1", "yes")
}

bounded_probability <- function(x) {
  pmin(pmax(as.numeric(x), 0), 1)
}

observation <- read_csv_first_available(
  c(
    out_path("model2c_observation_process_audit.csv"),
    file.path(root, "manuscript", "tables", "model2c_observation_process_audit.csv")
  ),
  c(
    "iso3", "country", "reporting_prior_mean", "reporting_prior_lower",
    "reporting_prior_upper", "reporting_prior_log_sd", "direct_testing_support",
    "reporting_prior_support_level", "post_reporting_multiplier_median",
    "post_reporting_multiplier_lower_95", "post_reporting_multiplier_upper_95",
    "post_transmission_multiplier_median", "post_transmission_multiplier_lower_95",
    "post_transmission_multiplier_upper_95", "excess_epidemic_pressure_scale_median",
    "outbreak_ignition_pulse_gain_median", "secondary_outbreak_ignition_pulse_gain_median",
    "posterior_reporting_shift_probability", "posterior_reporting_peak_probability",
    "posterior_transmission_context_shift_probability",
    "posterior_excess_epidemic_pressure_active_probability",
    "posterior_outbreak_ignition_pulse_active_probability",
    "posterior_secondary_outbreak_ignition_active_probability",
    "posterior_reporting_trend_probability", "observation_process_supported",
    "observation_process_role", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 5 observation audit") %>%
  mutate(
    across(
      c(
        reporting_prior_mean, reporting_prior_lower, reporting_prior_upper,
        reporting_prior_log_sd, post_reporting_multiplier_median,
        post_reporting_multiplier_lower_95, post_reporting_multiplier_upper_95,
        post_transmission_multiplier_median, post_transmission_multiplier_lower_95,
        post_transmission_multiplier_upper_95, excess_epidemic_pressure_scale_median,
        outbreak_ignition_pulse_gain_median, secondary_outbreak_ignition_pulse_gain_median
      ),
      as.numeric
    ),
    direct_testing_support = as_logical_flag(direct_testing_support),
    observation_process_supported = as_logical_flag(observation_process_supported),
    posterior_reporting_shift_probability = bounded_probability(posterior_reporting_shift_probability),
    posterior_reporting_peak_probability = bounded_probability(posterior_reporting_peak_probability),
    posterior_transmission_context_shift_probability = bounded_probability(posterior_transmission_context_shift_probability),
    posterior_pathogen_pressure_active_probability = bounded_probability(posterior_excess_epidemic_pressure_active_probability),
    posterior_outbreak_seeding_active_probability = bounded_probability(posterior_outbreak_ignition_pulse_active_probability),
    posterior_secondary_seeding_active_probability = bounded_probability(posterior_secondary_outbreak_ignition_active_probability),
    posterior_reporting_trend_probability = bounded_probability(posterior_reporting_trend_probability)
  )

external_validation <- read_csv_checked(
  out_path("model2c_external_validation_audit.csv"),
  c(
    "iso3", "country", "validation_domain", "evidence_available",
    "claim_strength", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 5 external validation audit") %>%
  mutate(evidence_available = as_logical_flag(evidence_available))

testing_summary <- read_csv_optional(out_path("direct_testing_denominator_evidence_summary.csv")) %>%
  mutate(
    direct_testing_series_years_available = as.numeric(direct_testing_series_years_available),
    direct_testing_series_relative_detection_prior_factor_median =
      as.numeric(direct_testing_series_relative_detection_prior_factor_median)
  )

reporting_summary <- read_csv_optional(out_path("reporting_practice_timeline_summary.csv")) %>%
  mutate(
    reporting_practice_high_evidence_fraction = as.numeric(reporting_practice_high_evidence_fraction),
    reporting_practice_detection_prior_available = as_logical_flag(reporting_practice_detection_prior_available)
  )

official_outcome_summary <- read_csv_optional(out_path("official_surveillance_outcome_summary.csv")) %>%
  mutate(
    hospitalization_available = as_logical_flag(hospitalization_available),
    death_available = as_logical_flag(death_available),
    official_surveillance_testing_denominator_available =
      as_logical_flag(official_surveillance_testing_denominator_available)
  )

posterior_draws <- read_csv_first_available(
  c(
    out_path("model2c_posterior_draws.csv"),
    out_path("model2c_posterior_draws_full.csv")
  ),
  c(
    "iso3", "log_post_reporting_multiplier", "log_post_transmission_multiplier",
    "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 5 posterior draws") %>%
  transmute(
    iso3,
    reporting_draw = exp(as.numeric(log_post_reporting_multiplier)),
    context_draw = exp(as.numeric(log_post_transmission_multiplier))
  )

posterior_middle_intervals <- posterior_draws %>%
  group_by(iso3) %>%
  summarise(
    post_reporting_multiplier_lower_50 = quantile(reporting_draw, 0.25, na.rm = TRUE),
    post_reporting_multiplier_upper_50 = quantile(reporting_draw, 0.75, na.rm = TRUE),
    post_transmission_multiplier_lower_50 = quantile(context_draw, 0.25, na.rm = TRUE),
    post_transmission_multiplier_upper_50 = quantile(context_draw, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

reporting_sensitivity_focus <- read_csv_checked(
  out_path("model2c_reporting_prior_sensitivity.csv"),
  c("iso3", "model2c_output_version")
) %>%
  require_model2c_current_version("Figure 5 reporting-prior sensitivity") %>%
  distinct(iso3) %>%
  mutate(reporting_prior_sensitive = TRUE)

contact_sensitivity_focus <- read_csv_checked(
  out_path("model2c_contact_sensitivity_boundary_audit.csv"),
  c("iso3", "boundary_flag", "model2c_output_version")
) %>%
  require_model2c_current_version("Figure 5 contact-path sensitivity") %>%
  transmute(
    iso3,
    contact_boundary_sensitive = as_logical_flag(boundary_flag)
  )

scenario_contrasts <- read_csv_first_available(
  c(
    fd_path("figure_4_model2c_scenario_contrasts_full.csv"),
    fd_path("figure_4_model2c_scenario_contrasts.csv"),
    out_path("model2c_counterfactual_scenario_summary_full.csv"),
    out_path("model2c_counterfactual_scenario_summary.csv")
  ),
  c(
    "iso3", "country", "scenario_id", "period_start", "period_end",
    "outcome_target", "scenario_to_full_ratio_median",
    "scenario_to_full_ratio_lower_95", "scenario_to_full_ratio_upper_95",
    "full_minus_scenario_median", "full_minus_scenario_lower_95",
    "full_minus_scenario_upper_95", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 5 audit scenario contrasts")

pilot <- observation %>%
  distinct(iso3, country) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_label = format_country_inline(iso3, country)
  ) %>%
  arrange(pilot_rank, country)

country_levels <- rev(pilot$country_label)

external_wide <- external_validation %>%
  filter(iso3 %in% pilot$iso3) %>%
  mutate(
    evidence_role = case_when(
      evidence_available & validation_domain == "direct_testing_denominator" ~ "External boundary",
      evidence_available ~ "External boundary",
      TRUE ~ "Not available"
    )
  ) %>%
  select(iso3, validation_domain, evidence_available, claim_strength, evidence_role) %>%
  pivot_wider(
    names_from = validation_domain,
    values_from = c(evidence_available, claim_strength, evidence_role),
    values_fill = list(evidence_available = FALSE, claim_strength = "not_available", evidence_role = "Not available")
  )

obs_plot <- observation %>%
  left_join(pilot %>% select(iso3, country_label, pilot_rank), by = "iso3") %>%
  left_join(posterior_middle_intervals, by = "iso3") %>%
  left_join(
    testing_summary %>%
      select(
        iso3,
        direct_testing_series_support_class,
        direct_testing_series_years_available,
        direct_testing_series_relative_detection_prior_factor_median
      ),
    by = "iso3"
  ) %>%
  left_join(
    reporting_summary %>%
      select(
        iso3,
        reporting_practice_high_evidence_fraction,
        reporting_practice_detection_prior_available
      ),
    by = "iso3"
  ) %>%
  left_join(
    official_outcome_summary %>%
      select(
        iso3,
        hospitalization_available,
        death_available,
        official_surveillance_testing_denominator_available,
        official_surveillance_outcome_support_level
      ),
    by = "iso3"
  ) %>%
  left_join(reporting_sensitivity_focus, by = "iso3") %>%
  left_join(contact_sensitivity_focus, by = "iso3") %>%
  left_join(external_wide, by = "iso3") %>%
  mutate(
    country_label = factor(country_label, levels = country_levels),
    direct_testing_series_support_class = coalesce(direct_testing_series_support_class, "not_available"),
    testing_prior_class = case_when(
      direct_testing_support & str_detect(direct_testing_series_support_class, "multi_year") ~ "Model-window testing support",
      direct_testing_support ~ "Single-year testing support",
      coalesce(evidence_available_direct_testing_denominator, FALSE) ~ "External testing only",
      TRUE ~ "No testing-denominator support"
    ),
    testing_prior_class = factor(
      testing_prior_class,
      levels = c(
        "Model-window testing support", "Single-year testing support",
        "External testing only", "No testing-denominator support"
      )
    ),
    reporting_prior_sensitive = coalesce(reporting_prior_sensitive, FALSE),
    contact_boundary_sensitive = coalesce(contact_boundary_sensitive, FALSE),
    audit_max_probability = pmax(
      posterior_transmission_context_shift_probability,
      posterior_pathogen_pressure_active_probability,
      posterior_outbreak_seeding_active_probability,
      posterior_secondary_seeding_active_probability,
      na.rm = TRUE
    ),
    audit_load = case_when(
      audit_max_probability >= 0.80 ~ "High audit signal",
      audit_max_probability >= 0.50 ~ "Moderate audit signal",
      TRUE ~ "Low audit signal"
    ),
    audit_load = factor(
      audit_load,
      levels = c("Low audit signal", "Moderate audit signal", "High audit signal")
    )
  ) %>%
  arrange(pilot_rank)

evidence_columns <- c(
  direct_testing_denominator = "Testing\ndenominator",
  reporting_practice_case_definition = "Reporting\npractice",
  official_severe_outcomes = "Severe\noutcomes",
  age_pattern_evidence = "Age-pattern\nvalidation",
  pathogen_amr_timeline = "Pathogen/AMR\ncontext",
  contact_validation = "Contact\ncontext"
)

evidence_layers <- external_validation %>%
  filter(iso3 %in% pilot$iso3, validation_domain %in% names(evidence_columns)) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  left_join(observation %>% select(iso3, direct_testing_support), by = "iso3") %>%
  mutate(
    country_label = factor(country_label, levels = country_levels),
    domain_index = match(validation_domain, names(evidence_columns)),
    domain_label = factor(evidence_columns[validation_domain], levels = unname(evidence_columns)),
    evidence_role = case_when(
      validation_domain == "direct_testing_denominator" & direct_testing_support ~ "In-model constraint",
      evidence_available ~ "External boundary",
      TRUE ~ "Not available"
    ),
    evidence_role = factor(evidence_role, levels = c("In-model constraint", "External boundary", "Not available"))
  )

evidence_count <- evidence_layers %>%
  count(country_label, evidence_role, name = "n_layers") %>%
  complete(
    country_label,
    evidence_role,
    fill = list(n_layers = 0)
  ) %>%
  mutate(
    country_label = factor(country_label, levels = country_levels),
    evidence_role = factor(evidence_role, levels = c("In-model constraint", "External boundary", "Not available"))
  )

evidence_totals <- evidence_count %>%
  group_by(country_label) %>%
  summarise(
    evidence_available_layers = sum(n_layers[evidence_role != "Not available"]),
    in_model_evidence_layers = sum(n_layers[evidence_role == "In-model constraint"]),
    external_boundary_layers = sum(n_layers[evidence_role == "External boundary"]),
    unavailable_evidence_layers = sum(n_layers[evidence_role == "Not available"]),
    evidence_label = paste0(evidence_available_layers, "/", length(evidence_columns)),
    .groups = "drop"
  )

audit_base <- obs_plot %>%
  left_join(
    evidence_totals %>%
      mutate(country_label_chr = as.character(country_label)) %>%
      select(
        country_label_chr,
        evidence_available_layers,
        in_model_evidence_layers,
        external_boundary_layers,
        unavailable_evidence_layers
      ),
    by = c("country_label" = "country_label_chr")
  ) %>%
  mutate(
    reporting_log2_multiplier = log2(post_reporting_multiplier_median),
    context_log2_multiplier = log2(post_transmission_multiplier_median),
    reporting_shift_abs_log2 = abs(log2(post_reporting_multiplier_median)),
    audit_max_probability = bounded_probability(audit_max_probability),
    evidence_available_layers = as.numeric(evidence_available_layers),
    reporting_audit_direction = case_when(
      post_reporting_multiplier_lower_95 > 1 ~ "Above no-shift",
      post_reporting_multiplier_upper_95 < 1 ~ "Below no-shift",
      post_reporting_multiplier_median > 1.10 ~ "Median above",
      TRUE ~ "Near no-shift"
    ),
    context_audit_direction = case_when(
      post_transmission_multiplier_lower_95 > 1 ~ "Above no-shift",
      post_transmission_multiplier_upper_95 < 1 ~ "Below no-shift",
      post_transmission_multiplier_median > 1.10 ~ "Median above",
      TRUE ~ "Near no-shift"
    ),
    interpretation_status = case_when(
      !direct_testing_support & evidence_available_layers <= 4 ~ "Audit-limited wording",
      reporting_prior_sensitive |
        contact_boundary_sensitive |
        post_reporting_multiplier_median > 1.20 |
        post_transmission_multiplier_median > 1.10 ~ "Boundary-sensitive",
      TRUE ~ "Transmission-consistent"
    ),
    interpretation_status = factor(
      interpretation_status,
      levels = c(
        "Transmission-consistent",
        "Boundary-sensitive",
        "Audit-limited wording"
      )
    )
  )

audit_order <- audit_base %>%
  mutate(
    max_upward_audit_multiplier = pmax(
      post_reporting_multiplier_median,
      post_transmission_multiplier_median,
      na.rm = TRUE
    )
  ) %>%
  arrange(desc(max_upward_audit_multiplier), desc(post_reporting_multiplier_median), pilot_rank)

country_levels_result <- rev(as.character(audit_order$country_label))
country_axis <- tibble(
  country_label = factor(country_levels_result, levels = country_levels_result),
  country_y = seq_along(country_levels_result)
)

audit_palette <- c(
  "Reporting visibility" = clr("black"),
  "Post-2023 context" = clr("teal")
)

audit_offsets <- tibble(
  audit_domain = factor(
    c("Reporting visibility", "Post-2023 context"),
    levels = c("Reporting visibility", "Post-2023 context")
  ),
  y_offset = c(0.13, -0.13)
)

audit_long <- bind_rows(
  audit_base %>%
    transmute(
      iso3,
      country_label,
      audit_domain = "Reporting visibility",
      median = post_reporting_multiplier_median,
      lower_95 = post_reporting_multiplier_lower_95,
      upper_95 = post_reporting_multiplier_upper_95
    ),
  audit_base %>%
    transmute(
      iso3,
      country_label,
      audit_domain = "Post-2023 context",
      median = post_transmission_multiplier_median,
      lower_95 = post_transmission_multiplier_lower_95,
      upper_95 = post_transmission_multiplier_upper_95
    )
) %>%
  mutate(
    audit_domain = factor(audit_domain, levels = c("Reporting visibility", "Post-2023 context")),
    country_label = factor(as.character(country_label), levels = country_levels_result)
  ) %>%
  left_join(country_axis, by = "country_label") %>%
  left_join(audit_offsets, by = "audit_domain") %>%
  mutate(country_y_dodged = country_y + y_offset)

audit_contrasts <- scenario_contrasts %>%
  filter(
    iso3 %in% pilot$iso3,
    scenario_id %in% c("no_reporting_testing_shift", "no_post_2023_transmission_context"),
    outcome_target == "reported_cases",
    period_start == 2023,
    period_end == 2025
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    across(
      c(
        scenario_to_full_ratio_median,
        scenario_to_full_ratio_lower_95,
        scenario_to_full_ratio_upper_95,
        full_minus_scenario_median,
        full_minus_scenario_lower_95,
        full_minus_scenario_upper_95
      ),
      as.numeric
    ),
    audit_domain = recode(
      scenario_id,
      no_reporting_testing_shift = "Reporting visibility",
      no_post_2023_transmission_context = "Post-2023 context"
    ),
    audit_domain = factor(audit_domain, levels = c("Reporting visibility", "Post-2023 context")),
    audit_key = recode(
      scenario_id,
      no_reporting_testing_shift = "reporting_visibility",
      no_post_2023_transmission_context = "post_2023_context"
    ),
    country_label = factor(as.character(country_label), levels = country_levels_result),
    reported_reduction_pct = 100 * (1 - scenario_to_full_ratio_median),
    reported_reduction_lower_95_pct = 100 * (1 - scenario_to_full_ratio_upper_95),
    reported_reduction_upper_95_pct = 100 * (1 - scenario_to_full_ratio_lower_95)
  ) %>%
  left_join(country_axis, by = "country_label") %>%
  left_join(audit_offsets, by = "audit_domain") %>%
  mutate(country_y_dodged = country_y + y_offset)

if (nrow(audit_contrasts) == 0) {
  stop("Figure 5 audit-neutralization scenario contrasts are empty.", call. = FALSE)
}

audit_contrast_data <- audit_contrasts %>%
  select(
    iso3,
    audit_key,
    scenario_to_full_ratio_median,
    scenario_to_full_ratio_lower_95,
    scenario_to_full_ratio_upper_95,
    reported_reduction_pct,
    reported_reduction_lower_95_pct,
    reported_reduction_upper_95_pct,
    full_minus_scenario_median,
    full_minus_scenario_lower_95,
    full_minus_scenario_upper_95
  ) %>%
  pivot_wider(
    names_from = audit_key,
    values_from = c(
      scenario_to_full_ratio_median,
      scenario_to_full_ratio_lower_95,
      scenario_to_full_ratio_upper_95,
      reported_reduction_pct,
      reported_reduction_lower_95_pct,
      reported_reduction_upper_95_pct,
      full_minus_scenario_median,
      full_minus_scenario_lower_95,
      full_minus_scenario_upper_95
    ),
    names_glue = "{audit_key}_{.value}"
  )

p_audit <- ggplot(audit_long, aes(y = country_y_dodged)) +
  geom_vline(xintercept = 1, colour = clr("grey"), linewidth = 0.28, linetype = "22") +
  geom_segment(
    aes(x = lower_95, xend = upper_95, yend = country_y_dodged, colour = audit_domain),
    linewidth = 0.58,
    alpha = 0.88
  ) +
  geom_point(
    aes(x = median, colour = audit_domain),
    shape = 16,
    size = 1.65,
    alpha = 0.95
  ) +
  scale_colour_manual(values = audit_palette, name = NULL) +
  scale_x_log10(
    labels = label_number(accuracy = 0.1),
    breaks = figure_breaks$ratio_2x[figure_breaks$ratio_2x <= 4],
    limits = c(0.38, 3.4),
    expand = expansion(mult = c(0.03, 0.06))
  ) +
  scale_y_continuous(
    breaks = country_axis$country_y,
    labels = as.character(country_axis$country_label),
    expand = expansion(add = c(0.42, 0.42))
  ) +
  labs(
    x = "Audit multiplier, log scale; point = median, line = 95% UI",
    y = NULL
  ) +
  theme_manuscript(grid = "x", plot_margin = margin(2, 3, 2, 3)) +
  theme(
    legend.position = "bottom",
    legend.justification = "left",
    legend.box.just = "left",
    legend.key.width = unit(4.4, "mm"),
    legend.margin = margin(0, 0, 0, 0)
  )

p_burden <- ggplot(audit_contrasts, aes(y = country_y_dodged)) +
  geom_vline(xintercept = 1, colour = clr("grey"), linewidth = 0.28, linetype = "22") +
  geom_segment(
    aes(
      x = scenario_to_full_ratio_lower_95,
      xend = scenario_to_full_ratio_upper_95,
      yend = country_y_dodged,
      colour = audit_domain
    ),
    linewidth = 0.58,
    alpha = 0.88
  ) +
  geom_point(
    aes(x = scenario_to_full_ratio_median, colour = audit_domain),
    shape = 16,
    size = 1.65,
    alpha = 0.95
  ) +
  scale_colour_manual(values = audit_palette, name = NULL, guide = "none") +
  scale_x_log10(
    breaks = c(0.05, 0.1, 0.25, 0.5, 1, 2, 4),
    labels = c("0.05", "0.1", "0.25", "0.5", "1", "2", "4"),
    limits = c(0.035, 3.8),
    expand = expansion(mult = c(0.02, 0.04))
  ) +
  scale_y_continuous(
    breaks = country_axis$country_y,
    labels = NULL,
    expand = expansion(add = c(0.42, 0.42))
  ) +
  labs(
    x = "Reported burden ratio after audit neutralization, log scale",
    y = NULL
  ) +
  theme_manuscript(grid = "x", plot_margin = margin(2, 3, 2, 3)) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "none"
  )

figure <- panel_tags(
  p_audit + p_burden +
    plot_layout(widths = c(1.34, 1.06), guides = "collect")
) &
  theme(
    legend.position = "bottom",
    legend.justification = "left",
    plot.tag = element_text(
      face = "bold",
      size = figure_text$tag,
      margin = margin(0, 2, 1, 0)
    )
  )

figure_order <- audit_order %>%
  mutate(figure5_country_order = row_number()) %>%
  select(iso3, figure5_country_order, max_upward_audit_multiplier)

figure_data <- audit_base %>%
  left_join(figure_order, by = "iso3") %>%
  left_join(audit_contrast_data, by = "iso3") %>%
  transmute(
    iso3,
    country,
    country_label = as.character(country_label),
    figure5_country_order,
    direct_testing_support,
    direct_testing_series_support_class,
    direct_testing_series_years_available,
    testing_prior_class = as.character(testing_prior_class),
    reporting_prior_mean,
    reporting_prior_lower,
    reporting_prior_upper,
    reporting_prior_log_sd,
    post_reporting_multiplier_median,
    post_reporting_multiplier_lower_50,
    post_reporting_multiplier_upper_50,
    post_reporting_multiplier_lower_95,
    post_reporting_multiplier_upper_95,
    reporting_log2_multiplier = log2(post_reporting_multiplier_median),
    post_transmission_multiplier_median,
    post_transmission_multiplier_lower_50,
    post_transmission_multiplier_upper_50,
    post_transmission_multiplier_lower_95,
    post_transmission_multiplier_upper_95,
    context_log2_multiplier = log2(post_transmission_multiplier_median),
    max_upward_audit_multiplier,
    reporting_visibility_scenario_to_full_ratio_median,
    reporting_visibility_scenario_to_full_ratio_lower_95,
    reporting_visibility_scenario_to_full_ratio_upper_95,
    reporting_visibility_reported_reduction_pct,
    reporting_visibility_reported_reduction_lower_95_pct,
    reporting_visibility_reported_reduction_upper_95_pct,
    reporting_visibility_full_minus_scenario_median,
    reporting_visibility_full_minus_scenario_lower_95,
    reporting_visibility_full_minus_scenario_upper_95,
    post_2023_context_scenario_to_full_ratio_median,
    post_2023_context_scenario_to_full_ratio_lower_95,
    post_2023_context_scenario_to_full_ratio_upper_95,
    post_2023_context_reported_reduction_pct,
    post_2023_context_reported_reduction_lower_95_pct,
    post_2023_context_reported_reduction_upper_95_pct,
    post_2023_context_full_minus_scenario_median,
    post_2023_context_full_minus_scenario_lower_95,
    post_2023_context_full_minus_scenario_upper_95,
    interpretation_status = case_when(
      !direct_testing_support & evidence_available_layers <= 4 ~ "Audit-limited wording",
      reporting_prior_sensitive |
        contact_boundary_sensitive |
        post_reporting_multiplier_median > 1.20 |
        post_transmission_multiplier_median > 1.10 ~ "Boundary-sensitive",
      TRUE ~ "Transmission-consistent"
    ),
    reporting_prior_sensitive,
    contact_boundary_sensitive,
    posterior_reporting_shift_probability,
    posterior_reporting_peak_probability,
    posterior_reporting_trend_probability,
    posterior_transmission_context_shift_probability,
    posterior_pathogen_pressure_active_probability,
    posterior_outbreak_seeding_active_probability,
    posterior_secondary_seeding_active_probability,
    observation_process_max_probability = pmax(
      posterior_reporting_shift_probability,
      posterior_reporting_peak_probability,
      posterior_reporting_trend_probability,
      na.rm = TRUE
    ),
    context_audit_max_probability = pmax(
      posterior_transmission_context_shift_probability,
      posterior_pathogen_pressure_active_probability,
      posterior_outbreak_seeding_active_probability,
      posterior_secondary_seeding_active_probability,
      na.rm = TRUE
    ),
    audit_max_probability,
    audit_load = as.character(audit_load),
    reporting_shift_abs_log2 = abs(log2(post_reporting_multiplier_median)),
    evidence_available_layers,
    in_model_evidence_layers,
    external_boundary_layers,
    unavailable_evidence_layers,
    observation_process_supported,
    observation_process_role,
    testing_denominator_evidence_available = coalesce(evidence_available_direct_testing_denominator, FALSE),
    testing_denominator_evidence_role = case_when(
      direct_testing_support ~ "In-model constraint",
      coalesce(evidence_available_direct_testing_denominator, FALSE) ~ "External boundary",
      TRUE ~ "Not available"
    ),
    reporting_practice_evidence_available = coalesce(evidence_available_reporting_practice_case_definition, FALSE),
    reporting_practice_evidence_role = coalesce(as.character(evidence_role_reporting_practice_case_definition), "Not available"),
    severe_outcome_evidence_available = coalesce(evidence_available_official_severe_outcomes, FALSE),
    severe_outcome_evidence_role = coalesce(as.character(evidence_role_official_severe_outcomes), "Not available"),
    age_pattern_evidence_available = coalesce(evidence_available_age_pattern_evidence, FALSE),
    age_pattern_evidence_role = coalesce(as.character(evidence_role_age_pattern_evidence), "Not available"),
    pathogen_context_evidence_available = coalesce(evidence_available_pathogen_amr_timeline, FALSE),
    pathogen_context_evidence_role = coalesce(as.character(evidence_role_pathogen_amr_timeline), "Not available"),
    contact_context_evidence_available = coalesce(evidence_available_contact_validation, FALSE),
    contact_context_evidence_role = coalesce(as.character(evidence_role_contact_validation), "Not available"),
    reporting_practice_high_evidence_fraction,
    reporting_practice_detection_prior_available,
    official_surveillance_outcome_support_level
  )

readr::write_csv(figure_data, fd_path("figure_5_model2c_observation_boundary.csv"))
save_supplement_figure(
  figure,
  "figure_5_model2c_observation_boundary",
  nature_full_width_mm,
  124
)
