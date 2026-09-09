#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

status_palette <- c(
  supported = clr("green"),
  partial = clr("orange"),
  caution = clr("red"),
  missing = pal["light_grey"]
)

status_labels <- c(
  supported = "Supported",
  partial = "Partial",
  caution = "Caution",
  missing = "Missing"
)

status_abbrev <- c(
  supported = "D",
  partial = "P",
  caution = "C",
  missing = "M"
)

score_caption <- str_wrap(
  "Score = mean of eight audited domains: mechanism claim, monthly detection prior, joint age-month evidence, source age-bin evidence, external validation, mechanism profile, policy implementation, and current-trend bridge. Supported = 1, partial = 0.67, caution = 0.33, missing = 0.",
  width = 110
)

score_map <- c(
  supported = 1.0,
  partial = 0.67,
  caution = 0.33,
  missing = 0.0
)

audit <- read_csv_checked(
  out_path("mechanism_claim_readiness_audit.csv"),
  c(
    "iso3",
    "country",
    "mechanism_claim_strength",
    "monthly_detection_prior_support_level",
    "joint_age_month_evidence_level",
    "source_age_bin_evidence_level",
    "external_validation_constraint_level",
    "mechanism_profile_fragility_level",
    "policy_implementation_evidence_level",
    "policy_current_trend_support_level"
  )
) %>%
  select(
    iso3,
    country,
    mechanism_claim_strength,
    monthly_detection_prior_support_level,
    joint_age_month_evidence_level,
    source_age_bin_evidence_level,
    external_validation_constraint_level,
    mechanism_profile_fragility_level,
    policy_implementation_evidence_level,
    policy_current_trend_support_level
  ) %>%
  mutate(
    mechanism_claim_status = case_when(
      mechanism_claim_strength == "moderate_reduced_model_attribution" ~ "supported",
      mechanism_claim_strength == "exploratory_sensitive_attribution" ~ "partial",
      mechanism_claim_strength == "low_negative_control_or_weak_resurgence" ~ "caution",
      TRUE ~ "missing"
    ),
    monthly_detection_status = case_when(
      monthly_detection_prior_support_level == "direct_testing_denominator_monthly_prior_supported" ~ "supported",
      monthly_detection_prior_support_level == "single_year_direct_testing_denominator_summary_supported" ~ "partial",
      monthly_detection_prior_support_level == "diagnostic_timeline_monthly_detection_prior_supported" ~ "partial",
      TRUE ~ "missing"
    ),
    joint_age_month_status = case_when(
      joint_age_month_evidence_level == "country_age_month_composite_likelihood_supported" ~ "supported",
      joint_age_month_evidence_level == "country_age_month_composite_likelihood_available_but_fragile" ~ "partial",
      TRUE ~ "missing"
    ),
    source_age_bin_status = case_when(
      source_age_bin_evidence_level == "source_age_bin_likelihood_supported" ~ "supported",
      source_age_bin_evidence_level == "source_age_bin_likelihood_available_but_fragile" ~ "partial",
      TRUE ~ "missing"
    ),
    external_validation_status = case_when(
      external_validation_constraint_level == "key_external_validation_sources_available" ~ "supported",
      external_validation_constraint_level == "external_validation_partial_proxy_contact_only" ~ "partial",
      TRUE ~ "missing"
    ),
    mechanism_profile_status = case_when(
      mechanism_profile_fragility_level == "moderate_annual_mechanism_profile_fragility" ~ "partial",
      mechanism_profile_fragility_level == "high_annual_mechanism_profile_fragility" ~ "caution",
      TRUE ~ "missing"
    ),
    policy_implementation_status = case_when(
      policy_implementation_evidence_level == "policy_grid_stable_with_local_program_support" ~ "supported",
      policy_implementation_evidence_level == "policy_requires_new_maternal_program_assumption" ~ "partial",
      TRUE ~ "missing"
    ),
    policy_current_trend_status = case_when(
      policy_current_trend_support_level == "current_year_partial_observation_policy_bridge" ~ "supported",
      policy_current_trend_support_level == "no_current_year_policy_bridge" ~ "missing",
      TRUE ~ "missing"
    )
  ) %>%
  mutate(
    readiness_score = rowMeans(
      cbind(
        score_map[mechanism_claim_status],
        score_map[monthly_detection_status],
        score_map[joint_age_month_status],
        score_map[source_age_bin_status],
        score_map[external_validation_status],
        score_map[mechanism_profile_status],
        score_map[policy_implementation_status],
        score_map[policy_current_trend_status]
      ),
      na.rm = TRUE
    )
  ) %>%
  arrange(desc(readiness_score), desc(score_map[mechanism_claim_status]), iso3)

summary_path <- fd_path("mechanism_claim_readiness_summary.csv")
summary_export <- audit %>%
  transmute(
    iso3,
    country,
    readiness_score = round(readiness_score, 3),
    mechanism_claim_status,
    monthly_detection_status,
    joint_age_month_status,
    source_age_bin_status,
    external_validation_status,
    mechanism_profile_status,
    policy_implementation_status,
    policy_current_trend_status
  )
write_csv(summary_export, summary_path)

plot_order <- summary_export$iso3

score_df <- summary_export %>%
  mutate(
    country = factor(iso3, levels = plot_order),
    mechanism_claim_status = factor(mechanism_claim_status, levels = names(status_palette)),
    readiness_score = as.numeric(readiness_score)
  )

domain_labels <- c(
  mechanism_claim_status = "Mechanism\nclaim",
  monthly_detection_status = "Monthly\ndetection",
  joint_age_month_status = "Joint\nage-month",
  source_age_bin_status = "Source\nage-bin",
  external_validation_status = "External\nvalidation",
  mechanism_profile_status = "Mechanism\nprofile",
  policy_implementation_status = "Policy\nimplementation",
  policy_current_trend_status = "Current-trend\nbridge"
)

matrix_df <- summary_export %>%
  select(iso3, all_of(names(domain_labels))) %>%
  pivot_longer(
    cols = all_of(names(domain_labels)),
    names_to = "domain",
    values_to = "status"
  ) %>%
  mutate(
    country = factor(iso3, levels = plot_order),
    domain = factor(domain, levels = names(domain_labels)),
    status = factor(status, levels = names(status_palette)),
    status_code = unname(status_abbrev[as.character(status)])
  )

p_score <- ggplot(score_df, aes(x = readiness_score, y = country)) +
  geom_segment(
    aes(x = 0, xend = readiness_score, yend = country),
    linewidth = 0.5,
    colour = pal["light_grey"]
  ) +
  geom_point(aes(fill = mechanism_claim_status), shape = 21, size = 3.2, colour = pal["black"], stroke = 0.35) +
  geom_text(
    aes(label = sprintf("%.2f", readiness_score)),
    hjust = -0.2,
    size = 2.25,
    colour = pal["dark_grey"]
  ) +
  scale_fill_manual(values = status_palette, breaks = names(status_labels), labels = status_labels, drop = FALSE) +
  scale_x_continuous(
    limits = c(0, 1.05),
    breaks = c(0, 0.25, 0.5, 0.75, 1.0),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_y_discrete(limits = rev(plot_order)) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Composite readiness score",
    y = NULL,
    fill = "Mechanism claim"
  ) +
  theme_nature(base_size = 7.8) +
  theme(
    axis.text.y = element_text(size = 6.2),
    axis.text.x = element_text(size = 6.0),
    legend.position = "bottom",
    plot.margin = margin(4, 4, 4, 4)
  )

p_matrix <- ggplot(matrix_df, aes(x = domain, y = country, fill = status)) +
  geom_tile(colour = "white", linewidth = 0.38) +
  geom_text(
    data = filter(matrix_df, !is.na(status_code)),
    aes(label = status_code),
    size = 2.5,
    colour = pal["black"]
  ) +
  scale_fill_manual(
    values = status_palette,
    breaks = names(status_labels),
    labels = status_labels,
    drop = FALSE
  ) +
  scale_x_discrete(labels = domain_labels) +
  scale_y_discrete(limits = rev(levels(score_df$country))) +
  labs(
    x = "Audited readiness domain",
    y = NULL,
    fill = "Evidence class"
  ) +
  theme_nature(base_size = 7.6) +
  theme(
    axis.text.x = element_text(size = 6.0, angle = 24, hjust = 1),
    axis.text.y = element_text(size = 6.0),
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(4, 4, 2, 4)
  )

figure <- panel_tags(
  (p_score | p_matrix) +
    plot_layout(widths = c(0.82, 1.38), guides = "keep")
)

save_nature(figure, "supplement_mechanism_claim_readiness_summary.png", nature_full_width_mm, 124)
