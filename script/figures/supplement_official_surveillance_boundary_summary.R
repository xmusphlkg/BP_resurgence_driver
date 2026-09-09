#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

boundary <- read_csv_checked(
  out_path("official_surveillance_boundary_summary.csv"),
  c(
    "iso3",
    "country",
    "region",
    "recommended_tier",
    "official_surveillance_boundary_label",
    "official_surveillance_outcome_available",
    "direct_testing_denominator_series_available",
    "annual_national_surveillance_summary_available",
    "age_pattern_case_distribution_available",
    "age_pattern_severe_outcome_signature_available",
    "hospitalization_available",
    "death_available"
  )
) %>%
  mutate(
    across(
      c(
        official_surveillance_outcome_available,
        direct_testing_denominator_series_available,
        annual_national_surveillance_summary_available,
        age_pattern_case_distribution_available,
        age_pattern_severe_outcome_signature_available,
        hospitalization_available,
        death_available
      ),
      as.logical
    ),
    region = factor(region, levels = names(region_palette)),
    recommended_tier = factor(
      recommended_tier,
      levels = c("Tier 1 dynamic pilot", "Tier 0 descriptive", "Needs data")
    ),
    boundary_group = factor(
      official_surveillance_boundary_label,
      levels = c(
        "annual_surveillance_boundary",
        "multi_year_direct_series_boundary",
        "official_surveillance_and_age_pattern_boundary",
        "official_surveillance_boundary"
      ),
      labels = c(
        "Annual surveillance\nboundary",
        "Multi-year direct\nseries boundary",
        "Official surveillance\n+ age-pattern boundary",
        "Official surveillance\nboundary"
      )
    )
  ) %>%
  arrange(boundary_group, region, recommended_tier, country) %>%
  mutate(country_label = factor(iso3, levels = iso3))

evidence_columns <- c(
  "official_surveillance_outcome_available",
  "direct_testing_denominator_series_available",
  "annual_national_surveillance_summary_available",
  "age_pattern_case_distribution_available",
  "age_pattern_severe_outcome_signature_available",
  "hospitalization_available",
  "death_available"
)

evidence_labels <- c(
  official_surveillance_outcome_available = "Official outcome",
  direct_testing_denominator_series_available = "Direct testing\nseries",
  annual_national_surveillance_summary_available = "Annual national\nsummary",
  age_pattern_case_distribution_available = "Age-pattern\ncase dist.",
  age_pattern_severe_outcome_signature_available = "Age-pattern\nsevere out.",
  hospitalization_available = "Hospitalization",
  death_available = "Death"
)

evidence_counts <- vapply(
  evidence_columns,
  function(column) sum(boundary[[column]], na.rm = TRUE),
  integer(1)
)

evidence_labels <- paste0(
  evidence_labels,
  "\n(",
  evidence_counts,
  "/",
  nrow(boundary),
  ")"
)

boundary_long <- boundary %>%
  select(country_label, boundary_group, all_of(evidence_columns)) %>%
  pivot_longer(
    cols = all_of(evidence_columns),
    names_to = "evidence_domain",
    values_to = "present"
  ) %>%
  mutate(
    present = as.logical(present),
    evidence_domain = factor(evidence_domain, levels = evidence_columns, labels = unname(evidence_labels)),
    status = if_else(present, "present", "missing")
  )

present_palette <- c(
  present = clr("green"),
  missing = clr("light_grey")
)

p_boundary <- ggplot(boundary_long, aes(evidence_domain, country_label, fill = status)) +
  geom_tile(colour = "white", linewidth = 0.35) +
  facet_grid(boundary_group ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(
    values = present_palette,
    breaks = c("present", "missing"),
    labels = c("Present", "Absent"),
    name = "Evidence"
  ) +
  labs(
    x = "Evidence domain",
    y = NULL
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text, lineheight = 0.95),
    axis.text.y = element_text(size = figure_text$axis_text_dense),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.spacing.y = unit(1.15, "mm"),
    strip.placement = "outside",
    strip.background = element_rect(fill = pal["paper"], colour = NA),
    strip.text.y.left = element_text(
      angle = 0,
      face = "bold",
      size = figure_text$strip_dense,
      lineheight = 0.90
    ),
    legend.position = "bottom",
    legend.title = element_text(size = figure_text$legend_title),
    legend.text = element_text(size = figure_text$legend_text),
    plot.margin = margin(4, 4, 6, 12)
  )

figure <- panel_tags(p_boundary)

save_nature(figure, "supplement_official_surveillance_boundary_summary.png", nature_full_width_mm, 168)
