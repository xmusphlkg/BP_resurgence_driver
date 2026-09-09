#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

lattice <- read_csv_checked(
  fd_path("age_pattern_validation_lattice_summary.csv"),
  c(
    "iso3",
    "country",
    "age_pattern_coverage_class",
    "validation_score",
    "validation_class",
    "age_pattern_case_distribution_available",
    "age_pattern_severe_outcome_signature_available",
    "testing_denominator_available",
    "direct_testing_denominator_series_available",
    "annual_national_surveillance_summary_available",
    "testing_reporting_case_definition_available",
    "serology_available",
    "contact_validation_available",
    "hospitalization_available",
    "death_available",
    "summary_note"
  )
) %>%
  mutate(
    validation_score = as.integer(validation_score),
    validation_class = factor(
      validation_class,
      levels = c("rich_validation", "mixed_validation", "sparse_validation"),
      labels = c("Dense evidence", "Intermediate evidence", "Sparse evidence")
    ),
    validation_status = case_when(
      validation_class == "Dense evidence" ~ "supported",
      validation_class == "Intermediate evidence" ~ "partial",
      TRUE ~ "missing"
    ),
    age_pattern_coverage_class = factor(
      age_pattern_coverage_class,
      levels = c("both_families", "reported_case_only", "official_severe_outcome_only"),
      labels = c("Both families", "Reported-case only", "Severe-outcome only")
    )
  ) %>%
  arrange(desc(validation_score), country) %>%
  mutate(
    iso3 = factor(iso3, levels = rev(unique(iso3)))
  )

domain_order <- c(
  "age_pattern_case_distribution_available",
  "age_pattern_severe_outcome_signature_available",
  "testing_denominator_available",
  "direct_testing_denominator_series_available",
  "annual_national_surveillance_summary_available",
  "testing_reporting_case_definition_available",
  "serology_available",
  "contact_validation_available",
  "hospitalization_available",
  "death_available"
)

domain_labels <- c(
  age_pattern_case_distribution_available = "Age pattern\ncase dist",
  age_pattern_severe_outcome_signature_available = "Age pattern\nsevere sig",
  testing_denominator_available = "Testing\navailable",
  direct_testing_denominator_series_available = "Direct testing\nseries",
  annual_national_surveillance_summary_available = "Annual national\nsummary",
  testing_reporting_case_definition_available = "Case-definition\ncovariates",
  serology_available = "Serology",
  contact_validation_available = "Direct contact\nvalidation",
  hospitalization_available = "Hospitalization",
  death_available = "Death"
)

domain_long <- lattice %>%
  select(iso3, country, validation_score, validation_status, all_of(domain_order)) %>%
  pivot_longer(
    cols = all_of(domain_order),
    names_to = "domain",
    values_to = "available"
  ) %>%
  mutate(
    domain = factor(domain, levels = domain_order, labels = unname(domain_labels)),
    domain_status = if_else(available, "supported", "missing"),
    validation_status = factor(validation_status, levels = c("supported", "partial", "missing")),
    iso3 = factor(iso3, levels = levels(lattice$iso3)),
    country = factor(country, levels = rev(unique(country)))
  )

bar_panel <- ggplot(
  lattice,
  aes(y = iso3, x = validation_score, fill = validation_status)
) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.15) +
  scale_fill_manual(
    values = c(
      supported = clr("green"),
      partial = clr("orange"),
      missing = pal["light_grey"]
    ),
    breaks = c("supported", "partial", "missing"),
    labels = c("Dense evidence", "Intermediate evidence", "Sparse evidence"),
    name = "Validation class"
  ) +
  scale_x_continuous(
    limits = c(0, 10.2),
    breaks = seq(0, 10, by = 2),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = "Evidence score",
    y = "Country"
  ) +
  theme_nature() +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text_dense),
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.title.y = element_text(size = figure_text$axis_title),
    plot.margin = margin(5, 5, 4, 5)
  )

heatmap_panel <- ggplot(
  domain_long,
  aes(x = domain, y = iso3, fill = domain_status)
) +
  geom_tile(colour = "white", linewidth = 0.18) +
  scale_fill_manual(
    values = c(
      supported = clr("green"),
      missing = pal["light_grey"]
    ),
    breaks = c("supported", "missing"),
    labels = c("Available", "Unavailable"),
    name = "Domain availability"
  ) +
  scale_x_discrete(expand = expansion(mult = c(0.01, 0.01))) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(
      size = figure_text$axis_text,
      angle = 34,
      hjust = 1,
      vjust = 1
    ),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.margin = margin(4, 5, 2, 5),
    legend.position = "bottom"
  )

figure <- panel_tags(
  (bar_panel | heatmap_panel) +
    plot_layout(widths = c(0.58, 1.42), guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = figure_text$legend_text),
      legend.title = element_text(size = figure_text$legend_title)
    )
)

save_nature(
  figure,
  "supplement_age_pattern_validation_lattice.png",
  nature_full_width_mm,
  132
)
