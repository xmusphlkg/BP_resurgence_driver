#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

summary <- read_csv_checked(
  fd_path("age_pattern_global_summary.csv"),
  c(
    "iso3",
    "country",
    "age_pattern_coverage_class",
    "total_rows",
    "reported_case_age_distribution_rows",
    "reported_case_age_distribution_year_min",
    "reported_case_age_distribution_year_max",
    "reported_case_age_distribution_year_span",
    "reported_case_age_distribution_dominant_signature",
    "reported_case_age_distribution_dominant_signature_count",
    "reported_case_age_distribution_dominant_signature_share_percent",
    "reported_case_age_distribution_unique_signatures",
    "official_severe_outcome_age_signature_rows",
    "official_severe_outcome_age_signature_year_min",
    "official_severe_outcome_age_signature_year_max",
    "official_severe_outcome_age_signature_year_span",
    "official_severe_outcome_age_signature_dominant_signature",
    "official_severe_outcome_age_signature_dominant_signature_count",
    "official_severe_outcome_age_signature_dominant_signature_share_percent",
    "official_severe_outcome_age_signature_unique_signatures",
    "summary_note"
  )
) %>%
  mutate(
    age_pattern_coverage_class = factor(
      age_pattern_coverage_class,
      levels = c("both_families", "reported_case_only", "official_severe_outcome_only"),
      labels = c("Both families", "Reported-case only", "Severe-outcome only")
    ),
    reported_case_age_distribution_year_min = as.integer(reported_case_age_distribution_year_min),
    reported_case_age_distribution_year_max = as.integer(reported_case_age_distribution_year_max),
    reported_case_age_distribution_year_span = as.integer(reported_case_age_distribution_year_span),
    reported_case_age_distribution_rows = as.integer(reported_case_age_distribution_rows),
    official_severe_outcome_age_signature_year_min = as.integer(official_severe_outcome_age_signature_year_min),
    official_severe_outcome_age_signature_year_max = as.integer(official_severe_outcome_age_signature_year_max),
    official_severe_outcome_age_signature_year_span = as.integer(official_severe_outcome_age_signature_year_span),
    official_severe_outcome_age_signature_rows = as.integer(official_severe_outcome_age_signature_rows)
  )

coverage_counts <- summary %>%
  count(age_pattern_coverage_class, name = "countries") %>%
  mutate(
    age_pattern_coverage_class = factor(
      age_pattern_coverage_class,
      levels = c("Both families", "Reported-case only", "Severe-outcome only")
    )
  ) %>%
  complete(
    age_pattern_coverage_class = factor(
      c("Both families", "Reported-case only", "Severe-outcome only"),
      levels = c("Both families", "Reported-case only", "Severe-outcome only")
    ),
    fill = list(countries = 0)
  )

coverage_palette <- c(
  "Both families" = clr("green"),
  "Reported-case only" = clr("blue"),
  "Severe-outcome only" = clr("orange")
)

signature_palette <- c(
  infant_or_early_child_peak = clr("blue"),
  school_age_peak = clr("orange"),
  young_adult_peak = clr("purple"),
  adult_peak = clr("red"),
  mixed_age_distribution = pal["grey"],
  young_infant_severe_outcome_signature = clr("teal"),
  infant_severe_outcome_signature = clr("green"),
  official_severe_outcome_summary = clr("gold")
)

signature_labels <- c(
  infant_or_early_child_peak = "Infant / early child",
  school_age_peak = "School-age",
  young_adult_peak = "Young adult",
  adult_peak = "Adult",
  mixed_age_distribution = "Mixed",
  young_infant_severe_outcome_signature = "Young infant",
  infant_severe_outcome_signature = "Infant",
  official_severe_outcome_summary = "Summary only"
)

reported <- summary %>%
  filter(reported_case_age_distribution_rows > 0) %>%
  arrange(
    desc(reported_case_age_distribution_year_span),
    desc(reported_case_age_distribution_rows),
    reported_case_age_distribution_year_min,
    iso3
  ) %>%
  mutate(iso3 = factor(iso3, levels = rev(unique(iso3))))

severe <- summary %>%
  filter(official_severe_outcome_age_signature_rows > 0) %>%
  arrange(
    desc(official_severe_outcome_age_signature_year_span),
    desc(official_severe_outcome_age_signature_rows),
    official_severe_outcome_age_signature_year_min,
    iso3
  ) %>%
  mutate(iso3 = factor(iso3, levels = rev(unique(iso3))))

p_coverage <- ggplot(
  coverage_counts,
  aes(x = age_pattern_coverage_class, y = countries, fill = age_pattern_coverage_class)
) +
  geom_col(width = 0.68) +
  geom_text(
    aes(label = paste0("n = ", countries)),
    vjust = -0.38,
    size = figure_text$annotation_large,
    colour = pal["dark_grey"]
  ) +
  scale_fill_manual(values = coverage_palette, name = "Coverage class") +
  scale_y_continuous(
    limits = c(0, max(coverage_counts$countries, na.rm = TRUE) + 4),
    breaks = seq(0, max(coverage_counts$countries, na.rm = TRUE), by = 5),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = NULL,
    y = "Countries"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    axis.title.y = element_text(size = figure_text$axis_title),
    plot.margin = margin(5, 5, 7, 5)
  )

p_reported <- ggplot(
  reported,
  aes(
    y = iso3,
    x = reported_case_age_distribution_year_min,
    xend = reported_case_age_distribution_year_max,
    colour = reported_case_age_distribution_dominant_signature
  )
) +
  geom_segment(linewidth = 1.0, lineend = "butt") +
  geom_point(size = 1.25) +
  scale_colour_manual(
    values = signature_palette,
    breaks = names(signature_labels),
    labels = unname(signature_labels),
    limits = names(signature_labels),
    drop = FALSE,
    name = "Dominant signature",
    guide = guide_legend(ncol = 4, byrow = TRUE)
  ) +
  scale_x_continuous(
    breaks = figure_breaks_year(1990, 2025, 5),
    limits = c(1990, 2026),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(
    x = "Year",
    y = "ISO3"
  ) +
  theme_nature() +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text_dense),
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.title.y = element_text(size = figure_text$axis_title),
    plot.margin = margin(5, 5, 3, 5),
    panel.grid.major.x = element_line(linewidth = 0.18, colour = pal["light_grey"]),
    panel.grid.major.y = element_blank()
  )

p_severe <- ggplot(
  severe,
  aes(
    y = iso3,
    x = official_severe_outcome_age_signature_year_min,
    xend = official_severe_outcome_age_signature_year_max,
    colour = official_severe_outcome_age_signature_dominant_signature
  )
) +
  geom_segment(linewidth = 1.0, lineend = "butt") +
  geom_point(size = 1.25) +
  scale_colour_manual(
    values = signature_palette,
    breaks = names(signature_labels),
    labels = unname(signature_labels),
    limits = names(signature_labels),
    drop = FALSE,
    name = "Dominant signature",
    guide = guide_legend(ncol = 4, byrow = TRUE)
  ) +
  scale_x_continuous(
    breaks = figure_breaks_year(1990, 2025, 5),
    limits = c(1990, 2026),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(
    x = "Year",
    y = "ISO3"
  ) +
  theme_nature() +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text_dense),
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.title.y = element_text(size = figure_text$axis_title),
    plot.margin = margin(5, 5, 2, 5),
    panel.grid.major.x = element_line(linewidth = 0.18, colour = pal["light_grey"]),
    panel.grid.major.y = element_blank()
  )

layout <- "
AB
CB
"

figure <- panel_tags(
  (p_coverage + p_reported + p_severe) +
    plot_layout(design = layout, widths = c(0.70, 1.30), guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = figure_text$legend_text),
      legend.title = element_text(size = figure_text$legend_title)
    )
)

save_nature(
  figure,
  "supplement_age_pattern_global_summary.png",
  nature_full_width_mm,
  168
)
