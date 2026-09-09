#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

ecdc_levels <- c(
  "all_overlaps",
  "country"
)

ecdc_labels <- c(
  all_overlaps = "All overlaps",
  country = "Country"
)

ecdc_palette <- c(
  all_overlaps = clr("dark_grey"),
  country = clr("blue")
)

ecdc_summary <- read_csv_checked(
  out_path("ecdc_who_reported_case_concordance_summary.csv"),
  c(
    "scope",
    "iso3",
    "country",
    "overlap_years",
    "mean_ecdc_to_who_ratio",
    "median_ecdc_to_who_ratio",
    "mean_absolute_relative_difference",
    "within_factor2_fraction",
    "ecdc_year_min",
    "ecdc_year_max",
    "comparison_note"
  )
) %>%
  mutate(
    scope = factor(scope, levels = ecdc_levels),
    display_label = if_else(scope == "all_overlaps", "All overlaps", iso3),
    display_label = if_else(display_label == "", iso3, display_label)
  ) %>%
  arrange(desc(scope), desc(median_ecdc_to_who_ratio), desc(overlap_years), display_label) %>%
  mutate(
    display_label = factor(display_label, levels = rev(unique(display_label))),
    point_size = if_else(scope == "all_overlaps", pmin(overlap_years, 25L), overlap_years)
  )

p_ecdc <- ggplot(ecdc_summary, aes(x = median_ecdc_to_who_ratio, y = display_label)) +
  geom_segment(
    aes(x = 1, xend = median_ecdc_to_who_ratio, yend = display_label),
    linewidth = 0.68,
    colour = pal["light_grey"]
  ) +
  geom_point(
    aes(size = point_size, colour = scope),
    alpha = 0.92
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.42, colour = pal["grey"]) +
  scale_colour_manual(
    values = ecdc_palette,
    breaks = ecdc_levels,
    labels = unname(ecdc_labels),
    name = "Scope"
  ) +
  scale_size_area(max_size = 4.8, guide = "none") +
  scale_x_continuous(
    breaks = figure_breaks$ratio_2x[figure_breaks$ratio_2x <= 4],
    limits = c(0.45, 4.75),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    x = "Median ECDC / WHO annual reported-case ratio",
    y = "EU/EEA country or aggregate"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    plot.margin = margin(5, 6, 8, 5)
  ) +
  coord_cartesian()

figure <- panel_tags(p_ecdc)

save_nature(figure, "supplement_ecdc_who_concordance.png", nature_full_width_mm, 112)
