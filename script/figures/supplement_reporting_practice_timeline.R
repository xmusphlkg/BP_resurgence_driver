#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

collapse_unique <- function(x) {
  values <- unique(trimws(as.character(x)))
  values <- values[!is.na(values) & nzchar(values) & values != "nan" & values != "not_available"]
  if (length(values) == 0) {
    "not_available"
  } else {
    paste(values, collapse = "; ")
  }
}

timeline <- read_csv_checked(
  fd_path("reporting_practice_timeline.csv"),
  c(
    "iso3",
    "country",
    "country_panel",
    "period_start",
    "period_end",
    "surveillance_regime",
    "primary_diagnostic_methods",
    "case_definition_or_reporting_change",
    "relative_detection_prior_mean",
    "relative_detection_prior_lower",
    "relative_detection_prior_upper",
    "effect_direction",
    "evidence_strength",
    "source_ids"
  )
) %>%
  mutate(
    period_start = as_plot_date(period_start),
    period_end = as_plot_date(period_end),
    period_span_years = as.numeric(period_end - period_start) / 365.25,
    relative_detection_prior_mean = as.numeric(relative_detection_prior_mean),
    relative_detection_prior_lower = as.numeric(relative_detection_prior_lower),
    relative_detection_prior_upper = as.numeric(relative_detection_prior_upper),
    country_panel = iso3
  )

summary <- timeline %>%
  group_by(iso3, country, country_panel) %>%
  summarise(
    reporting_rows = n(),
    high_evidence_rows = sum(as.character(evidence_strength) %in% c("high", "moderate"), na.rm = TRUE),
    timeline_start_min = min(period_start, na.rm = TRUE),
    timeline_end_max = max(period_end, na.rm = TRUE),
    prior_median = median(relative_detection_prior_mean, na.rm = TRUE),
    prior_min = min(relative_detection_prior_mean, na.rm = TRUE),
    prior_max = max(relative_detection_prior_mean, na.rm = TRUE),
    prior_lower_median = median(relative_detection_prior_lower, na.rm = TRUE),
    prior_upper_median = median(relative_detection_prior_upper, na.rm = TRUE),
    regime_count = n_distinct(surveillance_regime),
    method_count = n_distinct(primary_diagnostic_methods),
    effect_count = n_distinct(effect_direction),
    source_count = n_distinct(source_ids),
    regime_summary = collapse_unique(surveillance_regime),
    method_summary = collapse_unique(primary_diagnostic_methods),
    effect_summary = collapse_unique(effect_direction),
    source_summary = collapse_unique(source_ids),
    .groups = "drop"
  ) %>%
  mutate(
    timeline_span_years = as.numeric(timeline_end_max - timeline_start_min) / 365.25,
    high_evidence_fraction = high_evidence_rows / reporting_rows
  ) %>%
  arrange(prior_median, country)

country_levels <- summary %>%
  pull(country_panel) %>%
  as.character()

summary <- summary %>%
  mutate(country_panel = factor(country_panel, levels = country_levels))

timeline <- timeline %>%
  mutate(country_panel = factor(country_panel, levels = country_levels))

p_timeline <- ggplot(timeline, aes(y = country_panel)) +
  geom_segment(
    aes(
      x = period_start,
      xend = period_end,
      yend = country_panel,
      colour = relative_detection_prior_mean
    ),
    linewidth = 2.0,
    lineend = "round",
    alpha = 0.96
  ) +
  scale_colour_gradient2(
    low = clr("blue"),
    mid = pal["paper"],
    high = clr("red"),
    midpoint = 1,
    limits = c(0.5, 3.5),
    oob = scales::squish,
    labels = label_number(accuracy = 0.1),
    name = "Relative detection\nprior mean",
    guide = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(80, "mm"),
      barheight = unit(3.5, "mm")
    )
  ) +
  scale_x_date(
    date_breaks = "5 years",
    date_labels = "%Y",
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    x = NULL,
    y = NULL
  ) +
  coord_cartesian(xlim = c(as.Date("1990-01-01"), as.Date("2026-12-31"))) +
  theme_nature() +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text),
    axis.text.x = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    plot.margin = margin(4, 5, 4, 4)
  )

p_summary <- ggplot(summary, aes(y = country_panel)) +
  geom_segment(
    aes(x = prior_min, xend = prior_max, yend = country_panel),
    colour = pal["light_grey"],
    linewidth = 1.1,
    lineend = "round"
  ) +
  geom_segment(
    aes(x = prior_lower_median, xend = prior_upper_median, yend = country_panel),
    colour = pal["grey"],
    linewidth = 0.8,
    lineend = "round"
  ) +
  geom_point(
    aes(x = prior_median),
    shape = 21,
    fill = clr("blue"),
    colour = pal["black"],
    size = 2.3,
    stroke = 0.3
  ) +
  geom_vline(xintercept = 1, linetype = "22", linewidth = 0.28, colour = pal["grey"]) +
  scale_x_continuous(
    breaks = c(0.5, 0.75, 1, 1.5, 2, 3, 3.5),
    labels = label_number(accuracy = 0.1)
  ) +
  labs(
    x = "Relative detection prior mean",
    y = NULL
  ) +
  coord_cartesian(xlim = c(0.45, 3.55)) +
  theme_nature() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = figure_text$axis_text),
    plot.margin = margin(4, 5, 4, 4),
    legend.position = "none"
  )

figure <- panel_tags(
  (p_timeline | p_summary) +
    plot_layout(widths = c(1.35, 0.65), guides = "collect") &
    theme(legend.position = "bottom")
)

save_nature(figure, "supplement_reporting_practice_timeline.png", nature_full_width_mm, 110)
