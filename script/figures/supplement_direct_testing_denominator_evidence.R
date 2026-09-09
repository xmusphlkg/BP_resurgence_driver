#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

series <- read_csv_checked(
  fd_path("direct_testing_denominator_evidence.csv"),
  c(
    "iso3",
    "country",
    "year",
    "testing_denominator",
    "calibration_testing_denominator",
    "calibration_boundary",
    "test_positivity_percent",
    "direct_testing_series_priority",
    "direct_testing_series_years_available",
    "direct_testing_series_first_year",
    "direct_testing_series_last_year",
    "direct_testing_series_support_level",
    "direct_testing_series_source_ids",
    "direct_testing_series_source_type",
    "direct_testing_series_relative_detection_prior_factor",
    "direct_testing_series_relative_detection_prior_factor_lower",
    "direct_testing_series_relative_detection_prior_factor_upper"
  )
) %>%
  mutate(
    year = as.integer(as.numeric(year)),
    testing_denominator = as.numeric(testing_denominator),
    test_positivity_percent = as.numeric(test_positivity_percent),
    direct_testing_series_relative_detection_prior_factor = as.numeric(
      direct_testing_series_relative_detection_prior_factor
    ),
    direct_testing_series_relative_detection_prior_factor_lower = as.numeric(
      direct_testing_series_relative_detection_prior_factor_lower
    ),
    direct_testing_series_relative_detection_prior_factor_upper = as.numeric(
      direct_testing_series_relative_detection_prior_factor_upper
    ),
    direct_testing_series_years_available = as.integer(as.numeric(direct_testing_series_years_available)),
    direct_testing_series_first_year = as.integer(as.numeric(direct_testing_series_first_year)),
    direct_testing_series_last_year = as.integer(as.numeric(direct_testing_series_last_year)),
    model_window = year >= 2015 & year <= 2025
  )

summary <- read_csv_checked(
  out_path("direct_testing_denominator_evidence_summary.csv"),
  c(
    "iso3",
    "country",
    "direct_testing_series_rows",
    "direct_testing_series_years_available",
    "direct_testing_series_model_window_rows",
    "direct_testing_series_model_window_years_available",
    "direct_testing_series_first_year",
    "direct_testing_series_last_year",
    "direct_testing_series_model_window_first_year",
    "direct_testing_series_model_window_last_year",
    "direct_testing_series_support_level",
    "direct_testing_series_support_class",
    "direct_testing_series_support_rank",
    "direct_testing_series_source_ids",
    "direct_testing_series_source_type",
    "direct_testing_series_testing_denominator_median",
    "direct_testing_series_testing_denominator_min",
    "direct_testing_series_testing_denominator_max",
    "direct_testing_series_relative_detection_prior_factor_median",
    "direct_testing_series_relative_detection_prior_factor_min",
    "direct_testing_series_relative_detection_prior_factor_max",
    "direct_testing_series_test_positivity_percent_median",
    "direct_testing_series_test_positivity_percent_min",
    "direct_testing_series_test_positivity_percent_max"
  )
) %>%
  mutate(
    direct_testing_series_rows = as.integer(as.numeric(direct_testing_series_rows)),
    direct_testing_series_years_available = as.integer(as.numeric(direct_testing_series_years_available)),
    direct_testing_series_model_window_rows = as.integer(as.numeric(direct_testing_series_model_window_rows)),
    direct_testing_series_model_window_years_available = as.integer(
      as.numeric(direct_testing_series_model_window_years_available)
    ),
    direct_testing_series_first_year = as.integer(as.numeric(direct_testing_series_first_year)),
    direct_testing_series_last_year = as.integer(as.numeric(direct_testing_series_last_year)),
    direct_testing_series_model_window_first_year = as.integer(
      as.numeric(direct_testing_series_model_window_first_year)
    ),
    direct_testing_series_model_window_last_year = as.integer(
      as.numeric(direct_testing_series_model_window_last_year)
    ),
    direct_testing_series_support_rank = as.integer(as.numeric(direct_testing_series_support_rank)),
    direct_testing_series_testing_denominator_median = as.numeric(direct_testing_series_testing_denominator_median),
    direct_testing_series_testing_denominator_min = as.numeric(direct_testing_series_testing_denominator_min),
    direct_testing_series_testing_denominator_max = as.numeric(direct_testing_series_testing_denominator_max),
    direct_testing_series_relative_detection_prior_factor_median = as.numeric(
      direct_testing_series_relative_detection_prior_factor_median
    ),
    direct_testing_series_relative_detection_prior_factor_min = as.numeric(
      direct_testing_series_relative_detection_prior_factor_min
    ),
    direct_testing_series_relative_detection_prior_factor_max = as.numeric(
      direct_testing_series_relative_detection_prior_factor_max
    ),
    direct_testing_series_test_positivity_percent_median = as.numeric(
      direct_testing_series_test_positivity_percent_median
    ),
    direct_testing_series_test_positivity_percent_min = as.numeric(
      direct_testing_series_test_positivity_percent_min
    ),
    direct_testing_series_test_positivity_percent_max = as.numeric(
      direct_testing_series_test_positivity_percent_max
    )
  )

support_palette <- c(
  multi_year_model_window_series = clr("blue"),
  multi_year_historical_only_series = clr("grey"),
  single_year_model_window_summary = clr("orange"),
  single_year_historical_summary = clr("purple")
)

support_labels <- c(
  multi_year_model_window_series = "multi-year model-window",
  multi_year_historical_only_series = "multi-year historical only",
  single_year_model_window_summary = "single-year model-window",
  single_year_historical_summary = "single-year historical"
)

country_levels <- summary %>%
  pull(iso3) %>%
  as.character()

summary <- summary %>%
  mutate(
    country_factor = factor(iso3, levels = country_levels),
    support_class = factor(direct_testing_series_support_class, levels = names(support_palette))
  )

series <- series %>%
  mutate(
    country_factor = factor(iso3, levels = country_levels)
  )

year_min <- min(series$year, na.rm = TRUE)
# This supplement describes the archived calibration prior. The empirical
# positive/eligible count correction did not recompute these prior factors.
factor_range <- range(series$direct_testing_series_relative_detection_prior_factor)
stopifnot(all(is.finite(factor_range)), factor_range[1] > 0)
factor_limits <- c(floor(factor_range[1] * 10) / 10, ceiling(factor_range[2] * 10) / 10)

p_heatmap <- ggplot(series, aes(x = year, y = country_factor)) +
  annotate(
    "rect",
    xmin = 2014.5,
    xmax = 2025.5,
    ymin = -Inf,
    ymax = Inf,
    fill = pal["light_grey"],
    alpha = 0.20
  ) +
  geom_tile(
    aes(fill = direct_testing_series_relative_detection_prior_factor),
    width = 0.9,
    height = 0.82,
    colour = pal["white"],
    linewidth = 0.18
  ) +
  scale_fill_gradient2(
    low = clr("blue"),
    mid = pal["paper"],
    high = clr("red"),
    midpoint = 1,
    limits = factor_limits,
    labels = label_number(accuracy = 0.1),
    name = "Archived relative detection\nprior factor",
    guide = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(80, "mm"),
      barheight = unit(3.5, "mm")
    )
  ) +
  scale_x_continuous(
    breaks = figure_breaks_year(floor(year_min), 2025, 2),
    labels = label_number(accuracy = 1, big.mark = ""),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    x = NULL,
    y = NULL
  ) +
  coord_cartesian(xlim = c(floor(year_min) - 0.5, 2025.5)) +
  theme_nature() +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text),
    axis.text.x = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    plot.margin = margin(4, 5, 4, 4)
  )

p_summary <- ggplot(summary, aes(y = country_factor)) +
  geom_segment(
    aes(x = direct_testing_series_relative_detection_prior_factor_min, xend = direct_testing_series_relative_detection_prior_factor_max, yend = country_factor),
    colour = pal["light_grey"],
    linewidth = 1.1,
    lineend = "round"
  ) +
  geom_segment(
    aes(
      x = direct_testing_series_relative_detection_prior_factor_median,
      xend = direct_testing_series_relative_detection_prior_factor_median,
      yend = country_factor
    ),
    colour = pal["dark_grey"],
    linewidth = 0.35,
    lineend = "round"
  ) +
  geom_point(
    aes(x = direct_testing_series_relative_detection_prior_factor_median, fill = support_class),
    shape = 21,
    colour = pal["black"],
    size = 2.3,
    stroke = 0.3
  ) +
  geom_vline(xintercept = 1, linetype = "22", linewidth = 0.28, colour = pal["grey"]) +
  scale_fill_manual(
    values = support_palette,
    breaks = names(support_palette),
    labels = support_labels,
    name = "Support class",
    drop = FALSE,
    guide = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  scale_x_continuous(
    breaks = pretty(factor_limits, n = 4),
    labels = label_number(accuracy = 0.1)
  ) +
  labs(
    x = "Archived detection prior factor",
    y = NULL
  ) +
  coord_cartesian(xlim = factor_limits) +
  theme_nature() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = figure_text$axis_text),
    plot.margin = margin(4, 5, 4, 4),
    legend.position = "bottom"
  )

figure <- panel_tags(
  (p_heatmap | p_summary) +
    plot_layout(widths = c(1.28, 0.72), guides = "collect") &
    theme(legend.position = "bottom")
)

save_nature(figure, "supplement_direct_testing_denominator_evidence.png", nature_full_width_mm, 150)
