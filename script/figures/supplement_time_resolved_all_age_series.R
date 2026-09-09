#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

summary <- read_csv_checked(
  out_path("time_resolved_all_age_series_summary.csv"),
  c(
    "iso3",
    "country",
    "pipeline_role",
    "time_resolution",
    "source_ids",
    "source_dataset",
    "source",
    "period_start",
    "period_end",
    "periods_observed",
    "total_cases",
    "peak_period_start",
    "peak_period_end",
    "peak_period_cases",
    "summary_note"
  )
) %>%
  mutate(
    period_start = as_plot_date(period_start),
    period_end = as_plot_date(period_end),
    country = forcats::fct_reorder(iso3, period_start, .desc = FALSE),
    pipeline_role = factor(
      pipeline_role,
      levels = c(
        "default_model2_main_analysis",
        "sensitivity_only_small_population",
        "auxiliary_afro_monthly_extension",
        "time_resolved_all_age_only"
      )
    ),
    time_resolution = factor(time_resolution, levels = c("weekly", "monthly"))
  ) %>%
  arrange(pipeline_role, period_start, country)

role_palette <- c(
  default_model2_main_analysis = clr("blue"),
  sensitivity_only_small_population = clr("orange"),
  auxiliary_afro_monthly_extension = clr("green"),
  time_resolved_all_age_only = clr("purple")
)

role_labels <- c(
  default_model2_main_analysis = "Default dynamic bridge",
  sensitivity_only_small_population = "Singapore sensitivity",
  auxiliary_afro_monthly_extension = "AFRO auxiliary extension",
  time_resolved_all_age_only = "Other time-resolved only"
)

resolution_linetypes <- c(
  weekly = "22",
  monthly = "solid"
)

p <- ggplot(summary, aes(y = country)) +
  geom_vline(
    xintercept = as.Date(c("2015-01-01", "2020-01-01", "2023-01-01")),
    linewidth = 0.28,
    linetype = "22",
    colour = pal["light_grey"]
  ) +
  geom_segment(
    aes(
      x = period_start,
      xend = period_end,
      yend = country,
      colour = pipeline_role,
      linetype = time_resolution
    ),
    linewidth = 0.95,
    lineend = "round",
    alpha = 0.92
  ) +
  geom_point(
    aes(x = period_start, colour = pipeline_role),
    size = 1.2,
    alpha = 0.9
  ) +
  geom_point(
    aes(x = period_end, colour = pipeline_role),
    size = 1.2,
    alpha = 0.9
  ) +
  scale_colour_manual(values = role_palette, labels = role_labels, name = "Pipeline role") +
  scale_linetype_manual(values = resolution_linetypes, name = "Time resolution") +
  scale_x_date(
    date_breaks = "3 years",
    date_labels = "%Y",
    limits = c(as.Date("2008-01-01"), as.Date("2026-12-31")),
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  labs(
    x = "Calendar year",
    y = NULL
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 1, b = 0),
    plot.margin = margin(5, 8, 8, 5)
  ) +
  coord_cartesian(clip = "off")

save_nature(p, "supplement_time_resolved_all_age_series.png", width_mm = 170, height_mm = 108)
