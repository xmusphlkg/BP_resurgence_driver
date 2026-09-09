#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

japan <- read_csv_checked(
  out_path("japan_annual_national_surveillance_summary.csv"),
  c(
    "country",
    "iso3",
    "year",
    "summary_status",
    "reported_case_count",
    "guideline_compliant_case_count",
    "guideline_compliant_percent",
    "age_peak_group",
    "age_peak_share_percent",
    "diagnostic_methods_summary",
    "diagnostic_top_method",
    "diagnostic_top_method_share_percent"
  )
) %>%
  mutate(
    year = as.integer(year),
    summary_status = factor(summary_status, levels = c("full_year", "partial_week_44")),
    reported_case_count = as.numeric(reported_case_count),
    guideline_compliant_case_count = as.numeric(guideline_compliant_case_count),
    guideline_compliant_percent = as.numeric(guideline_compliant_percent),
    age_peak_share_percent = as.numeric(age_peak_share_percent),
    diagnostic_top_method_share_percent = as.numeric(diagnostic_top_method_share_percent),
    year_label = as.character(year),
    status_label = if_else(summary_status == "partial_week_44", "2025 partial", "Full year")
  )

reported_long <- japan %>%
  select(year, summary_status, status_label, reported_case_count, guideline_compliant_case_count) %>%
  pivot_longer(
    cols = c(reported_case_count, guideline_compliant_case_count),
    names_to = "series",
    values_to = "count"
  ) %>%
  mutate(
    series = factor(
      series,
      levels = c("reported_case_count", "guideline_compliant_case_count"),
      labels = c("Reported cases", "Guideline-compliant cases")
    )
  )

series_palette <- c("Reported cases" = clr("black"), "Guideline-compliant cases" = clr("blue"))
count_limits <- range(reported_long$count[reported_long$count > 0], finite = TRUE)
shade_ymin <- max(1, count_limits[1] / 1.25)
shade_ymax <- count_limits[2] * 1.25

audit_metric_levels <- c(
  "Guideline\ncompliance",
  "Dominant\nage peak",
  "Top diagnostic\nmethod"
)

audit <- bind_rows(
  japan %>%
    transmute(
      year,
      summary_status,
      metric = "Guideline\ncompliance",
      share = guideline_compliant_percent,
      label = sprintf("%.0f%%", guideline_compliant_percent)
    ),
  japan %>%
    transmute(
      year,
      summary_status,
      metric = "Dominant\nage peak",
      share = age_peak_share_percent,
      label = str_replace_all(age_peak_group, " years", "y")
    ),
  japan %>%
    transmute(
      year,
      summary_status,
      metric = "Top diagnostic\nmethod",
      share = diagnostic_top_method_share_percent,
      label = case_when(
        str_detect(diagnostic_top_method, regex("immunochromatography", ignore_case = TRUE)) ~ "Immunochrom.",
        str_detect(diagnostic_top_method, regex("serology", ignore_case = TRUE)) ~ "Serology",
        TRUE ~ diagnostic_top_method
      )
    )
) %>%
  mutate(metric = factor(metric, levels = rev(audit_metric_levels)))

panel_a <- ggplot(reported_long, aes(x = year, y = count, colour = series, group = series)) +
  annotate(
    "rect",
    xmin = 2024.5,
    xmax = 2025.5,
    ymin = shade_ymin,
    ymax = shade_ymax,
    fill = pal["light_grey"],
    alpha = 0.24
  ) +
  geom_line(linewidth = 0.5) +
  geom_point(aes(fill = series, shape = status_label), size = 2.5, stroke = 0.35) +
  scale_colour_manual(values = series_palette, name = NULL) +
  scale_fill_manual(values = series_palette, name = NULL) +
  scale_shape_manual(values = c("Full year" = 21, "2025 partial" = 24), name = NULL) +
  scale_y_log10(labels = label_short, expand = expansion(mult = c(0.03, 0.08))) +
  scale_x_continuous(breaks = japan$year, minor_breaks = NULL) +
  labs(
    x = NULL,
    y = "Cases (log scale)"
  ) +
  theme_nature() +
  theme(
    legend.position = "bottom",
    panel.grid.major.y = element_line(linewidth = 0.16, colour = pal["light_grey"])
  )

panel_b <- ggplot(audit, aes(x = year, y = metric, fill = share)) +
  geom_tile(width = 0.86, height = 0.76, colour = pal["white"], linewidth = 0.35) +
  geom_tile(
    data = filter(audit, summary_status == "partial_week_44"),
    width = 0.86,
    height = 0.76,
    fill = NA,
    colour = pal["black"],
    linewidth = 0.45
  ) +
  geom_text(
    aes(label = label),
    size = figure_text$annotation_large,
    colour = pal["black"],
    lineheight = 0.9
  ) +
  scale_fill_gradient(
    low = pal["paper"],
    high = clr("blue"),
    limits = c(0, 100),
    labels = label_number(accuracy = 1),
    name = "Share (%)",
    guide = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(42, "mm"),
      barheight = unit(3.2, "mm")
    )
  ) +
  scale_x_continuous(breaks = japan$year, minor_breaks = NULL) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_nature() +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text),
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )

figure <- panel_tags(
  (panel_a | panel_b) +
    plot_layout(widths = c(1.18, 0.82), guides = "collect") &
    theme(
      legend.position = "bottom",
      plot.margin = margin(4, 4, 4, 4)
    )
)

save_nature(figure, "supplement_japan_annual_national_surveillance_summary.png", nature_full_width_mm, 92)
