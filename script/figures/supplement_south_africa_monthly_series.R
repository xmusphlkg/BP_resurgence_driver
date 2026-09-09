#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

monthly <- read_csv_checked(
  file.path(root, "data", "clean", "cases_country_age_time.csv"),
  c(
    "iso3",
    "country",
    "date_start",
    "date_end",
    "year",
    "month",
    "time_resolution",
    "age_group",
    "cases_total"
  )
) %>%
  filter(
    iso3 == "ZAF",
    time_resolution == "monthly",
    age_group == "all_ages"
  ) %>%
  mutate(
    date_start = as_plot_date(date_start),
    date_end = as_plot_date(date_end),
    year = as.integer(year),
    cases_total = as.numeric(cases_total)
  ) %>%
  arrange(date_start)

annual <- read_csv_checked(
  out_path("south_africa_monthly_series_summary.csv"),
  c(
    "scope",
    "year",
    "total_cases",
    "share_of_series_cases_percent"
  )
) %>%
  filter(scope == "year") %>%
  mutate(
    year = factor(year, levels = c(2023, 2024, 2025))
  )

peak <- monthly %>%
  slice_max(cases_total, n = 1, with_ties = FALSE) %>%
  mutate(label = sprintf("Peak %s\n%s cases", format(date_start, "%Y-%m"), label_short(cases_total)))

annual_palette <- c(
  "2023" = clr("blue"),
  "2024" = clr("orange"),
  "2025" = clr("green")
)

p_monthly <- ggplot(monthly, aes(x = date_start, y = cases_total)) +
  geom_vline(
    xintercept = as.Date(c("2024-01-01", "2025-01-01")),
    linewidth = 0.3,
    linetype = "22",
    colour = pal["light_grey"]
  ) +
  geom_line(linewidth = 0.46, colour = clr("blue")) +
  geom_point(size = 1.05, colour = pal["black"], alpha = 0.82) +
  geom_point(data = peak, aes(x = date_start, y = cases_total), inherit.aes = FALSE, size = 1.45, colour = clr("red")) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b\n%Y",
    limits = c(as.Date("2023-02-01"), as.Date("2025-09-30")),
    expand = expansion(mult = c(0.01, 0.04))
  ) +
  scale_y_continuous(
    labels = label_short,
    expand = expansion(mult = c(0.01, 0.08))
  ) +
  labs(
    x = NULL,
    y = "Monthly cases"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    plot.margin = margin(5, 8, 8, 5)
  ) +
  coord_cartesian()

p_annual <- ggplot(annual, aes(x = year, y = total_cases, fill = year)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.25) +
  scale_fill_manual(values = annual_palette, guide = "none") +
  scale_y_continuous(
    labels = label_short,
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    x = NULL,
    y = "Annual cases"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    plot.margin = margin(5, 8, 6, 5)
  )

figure <- panel_tags(
  (p_monthly | p_annual) + plot_layout(widths = c(1.35, 0.65))
)

save_nature(figure, "supplement_south_africa_monthly_series.png", nature_full_width_mm, 88)
