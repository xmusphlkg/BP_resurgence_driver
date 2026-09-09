suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(readr)
  library(scales)
})

figure_data_dir <- file.path("manuscript", "figure_data")
figure_dir <- file.path("manuscript", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

annual <- read_csv(
  file.path(figure_data_dir, "figure_1_global_trends.csv"),
  show_col_types = FALSE
)
country <- read_csv(
  file.path(figure_data_dir, "figure_1_resurgence_index_by_country.csv"),
  show_col_types = FALSE
)
regions <- read_csv(
  file.path("manuscript", "tables", "country_tier_assignment.csv"),
  show_col_types = FALSE
) |>
  distinct(iso3, region)

region_levels <- c("AFRO", "AMRO", "EMRO", "EURO", "SEARO", "WPRO")
region_colours <- c(
  AFRO = "#2C6DB2", AMRO = "#66B82E", EMRO = "#E8751A",
  EURO = "#D62735", SEARO = "#16A6A1", WPRO = "#F0A700"
)

theme_lancet <- theme_classic(base_size = 9, base_family = "sans") +
  theme(
    axis.title = element_text(colour = "#202124"),
    axis.text = element_text(colour = "#202124"),
    plot.title = element_blank(),
    plot.tag = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = 0.25)
  )

global <- annual |>
  filter(series_type == "global", who_region == "GLOBAL")

p_a <- ggplot(global, aes(year, incidence_per_100k)) +
  annotate("rect", xmin = 2020, xmax = 2022.99, ymin = -Inf, ymax = Inf,
           fill = "#F1F3F4") +
  geom_line(linewidth = 0.7, colour = "#202124") +
  geom_point(
    data = global |> filter(year >= 2020),
    colour = "#D62839", size = 1.8
  ) +
  scale_x_continuous(breaks = c(1980, 2000, 2020, 2025)) +
  labs(
    x = NULL,
    y = "Reported incidence per 100,000"
  ) +
  theme_lancet

region_ratio <- annual |>
  filter(series_type == "who_region", who_region %in% region_levels) |>
  mutate(period = case_when(
    year >= 2015 & year <= 2019 ~ "baseline",
    year >= 2023 & year <= 2025 ~ "post",
    TRUE ~ NA_character_
  )) |>
  filter(!is.na(period)) |>
  group_by(who_region, period) |>
  summarise(incidence = mean(incidence_per_100k, na.rm = TRUE), .groups = "drop") |>
  tidyr::pivot_wider(names_from = period, values_from = incidence) |>
  mutate(
    ratio = post / baseline,
    who_region = factor(who_region, levels = rev(region_levels))
  )

p_b <- ggplot(region_ratio, aes(ratio, who_region, colour = who_region)) +
  geom_vline(xintercept = 1, linetype = 2, colour = "#6F7275", linewidth = 0.45) +
  geom_segment(aes(x = 1, xend = ratio, yend = who_region), linewidth = 1.1,
               alpha = 0.45) +
  geom_point(size = 3) +
  geom_text(aes(label = number(ratio, accuracy = 0.1)), nudge_x = 0.08,
            hjust = 0, colour = "#303134", size = 2.7) +
  scale_colour_manual(values = region_colours, guide = "none") +
  scale_x_log10(breaks = c(0.2, 0.5, 1, 2, 5, 10), limits = c(0.15, 10)) +
  labs(
    x = "2023–25 / 2015–19 reported incidence",
    y = NULL
  ) +
  theme_lancet

label_iso3 <- c(
  "AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA",
  "QAT", "PYF", "HUN", "DNK"
)

country_plot <- country |>
  left_join(regions, by = "iso3") |>
  filter(
    is.finite(resurgence_ratio), resurgence_ratio > 0,
    is.finite(absolute_post_minus_baseline_incidence_per_100k),
    !is.na(latest_reported_cases)
  ) |>
  mutate(
    region_display = if_else(is.na(region), "Other", as.character(region)),
    region_display = factor(region_display, levels = c(region_levels, "Other")),
    label = if_else(iso3 %in% label_iso3, iso3, NA_character_)
  )

country_plot_display <- country_plot |>
  filter(resurgence_ratio >= 0.05, resurgence_ratio <= 120)

p_c <- ggplot(
  country_plot_display,
  aes(
    resurgence_ratio,
    absolute_post_minus_baseline_incidence_per_100k,
    size = latest_reported_cases,
    colour = region_display
  )
) +
  geom_vline(xintercept = 1, linetype = 2, colour = "#6F7275", linewidth = 0.4) +
  geom_hline(yintercept = 0, linetype = 2, colour = "#6F7275", linewidth = 0.4) +
  geom_point(alpha = 0.55, stroke = 0.2) +
  geom_text_repel(
    data = country_plot_display |> filter(!is.na(label), iso3 != "QAT"),
    aes(label = label),
    min.segment.length = 0,
    max.overlaps = Inf,
    box.padding = 0.25,
    point.padding = 0.15,
    size = 2.5,
    colour = "#202124",
    seed = 20260724,
    show.legend = FALSE
  ) +
  scale_colour_manual(
    values = c(region_colours, Other = "#9AA0A6"),
    drop = FALSE
  ) +
  scale_size_area(
    max_size = 7,
    breaks = c(100, 1000, 10000, 100000),
    labels = label_number(scale_cut = cut_short_scale())
  ) +
  scale_x_log10(
    breaks = c(0.1, 0.3, 1, 3, 10, 100, 1000),
    labels = label_number(),
    limits = c(0.05, 120)
  ) +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 1, base = 10),
    breaks = c(-100, -10, 0, 10, 100),
    labels = label_number()
  ) +
  labs(
    x = "2023–25 incidence / periodic pre-pandemic baseline",
    y = "Absolute excess reported incidence per 100,000",
    colour = NULL,
    size = NULL
  ) +
  guides(
    colour = guide_legend(order = 1, nrow = 1),
    size = guide_legend(order = 2, nrow = 1)
  ) +
  annotate(
    "text", x = 105, y = -4, label = "QAT >100",
    hjust = 1, vjust = -0.4, size = 2.5, colour = "#202124"
  ) +
  theme_lancet

figure <- (p_a + p_b + plot_layout(widths = c(1.05, 0.95))) /
  p_c +
  plot_layout(heights = c(0.85, 1.15)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.margin = margin(5, 7, 5, 5))

ggsave(
  file.path(figure_dir, "lancet_figure_1_notification_signals.png"),
  figure, width = 183, height = 155, units = "mm", dpi = 600, bg = "white"
)
ggsave(
  file.path(figure_dir, "lancet_figure_1_notification_signals.pdf"),
  figure, width = 183, height = 155, units = "mm", device = cairo_pdf,
  bg = "white"
)
