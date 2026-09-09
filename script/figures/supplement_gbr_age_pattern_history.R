#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

history <- read_csv_checked(
  fd_path("gbr_age_pattern_history_summary.csv"),
  c(
    "year",
    "source_segment",
    "reported_case_total",
    "under1_share_percent",
    "child_1_4_share_percent",
    "school_age_5_17_share_percent",
    "adult_18plus_share_percent",
    "dominant_broad_age_band",
    "dominant_broad_age_band_share_percent",
    "age_pattern_signature"
  )
) %>%
  mutate(
    year = as.integer(year),
    source_segment = factor(
      source_segment,
      levels = c("official_summary_extension", "workbook_age_coverage"),
      labels = c("Official summary\n1994-2014", "Workbook coverage\n2015-2024")
    )
  )

band_palette <- c(
  under1 = clr("blue"),
  child_1_4y = clr("sky"),
  school_age_5_17y = clr("orange"),
  adult_18plus = clr("purple")
)

band_labels <- c(
  under1 = "Under 1 year",
  child_1_4y = "1-4 years",
  school_age_5_17y = "5-17 years",
  adult_18plus = "18+ years"
)

history_long <- history %>%
  select(
    year,
    source_segment,
    reported_case_total,
    under1_share_percent,
    child_1_4_share_percent,
    school_age_5_17_share_percent,
    adult_18plus_share_percent
  ) %>%
  pivot_longer(
    cols = c(
      under1_share_percent,
      child_1_4_share_percent,
      school_age_5_17_share_percent,
      adult_18plus_share_percent
    ),
    names_to = "band",
    values_to = "share_percent"
  ) %>%
  mutate(
    band = recode(
      band,
      under1_share_percent = "under1",
      child_1_4_share_percent = "child_1_4y",
      school_age_5_17_share_percent = "school_age_5_17y",
      adult_18plus_share_percent = "adult_18plus"
    ),
    band = factor(band, levels = c("adult_18plus", "school_age_5_17y", "child_1_4y", "under1"))
  )

p_history <- ggplot(history_long, aes(x = year, y = share_percent, fill = band)) +
  annotate(
    "rect",
    xmin = 1993.5,
    xmax = 2014.5,
    ymin = -Inf,
    ymax = Inf,
    fill = clr("blue"),
    alpha = 0.04
  ) +
  annotate(
    "rect",
    xmin = 2014.5,
    xmax = 2024.5,
    ymin = -Inf,
    ymax = Inf,
    fill = clr("orange"),
    alpha = 0.03
  ) +
  geom_col(width = 0.88, colour = "white", linewidth = 0.18) +
  geom_vline(xintercept = 2014.5, linewidth = 0.35, linetype = "22", colour = pal["grey"]) +
  scale_fill_manual(
    values = band_palette,
    breaks = names(band_labels),
    labels = unname(band_labels),
    name = "Age band"
  ) +
  scale_x_continuous(
    breaks = figure_breaks_year(1994, 2024, 4),
    limits = c(1993.45, 2024.55),
    expand = expansion(mult = c(0.005, 0.01))
  ) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = figure_breaks$percent,
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    x = "Year",
    y = "Share of reported cases"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    plot.margin = margin(5, 6, 10, 5)
  ) +
  coord_cartesian(clip = "off")

figure <- panel_tags(p_history)

save_nature(figure, "supplement_gbr_age_pattern_history.png", nature_full_width_mm, 114)
