#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

usa <- read_csv_checked(
  file.path(root, "data", "clean", "age_pattern_evidence.csv"),
  c(
    "iso3",
    "country",
    "evidence_family",
    "year",
    "reported_case_total",
    "infant_case_count",
    "infant_case_share_percent",
    "school_age_case_count",
    "school_age_case_share_percent",
    "adult_case_count",
    "adult_case_share_percent",
    "age_pattern_signature",
    "dominant_age_group",
    "dominant_model_age_group",
    "dominant_model_age_group_share_percent",
    "hospitalizations",
    "total_deaths",
    "infant_deaths",
    "source_ids",
    "notes"
  )
) %>%
  filter(iso3 == "USA") %>%
  mutate(
    year = as.integer(year),
    evidence_family = factor(
      evidence_family,
      levels = c("reported_case_age_distribution", "official_severe_outcome_age_signature")
    ),
    family_label = recode(
      as.character(evidence_family),
      reported_case_age_distribution = "Reported-case age distribution",
      official_severe_outcome_age_signature = "Severe-outcome age signature"
    )
  )

case_long <- usa %>%
  filter(evidence_family == "reported_case_age_distribution", year >= 2015) %>%
  select(
    year,
    reported_case_total,
    infant_case_share_percent,
    school_age_case_share_percent,
    adult_case_share_percent,
    age_pattern_signature
  ) %>%
  pivot_longer(
    cols = c(infant_case_share_percent, school_age_case_share_percent, adult_case_share_percent),
    names_to = "age_band",
    values_to = "share"
  ) %>%
  mutate(
    age_band = recode(
      age_band,
      infant_case_share_percent = "Infant",
      school_age_case_share_percent = "School age",
      adult_case_share_percent = "Adult"
    ),
    age_band = factor(age_band, levels = c("Infant", "School age", "Adult"))
  )

case_totals <- usa %>%
  filter(evidence_family == "reported_case_age_distribution", year >= 2015) %>%
  select(year, reported_case_total, age_pattern_signature)

severity_long <- usa %>%
  filter(evidence_family == "official_severe_outcome_age_signature", year >= 2003) %>%
  select(year, hospitalizations, total_deaths, infant_deaths) %>%
  pivot_longer(
    cols = c(hospitalizations, total_deaths, infant_deaths),
    names_to = "metric",
    values_to = "count"
  ) %>%
  filter(!is.na(count)) %>%
  mutate(
    metric = recode(
      metric,
      hospitalizations = "Hospitalization anchor",
      total_deaths = "Total deaths",
      infant_deaths = "Infant deaths"
    ),
    metric = factor(metric, levels = c("Hospitalization anchor", "Total deaths", "Infant deaths")),
    series_group = if_else(year == 2003 & metric == "Hospitalization anchor", "anchor", "annual")
  )

severity_series <- severity_long %>%
  filter(series_group == "annual")

severity_anchor <- severity_long %>%
  filter(series_group == "anchor")

age_palette <- c(
  Infant = clr("blue"),
  `School age` = clr("orange"),
  Adult = clr("purple")
)

severity_palette <- c(
  `Hospitalization anchor` = clr("orange"),
  `Total deaths` = clr("red"),
  `Infant deaths` = clr("purple")
)

p_age <- ggplot(case_long, aes(x = year, y = share, fill = age_band)) +
  geom_col(width = 0.74, colour = "white", linewidth = 0.28) +
  geom_text(
    data = case_totals,
    aes(x = year, y = 103.5, label = label_short(reported_case_total)),
    inherit.aes = FALSE,
    size = figure_text$annotation_large,
    colour = pal["dark_grey"]
  ) +
  scale_fill_manual(values = age_palette, name = NULL) +
  scale_y_continuous(
    limits = c(0, 110),
    breaks = figure_breaks$percent,
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.06))
  ) +
  labs(
    x = NULL,
    y = "Age share of annual reported cases"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    plot.margin = margin(4, 4, 8, 4)
  ) +
  coord_cartesian(clip = "off")

p_severity <- ggplot(severity_series, aes(x = year, y = count, colour = metric, group = metric)) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 2.0) +
  geom_point(
    data = severity_anchor,
    aes(x = year, y = count),
    inherit.aes = FALSE,
    colour = pal["grey"],
    fill = "white",
    shape = 21,
    size = 2.6,
    stroke = 0.35
  ) +
  facet_wrap(~metric, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = severity_palette, guide = "none") +
  scale_y_continuous(
    trans = scales::pseudo_log_trans(sigma = 1),
    labels = label_short,
    expand = expansion(mult = c(0.03, 0.18))
  ) +
  scale_x_continuous(
    breaks = c(2003, 2012, 2014, 2016, 2018, 2020, 2022, 2024),
    minor_breaks = NULL
  ) +
  labs(
    x = "Year",
    y = "Count"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    strip.text = element_text(size = figure_text$strip),
    legend.position = "bottom",
    plot.margin = margin(2, 4, 6, 4)
  )

figure <- panel_tags(
  (p_age | p_severity) +
    plot_layout(widths = c(1.08, 1.0))
)

save_nature(figure, "supplement_usa_annual_age_specific_surveillance_summary.png", nature_full_width_mm, 124)
