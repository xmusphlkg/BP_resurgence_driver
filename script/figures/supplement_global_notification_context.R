#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

if (!requireNamespace("ggrepel", quietly = TRUE)) {
  stop("Missing R package: ggrepel", call. = FALSE)
}

trends <- read_csv_checked(
  fd_path("figure_1_global_trends.csv"),
  c("series_type", "who_region", "year", "reported_cases", "incidence_per_100k")
) %>%
  mutate(year = as.integer(year))

resurgence <- read_csv_checked(
  fd_path("figure_1_resurgence_index_by_country.csv"),
  c(
    "iso3", "country", "baseline_mean_reported_incidence_per_100k",
    "periodic_expected_postpandemic_incidence_per_100k",
    "postpandemic_mean_reported_incidence_per_100k", "post_to_baseline_rr",
    "absolute_post_minus_baseline_incidence_per_100k",
    "excess_cases_per_year", "latest_case_year", "latest_reported_cases"
  )
) %>%
  mutate(
    baseline_mean_reported_incidence_per_100k =
      as.numeric(baseline_mean_reported_incidence_per_100k),
    periodic_expected_postpandemic_incidence_per_100k =
      as.numeric(periodic_expected_postpandemic_incidence_per_100k),
    postpandemic_mean_reported_incidence_per_100k =
      as.numeric(postpandemic_mean_reported_incidence_per_100k),
    post_to_baseline_rr = pmax(as.numeric(post_to_baseline_rr), 0.01),
    latest_reported_cases = as.numeric(latest_reported_cases),
    excess_cases_per_year = as.numeric(excess_cases_per_year),
    absolute_post_minus_baseline_incidence_per_100k =
      as.numeric(absolute_post_minus_baseline_incidence_per_100k)
  )

baseline_robustness <- read_csv_checked(
  file.path(
    outputs_dir,
    "global_baseline_robustness",
    "country_baseline_robustness.csv"
  ),
  c(
    "iso3", "requested_data_cutoff", "provisional_year_policy",
    "periodic_resurgence_ratio", "periodic_resurgence_ratio_lcl",
    "periodic_resurgence_ratio_ucl",
    "periodic_resurgence_ratio_ucl_unbounded",
    "periodic_bootstrap_successes", "periodic_bootstrap_status"
  )
) %>%
  filter(
    requested_data_cutoff == 2025,
    provisional_year_policy == "include"
  ) %>%
  transmute(
    iso3,
    robustness_resurgence_ratio = as.numeric(periodic_resurgence_ratio),
    resurgence_ratio_lower_95 = as.numeric(periodic_resurgence_ratio_lcl),
    resurgence_ratio_upper_95 = as.numeric(periodic_resurgence_ratio_ucl),
    resurgence_ratio_upper_unbounded = as.logical(periodic_resurgence_ratio_ucl_unbounded),
    periodic_bootstrap_successes = as.integer(periodic_bootstrap_successes),
    periodic_bootstrap_status
  )

metadata <- read_csv_checked(
  file.path(root, "data", "clean", "country_metadata.csv"),
  c("iso3", "who_region")
) %>%
  transmute(iso3, who_region_metadata = who_region)

selection <- read_csv_checked(
  out_path("model2c_country_selection_audit.csv"),
  c(
    "iso3", "who_region", "selected_for_current_model2c",
    "eligible_with_current_local_high_resolution_data",
    "excluded_from_default_model2",
    "weak_resurgence_or_negative_control_candidate"
  )
) %>%
  transmute(
    iso3,
    who_region_selection = who_region,
    selected_for_current_model2 = selected_for_current_model2c,
    eligible_with_current_local_high_resolution_data,
    excluded_from_default_model2,
    weak_resurgence_or_negative_control_candidate
  )

annual_year_limits <- c(1980, 2025)
annual_year_breaks <- figure_breaks_year(1980, 2020, 10)
region_year_breaks <- figure_breaks_year(1980, 2020, 20)
global_y_breaks <- seq(0, 50, 10)
regional_y_breaks <- c(0, 3, 10, 30, 100)
excess_y_breaks <- c(-100, -30, -10, -3, 0, 3, 10, 30, 100)
resurgence_y_breaks <- c(0.3, 1, 3, 10, 30, 100)

region_labels <- c(
  WPRO = "Western Pacific",
  EURO = "Europe",
  EMRO = "Eastern Mediterranean",
  AMRO = "Americas",
  AFRO = "Africa",
  SEARO = "South-East Asia"
)

global <- trends %>%
  filter(series_type == "global", year >= annual_year_limits[[1]]) %>%
  arrange(year)

region <- trends %>%
  filter(series_type != "global", year >= annual_year_limits[[1]], !is.na(who_region)) %>%
  mutate(who_region = factor(str_replace(who_region, "^WHO_", ""), levels = names(region_palette))) %>%
  arrange(year)

landscape <- resurgence %>%
  left_join(metadata, by = "iso3") %>%
  left_join(selection, by = "iso3") %>%
  filter(is.finite(post_to_baseline_rr), latest_reported_cases > 0) %>%
  mutate(
    who_region = str_replace(coalesce(who_region_metadata, who_region_selection, "Not classified"), "^WHO_", ""),
    who_region = factor(who_region, levels = names(region_palette)),
    selected_for_current_model2 = coalesce(selected_for_current_model2, FALSE),
    eligible_with_current_local_high_resolution_data =
      coalesce(eligible_with_current_local_high_resolution_data, FALSE),
    excluded_from_default_model2 = coalesce(excluded_from_default_model2, FALSE),
    focus_country = selected_for_current_model2 |
      eligible_with_current_local_high_resolution_data |
      excluded_from_default_model2,
    weak_resurgence_or_negative_control_candidate =
      coalesce(weak_resurgence_or_negative_control_candidate, FALSE),
    evidence_layer = case_when(
      selected_for_current_model2 ~ "Dynamic bridge pilot",
      eligible_with_current_local_high_resolution_data ~ "High-resolution source not default",
      TRUE ~ "Global descriptive layer"
    ),
    focus_alpha_class = if_else(focus_country, "Focus country", "Other country"),
    country_label = country_display_name(country),
    post_to_baseline_rr_display = pmin(post_to_baseline_rr, 120),
    country_scatter_label = if_else(
      post_to_baseline_rr > 120,
      paste0(str_wrap(country_label, width = 14), " >120x"),
      str_wrap(country_label, width = 14)
    )
  )

landscape_labels <- bind_rows(
  landscape %>%
    filter(
      focus_country,
      post_to_baseline_rr >= 3 |
        abs(absolute_post_minus_baseline_incidence_per_100k) >= 3
    ),
  landscape %>% slice_max(post_to_baseline_rr, n = 4, with_ties = FALSE),
  landscape %>% slice_max(abs(absolute_post_minus_baseline_incidence_per_100k), n = 4, with_ties = FALSE)
) %>%
  distinct(iso3, .keep_all = TRUE)

pilot_landscape <- landscape %>%
  filter(iso3 %in% pilot_iso3) %>%
  left_join(pilot_order, by = "iso3") %>%
  left_join(baseline_robustness, by = "iso3") %>%
  arrange(pilot_rank) %>%
  mutate(
    country_label = factor(country_display_name(country), levels = rev(country_display_name(country))),
    rr_band = case_when(
      post_to_baseline_rr >= 3 ~ "High resurgence",
      post_to_baseline_rr >= 1.25 ~ "Elevated",
      post_to_baseline_rr >= 0.75 ~ "Near baseline",
      TRUE ~ "Below baseline"
    ),
    rr_band = factor(rr_band, levels = c("High resurgence", "Elevated", "Near baseline", "Below baseline")),
    label = paste0(
      "RR ", label_number(accuracy = 0.1)(post_to_baseline_rr),
      "; ", label_short(latest_reported_cases), " cases"
    ),
    label_x = if_else(post_to_baseline_rr < 1, 1, post_to_baseline_rr)
  )

if (
  any(!is.finite(pilot_landscape$resurgence_ratio_lower_95)) ||
    any(pilot_landscape$periodic_bootstrap_successes != 2000L) ||
    any(
      abs(
        pilot_landscape$post_to_baseline_rr -
          pilot_landscape$robustness_resurgence_ratio
      ) > 1e-10,
      na.rm = TRUE
    )
) {
  stop(
    "Supplementary global context intervals do not match the primary resurgence point estimates.",
    call. = FALSE
  )
}

pilot_interval_cap <- max(
  32,
  pilot_landscape$resurgence_ratio_upper_95[
    is.finite(pilot_landscape$resurgence_ratio_upper_95)
  ],
  na.rm = TRUE
)

pilot_landscape <- pilot_landscape %>%
  mutate(
    resurgence_ratio_upper_plot = if_else(
      resurgence_ratio_upper_unbounded,
      pilot_interval_cap,
      resurgence_ratio_upper_95
    )
  )

region_summary <- region %>%
  group_by(who_region) %>%
  summarise(
    pre_2015_2019 = mean(incidence_per_100k[year %in% 2015:2019], na.rm = TRUE),
    post_2023_2025 = mean(incidence_per_100k[year %in% 2023:2025], na.rm = TRUE),
    latest_2025 = first(incidence_per_100k[year == 2025], default = NA_real_),
    .groups = "drop"
  ) %>%
  mutate(
    region_ratio = (post_2023_2025 + 0.01) / (pre_2015_2019 + 0.01),
    absolute_change = post_2023_2025 - pre_2015_2019,
    ratio_label = label_number(accuracy = 0.1)(region_ratio),
    who_region = factor(who_region, levels = names(region_palette))
  )

region_ratio_limits <- c(
  min(0.2, min(region_summary$region_ratio, na.rm = TRUE) * 0.82),
  max(region_summary$region_ratio, na.rm = TRUE) * 1.55
)

baseline_diag <- landscape %>%
  filter(
    is.finite(periodic_expected_postpandemic_incidence_per_100k),
    periodic_expected_postpandemic_incidence_per_100k > 0,
    is.finite(post_to_baseline_rr)
  ) %>%
  mutate(
    expected_baseline_for_plot = pmax(periodic_expected_postpandemic_incidence_per_100k, 0.001),
    low_baseline_class = case_when(
      expected_baseline_for_plot < 0.1 & post_to_baseline_rr >= 10 ~ "Low baseline, high RR",
      selected_for_current_model2 ~ "Dynamic pilot",
      TRUE ~ "Other"
    ),
    low_baseline_class = factor(
      low_baseline_class,
      levels = c("Other", "Low baseline, high RR", "Dynamic pilot")
    ),
    baseline_alpha_class = case_when(
      focus_country ~ "Focus country",
      low_baseline_class == "Low baseline, high RR" ~ "Low-baseline outlier",
      TRUE ~ "Other country"
    ),
    baseline_alpha_class = factor(
      baseline_alpha_class,
      levels = c("Other country", "Low-baseline outlier", "Focus country")
    ),
    resurgence_ratio_for_plot = pmin(post_to_baseline_rr, 120),
    resurgence_ratio_capped = post_to_baseline_rr > 120
  )

baseline_labels <- bind_rows(
  baseline_diag %>% filter(low_baseline_class == "Low baseline, high RR") %>% slice_max(post_to_baseline_rr, n = 7, with_ties = FALSE),
  baseline_diag %>% filter(focus_country)
) %>%
  distinct(iso3, .keep_all = TRUE) %>%
  mutate(
    label = if_else(
      resurgence_ratio_capped,
      paste0(str_wrap(country_display_name(country), width = 14), " >100"),
      str_wrap(country_display_name(country), width = 14)
    )
  )

# Annual observations are plotted at integer year centres; half-year bounds shade
# full calendar years 2020-2022, matching the Figure 2 coverage panel.
pandemic_reference_band <- tibble(
  xmin = 2019.5,
  xmax = 2022.5,
  ymin = -Inf,
  ymax = Inf
)

scale_colour_who_region <- function(show_legend = TRUE) {
  legend_guide <- if (show_legend) {
    guide_legend(
      nrow = 1,
      override.aes = list(alpha = 1, linewidth = 0.8)
    )
  } else {
    "none"
  }

  scale_colour_manual(
    values = region_palette,
    breaks = names(region_palette),
    labels = region_labels,
    name = "WHO region",
    drop = FALSE,
    guide = legend_guide
  )
}

p_global <- ggplot(global, aes(year, incidence_per_100k)) +
  geom_rect(
    data = pandemic_reference_band,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = clr("paper"),
    alpha = 0.95
  ) +
  geom_line(colour = pal["black"], linewidth = 0.55) +
  geom_point(data = filter(global, year >= 2020), colour = pal["red"], size = 1.2) +
  scale_y_continuous(breaks = global_y_breaks, labels = label_number(accuracy = 1)) +
  scale_x_continuous(limits = annual_year_limits, breaks = region_year_breaks, expand = expansion(mult = c(0.01, 0.02))) +
  labs(
    x = NULL,
    y = "Global reported incidence\nper 100,000"
  ) +
  theme_manuscript()

p_region <- ggplot(region, aes(year, incidence_per_100k, colour = who_region, group = who_region)) +
  geom_rect(
    data = pandemic_reference_band,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = clr("paper"),
    alpha = 0.95
  ) +
  geom_line(linewidth = 0.45) +
  facet_wrap(
    ~who_region,
    ncol = 2,
    labeller = as_labeller(region_labels)
  ) +
  scale_colour_who_region(show_legend = FALSE) +
  scale_y_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = regional_y_breaks,
    labels = label_number(accuracy = 1)
  ) +
  scale_x_continuous(limits = annual_year_limits, breaks = annual_year_breaks, expand = expansion(mult = c(0.01, 0.02))) +
  labs(
    x = NULL,
    y = "Regional incidence per 100,000"
  ) +
  theme_manuscript()

p_region_ratio <- ggplot(
  region_summary,
  aes(region_ratio, fct_reorder(who_region, region_ratio), colour = who_region)
) +
  geom_vline(xintercept = 1, colour = pal["grey"], linewidth = 0.28, linetype = "22") +
  geom_segment(aes(x = 1, xend = region_ratio, yend = fct_reorder(who_region, region_ratio)), linewidth = 0.62, alpha = 0.45) +
  geom_point(aes(size = abs(absolute_change)), alpha = 0.92) +
  geom_text(aes(label = ratio_label), nudge_x = 0.04, hjust = 0, size = figure_text$annotation, colour = pal["dark_grey"]) +
  scale_colour_who_region(show_legend = FALSE) +
  scale_y_discrete(labels = region_labels) +
  scale_size_area(max_size = 3.2, guide = "none") +
  scale_x_log10(
    breaks = figure_breaks$ratio_4x,
    labels = label_number(accuracy = 0.1),
    limits = region_ratio_limits,
    expand = expansion(mult = c(0.02, 0.16))
  ) +
  labs(
    x = "2023-2025 / 2015-2019 regional incidence",
    y = NULL
  ) +
  theme_manuscript()

p_landscape <- ggplot(
  landscape,
  aes(post_to_baseline_rr_display, absolute_post_minus_baseline_incidence_per_100k)
) +
  geom_hline(yintercept = 0, colour = pal["grey"], linewidth = 0.30, linetype = "22") +
  geom_vline(xintercept = 1, colour = pal["grey"], linewidth = 0.30, linetype = "22") +
  geom_point(
    aes(size = latest_reported_cases, colour = who_region, alpha = focus_alpha_class),
    stroke = 0.15
  ) +
  ggrepel::geom_text_repel(
    data = landscape_labels,
    aes(label = country_scatter_label),
    size = figure_text$annotation_dense,
    colour = pal["black"],
    seed = 20260729,
    box.padding = 0.35,
    point.padding = 0.16,
    force = 2,
    force_pull = 0.25,
    max.time = 2,
    max.iter = 20000,
    min.segment.length = 0,
    segment.colour = pal["grey"],
    segment.size = 0.18,
    max.overlaps = Inf
  ) +
  scale_x_log10(
    labels = c("0.01", "0.1", "1", "10", "100"),
    breaks = c(0.01, 0.1, 1, 10, 100),
    expand = expansion(mult = c(0.03, 0.03))
  ) +
  scale_y_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    labels = label_number(accuracy = 1),
    breaks = excess_y_breaks,
    expand = expansion(mult = c(0.04, 0.14))
  ) +
  coord_cartesian(
    xlim = c(NA, 125),
    clip = "off"
  ) +
  scale_size_area(
    max_size = 4.2,
    guide = "none"
  ) +
  scale_colour_who_region(show_legend = FALSE) +
  scale_alpha_manual(
    values = c(
      "Other country" = 0.18,
      "Focus country" = 1
    ),
    guide = "none"
  ) +
  labs(
    x = "2023-2025 incidence / periodic 2010-2019 baseline",
    y = "Absolute excess incidence\nper 100,000"
  ) +
  theme_manuscript() +
  theme(
    legend.position = "none",
    plot.margin = margin(6, 12, 4, 5)
  )

p_low_baseline <- ggplot(
  baseline_diag,
  aes(expected_baseline_for_plot, resurgence_ratio_for_plot)
) +
  geom_hline(yintercept = 1, colour = pal["grey"], linewidth = 0.28, linetype = "22") +
  geom_hline(yintercept = 10, colour = pal["light_grey"], linewidth = 0.24, linetype = "22") +
  geom_vline(xintercept = 0.1, colour = pal["light_grey"], linewidth = 0.24, linetype = "22") +
  geom_point(
    aes(size = latest_reported_cases, colour = who_region, alpha = baseline_alpha_class),
    stroke = 0.12
  ) +
  ggrepel::geom_text_repel(
    data = baseline_labels,
    aes(label = label),
    size = figure_text$annotation_dense,
    colour = pal["black"],
    seed = 20260729,
    box.padding = 0.32,
    point.padding = 0.14,
    force = 1.6,
    force_pull = 0.35,
    max.time = 2,
    max.iter = 20000,
    min.segment.length = 0,
    segment.colour = pal["grey"],
    segment.size = 0.16,
    max.overlaps = Inf
  ) +
  scale_x_log10(
    breaks = c(0.001, 0.01, 0.1, 1, 10, 100),
    labels = c("0.001", "0.01", "0.1", "1", "10", "100"),
    expand = expansion(mult = c(0.02, 0.03))
  ) +
  scale_y_log10(
    breaks = resurgence_y_breaks,
    labels = c("0.3", "1", "3", "10", "30", "100"),
    expand = expansion(mult = c(0.03, 0.07))
  ) +
  coord_cartesian(
    xlim = c(0.001, 100),
    ylim = c(0.2, 150),
    clip = "off"
  ) +
  scale_size_area(
    max_size = 3.8,
    breaks = c(100, 1000, 5000),
    labels = label_short,
    name = "Latest cases",
    guide = guide_legend(
      direction = "horizontal",
      title.position = "top",
      label.position = "bottom",
      override.aes = list(alpha = 0.55, colour = pal["dark_grey"])
    )
  ) +
  scale_colour_who_region(show_legend = FALSE) +
  scale_alpha_manual(
    values = c(
      "Other country" = 0.16,
      "Low-baseline outlier" = 0.78,
      "Focus country" = 1
    ),
    guide = "none"
  ) +
  labs(
    x = "Expected baseline incidence per 100,000",
    y = "Resurgence ratio"
  ) +
  theme_manuscript() +
  theme(
    axis.text = element_text(size = figure_text$axis_text_dense),
    legend.position = "inside",
    legend.position.inside = c(0.97, 0.90),
    legend.justification.inside = c(1, 1),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = scales::alpha(clr("white"), 0.86), colour = NA),
    legend.title = element_text(size = figure_text$legend_title, face = "bold"),
    legend.text = element_text(size = figure_text$legend_text_dense),
    legend.key.size = unit(2.5, "mm"),
    legend.margin = margin(1, 1, 1, 1)
  )

p_pilot <- ggplot(pilot_landscape, aes(post_to_baseline_rr, country_label, colour = who_region)) +
  geom_vline(xintercept = 1, colour = pal["grey"], linewidth = 0.30, linetype = "22") +
  geom_segment(
    aes(
      x = resurgence_ratio_lower_95,
      xend = resurgence_ratio_upper_plot,
      yend = country_label
    ),
    linewidth = 0.45,
    alpha = 0.40,
    lineend = "round"
  ) +
  geom_segment(
    data = filter(pilot_landscape, resurgence_ratio_upper_unbounded),
    aes(
      x = resurgence_ratio_lower_95,
      xend = resurgence_ratio_upper_plot,
      yend = country_label
    ),
    linewidth = 0.45,
    alpha = 0.40,
    arrow = arrow(type = "open", length = unit(1.2, "mm"))
  ) +
  geom_point(shape = 21, fill = clr("white"), size = 1.9, stroke = 0.36) +
  scale_x_log10(
    labels = label_number(accuracy = 0.1),
    breaks = c(0.3, 1, 3, 10, 30),
    limits = c(0.25, pilot_interval_cap * 1.08)
  ) +
  scale_colour_who_region(show_legend = FALSE) +
  labs(
    x = "Modelled-country resurgence ratio (95% bootstrap interval)",
    y = NULL
  ) +
  theme_manuscript()

figure_layout <- "
AB
CD
CE
FE
"

figure <- panel_tags(
  free(p_global) + p_region_ratio + free(p_region) + free(p_landscape) + free(p_low_baseline) + p_pilot +
    plot_layout(design = figure_layout, heights = c(0.57, 0.80, 0.35, 0.45), guides = "keep")
) &
  theme(
    plot.tag = element_text(
      face = "bold",
      size = figure_text$tag,
      margin = margin(0, 2, 1, 0)
    )
  )

save_supplement_figure(figure, "supplement_global_notification_context", nature_full_width_mm, 155)
