#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(ragg)
  library(scales)
  library(tidyr)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
repo_root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg)), "..", ".."), mustWork = TRUE)
setwd(repo_root)
source(file.path("script", "figures", "_common.R"))

# Figure contract
# Core conclusion: exact notification compensation conceals a strong infant
# burden contrast, while source and effective notification ascertainment differ;
# notification-only discrimination returns gradually as matching is relaxed.
# Archetype: asymmetric quantitative composite; panel b is the hero evidence.

matching_tolerance <- 1e-10

path <- read_csv_checked(
  fd_path("figure_4_representative_matched_path.csv"),
  required = c(
    "iso3", "group_id", "scenario_id", "trajectory_class", "date", "expected_notifications",
    "max_absolute_notification_difference_all_720_pairs", "surveillance_design_version"
  )
) |>
  mutate(
    date = as.Date(date),
    trajectory_class = factor(recode(
      scenario_id,
      matched_transmission_truth = "Transmission-context retained",
      matched_reporting_truth = "Ascertainment-change retained"
    ), levels = c("Transmission-context retained", "Ascertainment-change retained"))
  )
ratios <- read_csv_checked(
  fd_path("figure_4_matched_latent_ratios.csv"),
  required = c(
    "iso3", "group_id", "infant_symptomatic_burden_ratio_2024",
    "source_relative_ascertainment_ratio_2024",
    "effective_notification_ascertainment_ratio_2024", "surveillance_design_version"
  )
)
partial <- read_csv_checked(
  fd_path("figure_5_partial_notification_matching.csv"),
  required = c(
    "match_strength", "package_id", "validation_scheme", "auc",
    "auc_lower_95", "auc_upper_95", "n_bootstrap", "surveillance_design_version"
  )
) |>
  filter(package_id == "notifications") |>
  mutate(
    validation_scheme = recode(
      validation_scheme,
      grouped_five_fold = "Grouped CV",
      whole_country_holdout = "Whole-country holdout"
  )
)

input_versions <- unique(c(
  path$surveillance_design_version,
  ratios$surveillance_design_version,
  partial$surveillance_design_version
))
stopifnot(
  length(input_versions) == 1L,
  !is.na(input_versions),
  nzchar(input_versions)
)

stopifnot(
  nrow(ratios) == 720L,
  n_distinct(ratios$iso3) == 9L,
  all(is.finite(as.matrix(select(ratios, infant_symptomatic_burden_ratio_2024, source_relative_ascertainment_ratio_2024, effective_notification_ascertainment_ratio_2024)))),
  all(as.matrix(select(ratios, infant_symptomatic_burden_ratio_2024, source_relative_ascertainment_ratio_2024, effective_notification_ascertainment_ratio_2024)) > 0),
  max(path$max_absolute_notification_difference_all_720_pairs) <= matching_tolerance,
  all(partial$n_bootstrap == 2000L)
)

class_pal <- c("Transmission-context retained" = clr("blue"), "Ascertainment-change retained" = clr("orange"))
ascertainment_pal <- c("Source process" = clr("teal"), "Effective after matching" = clr("orange"))
validation_pal <- c("Grouped CV" = clr("blue"), "Whole-country holdout" = clr("orange"))

p_a <- ggplot(path, aes(date, expected_notifications, colour = trajectory_class, linetype = trajectory_class)) +
  geom_line(linewidth = 0.72) +
  scale_colour_manual(values = class_pal) +
  scale_linetype_manual(
    values = c("Transmission-context retained" = "solid", "Ascertainment-change retained" = "22")
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y", expand = expansion(mult = c(0.02, 0.02))) +
  scale_y_continuous(
    labels = label_number(scale_cut = cut_short_scale()),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(tag = "a", x = NULL, y = "Expected notifications", colour = NULL, linetype = NULL) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = 0.9))) +
  theme_nature(grid = "y") +
  theme(
    legend.position = "bottom", legend.direction = "horizontal",
    legend.box = "horizontal", legend.margin = margin(0, 0, 0, 0)
  )

rho <- cor(
  log2(ratios$infant_symptomatic_burden_ratio_2024),
  log2(ratios$effective_notification_ascertainment_ratio_2024),
  method = "spearman"
)
country_medians <- ratios |>
  group_by(iso3) |>
  summarise(
    burden = median(infant_symptomatic_burden_ratio_2024),
    ascertainment = median(effective_notification_ascertainment_ratio_2024),
    .groups = "drop"
  ) |>
  mutate(
    label_x = burden * case_when(iso3 == "SWE" ~ 0.82, iso3 == "AUS" ~ 1.08, TRUE ~ 1),
    label_y = ascertainment * case_when(
      iso3 == "SWE" ~ 1.16, iso3 == "GBR" ~ 0.87,
      iso3 == "THA" ~ 0.91, TRUE ~ 1.09
    )
  )
p_b <- ggplot(
  ratios,
  aes(infant_symptomatic_burden_ratio_2024, effective_notification_ascertainment_ratio_2024)
) +
  geom_abline(intercept = 0, slope = -1, linetype = "22", linewidth = 0.42, colour = clr("grey")) +
  geom_point(shape = 16, size = 0.75, alpha = 0.25, colour = clr("dark_grey")) +
  geom_point(
    data = country_medians, aes(x = burden, y = ascertainment),
    inherit.aes = FALSE, shape = 21, size = 2.25, stroke = 0.45,
    fill = clr("blue"), colour = "white"
  ) +
  geom_text(
    data = country_medians, aes(x = label_x, y = label_y, label = iso3),
    inherit.aes = FALSE, family = figure_font_family,
    size = 1.45, fontface = "bold"
  ) +
  annotate(
    "label", x = 1.55, y = 0.045,
    label = sprintf("Spearman ρ = %.3f\nn = 720 matched pairs\nDashed line: reciprocal reference", rho),
    family = figure_font_family, size = 1.62, hjust = 0,
    fill = "white", linewidth = 0.18, label.padding = unit(0.85, "mm")
  ) +
  scale_x_log10(breaks = c(1, 2, 4, 8, 16, 32), labels = label_number(suffix = "×")) +
  scale_y_log10(breaks = c(0.03, 0.05, 0.1, 0.2, 0.5, 1), labels = label_number(suffix = "×", accuracy = 0.01)) +
  coord_cartesian(
    xlim = c(min(0.9, min(ratios$infant_symptomatic_burden_ratio_2024) * 0.9), max(36, max(ratios$infant_symptomatic_burden_ratio_2024) * 1.08)),
    ylim = c(min(0.028, min(ratios$effective_notification_ascertainment_ratio_2024) * 0.9), max(1.25, max(ratios$effective_notification_ascertainment_ratio_2024) * 1.08))
  ) +
  labs(
    tag = "b",
    x = "Infant symptomatic-burden ratio\n(transmission-context / ascertainment-change)",
    y = "Effective notification-ascertainment ratio"
  ) +
  theme_nature(grid = "both")

ratio_country <- ratios |>
  group_by(iso3) |>
  summarise(
    burden_median = median(infant_symptomatic_burden_ratio_2024),
    burden_lower = quantile(infant_symptomatic_burden_ratio_2024, 0.025),
    burden_upper = quantile(infant_symptomatic_burden_ratio_2024, 0.975),
    .groups = "drop"
  ) |>
  mutate(iso3 = factor(iso3, levels = rev(pilot_iso3)))

p_c <- ggplot(ratio_country, aes(burden_median, iso3)) +
  geom_vline(xintercept = 1, linetype = "dotted", colour = clr("grey")) +
  geom_errorbar(
    aes(xmin = burden_lower, xmax = burden_upper), orientation = "y",
    width = 0, linewidth = 0.48, colour = clr("blue")
  ) +
  geom_point(shape = 21, size = 2.0, fill = clr("blue"), colour = "white", stroke = 0.4) +
  scale_x_log10(breaks = c(1, 2, 4, 8, 16, 32), labels = label_number(suffix = "×")) +
  scale_y_discrete(labels = country_abbrev) +
  labs(tag = "c", x = "Infant burden ratio\nmedian (95% ensemble range)", y = NULL) +
  theme_nature(grid = "x") +
  theme(axis.ticks.y = element_blank())

ascertainment_country <- ratios |>
  pivot_longer(
    c(source_relative_ascertainment_ratio_2024, effective_notification_ascertainment_ratio_2024),
    names_to = "ascertainment_type", values_to = "ratio"
  ) |>
  mutate(ascertainment_type = factor(recode(
    ascertainment_type,
    source_relative_ascertainment_ratio_2024 = "Source process",
    effective_notification_ascertainment_ratio_2024 = "Effective after matching"
  ), levels = names(ascertainment_pal))) |>
  group_by(iso3, ascertainment_type) |>
  summarise(
    median = median(ratio), lower = quantile(ratio, 0.025),
    upper = quantile(ratio, 0.975), .groups = "drop"
  ) |>
  mutate(iso3 = factor(iso3, levels = rev(pilot_iso3)))

p_d <- ggplot(ascertainment_country, aes(median, iso3, colour = ascertainment_type, group = ascertainment_type)) +
  geom_vline(xintercept = 1, linetype = "dotted", colour = clr("grey")) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper), orientation = "y",
    width = 0, linewidth = 0.48, position = position_dodge(width = 0.45)
  ) +
  geom_point(aes(shape = ascertainment_type), size = 1.7, fill = "white", stroke = 0.5, position = position_dodge(width = 0.45)) +
  scale_colour_manual(values = ascertainment_pal) +
  scale_shape_manual(values = c("Source process" = 21, "Effective after matching" = 16)) +
  scale_x_log10(breaks = c(0.03, 0.1, 0.3, 1, 3), labels = label_number(suffix = "×", accuracy = 0.01)) +
  scale_y_discrete(labels = country_abbrev) +
  labs(tag = "d", x = "Ascertainment ratio\nmedian (95% ensemble range)", y = NULL, colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_nature(grid = "x") +
  theme(
    axis.ticks.y = element_blank(), legend.position = "bottom",
    legend.direction = "horizontal", legend.box = "horizontal",
    legend.margin = margin(0, 0, 0, 0)
  )

p_e <- ggplot(partial, aes(match_strength, auc, colour = validation_scheme, fill = validation_scheme)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = clr("grey")) +
  geom_ribbon(aes(ymin = auc_lower_95, ymax = auc_upper_95), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.75) +
  scale_x_reverse(breaks = c(1, 0.95, 0.90, 0.80, 0.75)) +
  scale_y_continuous(breaks = c(0.5, 0.6, 0.7, 0.8)) +
  coord_cartesian(ylim = c(0.46, max(0.86, max(partial$auc_upper_95) * 1.02))) +
  scale_colour_manual(values = validation_pal) +
  scale_fill_manual(values = validation_pal) +
  labs(
    tag = "e", x = "Notification-match strength", y = "Notification-only AUC",
    colour = NULL, fill = NULL
  ) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_nature(grid = "y") +
  theme(
    legend.position = "bottom", legend.direction = "horizontal",
    legend.box = "horizontal", legend.margin = margin(0, 0, 0, 0),
    legend.text = element_text(size = figure_text$legend_text_dense)
  )

figure <- (patchwork::free(p_a) | p_b) / (p_c | p_d | p_e) +
  plot_layout(heights = c(1.12, 1), widths = c(1, 1, 1.16))

stem <- "figure_2_matched_notification_benchmark"
width <- 183 / 25.4
height <- 148 / 25.4

grDevices::cairo_pdf(fig_path(paste0(stem, ".pdf")), width = width, height = height, family = "sans")
print(figure)
dev.off()
ragg::agg_png(fig_path(paste0(stem, ".png")), width = width, height = height, units = "in", res = 600, background = "white")
print(figure)
dev.off()
message(
  "Wrote Figure 2 ambiguity composite from surveillance-design version ",
  input_versions[[1]],
  "."
)
