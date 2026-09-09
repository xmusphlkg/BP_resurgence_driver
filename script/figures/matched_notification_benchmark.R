#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(ragg)
  library(readr)
  library(scales)
  library(svglite)
  library(tidyr)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Run this file with Rscript so the project root can be resolved.", call. = FALSE)
}
script_file <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_file), "..", ".."), mustWork = TRUE)
setwd(repo_root)

source(file.path("script", "figures", "_common.R"))

# Archived diagnostic renderers write outside the active manuscript figure set.
diagnostic_dir <- file.path(root, "manuscript", "reports", "figures")
dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)
fig_path <- function(name) file.path(diagnostic_dir, name)

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

# Figure contract:
# Conclusion: exact notification matching can hide distinct infant burdens and effective
# notification ascertainment states, leaving notification-only
# discrimination at chance across country backgrounds.
# Archetype: Figure-4-aligned quantitative grid with one hero panel.
# Evidence: exact notification match -> country-resolved latent separation ->
# paired compensation -> ROC failure -> cross-structure negative control.

matched_path <- read_csv_checked(
  fd_path("figure_4_representative_matched_path.csv"),
  required = c(
    "iso3", "group_id", "trajectory_class", "date",
    "expected_notifications",
    "max_absolute_notification_difference_all_720_pairs",
    "surveillance_design_version"
  )
) |>
  mutate(
    date = as.Date(date),
    trajectory_class = recode(
      trajectory_class,
      "Reporting-dominant" = "Ascertainment-change retained"
    )
  )

matched_ratios <- read_csv_checked(
  fd_path("figure_4_matched_latent_ratios.csv"),
  required = c(
    "iso3", "group_id", "infant_symptomatic_burden_ratio_2024",
    "effective_notification_ascertainment_ratio_2024", "surveillance_design_version"
  )
)

country_auc <- read_csv_checked(
  fd_path("figure_4_country_holdout_auc.csv"),
  required = c(
    "measurement_design", "package_id", "iso3", "validation_scheme",
    "n_country_draw_groups", "n_observations", "auc",
    "surveillance_design_version"
  )
) |>
  filter(
    measurement_design == "main",
    package_id == "notifications",
    validation_scheme %in% c("leave_one_country_out", "whole_country_holdout")
  )

grouped_auc <- read_csv_checked(
  fd_path("figure_4_package_performance.csv"),
  required = c(
    "package_id", "validation", "auc", "auc_lower_95", "auc_upper_95",
    "n_bootstrap", "bootstrap_seed", "interval_method",
    "surveillance_design_version"
  )
) |>
  filter(package_id == "notifications", validation == "Grouped 5-fold CV")

notification_roc <- read_csv_checked(
  fd_path("figure_4_grouped_cv_roc.csv"),
  required = c(
    "package_id", "point_order", "false_positive_rate",
    "true_positive_rate", "validation_scheme", "auc",
    "n_country_draw_groups", "n_observations",
    "surveillance_design_version"
  )
) |>
  filter(
    package_id == "notifications",
    validation_scheme == "grouped_five_fold_oof"
  ) |>
  arrange(point_order)

latent_signal_path <- file.path(
  outputs_dir,
  "model2c_surveillance_design",
  "monitoring_packages_v4_source_preserved_20260908",
  "latent_surveillance_signals.csv"
)
assert_true(
  file.exists(latent_signal_path),
  paste0("Required exact-match audit input is missing: ", latent_signal_path)
)
latent_signal_columns <- names(
  readr::read_csv(
    latent_signal_path,
    n_max = 0,
    show_col_types = FALSE,
    progress = FALSE
  )
)
notification_columns <- grep(
  "^latent_notification_mean_20(23|24|25)_m(0[1-9]|1[0-2])$",
  latent_signal_columns,
  value = TRUE
)
assert_true(
  length(notification_columns) == 36L,
  "The exact-match audit input must contain 36 monthly notification columns."
)
matched_all <- readr::read_csv(
  latent_signal_path,
  col_select = tidyselect::all_of(c(
    "iso3", "group_id", "scenario_id", "surveillance_design_version",
    notification_columns
  )),
  show_col_types = FALSE,
  progress = FALSE
) |>
  filter(scenario_id %in% c(
    "matched_transmission_truth", "matched_reporting_truth"
  ))

all_versions <- unique(c(
  matched_path$surveillance_design_version,
  matched_ratios$surveillance_design_version,
  country_auc$surveillance_design_version,
  grouped_auc$surveillance_design_version,
  notification_roc$surveillance_design_version,
  matched_all$surveillance_design_version
))
assert_true(
  length(all_versions) == 1L,
  "Figure 5 inputs do not share one frozen surveillance-design version."
)

assert_true(
  nrow(matched_path) == 72L &&
    n_distinct(matched_path$group_id) == 1L &&
    n_distinct(matched_path$date) == 36L &&
    setequal(
      unique(matched_path$trajectory_class),
      c("Transmission-context retained", "Ascertainment-change retained")
    ),
  "Panel a requires one 36-month representative path for both latent classes."
)

path_wide <- matched_path |>
  select(date, trajectory_class, expected_notifications) |>
  pivot_wider(names_from = trajectory_class, values_from = expected_notifications)
assert_true(
  all(
    abs(
      path_wide[["Transmission-context retained"]] -
        path_wide[["Ascertainment-change retained"]]
    ) <= 1e-10
  ) &&
    max(matched_path$max_absolute_notification_difference_all_720_pairs) <= 1e-10,
  "Panel a notification paths are not exactly matched."
)

matched_agreement <- matched_all |>
  pivot_longer(
    cols = all_of(notification_columns),
    names_to = "signal",
    values_to = "expected_notifications"
  ) |>
  mutate(
    year = as.integer(sub(
      ".*mean_([0-9]{4})_m[0-9]{2}$", "\\1", signal
    )),
    month = as.integer(sub(
      ".*mean_[0-9]{4}_m([0-9]{2})$", "\\1", signal
    )),
    date = as.Date(sprintf("%04d-%02d-01", year, month)),
    trajectory_class = recode(
      scenario_id,
      matched_transmission_truth = "Transmission-context retained",
      matched_reporting_truth = "Ascertainment-change retained"
    )
  ) |>
  select(
    iso3, group_id, date, trajectory_class, expected_notifications,
    surveillance_design_version
  ) |>
  pivot_wider(
    names_from = trajectory_class,
    values_from = expected_notifications
  )

assert_true(
  nrow(matched_all) == 1440L &&
    n_distinct(matched_all$group_id) == 720L &&
    nrow(matched_agreement) == 25920L &&
    all(matched_agreement[["Transmission-context retained"]] > 0) &&
    all(matched_agreement[["Ascertainment-change retained"]] > 0) &&
    max(abs(
      matched_agreement[["Transmission-context retained"]] -
        matched_agreement[["Ascertainment-change retained"]]
    )) <= 1e-10,
  "Panel a audit requires exact agreement for 720 pairs across all 36 months."
)

groups_per_country <- matched_ratios |> count(iso3, name = "n_pairs")
assert_true(
  nrow(matched_ratios) == 720L &&
    n_distinct(matched_ratios$group_id) == 720L &&
    n_distinct(matched_ratios$iso3) == 9L &&
    all(groups_per_country$n_pairs == 80L),
  paste0(
    "Panels b-c require 720 paired country–ensemble calibration backgrounds ",
    "(nine countries by 80 retained members)."
  )
)
assert_true(
  nrow(country_auc) == 9L &&
    setequal(country_auc$iso3, unique(matched_ratios$iso3)) &&
    nrow(grouped_auc) == 1L &&
    grouped_auc$n_bootstrap == 2000L &&
    grouped_auc$bootstrap_seed == 20260730L &&
    grouped_auc$auc_lower_95 <= grouped_auc$auc &&
    grouped_auc$auc <= grouped_auc$auc_upper_95,
  "Panel f requires nine notification-only holdouts and the audited grouped-CV interval."
)
assert_true(
  nrow(notification_roc) > 100L &&
    notification_roc$false_positive_rate[[1]] == 0 &&
    notification_roc$true_positive_rate[[1]] == 0 &&
    dplyr::last(notification_roc$false_positive_rate) == 1 &&
    dplyr::last(notification_roc$true_positive_rate) == 1 &&
    all(diff(notification_roc$false_positive_rate) >= 0) &&
    all(diff(notification_roc$true_positive_rate) >= 0) &&
    all(abs(notification_roc$auc - grouped_auc$auc) <= 1e-12),
  "Panel e requires the complete notifications-only grouped-CV ROC curve."
)

# Manuscript-facing source data.
path_export <- matched_path |> mutate(panel = "a") |> select(panel, everything())
agreement_export <- matched_agreement |>
  transmute(
    panel = "a_audit", iso3, group_id, date,
    ascertainment_change_notifications = `Ascertainment-change retained`,
    transmission_context_notifications = `Transmission-context retained`,
    absolute_difference = abs(
      transmission_context_notifications -
        ascertainment_change_notifications
    ),
    surveillance_design_version
  )
ratio_export <- matched_ratios |> mutate(panel = "b-c") |> select(panel, everything())

ratio_long <- matched_ratios |>
  transmute(
    iso3,
    group_id,
    burden = infant_symptomatic_burden_ratio_2024,
    ascertainment = effective_notification_ascertainment_ratio_2024
  ) |>
  pivot_longer(
    cols = c(burden, ascertainment),
    names_to = "metric_id",
    values_to = "ratio"
  ) |>
  mutate(
    metric = recode(
      metric_id,
      burden = "Infant symptomatic burden",
      ascertainment = "Effective notification ascertainment"
    ),
    metric = factor(
      metric,
      levels = c("Effective notification ascertainment", "Infant symptomatic burden")
    ),
    log2_ratio = log2(ratio)
  )

country_order <- ratio_long |>
  filter(metric == "Infant symptomatic burden") |>
  group_by(iso3) |>
  summarise(median_ratio = median(ratio), .groups = "drop") |>
  arrange(desc(median_ratio)) |>
  pull(iso3)

country_names <- c(
  AUS = "Australia",
  BRA = "Brazil",
  CHN = "China",
  GBR = "United Kingdom",
  JPN = "Japan",
  NZL = "New Zealand",
  SWE = "Sweden",
  THA = "Thailand",
  USA = "United States"
)
assert_true(
  setequal(names(country_names), country_order),
  "Figure 5 country labels do not match the nine fitted structures."
)

y_lookup <- setNames(c(10:2, 1), c(country_order, "Pooled"))

ratio_country_summary <- ratio_long |>
  group_by(iso3, metric) |>
  summarise(
    lower_95 = quantile(ratio, 0.025),
    median = median(ratio),
    upper_95 = quantile(ratio, 0.975),
    n_pairs = n(),
    .groups = "drop"
  ) |>
  mutate(scope = "country")

ratio_pooled_summary <- ratio_long |>
  group_by(metric) |>
  summarise(
    lower_95 = quantile(ratio, 0.025),
    median = median(ratio),
    upper_95 = quantile(ratio, 0.975),
    n_pairs = n(),
    .groups = "drop"
  ) |>
  mutate(scope = "pooled", iso3 = "Pooled")

ratio_summary <- bind_rows(ratio_country_summary, ratio_pooled_summary) |>
  mutate(
    y = unname(y_lookup[iso3]),
    panel = if_else(metric == "Infant symptomatic burden", "b", "c"),
    log2_lower_95 = log2(lower_95),
    log2_median = log2(median),
    log2_upper_95 = log2(upper_95),
    interval_label = "95% UI",
    ui_method = "calibration_ensemble_percentile_2.5_97.5"
  )

tradeoff_pairs <- matched_ratios |>
  transmute(
    panel = "d",
    scope = "paired background",
    iso3,
    group_id,
    infant_symptomatic_burden_ratio_2024,
    effective_notification_ascertainment_ratio_2024,
    log2_burden_ratio = log2(infant_symptomatic_burden_ratio_2024),
    log2_ascertainment_ratio = log2(effective_notification_ascertainment_ratio_2024),
    surveillance_design_version
  )

tradeoff_country_medians <- tradeoff_pairs |>
  group_by(iso3) |>
  summarise(
    log2_burden_ratio = median(log2_burden_ratio),
    log2_ascertainment_ratio = median(log2_ascertainment_ratio),
    .groups = "drop"
  )

tradeoff_pooled <- tradeoff_pairs |>
  summarise(
    log2_burden_ratio = median(log2_burden_ratio),
    log2_ascertainment_ratio = median(log2_ascertainment_ratio)
  )

tradeoff_spearman <- cor(
  tradeoff_pairs$log2_burden_ratio,
  tradeoff_pairs$log2_ascertainment_ratio,
  method = "spearman"
)
assert_true(
  is.finite(tradeoff_spearman),
  "Panel d requires a finite descriptive rank correlation."
)

tradeoff_export <- bind_rows(
  tradeoff_pairs |>
    select(
      panel, scope, iso3, group_id,
      infant_symptomatic_burden_ratio_2024,
      effective_notification_ascertainment_ratio_2024,
      log2_burden_ratio, log2_ascertainment_ratio,
      surveillance_design_version
    ),
  tradeoff_country_medians |>
    transmute(
      panel = "d", scope = "country median", iso3,
      group_id = NA_character_,
      infant_symptomatic_burden_ratio_2024 = 2 ^ log2_burden_ratio,
      effective_notification_ascertainment_ratio_2024 = 2 ^ log2_ascertainment_ratio,
      log2_burden_ratio, log2_ascertainment_ratio,
      surveillance_design_version = all_versions[[1]]
    ),
  tradeoff_pooled |>
    transmute(
      panel = "d", scope = "pooled median", iso3 = "Pooled",
      group_id = NA_character_,
      infant_symptomatic_burden_ratio_2024 = 2 ^ log2_burden_ratio,
      effective_notification_ascertainment_ratio_2024 = 2 ^ log2_ascertainment_ratio,
      log2_burden_ratio, log2_ascertainment_ratio,
      surveillance_design_version = all_versions[[1]]
    )
)

roc_export <- notification_roc |>
  mutate(panel = "e", .before = 1)

negative_control_export <- bind_rows(
  country_auc |>
    transmute(
      panel = "f",
      validation = "Whole-country holdout",
      iso3,
      estimate = auc,
      lower_95 = NA_real_,
      upper_95 = NA_real_,
      n_country_draw_groups,
      n_observations,
      interval_method = NA_character_,
      n_bootstrap = NA_integer_,
      bootstrap_seed = NA_integer_,
      surveillance_design_version
    ),
  grouped_auc |>
    transmute(
      panel = "f",
      validation,
      iso3 = "Pooled",
      estimate = auc,
      lower_95 = auc_lower_95,
      upper_95 = auc_upper_95,
      n_country_draw_groups = 720L,
      n_observations = 4320L,
      interval_method,
      n_bootstrap,
      bootstrap_seed,
      surveillance_design_version
    )
)

write_csv(path_export, fd_path("matched_notification_path.csv"), na = "")
write_csv(
  agreement_export,
  fd_path("matched_notification_exact_agreement.csv"),
  na = ""
)
write_csv(ratio_export, fd_path("matched_notification_latent_ratios.csv"), na = "")
write_csv(
  ratio_summary |>
    transmute(
      panel, scope, iso3, metric, n_pairs,
      lower_95, median, upper_95,
      log2_lower_95, log2_median, log2_upper_95,
      interval_label, ui_method
    ),
  fd_path("matched_notification_country_latent_summary.csv"),
  na = ""
)
write_csv(
  tradeoff_export,
  fd_path("matched_notification_latent_tradeoff.csv"),
  na = ""
)
write_csv(
  roc_export,
  fd_path("matched_notification_notifications_roc.csv"),
  na = ""
)
write_csv(
  negative_control_export,
  fd_path("matched_notification_negative_control_auc.csv"),
  na = ""
)

# Visual contract.
ink <- figure_colours$text
muted <- figure_colours$muted_text
# Match Figure 4's palette while preserving one colour per scientific role:
# orange and blue identify the two latent mechanisms, purple their paired
# trade-off, and teal the notification-only diagnostic benchmark.
transmission_colour <- clr("blue")
ascertainment_colour <- clr("orange")
tradeoff_colour <- clr("purple")
notification_colour <- clr("teal")
class_colours <- c(
  "Transmission-context retained" = transmission_colour,
  "Ascertainment-change retained" = ascertainment_colour
)

# Axis contract. Limits are fixed to interpretable ratio/probability bounds and
# every scale uses zero expansion. Ratio and scatter axes include all observed
# intervals and paired states without assuming which side of parity they occupy.
path_x_limits <- range(matched_path$date)
path_x_breaks <- as.Date(sprintf("%04d-01-01", 2023:2025))
path_y_limits <- range(matched_path$expected_notifications) * c(0.85, 1.2)
path_y_breaks <- c(100, 300, 1000, 3000)

burden_log2_limits <- range(c(0, floor(log2(ratio_summary$lower_95[ratio_summary$metric == "Infant symptomatic burden"])), ceiling(log2(ratio_summary$upper_95[ratio_summary$metric == "Infant symptomatic burden"]))))
burden_log2_breaks <- seq(burden_log2_limits[[1]], burden_log2_limits[[2]], by = 1)
burden_ratio_labels <- as.character(2 ^ burden_log2_breaks)

ascertainment_log2_limits <- range(c(0, floor(log2(ratio_summary$lower_95[ratio_summary$metric == "Effective notification ascertainment"])), ceiling(log2(ratio_summary$upper_95[ratio_summary$metric == "Effective notification ascertainment"]))))
ascertainment_log2_breaks <- seq(ascertainment_log2_limits[[1]], ascertainment_log2_limits[[2]], by = 1)
ascertainment_ratio_labels <- format(2^ascertainment_log2_breaks, digits = 2)

tradeoff_x_limits <- range(c(0, tradeoff_pairs$log2_burden_ratio)) + c(-0.15, 0.15)
tradeoff_y_limits <- range(c(0, tradeoff_pairs$log2_ascertainment_ratio)) + c(-0.15, 0.15)
tradeoff_x_breaks <- seq(0, 5, by = 1)
tradeoff_y_breaks <- seq(-5, 0, by = 1)

roc_limits <- c(0, 1)
roc_breaks <- figure_breaks$probability

control_auc_pretty <- pretty(
  c(
    country_auc$auc,
    grouped_auc$auc_lower_95,
    grouped_auc$auc,
    grouped_auc$auc_upper_95,
    0.5
  ),
  n = 5
)
control_auc_limits <- range(control_auc_pretty)
control_auc_breaks <- sort(unique(c(control_auc_limits, 0.5)))

assert_true(
  all(range(log2(ratio_summary$lower_95[
    ratio_summary$metric == "Infant symptomatic burden"
  ])) >= burden_log2_limits[[1]]) &&
    all(range(log2(ratio_summary$upper_95[
      ratio_summary$metric == "Infant symptomatic burden"
    ])) <= burden_log2_limits[[2]]) &&
    all(range(log2(ratio_summary$lower_95[
      ratio_summary$metric == "Effective notification ascertainment"
    ])) >= ascertainment_log2_limits[[1]]) &&
    all(range(log2(ratio_summary$upper_95[
      ratio_summary$metric == "Effective notification ascertainment"
    ])) <= ascertainment_log2_limits[[2]]),
  "Figure 5b-c ratio intervals exceed their data-derived axes."
)
assert_true(
  all(range(tradeoff_pairs$log2_burden_ratio) >= tradeoff_x_limits[[1]]) &&
    all(range(tradeoff_pairs$log2_burden_ratio) <= tradeoff_x_limits[[2]]) &&
    all(range(tradeoff_pairs$log2_ascertainment_ratio) >= tradeoff_y_limits[[1]]) &&
    all(range(tradeoff_pairs$log2_ascertainment_ratio) <= tradeoff_y_limits[[2]]) &&
    min(c(country_auc$auc, grouped_auc$auc_lower_95)) >= control_auc_limits[[1]] &&
    max(c(country_auc$auc, grouped_auc$auc_upper_95)) <= control_auc_limits[[2]],
  "Figure 5d or 5f data exceed their plotting limits."
)

panel_style <- function(
    grid = "none",
    plot_margin = margin(3.5, 4.5, 3.5, 4.5)) {
  theme_manuscript(
    grid = grid,
    show_titles = FALSE,
    plot_margin = plot_margin
  ) +
    theme(
      axis.text = element_text(size = 5.2),
      axis.title = element_text(size = 5.8),
      legend.text = element_text(size = 5.0),
      legend.title = element_text(size = 5.2),
      plot.tag = element_text(size = 8, face = "bold"),
      plot.tag.position = c(0, 1)
    )
}

# a, exact overlap in a representative notification trajectory.
p_a <- ggplot(matched_path, aes(date, expected_notifications)) +
  geom_line(
    data = filter(matched_path, trajectory_class == "Transmission-context retained"),
    aes(colour = trajectory_class, linetype = trajectory_class),
    linewidth = 0.92
  ) +
  geom_line(
    data = filter(matched_path, trajectory_class == "Ascertainment-change retained"),
    aes(colour = trajectory_class, linetype = trajectory_class),
    linewidth = 0.60
  ) +
  scale_colour_manual(
    values = class_colours,
    breaks = names(class_colours),
    name = NULL
  ) +
  scale_linetype_manual(
    values = c(
      "Transmission-context retained" = "solid",
      "Ascertainment-change retained" = "22"
    ),
    breaks = names(class_colours),
    name = NULL
  ) +
  scale_x_date(
    limits = path_x_limits,
    breaks = path_x_breaks,
    date_labels = "%Y",
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_log10(
    limits = path_y_limits,
    breaks = path_y_breaks,
    labels = label_number(scale_cut = cut_short_scale()),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(tag = "a", x = NULL, y = "Expected monthly notifications") +
  panel_style(plot_margin = margin(3.5, 4.5, 2.5, 4.5)) +
  guides(
    colour = guide_legend(ncol = 1, byrow = TRUE),
    linetype = guide_legend(ncol = 1, byrow = TRUE)
  ) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(1, 0.01),
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    legend.background = element_rect(fill = alpha("white", 0.92), colour = NA),
    legend.key.width = unit(3.6, "mm"),
    legend.key.height = unit(2.2, "mm")
  )

# b, country-resolved infant symptomatic-burden separation.
p_b <- ggplot(
  filter(ratio_summary, metric == "Infant symptomatic burden"),
  aes(y = y)
) +
  geom_rect(
    data = tibble(xmin = 0, xmax = 5, ymin = 0.5, ymax = 1.5),
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE, fill = clr("paper"), colour = NA
  ) +
  geom_segment(
    aes(x = log2_lower_95, xend = log2_upper_95, yend = y),
    linewidth = 0.62, lineend = "round", colour = transmission_colour
  ) +
  geom_point(
    aes(x = log2_median, shape = scope), colour = transmission_colour,
    size = 1.65, stroke = 0.52, fill = "white"
  ) +
  scale_shape_manual(values = c(country = 21, pooled = 23), guide = "none") +
  scale_x_continuous(
    limits = burden_log2_limits,
    breaks = burden_log2_breaks,
    labels = burden_ratio_labels,
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0.5, 10.5),
    breaks = 1:10,
    labels = c("Pooled", unname(country_names[rev(country_order)])),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    tag = "b",
    x = "Infant symptomatic-burden ratio\nmedian (95% ensemble range)",
    y = NULL
  ) +
  panel_style(grid = "x", plot_margin = margin(3.5, 4.5, 4.0, 4.5)) +
  theme(
    axis.text.y = element_text(size = 5.25),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = figure_colours$grid)
  )

# c, country-resolved relative-ascertainment separation.
p_c <- ggplot(
  filter(ratio_summary, metric == "Effective notification ascertainment"),
  aes(y = y)
) +
  geom_rect(
    data = tibble(xmin = -5, xmax = 0, ymin = 0.5, ymax = 1.5),
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE, fill = clr("paper"), colour = NA
  ) +
  geom_segment(
    aes(x = log2_lower_95, xend = log2_upper_95, yend = y),
    linewidth = 0.62, lineend = "round", colour = ascertainment_colour
  ) +
  geom_point(
    aes(x = log2_median, shape = scope), colour = ascertainment_colour,
    size = 1.65, stroke = 0.52, fill = "white"
  ) +
  scale_shape_manual(values = c(country = 21, pooled = 23), guide = "none") +
  scale_x_continuous(
    limits = ascertainment_log2_limits,
    breaks = ascertainment_log2_breaks,
    labels = ascertainment_ratio_labels,
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0.5, 10.5),
    breaks = 1:10,
    labels = rep("", 10),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    tag = "c",
    x = "Effective ascertainment ratio\nmedian (95% ensemble range)",
    y = NULL
  ) +
  panel_style(grid = "x", plot_margin = margin(3.5, 4.5, 4.0, 4.5)) +
  theme(
    axis.text.y = element_text(size = 5.25),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = figure_colours$grid)
  )

# d, paired latent compensation across all 720 backgrounds.
p_d <- ggplot(
  tradeoff_pairs,
  aes(x = log2_burden_ratio, y = log2_ascertainment_ratio)
) +
  geom_abline(
    intercept = 0, slope = -1,
    linewidth = 0.38, linetype = "dotted", colour = muted
  ) +
  geom_point(
    size = 0.64, alpha = 0.16, stroke = 0,
    colour = tradeoff_colour
  ) +
  geom_point(
    data = tradeoff_country_medians,
    shape = 21, size = 1.50, stroke = 0.45,
    fill = "white", colour = tradeoff_colour
  ) +
  geom_point(
    data = tradeoff_pooled,
    shape = 23, size = 2.05, stroke = 0.42,
    fill = tradeoff_colour, colour = tradeoff_colour
  ) +
  scale_x_continuous(
    limits = tradeoff_x_limits,
    breaks = tradeoff_x_breaks,
    labels = as.character(2 ^ tradeoff_x_breaks),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = tradeoff_y_limits,
    breaks = tradeoff_y_breaks,
    labels = c("1/32", "1/16", "1/8", "1/4", "1/2", "1"),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_fixed() +
  labs(
    tag = "d",
    x = "Infant burden ratio",
    y = "Effective ascertainment ratio"
  ) +
  panel_style(grid = "both", plot_margin = margin(3.5, 4.5, 4.0, 4.5)) +
  theme(
    panel.grid.major = element_line(
      linewidth = 0.16,
      colour = figure_colours$grid
    )
  )

# e, shape of notification-only discrimination failure.
p_e <- ggplot(
  notification_roc,
  aes(x = false_positive_rate, y = true_positive_rate)
) +
  geom_abline(
    intercept = 0, slope = 1,
    linewidth = 0.38, linetype = "dotted", colour = muted
  ) +
  geom_step(
    linewidth = 0.78, direction = "hv",
    colour = notification_colour
  ) +
  scale_x_continuous(
    limits = roc_limits, breaks = roc_breaks,
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = roc_limits, breaks = roc_breaks,
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_fixed() +
  labs(
    tag = "e",
    x = "False-positive rate",
    y = "True-positive rate"
  ) +
  panel_style(grid = "both", plot_margin = margin(3.5, 4.5, 4.0, 4.5)) +
  theme(
    panel.grid.major = element_line(
      linewidth = 0.16,
      colour = figure_colours$grid
    )
  )

# f, compact cross-structure and grouped-validation negative control.
country_auc_plot <- country_auc |> mutate(y = 2)
country_auc_median <- median(country_auc_plot$auc)

p_f <- ggplot() +
  geom_vline(xintercept = 0.5, linewidth = 0.34, linetype = "dotted", colour = muted) +
  geom_segment(
    x = min(country_auc_plot$auc), xend = max(country_auc_plot$auc),
    y = 2, yend = 2,
    linewidth = 0.62, lineend = "round",
    colour = notification_colour, alpha = 0.68
  ) +
  geom_point(
    data = country_auc_plot,
    aes(x = auc, y = y),
    position = position_jitter(height = 0.075, width = 0, seed = 20260730),
    shape = 21, size = 1.35, stroke = 0.38,
    fill = "white", colour = notification_colour
  ) +
  geom_point(
    aes(x = country_auc_median, y = 2),
    shape = 23, size = 1.85, stroke = 0.45,
    fill = "white", colour = notification_colour
  ) +
  geom_segment(
    data = grouped_auc,
    aes(x = auc_lower_95, xend = auc_upper_95, y = 1, yend = 1),
    linewidth = 1.05, lineend = "round", colour = notification_colour
  ) +
  geom_point(
    data = grouped_auc,
    aes(x = auc, y = 1),
    shape = 23, size = 2.0, stroke = 0.48,
    fill = notification_colour, colour = notification_colour
  ) +
  scale_x_continuous(
    limits = control_auc_limits,
    breaks = control_auc_breaks,
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0.65, 2.35), breaks = c(1, 2),
    labels = c("Grouped 5-fold", "Whole-country holdouts"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(tag = "f", x = "AUC using notifications only", y = NULL) +
  panel_style(grid = "x", plot_margin = margin(3.5, 4.5, 4.0, 4.5)) +
  theme(
    axis.text.y = element_text(size = 4.8, lineheight = 0.92),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = figure_colours$grid)
  )

top_row <- wrap_plots(
  list(p_a, p_b, p_c),
  ncol = 3,
  widths = c(1.10, 1.15, 0.90)
)

bottom_row <- wrap_plots(
  list(p_d, p_e, p_f),
  ncol = 3,
  widths = c(1.0, 1.0, 1.0)
)

figure <- wrap_plots(
  list(top_row, bottom_row),
  ncol = 1,
  heights = c(1.08, 0.92)
) +
  plot_annotation(
    theme = theme(
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(2, 2, 2, 2)
    )
  )

# R-only export bundle. Cairo PDF and svglite retain editable vector text;
# ragg produces submission-grade 600-dpi raster files.
figure_stem <- "matched_notification_benchmark"
width_mm <- nature_full_width_mm
height_mm <- 124
width_in <- width_mm / 25.4
height_in <- height_mm / 25.4
dpi <- 600

export_figure <- function(extension) {
  filename <- fig_path(paste0(figure_stem, ".", extension))
  if (extension == "pdf") {
    grDevices::cairo_pdf(
      filename, width = width_in, height = height_in,
      family = figure_font_family, bg = "white"
    )
  } else if (extension == "svg") {
    svglite::svglite(
      filename, width = width_in, height = height_in,
      bg = "white"
    )
  } else if (extension == "png") {
    ragg::agg_png(
      filename, width = width_in, height = height_in,
      units = "in", res = dpi, background = "white"
    )
  } else if (extension == "tiff") {
    ragg::agg_tiff(
      filename, width = width_in, height = height_in,
      units = "in", res = dpi, compression = "lzw",
      background = "white"
    )
  } else {
    stop("Unsupported export extension: ", extension, call. = FALSE)
  }
  print(figure)
  grDevices::dev.off()
}

invisible(lapply(c("png", "pdf", "svg", "tiff"), export_figure))

export_paths <- fig_path(
  paste0(figure_stem, ".", c("png", "pdf", "svg", "tiff"))
)
assert_true(
  all(file.exists(export_paths)) && all(file.info(export_paths)$size > 10000),
  "Matched-notification benchmark export bundle is incomplete."
)

message(
  "Wrote the redesigned six-panel matched-notification benchmark, ",
  "publication exports and canonical source-data files."
)
