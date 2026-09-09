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
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

holdout_palette <- c("#E9E9E9", "#AFCFE3", "#0072B2")
holdout_text_colour <- function(auc) {
  fills <- scales::gradient_n_pal(holdout_palette)(squish(auc, c(0, 1)))
  rgb <- t(grDevices::col2rgb(fills)) / 255
  linear <- ifelse(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)^2.4)
  luminance <- as.vector(linear %*% c(0.2126, 0.7152, 0.0722))
  white_contrast <- 1.05 / (luminance + 0.05)
  black_contrast <- (luminance + 0.05) / 0.05
  stopifnot(all(pmax(white_contrast, black_contrast) >= 4.5))
  ifelse(white_contrast > black_contrast, "white", "#000000")
}

package_auc <- read_csv_checked(
  fd_path("surveillance_design_package_auc.csv"),
  required = c(
    "package_id", "auc", "auc_lower_95", "auc_upper_95",
    "surveillance_design_version"
  )
)
roc_curve <- read_csv_checked(
  fd_path("figure_4_grouped_cv_roc.csv"),
  required = c(
    "package_id", "point_order", "threshold", "false_positive_rate",
    "true_positive_rate", "validation_scheme", "auc", "auc_lower_95",
    "auc_upper_95", "n_features", "n_country_draw_groups", "n_observations",
    "surveillance_design_version"
  )
)
country_holdout <- read_csv_checked(
  file.path("manuscript", "tables", "surveillance_design_country_holdout.csv"),
  required = c(
    "measurement_design", "package_id", "validation_scheme", "auc",
    "n_countries", "n_countries_auc_ge_0_70", "surveillance_design_version"
  )
) |>
  filter(
    measurement_design == "main",
    validation_scheme %in% c("leave_one_country_out", "whole_country_holdout")
  )
matched_path <- read_csv_checked(
  fd_path("figure_4_representative_matched_path.csv"),
  required = c(
    "iso3", "group_id", "trajectory_class", "date",
    "expected_notifications",
    "max_absolute_notification_difference_all_720_pairs",
    "surveillance_design_version"
  )
) |>
  mutate(date = as.Date(date))
matched_ratios <- read_csv_checked(
  fd_path("figure_4_matched_latent_ratios.csv"),
  required = c(
    "iso3", "group_id", "infant_symptomatic_burden_ratio_2024",
    "effective_notification_ascertainment_ratio_2024", "surveillance_design_version"
  )
)
shapley_gain <- read_csv_checked(
  fd_path("surveillance_design_shapley_auc_gain.csv"),
  required = c("stream", "shapley_auc_gain", "surveillance_design_version")
)
country_holdout_detail <- read_csv_checked(
  fd_path("figure_4_country_holdout_auc.csv"),
  required = c(
    "measurement_design", "package_id", "iso3", "validation_scheme",
    "n_country_draw_groups", "n_observations", "auc",
    "surveillance_design_version"
  )
)
precision_robustness <- read_csv_checked(
  fd_path("figure_4_precision_robustness_auc.csv"),
  required = c(
    "measurement_design", "package_id", "auc", "n_country_draw_groups",
    "n_observations", "surveillance_design_version"
  )
)

package_order <- c(
  "notifications",
  "notifications+age",
  "notifications+serology",
  "notifications+testing",
  "notifications+severe",
  "notifications+age+testing+severe+serology"
)
package_auc <- package_auc |> filter(package_id %in% package_order)
country_holdout <- country_holdout |> filter(package_id %in% package_order)
package_labels_short <- c(
  notifications = "Notifications",
  `notifications+age` = "+ age",
  `notifications+serology` = "+ serology",
  `notifications+testing` = "+ testing",
  `notifications+severe` = "+ severe",
  `notifications+age+testing+severe+serology` = "Complete"
)
package_colours <- c(
  notifications = "#5B5B5B",
  `notifications+age` = "#A6A6A6",
  `notifications+serology` = "#CC79A7",
  `notifications+testing` = "#0072B2",
  `notifications+severe` = "#D55E00",
  `notifications+age+testing+severe+serology` = "#1B1B1B"
)
package_linewidths <- c(
  notifications = 0.55,
  `notifications+age` = 0.55,
  `notifications+serology` = 0.64,
  `notifications+testing` = 0.76,
  `notifications+severe` = 0.76,
  `notifications+age+testing+severe+serology` = 0.86
)
class_colours <- c(
  "Transmission-context retained" = "#0072B2",
  "Ascertainment-change retained" = "#D55E00"
)
stream_labels <- c(
  age = "Age composition",
  serology = "Serology",
  testing = "Testing index",
  severe = "Severe outcomes"
)
stream_colours <- c(
  age = "#A6A6A6",
  serology = "#CC79A7",
  testing = "#0072B2",
  severe = "#D55E00"
)
precision_labels <- c(
  low_precision = "Lower",
  main = "Main",
  high_precision = "Higher"
)

assert_true(
  nrow(package_auc) == length(package_order) &&
    nrow(country_holdout) == length(package_order) &&
    setequal(package_auc$package_id, package_order) &&
    setequal(country_holdout$package_id, package_order),
  "Expected one grouped-CV and one whole-country-holdout row per displayed package."
)
roc_endpoints <- roc_curve |>
  arrange(package_id, point_order) |>
  group_by(package_id) |>
  summarise(
    starts_at_origin = first(false_positive_rate) == 0 &&
      first(true_positive_rate) == 0,
    ends_at_one = last(false_positive_rate) == 1 &&
      last(true_positive_rate) == 1,
    monotone_fpr = all(diff(false_positive_rate) >= 0),
    monotone_tpr = all(diff(true_positive_rate) >= 0),
    .groups = "drop"
  )
roc_auc_check <- roc_curve |>
  distinct(package_id, auc) |>
  inner_join(
    package_auc |> select(package_id, frozen_auc = auc),
    by = "package_id"
  )
assert_true(
  n_distinct(roc_curve$package_id) == length(package_order) &&
    setequal(roc_curve$package_id, package_order) &&
    all(roc_curve$validation_scheme == "grouped_five_fold_oof") &&
    all(roc_curve$false_positive_rate >= 0 & roc_curve$false_positive_rate <= 1) &&
    all(roc_curve$true_positive_rate >= 0 & roc_curve$true_positive_rate <= 1) &&
    all(roc_endpoints$starts_at_origin) &&
    all(roc_endpoints$ends_at_one) &&
    all(roc_endpoints$monotone_fpr) &&
    all(roc_endpoints$monotone_tpr) &&
    nrow(roc_auc_check) == length(package_order) &&
    max(abs(roc_auc_check$auc - roc_auc_check$frozen_auc)) < 1e-12,
  "Grouped-CV ROC coordinates are incomplete or disagree with the frozen AUC results."
)
assert_true(
  nrow(matched_ratios) == 720L &&
    n_distinct(matched_ratios$group_id) == 720L,
  "Expected exactly 720 matched country-draw ratio rows."
)
assert_true(
  n_distinct(matched_path$group_id) == 1L &&
    nrow(matched_path) == 72L &&
    n_distinct(matched_path$trajectory_class) == 2L,
  "The representative path must contain two 36-month matched trajectories."
)
assert_true(
  all(matched_path$max_absolute_notification_difference_all_720_pairs <= 1e-10),
  "Expected monthly notifications are not exactly matched in all 720 pairs."
)
assert_true(
  nrow(shapley_gain) == 4L &&
    setequal(shapley_gain$stream, names(stream_labels)),
  "Expected one order-averaged AUC gain for each optional stream."
)
assert_true(
  nrow(country_holdout_detail) == 54L &&
    n_distinct(country_holdout_detail$iso3) == 9L &&
    setequal(country_holdout_detail$package_id, package_order) &&
    !anyDuplicated(country_holdout_detail[c("iso3", "package_id")]),
  "Expected 54 unique country-by-package holdout results."
)
assert_true(
  nrow(precision_robustness) == 6L &&
    setequal(
      precision_robustness$package_id,
      c("notifications+testing", "notifications+severe")
    ) &&
    setequal(
      precision_robustness$measurement_design,
      names(precision_labels)
    ) &&
    !anyDuplicated(
      precision_robustness[c("measurement_design", "package_id")]
    ),
  "Expected testing and severe-outcome AUC under all three precision designs."
)
assert_true(
  all(is.finite(package_auc$auc)) &&
    all(is.finite(package_auc$auc_lower_95)) &&
    all(is.finite(package_auc$auc_upper_95)) &&
    all(package_auc$auc_lower_95 <= package_auc$auc) &&
    all(package_auc$auc <= package_auc$auc_upper_95) &&
    all(is.finite(country_holdout$auc)) &&
    all(is.finite(country_holdout_detail$auc)) &&
    all(is.finite(precision_robustness$auc)),
  "AUC estimates or intervals are invalid."
)
ratio_columns <- c(
  "infant_symptomatic_burden_ratio_2024",
  "effective_notification_ascertainment_ratio_2024"
)
assert_true(
  all(is.finite(as.matrix(matched_ratios[ratio_columns]))) &&
    all(as.matrix(matched_ratios[ratio_columns]) > 0),
  "Matched-pair burden or ascertainment ratios are invalid."
)

versions <- unique(c(
  package_auc$surveillance_design_version,
  roc_curve$surveillance_design_version,
  country_holdout$surveillance_design_version,
  matched_path$surveillance_design_version,
  matched_ratios$surveillance_design_version,
  shapley_gain$surveillance_design_version,
  country_holdout_detail$surveillance_design_version,
  precision_robustness$surveillance_design_version
))
assert_true(
  length(versions) == 1L,
  "Figure 2 inputs do not share one frozen surveillance-design version."
)

package_auc_lookup <- setNames(package_auc$auc, package_auc$package_id)
package_roc_labels <- setNames(
  sprintf(
    "%s (%.3f)",
    unname(package_labels_short[package_order]),
    package_auc_lookup[package_order]
  ),
  package_order
)
roc_plot <- roc_curve |>
  mutate(
    package_id = factor(package_id, levels = package_order)
  ) |>
  arrange(package_id, point_order)

ratio_long <- matched_ratios |>
  pivot_longer(
    cols = all_of(ratio_columns),
    names_to = "metric_id",
    values_to = "ratio"
  ) |>
  mutate(
    metric = recode(
      metric_id,
      infant_symptomatic_burden_ratio_2024 = "Infant symptomatic burden",
      effective_notification_ascertainment_ratio_2024 = "Effective ascertainment"
    ),
    metric = factor(
      metric,
      levels = c("Effective ascertainment", "Infant symptomatic burden")
    ),
    signal = recode(
      metric_id,
      infant_symptomatic_burden_ratio_2024 = "Transmission-context retained",
      effective_notification_ascertainment_ratio_2024 = "Ascertainment-change retained"
    )
  )
ratio_country <- ratio_long |>
  group_by(metric, signal, iso3) |>
  summarise(ratio = median(ratio), .groups = "drop")
ratio_summary <- ratio_long |>
  group_by(metric, signal) |>
  summarise(median = median(ratio), .groups = "drop") |>
  mutate(
    label = if_else(
      metric == "Infant symptomatic burden",
      sprintf("%.2f×", median),
      sprintf("%.3f×", median)
    ),
    label_x = median
  )

shapley_plot <- shapley_gain |>
  mutate(
    stream_label = factor(
      stream_labels[stream],
      levels = unname(stream_labels[c("age", "serology", "testing", "severe")])
    ),
    value_label = sprintf("%.3f", shapley_auc_gain)
  )

country_plot <- bind_rows(
  country_holdout_detail |>
    transmute(package_id, iso3, auc),
  country_holdout |>
    transmute(package_id, iso3 = "Pooled", auc)
) |>
  mutate(
    iso3 = factor(
      iso3,
      levels = c(
        "AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA",
        "Pooled"
      )
    ),
    package_label = factor(
      package_labels_short[package_id],
      levels = rev(unname(package_labels_short[package_order]))
    ),
    value_label = sprintf("%.3f", auc),
    text_colour = holdout_text_colour(auc)
  )

precision_plot <- precision_robustness |>
  mutate(
    precision = factor(
      precision_labels[measurement_design],
      levels = unname(precision_labels)
    ),
    stream = recode(
      package_id,
      `notifications+testing` = "Testing index",
      `notifications+severe` = "Severe outcomes"
    ),
    stream = factor(stream, levels = c("Testing index", "Severe outcomes")),
    value_label = sprintf("%.3f", auc),
    label_y = auc + if_else(stream == "Testing index", 0.018, -0.018)
  )

ink <- "#1B1B1B"
neutral_mid <- "#767676"

theme_figure4 <- function(base_size = 6.3) {
  theme_nature(base_size = base_size) +
    theme(
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.tag = element_text(size = 8, face = "bold", colour = ink),
      plot.tag.position = c(0, 1),
      axis.line = element_line(linewidth = 0.28, colour = ink),
      axis.ticks = element_line(linewidth = 0.24, colour = ink),
      axis.ticks.length = unit(1.1, "mm"),
      legend.position = "bottom",
      legend.justification = "left",
      legend.box.just = "left",
      legend.key.width = unit(4.0, "mm"),
      legend.key.height = unit(2.6, "mm"),
      plot.margin = margin(4, 4, 4, 4)
    )
}

p_a <- ggplot() +
  geom_line(
    data = filter(matched_path, trajectory_class == "Transmission-context retained"),
    aes(x = date, y = expected_notifications, colour = trajectory_class),
    linewidth = 0.90
  ) +
  geom_line(
    data = filter(matched_path, trajectory_class == "Ascertainment-change retained"),
    aes(x = date, y = expected_notifications, colour = trajectory_class),
    linewidth = 0.58,
    linetype = "22"
  ) +
  annotate(
    "label",
    x = as.Date("2024-08-01"),
    y = max(matched_path$expected_notifications) * 0.8,
    label = "Maximum absolute\nmonthly difference <1e-10",
    family = figure_font_family,
    size = 1.48,
    lineheight = 0.9,
    linewidth = 0.18,
    label.padding = unit(0.9, "mm"),
    fill = "white",
    colour = ink
  ) +
  scale_colour_manual(values = class_colours, name = NULL) +
  scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y",
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  scale_y_log10(
    labels = label_number(scale_cut = cut_short_scale()),
    expand = expansion(mult = c(0.04, 0.10))
  ) +
  labs(
    tag = "a",
    x = NULL,
    y = "Expected monthly notifications"
  ) +
  theme_figure4() +
  theme(
    legend.text = element_text(size = 4.9),
    legend.spacing.x = unit(0.9, "mm"),
    panel.grid = element_blank()
  ) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

p_b <- ggplot(ratio_long, aes(x = ratio, y = metric)) +
  geom_vline(
    xintercept = 1,
    colour = neutral_mid,
    linewidth = 0.32,
    linetype = "dotted"
  ) +
  geom_point(
    aes(colour = signal),
    alpha = 0.11,
    size = 0.68,
    position = position_jitter(height = 0.07, width = 0, seed = 20260728),
    show.legend = FALSE
  ) +
  geom_boxplot(
    aes(colour = signal),
    width = 0.32,
    outlier.shape = NA,
    fill = "white",
    linewidth = 0.46,
    show.legend = FALSE
  ) +
  geom_point(
    data = ratio_country,
    aes(colour = signal),
    shape = 5,
    size = 1.35,
    stroke = 0.48,
    show.legend = FALSE
  ) +
  geom_text(
    data = ratio_summary,
    aes(x = label_x, label = label),
    family = figure_font_family,
    fontface = "bold",
    hjust = 0.5,
    position = position_nudge(y = 0.33),
    size = 1.65,
    colour = ink
  ) +
  scale_colour_manual(values = class_colours) +
  scale_x_log10(
    limits = range(c(ratio_long$ratio, ratio_summary$label_x)) * c(0.8, 1.15),
    breaks = c(0.03, 0.1, 0.3, 1, 3, 10, 30),
    labels = c("0.03", "0.1", "0.3", "1", "3", "10", "30"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    tag = "b",
    x = "Matched-pair ratio (log scale)",
    y = NULL
  ) +
  theme_figure4() +
  theme(
    axis.text.y = element_text(size = 5.0),
    axis.text.x = element_text(size = 4.9),
    panel.grid = element_blank(),
    legend.position = "none"
  )

p_c <- ggplot(
  roc_plot,
  aes(
    x = false_positive_rate,
    y = true_positive_rate,
    colour = package_id,
    linewidth = package_id,
    group = package_id
  )
) +
  geom_abline(
    slope = 1,
    intercept = 0,
    colour = neutral_mid,
    linewidth = 0.34,
    linetype = "dotted"
  ) +
  geom_step(direction = "vh", alpha = 0.97) +
  scale_colour_manual(
    values = package_colours,
    breaks = package_order,
    labels = package_roc_labels,
    name = NULL
  ) +
  scale_linewidth_manual(
    values = package_linewidths,
    breaks = package_order,
    guide = "none"
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.5, 0.75, 1),
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.5, 0.75, 1),
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_equal(clip = "off") +
  labs(
    tag = "c",
    x = "False-positive rate",
    y = "True-positive rate"
  ) +
  theme_figure4() +
  guides(
    colour = guide_legend(
      ncol = 2,
      byrow = TRUE,
      override.aes = list(linewidth = 0.9, alpha = 1)
    )
  ) +
  theme(
    axis.text = element_text(size = 4.8),
    legend.text = element_text(size = 4.55),
    legend.key.width = unit(4.4, "mm"),
    legend.spacing.x = unit(0.8, "mm"),
    legend.spacing.y = unit(0.2, "mm"),
    legend.margin = margin(t = 0.5),
    panel.grid = element_blank()
  )

p_d <- ggplot(
  shapley_plot,
  aes(x = shapley_auc_gain, y = stream_label, colour = stream)
) +
  geom_segment(
    aes(x = 0, xend = shapley_auc_gain, yend = stream_label),
    linewidth = 1.15,
    alpha = 0.45
  ) +
  geom_point(size = 2.1) +
  geom_text(
    aes(label = value_label),
    hjust = -0.18,
    family = figure_font_family,
    fontface = "bold",
    size = 1.55,
    colour = ink
  ) +
  scale_colour_manual(values = stream_colours, guide = "none") +
  scale_x_continuous(
    limits = c(min(0, min(shapley_plot$shapley_auc_gain)) - 0.005, max(0.255, max(shapley_plot$shapley_auc_gain) * 1.15)),
    breaks = c(0, 0.1, 0.2),
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    tag = "d",
    x = "Order-averaged marginal AUC gain",
    y = NULL
  ) +
  theme_figure4() +
  theme(
    axis.text.y = element_text(size = 5.1),
    panel.grid = element_blank(),
    legend.position = "none"
  )

p_e <- ggplot(country_plot, aes(x = iso3, y = package_label, fill = auc)) +
  geom_tile(colour = "white", linewidth = 0.42) +
  geom_vline(xintercept = 9.5, colour = "white", linewidth = 1.25) +
  geom_vline(xintercept = 9.5, colour = neutral_mid, linewidth = 0.25) +
  geom_text(
    aes(label = value_label, colour = text_colour),
    family = figure_font_family,
    fontface = "bold",
    size = 1.38,
    show.legend = FALSE
  ) +
  scale_colour_identity() +
  scale_fill_gradientn(
    colours = holdout_palette,
    values = rescale(c(0.00, 0.50, 1.00)),
    limits = c(0.00, 1.00),
    oob = squish,
    breaks = c(0.0, 0.5, 1.0),
    labels = label_number(accuracy = 0.1),
    name = "Whole-country holdout AUC"
  ) +
  labs(
    tag = "e",
    x = "Held-out country / pooled",
    y = NULL
  ) +
  theme_figure4() +
  guides(
    fill = guide_colourbar(
      title.position = "top",
      title.hjust = 0,
      barwidth = unit(21, "mm"),
      barheight = unit(2.0, "mm"),
      ticks.colour = ink,
      frame.colour = NA
    )
  ) +
  theme(
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 4.8),
    axis.text.y = element_text(size = 4.9),
    panel.grid = element_blank(),
    legend.title = element_text(size = 4.9, face = "plain"),
    legend.text = element_text(size = 4.6),
    legend.margin = margin(t = 0)
  )

p_f <- ggplot(
  precision_plot,
  aes(x = precision, y = auc, group = stream, colour = stream, shape = stream)
) +
  geom_hline(
    yintercept = 0.8,
    colour = ink,
    linewidth = 0.35,
    linetype = "dashed"
  ) +
  geom_line(linewidth = 0.68) +
  geom_point(size = 2.05, stroke = 0.58) +
  geom_text(
    aes(y = label_y, label = value_label),
    vjust = 0.5,
    family = figure_font_family,
    size = 1.42,
    show.legend = FALSE
  ) +
  scale_colour_manual(
    values = c("Testing index" = "#0072B2", "Severe outcomes" = "#D55E00"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Testing index" = 16, "Severe outcomes" = 15),
    name = NULL
  ) +
  scale_y_continuous(
    limits = c(0.45, 1.005),
    breaks = c(0.5, 0.7, 0.9, 1.0),
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    tag = "f",
    x = "Measurement precision",
    y = "Area under the ROC curve"
  ) +
  theme_figure4() +
  guides(
    colour = guide_legend(nrow = 2, byrow = TRUE),
    shape = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  theme(
    axis.text.x = element_text(size = 4.9),
    legend.text = element_text(size = 4.8),
    legend.spacing.y = unit(0.2, "mm"),
    legend.margin = margin(t = 0),
    panel.grid = element_blank()
  )

top_row <- p_a + p_b + p_c +
  plot_layout(widths = c(0.92, 1.02, 1.36))
bottom_row <- p_d + p_e + p_f +
  plot_layout(widths = c(0.82, 1.48, 0.96))
figure <- top_row / bottom_row +
  plot_layout(heights = c(1.00, 1.02)) +
  plot_annotation(
    theme = theme(
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(2, 2, 2, 2)
    )
  )

width_mm <- 183
height_mm <- 156
width_in <- width_mm / 25.4
height_in <- height_mm / 25.4
output_stem <- fig_path("supplement_figure_4_surveillance_auc_diagnostics")

grDevices::cairo_pdf(
  paste0(output_stem, ".pdf"),
  width = width_in,
  height = height_in,
  family = "sans",
  bg = "white",
  onefile = TRUE
)
print(figure)
grDevices::dev.off()

ragg::agg_png(
  paste0(output_stem, ".png"),
  width = width_in,
  height = height_in,
  units = "in",
  res = 600,
  background = "white"
)
print(figure)
grDevices::dev.off()

export_paths <- paste0(output_stem, c(".png", ".pdf"))
assert_true(
  all(file.exists(export_paths)) &&
    all(file.info(export_paths)$size > 1000),
  "One or more Figure 2 exports are missing or unexpectedly small."
)

message(
  "Wrote legacy six-panel surveillance-design diagnostic exports for version ",
  versions[[1]],
  "."
)
