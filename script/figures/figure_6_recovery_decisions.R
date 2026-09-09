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
# Core conclusion: process-linked streams differ in their conditional
# information, and annual auxiliary exports delay signed
# latent-contrast recovery; linked packages improve the miss–false-alarm
# trade-off once process-specific streams are available.
# Archetype: quantitative grid; panels b and c are the transport and timeliness
# evidence.

grouped <- read_csv_checked(
  file.path("manuscript", "tables", "surveillance_design_key_results.csv"),
  required = c(
    "measurement_design", "package_id", "auc", "auc_lower_95", "auc_upper_95",
    "n_bootstrap", "n_country_draw_groups", "n_observations", "surveillance_design_version"
  )
) |>
  filter(measurement_design == "main")
holdout <- read_csv_checked(
  file.path("manuscript", "tables", "surveillance_design_country_holdout.csv"),
  required = c(
    "measurement_design", "package_id", "validation_scheme", "auc",
    "median_country_auc", "minimum_country_auc", "maximum_country_auc", "n_countries",
    "surveillance_design_version"
  )
) |>
  filter(measurement_design == "main", validation_scheme %in% c("whole_country_holdout", "leave_one_country_out"))
holdout_detail <- read_csv_checked(
  fd_path("figure_4_country_holdout_auc.csv"),
  required = c("measurement_design", "package_id", "iso3", "validation_scheme", "auc", "surveillance_design_version")
) |>
  filter(measurement_design == "main", validation_scheme %in% c("whole_country_holdout", "leave_one_country_out"))
decisions <- read_csv_checked(
  fd_path("figure_6_decision_operating_points.csv"),
  required = c(
    "aggregation", "package_id", "threshold_policy", "missed_transmission_rate",
    "false_alarm_rate", "missed_transmission_rate_lower_95",
    "missed_transmission_rate_upper_95", "false_alarm_rate_lower_95",
    "false_alarm_rate_upper_95", "n_bootstrap", "surveillance_design_version"
  )
) |>
  filter(aggregation == "all_holdouts")
timeliness <- read_csv_checked(
  file.path("manuscript", "tables", "continuous_attribution_timeliness_summary.csv"),
  required = c(
    "horizon_months", "package_label", "spearman_rho", "auc",
    "available_streams_at_horizon", "surveillance_design_version"
  )
) |>
  filter(package_label %in% c("complete", "notifications"))

input_versions <- unique(c(
  grouped$surveillance_design_version,
  holdout$surveillance_design_version,
  holdout_detail$surveillance_design_version,
  decisions$surveillance_design_version,
  timeliness$surveillance_design_version
))
stopifnot(
  length(input_versions) == 1L,
  !is.na(input_versions),
  nzchar(input_versions)
)

package_order <- c(
  "notifications",
  "notifications+age",
  "notifications+serology",
  "notifications+testing",
  "notifications+severe",
  "notifications+age+testing+severe+serology"
)
package_labels <- c(
  notifications = "Notifications",
  `notifications+age` = "+ age composition",
  `notifications+serology` = "+ two-round serology",
  `notifications+testing` = "+ testing-linked",
  `notifications+severe` = "+ severe outcomes",
  `notifications+age+testing+severe+serology` = "Complete package"
)
package_colours <- c(
  notifications = "#707274",
  `notifications+age` = "#A9ACAF",
  `notifications+serology` = clr("purple"),
  `notifications+testing` = clr("blue"),
  `notifications+severe` = clr("orange"),
  `notifications+age+testing+severe+serology` = clr("black")
)

stopifnot(
  setequal(grouped$package_id, package_order),
  setequal(holdout$package_id, package_order),
  nrow(holdout_detail) == 54L,
  all(grouped$n_bootstrap == 2000L),
  nrow(timeliness) == 10L,
  all(decisions$n_bootstrap == 2000L)
)

performance <- bind_rows(
  grouped |>
    transmute(
      package_id, validation = "Grouped five-fold",
      estimate = auc, lower = auc_lower_95, upper = auc_upper_95
    ),
  holdout |>
    transmute(
      package_id, validation = "Country-structure holdout",
      estimate = auc, lower = NA_real_, upper = NA_real_
    )
) |>
  mutate(
    package_id = factor(package_id, levels = rev(package_order)),
    validation = factor(validation, levels = c("Grouped five-fold", "Country-structure holdout"))
  )

p_a <- ggplot(performance, aes(estimate, package_id, colour = validation, shape = validation)) +
  geom_vline(xintercept = 0.5, linetype = "dotted", colour = clr("grey")) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper), orientation = "y", width = 0,
    position = position_dodge(width = 0.42), linewidth = 0.46, na.rm = TRUE
  ) +
  geom_point(position = position_dodge(width = 0.42), size = 2.05, stroke = 0.55) +
  scale_x_continuous(breaks = seq(0, 1, 0.1)) +
  coord_cartesian(xlim = c(min(0.45, performance$estimate, performance$lower, na.rm = TRUE) - 0.01, 1.005)) +
  scale_y_discrete(labels = package_labels) +
  scale_colour_manual(values = c("Grouped five-fold" = clr("blue"), "Country-structure holdout" = clr("orange"))) +
  scale_shape_manual(values = c("Grouped five-fold" = 16, "Country-structure holdout" = 23)) +
  labs(
    tag = "a", x = "AUC", y = NULL, colour = NULL, shape = NULL
  ) +
  theme_nature(grid = "x") +
  theme(legend.position = "top", axis.ticks.y = element_blank())

holdout_limits <- c(floor(min(0.45, holdout_detail$auc) * 10) / 10, 1)
holdout_palette <- c("#EFEFF0", "#C8D9E8", "#78A6CF", "#235F98")
holdout_text_colour <- function(auc) {
  fills <- scales::gradient_n_pal(holdout_palette)(scales::rescale(squish(auc, holdout_limits), from = holdout_limits))
  rgb <- t(grDevices::col2rgb(fills)) / 255
  linear <- ifelse(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)^2.4)
  luminance <- as.vector(linear %*% c(0.2126, 0.7152, 0.0722))
  white_contrast <- 1.05 / (luminance + 0.05)
  black_contrast <- (luminance + 0.05) / 0.05
  stopifnot(all(pmax(white_contrast, black_contrast) >= 4.5))
  ifelse(white_contrast > black_contrast, "white", "#000000")
}
holdout_heat <- bind_rows(
  holdout_detail |> transmute(package_id, column = iso3, auc),
  holdout |> transmute(package_id, column = "Median", auc = median_country_auc)
) |>
  mutate(
    package_id = factor(package_id, levels = rev(package_order)),
    column = factor(column, levels = c(pilot_iso3, "Median")),
    value_label = sprintf("%.3f", auc),
    text_colour = holdout_text_colour(auc)
  )
country_labels <- c(
  AUS = "AUS", BRA = "BRA", CHN = "CHN", GBR = "GBR", JPN = "JPN",
  NZL = "NZL", SWE = "SWE", THA = "THA", USA = "USA", Median = "Median"
)
p_b <- ggplot(holdout_heat, aes(column, package_id, fill = auc)) +
  geom_tile(colour = "white", linewidth = 0.45) +
  geom_text(aes(label = value_label, colour = text_colour), size = 1.48, family = figure_font_family) +
  scale_fill_gradientn(
    colours = holdout_palette,
    limits = holdout_limits, breaks = sort(unique(c(holdout_limits[[1]], 0.5, 0.8, 1))), oob = squish
  ) +
  scale_colour_identity() +
  scale_x_discrete(labels = country_labels) +
  scale_y_discrete(labels = package_labels) +
  labs(
    tag = "b", x = "Held-out country structure", y = NULL, fill = "AUC"
  ) +
  theme_nature() +
  theme(
    axis.ticks = element_blank(), axis.line = element_blank(),
    axis.text.x = element_text(size = figure_text$axis_text_dense),
    axis.text.y = element_text(size = figure_text$axis_text_dense),
    legend.position = "right", legend.key.height = unit(7, "mm")
  )

timeliness_plot <- timeliness |>
  transmute(
    horizon_months,
    package = recode(
      package_label,
      notifications = "Notifications",
      complete = "Complete package"
    ),
    `Spearman rho` = spearman_rho,
    `Sign AUC` = auc
  ) |>
  pivot_longer(
    cols = c("Spearman rho", "Sign AUC"),
    names_to = "metric",
    values_to = "value"
  ) |>
  mutate(
    package = factor(package, levels = c("Notifications", "Complete package")),
    metric = factor(metric, levels = c("Spearman rho", "Sign AUC"))
  )
timeliness_baselines <- tibble::tibble(
  metric = factor(c("Spearman rho", "Sign AUC"), levels = c("Spearman rho", "Sign AUC")),
  baseline = c(0, 0.5)
)
timeliness_colours <- c("Notifications" = "#707274", "Complete package" = clr("black"))
p_c <- ggplot(timeliness_plot, aes(horizon_months, value, colour = package, shape = package)) +
  geom_hline(
    data = timeliness_baselines,
    aes(yintercept = baseline),
    inherit.aes = FALSE, linetype = "dotted", colour = clr("grey"), linewidth = 0.35
  ) +
  geom_vline(xintercept = 12, linetype = "22", linewidth = 0.3, colour = clr("dark_grey")) +
  geom_line(linewidth = 0.58) +
  geom_point(size = 1.45, stroke = 0.55) +
  facet_wrap(~metric, nrow = 1) +
  scale_colour_manual(values = timeliness_colours) +
  scale_shape_manual(values = c("Notifications" = 16, "Complete package" = 15)) +
  scale_x_continuous(breaks = c(3, 6, 12, 24, 36), limits = c(2, 37)) +
  scale_y_continuous(breaks = c(0, 0.5, 1), limits = c(-0.08, 1.02)) +
  labs(
    tag = "c", x = "Surveillance horizon (months)", y = "Held-out performance",
    colour = NULL, shape = NULL
  ) +
  theme_nature(grid = "y") +
  theme(
    legend.position = "top",
    strip.text = element_text(size = figure_text$strip_dense),
    panel.spacing.x = unit(3, "mm")
  )

decision_packages <- c(
  "notifications", "notifications+testing", "notifications+severe", "notifications+testing+severe"
)
decision_labels <- c(
  notifications = "Notifications", `notifications+testing` = "Testing",
  `notifications+severe` = "Severe", `notifications+testing+severe` = "Testing–severe"
)
decision_colours <- c(
  notifications = "#707274", `notifications+testing` = clr("blue"),
  `notifications+severe` = clr("orange"), `notifications+testing+severe` = clr("black")
)
decision_plot <- decisions |>
  filter(package_id %in% decision_packages) |>
  mutate(
    package_id = factor(package_id, levels = decision_packages),
    threshold_policy = factor(
      threshold_policy,
      levels = c("sensitivity", "youden", "specificity"),
      labels = c("Sensitivity ≥0.90", "Maximum Youden", "Specificity ≥0.90")
    )
  )
p_d <- ggplot(
  decision_plot,
  aes(false_alarm_rate, missed_transmission_rate, colour = package_id, shape = threshold_policy, group = package_id)
) +
  annotate(
    "rect", xmin = 0, xmax = 0.2, ymin = 0, ymax = 0.2,
    fill = "#E2F0EB", colour = NA, alpha = 0.65
  ) +
  geom_path(linewidth = 0.48, alpha = 0.65) +
  geom_errorbar(
    aes(xmin = false_alarm_rate_lower_95, xmax = false_alarm_rate_upper_95),
    orientation = "y", width = 0, linewidth = 0.3, alpha = 0.32
  ) +
  geom_errorbar(
    aes(ymin = missed_transmission_rate_lower_95, ymax = missed_transmission_rate_upper_95),
    width = 0, linewidth = 0.3, alpha = 0.32
  ) +
  geom_point(size = 2.0, stroke = 0.55, fill = "white") +
  scale_colour_manual(values = decision_colours, labels = decision_labels) +
  scale_shape_manual(values = c("Sensitivity ≥0.90" = 24, "Maximum Youden" = 21, "Specificity ≥0.90" = 22)) +
  scale_x_continuous(labels = label_percent(), breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1)) +
  scale_y_continuous(labels = label_percent(), breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1)) +
  labs(
    tag = "d", x = "False transmission-context alarm rate",
    y = "Missed transmission-context scenario rate",
    colour = NULL, shape = NULL
  ) +
  theme_nature(grid = "both") +
  theme(
    legend.position = "bottom", legend.box = "vertical",
    legend.text = element_text(size = figure_text$legend_text_dense),
    plot.margin = margin(3, 4, 3, 5)
  )

figure <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1.02, 1), widths = c(0.88, 1.12))

stem <- "figure_6_recovery_decisions"
width <- 183 / 25.4
height <- 153 / 25.4

ragg::agg_png(supplement_fig_path(paste0(stem, ".png")), width = width, height = height, units = "in", res = 600, background = "white")
print(figure)
dev.off()

message(
  "Wrote Supplementary Figure 25 as PNG from surveillance-design version ",
  input_versions[[1]],
  "."
)
