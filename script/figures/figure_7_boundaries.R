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
# Core conclusion: measurement precision, latent contrast and event density set
# practical discrimination boundaries. Loading-rank and cross-loading checks
# remain supplementary diagnostics rather than main-text results. Panel a
# pairs pooled scores with country medians so that a pooled trend is not read as
# a universal country effect.

atten <- read_csv_checked(
  fd_path("figure_4_attenuation.csv"),
  required = c(
    "retained_between_class_contrast", "package_id", "grouped_cv_auc",
    "grouped_cv_auc_lower_95", "grouped_cv_auc_upper_95"
  )
) |>
  filter(package_id %in% c("notifications+testing", "notifications+severe", "notifications+testing+severe"))
severe <- read_csv_checked(
  fd_path("figure_7_severe_resource_frontier.csv"),
  required = c(
    "severe_baseline_events_per_country_year", "severe_annual_log_rate_drift_sd",
    "auc", "auc_lower_95", "auc_upper_95", "meets_auc_0_80", "n_bootstrap"
  )
)
precision <- read_csv_checked(
  fd_path("surveillance_precision_frontier.csv"),
  required = c(
    "frontier_type", "parameter_value", "auc", "auc_lower_95", "auc_upper_95",
    "operational_success", "conservative_success"
  )
) |>
  filter(frontier_type %in% c("testing_log_sd", "severe_annual_log_sd"))
stopifnot(
  nrow(severe) == 21L,
  all(severe$n_bootstrap == 2000L),
  all(is.finite(severe$auc))
)

package_labels <- c(
  `notifications+testing` = "Testing",
  `notifications+severe` = "Severe",
  `notifications+testing+severe` = "Testing–severe"
)
package_colours <- c("Testing" = clr("blue"), "Severe" = clr("orange"), "Testing–severe" = clr("black"))
atten <- atten |> mutate(package = unname(package_labels[package_id]))

green_scale <- c("#F1F4F2", "#CDE4DA", "#79BDA4", "#176A50")
blue_scale <- c("#F1F4F7", "#D0DEEB", "#76A3CC", "#235F98")
heatmap_text_colour <- function(values, colours, limits) {
  fills <- scales::gradient_n_pal(colours)(scales::rescale(squish(values, limits), from = limits))
  luminance <- function(colours) {
    rgb <- t(grDevices::col2rgb(colours)) / 255
    linear <- ifelse(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)^2.4)
    as.vector(linear %*% c(0.2126, 0.7152, 0.0722))
  }
  background <- luminance(fills)
  white_contrast <- 1.05 / (background + 0.05)
  dark_contrast <- (background + 0.05) / 0.05
  stopifnot(all(pmax(white_contrast, dark_contrast) >= 4.5))
  ifelse(white_contrast > dark_contrast, "white", "#000000")
}
format_boundary_auc <- function(auc) {
  ifelse(abs(auc - 0.8) < 0.015, sprintf("%.3f", auc), sprintf("%.2f", auc))
}

contrast <- read_csv_checked(
  file.path("manuscript", "tables", "nc_equal_contrast_comparison.csv"),
  required = c("target_standardised_contrast", "stream", "whole_country_holdout_auc",
               "holdout_country_auc_median", "n_country_structures", "measurement_noise_sd")
)
stopifnot(
  nrow(contrast) == 8L, all(contrast$n_country_structures == 9L),
  all(contrast$measurement_noise_sd == 0.20),
  setequal(contrast$stream, c("testing", "severe")),
  all(vapply(split(contrast$target_standardised_contrast, contrast$stream),
             function(x) setequal(x, c(0.5, 1, 2, 4)), logical(1)))
)
contrast_long <- contrast |>
  mutate(package = if_else(stream == "testing", "Testing", "Severe")) |>
  pivot_longer(c(whole_country_holdout_auc, holdout_country_auc_median),
               names_to = "summary", values_to = "auc") |>
  mutate(summary = recode(summary, whole_country_holdout_auc = "Pooled",
                           holdout_country_auc_median = "Country median"))
stopifnot(all(is.finite(contrast_long$auc)), all(between(contrast_long$auc, 0, 1)))
p_contrast <- ggplot(contrast_long, aes(target_standardised_contrast, auc,
                                       colour = package, linetype = summary, shape = summary)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = clr("grey"), linewidth = 0.3) +
  geom_line(linewidth = 0.62) +
  geom_point(size = 1.55, fill = "white") +
  geom_text(
    data = contrast_long |> filter(summary == "Pooled", target_standardised_contrast == 4),
    aes(label = sprintf("%.3f", auc)), hjust = -0.18, size = 1.6,
    family = figure_font_family, show.legend = FALSE
  ) +
  scale_colour_manual(values = package_colours[c("Testing", "Severe")]) +
  scale_linetype_manual(values = c(Pooled = "solid", `Country median` = "22")) +
  scale_shape_manual(values = c(Pooled = 16, `Country median` = 1)) +
  scale_x_log10(breaks = c(0.5, 1, 2, 4), labels = c("0.5", "1", "2", "4")) +
  scale_y_continuous(breaks = seq(0.4, 1, 0.2)) +
  coord_cartesian(xlim = c(0.46, 6.1), ylim = c(0.34, 1.015)) +
  labs(tag = "a", x = "Median latent contrast / reference SD (0.20)",
       y = "Country-transfer AUC", colour = NULL, linetype = NULL, shape = NULL) +
  guides(colour = guide_legend(order = 1, nrow = 1),
         linetype = guide_legend(order = 2, nrow = 1), shape = guide_legend(order = 2, nrow = 1)) +
  theme_nature(grid = "y") +
  theme(legend.position = "bottom", legend.box = "vertical")

p_a <- ggplot(atten, aes(retained_between_class_contrast, grouped_cv_auc, colour = package, fill = package)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = clr("grey")) +
  geom_hline(yintercept = 0.8, linetype = "22", linewidth = 0.3, colour = clr("dark_grey")) +
  geom_ribbon(aes(ymin = grouped_cv_auc_lower_95, ymax = grouped_cv_auc_upper_95), alpha = 0.1, colour = NA) +
  geom_line(linewidth = 0.68) +
  geom_point(size = 1.45) +
  annotate(
    "text", x = 0.02, y = 0.81, label = "AUC 0.80",
    hjust = 0, vjust = 0, size = 1.42, family = figure_font_family, colour = clr("dark_grey")
  ) +
  scale_colour_manual(values = package_colours) +
  scale_fill_manual(values = package_colours) +
  scale_x_continuous(labels = label_percent(), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  scale_y_continuous(breaks = c(0.5, 0.6, 0.7, 0.8, 0.9, 1)) +
  coord_cartesian(ylim = c(min(0.46, atten$grouped_cv_auc_lower_95) - 0.005, 1.005)) +
  labs(
    tag = "b", x = "Retained between-class contrast", y = "Grouped-CV AUC",
    colour = NULL, fill = NULL
  ) +
  theme_nature(grid = "y") +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
  theme(legend.position = "bottom")

event_levels <- sort(unique(severe$severe_baseline_events_per_country_year))
drift_levels <- sort(unique(severe$severe_annual_log_rate_drift_sd), decreasing = TRUE)
severe_plot <- severe |>
  mutate(
    events = factor(severe_baseline_events_per_country_year, levels = event_levels),
    drift = factor(severe_annual_log_rate_drift_sd, levels = drift_levels),
    meets = as.logical(meets_auc_0_80),
    label = paste0(format_boundary_auc(auc), "\n", if_else(meets, "≥0.80", "<0.80")),
    text_colour = heatmap_text_colour(auc, green_scale, c(0.5, 1))
  )
stopifnot(all(severe_plot$meets == (severe_plot$auc >= 0.8)))
p_c <- ggplot(severe_plot, aes(events, drift, fill = auc)) +
  geom_tile(colour = "white", linewidth = 0.68) +
  geom_tile(data = severe_plot |> filter(!meets), fill = NA, colour = "#924B32", linewidth = 0.68) +
  geom_text(
    aes(label = label, colour = text_colour), size = 1.5, lineheight = 0.9,
    family = figure_font_family
  ) +
  scale_fill_gradientn(
    colours = green_scale,
    limits = c(0.5, 1), breaks = c(0.5, 0.7, 0.85, 1), oob = squish
  ) +
  scale_colour_identity() +
  labs(
    tag = "c", x = "Baseline expected infant events\nper sentinel catchment-year",
    y = "Annual log-rate-drift SD", fill = "AUC"
  ) +
  guides(fill = guide_colorbar(barheight = unit(34, "mm"), barwidth = unit(3.2, "mm"))) +
  theme_nature() +
  theme(
    axis.ticks = element_blank(), axis.line = element_blank(),
    axis.text.x = element_text(size = figure_text$axis_text), legend.position = "right"
  )

precision_plot <- precision |>
  mutate(
    stream = recode(
      frontier_type,
      testing_log_sd = "Testing-index error",
      severe_annual_log_sd = "Severe-rate drift"
    )
  )
precision_cols <- c("Testing-index error" = clr("blue"), "Severe-rate drift" = clr("orange"))
p_d <- ggplot(precision_plot, aes(parameter_value, auc, colour = stream, fill = stream)) +
  geom_hline(yintercept = 0.8, linetype = "22", linewidth = 0.35, colour = clr("dark_grey")) +
  geom_ribbon(aes(ymin = auc_lower_95, ymax = auc_upper_95), alpha = 0.11, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.35) +
  scale_colour_manual(values = precision_cols) +
  scale_fill_manual(values = precision_cols) +
  scale_x_log10(breaks = c(0.1, 0.2, 0.5, 1, 2, 4), labels = label_number()) +
  scale_y_continuous(breaks = c(0.5, 0.6, 0.7, 0.8, 0.9, 1)) +
  coord_cartesian(ylim = c(min(0.46, precision_plot$auc_lower_95) - 0.005, 1.005)) +
  labs(
    tag = "d", x = "Log-scale measurement-noise SD", y = "Grouped-CV AUC",
    colour = NULL, fill = NULL
  ) +
  theme_nature(grid = "both") +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
  theme(legend.position = "bottom")

mapping <- read_csv_checked('manuscript/tables/primary_severe_mapping_sensitivity.csv', required=c('severe_contrast_factor','drift_multiplier','country_macro_auc','lower_95','upper_95')) |>
 mutate(drift=factor(if_else(drift_multiplier==1,'Native drift (SD 0.736)','Doubled drift (SD 1.471)')))
p_contrast <- ggplot(mapping,aes(severe_contrast_factor,country_macro_auc,colour=drift,fill=drift,linetype=drift))+
 geom_hline(yintercept=.5,linetype='dotted',colour=clr('grey'))+
 geom_ribbon(aes(ymin=lower_95,ymax=upper_95),alpha=.10,colour=NA)+geom_line(linewidth=.65)+geom_point(size=1.6)+
 scale_colour_manual(values=c(clr('blue'),clr('orange')))+scale_fill_manual(values=c(clr('blue'),clr('orange')))+
 scale_x_continuous(breaks=c(0,.25,.5,.75,1),labels=label_percent())+
 scale_y_continuous(limits=c(.45,.85),breaks=seq(.5,.8,.1))+
 labs(tag='a',x='Retained severe-event log contrast',y='Primary country-macro AUC',colour=NULL,fill=NULL,linetype=NULL)+
 theme_nature(grid='y')+theme(legend.position='bottom',legend.text=element_text(size=5.5))+guides(colour=guide_legend(ncol=1))

figure <- (p_contrast | p_a) / (p_c | p_d) +
  plot_layout(heights = c(1.02, 0.95), widths = c(0.94, 1.06))

stem <- "figure_3_measurement_boundaries"
width <- 183 / 25.4
height <- 125 / 25.4

grDevices::cairo_pdf(fig_path(paste0(stem, ".pdf")), width = width, height = height, family = "sans")
print(figure)
dev.off()
ragg::agg_png(fig_path(paste0(stem, ".png")), width = width, height = height, units = "in", res = 600, background = "white")
print(figure)
dev.off()
message("Wrote main Figure 3 with contrast, attenuation, event-density and measurement-noise boundaries.")
