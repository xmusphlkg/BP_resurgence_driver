#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(stringr)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Run this file with Rscript so the project root can be resolved.", call. = FALSE)
}
script_file <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_file), "..", ".."), mustWork = TRUE)
setwd(repo_root)

source(file.path("script", "figures", "_common.R"))

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

holdout_palette <- c("#EEEEEE", "#B7D8E8", "#0072B2")
holdout_text_colour <- function(auc, limits) {
  fills <- scales::gradient_n_pal(holdout_palette)(scales::rescale(squish(auc, limits), from = limits))
  rgb <- t(grDevices::col2rgb(fills)) / 255
  linear <- ifelse(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)^2.4)
  luminance <- as.vector(linear %*% c(0.2126, 0.7152, 0.0722))
  white_contrast <- 1.05 / (luminance + 0.05)
  black_contrast <- (luminance + 0.05) / 0.05
  stopifnot(all(pmax(white_contrast, black_contrast) >= 4.5))
  ifelse(white_contrast > black_contrast, "white", "#000000")
}

nested_output_path <- out_path("nc_country_draw_nested_auc.csv")
nested_table_path <- file.path(
  "manuscript", "tables", "nc_country_draw_nested_auc.csv"
)
if (!file.exists(nested_output_path) || !file.exists(nested_table_path)) {
  stop(
    paste0(
      "The named-country supplement requires both audited nested-AUC copies. ",
      "Run `python script/build_nc_country_draw_nested_auc.py` first."
    ),
    call. = FALSE
  )
}

nested_required <- c(
  "mechanism", "analysis_population", "measurement_design",
  "validation_scheme", "uncertainty_method", "package_id",
  "n_countries", "n_country_draw_groups", "n_observations",
  "n_bootstrap", "bootstrap_seed", "pooled_auc",
  "nested_auc_lower_95", "nested_auc_upper_95",
  "country_auc_median", "country_auc_minimum", "country_auc_maximum",
  "surveillance_design_version"
)
nested_auc <- read_csv_checked(nested_table_path, required = nested_required)
nested_output <- read_csv_checked(nested_output_path, required = nested_required)
assert_true(
  isTRUE(all.equal(
    nested_auc |>
      arrange(package_id) |>
      select(all_of(nested_required)),
    nested_output |>
      arrange(package_id) |>
      select(all_of(nested_required)),
    check.attributes = FALSE,
    tolerance = 1e-12
  )),
  "The output and manuscript-facing nested-AUC tables differ."
)

holdout_summary <- read_csv_checked(
  file.path("manuscript", "tables", "surveillance_design_country_holdout.csv"),
  required = c(
    "measurement_design", "package_id", "validation_scheme", "auc",
    "median_country_auc", "minimum_country_auc", "maximum_country_auc",
    "surveillance_design_version"
  )
) |>
  filter(
    measurement_design == "main",
    validation_scheme %in% c("leave_one_country_out", "whole_country_holdout")
  ) |>
  mutate(validation_scheme = "leave_one_country_out")
holdout_detail <- read_csv_checked(
  file.path(
    "outputs", "model2c_surveillance_design", "monitoring_packages_v4_source_preserved_20260908",
    "country_holdout_country_discrimination.csv"
  ),
  required = c(
    "measurement_design", "package_id", "iso3", "validation_scheme",
    "n_country_draw_groups", "n_observations", "auc",
    "balanced_accuracy", "surveillance_design_version"
  )
) |>
  filter(
    measurement_design == "main",
    validation_scheme == "leave_one_country_out"
  )

optional_stream_order <- c("age", "testing", "severe", "serology")
package_order <- c(
  "notifications",
  unlist(
    lapply(
      seq_along(optional_stream_order),
      function(n_streams) {
        combn(
          optional_stream_order,
          n_streams,
          FUN = function(streams) {
            paste(c("notifications", streams), collapse = "+")
          }
        )
      }
    ),
    use.names = FALSE
  )
)
package_labels <- setNames(
  if_else(
    package_order == "notifications",
    "Notifications",
    paste0(
      "+ ",
      str_replace_all(
        str_remove(package_order, fixed("notifications+")),
        fixed("+"),
        " + "
      )
    )
  ),
  package_order
)
holdout_order <- c(
  "notifications",
  "notifications+age",
  "notifications+serology",
  "notifications+testing",
  "notifications+severe",
  "notifications+age+testing+severe+serology"
)
holdout_labels <- c(
  notifications = "Notifications",
  `notifications+age` = "+ age composition",
  `notifications+serology` = "+ serology",
  `notifications+testing` = "+ testing index",
  `notifications+severe` = "+ severe outcomes",
  `notifications+age+testing+severe+serology` = "All five streams"
)
holdout_detail_focus <- holdout_detail |>
  filter(package_id %in% holdout_order)
country_order <- c(pilot_iso3, "Median", "Pooled")
country_labels <- c(
  AUS = "Australia",
  BRA = "Brazil",
  CHN = "China",
  GBR = "United\nKingdom",
  JPN = "Japan",
  NZL = "New\nZealand",
  SWE = "Sweden",
  THA = "Thailand",
  USA = "United\nStates",
  Median = "Country\nmedian",
  Pooled = "Pooled\nsecondary"
)

assert_true(
  nrow(nested_auc) == 16L &&
    setequal(nested_auc$package_id, package_order) &&
    !anyDuplicated(nested_auc$package_id) &&
    all(nested_auc$n_countries == 9L) &&
    all(nested_auc$n_country_draw_groups == 720L) &&
    all(nested_auc$n_bootstrap == 2000L) &&
    all(nested_auc$bootstrap_seed == 20260730L) &&
    all(
      nested_auc$uncertainty_method ==
        "country_first_draw_second_nested_bootstrap"
    ),
    "The nested-AUC input is not the audited 16-package, 2,000-replicate result."
)
assert_true(
  nrow(holdout_summary) == length(holdout_order) &&
    setequal(holdout_summary$package_id, holdout_order),
  "The holdout summary must contain the six prespecified packages."
)
assert_true(
  nrow(holdout_detail_focus) == length(holdout_order) * length(pilot_iso3) &&
    setequal(holdout_detail_focus$package_id, holdout_order) &&
    setequal(holdout_detail_focus$iso3, pilot_iso3) &&
    !anyDuplicated(holdout_detail_focus[c("package_id", "iso3")]),
  "The holdout detail must contain a complete 9-country by 6-package grid."
)

package_source <- nested_auc |>
  mutate(
    panel = "a",
    package_rank = match(package_id, package_order),
    package_label = unname(package_labels[package_id])
  )
package_layers <- bind_rows(
  package_source |>
    transmute(
      package_id,
      package_rank,
      y = 17L - package_rank + 0.13,
      layer = "Pooled AUC\n95% UI",
      estimate = pooled_auc,
      lower = nested_auc_lower_95,
      upper = nested_auc_upper_95
    ),
  package_source |>
    transmute(
      package_id,
      package_rank,
      y = 17L - package_rank - 0.13,
      layer = "Grouped-CV country median\ncountry range",
      estimate = country_auc_median,
      lower = country_auc_minimum,
      upper = country_auc_maximum
    )
) |>
  mutate(
    layer = factor(
      layer,
      levels = c(
        "Pooled AUC\n95% UI",
        "Grouped-CV country median\ncountry range"
      )
    )
  )

holdout_country <- holdout_detail_focus |>
  transmute(
    panel = "b",
    measurement_design,
    package_id,
    iso3,
    validation_scheme,
    n_country_draw_groups,
    n_observations,
    auc,
    balanced_accuracy,
    summary_type = "country",
    surveillance_design_version
  )
holdout_median <- holdout_summary |>
  filter(package_id %in% holdout_order) |>
  transmute(
    panel = "b",
    measurement_design,
    package_id,
    iso3 = "Median",
    validation_scheme,
    n_country_draw_groups = NA_integer_,
    n_observations = NA_integer_,
    auc = median_country_auc,
    balanced_accuracy = NA_real_,
    summary_type = "country_median",
    surveillance_design_version
  )
holdout_pooled <- holdout_summary |>
  filter(package_id %in% holdout_order) |>
  transmute(
    panel = "b",
    measurement_design,
    package_id,
    iso3 = "Pooled",
    validation_scheme,
    n_country_draw_groups = NA_integer_,
    n_observations = NA_integer_,
    auc,
    balanced_accuracy = NA_real_,
    summary_type = "pooled_secondary",
    surveillance_design_version
  )
holdout_limits <- c(floor(min(0.45, holdout_country$auc, holdout_median$auc, holdout_pooled$auc) * 10) / 10, 1)
holdout_source <- bind_rows(
  holdout_country,
  holdout_median,
  holdout_pooled
) |>
  mutate(
    iso3 = factor(iso3, levels = country_order),
    package_id = factor(package_id, levels = rev(holdout_order)),
    value_label = sprintf("%.3f", auc),
    text_colour = holdout_text_colour(auc, holdout_limits)
  )

write_csv(
  package_source,
  fd_path("supplement_named_country_package_performance.csv"),
  na = ""
)
write_csv(
  holdout_source |>
    mutate(
      iso3 = as.character(iso3),
      package_id = as.character(package_id)
    ) |>
    select(-value_label, -text_colour),
  fd_path("supplement_named_country_holdout_auc.csv"),
  na = ""
)

ink <- figure_colours$text
muted <- figure_colours$muted_text
layer_palette <- c(
  "Pooled AUC\n95% UI" = clr("black"),
  "Grouped-CV country median\ncountry range" = clr("green")
)
layer_shapes <- c(
  "Pooled AUC\n95% UI" = 16,
  "Grouped-CV country median\ncountry range" = 5
)

panel_style <- function(plot_margin = margin(4, 4, 4, 4)) {
  theme_manuscript(show_titles = FALSE, plot_margin = plot_margin) +
    theme(
      axis.text = element_text(size = 5.1),
      legend.text = element_text(size = 5.1),
      legend.title = element_text(size = 5.2),
      plot.tag = element_text(size = 8, face = "bold"),
      plot.tag.position = c(0, 1),
      panel.grid = element_blank()
    )
}

package_breaks <- seq_along(package_order)
package_axis_labels <- setNames(
  unname(package_labels[package_order]),
  17L - seq_along(package_order)
)
p_a <- ggplot(package_layers, aes(y = y, colour = layer, shape = layer)) +
  geom_vline(
    xintercept = 0.5,
    linewidth = 0.30,
    linetype = "dotted",
    colour = muted
  ) +
  geom_vline(
    xintercept = 0.8,
    linewidth = 0.34,
    linetype = "dashed",
    colour = ink
  ) +
  geom_segment(
    aes(x = lower, xend = upper, yend = y),
    linewidth = 0.42
  ) +
  geom_point(aes(x = estimate), size = 1.55, stroke = 0.50) +
  scale_colour_manual(values = layer_palette, name = NULL) +
  scale_shape_manual(values = layer_shapes, name = NULL) +
  scale_x_continuous(
    breaks = c(0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  # Physical space beyond AUC = 1 keeps the open country-median diamonds whole.
  coord_cartesian(xlim = c(min(0.45, min(package_layers$lower)) - 0.005, 1.03)) +
  scale_y_continuous(
    limits = c(0.4, 16.6),
    breaks = package_breaks,
    labels = unname(package_axis_labels[as.character(package_breaks)]),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(tag = "b", x = "Area under the ROC curve", y = NULL) +
  panel_style(plot_margin = margin(4, 5, 4, 4)) +
  guides(
    colour = guide_legend(nrow = 2, byrow = TRUE),
    shape = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  theme(
    axis.text.y = element_text(size = 5.0),
    legend.position = "bottom",
    legend.key.height = unit(2.4, "mm")
  )

p_b <- ggplot(holdout_source, aes(x = iso3, y = package_id, fill = auc)) +
  geom_tile(colour = "white", linewidth = 0.45) +
  geom_vline(
    xintercept = c(9.5, 10.5),
    colour = muted,
    linewidth = 0.32
  ) +
  geom_text(
    aes(label = value_label, colour = text_colour),
    family = figure_font_family,
    fontface = "bold",
    size = 1.82,
    show.legend = FALSE
  ) +
  scale_colour_identity() +
  scale_fill_gradientn(
    colours = holdout_palette,
    limits = holdout_limits,
    oob = squish,
    breaks = sort(unique(c(holdout_limits[[1]], 0.5, 0.8, 1.0))),
    labels = label_number(accuracy = 0.1),
    name = "Whole-country holdout AUC"
  ) +
  scale_x_discrete(labels = country_labels, drop = FALSE) +
  scale_y_discrete(labels = holdout_labels, drop = FALSE) +
  labs(tag = "a", x = "Held-out country / summary", y = NULL) +
  guides(
    fill = guide_colourbar(
      title.position = "top",
      title.hjust = 0,
      barwidth = unit(24, "mm"),
      barheight = unit(2.0, "mm"),
      ticks.colour = ink,
      frame.colour = NA
    )
  ) +
  panel_style(plot_margin = margin(4, 4, 4, 5)) +
  theme(
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 5.0, angle = 42, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 5.0),
    legend.position = "bottom"
  )

figure <- p_b + p_a +
  plot_layout(widths = c(1.8, 1.0)) +
  plot_annotation(
    caption = paste0(
      "Panel a: named whole-country holdout with country median and pooled ",
      "same-model secondary summary; this is internal transport, not external validation.\n",
      "Panel b: 2,000-replicate country-first, calibration-ensemble-member-second ",
      "95% UIs from nested-bootstrap replicates and the nine-country median/range."
    ),
    theme = theme(
      plot.background = element_rect(fill = "white", colour = NA),
      plot.caption = element_text(
        family = figure_font_family,
        size = 5.0,
        colour = muted,
        hjust = 0,
        lineheight = 1.05,
        margin = margin(t = 2)
      ),
      plot.margin = margin(2, 2, 2, 2)
    )
  )

output_stem <- "supplement_named_country_auc"
ggsave(
  filename = supplement_fig_path(paste0(output_stem, ".png")),
  plot = figure,
  width = nature_full_width_mm,
  height = 118,
  units = "mm",
  dpi = 600,
  bg = "white"
)
assert_true(
  file.exists(supplement_fig_path("supplement_named_country_auc.png")),
  "Named-country supplement PNG export is missing."
)

message("Wrote the named-country hierarchical AUC supplement.")
