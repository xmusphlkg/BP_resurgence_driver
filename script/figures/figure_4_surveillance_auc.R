#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
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
fig_dir <- diagnostic_dir
fig_path <- function(name) file.path(diagnostic_dir, name)

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

assert_auc_interval <- function(data, estimate, lower, upper, label) {
  estimate_value <- data[[estimate]]
  lower_value <- data[[lower]]
  upper_value <- data[[upper]]
  assert_true(
    all(is.finite(estimate_value)) &&
      all(is.finite(lower_value)) &&
      all(is.finite(upper_value)),
    paste0(label, " contains non-finite AUC values.")
  )
  assert_true(
    all(lower_value >= 0 & upper_value <= 1) &&
      all(lower_value <= estimate_value & estimate_value <= upper_value),
    paste0(label, " contains invalid AUC intervals.")
  )
}

package_auc_flat <- read_csv_checked(
  file.path(
    "outputs", "model2c_surveillance_design", "monitoring_packages_v4_source_preserved_20260908",
    "package_discrimination.csv"
  ),
  required = c(
    "mechanism", "analysis_population", "measurement_design",
    "package_id", "auc", "auc_lower_95", "auc_upper_95",
    "balanced_accuracy", "surveillance_design_version"
  )
) |>
  filter(
    mechanism == "transmission_vs_reporting",
    analysis_population == "all_nine",
    measurement_design == "main"
  )
nested_output_path <- out_path("nc_country_draw_nested_auc.csv")
nested_table_path <- file.path(
  "manuscript", "tables", "nc_country_draw_nested_auc.csv"
)
if (!file.exists(nested_output_path) || !file.exists(nested_table_path)) {
  stop(
    paste0(
      "Figure 6 panel a requires both outputs/nc_country_draw_nested_auc.csv ",
      "and manuscript/tables/nc_country_draw_nested_auc.csv. Run ",
      "`python script/build_nc_country_draw_nested_auc.py` before rendering; ",
      "flat cluster intervals are not an allowed fallback."
    ),
    call. = FALSE
  )
}
nested_required <- c(
  "analysis_version", "package_id", "pooled_auc", "nested_auc_lower_95",
  "nested_auc_upper_95", "country_auc_median",
  "country_auc_minimum", "country_auc_maximum",
  "n_countries", "n_country_draw_groups", "n_bootstrap",
  "bootstrap_seed", "uncertainty_method", "surveillance_design_version"
)
nested_auc <- read_csv_checked(
  nested_table_path,
  required = nested_required
)
nested_output <- read_csv_checked(
  nested_output_path,
  required = nested_required
)
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
  paste0(
    "The output and manuscript-facing nested-AUC tables differ. ",
    "Rebuild them together before rendering Figure 6."
  )
)
package_auc <- package_auc_flat |>
  select(-any_of(c(
    "auc", "auc_lower_95", "auc_upper_95", "n_bootstrap", "bootstrap_seed",
    "uncertainty_method", "country_auc_median", "country_auc_minimum", "country_auc_maximum"
  ))) |>
  left_join(
    nested_auc |>
      transmute(
        package_id,
        auc = pooled_auc,
        auc_lower_95 = nested_auc_lower_95,
        auc_upper_95 = nested_auc_upper_95,
        country_auc_median,
        country_auc_minimum,
        country_auc_maximum,
        n_bootstrap,
        bootstrap_seed,
        uncertainty_method,
        nested_analysis_version = analysis_version
      ),
    by = "package_id"
  )
country_holdout <- read_csv_checked(
  file.path("manuscript", "tables", "surveillance_design_country_holdout.csv"),
  required = c(
    "measurement_design", "package_id", "validation_scheme", "auc",
    "n_countries", "n_countries_auc_ge_0_70", "surveillance_design_version"
  )
)
country_holdout_country <- read_csv_checked(
  file.path(
    "outputs", "model2c_surveillance_design", "monitoring_packages_v4_source_preserved_20260908",
    "country_holdout_country_discrimination.csv"
  ),
  required = c(
    "measurement_design", "package_id", "iso3", "validation_scheme",
    "n_country_draw_groups", "n_observations", "auc", "balanced_accuracy",
    "surveillance_design_version"
  )
)
attenuation <- read_csv_checked(
  out_path("nc_contrast_attenuation_frontier.csv"),
  required = c(
    "contrast_level", "package_id", "grouped_cv_auc",
    "grouped_cv_auc_lower_95", "grouped_cv_auc_upper_95",
    "grouped_cv_countries_auc_ge_0_70", "whole_country_holdout_auc",
    "whole_country_holdout_countries_auc_ge_0_70",
    "interpretation_boundary"
  )
)
cross_loading <- read_csv_checked(
  out_path("nc_measurement_contamination_frontier.csv"),
  required = c(
    "contamination_regime", "contamination_fraction",
    "testing_cross_loading_fraction", "severe_cross_loading_fraction",
    "package_id", "grouped_cv_auc", "grouped_cv_auc_lower_95",
    "grouped_cv_auc_upper_95", "grouped_cv_countries_auc_ge_0_70",
    "whole_country_holdout_auc",
    "whole_country_holdout_countries_auc_ge_0_70",
    "contamination_definition", "interpretation_boundary"
  )
)
cross_loading_surface <- read_csv_checked(
  out_path("nc_measurement_contamination_surface.csv"),
  required = c(
    "testing_cross_loading_fraction", "severe_cross_loading_fraction",
    "package_id", "grouped_cv_auc", "grouped_cv_auc_lower_95",
    "grouped_cv_auc_upper_95", "grouped_cv_countries_auc_ge_0_70",
    "whole_country_holdout_auc",
    "whole_country_holdout_countries_auc_ge_0_70",
    "analysis_version", "grid_definition", "contamination_definition",
    "interpretation_boundary"
  )
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
holdout_package_order <- c(
  "notifications",
  "notifications+age",
  "notifications+testing",
  "notifications+severe",
  "notifications+serology",
  "notifications+age+testing+severe+serology"
)
country_holdout <- country_holdout |>
  filter(measurement_design == "main", package_id %in% holdout_package_order,
         validation_scheme %in% c("leave_one_country_out", "whole_country_holdout")) |>
  mutate(validation_scheme = "leave_one_country_out")
country_holdout_country <- country_holdout_country |>
  filter(measurement_design == "main", package_id %in% holdout_package_order)
holdout_plot_order <- c(
  "notifications",
  "notifications+age",
  "notifications+serology",
  "notifications+testing",
  "notifications+severe",
  "notifications+age+testing+severe+serology"
)
holdout_plot_labels <- c(
  notifications = "Notifications",
  `notifications+age` = "+ Age\ncomposition",
  `notifications+serology` = "+ Serology",
  `notifications+testing` = "+ Testing\nindex",
  `notifications+severe` = "+ Severe\noutcomes",
  `notifications+age+testing+severe+serology` = "All five\nstreams"
)
package_component_order <- c(
  "notifications", "age", "testing", "severe", "serology"
)
package_component_labels <- c(
  notifications = "Notifications",
  age = "Age composition",
  testing = "Testing index",
  severe = "Severe outcomes",
  serology = "Serology"
)
stream_ids <- c(
  "notifications+testing",
  "notifications+severe",
  "notifications+testing+severe"
)
stream_labels <- c(
  `notifications+testing` = "Testing",
  `notifications+severe` = "Severe outcomes",
  `notifications+testing+severe` = "Testing + severe"
)
stream_palette <- c(
  `notifications+testing` = clr("blue"),
  `notifications+severe` = clr("orange"),
  `notifications+testing+severe` = clr("black")
)
stream_shapes <- c(
  Testing = 16,
  `Severe outcomes` = 15,
  `Testing + severe` = 17
)

assert_true(
  nrow(package_auc) == length(package_order) &&
    setequal(package_auc$package_id, package_order) &&
    !anyDuplicated(package_auc$package_id),
  "The grouped-validation summary must contain all 16 monitoring packages."
)
assert_true(
  all(package_auc$n_bootstrap == 2000L) &&
    all(package_auc$bootstrap_seed == 20260730L) &&
    all(
      package_auc$uncertainty_method ==
        "country_first_draw_second_nested_bootstrap"
    ),
  paste0(
    "Figure 6a requires the audited 2,000-replicate country-first, ",
    "ensemble-member-second nested intervals."
  )
)
assert_true(
  nrow(country_holdout) == length(holdout_package_order) &&
    setequal(country_holdout$package_id, holdout_package_order) &&
    all(country_holdout$measurement_design == "main") &&
    all(country_holdout$validation_scheme == "leave_one_country_out") &&
    !anyDuplicated(country_holdout$package_id),
  "The country-holdout summary must contain the six prespecified representative packages."
)
assert_true(
  nrow(country_holdout_country) ==
    length(holdout_package_order) * length(pilot_iso3) &&
    setequal(country_holdout_country$package_id, holdout_package_order) &&
    setequal(country_holdout_country$iso3, pilot_iso3) &&
    all(country_holdout_country$measurement_design == "main") &&
    all(country_holdout_country$validation_scheme == "leave_one_country_out") &&
    !anyDuplicated(country_holdout_country[c("package_id", "iso3")]),
  paste0(
    "Country-specific holdout results must contain a complete six-package ",
    "by nine-country grid."
  )
)
assert_true(
  length(unique(package_auc$surveillance_design_version)) == 1L &&
    identical(
      unique(package_auc$surveillance_design_version),
      unique(country_holdout$surveillance_design_version)
    ) &&
    identical(
      unique(package_auc$surveillance_design_version),
      unique(country_holdout_country$surveillance_design_version)
    ),
  "Package and country-holdout summaries do not share one frozen design version."
)
assert_auc_interval(
  package_auc, "auc", "auc_lower_95", "auc_upper_95",
  "All-package grouped-validation summary"
)
assert_auc_interval(
  attenuation, "grouped_cv_auc", "grouped_cv_auc_lower_95",
  "grouped_cv_auc_upper_95", "Contrast-attenuation frontier"
)
assert_auc_interval(
  cross_loading, "grouped_cv_auc", "grouped_cv_auc_lower_95",
  "grouped_cv_auc_upper_95", "Cross-loading frontier"
)
assert_auc_interval(
  cross_loading_surface, "grouped_cv_auc", "grouped_cv_auc_lower_95",
  "grouped_cv_auc_upper_95", "Cross-loading response surface"
)

expected_contrast_levels <- c(0, 0.125, 0.25, 0.5, 0.75, 1)
expected_cross_loading_levels <- c(0, 0.25, 0.5, 0.75, 1)
assert_true(
  setequal(unique(attenuation$contrast_level), expected_contrast_levels) &&
    !anyDuplicated(attenuation[c("contrast_level", "package_id")]),
  "Unexpected or duplicated contrast-attenuation rows."
)
assert_true(
  setequal(unique(cross_loading$contamination_fraction), expected_cross_loading_levels) &&
    setequal(
      unique(cross_loading$contamination_regime),
      c("both_contaminated", "testing_only", "severe_only")
    ) &&
    !anyDuplicated(
      cross_loading[c("contamination_regime", "contamination_fraction", "package_id")]
    ),
  "Unexpected or duplicated cross-loading rows."
)
assert_true(
  nrow(cross_loading_surface) == length(expected_cross_loading_levels)^2 &&
    setequal(
      unique(cross_loading_surface$testing_cross_loading_fraction),
      expected_cross_loading_levels
    ) &&
    setequal(
      unique(cross_loading_surface$severe_cross_loading_fraction),
      expected_cross_loading_levels
    ) &&
    all(
      cross_loading_surface$package_id ==
        "notifications+testing+severe"
    ) &&
    !anyDuplicated(
      cross_loading_surface[
        c(
          "testing_cross_loading_fraction",
          "severe_cross_loading_fraction"
        )
      ]
    ),
  "The combined-package cross-loading response surface must be a complete 5 x 5 grid."
)

package_cv <- package_auc |>
  transmute(
    package_id,
    validation = "Grouped 5-fold CV",
    auc,
    auc_lower_95,
    auc_upper_95,
    country_auc_median,
    country_auc_minimum,
    country_auc_maximum,
    interval_method =
    "2,000-replicate country-first, ensemble-member-second nested bootstrap",
    n_bootstrap,
    bootstrap_seed,
    n_countries = 9L,
    n_countries_auc_ge_0_70 = NA_integer_,
    surveillance_design_version
  )
package_source <- package_cv |>
  mutate(
    panel = "a",
    package_id = factor(
      package_id,
      levels = package_order
    )
  )

holdout_country_source <- country_holdout_country |>
  filter(package_id %in% holdout_plot_order) |>
  transmute(
    panel = "b",
    measurement_design,
    package_id = factor(package_id, levels = rev(holdout_plot_order)),
    iso3,
    validation_scheme,
    n_country_draw_groups,
    n_observations,
    auc,
    balanced_accuracy,
    surveillance_design_version
  )

holdout_country_summary <- holdout_country_source |>
  group_by(package_id) |>
  summarise(
    median_auc = median(auc),
    min_auc = min(auc),
    max_auc = max(auc),
    n_countries = n(),
    n_countries_auc_ge_0_70 = sum(auc >= 0.70),
    .groups = "drop"
  )

package_matrix_source <- crossing(
  package_id = package_order,
  component_id = package_component_order
) |>
  mutate(
    included = str_detect(
      paste0("+", package_id, "+"),
      fixed(paste0("+", component_id, "+"))
    ),
    package_id = factor(package_id, levels = package_order),
    component_id = factor(
      component_id,
      levels = rev(package_component_order)
    )
  )

package_membership_export <- package_matrix_source |>
  mutate(
    package_id = as.character(package_id),
    component_id = as.character(component_id)
  ) |>
  select(package_id, component_id, included) |>
  pivot_wider(
    names_from = component_id,
    values_from = included,
    names_prefix = "includes_"
  )

assert_true(
  nrow(package_membership_export) == length(package_order) &&
    nrow(distinct(
      package_membership_export,
      across(starts_with("includes_"))
    )) == length(package_order),
  "The package membership matrix must encode 16 unique monitoring combinations."
)

attenuation_source <- attenuation |>
  filter(package_id %in% stream_ids) |>
  transmute(
    panel = "c",
    retained_between_class_contrast = contrast_level,
    package_id,
    monitoring_package = factor(
      stream_labels[package_id],
      levels = unname(stream_labels[stream_ids])
    ),
    grouped_cv_auc,
    grouped_cv_auc_lower_95,
    grouped_cv_auc_upper_95,
    grouped_cv_countries_auc_ge_0_70,
    whole_country_holdout_auc,
    whole_country_holdout_countries_auc_ge_0_70,
    interpretation_boundary
  )

cross_loading_both_source <- cross_loading |>
  filter(
    contamination_regime == "both_contaminated",
    package_id %in% stream_ids
  ) |>
  transmute(
    panel = "d_shared",
    cross_loading_regime = contamination_regime,
    cross_loading_coefficient = contamination_fraction,
    testing_cross_loading_fraction,
    severe_cross_loading_fraction,
    package_id,
    monitoring_package = factor(
      stream_labels[package_id],
      levels = unname(stream_labels[stream_ids])
    ),
    grouped_cv_auc,
    grouped_cv_auc_lower_95,
    grouped_cv_auc_upper_95,
    grouped_cv_countries_auc_ge_0_70,
    whole_country_holdout_auc,
    whole_country_holdout_countries_auc_ge_0_70,
    contamination_definition,
    interpretation_boundary
  )

cross_loading_surface_source <- cross_loading_surface |>
  transmute(
    panel = "e_surface",
    testing_cross_loading_fraction,
    severe_cross_loading_fraction,
    package_id,
    grouped_cv_auc,
    grouped_cv_auc_lower_95,
    grouped_cv_auc_upper_95,
    grouped_cv_countries_auc_ge_0_70,
    whole_country_holdout_auc,
    whole_country_holdout_countries_auc_ge_0_70,
    analysis_version,
    grid_definition,
    contamination_definition,
    interpretation_boundary
  )

surface_auc_data_range <- range(
  0.8,
  cross_loading_surface_source$grouped_cv_auc,
  na.rm = TRUE
)
surface_auc_pretty <- pretty(surface_auc_data_range, n = 4)
surface_auc_limits <- c(
  max(0, min(surface_auc_pretty)),
  min(1, max(surface_auc_pretty))
)
surface_auc_breaks <- pretty(surface_auc_limits, n = 3)
surface_auc_breaks <- surface_auc_breaks[
  surface_auc_breaks >= surface_auc_limits[[1]] &
    surface_auc_breaks <= surface_auc_limits[[2]]
]
surface_auc_breaks <- sort(unique(c(surface_auc_limits, surface_auc_breaks)))
surface_auc_midpoint <- 0.80
surface_auc_span <- diff(surface_auc_limits)
surface_auc_text_low <- surface_auc_limits[[1]] + 0.30 * surface_auc_span
surface_auc_text_high <- surface_auc_limits[[1]] + 0.78 * surface_auc_span

assert_true(
  all(surface_auc_data_range >= surface_auc_limits[[1]]) &&
    all(surface_auc_data_range <= surface_auc_limits[[2]]) &&
    surface_auc_midpoint >= surface_auc_limits[[1]] &&
    surface_auc_midpoint <= surface_auc_limits[[2]],
  "Figure 6e AUC limits must contain the response surface and the 0.80 reference midpoint."
)

write_csv(
  package_source |>
    mutate(package_id = as.character(package_id)) |>
    left_join(package_membership_export, by = "package_id"),
  fd_path("figure_4_package_performance.csv"),
  na = ""
)
write_csv(
  holdout_country_source |>
    mutate(package_id = as.character(package_id)),
  fd_path("figure_4_country_holdout_auc.csv"),
  na = ""
)
write_csv(
  attenuation_source |>
    mutate(monitoring_package = as.character(monitoring_package)),
  fd_path("figure_4_attenuation.csv"),
  na = ""
)
write_csv(
  cross_loading_both_source |>
    mutate(monitoring_package = as.character(monitoring_package)) |>
    select(
      panel, cross_loading_regime, cross_loading_coefficient,
      testing_cross_loading_fraction, severe_cross_loading_fraction,
      package_id, monitoring_package,
      grouped_cv_auc, grouped_cv_auc_lower_95, grouped_cv_auc_upper_95,
      grouped_cv_countries_auc_ge_0_70, whole_country_holdout_auc,
      whole_country_holdout_countries_auc_ge_0_70,
      contamination_definition, interpretation_boundary
    ),
  fd_path("figure_4_cross_loading.csv"),
  na = ""
)
write_csv(
  cross_loading_surface_source,
  fd_path("figure_4_cross_loading_surface.csv"),
  na = ""
)

ink <- figure_colours$text
neutral_mid <- figure_colours$muted_text
neutral_light <- clr("light_grey")
matrix_green <- clr("green")
matrix_green_border <- "#4B9323"
matrix_empty_border <- "#73787D"
package_optional_counts <- str_count(package_order, fixed("+"))
package_group_breaks <- which(diff(package_optional_counts) != 0) + 0.5

reference_layers <- function() {
  list(
    geom_hline(
      yintercept = 0.5, linewidth = 0.32,
      linetype = "dotted", colour = neutral_mid
    ),
    geom_hline(
      yintercept = 0.8, linewidth = 0.38,
      linetype = "dashed", colour = ink
    )
  )
}

p_a_matrix <- ggplot(
  package_matrix_source,
  aes(x = package_id, y = component_id)
) +
  geom_hline(
    yintercept = seq(1.5, length(package_component_order) - 0.5, by = 1),
    linewidth = 0.18,
    colour = alpha(neutral_light, 0.72)
  ) +
  geom_vline(
    xintercept = package_group_breaks,
    linewidth = 0.22,
    colour = neutral_light
  ) +
  geom_point(
    data = filter(package_matrix_source, !included),
    shape = 21, size = 1.44, stroke = 0.34,
    fill = clr("white"), colour = matrix_empty_border
  ) +
  geom_point(
    data = filter(package_matrix_source, included),
    shape = 21, size = 1.58, stroke = 0.26,
    fill = matrix_green, colour = matrix_green_border
  ) +
  scale_x_discrete(
    drop = FALSE,
    expand = expansion(add = c(0.35, 0.35))
  ) +
  scale_y_discrete(
    labels = package_component_labels,
    expand = expansion(add = 0.30)
  ) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_manuscript(
    show_titles = FALSE,
    plot_margin = margin(0, 5, 3, 6)
  ) +
  theme(
    axis.line.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(
      size = figure_text$axis_text_dense,
      colour = ink,
      margin = margin(r = 1.5)
    ),
    legend.position = "none",
    plot.margin = margin(0, 5, 3, 6)
  )

p_a_auc <- ggplot() +
  reference_layers() +
  geom_vline(
    xintercept = package_group_breaks,
    linewidth = 0.22,
    colour = neutral_light
  ) +
  geom_segment(
    data = package_source,
    aes(
      x = package_id, xend = package_id,
      y = auc_lower_95, yend = auc_upper_95
    ),
    linewidth = 0.58, lineend = "round",
    colour = ink, alpha = 0.92
  ) +
  geom_point(
    data = package_source,
    aes(x = package_id, y = auc),
    shape = 21, fill = ink, colour = ink,
    size = 1.50, stroke = 0.25
  ) +
  scale_x_discrete(
    drop = FALSE,
    expand = expansion(add = c(0.35, 0.35))
  ) +
  scale_y_continuous(
    limits = c(0.45, 1.005),
    breaks = figure_breaks$auc,
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    tag = "a",
    x = NULL,
    y = "Grouped-CV ROC AUC"
  ) +
  theme_manuscript(show_titles = FALSE, plot_margin = margin(4, 5, 0, 6)) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.line.x = element_blank(),
    plot.tag.position = c(0.018, 0.982),
    plot.margin = margin(4, 5, 0, 6)
  )

p_a <- wrap_plots(
  list(p_a_auc, p_a_matrix),
  ncol = 1,
  heights = c(1.54, 1.02)
)

p_b <- ggplot() +
  geom_vline(
    xintercept = 0.5, linewidth = 0.32,
    linetype = "dotted", colour = neutral_mid
  ) +
  geom_vline(
    xintercept = 0.8, linewidth = 0.38,
    linetype = "dashed", colour = ink
  ) +
  geom_segment(
    data = holdout_country_summary,
    aes(
      x = min_auc, xend = max_auc,
      y = package_id, yend = package_id
    ),
    linewidth = 0.58, lineend = "round",
    colour = clr("blue"), alpha = 0.86
  ) +
  geom_point(
    data = holdout_country_source,
    aes(x = auc, y = package_id),
    shape = 21, size = 0.92, stroke = 0.25,
    fill = clr("white"), colour = neutral_mid, alpha = 0.78,
    position = position_jitter(width = 0, height = 0.085, seed = 404)
  ) +
  geom_point(
    data = holdout_country_summary,
    aes(x = median_auc, y = package_id),
    shape = 21, size = 1.52, stroke = 0.25,
    fill = clr("blue"), colour = clr("blue")
  ) +
  scale_x_continuous(
    limits = c(max(0, min(holdout_country_source$auc) - 0.02), 1.005),
    breaks = c(0, 0.2, 0.5, 0.8, 1.0),
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_discrete(labels = holdout_plot_labels) +
  labs(
    tag = "b",
    x = "Country-specific ROC AUC\nmedian (country range)",
    y = NULL
  ) +
  theme_manuscript(
    grid = "x",
    show_titles = FALSE,
    plot_margin = margin(4, 4, 3, 4)
  ) +
  theme(
    axis.text.y = element_text(
      size = figure_text$axis_text_dense,
      lineheight = 0.88,
      margin = margin(r = 1.5)
    ),
    plot.tag.position = c(0, 1),
    plot.margin = margin(4, 4, 3, 4)
  )

p_c <- ggplot(
  attenuation_source,
  aes(
    x = retained_between_class_contrast,
    y = grouped_cv_auc,
    colour = monitoring_package,
    shape = monitoring_package,
    group = monitoring_package
  )
) +
  reference_layers() +
  geom_ribbon(
    aes(
      ymin = grouped_cv_auc_lower_95,
      ymax = grouped_cv_auc_upper_95,
      fill = monitoring_package
    ),
    colour = NA, alpha = 0.13, show.legend = FALSE
  ) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.55, stroke = 0.35) +
  scale_colour_manual(
    values = unname(stream_palette[stream_ids]),
    name = NULL
  ) +
  scale_fill_manual(values = unname(stream_palette[stream_ids])) +
  scale_shape_manual(values = stream_shapes, name = NULL) +
  scale_x_continuous(
    breaks = figure_breaks$probability,
    limits = c(0, 1),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    limits = c(0.45, 1.01),
    breaks = figure_breaks$auc,
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    tag = "c",
    x = "Retained between-class contrast, d",
    y = "Area under the ROC curve"
  ) +
  theme_manuscript(show_titles = FALSE, plot_margin = margin(4, 5, 4, 5)) +
  guides(
    colour = guide_legend(nrow = 1, byrow = TRUE),
    shape = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme(legend.spacing.x = unit(1.5, "mm"))

p_d <- ggplot(
  cross_loading_both_source,
  aes(
    x = cross_loading_coefficient,
    y = grouped_cv_auc,
    colour = monitoring_package,
    shape = monitoring_package,
    group = monitoring_package
  )
) +
  reference_layers() +
  geom_ribbon(
    aes(
      ymin = grouped_cv_auc_lower_95,
      ymax = grouped_cv_auc_upper_95,
      fill = monitoring_package
    ),
    colour = NA, alpha = 0.13, show.legend = FALSE
  ) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.55, stroke = 0.35) +
  scale_colour_manual(
    values = unname(stream_palette[stream_ids]),
    name = NULL
  ) +
  scale_fill_manual(values = unname(stream_palette[stream_ids])) +
  scale_shape_manual(values = stream_shapes, name = NULL) +
  scale_x_continuous(
    breaks = figure_breaks$probability,
    limits = c(0, 1),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    limits = c(0.45, 1.01),
    breaks = figure_breaks$auc,
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    tag = "d",
    x = "Shared source-stream loading",
    y = "Area under the ROC curve"
  ) +
  theme_manuscript(show_titles = FALSE, plot_margin = margin(4, 5, 4, 5)) +
  guides(
    colour = guide_legend(nrow = 1, byrow = TRUE),
    shape = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme(
    axis.title.x = element_text(face = "bold"),
    legend.spacing.x = unit(1.5, "mm")
  )

p_e <- ggplot(
  cross_loading_surface_source,
  aes(
    x = testing_cross_loading_fraction,
    y = severe_cross_loading_fraction,
    fill = grouped_cv_auc
  )
) +
  geom_tile(
    width = 0.245, height = 0.245,
    colour = clr("white"), linewidth = 0.35
  ) +
  geom_tile(
    data = filter(
      cross_loading_surface_source,
      testing_cross_loading_fraction == severe_cross_loading_fraction
    ),
    width = 0.225, height = 0.225,
    fill = NA, colour = ink, linewidth = 0.38,
    linetype = "dashed"
  ) +
  geom_tile(
    data = filter(cross_loading_surface_source, grouped_cv_auc < 0.80),
    width = 0.225, height = 0.225,
    fill = NA, colour = clr("orange"), linewidth = 0.75
  ) +
  geom_text(
    aes(
      label = sprintf("%.3f", grouped_cv_auc),
      colour = case_when(
        grouped_cv_auc < surface_auc_text_low ~ "reversed",
        grouped_cv_auc > surface_auc_text_high ~ "reversed",
        TRUE ~ "standard"
      )
    ),
    size = figure_text$annotation_dense,
    family = figure_font_family
  ) +
  scale_fill_gradient2(
    low = clr("orange"),
    mid = clr("paper"),
    high = clr("blue"),
    midpoint = surface_auc_midpoint,
    limits = surface_auc_limits,
    breaks = surface_auc_breaks,
    labels = label_number(accuracy = 0.01),
    name = "Combined-package grouped CV AUC",
    oob = squish
  ) +
  scale_colour_manual(
    values = c(
      reversed = figure_colours$background,
      standard = figure_colours$text
    ),
    guide = "none"
  ) +
  scale_x_continuous(
    limits = c(-0.125, 1.125),
    breaks = expected_cross_loading_levels,
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(-0.125, 1.125),
    breaks = expected_cross_loading_levels,
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_fixed(clip = "off") +
  labs(
    tag = "e",
    x = expression("Testing-link cross-loading, " * gamma[t]),
    y = expression("Severe-link cross-loading, " * gamma[s])
  ) +
  guides(
    fill = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(30, "mm"),
      barheight = unit(1.6, "mm"),
      ticks.colour = ink,
      frame.colour = neutral_mid
    )
  ) +
  theme_manuscript(show_titles = FALSE, plot_margin = margin(4, 4, 4, 2)) +
  theme(
    legend.position = "bottom",
    legend.margin = margin(t = 0),
    legend.box.margin = margin(0, 0, 0, 0),
    plot.margin = margin(4, 4, 4, 2)
  )

top_row <- wrap_plots(
  list(p_a, p_b),
  ncol = 2,
  widths = c(1.82, 1.18)
)

cd_row <- wrap_plots(
  list(free(p_c), p_d),
  ncol = 2,
  widths = c(1.0, 1.0),
  guides = "collect"
) &
  theme(legend.position = "bottom")

bottom_row <- wrap_plots(
  list(cd_row, p_e),
  ncol = 2,
  widths = c(2.0, 0.95)
)

figure <- wrap_plots(
  list(top_row, bottom_row),
  ncol = 1,
  heights = c(1.10, 0.90)
) +
  plot_annotation(
    theme = theme(
      plot.background = element_rect(
        fill = figure_colours$background,
        colour = NA
      ),
      plot.margin = margin(2, 2, 2, 2)
    )
  )

save_nature(
  figure,
  "figure_4_surveillance_auc.png",
  nature_full_width_mm,
  136,
  dpi = 600
)
ggsave(
  filename = fig_path("figure_4_surveillance_auc.pdf"),
  plot = figure,
  device = grDevices::cairo_pdf,
  width = nature_full_width_mm,
  height = 136,
  units = "mm",
  bg = "white"
)
message("Wrote manuscript/figures/figure_4_surveillance_auc.pdf")

message("Wrote figure source data:")
message("  manuscript/figure_data/figure_4_package_performance.csv")
message("  manuscript/figure_data/figure_4_country_holdout_auc.csv")
message("  manuscript/figure_data/figure_4_attenuation.csv")
message("  manuscript/figure_data/figure_4_cross_loading.csv")
message("  manuscript/figure_data/figure_4_cross_loading_surface.csv")
