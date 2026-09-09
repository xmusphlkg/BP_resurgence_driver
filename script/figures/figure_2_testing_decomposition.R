#!/usr/bin/env Rscript
# Shared laboratory panel constructors and Supplementary Figure 28.
# Main Figure 4 reuses the France/England constructors from the consolidated renderer.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(ragg)
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

d <- read_csv_checked(
  out_path("tiered_testing_decomposition.csv"),
  required = c(
    "iso3", "country", "year", "system", "tests", "positivity_percent",
    "ref_year", "tests_fold_vs_ref", "positivity_fold_vs_ref",
    "positive_tests", "notifications", "comparison_status"
  )
) |>
  filter(iso3 %in% c("FRA", "GBR", "USA", "THA")) |>
  mutate(panel_order = match(iso3, c("FRA", "GBR", "USA", "THA"))) |>
  arrange(panel_order, year)

comparable <- d |> filter(comparison_status == "same_system_longitudinal")
stopifnot(
  n_distinct(d$iso3) == 4L,
  all(is.finite(d$tests)), all(d$tests > 0),
  all(comparable$positivity_percent >= 0 & comparable$positivity_percent <= 100),
  all(is.finite(comparable$positive_tests)),
  all(abs(comparable$positive_tests - comparable$tests * comparable$positivity_percent / 100) < 1e-6),
  all(vapply(split(d, d$iso3), function(x) any(x$year == x$ref_year), logical(1))),
  identical(
    sort(unique(d$comparison_status)),
    sort(c("same_system_longitudinal", "unlinked_cross_system", "noncomparable_snapshots"))
  )
)

tests_colour <- clr("blue")
positivity_colour <- clr("orange")
notification_colour <- clr("dark_grey")
pandemic_fill <- "#F1F3F4"

aligned_track_axes <- function(colour, top = FALSE) {
  theme(
      axis.title.y = element_text(colour = colour),
      axis.text.y = element_text(colour = colour),
      axis.text.x = if (top) element_blank() else element_text(),
      axis.ticks.x = if (top) element_blank() else element_line(),
      panel.grid.major.y = element_line(colour = figure_colours$grid, linewidth = 0.16),
      plot.margin = margin(if (top) 4 else 1, 5, if (top) 1 else 4, 5)
    )
}

make_temporal_panel <- function(iso_code, tag) {
  sub <- d |> filter(iso3 == iso_code)
  unlinked <- iso_code == "USA"
  stopifnot(unique(sub$comparison_status) == if (unlinked) "unlinked_cross_system" else "same_system_longitudinal")
  ref <- sub |> filter(year == first(ref_year)) |> slice(1)
  sub$second_value <- if (unlinked) sub$notifications else sub$positivity_percent
  second_colour <- if (unlinked) notification_colour else positivity_colour
  count_title <- if (iso_code == "GBR") "Eligible samples\n(thousands)" else if (unlinked) "Test orders\n(thousands)" else "PCR tests\n(thousands)"
  second_title <- if (unlinked) "National notifications" else "Positive samples (%)"
  second_reference <- if (unlinked) ref$notifications else ref$positivity_percent
  stopifnot(all(is.finite(sub$second_value)))

  year_axis <- function() scale_x_continuous(
    breaks = seq(min(sub$year), max(sub$year)),
    limits = range(sub$year) + c(-0.55, 0.55), expand = expansion(mult = 0)
  )
  pandemic_band <- function() annotate(
    "rect", xmin = 2019.5, xmax = 2022.5, ymin = -Inf, ymax = Inf,
    fill = pandemic_fill, colour = NA
  )

  p_count <- ggplot(sub, aes(year, tests / 1000)) +
    pandemic_band() +
    geom_col(width = 0.68, fill = alpha(tests_colour, 0.72), colour = tests_colour, linewidth = 0.18) +
    geom_hline(yintercept = ref$tests / 1000, linetype = "dashed", colour = figure_colours$reference, linewidth = 0.32) +
    year_axis() +
    scale_y_continuous(labels = label_number(accuracy = 1), limits = c(0, NA), expand = expansion(mult = c(0, 0.08))) +
    labs(tag = tag, title = switch(iso_code, FRA = "France: 3Labos PCR network",
                                  GBR = "England: mixed assay pathways",
                                  USA = "USA: unlinked systems"), x = NULL, y = count_title) +
    theme_nature(grid = "y") +
    aligned_track_axes(tests_colour, top = TRUE)

  p_second <- ggplot(sub, aes(year, second_value)) +
    pandemic_band() +
    geom_hline(yintercept = second_reference, linetype = "dashed", colour = figure_colours$reference, linewidth = 0.32) +
    geom_line(colour = second_colour, linewidth = 0.74) +
    geom_point(shape = 21, size = 1.65, stroke = 0.48, fill = "white", colour = second_colour) +
    year_axis() +
    scale_y_continuous(
      labels = if (unlinked) label_number(scale_cut = cut_short_scale()) else label_number(suffix = "%", accuracy = 1),
      limits = c(0, NA), expand = expansion(mult = c(0, 0.12))
    ) +
    labs(x = NULL, y = second_title) +
    theme_nature(grid = "y") +
    aligned_track_axes(second_colour)

  # Shared x coordinates are the only alignment between tracks. Each outcome
  # keeps its own units; maxima cannot force a visual coincidence.
  (p_count / p_second) + plot_layout(heights = c(1, 1))
}

make_thailand_boundary_panel <- function(tag) {
  sub <- d |>
    filter(iso3 == "THA") |>
    mutate(snapshot = factor(
      if_else(year == 2019, "Bangkok referral\nhospital, 2019", "National outbreak\ninvestigation, 2023"),
      levels = c("Bangkok referral\nhospital, 2019", "National outbreak\ninvestigation, 2023")
    ))
  stopifnot(unique(sub$comparison_status) == "noncomparable_snapshots")

  p_count <- ggplot(sub, aes(snapshot, tests)) +
    geom_col(width = 0.58, fill = alpha(tests_colour, 0.72), colour = tests_colour, linewidth = 0.18) +
    geom_text(aes(label = comma(tests)), vjust = -0.45, size = figure_text$annotation, family = figure_font_family, colour = tests_colour) +
    scale_y_continuous(labels = comma, limits = c(0, NA), expand = expansion(mult = c(0, 0.18))) +
    labs(tag = tag, x = NULL, y = "Samples tested") +
    theme_nature(grid = "y") +
    aligned_track_axes(tests_colour, top = TRUE)

  p_positivity <- ggplot(sub, aes(snapshot, positivity_percent)) +
    geom_point(shape = 21, size = 2.1, stroke = 0.55, fill = "white", colour = positivity_colour) +
    geom_text(
      aes(label = sprintf("%.1f%%", positivity_percent)), vjust = -1.0,
      size = figure_text$annotation, family = figure_font_family, fontface = "bold", colour = positivity_colour
    ) +
    scale_y_continuous(labels = label_number(suffix = "%", accuracy = 1), limits = c(0, NA), expand = expansion(mult = c(0, 0.28))) +
    labs(x = NULL, y = "Positive samples (%)") +
    theme_nature(grid = "y") +
    aligned_track_axes(positivity_colour)

  (p_count / p_positivity) + plot_layout(heights = c(1, 1))
}

render_testing_boundaries <- function() {
  figure <- make_temporal_panel("USA", "a") | make_thailand_boundary_panel("b")
  ragg::agg_png(file.path("manuscript", "appendix", "figures", "supplement_testing_comparability.png"),
                width = nature_full_width_mm, height = 78, units = "mm", res = 600, background = "white")
  print(figure)
  dev.off()
  message("Wrote Supplementary Figure 28: unlinked US series and non-comparable Thai snapshots.")
}
if (sys.nframe() == 0L) render_testing_boundaries()
