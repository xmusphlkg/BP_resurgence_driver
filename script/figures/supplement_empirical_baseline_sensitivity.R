#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

# Figure job: show whether the direction and magnitude of each empirical 2024
# comparison survives a change from an adjacent-year baseline to pre-pandemic
# last-year, mean and maximum comparators.

required <- c(
  "iso3", "country", "pathway", "display_metric", "value_2024",
  "comparator_type", "comparator_label", "n_baseline_years",
  "baseline_value", "ratio_2024_to_baseline", "comparison_status",
  "evidence_role"
)
dat <- read_csv_checked(
  fd_path("supplement_empirical_baseline_sensitivity.csv"),
  required = required
)

stopifnot(
  nrow(dat) == 38L,
  all(dat$comparison_status %in% c("available", "unavailable")),
  all(dat$n_baseline_years[dat$comparator_type == "prepandemic_mean"] >= 3),
  all(is.finite(dat$ratio_2024_to_baseline[dat$comparison_status == "available"])),
  all(dat$ratio_2024_to_baseline[dat$comparison_status == "available"] > 0)
)

comparator_levels <- c(
  "adjacent_2023", "last_prepandemic_2019",
  "prepandemic_mean", "prepandemic_maximum"
)
comparator_labels <- c(
  adjacent_2023 = "2023",
  last_prepandemic_2019 = "2019",
  prepandemic_mean = "Pre-pandemic mean",
  prepandemic_maximum = "Pre-pandemic maximum"
)
comparator_colours <- c(
  adjacent_2023 = clr("orange"),
  last_prepandemic_2019 = clr("grey"),
  prepandemic_mean = clr("blue"),
  prepandemic_maximum = clr("black")
)
comparator_shapes <- c(
  adjacent_2023 = 16,
  last_prepandemic_2019 = 17,
  prepandemic_mean = 15,
  prepandemic_maximum = 18
)

plot_dat <- dat |>
  filter(comparison_status == "available") |>
  mutate(
    comparator_type = factor(comparator_type, levels = comparator_levels),
    ratio_label = paste0(formatC(ratio_2024_to_baseline, digits = 2, format = "fg"), "×")
  )

ratio_scale <- scale_x_log10(
  limits = c(0.1, 64),
  breaks = c(0.125, 0.5, 1, 2, 8, 32),
  labels = c("0.125×", "0.5×", "1×", "2×", "8×", "32×"),
  expand = expansion(mult = c(0.02, 0.10))
)

base_plot <- function(data, tag, metric_levels, annotate = NULL) {
  stopifnot(all(data$display_metric %in% metric_levels))
  p <- data |>
    mutate(display_metric = factor(display_metric, levels = rev(metric_levels))) |>
    ggplot(aes(ratio_2024_to_baseline, display_metric)) +
    geom_vline(xintercept = 1, linetype = "dotted", linewidth = 0.38, colour = clr("dark_grey")) +
    geom_segment(
      aes(x = pmin(ratio_2024_to_baseline, 1), xend = pmax(ratio_2024_to_baseline, 1), yend = display_metric),
      linewidth = 0.24, colour = clr("light_grey"), alpha = 0.72
    ) +
    geom_point(
      aes(colour = comparator_type, shape = comparator_type),
      size = 2.15, stroke = 0.55
    ) +
    ratio_scale +
    scale_colour_manual(
      values = comparator_colours,
      labels = comparator_labels,
      drop = FALSE
    ) +
    scale_shape_manual(
      values = comparator_shapes,
      labels = comparator_labels,
      drop = FALSE
    ) +
    scale_y_discrete(labels = function(x) ifelse(x == "Tests eligible for positivity", "Eligible tests", x)) +
    labs(tag = tag, x = "2024-to-comparator ratio", y = NULL, colour = NULL, shape = NULL) +
    theme_nature(grid = "x") +
    theme(
      legend.position = "none",
      axis.text.y = element_text(size = figure_text$axis_text)
    )
  if (!is.null(annotate)) {
    p <- p + annotate
  }
  p
}

p_a <- base_plot(
  plot_dat |> filter(pathway == "testing"),
  "a",
  c("Tests eligible for positivity", "Test positivity")
) +
  facet_grid(country ~ ., scales = "free_y", space = "free_y", switch = "y") +
  theme(strip.placement = "outside", strip.background = element_blank(),
        strip.text.y.left = element_text(angle = 0, face = "bold"))

p_b <- base_plot(
  plot_dat |> filter(country %in% c("France", "England"), pathway == "severe"),
  "b",
  c("Infant hospitalisations", "Infant deaths")
) +
  facet_grid(country ~ ., scales = "free_y", space = "free_y", switch = "y") +
  theme(strip.placement = "outside", strip.background = element_blank(),
        strip.text.y.left = element_text(angle = 0, face = "bold"))

p_c <- base_plot(
  plot_dat |> filter(country == "Brazil"),
  "c",
  c("Infant hospitalisations", "Infant deaths", "Infant reported cases")
) +
  annotate(
    "label", x = 0.14, y = 3.40,
    label = "2024 was below the 2010–19\nmean for all three streams",
    hjust = 0, vjust = 1, size = 1.9, family = figure_font_family,
    linewidth = 0.18, fill = "white", colour = clr("dark_grey")
  )

p_d <- base_plot(
  plot_dat |> filter(country == "China"),
  "d",
  c("Reported cases", "All-age reported deaths")
) +
  annotate(
    "label", x = 0.14, y = 2.34,
    label = "Cases and deaths share\nthe notification channel",
    hjust = 0, vjust = 1, size = 1.9, family = figure_font_family,
    linewidth = 0.18, fill = "#FFF7F0", colour = clr("red")
  )

legend <- cowplot::get_legend(
  ggplot(plot_dat, aes(ratio_2024_to_baseline, display_metric)) +
    geom_point(aes(colour = comparator_type, shape = comparator_type), size = 2.2) +
    scale_colour_manual(values = comparator_colours, labels = comparator_labels, drop = FALSE) +
    scale_shape_manual(values = comparator_shapes, labels = comparator_labels, drop = FALSE) +
    labs(colour = NULL, shape = NULL) +
    guides(
      colour = guide_legend(nrow = 1, byrow = TRUE,
                            override.aes = list(size = 2.4, shape = unname(comparator_shapes))),
      shape = "none"
    ) +
    theme_manuscript_void() +
    theme(legend.position = "bottom", legend.margin = margin(0, 0, 0, 0))
)

figure <- cowplot::plot_grid(
  cowplot::plot_grid(p_a, p_b, p_c, p_d, ncol = 2, rel_widths = c(1.08, 0.92)),
  legend,
  ncol = 1,
  rel_heights = c(1, 0.08)
)

save_supplement_figure(
  figure,
  "supplement_empirical_baseline_sensitivity",
  nature_full_width_mm,
  150
)
message("Wrote Supplementary Figure 26 baseline-sensitivity forest plots.")
