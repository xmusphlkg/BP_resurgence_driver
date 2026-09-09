#!/usr/bin/env Rscript
# One claim: laboratory growth has an exact sample-count decomposition, while
# infant-outcome comparisons depend on source scope and historical denominator.
# Reuse the established typography/palette and independent laboratory axes.
source(file.path("script", "figures", "figure_2_testing_decomposition.R"))
library(tidyr)

baseline <- read_csv_checked(
  fd_path("supplement_empirical_baseline_sensitivity.csv"),
  required = c("iso3", "metric", "display_metric", "comparator_type", "baseline_value",
               "value_2024", "ratio_2024_to_baseline", "comparison_status")
)
events <- read_csv_checked(
  fd_path("figure_8_external_empirical_challenge.csv"),
  required = c("iso3", "metric", "value_2023", "value_2024",
               "value_2023_poisson_lower_95", "value_2023_poisson_upper_95",
               "value_2024_poisson_lower_95", "value_2024_poisson_upper_95")
)

fr <- d |> filter(iso3 == "FRA", year %in% c(2019, 2024)) |> arrange(year)
stopifnot(nrow(fr) == 2L, identical(fr$year, c(2019, 2024)))
log_positive <- log(fr$positive_tests[2] / fr$positive_tests[1])
components <- tibble(
  component = factor(c("Test-count term", "Positivity term"), levels = c("Test-count term", "Positivity term")),
  value = c(log(fr$tests[2] / fr$tests[1]), log(fr$positivity_percent[2] / fr$positivity_percent[1]))
) |> mutate(share = value / log_positive,
            label = sprintf("%.2f log units\n%.1f%%", value, share * 100))
stopifnot(abs(sum(components$value) - log_positive) < 1e-10)
# This is an identity applied to observed counts, not an estimated causal effect.
p_decomposition <- ggplot(components, aes(component, value, fill = component)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = label), vjust = -0.35, size = 1.85, family = figure_font_family) +
  scale_fill_manual(values = c(`Test-count term` = clr("blue"), `Positivity term` = clr("orange"))) +
  scale_y_continuous(limits = c(0, 2.6), breaks = c(0, 1, 2)) +
  labs(tag = "c", title = "France: 2019–2024 positive-count growth",
       x = NULL, y = "Contribution to log growth") + theme_nature(grid = "y")

fr_events <- events |> filter(iso3 == "FRA", metric %in% c("infant_hospitalizations", "infant_deaths"))
stopifnot(nrow(fr_events) == 2L)
fr_long <- bind_rows(
  fr_events |> transmute(metric, year = "2023", value = value_2023,
                         lower = value_2023_poisson_lower_95, upper = value_2023_poisson_upper_95),
  fr_events |> transmute(metric, year = "2024", value = value_2024,
                         lower = value_2024_poisson_lower_95, upper = value_2024_poisson_upper_95)
) |> mutate(metric = recode(metric, infant_hospitalizations = "Infant hospitalisations",
                             infant_deaths = "Infant deaths"))
stopifnot(all(is.finite(fr_long$lower)), all(fr_long$lower <= fr_long$value),
          all(fr_long$value <= fr_long$upper))
p_fr_events <- ggplot(fr_long, aes(year, value)) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.10, linewidth = 0.32, colour = clr("orange")) +
  geom_point(size = 1.7, colour = clr("orange")) +
  geom_text(aes(y = upper, label = value), vjust = -0.5, size = 1.8, family = figure_font_family) +
  facet_wrap(vars(metric), scales = "free_y", nrow = 1) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.18))) +
  labs(tag = "d", title = "France: RENACOQ post-trough counts", x = NULL, y = "Recorded infant events") +
  theme_nature(grid = "y")

eng <- baseline |> filter(iso3 == "GBR", metric == "infant_deaths",
                           comparator_type %in% c("prepandemic_mean", "prepandemic_maximum"))
stopifnot(nrow(eng) == 2L, all(eng$value_2024 == 11))
eng_values <- bind_rows(
  eng |> transmute(period = if_else(comparator_type == "prepandemic_mean", "2001–19\nmean", "2001–19\nmaximum"), value = baseline_value),
  tibble(period = "2024", value = unique(eng$value_2024))
) |> mutate(period = factor(period, levels = c("2001–19\nmean", "2024", "2001–19\nmaximum")))
p_eng <- ggplot(eng_values, aes(period, value)) +
  geom_col(aes(fill = period == "2024"), width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = format(round(value, 2), trim = TRUE)), vjust = -0.5,
            size = 1.8, family = figure_font_family) +
  scale_fill_manual(values = c(`FALSE` = clr("grey"), `TRUE` = clr("orange"))) +
  scale_y_continuous(limits = c(0, 17), breaks = c(0, 5, 10, 15)) +
  labs(tag = "e", title = "England: infant-death comparators", x = NULL, y = "Annual infant deaths") +
  theme_nature(grid = "y")

bra <- baseline |> filter(iso3 == "BRA", comparison_status == "available",
                           comparator_type %in% c("last_prepandemic_2019", "prepandemic_mean", "prepandemic_maximum")) |>
  mutate(comparator = factor(recode(comparator_type, last_prepandemic_2019 = "2019",
                                    prepandemic_mean = "2010–19 mean", prepandemic_maximum = "2010–19 maximum"),
                              levels = c("2019", "2010–19 mean", "2010–19 maximum")),
         display_metric = factor(display_metric, levels = c("Infant reported cases", "Infant deaths", "Infant hospitalisations")))
stopifnot(nrow(bra) == 9L, !anyNA(bra$display_metric), all(is.finite(bra$ratio_2024_to_baseline)),
          all(bra$ratio_2024_to_baseline > 0))
p_bra <- ggplot(bra, aes(ratio_2024_to_baseline, display_metric, colour = comparator, shape = comparator)) +
  geom_vline(xintercept = 1, linetype = "dotted", linewidth = 0.3, colour = clr("grey")) +
  geom_point(size = 1.9) +
  scale_colour_manual(values = c(`2019` = clr("orange"), `2010–19 mean` = clr("blue"), `2010–19 maximum` = clr("black"))) +
  scale_shape_manual(values = c(`2019` = 17, `2010–19 mean` = 16, `2010–19 maximum` = 18)) +
  scale_x_log10(breaks = c(0.125, 0.5, 1, 2, 4), labels = function(x) paste0(x, "×")) +
  labs(tag = "f", title = "Brazil: infant-stream comparators", x = "2024-to-comparator count ratio",
       y = NULL, colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2)) +
  theme_nature(grid = "x") + theme(legend.position = "bottom")

panel_fr <- make_temporal_panel("FRA", "a") + plot_annotation(title = "France: 3Labos PCR network", theme = theme(plot.title = element_text(size = 7)))
panel_eng <- make_temporal_panel("GBR", "b") + plot_annotation(title = "England: mixed assay pathways", theme = theme(plot.title = element_text(size = 7)))
figure <- (panel_fr | panel_eng) / (p_decomposition | p_fr_events) / (p_eng | p_bra) +
  plot_layout(heights = c(1.35, 0.9, 0.9)) &
  theme(plot.title = element_text(size = 6.5, face = "bold", hjust = 0))
stem <- "figure_4_empirical_comparability"
grDevices::cairo_pdf(fig_path(paste0(stem, ".pdf")), width = 183 / 25.4, height = 178 / 25.4)
print(figure)
dev.off()
ragg::agg_png(fig_path(paste0(stem, ".png")), width = 183, height = 178, units = "mm", res = 600, background = "white")
print(figure)
dev.off()
render_testing_boundaries()
message("Wrote consolidated main Figure 4 and Supplementary Figure 28.")
