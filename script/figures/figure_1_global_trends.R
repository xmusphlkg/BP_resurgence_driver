#!/usr/bin/env Rscript
# Main Figure 1: nine-country trajectories, comparator sensitivity and baseline uncertainty.
# The legacy renderer filename is retained by the established build interface.
source(file.path("script", "figures", "_common.R"))

monthly <- read_csv_checked(fd_path("figure_1_context_monthly.csv"),
  c("iso3", "country", "date", "year", "observed_cases", "reported_median",
    "reported_predictive_lower", "reported_predictive_upper", "observation_available", "analysis_role"))
baselines <- read_csv_checked(fd_path("figure_1_context_baselines.csv"),
  c("iso3", "country", "baseline_id", "ratio", "baseline_incidence", "post_incidence",
    "audit_periodic_ratio_lcl", "audit_periodic_ratio_ucl", "audit_periodic_ratio_ucl_unbounded",
    "audit_periodic_bootstrap_status", "audit_periodic_bootstrap_successes"))
stopifnot(nrow(monthly) == 1188, setequal(unique(monthly$iso3), pilot_iso3),
  all(monthly$year >= 2015 & monthly$year <= 2025),
  all(monthly$analysis_role == "primary_calibration_through_2025"),
  nrow(baselines) == 27, setequal(unique(baselines$iso3), pilot_iso3),
  all(is.finite(baselines$ratio)), all(baselines$ratio > 0),
  all(abs(baselines$ratio - baselines$post_incidence / baselines$baseline_incidence) < 1e-10))
paired <- baselines %>% group_by(iso3) %>%
  summarise(n_numerators = n_distinct(post_incidence), n_baselines = n_distinct(baseline_id), .groups = "drop")
stopifnot(all(paired$n_numerators == 1), all(paired$n_baselines == 3))
country_order <- pilot_iso3
country_names <- baselines %>% distinct(iso3, country) %>%
  mutate(country = country_display_name(country))
country_labels <- country_names %>%
  mutate(country_y = length(country_order) + 1 - match(iso3, country_order)) %>% arrange(country_y)
country_axis <- function() scale_y_continuous(
  breaks = country_labels$country_y, labels = country_labels$country,
  limits = c(0.5, length(country_order) + 0.5), expand = expansion(mult = 0))

# a: Promote the monthly trajectories from Supplementary Figure 17. The full
# country-month grid includes NA gaps so lines do not bridge missing observations.
monthly <- monthly %>% mutate(date = as.Date(date),
  country_panel = factor(country_display_name(country),
    levels = country_names$country[match(country_order, country_names$iso3)]))
p_monthly <- ggplot(monthly, aes(date)) +
  annotate("rect", xmin = as.Date("2020-01-01"), xmax = as.Date("2022-12-31"),
    ymin = -Inf, ymax = Inf, fill = "#F0F1F2") +
  geom_ribbon(aes(ymin = reported_predictive_lower, ymax = reported_predictive_upper),
    fill = clr("blue"), alpha = .17, colour = NA, na.rm = TRUE) +
  geom_line(aes(y = reported_median, colour = "Fitted median (95% predictive band)"),
    linewidth = .34, na.rm = TRUE) +
  geom_point(aes(y = observed_cases, colour = "Observed"), size = .55, alpha = .75, na.rm = TRUE) +
  facet_wrap(~country_panel, ncol = 3, scales = "free_y") +
  scale_colour_manual(values = c("Observed" = clr("dark_grey"),
    "Fitted median (95% predictive band)" = clr("blue")),
    breaks = c("Observed", "Fitted median (95% predictive band)"), name = NULL) +
  guides(colour = guide_legend(override.aes = list(linetype = c(0, 1), shape = c(16, NA),
    alpha = 1, linewidth = .45, size = .9))) +
  scale_x_date(breaks = as.Date(c("2015-01-01", "2020-01-01", "2025-01-01")),
    date_labels = "%Y", limits = as.Date(c("2015-01-01", "2025-12-31")),
    expand = expansion(mult = c(0, .015))) +
  scale_y_continuous(breaks = scales::breaks_pretty(n = 3),
    labels = scales::label_number(scale_cut = scales::cut_short_scale()),
    expand = expansion(mult = c(0, .08))) +
  expand_limits(y = 0) +
  labs(tag = "a", x = NULL, y = "Monthly reports (country-specific scales)") +
  theme_manuscript(plot_margin = margin(3, 4, 2, 4)) +
  theme(panel.spacing.x = unit(4, "mm"), panel.spacing.y = unit(2.0, "mm"),
    legend.margin = margin(0, 0, 0, 0))

# b: The numerator is held fixed; the range is not an uncertainty interval.
baseline_order <- c("mean_2010_2019", "mean_2015_2019", "periodic")
baseline_labels <- c("2010-19 mean", "2015-19 mean", "Periodic")
baselines <- baselines %>% mutate(country_y = length(country_order) + 1 - match(iso3, country_order),
  baseline_id = factor(baseline_id, levels = baseline_order),
  point_y = country_y + c(.15, 0, -.15)[as.integer(baseline_id)])
baseline_ranges <- baselines %>% group_by(iso3, country_y) %>%
  summarise(lower = min(ratio), upper = max(ratio), .groups = "drop")
p_baselines <- ggplot(baselines, aes(ratio, point_y)) +
  geom_vline(xintercept = 1, linewidth = .28, linetype = "22", colour = clr("grey")) +
  geom_segment(data = baseline_ranges,
    aes(x = lower, xend = upper, y = country_y, yend = country_y),
    inherit.aes = FALSE, linewidth = .48, colour = clr("light_grey")) +
  geom_point(aes(colour = baseline_id, shape = baseline_id), size = 1.7) +
  scale_colour_manual(values = c(clr("blue"), clr("orange"), clr("black")),
    breaks = baseline_order, labels = baseline_labels, name = NULL) +
  scale_shape_manual(values = c(16, 17, 18), breaks = baseline_order,
    labels = baseline_labels, name = NULL) +
  scale_x_log10(limits = c(min(baselines$ratio) * .78, max(baselines$ratio) * 1.2),
    breaks = c(.5, 1, 2, 4, 8, 16), labels = c("0.5", "1", "2", "4", "8", "16")) +
  country_axis() + labs(tag = "b", x = "2023-25 mean incidence / comparator", y = NULL) +
  theme_manuscript(plot_margin = margin(4, 6, 3, 4)) +
  theme(legend.key.width = unit(3.1, "mm"), legend.spacing.x = unit(.6, "mm"))

# c: Promote Supplementary Figure 27f. The point estimates match the black
# diamonds in b; uncertainty is conditional on the periodic baseline model.
# An unbounded upper interval remains infinite in the data and receives an arrow.
periodic <- baselines %>% filter(baseline_id == "periodic")
stopifnot(nrow(periodic) == 9,
  all(is.finite(periodic$audit_periodic_ratio_lcl)),
  all(periodic$audit_periodic_ratio_lcl > 0),
  all(periodic$audit_periodic_ratio_lcl <= periodic$ratio),
  all(periodic$audit_periodic_ratio_ucl >= periodic$ratio),
  all(is.infinite(periodic$audit_periodic_ratio_ucl) == periodic$audit_periodic_ratio_ucl_unbounded),
  all(periodic$audit_periodic_bootstrap_successes == 2000),
  all(periodic$audit_periodic_bootstrap_status == "hc2_rademacher_wild_residual_bootstrap"))
interval_cap <- max(periodic$audit_periodic_ratio_ucl[is.finite(periodic$audit_periodic_ratio_ucl)]) * 1.2
periodic <- periodic %>% mutate(upper_for_plot = ifelse(audit_periodic_ratio_ucl_unbounded,
  interval_cap, audit_periodic_ratio_ucl))
p_uncertainty <- ggplot(periodic, aes(ratio, country_y)) +
  geom_vline(xintercept = 1, linewidth = .28, linetype = "22", colour = clr("grey")) +
  geom_segment(aes(x = audit_periodic_ratio_lcl, xend = upper_for_plot,
    yend = country_y), linewidth = .48, colour = clr("grey")) +
  geom_segment(data = filter(periodic, audit_periodic_ratio_ucl_unbounded),
    aes(x = upper_for_plot / 1.2, xend = upper_for_plot, yend = country_y),
    arrow = arrow(length = unit(1.3, "mm"), type = "open"),
    linewidth = .48, colour = clr("grey")) +
  geom_point(shape = 18, size = 1.9, colour = clr("black")) +
  scale_x_log10(breaks = c(.25, .5, 1, 2, 4, 8, 16, 32),
    labels = c("0.25", "0.5", "1", "2", "4", "8", "16", "32"),
    limits = c(min(periodic$audit_periodic_ratio_lcl) * .84, interval_cap * 1.08)) +
  country_axis() + labs(tag = "c", x = "2023-25 incidence / periodic comparator\n(95% conditional bootstrap interval)", y = NULL) +
  theme_manuscript(plot_margin = margin(4, 4, 3, 4))

figure <- p_monthly + p_baselines + p_uncertainty +
  plot_layout(design = "AA\nBC", widths = c(1.04, 1), heights = c(1.35, 1))
save_main_figure(figure, "figure_1_nine_country_context", nature_full_width_mm, 170)
