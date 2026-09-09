#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

ppc_path <- out_path("model2c_posterior_predictive.csv")
if (file.exists(ppc_path)) {
  ppc <- read_csv_checked(
    ppc_path,
    c(
      "iso3", "country", "draw", "year", "month", "observed_cases",
      "reported_cases_predicted", "reported_cases_predictive", "fit_role", "model2c_output_version"
    )
  ) %>%
    require_model2c_current_version("Supplement monthly fit posterior predictive") %>%
    mutate(
      year = as.integer(year),
      month = as.integer(month),
      observed_cases = as.numeric(observed_cases),
      reported_cases_predicted = as.numeric(reported_cases_predicted),
      reported_cases_predictive = as.numeric(reported_cases_predictive)
    )

  monthly_summary <- ppc %>%
    group_by(iso3, country, year, month, fit_role) %>%
    summarise(
      observed_cases = first(na.omit(observed_cases), default = NA_real_),
      predicted_median = median(reported_cases_predicted, na.rm = TRUE),
      predicted_lower_95 = quantile(reported_cases_predicted, 0.025, na.rm = TRUE),
      predicted_upper_95 = quantile(reported_cases_predicted, 0.975, na.rm = TRUE),
      predictive_lower_95 = quantile(reported_cases_predictive, 0.025, na.rm = TRUE),
      predictive_upper_95 = quantile(reported_cases_predictive, 0.975, na.rm = TRUE),
      .groups = "drop"
    )
} else {
  monthly_summary <- read_csv_checked(
    fd_path("supplement_model2c_monthly_fit_diagnostics.csv"),
    c(
      "iso3", "country", "year", "month", "fit_role", "observed_cases",
      "predicted_median", "predicted_lower_95", "predicted_upper_95", "model2c_output_version"
    )
  ) %>%
    require_model2c_current_version("Supplement monthly fit diagnostics fallback")
}

if (!"predictive_lower_95" %in% names(monthly_summary)) {
  monthly_summary$predictive_lower_95 <- monthly_summary$predicted_lower_95
}
if (!"predictive_upper_95" %in% names(monthly_summary)) {
  monthly_summary$predictive_upper_95 <- monthly_summary$predicted_upper_95
}

monthly_summary <- monthly_summary %>%
  mutate(
    year = as.integer(year),
    month = as.integer(month),
    observed_cases = as.numeric(observed_cases),
    predicted_median = as.numeric(predicted_median),
    predicted_lower_95 = as.numeric(predicted_lower_95),
    predicted_upper_95 = as.numeric(predicted_upper_95),
    predictive_lower_95 = as.numeric(predictive_lower_95),
    predictive_upper_95 = as.numeric(predictive_upper_95),
    date = as.Date(sprintf("%04d-%02d-15", year, month)),
    fit_role_label = case_when(
      fit_role == "fit_likelihood" ~ "Monthly likelihood",
      fit_role == "state_history_no_monthly_observation" ~ "State history",
      fit_role == "posterior_predictive_check" ~ "2026 PPC",
      fit_role == "posterior_predictive_unobserved_month" ~ "2026 unobserved",
      fit_role == "2026 PPC" ~ "2026 PPC",
      TRUE ~ "Monthly likelihood"
    )
  ) %>%
  select(-any_of(c("pilot_rank", "country_label", "country_panel"))) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_panel = format_country_panel(iso3, country),
    country_label = format_country_inline(iso3, country),
    country_panel = factor(country_panel, levels = format_country_panel(pilot_iso3, country_abbrev[pilot_iso3])),
    country_label = factor(country_label, levels = rev(format_country_inline(pilot_iso3, country_abbrev[pilot_iso3])))
  )

plot_data <- monthly_summary %>%
  filter(is.finite(predicted_median), is.finite(predicted_lower_95), is.finite(predicted_upper_95))

p_monthly <- ggplot(plot_data, aes(date)) +
  annotate("rect", xmin = as.Date("2020-01-01"), xmax = as.Date("2022-12-31"), ymin = -Inf, ymax = Inf, fill = pal["light_grey"], alpha = 0.34) +
  annotate("rect", xmin = as.Date("2023-01-01"), xmax = as.Date("2025-12-31"), ymin = -Inf, ymax = Inf, fill = pal["paper"], alpha = 0.70) +
  geom_ribbon(aes(ymin = pmax(predictive_lower_95, 0), ymax = pmax(predictive_upper_95, 0)), fill = clr("blue"), alpha = 0.10) +
  geom_ribbon(aes(ymin = pmax(predicted_lower_95, 0), ymax = pmax(predicted_upper_95, 0)), fill = clr("blue"), alpha = 0.22) +
  geom_line(aes(y = pmax(predicted_median, 0)), colour = clr("blue"), linewidth = 0.38) +
  geom_point(aes(y = pmax(observed_cases, 0), shape = fit_role_label), colour = pal["black"], size = 0.55, alpha = 0.78, na.rm = TRUE) +
  facet_wrap(~ country_panel, ncol = 3, scales = "free_y") +
  scale_y_continuous(labels = label_short, expand = expansion(mult = c(0.02, 0.08))) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y", expand = expansion(mult = c(0.01, 0.03))) +
  scale_shape_manual(values = c("Monthly likelihood" = 16, "2026 PPC" = 1, "State history" = 4, "2026 unobserved" = 3)) +
  labs(x = NULL, y = "Monthly reported cases", shape = NULL) +
  theme_nature() +
  theme(
    strip.text = element_text(size = figure_text$strip),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom"
  )

fit_quality <- monthly_summary %>%
  filter(fit_role_label == "Monthly likelihood", is.finite(observed_cases), is.finite(predicted_median)) %>%
  group_by(iso3, country, country_label) %>%
  summarise(
    n_months = n(),
    log_rmse = sqrt(mean((log1p(observed_cases) - log1p(predicted_median))^2, na.rm = TRUE)),
    median_abs_fold_error = exp(median(abs(log((observed_cases + 0.5) / (predicted_median + 0.5))), na.rm = TRUE)),
    expected_ui_inclusion_95 = mean(observed_cases >= predicted_lower_95 & observed_cases <= predicted_upper_95, na.rm = TRUE),
    predictive_interval_coverage_95 = mean(observed_cases >= predictive_lower_95 & observed_cases <= predictive_upper_95, na.rm = TRUE),
    median_expected_interval_relative_width = median((predicted_upper_95 - predicted_lower_95) / pmax(predicted_median, 1), na.rm = TRUE),
    median_predictive_relative_width = median((predictive_upper_95 - predictive_lower_95) / pmax(predicted_median, 1), na.rm = TRUE),
    .groups = "drop"
  )

peak_quality <- monthly_summary %>%
  filter(fit_role_label == "Monthly likelihood", is.finite(observed_cases), is.finite(predicted_median)) %>%
  group_by(iso3, country, country_label) %>%
  arrange(desc(observed_cases), year, month, .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  transmute(
    iso3,
    country,
    country_label,
    peak_year = year,
    peak_month = month,
    peak_observed_cases = observed_cases,
    peak_predicted_median = predicted_median,
    peak_predicted_lower_95 = predicted_lower_95,
    peak_predicted_upper_95 = predicted_upper_95,
    peak_predictive_lower_95 = predictive_lower_95,
    peak_predictive_upper_95 = predictive_upper_95,
    peak_capture_ratio = predicted_median / pmax(observed_cases, 1),
    peak_expected_interval_included = observed_cases >= predicted_lower_95 & observed_cases <= predicted_upper_95,
    peak_predictive_interval_covered = observed_cases >= predictive_lower_95 & observed_cases <= predictive_upper_95
  )

fit_quality <- fit_quality %>%
  left_join(
    peak_quality %>%
      select(
        iso3, peak_year, peak_month, peak_observed_cases, peak_predicted_median,
        peak_capture_ratio, peak_expected_interval_included, peak_predictive_interval_covered
      ),
    by = "iso3"
  )

quality_long <- fit_quality %>%
  transmute(
    country_label,
    `Monthly log RMSE` = log_rmse,
    `Median fold error` = median_abs_fold_error,
    `Predictive coverage` = predictive_interval_coverage_95,
    `Mean UI width` = median_expected_interval_relative_width,
    `Highest peak captured` = peak_capture_ratio
  ) %>%
  pivot_longer(-country_label, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(metric, levels = c("Monthly log RMSE", "Median fold error", "Predictive coverage", "Mean UI width", "Highest peak captured")),
    status = case_when(
      metric == "Monthly log RMSE" & value <= 0.70 ~ "supported",
      metric == "Monthly log RMSE" & value <= 1.10 ~ "partial",
      metric == "Median fold error" & value <= 2.0 ~ "supported",
      metric == "Median fold error" & value <= 3.5 ~ "partial",
      metric == "Predictive coverage" & value >= 0.75 ~ "supported",
      metric == "Predictive coverage" & value >= 0.50 ~ "partial",
      metric == "Mean UI width" & value <= 0.75 ~ "supported",
      metric == "Mean UI width" & value <= 1.25 ~ "partial",
      metric == "Highest peak captured" & value >= 0.75 ~ "supported",
      metric == "Highest peak captured" & value >= 0.50 ~ "partial",
      TRUE ~ "caution"
    ),
    label = case_when(
      metric == "Predictive coverage" ~ paste0(round(100 * value), "%"),
      metric == "Highest peak captured" ~ paste0(round(100 * value), "%"),
      metric == "Median fold error" ~ label_number(accuracy = 0.1)(value),
      metric == "Mean UI width" ~ label_number(accuracy = 0.1)(value),
      TRUE ~ label_number(accuracy = 0.01)(value)
    ),
    country_label = factor(country_label, levels = levels(monthly_summary$country_label)),
    status = factor(status, levels = names(evidence_palette))
  ) %>%
  group_by(metric) %>%
  mutate(
    metric_span = diff(range(value, na.rm = TRUE)),
    metric_span = if_else(is.finite(metric_span) & metric_span > 0, metric_span, 1),
    metric_high = max(value, na.rm = TRUE),
    label_at_right_edge = value >= metric_high - 0.12 * metric_span,
    label_x = if_else(
      label_at_right_edge,
      value - 0.04 * metric_span,
      value + 0.04 * metric_span
    ),
    label_x = if_else(is.finite(label_x), label_x, value + 0.04)
  ) %>%
  ungroup()

quality_refs <- tibble(
  metric = factor(
    c("Monthly log RMSE", "Median fold error", "Predictive coverage", "Mean UI width", "Highest peak captured"),
    levels = levels(quality_long$metric)
  ),
  reference = c(0.70, 2.0, 0.75, 0.75, 0.75)
)

p_quality <- ggplot(quality_long, aes(value, country_label, colour = status)) +
  geom_vline(
    data = quality_refs,
    aes(xintercept = reference),
    inherit.aes = FALSE,
    colour = pal["grey"],
    linewidth = 0.25,
    linetype = "22"
  ) +
  geom_point(size = 1.85, alpha = 0.95) +
  geom_text(
    aes(x = label_x, label = label, hjust = if_else(label_at_right_edge, 1, 0)),
    size = figure_text$annotation,
    colour = pal["dark_grey"],
    show.legend = FALSE
  ) +
  facet_wrap(~metric, nrow = 1, scales = "free_x") +
  scale_colour_manual(
    values = evidence_palette,
    breaks = c("supported", "partial", "caution"),
    labels = c(supported = "Better", partial = "Mixed", caution = "Poor")
  ) +
  labs(x = NULL, y = NULL, colour = "Monthly fit") +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text_dense),
    axis.text.y = element_text(size = figure_text$axis_text),
    strip.text = element_text(size = figure_text$strip_dense),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom"
  )

figure <- panel_tags(
  (p_monthly / p_quality) +
    plot_layout(heights = c(1.4, 0.62), guides = "collect")
) &
  theme(legend.position = "bottom")

readr::write_csv(monthly_summary, fd_path("supplement_model2c_monthly_fit_diagnostics.csv"))
readr::write_csv(fit_quality, fd_path("supplement_model2c_monthly_fit_quality.csv"))
readr::write_csv(peak_quality, fd_path("supplement_model2c_monthly_peak_fit_diagnostics.csv"))
save_nature(figure, "supplement_model2c_monthly_fit_diagnostics.png", nature_full_width_mm, nature_max_height_mm)
