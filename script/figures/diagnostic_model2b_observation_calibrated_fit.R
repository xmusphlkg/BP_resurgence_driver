#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

rolling_mean_center <- function(x, window = 3L) {
  n <- length(x)
  out <- rep(NA_real_, n)
  half <- floor(window / 2)
  for (i in seq_len(n)) {
    lo <- max(1L, i - half)
    hi <- min(n, i + half)
    values <- x[lo:hi]
    out[[i]] <- mean(values[is.finite(values)], na.rm = TRUE)
  }
  out[!is.finite(out)] <- 0
  out
}

monthly <- read_csv_checked(
  fd_path("supplement_model2b_monthly_fit_diagnostics.csv"),
  c(
    "iso3", "country", "year", "month", "observed_cases",
    "predicted_median", "predicted_lower_95", "predicted_upper_95",
    "fit_role_label"
  )
) %>%
  mutate(
    year = as.integer(year),
    month = as.integer(month),
    observed_cases = as.numeric(observed_cases),
    predicted_median = as.numeric(predicted_median),
    predicted_lower_95 = as.numeric(predicted_lower_95),
    predicted_upper_95 = as.numeric(predicted_upper_95),
    date = as.Date(sprintf("%04d-%02d-15", year, month)),
    fit_role_label = if_else(fit_role_label == "fit_likelihood", "Fitted 2015-2025", fit_role_label)
  )

annual_bridge <- monthly %>%
  filter(fit_role_label == "Fitted 2015-2025", is.finite(observed_cases), is.finite(predicted_median)) %>%
  group_by(iso3, year) %>%
  summarise(
    annual_residual_ratio = (sum(observed_cases, na.rm = TRUE) + 1.0) /
      (sum(predicted_median, na.rm = TRUE) + 1.0),
    .groups = "drop"
  ) %>%
  mutate(
    annual_residual_ratio = pmin(pmax(annual_residual_ratio, 0.02), 80.0),
    log_annual_bridge = log(annual_residual_ratio)
  )

calibrated <- monthly %>%
  left_join(annual_bridge, by = c("iso3", "year")) %>%
  arrange(iso3, year, month) %>%
  group_by(iso3) %>%
  mutate(
    raw_month_ratio = if_else(
      fit_role_label == "Fitted 2015-2025" & is.finite(observed_cases) & is.finite(predicted_median),
      (observed_cases + 1.0) / (predicted_median + 1.0),
      NA_real_
    ),
    raw_month_ratio = pmin(pmax(raw_month_ratio, 0.02), 80.0),
    log_month_raw = log(raw_month_ratio),
    log_month_raw = if_else(is.finite(log_month_raw), log_month_raw, log_annual_bridge),
    log_month_smooth = rolling_mean_center(log_month_raw, 3L),
    log_annual_bridge = if_else(is.finite(log_annual_bridge), log_annual_bridge, log_month_smooth),
    log_calibration_factor = 0.65 * log_month_smooth + 0.35 * log_annual_bridge,
    calibration_factor = pmin(pmax(exp(log_calibration_factor), 0.02), 80.0),
    calibrated_median = predicted_median * calibration_factor,
    calibrated_lower_95 = predicted_lower_95 * calibration_factor,
    calibrated_upper_95 = predicted_upper_95 * calibration_factor
  ) %>%
  ungroup()

fit_quality <- calibrated %>%
  filter(fit_role_label == "Fitted 2015-2025", is.finite(observed_cases), is.finite(predicted_median)) %>%
  group_by(iso3, country) %>%
  summarise(
    n_months = n(),
    uncalibrated_log_rmse = sqrt(mean((log1p(observed_cases) - log1p(predicted_median))^2, na.rm = TRUE)),
    calibrated_log_rmse = sqrt(mean((log1p(observed_cases) - log1p(calibrated_median))^2, na.rm = TRUE)),
    uncalibrated_median_fold_error = exp(median(abs(log((observed_cases + 0.5) / (predicted_median + 0.5))), na.rm = TRUE)),
    calibrated_median_fold_error = exp(median(abs(log((observed_cases + 0.5) / (calibrated_median + 0.5))), na.rm = TRUE)),
    uncalibrated_interval_coverage_95 = mean(
      observed_cases >= predicted_lower_95 & observed_cases <= predicted_upper_95,
      na.rm = TRUE
    ),
    calibrated_interval_coverage_95 = mean(
      observed_cases >= calibrated_lower_95 & observed_cases <= calibrated_upper_95,
      na.rm = TRUE
    ),
    median_calibration_factor = median(calibration_factor, na.rm = TRUE),
    max_calibration_factor = max(calibration_factor, na.rm = TRUE),
    months_outside_twofold_calibration = mean(calibration_factor < 0.5 | calibration_factor > 2.0, na.rm = TRUE),
    .groups = "drop"
  )

country_order <- calibrated %>%
  distinct(iso3, country) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_panel = format_country_panel(iso3, country),
    country_label = format_country_inline(iso3, country)
  ) %>%
  arrange(pilot_rank, country)

plot_data <- calibrated %>%
  select(-any_of(c("country_panel", "country_label", "pilot_rank"))) %>%
  left_join(country_order %>% select(iso3, country_panel), by = "iso3") %>%
  mutate(country_panel = factor(country_panel, levels = country_order$country_panel)) %>%
  filter(fit_role_label == "Fitted 2015-2025")

p_fit <- ggplot(plot_data, aes(date)) +
  annotate("rect", xmin = as.Date("2020-01-01"), xmax = as.Date("2022-12-31"), ymin = -Inf, ymax = Inf, fill = pal["light_grey"], alpha = 0.34) +
  annotate("rect", xmin = as.Date("2023-01-01"), xmax = as.Date("2025-12-31"), ymin = -Inf, ymax = Inf, fill = pal["paper"], alpha = 0.70) +
  geom_ribbon(aes(ymin = pmax(calibrated_lower_95, 0), ymax = pmax(calibrated_upper_95, 0)), fill = clr("orange"), alpha = 0.15) +
  geom_line(aes(y = pmax(calibrated_median, 0), colour = "Observation-calibrated"), linewidth = 0.36) +
  geom_line(aes(y = pmax(predicted_median, 0), colour = "Uncalibrated Model 2B"), linewidth = 0.28, alpha = 0.62) +
  geom_point(aes(y = pmax(observed_cases, 0)), colour = clr("black"), size = 0.55, alpha = 0.75, na.rm = TRUE) +
  facet_wrap(~ country_panel, ncol = 3, scales = "free_y") +
  scale_y_continuous(labels = label_short, expand = expansion(mult = c(0.02, 0.08))) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y", expand = expansion(mult = c(0.01, 0.03))) +
  scale_colour_manual(values = c("Observation-calibrated" = clr("red"), "Uncalibrated Model 2B" = clr("blue"))) +
  labs(x = NULL, y = "Monthly reported cases", colour = NULL) +
  theme_nature(base_size = 7.2) +
  theme(
    strip.text = element_text(size = 6.5),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom"
  )

quality_long <- fit_quality %>%
  left_join(country_order %>% select(iso3, country_label), by = "iso3") %>%
  transmute(
    country_label,
    `Raw RMSE` = uncalibrated_log_rmse,
    `Calibrated RMSE` = calibrated_log_rmse,
    `Raw coverage` = uncalibrated_interval_coverage_95,
    `Calibrated coverage` = calibrated_interval_coverage_95,
    `Months >2x adj.` = months_outside_twofold_calibration
  ) %>%
  pivot_longer(-country_label, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(metric, levels = c("Raw RMSE", "Calibrated RMSE", "Raw coverage", "Calibrated coverage", "Months >2x adj.")),
    status = case_when(
      str_detect(metric, "RMSE") & value <= 0.50 ~ "supported",
      str_detect(metric, "RMSE") & value <= 1.00 ~ "partial",
      str_detect(metric, "coverage") & value >= 0.75 ~ "supported",
      str_detect(metric, "coverage") & value >= 0.50 ~ "partial",
      str_detect(metric, ">2x") & value <= 0.15 ~ "supported",
      str_detect(metric, ">2x") & value <= 0.35 ~ "partial",
      TRUE ~ "caution"
    ),
    label = if_else(
      str_detect(metric, "coverage|>2x"),
      paste0(round(100 * value), "%"),
      label_number(accuracy = 0.01)(value)
    ),
    country_label = factor(country_label, levels = rev(country_order$country_label)),
    status = factor(status, levels = names(evidence_palette))
  )

p_quality <- ggplot(quality_long, aes(metric, country_label, fill = status)) +
  geom_tile(colour = "white", linewidth = 0.42) +
  geom_text(aes(label = label), size = 1.85, colour = pal["black"]) +
  scale_fill_manual(
    values = evidence_palette,
    breaks = c("supported", "partial", "caution"),
    labels = c(supported = "Better", partial = "Mixed", caution = "Poor")
  ) +
  labs(x = NULL, y = NULL, fill = "Diagnostic") +
  theme_nature(base_size = 7.1) +
  theme(
    axis.text.x = element_text(size = 5.7),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )

figure <- panel_tags(
  (p_fit / p_quality) +
    plot_layout(heights = c(1.42, 0.58), guides = "collect")
) &
  theme(legend.position = "bottom")

readr::write_csv(calibrated, fd_path("diagnostic_model2b_observation_calibrated_monthly_fit.csv"))
readr::write_csv(fit_quality, fd_path("diagnostic_model2b_observation_calibrated_fit_quality.csv"))
save_nature(figure, "diagnostic_model2b_observation_calibrated_monthly_fit.png", nature_full_width_mm, nature_max_height_mm)
