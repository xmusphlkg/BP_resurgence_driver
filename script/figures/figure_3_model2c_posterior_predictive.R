#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

safe_sum <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else sum(x)
}

safe_first <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else x[[1]]
}

safe_median <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else stats::median(x)
}

safe_quantile <- function(x, prob) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else unname(stats::quantile(x, prob, na.rm = TRUE))
}

label_fit_role <- function(x) {
  case_when(
    x == "fit_likelihood" ~ "Monthly likelihood",
    x == "state_history_no_monthly_observation" ~ "State history",
    x == "posterior_predictive_check" ~ "2026 PPC",
    x == "posterior_predictive_unobserved_month" ~ "2026 unobserved",
    TRUE ~ as.character(x)
  )
}

month_index_to_date <- function(index) {
  index <- round(as.numeric(index))
  year <- 2023 + floor((index - 1) / 12)
  month <- ((index - 1) %% 12) + 1
  as.Date(sprintf("%04d-%02d-15", year, month))
}

label_monthly_cases_axis <- function(x) {
  case_when(
    is.na(x) ~ NA_character_,
    abs(x) >= 1e6 ~ paste0(str_replace(scales::number(x / 1e6, accuracy = 0.1), "\\.0$", ""), "M"),
    abs(x) >= 1e4 ~ paste0(scales::number(x / 1e3, accuracy = 1), "k"),
    abs(x) >= 1e3 ~ paste0(str_replace(scales::number(x / 1e3, accuracy = 0.1), "\\.0$", ""), "k"),
    TRUE ~ scales::comma(x, accuracy = 1)
  )
}

monthly_case_breaks <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(NULL)
  }
  breaks <- pretty(range(c(0, x), na.rm = TRUE), n = 4)
  breaks[breaks >= 0]
}

fit_summary <- read_csv_first_available(
  c(
    out_path("model2c_fit_summary.csv"),
    file.path(root, "manuscript", "tables", "model2c_fit_summary.csv")
  ),
  c(
    "iso3", "country", "posterior_draws", "max_rhat", "min_ess_bulk",
    "n_month_observations", "n_month_ppc_observations", "n_age_observations",
    "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 3 fit summary")

ppc <- read_csv_first_available(
  c(
    out_path("model2c_posterior_predictive.csv")
  ),
  c(
    "iso3", "country", "draw", "year", "month", "observed_cases",
    "true_infections_predicted", "symptomatic_infections_predicted",
    "reported_cases_predicted", "reported_cases_predictive", "fit_role",
    "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 3 posterior predictive") %>%
  mutate(
    year = as.integer(year),
    month = as.integer(month),
    draw = as.integer(draw),
    observed_cases = as.numeric(observed_cases),
    true_infections_predicted = as.numeric(true_infections_predicted),
    symptomatic_infections_predicted = as.numeric(symptomatic_infections_predicted),
    reported_cases_predicted = as.numeric(reported_cases_predicted),
    reported_cases_predictive = as.numeric(reported_cases_predictive)
  ) %>%
  mutate(
    reported_cases_predicted_comparison = if_else(
      (str_detect(fit_role, "^posterior_predictive") | fit_role == "state_history_no_monthly_observation") &
        !is.finite(observed_cases),
      NA_real_,
      reported_cases_predicted
    ),
    reported_cases_predictive_comparison = if_else(
      (str_detect(fit_role, "^posterior_predictive") | fit_role == "state_history_no_monthly_observation") &
        !is.finite(observed_cases),
      NA_real_,
      reported_cases_predictive
    ),
    true_infections_predicted_comparison = if_else(
      (str_detect(fit_role, "^posterior_predictive") | fit_role == "state_history_no_monthly_observation") &
        !is.finite(observed_cases),
      NA_real_,
      true_infections_predicted
    ),
    symptomatic_infections_predicted_comparison = if_else(
      (str_detect(fit_role, "^posterior_predictive") | fit_role == "state_history_no_monthly_observation") &
        !is.finite(observed_cases),
      NA_real_,
      symptomatic_infections_predicted
    )
  )

pilot <- ppc %>%
  distinct(iso3, country) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_panel = country_display_name(country),
    country_label = country_display_name(country)
  ) %>%
  arrange(pilot_rank, country)

annual_draw <- ppc %>%
  group_by(iso3, country, draw, year, fit_role) %>%
  summarise(
    observed_cases = safe_sum(observed_cases),
    reported_cases_predicted = safe_sum(reported_cases_predicted_comparison),
    reported_cases_predictive = safe_sum(reported_cases_predictive_comparison),
    true_infections_predicted = safe_sum(true_infections_predicted_comparison),
    symptomatic_infections_predicted = safe_sum(symptomatic_infections_predicted_comparison),
    .groups = "drop"
  )

annual_summary <- annual_draw %>%
  group_by(iso3, country, year, fit_role) %>%
  summarise(
    reported_median = safe_median(reported_cases_predicted),
    reported_lower = safe_quantile(reported_cases_predicted, 0.025),
    reported_upper = safe_quantile(reported_cases_predicted, 0.975),
    reported_predictive_lower = safe_quantile(reported_cases_predictive, 0.025),
    reported_predictive_upper = safe_quantile(reported_cases_predictive, 0.975),
    true_median = safe_median(true_infections_predicted),
    observed_cases = safe_first(observed_cases),
    .groups = "drop"
  ) %>%
  mutate(
    year = as.integer(year),
    fit_role = label_fit_role(fit_role),
    interval_label = "95% UI",
    expected_ui_method = "calibration_ensemble_percentile_2.5_97.5",
    predictive_ui_method = "negative_binomial_predictive_percentile_2.5_97.5"
  ) %>%
  left_join(pilot %>% select(iso3, country_panel), by = "iso3") %>%
  mutate(country_panel = factor(country_panel, levels = pilot$country_panel))

monthly_summary <- ppc %>%
  group_by(iso3, country, year, month, fit_role) %>%
  summarise(
    observed_cases = safe_first(observed_cases),
    reported_median = safe_median(reported_cases_predicted),
    reported_lower = safe_quantile(reported_cases_predicted, 0.025),
    reported_upper = safe_quantile(reported_cases_predicted, 0.975),
    reported_predictive_lower = safe_quantile(reported_cases_predictive, 0.025),
    reported_predictive_upper = safe_quantile(reported_cases_predictive, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    year = as.integer(year),
    month = as.integer(month),
    date = as.Date(sprintf("%04d-%02d-15", year, month)),
    fit_role = label_fit_role(fit_role),
    interval_label = "95% UI",
    expected_ui_method = "calibration_ensemble_percentile_2.5_97.5",
    predictive_ui_method = "negative_binomial_predictive_percentile_2.5_97.5"
  ) %>%
  left_join(pilot %>% select(iso3, country_panel, country_label), by = "iso3") %>%
  mutate(
    country_panel = factor(country_panel, levels = pilot$country_panel),
    country_label = factor(country_label, levels = rev(pilot$country_label))
  )

monthly_summary_plot <- monthly_summary %>%
  filter(is.finite(reported_median), is.finite(reported_lower), is.finite(reported_upper))

fit_role_shapes <- c(
  "Monthly likelihood" = 16,
  "2026 PPC" = 1,
  "State history" = 4,
  "2026 unobserved" = 3
)

fig3_status_palette <- c(
  supported = "#009E73",
  partial = "#E69F00",
  missing = "#BDBDBD",
  caution = "#D55E00",
  selected = "#0072B2",
  excluded = "#7A7A7A"
)

fig3_status_tile_palette <- c(
  supported = "#D8EFE6",
  partial = "#F6E2B3",
  missing = "#ECECEC",
  caution = "#F3C5B7",
  selected = "#CFE4F2",
  excluded = "#D8D8D8"
)

fig3_status_breaks <- c("supported", "partial", "caution", "missing")
fig3_status_labels <- c("Supported", "Partial", "Caution", "Missing")

fig3_interval_palette <- c(
  "Median (95% UI)" = clr("blue")
)

# Monthly observations are plotted at month centres; these bounds shade the
# same full 2020-2022 pandemic reference window used in Figure 1.
pandemic_reference_band_monthly <- tibble(
  xmin = as.Date("2020-01-01"),
  xmax = as.Date("2022-12-31"),
  ymin = -Inf,
  ymax = Inf
)

p_fit <- ggplot(monthly_summary_plot, aes(date)) +
  geom_rect(
    data = pandemic_reference_band_monthly,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = clr("paper"),
    alpha = 0.95
  ) +
  geom_ribbon(
    aes(ymin = pmax(reported_lower, 0), ymax = pmax(reported_upper, 0), fill = "Median (95% UI)"),
    alpha = 0.26,
    colour = NA
  ) +
  geom_line(aes(y = pmax(reported_median, 0), colour = "Median (95% UI)"), linewidth = 0.46) +
  geom_point(
    aes(y = pmax(observed_cases, 0), shape = fit_role),
    colour = clr("black"),
    size = 0.78,
    alpha = 0.46,
    na.rm = TRUE
  ) +
  facet_wrap(~country_panel, ncol = 3, scales = "free_y") +
  scale_y_continuous(
    breaks = monthly_case_breaks,
    labels = label_monthly_cases_axis,
    expand = expansion(mult = c(0.02, 0.08))
  ) +
  scale_x_date(
    breaks = figure_breaks_date(2010, 2025, 5),
    date_labels = "%Y",
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  scale_fill_manual(
    values = fig3_interval_palette,
    breaks = c("Median (95% UI)"),
    name = NULL
  ) +
  scale_colour_manual(
    values = fig3_interval_palette,
    breaks = c("Median (95% UI)"),
    name = NULL
  ) +
  scale_shape_manual(
    values = fit_role_shapes,
    breaks = c("Monthly likelihood", "2026 PPC"),
    labels = c("Fitted", "2026 PPC"),
    drop = FALSE
  ) +
  labs(x = NULL, y = "Monthly reported cases", fill = NULL, colour = NULL, shape = NULL) +
  guides(
    fill = guide_legend(order = 1),
    colour = guide_legend(order = 1),
    shape = guide_legend(order = 2, override.aes = list(alpha = 0.55, size = 0.65))
  ) +
  theme_manuscript() +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.direction = "horizontal",
    legend.key.size = unit(2.1, "mm"),
    legend.spacing.x = unit(1.2, "mm"),
    legend.margin = margin(0, 0, 0, 0),
    plot.margin = margin(2, 4, 2, 4)
  )

calibration_months <- monthly_summary %>%
  filter(
    fit_role == "Monthly likelihood",
    is.finite(observed_cases),
    is.finite(reported_median),
    is.finite(reported_predictive_lower),
    is.finite(reported_predictive_upper)
  )

post_cumulative_draw <- ppc %>%
  filter(fit_role == "fit_likelihood", year >= 2023, year <= 2025) %>%
  group_by(iso3, country, draw) %>%
  summarise(
    observed_cases = safe_sum(observed_cases),
    reported_expected = safe_sum(reported_cases_predicted),
    reported_predictive = safe_sum(reported_cases_predictive),
    true_infections = safe_sum(true_infections_predicted),
    symptomatic_infections = safe_sum(symptomatic_infections_predicted),
    .groups = "drop"
  )

post_cumulative <- post_cumulative_draw %>%
  group_by(iso3, country) %>%
  summarise(
    observed_cases = safe_first(observed_cases),
    reported_median = safe_median(reported_expected),
    reported_predictive_lower = safe_quantile(reported_predictive, 0.025),
    reported_predictive_upper = safe_quantile(reported_predictive, 0.975),
    true_infections_median = safe_median(true_infections),
    symptomatic_infections_median = safe_median(symptomatic_infections),
    .groups = "drop"
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    country_label = factor(country_label, levels = rev(pilot$country_label)),
    ratio_median = reported_median / pmax(observed_cases, 1),
    ratio_lower = reported_predictive_lower / pmax(observed_cases, 1),
    ratio_upper = reported_predictive_upper / pmax(observed_cases, 1),
    cumulative_status = case_when(
      ratio_lower <= 1 & ratio_upper >= 1 & ratio_median >= 0.75 & ratio_median <= 1.33 ~ "supported",
      ratio_lower <= 1 & ratio_upper >= 1 ~ "partial",
      TRUE ~ "caution"
    ),
    cumulative_status = factor(cumulative_status, levels = names(evidence_palette)),
    cumulative_label = paste0(
      label_number(accuracy = 0.01)(ratio_median),
      "x; ",
      label_short(observed_cases),
      " obs"
    )
  )

p_cumulative <- ggplot(post_cumulative, aes(ratio_median, country_label)) +
  geom_vline(xintercept = 1, colour = clr("grey"), linewidth = 0.28, linetype = "22") +
  geom_pointrange(
    aes(xmin = ratio_lower, xmax = ratio_upper, colour = cumulative_status, fill = cumulative_status),
    orientation = "y",
    shape = 21,
    linewidth = 0.52,
    size = 0.39,
    stroke = 0.28,
    alpha = 0.92,
    na.rm = TRUE
  ) +
  geom_text(aes(x = 1.5, label = cumulative_label), hjust = 0, size = figure_text$annotation, colour = clr("dark_grey")) +
  scale_colour_manual(values = fig3_status_palette, guide = "none") +
  scale_fill_manual(values = fig3_status_palette, guide = "none") +
  scale_x_log10(
    breaks = figure_breaks$ratio_fine,
    labels = c("0.5x", "0.75x", "1x", "1.5x", "2x"),
    limits = c(0.48, 2.70),
    expand = expansion(mult = c(0.02, 0.04))
  ) +
  labs(x = "2023-2025 expected / observed\nmedian (95% predictive UI)", y = NULL) +
  theme_manuscript() +
  theme(
    legend.position = "none"
  )

fit_quality <- calibration_months %>%
  group_by(iso3, country, country_label) %>%
  summarise(
    n_months = n(),
    log_rmse = sqrt(mean((log1p(observed_cases) - log1p(reported_median))^2, na.rm = TRUE)),
    median_abs_fold_error = exp(stats::median(abs(log((observed_cases + 0.5) / (reported_median + 0.5))), na.rm = TRUE)),
    expected_interval_inclusion_95 = mean(observed_cases >= reported_lower & observed_cases <= reported_upper, na.rm = TRUE),
    predictive_interval_coverage_95 = mean(observed_cases >= reported_predictive_lower & observed_cases <= reported_predictive_upper, na.rm = TRUE),
    median_expected_interval_relative_width = stats::median((reported_upper - reported_lower) / pmax(reported_median, 1), na.rm = TRUE),
    median_predictive_relative_width = stats::median((reported_predictive_upper - reported_predictive_lower) / pmax(reported_median, 1), na.rm = TRUE),
    .groups = "drop"
  )

peak_observed <- calibration_months %>%
  filter(year >= 2023, year <= 2025) %>%
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
    peak_predicted_median = reported_median,
    peak_predicted_lower_95 = reported_lower,
    peak_predicted_upper_95 = reported_upper,
    peak_predictive_lower_95 = reported_predictive_lower,
    peak_predictive_upper_95 = reported_predictive_upper,
    peak_capture_ratio = reported_median / pmax(observed_cases, 1),
    peak_expected_interval_included = observed_cases >= reported_lower & observed_cases <= reported_upper,
    peak_predictive_interval_covered = observed_cases >= reported_predictive_lower & observed_cases <= reported_predictive_upper
  )

peak_draw <- ppc %>%
  filter(fit_role == "fit_likelihood", year >= 2023, year <= 2025) %>%
  group_by(iso3, country, draw) %>%
  arrange(desc(reported_cases_predicted), year, month, .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  transmute(
    iso3,
    country,
    draw,
    predicted_peak_year = year,
    predicted_peak_month = month,
    predicted_peak_index = (year - 2023) * 12 + month,
    predicted_peak_cases = reported_cases_predicted
  )

peak_timing <- peak_draw %>%
  group_by(iso3, country) %>%
  summarise(
    predicted_peak_index_median = safe_median(predicted_peak_index),
    predicted_peak_index_lower = safe_quantile(predicted_peak_index, 0.025),
    predicted_peak_index_upper = safe_quantile(predicted_peak_index, 0.975),
    predicted_peak_cases_median = safe_median(predicted_peak_cases),
    .groups = "drop"
  ) %>%
  left_join(peak_observed, by = c("iso3", "country")) %>%
  mutate(
    observed_peak_index = (peak_year - 2023) * 12 + peak_month,
    predicted_peak_date = month_index_to_date(predicted_peak_index_median),
    predicted_peak_date_lower = month_index_to_date(predicted_peak_index_lower),
    predicted_peak_date_upper = month_index_to_date(predicted_peak_index_upper),
    observed_peak_date = month_index_to_date(observed_peak_index),
    peak_lag_months = predicted_peak_index_median - observed_peak_index,
    peak_status = case_when(
      peak_predictive_interval_covered & abs(peak_lag_months) <= 1 & peak_capture_ratio >= 0.75 & peak_capture_ratio <= 1.33 ~ "supported",
      peak_predictive_interval_covered & abs(peak_lag_months) <= 2 ~ "partial",
      TRUE ~ "caution"
    ),
    peak_status = factor(peak_status, levels = names(evidence_palette)),
    peak_label = paste0(
      if_else(peak_lag_months > 0, "+", ""),
      label_number(accuracy = 1)(peak_lag_months),
      " mo; ",
      label_number(accuracy = 0.01)(peak_capture_ratio),
      "x"
    )
  )

fit_quality <- fit_quality %>%
  left_join(
    peak_timing %>%
      select(
        iso3, peak_year, peak_month, peak_observed_cases, peak_predicted_median,
        peak_capture_ratio, peak_expected_interval_included, peak_predictive_interval_covered
      ),
    by = "iso3"
  )

p_peak <- ggplot(peak_timing, aes(predicted_peak_date, country_label)) +
  geom_pointrange(
    aes(
      xmin = predicted_peak_date_lower,
      xmax = predicted_peak_date_upper,
      colour = peak_status,
      fill = peak_status,
      shape = "Predicted"
    ),
    orientation = "y",
    linewidth = 0.70,
    size = 0.70,
    stroke = 0.28,
    alpha = 0.92,
    na.rm = TRUE
  ) +
  geom_point(
    aes(x = observed_peak_date, shape = "Observed"),
    size = 2.15,
    colour = clr("black"),
    stroke = 0.55
  ) +
  geom_label(
    aes(x = as.Date("2025-07-15"), label = peak_label),
    hjust = 0,
    size = figure_text$annotation_dense,
    colour = clr("dark_grey"),
    fill = scales::alpha(clr("white"), 0.82),
    linewidth = 0,
    label.padding = unit(0.45, "mm")
  ) +
  scale_colour_manual(values = fig3_status_palette, guide = "none") +
  scale_fill_manual(values = fig3_status_palette, guide = "none") +
  scale_shape_manual(
    values = c(
      "Predicted" = 21,
      "Observed" = 1
    ),
    breaks = c("Predicted", "Observed"),
    name = NULL,
    guide = guide_legend(
      nrow = 2,
      byrow = TRUE,
      override.aes = list(
        colour = clr("black"),
        fill = c(clr("black"), clr("white"))
      )
    )
  ) +
  scale_x_date(
    breaks = figure_breaks_date(2023, 2026, 1, month = 1, day = 15),
    date_labels = "%Y",
    limits = as.Date(c("2023-01-01", "2026-05-01")),
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  labs(x = "Predicted peak month\nmedian (95% UI)", y = NULL) +
  theme_manuscript() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.02, 0.98),
    legend.justification.inside = c(0, 1),
    legend.background = element_rect(
      fill = scales::alpha(clr("white"), 0.88),
      colour = NA
    ),
    legend.key.height = unit(2.0, "mm"),
    legend.key.width = unit(4.8, "mm"),
    legend.text = element_text(size = figure_text$legend_text_dense),
    legend.margin = margin(0.5, 0.8, 0.5, 0.8)
  )

ppc_2026_draw <- ppc %>%
  filter(fit_role == "posterior_predictive_check") %>%
  group_by(iso3, country, draw) %>%
  summarise(
    observed_cases = safe_sum(observed_cases),
    reported_expected = safe_sum(reported_cases_predicted),
    reported_predictive = safe_sum(reported_cases_predictive),
    .groups = "drop"
  )

ppc_2026_plot <- pilot %>%
  select(iso3, country_label) %>%
  mutate(country_label = factor(country_label, levels = rev(pilot$country_label))) %>%
  left_join(
    ppc_2026_draw %>%
      group_by(iso3, country) %>%
      summarise(
        ppc_observed = safe_first(observed_cases),
        ppc_median = safe_median(reported_predictive),
        ppc_lower = safe_quantile(reported_predictive, 0.025),
        ppc_upper = safe_quantile(reported_predictive, 0.975),
        .groups = "drop"
      ),
    by = "iso3"
  ) %>%
  mutate(
    ratio_median = ppc_median / pmax(ppc_observed, 1),
    ratio_lower = ppc_lower / pmax(ppc_observed, 1),
    ratio_upper = ppc_upper / pmax(ppc_observed, 1),
    ppc_status = case_when(
      !is.finite(ppc_observed) ~ "missing",
      ratio_lower <= 1 & ratio_upper >= 1 & ratio_median >= 0.75 & ratio_median <= 1.33 ~ "supported",
      ratio_lower <= 1 & ratio_upper >= 1 ~ "partial",
      TRUE ~ "caution"
    ),
    ppc_status = factor(ppc_status, levels = names(evidence_palette)),
    ppc_label = case_when(
      !is.finite(ppc_observed) ~ "No 2026",
      TRUE ~ paste0(label_number(accuracy = 0.01)(ratio_median), "x")
    ),
    ppc_text_x = if_else(is.finite(ratio_upper), ratio_upper * 1.16, 1.16)
  )

p_ppc <- ggplot(ppc_2026_plot, aes(ratio_median, country_label)) +
  geom_vline(xintercept = 1, colour = clr("grey"), linewidth = 0.28, linetype = "22") +
  geom_pointrange(
    aes(xmin = ratio_lower, xmax = ratio_upper, colour = ppc_status, fill = ppc_status),
    orientation = "y",
    shape = 21,
    linewidth = 0.52,
    size = 0.39,
    stroke = 0.28,
    alpha = 0.92,
    na.rm = TRUE
  ) +
  geom_point(
    data = filter(ppc_2026_plot, !is.finite(ratio_median)),
    aes(x = 1),
    shape = 4,
    size = 1.45,
    colour = clr("grey"),
    stroke = 0.45,
    inherit.aes = TRUE
  ) +
  geom_text(aes(x = ppc_text_x, label = ppc_label), hjust = 0, size = figure_text$annotation, colour = clr("dark_grey")) +
  scale_colour_manual(values = fig3_status_palette, guide = "none") +
  scale_fill_manual(values = fig3_status_palette, guide = "none") +
  scale_x_log10(
    breaks = figure_breaks$ratio_2x,
    labels = c("0.5x", "1x", "2x", "4x", "8x"),
    limits = c(0.45, 13),
    expand = expansion(mult = c(0.02, 0.06))
  ) +
  labs(x = "2026 predictive / observed\nmedian (95% UI)", y = NULL) +
  theme_manuscript() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "none"
  )

metric_long <- fit_quality %>%
  select(iso3, country_label, log_rmse, median_abs_fold_error, predictive_interval_coverage_95, peak_capture_ratio) %>%
  left_join(post_cumulative %>% select(iso3, cumulative_ratio = ratio_median), by = "iso3") %>%
  transmute(
    country_label,
    `Log RMSE` = log_rmse,
    `Median fold` = median_abs_fold_error,
    `PPC coverage` = predictive_interval_coverage_95,
    `2023-2025 ratio` = cumulative_ratio,
    `Peak capture` = peak_capture_ratio
  ) %>%
  pivot_longer(-country_label, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(
      metric,
      levels = c("Log RMSE", "Median fold", "PPC coverage", "2023-2025 ratio", "Peak capture")
    ),
    metric_status = case_when(
      metric == "Log RMSE" & value <= 0.70 ~ "supported",
      metric == "Log RMSE" & value <= 1.00 ~ "partial",
      metric == "Median fold" & value <= 2.00 ~ "supported",
      metric == "Median fold" & value <= 3.00 ~ "partial",
      metric == "PPC coverage" & value >= 0.85 ~ "supported",
      metric == "PPC coverage" & value >= 0.75 ~ "partial",
      metric == "2023-2025 ratio" & value >= 0.75 & value <= 1.33 ~ "supported",
      metric == "2023-2025 ratio" & value >= 0.50 & value <= 2.00 ~ "partial",
      metric == "Peak capture" & value >= 0.75 & value <= 1.33 ~ "supported",
      metric == "Peak capture" & value >= 0.50 & value <= 2.00 ~ "partial",
      TRUE ~ "caution"
    ),
    metric_status = factor(metric_status, levels = names(evidence_palette)),
    label = case_when(
      metric == "PPC coverage" ~ paste0(label_number(accuracy = 1)(100 * value), "%"),
      metric %in% c("Median fold", "2023-2025 ratio", "Peak capture") ~ paste0(label_number(accuracy = 0.01)(value), "x"),
      TRUE ~ label_number(accuracy = 0.01)(value)
    )
  )

status_legend_keys <- tibble(
  metric = factor(rep(levels(metric_long$metric)[[1]], length(fig3_status_breaks)), levels = levels(metric_long$metric)),
  country_label = factor(rep(levels(metric_long$country_label)[[1]], length(fig3_status_breaks)), levels = levels(metric_long$country_label)),
  metric_status = factor(fig3_status_breaks, levels = names(evidence_palette))
)

p_metrics <- ggplot(metric_long, aes(metric, country_label)) +
  geom_tile(aes(fill = metric_status), colour = clr("white"), linewidth = 0.32, width = 0.94, height = 0.78, show.legend = FALSE) +
  geom_text(aes(label = label), size = figure_text$annotation_dense, colour = clr("black")) +
  geom_point(
    data = status_legend_keys,
    aes(metric, country_label, fill = metric_status),
    inherit.aes = FALSE,
    shape = 22,
    size = 0.01,
    alpha = 0,
    show.legend = TRUE
  ) +
  scale_fill_manual(
    values = fig3_status_tile_palette,
    breaks = fig3_status_breaks,
    labels = fig3_status_labels,
    limits = fig3_status_breaks,
    name = "Status",
    drop = FALSE,
    guide = guide_legend(
      nrow = 1,
      override.aes = list(
        shape = 22,
        size = 2.2,
        alpha = 1,
        fill = unname(fig3_status_palette[fig3_status_breaks]),
        colour = NA
      )
    )
  ) +
  labs(x = NULL, y = NULL) +
  theme_manuscript() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text_dense),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = figure_text$legend_title, face = "bold"),
    legend.text = element_text(size = figure_text$legend_text),
    legend.key.size = unit(2.4, "mm"),
    legend.spacing.x = unit(1.0, "mm"),
    legend.margin = margin(0, 0, 0, 0)
  )

latent_multiplier_draw <- post_cumulative_draw %>%
  transmute(
    iso3,
    country,
    draw,
    `Total infections` = true_infections / pmax(reported_expected, 1),
    `Symptomatic infections` = symptomatic_infections / pmax(reported_expected, 1)
  ) %>%
  pivot_longer(c(`Total infections`, `Symptomatic infections`), names_to = "infection_target", values_to = "multiplier")

latent_multiplier <- latent_multiplier_draw %>%
  group_by(iso3, country, infection_target) %>%
  summarise(
    multiplier_median = safe_median(multiplier),
    multiplier_lower = safe_quantile(multiplier, 0.025),
    multiplier_upper = safe_quantile(multiplier, 0.975),
    .groups = "drop"
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    country_label = factor(country_label, levels = rev(pilot$country_label)),
    infection_target = factor(infection_target, levels = c("Symptomatic infections", "Total infections")),
    label = paste0(label_number(accuracy = 1)(multiplier_median), "x")
  )

latent_palette <- c(
  "Symptomatic infections" = "#CC79A7",
  "Total infections" = "#0072B2"
)

p_latent <- ggplot(
  latent_multiplier,
  aes(multiplier_median, country_label, colour = infection_target, fill = infection_target)
) +
  geom_pointrange(
    aes(xmin = multiplier_lower, xmax = multiplier_upper),
    orientation = "y",
    position = position_dodge(width = 0.34),
    linewidth = 0.52,
    size = 0.39,
    shape = 21,
    stroke = 0.28,
    alpha = 0.92
  ) +
  scale_colour_manual(
    values = latent_palette,
    breaks = c("Symptomatic infections", "Total infections"),
    labels = c("Symptomatic", "Total"),
    name = "Infections",
    guide = guide_legend(
      title.position = "left",
      title.vjust = 0.5,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(linewidth = 0.50, alpha = 1, size = 0.12)
    )
  ) +
  scale_fill_manual(values = latent_palette, guide = "none") +
  scale_x_log10(
    breaks = c(5, 10, 30, 100, 300, 1000),
    labels = c("5x", "10x", "30x", "100x", "300x", "1000x"),
    expand = expansion(mult = c(0.02, 0.18))
  ) +
  labs(x = "2023-2025 model-implied infections /\nexpected reports, median (95% UI)", y = NULL) +
  theme_manuscript() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom",
    legend.key.height = unit(3.4, "mm"),
    legend.key.width = unit(2.2, "mm"),
    legend.margin = margin(0, 0, 0, 0)
  )

monthly_residual <- calibration_months %>%
  mutate(
    log2_fold_error = log2((observed_cases + 0.5) / (reported_median + 0.5)),
    residual_status = case_when(
      abs(log2_fold_error) <= 1 ~ "supported",
      abs(log2_fold_error) <= 2 ~ "partial",
      TRUE ~ "caution"
    ),
    residual_status = factor(residual_status, levels = names(evidence_palette))
)

diagnostic_long <- fit_summary %>%
  select(
    iso3,
    n_month_observations,
    n_month_ppc_observations,
    n_age_observations,
    max_rhat,
    min_ess_bulk
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(country_label = factor(country_label, levels = rev(pilot$country_label))) %>%
  transmute(
    country_label,
    `Monthly obs.` = as.numeric(n_month_observations),
    `Age rows` = as.numeric(n_age_observations),
    `2026 PPC mo.` = as.numeric(n_month_ppc_observations),
    `Max R-hat` = as.numeric(max_rhat),
    `Min bulk ESS` = as.numeric(min_ess_bulk)
  ) %>%
  pivot_longer(-country_label, names_to = "metric", values_to = "value")

figure_layout <- "
AAA
AAA
BCD
EEF
"

figure <- panel_tags(
  free(p_fit, side = "l") + p_cumulative + p_peak + p_ppc + free(p_metrics) + p_latent +
    plot_layout(
      design = figure_layout,
      heights = c(0.50, 0.50, 0.48, 0.50),
      guides = "keep"
    )
) &
  theme(
    plot.tag = element_text(
      face = "bold",
      size = figure_text$tag,
      margin = margin(0, 2, 1, 0)
    )
  )

readr::write_csv(annual_summary, fd_path("figure_3_model2c_posterior_predictive.csv"))
readr::write_csv(monthly_summary, fd_path("figure_3_model2c_monthly_posterior_predictive.csv"))
readr::write_csv(post_cumulative, fd_path("figure_3_model2c_post_cumulative_calibration.csv"))
readr::write_csv(peak_timing, fd_path("figure_3_model2c_peak_timing_magnitude.csv"))
readr::write_csv(ppc_2026_plot, fd_path("figure_3_model2c_2026_ppc_check.csv"))
readr::write_csv(fit_quality, fd_path("figure_3_model2c_fit_quality.csv"))
readr::write_csv(metric_long, fd_path("figure_3_model2c_validation_metrics.csv"))
readr::write_csv(latent_multiplier, fd_path("figure_3_model2c_latent_reported_multipliers.csv"))
readr::write_csv(diagnostic_long, fd_path("figure_3_model2c_sampling_data_diagnostics.csv"))
readr::write_csv(monthly_residual, fd_path("figure_3_model2c_monthly_residuals.csv"))
save_main_figure(
  figure,
  "figure_3_model2c_posterior_predictive",
  nature_full_width_mm,
  nature_max_height_mm
)
