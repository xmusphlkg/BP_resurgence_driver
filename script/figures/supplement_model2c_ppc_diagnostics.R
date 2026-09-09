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
    require_model2c_current_version("Supplement posterior predictive diagnostics") %>%
    mutate(
      draw = as.integer(draw),
      year = as.integer(year),
      observed_cases = as.numeric(observed_cases),
      reported_cases_predicted = as.numeric(reported_cases_predicted),
      reported_cases_predictive = as.numeric(reported_cases_predictive),
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
      )
    )

  annual_draws <- ppc %>%
    group_by(iso3, country, draw, year, fit_role) %>%
    summarise(
      observed_cases = if_else(any(is.finite(observed_cases)), sum(observed_cases, na.rm = TRUE), NA_real_),
      predicted_cases = if_else(
        any(is.finite(reported_cases_predicted_comparison)),
        sum(reported_cases_predicted_comparison, na.rm = TRUE),
        NA_real_
      ),
      predictive_cases = if_else(
        any(is.finite(reported_cases_predictive_comparison)),
        sum(reported_cases_predictive_comparison, na.rm = TRUE),
        NA_real_
      ),
      .groups = "drop"
    )

  annual_summary <- annual_draws %>%
    group_by(iso3, country, year, fit_role) %>%
    summarise(
      observed_cases = first(na.omit(observed_cases), default = NA_real_),
      predicted_median = median(predicted_cases, na.rm = TRUE),
      predicted_lower_95 = quantile(predictive_cases, 0.025, na.rm = TRUE),
      predicted_upper_95 = quantile(predictive_cases, 0.975, na.rm = TRUE),
      .groups = "drop"
    )
} else {
  annual_summary <- read_csv_checked(
    fd_path("supplement_model2c_ppc_diagnostics.csv"),
    c(
      "iso3", "country", "year", "fit_role", "observed_cases",
      "predicted_median", "predicted_lower_95", "predicted_upper_95"
    )
  )
}

annual_summary <- annual_summary %>%
  mutate(
    year = as.integer(year),
    observed_cases = as.numeric(observed_cases),
    predicted_median = as.numeric(predicted_median),
    predicted_lower_95 = as.numeric(predicted_lower_95),
    predicted_upper_95 = as.numeric(predicted_upper_95),
    fit_role_label = case_when(
      fit_role == "fit_likelihood" ~ "Monthly likelihood",
      fit_role == "state_history_no_monthly_observation" ~ "State history",
      fit_role == "posterior_predictive_check" ~ "2026 PPC",
      fit_role == "posterior_predictive_unobserved_month" ~ "2026 unobserved",
      fit_role == "2026 PPC" ~ "2026 PPC",
      TRUE ~ "Fitted period"
    )
  ) %>%
  select(-any_of(c("pilot_rank", "country_label"))) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_label = format_country_inline(iso3, country),
    country_label = fct_reorder(country_label, pilot_rank, .desc = FALSE)
  )

annual_summary_plot <- annual_summary %>%
  filter(is.finite(predicted_median), is.finite(predicted_lower_95), is.finite(predicted_upper_95))

p <- ggplot(annual_summary_plot, aes(year)) +
  geom_ribbon(
    aes(ymin = predicted_lower_95, ymax = predicted_upper_95, fill = fit_role_label),
    alpha = 0.22,
    colour = NA
  ) +
  geom_line(aes(y = predicted_median, colour = fit_role_label), linewidth = 0.35) +
  geom_point(aes(y = observed_cases), shape = 21, size = 1.45, stroke = 0.35, fill = pal["white"], colour = pal["black"], na.rm = TRUE) +
  facet_wrap(~ country_label, scales = "free_y", ncol = 3) +
  scale_colour_manual(values = c("Monthly likelihood" = clr("blue"), "Fitted period" = clr("blue"), "2026 PPC" = clr("red"))) +
  scale_fill_manual(values = c("Monthly likelihood" = clr("blue"), "Fitted period" = clr("blue"), "2026 PPC" = clr("red"))) +
  scale_y_continuous(labels = label_short, expand = expansion(mult = c(0.02, 0.08))) +
  scale_x_continuous(breaks = c(2015, 2019, 2023, 2026)) +
  labs(
    x = NULL,
    y = "Annual reported cases",
    colour = NULL,
    fill = NULL
  ) +
  theme_nature() +
  theme(
    legend.position = "bottom",
    panel.grid.major.x = element_line(linewidth = 0.14, colour = pal["light_grey"])
  )

readr::write_csv(annual_summary, fd_path("supplement_model2c_ppc_diagnostics.csv"))
save_nature(p, "supplement_model2c_ppc_diagnostics.png", nature_full_width_mm, 168)
