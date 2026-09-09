#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

draws_path <- out_path("model2c_posterior_draws.csv")
if (file.exists(draws_path)) {
  draws <- read_csv_checked(
    draws_path,
    c(
      "iso3", "country", "draw", "contact_sensitivity", "waning_rate", "log_post_transmission_multiplier",
      "log_post_reporting_multiplier", "rebound_peak_month", "rebound_width",
      "rebound_asymmetry", "rebound_release_gain", "secondary_rebound_delay",
      "secondary_rebound_width", "secondary_rebound_release_gain", "outbreak_seeding_gain",
      "seasonal_strength", "natural_boosting_efficiency",
      "pathogen_pressure_scale", "epidemic_feedback_gain",
      "endemic_reseed_pressure", "epidemic_pressure_retention", "post_reporting_peak_gain",
      "post_reporting_peak_multiplier", "post_reporting_trend_per_year",
      "missed_natural_boosting_2020_2022", "natural_boosting_flow_2020_2022",
      "routine_vaccination_shortfall_2020_2022", "natural_waned_stock_2023_start",
      "nb_phi_month", "nb_phi_age", "model2c_output_version"
    )
  ) %>%
    require_model2c_current_version("Supplement parameter posterior draws")

  parameter_draws <- draws %>%
    transmute(
      iso3,
      country,
      draw,
      `Contact sensitivity` = as.numeric(contact_sensitivity),
      `Waning rate` = as.numeric(waning_rate),
      `Post-2023 transmission shift multiplier` = exp(as.numeric(log_post_transmission_multiplier)),
      `Reporting multiplier` = exp(as.numeric(log_post_reporting_multiplier)),
      `Reporting peak multiplier` = as.numeric(post_reporting_peak_multiplier),
      `Reporting peak gain` = as.numeric(post_reporting_peak_gain),
      `Post-reporting trend per year` = as.numeric(post_reporting_trend_per_year),
      `Rebound peak month since Jan 2023` = as.numeric(rebound_peak_month),
      `Rebound width months` = as.numeric(rebound_width),
      `Rebound asymmetry` = as.numeric(rebound_asymmetry),
      `Rebound release gain` = as.numeric(rebound_release_gain),
      `Secondary rebound delay months` = as.numeric(secondary_rebound_delay),
      `Secondary rebound width months` = as.numeric(secondary_rebound_width),
      `Secondary rebound release gain` = as.numeric(secondary_rebound_release_gain),
      `Outbreak ignition pulse gain` = as.numeric(outbreak_seeding_gain),
      `Seasonal strength` = as.numeric(seasonal_strength),
      `Natural boosting efficiency` = as.numeric(natural_boosting_efficiency),
      `Excess epidemic pressure scale` = as.numeric(pathogen_pressure_scale),
      `Epidemic feedback gain` = as.numeric(epidemic_feedback_gain),
      `Endemic background force` = as.numeric(endemic_reseed_pressure),
      `Epidemic pressure retention` = as.numeric(epidemic_pressure_retention),
      `Natural-waned exposure-boosting interruption diagnostic 2020-2022` = as.numeric(missed_natural_boosting_2020_2022),
      `Natural-waned boosting flow diagnostic 2020-2022` = as.numeric(natural_boosting_flow_2020_2022),
      `Routine vaccination shortfall 2020-2022` = as.numeric(routine_vaccination_shortfall_2020_2022),
      `Natural-waned stock Jan 2023` = as.numeric(natural_waned_stock_2023_start),
      `Monthly NB dispersion` = as.numeric(nb_phi_month),
      `Age NB dispersion` = as.numeric(nb_phi_age)
    ) %>%
    pivot_longer(
      cols = -c(iso3, country, draw),
      names_to = "parameter",
      values_to = "value"
    )

  summary <- parameter_draws %>%
    group_by(iso3, country, parameter) %>%
    summarise(
      median = median(value, na.rm = TRUE),
      lower_95 = quantile(value, 0.025, na.rm = TRUE),
      upper_95 = quantile(value, 0.975, na.rm = TRUE),
      .groups = "drop"
    )
} else {
  summary <- read_csv_checked(
    fd_path("supplement_model2c_parameter_posteriors.csv"),
    c("iso3", "country", "parameter", "median", "lower_95", "upper_95")
  )
}

summary <- summary %>%
  mutate(
    median = as.numeric(median),
    lower_95 = as.numeric(lower_95),
    upper_95 = as.numeric(upper_95)
  ) %>%
  select(-any_of(c("pilot_rank", "country_label"))) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_label = format_country_inline(iso3, country),
    country_label = fct_reorder(country_label, pilot_rank, .desc = TRUE),
    parameter = factor(
      parameter,
      levels = c(
        "Contact sensitivity",
        "Waning rate",
        "Post-2023 transmission shift multiplier",
        "Reporting multiplier",
        "Rebound peak month since Jan 2023",
        "Rebound width months",
        "Rebound release gain",
        "Outbreak ignition pulse gain",
        "Seasonal strength",
        "Natural boosting efficiency",
        "Excess epidemic pressure scale",
        "Epidemic feedback gain",
        "Endemic background force",
        "Epidemic pressure retention",
        "Natural-waned exposure-boosting interruption diagnostic 2020-2022",
        "Natural-waned boosting flow diagnostic 2020-2022",
        "Routine vaccination shortfall 2020-2022",
        "Natural-waned stock Jan 2023",
        "Monthly NB dispersion",
        "Age NB dispersion"
      )
    ),
    parameter_short = fct_recode(
      parameter,
      "Transmission\nmultiplier" = "Post-2023 transmission shift multiplier",
      "Reporting\nmultiplier" = "Reporting multiplier",
      "Rebound peak\nmonth" = "Rebound peak month since Jan 2023",
      "Rebound width\nmonths" = "Rebound width months",
      "Outbreak ignition\npulse gain" = "Outbreak ignition pulse gain",
      "Natural boosting\nefficiency" = "Natural boosting efficiency",
      "Excess epidemic\npressure" = "Excess epidemic pressure scale",
      "Endemic\nbackground force" = "Endemic background force",
      "Epidemic pressure\nretention" = "Epidemic pressure retention",
      "Boosting interruption\n2020-2022" = "Natural-waned exposure-boosting interruption diagnostic 2020-2022",
      "Boosting flow\n2020-2022" = "Natural-waned boosting flow diagnostic 2020-2022",
      "Routine shortfall\n2020-2022" = "Routine vaccination shortfall 2020-2022",
      "Natural-waned stock\nJan 2023" = "Natural-waned stock Jan 2023",
      "Monthly NB\ndispersion" = "Monthly NB dispersion",
      "Age NB\ndispersion" = "Age NB dispersion"
    )
  ) %>%
  # Retain only the prespecified display set; undeclared factor levels would
  # otherwise collapse into a spurious facet labelled "NA".
  filter(!is.na(parameter))

p <- ggplot(summary, aes(median, country_label)) +
  geom_segment(aes(x = lower_95, xend = upper_95, yend = country_label), linewidth = 0.75, alpha = 0.34, colour = clr("blue")) +
  geom_point(size = 1.05, colour = clr("blue")) +
  geom_vline(
    data = tibble(
      parameter_short = fct_recode(
        factor(
          c("Post-2023 transmission shift multiplier", "Reporting multiplier"),
          levels = levels(summary$parameter)
        ),
        "Transmission\nmultiplier" = "Post-2023 transmission shift multiplier",
        "Reporting\nmultiplier" = "Reporting multiplier"
      )
    ),
    aes(xintercept = 1),
    inherit.aes = FALSE,
    linetype = "22",
    linewidth = 0.25,
    colour = pal["grey"]
  ) +
  facet_wrap(~ parameter_short, scales = "free_x", ncol = 4) +
  labs(
    x = "Median and 95% UI; fixed quantities have zero-width UIs",
    y = NULL
  ) +
  theme_nature() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.x = element_text(size = figure_text$axis_text_dense),
    axis.text.y = element_text(size = figure_text$axis_text_dense),
    strip.text = element_text(size = figure_text$strip_dense, lineheight = 0.88),
    panel.spacing = unit(2.0, "mm")
  )

readr::write_csv(summary, fd_path("supplement_model2c_parameter_posteriors.csv"))
save_nature(p, "supplement_model2c_parameter_posteriors.png", nature_full_width_mm, nature_max_height_mm)
