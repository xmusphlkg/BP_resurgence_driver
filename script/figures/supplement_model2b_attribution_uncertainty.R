#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

factor_summary <- read_csv_first_available(
  c(
    out_path("model2b_bayes_factor_weight_summary.csv"),
    file.path(root, "manuscript", "tables", "model2b_bayes_factor_weight_summary.csv")
  ),
  c(
    "iso3", "country", "factor_id", "factor_label", "outcome_target",
    "share_percent_median", "share_percent_lower_95", "share_percent_upper_95",
    "dominant_probability", "model2b_output_version"
  )
) %>%
  require_model2b_current_version("Model 2B attribution uncertainty input")

factor_levels <- c(
  "low_circulation_missed_boosting",
  "routine_immunization_disruption",
  "vaccine_waning_booster_gap",
  "birth_cohort_demography"
)

factor_labels <- c(
  low_circulation_missed_boosting = "Low circulation/\nmissed boosting",
  routine_immunization_disruption = "Routine immunization\ninterruption",
  vaccine_waning_booster_gap = "Waning/\nbooster gap",
  birth_cohort_demography = "Birth-cohort\ndemography"
)

plot_data <- factor_summary %>%
  filter(
    outcome_target == "true_infections_2023_2025",
    factor_id %in% factor_levels,
    attribution_layer == "primary_susceptibility"
  ) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_label = format_country_inline(iso3, country),
    share_percent_median = as.numeric(share_percent_median),
    share_percent_lower_95 = as.numeric(share_percent_lower_95),
    share_percent_upper_95 = as.numeric(share_percent_upper_95),
    dominant_probability = as.numeric(dominant_probability),
    factor = factor(unname(factor_labels[factor_id]), levels = unname(factor_labels)),
    interval_label = case_when(
      !is.finite(share_percent_median) ~ "",
      TRUE ~ paste0(
        round(share_percent_median), "%\n(",
        round(share_percent_lower_95), "-",
        round(share_percent_upper_95), ")"
      )
    ),
    country_label = factor(country_label, levels = rev(unique(country_label[order(pilot_rank, country_label)])))
  )

p <- ggplot(plot_data, aes(factor, country_label, fill = share_percent_median)) +
  geom_tile(colour = pal["white"], linewidth = 0.45) +
  geom_text(aes(label = interval_label), size = 1.75, lineheight = 0.82, colour = pal["black"]) +
  scale_fill_gradient2(
    low = clr("blue"),
    mid = pal["paper"],
    high = clr("red"),
    midpoint = 0,
    breaks = c(-40, -20, 0, 20, 40, 60),
    labels = function(x) paste0(x, "%"),
    na.value = pal["light_grey"]
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Median\nshare",
    caption = "Cell text shows the median share and 95% uncertainty interval (UI)."
  ) +
  theme_nature(base_size = 7.8) +
  theme(
    axis.text.x = element_text(size = 6.0),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )

readr::write_csv(plot_data, fd_path("supplement_model2b_attribution_uncertainty.csv"))
save_nature(p, "supplement_model2b_attribution_uncertainty.png", nature_full_width_mm, 140)
