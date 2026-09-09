suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
})

figure_data_dir <- file.path("manuscript", "figure_data")
figure_dir <- file.path("manuscript", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

scenario_order <- c(
  "low_vaccine_waning",
  "no_missed_exposure_boosting",
  "no_routine_immunization_disruption",
  "restored_booster_reach",
  "fixed_birth_cohort_demography"
)

scenario_labels <- c(
  low_vaccine_waning = "No further\nwaning",
  no_missed_exposure_boosting = "Exposure boost\nrestored",
  no_routine_immunization_disruption = "DTP flow\nrestored",
  restored_booster_reach = "Booster reach\nrestored",
  fixed_birth_cohort_demography = "Fixed cohort\ndemography"
)

country_order <- c("USA", "CHN", "AUS", "GBR", "NZL", "JPN", "THA", "SWE", "BRA")

scenario_colours <- c(
  low_vaccine_waning = "#2C73B9",
  no_missed_exposure_boosting = "#0AA6A6",
  no_routine_immunization_disruption = "#E67E22",
  restored_booster_reach = "#7F8C8D",
  fixed_birth_cohort_demography = "#B65F32"
)

contrasts <- read_csv(
  file.path(figure_data_dir, "figure_4_model2c_scenario_contrasts.csv"),
  show_col_types = FALSE
) |>
  filter(
    scenario_id %in% scenario_order,
    outcome_target == "reported_cases",
    period_start == 2023,
    period_end == 2025
  ) |>
  mutate(
    reduction = 100 * (1 - scenario_to_full_ratio_median),
    reduction_lower = 100 * (1 - scenario_to_full_ratio_upper_95),
    reduction_upper = 100 * (1 - scenario_to_full_ratio_lower_95),
    country = factor(iso3, levels = rev(country_order)),
    scenario = factor(
      scenario_id,
      levels = scenario_order,
      labels = unname(scenario_labels[scenario_order])
    ),
    label = case_when(
      abs(reduction) < 0.05 ~ "0",
      abs(reduction) < 10 ~ paste0(
        if_else(reduction > 0, "+", ""),
        number(reduction, accuracy = 0.1), "%"
      ),
      TRUE ~ paste0(
        if_else(reduction > 0, "+", ""),
        number(reduction, accuracy = 1), "%"
      )
    )
  )

write_csv(
  contrasts |>
    transmute(
      iso3, scenario_id, outcome_target, period_start, period_end,
      reduction_median = reduction,
      reduction_lower_95 = reduction_lower,
      reduction_upper_95 = reduction_upper
    ),
  file.path(figure_data_dir, "lancet_v2_figure_3_conditional_mechanisms.csv")
)

figure <- ggplot(contrasts, aes(reduction, country, colour = scenario_id)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "#777777", linewidth = 0.4) +
  geom_errorbar(
    aes(xmin = reduction_lower, xmax = reduction_upper),
    width = 0, linewidth = 0.8, orientation = "y"
  ) +
  geom_point(size = 2.6) +
  geom_text(
    aes(label = label),
    nudge_x = 3.2, hjust = 0, colour = "#202124", size = 2.5
  ) +
  facet_grid(. ~ scenario) +
  scale_colour_manual(values = scenario_colours, guide = "none") +
  scale_x_continuous(
    limits = c(-15, 70),
    breaks = c(0, 20, 40, 60),
    labels = label_percent(scale = 1),
    expand = expansion(mult = c(0.08, 0.28))
  ) +
  labs(
    x = "Conditional reduction in modelled reports, 2023–25",
    y = NULL
  ) +
  theme_classic(base_size = 9, base_family = "sans") +
  theme(
    axis.title = element_text(colour = "#202124"),
    axis.text = element_text(colour = "#202124"),
    panel.grid.major.x = element_line(colour = "#E8EAED", linewidth = 0.25),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9, lineheight = 0.95),
    panel.spacing.x = unit(5, "mm"),
    plot.margin = margin(6, 8, 5, 6)
  )

ggsave(
  file.path(figure_dir, "lancet_v2_figure_3_conditional_mechanisms.png"),
  figure, width = 183, height = 103, units = "mm", dpi = 600, bg = "white"
)

ggsave(
  file.path(figure_dir, "lancet_v2_figure_3_conditional_mechanisms.pdf"),
  figure, width = 183, height = 103, units = "mm", device = cairo_pdf,
  bg = "white"
)
