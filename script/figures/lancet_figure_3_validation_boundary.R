suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
})

output_dir <- "outputs"
figure_data_dir <- file.path("manuscript", "figure_data")
figure_dir <- file.path("manuscript", "figures")
dir.create(figure_data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

primary <- read_csv(file.path(output_dir, "model2c_fit_summary.csv"), show_col_types = FALSE) |>
  filter(iso3 %in% c("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")) |>
  mutate(pass = max_rhat < 1.01 & min_ess_bulk >= 400)

sensitivity <- read_csv(
  file.path(output_dir, "model2c_sensitivity_comparison_status.csv"),
  show_col_types = FALSE
) |>
  mutate(pass = diagnostic_status == "pass")

recovery <- read_csv(
  file.path(
    output_dir, "model2c_recovery", "posterior_truth_nine_country_r5",
    "sampling_diagnostics_by_fit.csv"
  ),
  show_col_types = FALSE
) |>
  mutate(pass = diagnostic_gate_pass)

structural <- read_csv(
  file.path(output_dir, "model2c_structural_refit_sampling_diagnostics.csv"),
  show_col_types = FALSE
) |>
  mutate(pass = passes_core_sampling_gate)

temporal <- read_csv(
  file.path(
    output_dir, "model2c_temporal_validation", "temporal_validation_summary.csv"
  ),
  show_col_types = FALSE
) |>
  filter(scope == "pooled_all_horizons")

gate_summary <- bind_rows(
  tibble(
    analysis = "Primary country fits",
    passed = sum(primary$pass, na.rm = TRUE),
    total = nrow(primary),
    unit = "countries"
  ),
  tibble(
    analysis = "Sensitivity specifications",
    passed = sum(sensitivity$pass, na.rm = TRUE),
    total = nrow(sensitivity),
    unit = "specifications"
  ),
  tibble(
    analysis = "Synthetic-data recovery",
    passed = sum(recovery$pass, na.rm = TRUE),
    total = nrow(recovery),
    unit = "fits"
  ),
  structural |>
    group_by(tag) |>
    summarise(global_pass = all(pass), .groups = "drop") |>
    summarise(
      analysis = "Alternative structures",
      passed = sum(global_pass),
      total = n(),
      unit = "structures"
    )
) |>
  mutate(
    proportion = passed / total,
    label = paste0(passed, "/", total, " ", unit),
    analysis = factor(
      analysis,
      levels = rev(c(
        "Primary country fits", "Sensitivity specifications",
        "Synthetic-data recovery", "Alternative structures"
      ))
    )
  )

coverage <- tibble(
  nominal = c(0.50, 0.80, 0.95),
  observed = c(temporal$coverage_50, temporal$coverage_80, temporal$coverage_95)
)

structure_labels <- c(
  structural_full = "Full structure",
  structural_no_exposure_interruption = "No exposure interruption",
  structural_no_post2023_waning = "No post-2023 waning",
  structural_no_reporting_shift = "No reporting shift",
  structural_flexible_seeding = "Flexible seeding"
)

structure_summary <- structural |>
  group_by(tag) |>
  summarise(passed = sum(pass, na.rm = TRUE), total = n(), .groups = "drop") |>
  mutate(
    label = unname(structure_labels[tag]),
    label = factor(label, levels = rev(unname(structure_labels)))
  )

write_csv(
  gate_summary |>
    transmute(
      panel = "convergence_summary", analysis = as.character(analysis),
      passed, total, value = proportion, label
    ),
  file.path(figure_data_dir, "lancet_figure_3_validation_boundary.csv")
)

theme_lancet <- theme_classic(base_size = 9, base_family = "sans") +
  theme(
    axis.title = element_text(colour = "#202124"),
    axis.text = element_text(colour = "#202124"),
    plot.title = element_blank(),
    plot.tag = element_text(face = "bold", size = 12),
    panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = 0.25),
    legend.position = "none"
  )

p_a <- ggplot(gate_summary, aes(proportion, analysis)) +
  geom_segment(aes(x = 0, xend = proportion, yend = analysis),
               linewidth = 1.4, colour = "#9CC3E8") +
  geom_point(size = 3.2, colour = "#226DB4") +
  geom_text(aes(label = label), nudge_x = 0.035, hjust = 0, size = 2.7) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 0.95),
    breaks = c(0, 0.25, 0.5, 0.75)
  ) +
  labs(
    x = "Proportion meeting criteria",
    y = NULL
  ) +
  theme_lancet

p_b <- ggplot(coverage, aes(nominal, observed)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "#6F7275") +
  geom_segment(
    aes(x = nominal, xend = nominal, y = nominal, yend = observed),
    colour = "#D55E00", linewidth = 1
  ) +
  geom_point(size = 3.2, colour = "#D55E00") +
  scale_x_continuous(labels = percent, limits = c(0.25, 1), breaks = c(0.5, 0.8, 0.95)) +
  scale_y_continuous(labels = percent, limits = c(0.25, 1), breaks = c(0.5, 0.8, 0.95)) +
  coord_equal() +
  labs(
    x = "Nominal interval",
    y = "Empirical coverage"
  ) +
  theme_lancet

p_c <- ggplot(structure_summary, aes(passed, label)) +
  geom_segment(aes(x = 0, xend = passed, yend = label),
               linewidth = 1.4, colour = "#B8D8C7") +
  geom_point(size = 3.2, colour = "#008F5D") +
  geom_text(aes(label = paste0(passed, "/", total, " countries")),
            nudge_x = 0.15, hjust = 0, size = 2.7) +
  scale_x_continuous(breaks = 0:9, limits = c(0, 9.8)) +
  labs(
    x = "Countries meeting convergence criteria",
    y = NULL
  ) +
  theme_lancet

figure <- (p_a + p_b + plot_layout(widths = c(1.35, 1))) /
  p_c +
  plot_annotation(tag_levels = "a") &
  theme(plot.margin = margin(5, 6, 5, 5))

ggsave(
  file.path(figure_dir, "lancet_figure_3_validation_boundary.png"),
  figure, width = 183, height = 150, units = "mm", dpi = 600, bg = "white"
)
ggsave(
  file.path(figure_dir, "lancet_figure_3_validation_boundary.pdf"),
  figure, width = 183, height = 150, units = "mm", device = cairo_pdf,
  bg = "white"
)
