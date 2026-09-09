suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
})

figure_data_dir <- file.path("manuscript", "figure_data")
figure_dir <- file.path("manuscript", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

dat <- read_csv(
  file.path(figure_data_dir, "figure_3_model2c_posterior_predictive.csv"),
  show_col_types = FALSE
)

country_order <- c("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")
country_labels <- c(
  AUS = "Australia", BRA = "Brazil", CHN = "China", GBR = "UK",
  JPN = "Japan", NZL = "New Zealand", SWE = "Sweden",
  THA = "Thailand", USA = "USA"
)

theme_lancet <- theme_classic(base_size = 8.5, base_family = "sans") +
  theme(
    axis.title = element_text(colour = "#202124"),
    axis.text = element_text(colour = "#202124"),
    strip.text = element_text(face = "bold", size = 8.5),
    strip.background = element_blank(),
    plot.title = element_blank(),
    plot.tag = element_text(face = "bold", size = 12),
    legend.position = "bottom",
    panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = 0.25)
  )

fit <- dat |>
  filter(fit_role == "Monthly likelihood") |>
  mutate(
    iso3 = factor(iso3, levels = country_order),
    country_short = factor(
      country_labels[as.character(iso3)],
      levels = unname(country_labels[country_order])
    )
  )

check_2026 <- dat |>
  filter(fit_role == "2026 PPC", !is.na(observed_cases), observed_cases > 0) |>
  mutate(
    iso3 = factor(iso3, levels = rev(country_order)),
    country_short = factor(
      country_labels[as.character(iso3)],
      levels = unname(country_labels[country_order])
    ),
    ratio = reported_median / observed_cases,
    lower_ratio = reported_predictive_lower / observed_cases,
    upper_ratio = reported_predictive_upper / observed_cases
  )

p_a <- ggplot(fit, aes(year)) +
  annotate("rect", xmin = 2020, xmax = 2022.99, ymin = -Inf, ymax = Inf,
           fill = "#F1F3F4") +
  geom_ribbon(
    aes(ymin = reported_lower, ymax = reported_upper),
    fill = "#9CC3E8", alpha = 0.5
  ) +
  geom_line(aes(y = reported_median), colour = "#226DB4", linewidth = 0.55) +
  geom_point(aes(y = observed_cases), colour = "#3C4043", alpha = 0.72, size = 0.9) +
  geom_point(
    data = check_2026,
    aes(x = year, y = observed_cases),
    inherit.aes = FALSE,
    shape = 21, fill = "white", colour = "#3C4043", size = 1.8, stroke = 0.6
  ) +
  facet_wrap(~country_short, ncol = 3, scales = "free_y") +
  scale_x_continuous(breaks = c(2010, 2015, 2020, 2025)) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    x = NULL,
    y = "Annual reported cases"
  ) +
  theme_lancet +
  theme(legend.position = "none")

p_b <- ggplot(check_2026, aes(ratio, iso3)) +
  geom_vline(xintercept = 1, linetype = 2, colour = "#6F7275", linewidth = 0.45) +
  geom_errorbar(
    aes(xmin = lower_ratio, xmax = upper_ratio),
    width = 0, linewidth = 0.8, colour = "#D55E00", orientation = "y"
  ) +
  geom_point(size = 2.7, colour = "#D55E00") +
  scale_x_log10(
    breaks = c(0.25, 0.5, 1, 2, 4, 8),
    labels = c("0.25×", "0.5×", "1×", "2×", "4×", "8×")
  ) +
  scale_y_discrete(labels = country_labels) +
  labs(
    x = "Predictive / observed reports",
    y = NULL
  ) +
  theme_lancet +
  theme(legend.position = "none")

figure <- p_a + p_b +
  plot_layout(widths = c(2.25, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.margin = margin(5, 6, 5, 5))

ggsave(
  file.path(figure_dir, "lancet_figure_2_model_performance.png"),
  figure, width = 183, height = 128, units = "mm", dpi = 600, bg = "white"
)
ggsave(
  file.path(figure_dir, "lancet_figure_2_model_performance.pdf"),
  figure, width = 183, height = 128, units = "mm", device = cairo_pdf,
  bg = "white"
)
