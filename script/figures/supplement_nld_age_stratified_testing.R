#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

testing <- read_csv_checked(
  out_path("diagnostic_age_testing_summary.csv"),
  c(
    "country",
    "iso3",
    "period_start",
    "period_end",
    "age_group",
    "recommended_test",
    "total_tests",
    "serology_tests",
    "serology_percent",
    "source_ids",
    "source",
    "source_type",
    "notes"
  )
) %>%
  mutate(
    age_group = factor(age_group, levels = c("<1 year", ">1 year with >3 weeks of coughing")),
    recommended_test = factor(recommended_test, levels = c("PCR or culture", "Serology")),
    label = sprintf("%d/%d tests\n%d%% serology", serology_tests, total_tests, serology_percent)
  )

test_palette <- c(
  "PCR or culture" = clr("blue"),
  "Serology" = clr("orange")
)

p_testing <- ggplot(testing, aes(x = serology_percent, y = age_group, fill = recommended_test)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.22) +
  geom_vline(xintercept = 50, linewidth = 0.35, linetype = "22", colour = pal["grey"]) +
  scale_fill_manual(
    values = test_palette,
    name = "Recommended test"
  ) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = figure_breaks$percent,
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0.02, 0.03))
  ) +
  labs(
    x = "Share of serology tests",
    y = "Age group"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    legend.position = "bottom",
    plot.margin = margin(5, 6, 8, 5)
  ) +
  coord_cartesian()

figure <- panel_tags(p_testing)

save_nature(figure, "supplement_nld_age_stratified_testing.png", nature_full_width_mm, 94)
