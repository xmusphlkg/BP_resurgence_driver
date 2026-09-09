#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

crosswalk <- read_csv_checked(
  fd_path("model2c_literature_scenario_evidence_heatmap.csv"),
  c(
    "literature_mechanism",
    "model2c_scenario_id",
    "countries_with_support",
    "countries_with_boundary_concern",
    "claim_strength",
    "support_count",
    "boundary_count",
    "claim_strength_score",
    "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Model 2C literature scenario crosswalk") %>%
  mutate(
    literature_mechanism = factor(literature_mechanism, levels = rev(unique(literature_mechanism))),
    scenario_label = unname(scenario2c_labels[model2c_scenario_id]),
    scenario_label = factor(scenario_label, levels = unname(scenario2c_labels[scenario2c_levels[scenario2c_levels %in% model2c_scenario_id]])),
    claim_class = case_when(
      claim_strength == "strong_scenario_supported" ~ "strong",
      str_detect(claim_strength, "supported") ~ "supported",
      str_detect(claim_strength, "boundary") ~ "boundary",
      str_detect(claim_strength, "audit") ~ "audit",
      TRUE ~ "not_supported"
    ),
    claim_class = factor(claim_class, levels = c("strong", "supported", "boundary", "audit", "not_supported")),
    evidence_total = pmax(support_count + boundary_count, 1),
    support_label = paste0("S", support_count, "/B", boundary_count)
  )

claim_palette <- c(
  strong = clr("green"),
  supported = clr("blue"),
  boundary = clr("orange"),
  audit = clr("dark_grey"),
  not_supported = clr("light_grey")
)

crosswalk_panel <- ggplot(crosswalk, aes(x = scenario_label, y = literature_mechanism, fill = claim_class)) +
  geom_point(aes(size = evidence_total), shape = 21, colour = "white", stroke = 0.35, alpha = 0.96) +
  geom_text(aes(label = support_label), size = figure_text$annotation, colour = clr("black")) +
  scale_fill_manual(
    values = claim_palette,
    breaks = c("strong", "supported", "boundary", "audit", "not_supported"),
    labels = c("Strong", "Supported", "Boundary", "Audit", "No clear signal"),
    name = "Claim class"
  ) +
  scale_size_continuous(
    range = c(4.0, 7.0),
    guide = "none"
  ) +
  labs(x = "Model 2C scenario", y = "Literature mechanism") +
  theme_nature() +
  theme(
    axis.text.x = element_text(
      angle = 34,
      hjust = 1,
      vjust = 1,
      size = figure_text$axis_text
    ),
    axis.text.y = element_text(size = figure_text$axis_text),
    panel.grid = element_blank()
  )

support_panel <- crosswalk %>%
  mutate(
    literature_mechanism = factor(literature_mechanism, levels = levels(crosswalk$literature_mechanism)),
    support_label_x = support_count + 0.25,
    boundary_label_x = -boundary_count - 0.25
  ) %>%
  ggplot(aes(y = literature_mechanism)) +
  geom_col(aes(x = support_count, fill = "support"), width = 0.58, alpha = 0.95) +
  geom_col(aes(x = -boundary_count, fill = "boundary"), width = 0.58, alpha = 0.95) +
  geom_text(
    aes(x = support_label_x, label = support_count),
    size = figure_text$annotation_large,
    hjust = 0,
    colour = clr("black")
  ) +
  geom_text(
    aes(x = boundary_label_x, label = boundary_count),
    size = figure_text$annotation_large,
    hjust = 1,
    colour = clr("black")
  ) +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = clr("black")) +
  scale_fill_manual(
    values = c(support = clr("green"), boundary = clr("orange")),
    breaks = c("support", "boundary"),
    labels = c("Supporting countries", "Boundary countries"),
    name = NULL
  ) +
  scale_x_continuous(
    breaks = c(-9, -6, -3, 0, 3, 6, 9),
    labels = c("9", "6", "3", "0", "3", "6", "9"),
    limits = c(-10.5, 10.5),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(x = "Countries", y = NULL) +
  theme_nature() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

figure <- panel_tags(
  (crosswalk_panel | support_panel) +
    plot_layout(widths = c(1.55, 0.75), guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = figure_text$legend_text),
      legend.title = element_text(size = figure_text$legend_title)
    )
)

save_nature(figure, "supplement_model2c_literature_scenario_evidence.png", nature_full_width_mm, 128)
