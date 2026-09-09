#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

audit <- read_csv_checked(
  fd_path("model2c_contact_sensitivity_boundary.csv"),
  c(
    "iso3",
    "country",
    "contact_sensitivity_upper_main_default",
    "contact_sensitivity_median",
    "contact_sensitivity_lower_95",
    "contact_sensitivity_upper_95",
    "p_near_upper",
    "boundary_flag",
    "boundary_flag_reason",
    "reported_no_contact_full_minus_scenario_median",
    "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Model 2C contact sensitivity audit") %>%
  mutate(
    boundary_flag = as.logical(boundary_flag),
    boundary_status = if_else(boundary_flag, "boundary_focus", "not_flagged"),
    iso3 = factor(iso3, levels = rev(iso3)),
    contact_sensitivity_upper_main_default = as.numeric(contact_sensitivity_upper_main_default),
    contact_sensitivity_median = as.numeric(contact_sensitivity_median),
    contact_sensitivity_lower_95 = as.numeric(contact_sensitivity_lower_95),
    contact_sensitivity_upper_95 = as.numeric(contact_sensitivity_upper_95),
    p_near_upper = as.numeric(p_near_upper),
    reported_no_contact_full_minus_scenario_median = as.numeric(reported_no_contact_full_minus_scenario_median),
    contact_label_x = if_else(contact_sensitivity_median > 4.45, contact_sensitivity_median - 0.22, contact_sensitivity_median + 0.28),
    contact_label_hjust = if_else(contact_sensitivity_median > 4.45, 1, 0),
    near_label_x = if_else(p_near_upper > 0.75, p_near_upper - 0.035, p_near_upper + 0.035),
    near_label_hjust = if_else(p_near_upper > 0.75, 1, 0)
  )

interval_panel <- ggplot(audit, aes(y = iso3, colour = boundary_status)) +
  geom_segment(
    aes(x = contact_sensitivity_lower_95, xend = contact_sensitivity_upper_95, yend = iso3),
    linewidth = 0.65,
    alpha = 0.95
  ) +
  geom_point(aes(x = contact_sensitivity_median), size = 1.8) +
  geom_text(
    aes(x = contact_label_x, label = sprintf("%.2f", contact_sensitivity_median), hjust = contact_label_hjust),
    size = figure_text$annotation_large,
    show.legend = FALSE
  ) +
  geom_vline(aes(xintercept = contact_sensitivity_upper_main_default), linetype = "22", linewidth = 0.28, colour = clr("dark_grey")) +
  scale_colour_manual(
    values = c(boundary_focus = clr("red"), not_flagged = clr("blue")),
    breaks = c("boundary_focus", "not_flagged"),
    labels = c("Boundary focus", "Not flagged"),
    name = "Audit flag"
  ) +
  scale_x_continuous(limits = c(0, 5.2), breaks = 0:5, expand = expansion(mult = c(0.01, 0.03))) +
  labs(x = "Contact sensitivity; point = median, line = 95% UI", y = "Country") +
  theme_nature() +
  theme(axis.text.y = element_text(size = figure_text$axis_text))

near_panel <- ggplot(audit, aes(y = iso3, x = p_near_upper, fill = boundary_status)) +
  geom_col(width = 0.70, colour = "white", linewidth = 0.18) +
  geom_text(
    aes(x = near_label_x, label = percent(p_near_upper, accuracy = 1), hjust = near_label_hjust),
    size = figure_text$annotation_large,
    colour = clr("black")
  ) +
  scale_fill_manual(
    values = c(boundary_focus = clr("red"), not_flagged = clr("light_grey")),
    breaks = c("boundary_focus", "not_flagged"),
    labels = c("Boundary focus", "Not flagged"),
    name = "Audit flag"
  ) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Posterior mass near upper bound", y = NULL) +
  theme_nature() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

contrast_panel <- audit %>%
  mutate(
    contrast_k = reported_no_contact_full_minus_scenario_median / 1000,
    contrast_label = label_short(reported_no_contact_full_minus_scenario_median),
    contrast_label_x = if_else(contrast_k > -2, -0.55, contrast_k + 0.55),
    contrast_label_hjust = if_else(contrast_k > -2, 1, 0)
  ) %>%
  ggplot(aes(y = iso3, x = contrast_k, fill = boundary_status)) +
  geom_col(width = 0.70, colour = "white", linewidth = 0.18) +
  geom_text(
    aes(
      x = contrast_label_x,
      label = contrast_label,
      hjust = contrast_label_hjust
    ),
    size = figure_text$annotation_large,
    colour = clr("black")
  ) +
  geom_vline(xintercept = 0, linewidth = 0.25, colour = clr("black")) +
  scale_fill_manual(
    values = c(boundary_focus = clr("red"), not_flagged = clr("light_grey")),
    guide = "none"
  ) +
  scale_x_continuous(limits = c(-40, 1.5), breaks = c(-40, -20, 0), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "No-contact contrast\n(k reported cases)", y = NULL) +
  theme_nature() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

figure <- panel_tags(
  (interval_panel | near_panel | contrast_panel) +
    plot_layout(widths = c(1.08, 0.92, 1.15), guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = figure_text$legend_text),
      legend.title = element_text(size = figure_text$legend_title)
    )
)

save_nature(figure, "supplement_model2c_contact_sensitivity_boundary.png", nature_full_width_mm, 124)
