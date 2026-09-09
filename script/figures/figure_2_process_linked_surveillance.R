#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(grid)
  library(patchwork)
  library(readr)
  library(ragg)
  library(scales)
  library(tidyr)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
repo_root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg)), "..", ".."), mustWork = TRUE)
setwd(repo_root)
source(file.path("script", "figures", "_common.R"))

# Figure contract
# Core conclusion: identical expected notifications can conceal inverse
# burden-ascertainment states; process-linked measurements define complementary
# information positions within a constructed benchmark.
# Archetype: compact asymmetric four-panel composite with a full-width pathway,
# a concise study-flow panel, a faceted exact-match curve and a full-width
# income-group readiness panel.
# Output: Nature Communications full width (183 mm), editable panel a-d SVGs
# plus a merged SVG/PDF and 600-dpi PNG/TIFF.
# Reviewer risks addressed here: the benchmark is constructed rather than
# empirical; 720 backgrounds are nested within nine country structures; the
# four ecological episodes are a separate observed-stream comparison arm.

layout <- read_csv_checked(
  fd_path("figure_2_process_linked_surveillance.csv"),
  required = c("record_type", "id", "label", "x", "y", "anchor_x", "anchor_y", "position", "value_class")
) |>
  mutate(label = gsub("\\\\n", "\n", label))
flow <- read_csv_checked(
  fd_path("figure_4_surveillance_design_flow.csv"),
  required = c("stage_order", "stage_id", "stage_label", "primary_count", "count_unit", "detail")
)
readiness <- read_csv_checked(
  file.path("manuscript", "tables", "income_group_readiness_summary.csv"),
  required = c(
    "world_bank_income_group", "n_settings", "n_joint_high_resolution",
    "n_selected_calibration_backgrounds", "joint_high_resolution_percent"
  )
)
empirical <- read_csv_checked(
  file.path("manuscript", "tables", "external_empirical_challenge_summary.csv"),
  required = c("iso3", "country")
)
matched_path <- read_csv_checked(
  fd_path("figure_4_representative_matched_path.csv"),
  required = c(
    "iso3", "group_id", "trajectory_class", "date", "expected_notifications",
    "max_absolute_notification_difference_all_720_pairs"
  )
) |>
  mutate(date = as.Date(date))

flow_count <- function(stage_id) {
  value <- flow$primary_count[flow$stage_id == stage_id]
  if (length(value) != 1L) {
    stop("Expected one flow row for stage_id=", stage_id, call. = FALSE)
  }
  as.integer(value)
}

detail_number <- function(stage_id, pattern) {
  detail <- flow$detail[flow$stage_id == stage_id]
  value <- stringr::str_extract(detail, pattern)
  value <- readr::parse_number(value)
  if (length(value) != 1L || is.na(value)) {
    stop("Could not extract a design count for stage_id=", stage_id, call. = FALSE)
  }
  as.integer(value)
}

n_screened <- sum(readiness$n_settings)
n_source_ready <- sum(readiness$n_joint_high_resolution)
n_calibration <- flow_count("calibration_backgrounds")
n_members_per_country <- detail_number("calibration_backgrounds", "[0-9]+ retained")
n_matched_backgrounds <- detail_number("calibration_backgrounds", "[0-9]+ paired")
n_profiles <- flow_count("matched_trajectories")
n_matched_months <- flow_count("exact_notification_matching")
n_packages <- flow_count("observation_packages")
n_replicates <- detail_number("observation_packages", "[0-9]+ observation replicates")
n_grouped_folds <- flow_count("held_out_evaluation")
n_country_holdouts <- detail_number("held_out_evaluation", "[0-9]+ whole-country")
n_episodes <- nrow(empirical)
matched_path_agreement <- matched_path |>
  select(date, trajectory_class, expected_notifications) |>
  pivot_wider(names_from = trajectory_class, values_from = expected_notifications) |>
  summarise(
    within_tolerance = all(
      abs(`Transmission-context retained` - `Ascertainment-change retained`) <= 1e-10
    )
  ) |>
  pull(within_tolerance)
stopifnot(
  n_screened == 217L,
  n_source_ready == 10L,
  n_calibration == 9L,
  n_members_per_country == 80L,
  n_matched_backgrounds == n_calibration * n_members_per_country,
  n_matched_backgrounds == 720L,
  n_profiles == 2L,
  n_matched_months == 36L,
  n_packages == 16L,
  n_replicates == 3L,
  n_grouped_folds == 5L,
  n_country_holdouts == 9L,
  n_episodes == 4L,
  nrow(matched_path) == n_profiles * n_matched_months,
  n_distinct(matched_path$trajectory_class) == n_profiles,
  n_distinct(matched_path$date) == n_matched_months,
  max(matched_path$max_absolute_notification_difference_all_720_pairs) <= 1e-10,
  matched_path_agreement
)

# A restrained semantic palette is shared across the four panels.
ink <- "#26323B"
muted <- "#62717B"
scaffold <- "#AEB8BF"
paper <- "#F4F6F7"
burden <- "#2F6690"
burden_light <- "#DCEAF5"
ascertainment <- "#C76B2A"
ascertainment_light <- "#F7E2D2"
linked <- "#2A8C82"
linked_light <- "#DCEFE8"
immune_light <- "#E9E6F2"
warning_light <- "#F5EBD9"

panel_theme <- theme_manuscript_void(plot_margin = margin(3, 4, 2, 4)) +
  theme(
    plot.tag = element_text(size = 8, face = "bold", colour = ink),
    plot.tag.position = c(0.002, 0.998)
  )

# -----------------------------------------------------------------------------
# Panel a: information positions along the infection-to-reporting pathway.
# Keep the earlier node-and-branch grammar; it is more legible than broad
# background bands when the panel is used as a compact full-width schematic.
# -----------------------------------------------------------------------------

nodes <- layout |> filter(record_type == "pathway")
streams <- layout |> filter(record_type == "stream")
edges <- tibble(
  x = nodes$x[-nrow(nodes)] + 0.38,
  xend = nodes$x[-1] - 0.38,
  y = 3,
  yend = 3
)

node_fill <- c(
  biological = burden_light,
  burden = burden_light,
  observation = "#ECEDEE",
  ascertainment = ascertainment_light,
  reported = "#E5E6E7"
)
stream_fill <- c(
  high = linked_light,
  intermediate = immune_light,
  limited = "#F0F0F0",
  warning = warning_light
)

p_a <- ggplot() +
  geom_segment(
    data = edges,
    aes(x = x, xend = xend, y = y, yend = yend),
    linewidth = 0.34, colour = scaffold,
    arrow = arrow(length = unit(1.25, "mm"), type = "closed")
  ) +
  geom_label(
    data = nodes,
    aes(x = x, y = y, label = label, fill = value_class),
    family = figure_font_family, fontface = "bold", size = 1.82,
    lineheight = 0.92, linewidth = 0.18,
    label.padding = unit(0.98, "mm"), label.r = unit(0.9, "mm"),
    colour = ink
  ) +
  geom_segment(
    data = streams,
    aes(x = anchor_x, xend = x, y = anchor_y, yend = y),
    linewidth = 0.26, linetype = "22", colour = scaffold
  ) +
  geom_label(
    data = streams,
    aes(x = x, y = y, label = label, fill = value_class),
    family = figure_font_family, size = 1.55, lineheight = 0.91,
    linewidth = 0.16, label.padding = unit(0.72, "mm"),
    label.r = unit(0.8, "mm"), colour = ink
  ) +
  scale_fill_manual(values = c(node_fill, stream_fill), guide = "none") +
  coord_cartesian(xlim = c(0.42, 6.58), ylim = c(0.78, 4.95), clip = "off") +
  labs(tag = "a") +
  panel_theme +
  theme(plot.margin = margin(2.5, 4, 1, 5))

# -----------------------------------------------------------------------------
# Panel b: compact study flow. The earlier card-and-arrow grammar is retained,
# but related counts are consolidated to avoid a tall seven-card stack.
# -----------------------------------------------------------------------------

study_flow <- tribble(
  ~step, ~value, ~descriptor, ~family,
  1, paste0(n_screened, " → ", n_source_ready, " → ", n_calibration), "screened → source-ready → primary backgrounds", "screen",
  2, paste0(n_calibration, " × ", n_members_per_country, " = ", n_matched_backgrounds), "country structures × retained members = paired backgrounds", "model",
  3, paste0(n_profiles, " × ", n_matched_months), "scenario profiles × exactly matched months", "model",
  4, paste0(n_packages, " × ", n_replicates), "packages × observation replicates", "benchmark",
  5, paste0(n_grouped_folds, " + ", n_country_holdouts), "grouped folds + whole-country holdouts", "benchmark",
  6, as.character(n_episodes), "observed 2024 episodes", "empirical"
)

flow_cols <- c(
  screen = "#ECEFF1",
  model = burden_light,
  benchmark = linked_light,
  empirical = ascertainment_light
)

p_b <- ggplot(study_flow, aes(x = 1, y = -step)) +
  geom_segment(
    data = study_flow |> filter(step < 5),
    aes(x = 1, xend = 1, y = -step - 0.37, yend = -step - 0.66),
    linewidth = 0.30, colour = scaffold,
    arrow = arrow(length = unit(1.15, "mm"), type = "closed")
  ) +
  geom_segment(
    data = tibble(y = -5.48),
    aes(x = 0.24, xend = 1.76, y = y, yend = y),
    inherit.aes = FALSE, linewidth = 0.23,
    colour = ascertainment, linetype = "22"
  ) +
  geom_label(
    aes(label = paste0(value, "\n", descriptor), fill = family),
    family = figure_font_family, size = 1.47, lineheight = 0.92,
    linewidth = 0.18, label.padding = unit(0.72, "mm"),
    label.r = unit(0.9, "mm"), colour = ink
  ) +
  annotate(
    "text", x = 0.24, y = -5.62, label = "Separate observed-stream arm",
    family = figure_font_family, fontface = "bold", size = 1.45,
    hjust = 0, vjust = 1, colour = ascertainment
  ) +
  scale_fill_manual(values = flow_cols, guide = "none") +
  coord_cartesian(xlim = c(0.12, 1.88), ylim = c(-6.46, -0.54), clip = "off") +
  labs(tag = "b") +
  panel_theme +
  theme(plot.margin = margin(2, 3, 2, 5))

# -----------------------------------------------------------------------------
# Panel c: the representative constructed pair retains the exact notification
# curve while exposing inverse latent burden-ascertainment states.
# -----------------------------------------------------------------------------

class_palette <- c(
  "Transmission-context retained" = burden,
  "Ascertainment-change retained" = ascertainment
)
class_linetypes <- c(
  "Transmission-context retained" = "solid",
  "Ascertainment-change retained" = "22"
)

trajectory_order <- c("Transmission-context retained", "Ascertainment-change retained")
state_labels <- c(
  "Transmission-context retained" = "Transmission-context retained\nControlled matched-notification profile",
  "Ascertainment-change retained" = "Ascertainment-change retained\nControlled matched-notification profile"
)
matched_path_panel <- matched_path |>
  mutate(
    trajectory_class = factor(trajectory_class, levels = trajectory_order),
    state_label = factor(
      unname(state_labels[as.character(trajectory_class)]),
      levels = unname(state_labels[trajectory_order])
    )
  )

p_c <- ggplot(
  matched_path_panel,
  aes(date, expected_notifications, colour = trajectory_class, linetype = trajectory_class)
) +
  geom_line(linewidth = 0.76, lineend = "round") +
  facet_wrap(vars(state_label), ncol = 1, scales = "fixed") +
  scale_colour_manual(values = class_palette) +
  scale_linetype_manual(values = class_linetypes) +
  scale_x_date(
    breaks = as.Date(c("2023-01-01", "2024-01-01", "2025-01-01")),
    labels = c("2023", "2024", "2025"),
    expand = expansion(mult = c(0.015, 0.02))
  ) +
  scale_y_continuous(
    breaks = c(0, 2000, 4000, 6000),
    labels = label_number(scale = 1e-3, accuracy = 1, suffix = "k"),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    tag = "c",
    x = NULL,
    y = "Expected notifications",
    colour = NULL,
    linetype = NULL
  ) +
  theme_manuscript(grid = "y", plot_margin = margin(3, 5, 3, 5)) +
  theme(
    plot.tag = element_text(size = 8, face = "bold", colour = ink),
    plot.tag.position = c(0.002, 1.00),
    axis.title.y = element_text(size = 5.3, margin = margin(r = 2.0)),
    axis.text = element_text(size = 5.1),
    strip.text = element_text(
      family = figure_font_family, size = 5.15, face = "bold",
      colour = ink, lineheight = 0.95, margin = margin(b = 1.8)
    ),
    strip.background = element_rect(fill = paper, colour = NA),
    panel.spacing.y = unit(1.45, "mm"),
    legend.position = "none"
  )

# -----------------------------------------------------------------------------
# Panel d: income-group source-readiness gap. This panel keeps the readiness
# boundary in the figure rather than only in the caption or text.
# -----------------------------------------------------------------------------

income_order <- c(
  "Low income", "Lower middle income", "Upper middle income",
  "High income", "Unclassified"
)
readiness_panel <- readiness |>
  mutate(
    world_bank_income_group = factor(world_bank_income_group, levels = rev(income_order)),
    ready_label = paste0(
      n_joint_high_resolution, "/", n_settings, " ready; ",
      n_selected_calibration_backgrounds, " calibrated"
    ),
    label_x = if_else(joint_high_resolution_percent > 0, joint_high_resolution_percent + 0.35, 0.65)
  )
low_resource_total <- readiness |>
  filter(world_bank_income_group %in% c("Low income", "Lower middle income")) |>
  summarise(n = sum(n_settings), ready = sum(n_joint_high_resolution), .groups = "drop")
low_resource_rows <- readiness_panel |>
  filter(world_bank_income_group %in% c("Low income", "Lower middle income"))

p_d <- ggplot(readiness_panel, aes(joint_high_resolution_percent, world_bank_income_group)) +
  geom_col(width = 0.58, fill = linked_light, colour = linked, linewidth = 0.28) +
  geom_point(
    data = low_resource_rows,
    aes(x = 0.18, y = world_bank_income_group),
    inherit.aes = FALSE, shape = 21, size = 2.05, stroke = 0.45,
    fill = ascertainment_light, colour = ascertainment
  ) +
  geom_text(
    aes(x = label_x, label = ready_label),
    family = figure_font_family, size = 1.72,
    hjust = 0, colour = ink
  ) +
  scale_x_continuous(
    limits = c(0, 12.2),
    breaks = c(0, 5, 10),
    labels = label_percent(scale = 1, accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    tag = "d",
    x = "Settings meeting the joint high-resolution criterion",
    y = NULL
  ) +
  theme_manuscript(grid = "x", plot_margin = margin(3, 5, 2.5, 5)) +
  theme(
    plot.tag = element_text(size = 8, face = "bold", colour = ink),
    plot.tag.position = c(0.002, 1.00),
    axis.title.x = element_text(size = 5.3, margin = margin(t = 2.0)),
    axis.text = element_text(size = 5.1)
  )

# -----------------------------------------------------------------------------
# Compose the native ggplot objects into the supplementary figure.
# -----------------------------------------------------------------------------

figure_design <- "
AA
BC
DD
"

figure <- patchwork::wrap_plots(
  A = p_a,
  B = p_b,
  C = p_c,
  D = p_d,
  design = figure_design,
  widths = c(0.39, 0.61),
  heights = c(0.30, 0.52, 0.18)
) +
  plot_annotation(theme = theme(plot.margin = margin(1.5, 1.5, 1.5, 1.5)))

stem <- "figure_2_process_linked_surveillance"
width_mm <- 183
height_mm <- 128

ragg::agg_png(
  supplement_fig_path(paste0(stem, ".png")),
  width = width_mm / 25.4,
  height = height_mm / 25.4,
  units = "in",
  res = 600,
  background = "white"
)
print(figure)
dev.off()
message("Wrote Supplementary Figure 24 as PNG.")
