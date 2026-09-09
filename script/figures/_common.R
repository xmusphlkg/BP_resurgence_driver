suppressPackageStartupMessages({
  library(dplyr)
  library(forcats)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(stringr)
  library(tidyr)
})

required_packages <- c("ggplot2", "ggsci", "patchwork", "dplyr", "readr", "scales", "tidyr", "stringr", "forcats")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "), call. = FALSE)
}

root <- normalizePath(getwd(), mustWork = TRUE)
fig_dir <- file.path(root, "manuscript", "figures")
supplement_fig_dir <- file.path(root, "manuscript", "appendix", "figures")
figure_data_dir <- file.path(root, "manuscript", "figure_data")
outputs_dir <- file.path(root, "outputs")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supplement_fig_dir, recursive = TRUE, showWarnings = FALSE)
model2b_output_version <- "model2b_v2_20260616_layered_endemic_background_attribution"
model2c_output_version <- Sys.getenv(
  "MODEL2C_OUTPUT_VERSION",
  unset = "model2c_v10_20260705_cohort_routine_natural_exposure"
)

nature_full_width_mm <- 183
nature_single_width_mm <- 89
nature_max_height_mm <- 170

fig_path <- function(name) file.path(fig_dir, name)
supplement_fig_path <- function(name) file.path(supplement_fig_dir, name)
fd_path <- function(name) file.path(figure_data_dir, name)
out_path <- function(name) file.path(outputs_dir, name)

read_csv_checked <- function(path, required = character()) {
  if (!file.exists(path)) {
    stop("Required figure input is missing: ", path, call. = FALSE)
  }
  df <- readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop("Figure input ", path, " is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  df
}

read_csv_optional <- function(path) {
  if (!file.exists(path)) {
    return(tibble())
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

read_csv_first_available <- function(paths, required = character()) {
  existing <- paths[file.exists(paths)]
  if (length(existing) == 0) {
    stop(
      "None of the candidate figure inputs exists: ",
      paste(paths, collapse = "; "),
      call. = FALSE
    )
  }
  version_expectations <- c(
    model2b_output_version = model2b_output_version,
    model2c_output_version = model2c_output_version
  )
  version_columns <- intersect(names(version_expectations), required)
  if (length(version_columns) > 0) {
    mismatches <- character()
    for (path in existing) {
      df <- read_csv_checked(path, required)
      matches <- vapply(
        version_columns,
        function(column) {
          versions <- unique(stats::na.omit(as.character(df[[column]])))
          length(versions) == 1 && versions[[1]] == version_expectations[[column]]
        },
        logical(1)
      )
      if (all(matches)) {
        return(df)
      }
      observed <- vapply(
        version_columns,
        function(column) {
          versions <- unique(stats::na.omit(as.character(df[[column]])))
          if (length(versions) == 0) "<none>" else paste(versions, collapse = ", ")
        },
        character(1)
      )
      mismatches <- c(mismatches, paste0(path, " [", paste(names(observed), observed, sep = "=", collapse = "; "), "]"))
    }
    stop(
      "None of the candidate figure inputs matches the current output version(s). Expected ",
      paste(version_columns, version_expectations[version_columns], sep = "=", collapse = "; "),
      ". Observed: ",
      paste(mismatches, collapse = " | "),
      call. = FALSE
    )
  }
  read_csv_checked(existing[[1]], required)
}

require_model2b_current_version <- function(df, label = "Model 2B input") {
  if (!"model2b_output_version" %in% names(df)) {
    stop(label, " is missing model2b_output_version.", call. = FALSE)
  }
  versions <- unique(stats::na.omit(as.character(df$model2b_output_version)))
  if (length(versions) != 1 || versions[[1]] != model2b_output_version) {
    stop(
      label,
      " is not from the current Model 2B output version. Expected ",
      model2b_output_version,
      "; observed ",
      paste(versions, collapse = ", "),
      call. = FALSE
    )
  }
  df
}

require_model2c_current_version <- function(df, label = "Model 2C input") {
  if (!"model2c_output_version" %in% names(df)) {
    stop(label, " is missing model2c_output_version.", call. = FALSE)
  }
  versions <- unique(stats::na.omit(as.character(df$model2c_output_version)))
  if (length(versions) != 1 || versions[[1]] != model2c_output_version) {
    stop(
      label,
      " is not from the current Model 2C output version. Expected ",
      model2c_output_version,
      "; observed ",
      paste(versions, collapse = ", "),
      call. = FALSE
    )
  }
  df
}

label_short <- function(x) {
  dplyr::case_when(
    is.na(x) ~ NA_character_,
    abs(x) >= 1e6 ~ paste0(format(round(x / 1e6, 1), trim = TRUE), "M"),
    abs(x) >= 1e3 ~ paste0(format(round(x / 1e3, 0), trim = TRUE), "k"),
    TRUE ~ scales::comma(x, accuracy = 1)
  )
}

bmj_palette <- unname(ggsci::pal_bmj("default")(9))
names(bmj_palette) <- c("blue", "gold", "pink", "purple", "orange", "green", "teal", "red", "grey")

discrete_palette_primary <- unname(bmj_palette)

discrete_palette_alt <- unname(bmj_palette[c(
  "blue", "teal", "orange", "red", "purple", "green", "gold", "pink", "grey"
)])

continuous_palette_warm <- colorRampPalette(unname(bmj_palette[c("red", "orange", "gold")]))(7)

continuous_palette_teal <- colorRampPalette(unname(bmj_palette[c("blue", "teal", "green")]))(7)

pal <- c(
  white = "#FFFFFF",
  black = "#1B1B1B",
  grey = bmj_palette[["grey"]],
  dark_grey = "#4B4D4F",
  light_grey = "#DADCDD",
  paper = "#F7F8F8",
  blue = bmj_palette[["blue"]],
  sky = bmj_palette[["teal"]],
  green = bmj_palette[["green"]],
  orange = bmj_palette[["orange"]],
  gold = bmj_palette[["gold"]],
  pink = bmj_palette[["pink"]],
  red = bmj_palette[["red"]],
  crimson = bmj_palette[["pink"]],
  purple = bmj_palette[["purple"]],
  plum = bmj_palette[["pink"]],
  teal = bmj_palette[["teal"]],
  navy = bmj_palette[["blue"]],
  brown = bmj_palette[["pink"]],
  sand = bmj_palette[["gold"]]
)

clr <- function(name) unname(pal[[name]])

figure_font_family <- "sans"

figure_text <- list(
  base = 6.3,
  title = 6.8,
  subtitle = 5.5,
  axis_title = 6.3,
  axis_text = 5.6,
  axis_text_dense = 4.8,
  strip = 6.0,
  strip_dense = 5.2,
  legend_title = 5.9,
  legend_text = 5.4,
  legend_text_dense = 4.8,
  annotation_dense = 1.55,
  annotation = 1.75,
  annotation_large = 2.05,
  annotation_emphasis = 2.55,
  tag = 8.0
)

figure_colours <- list(
  text = clr("black"),
  muted_text = clr("dark_grey"),
  axis = clr("black"),
  reference = clr("grey"),
  grid = clr("paper"),
  background = clr("white")
)

figure_breaks <- list(
  auc = seq(0.5, 1, by = 0.1),
  probability = c(0, 0.25, 0.50, 0.75, 1),
  percent = c(0, 25, 50, 75, 100),
  percent_coarse = c(0, 50, 100),
  ratio_4x = c(0.25, 0.5, 1, 2, 4),
  ratio_2x = c(0.5, 1, 2, 4, 8),
  ratio_fine = c(0.5, 0.75, 1, 1.5, 2)
)

figure_breaks_linear <- function(n = 4) {
  scales::breaks_pretty(n = n)
}

figure_breaks_log <- function(n = 5, base = 10) {
  scales::breaks_log(n = n, base = base)
}

figure_breaks_year <- function(start, end, by = 5) {
  seq.int(start, end, by = by)
}

figure_breaks_date <- function(start, end, by = 5, month = 1L, day = 1L) {
  years <- figure_breaks_year(start, end, by)
  as.Date(sprintf("%04d-%02d-%02d", years, as.integer(month), as.integer(day)))
}

theme_manuscript <- function(
    base_size = figure_text$base,
    grid = c("none", "x", "y", "both"),
    border = FALSE,
    show_titles = FALSE,
    plot_margin = margin(3, 4, 3, 4)) {
  grid <- match.arg(grid)
  size_delta <- base_size - figure_text$base
  scaled_size <- function(size) pmax(1, size + size_delta)
  grid_line <- element_line(linewidth = 0.14, colour = figure_colours$grid)

  theme_classic(base_size = base_size, base_family = figure_font_family) +
    theme(
      text = element_text(
        family = figure_font_family,
        colour = figure_colours$text,
        size = scaled_size(figure_text$base)
      ),
      axis.line = element_line(linewidth = 0.26, colour = figure_colours$axis),
      axis.ticks = element_line(linewidth = 0.22, colour = figure_colours$axis),
      axis.ticks.length = unit(1.0, "mm"),
      axis.text = element_text(
        colour = figure_colours$text,
        size = scaled_size(figure_text$axis_text)
      ),
      axis.title = element_text(
        colour = figure_colours$text,
        size = scaled_size(figure_text$axis_title),
        face = "bold"
      ),
      legend.title = element_text(
        size = scaled_size(figure_text$legend_title),
        face = "bold",
        colour = figure_colours$text
      ),
      legend.text = element_text(
        size = scaled_size(figure_text$legend_text),
        colour = figure_colours$text
      ),
      legend.key.height = unit(2.6, "mm"),
      legend.key.width = unit(4.0, "mm"),
      legend.spacing.x = unit(1.4, "mm"),
      legend.box = "horizontal",
      legend.position = "bottom",
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      panel.grid.major.x = if (grid %in% c("x", "both")) grid_line else element_blank(),
      panel.grid.major.y = if (grid %in% c("y", "both")) grid_line else element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = if (border) {
        element_rect(colour = figure_colours$axis, fill = NA, linewidth = 0.24)
      } else {
        element_blank()
      },
      # Figure-level interpretation belongs in the manuscript legend. Keeping
      # both elements blank here prevents an individual renderer from adding
      # a competing title or subtitle to a publication figure.
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.caption = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(
        face = "bold",
        colour = figure_colours$text,
        size = scaled_size(figure_text$strip)
      ),
      plot.tag = element_text(
        face = "bold",
        colour = figure_colours$text,
        size = scaled_size(figure_text$tag)
      ),
      plot.margin = plot_margin
    )
}

# Backward-compatible entry point for active supplementary figures.
theme_nature <- function(
    base_size = figure_text$base,
    grid = c("none", "x", "y", "both"),
    border = FALSE,
    show_titles = FALSE,
    plot_margin = margin(3, 4, 3, 4)) {
  theme_manuscript(
    base_size = base_size,
    grid = match.arg(grid),
    border = border,
    show_titles = show_titles,
    plot_margin = plot_margin
  )
}

theme_manuscript_void <- function(plot_margin = margin(3, 4, 3, 4)) {
  theme_void(base_family = figure_font_family, base_size = figure_text$base) +
    theme(
      text = element_text(
        family = figure_font_family,
        colour = figure_colours$text,
        size = figure_text$base
      ),
      plot.tag = element_text(
        face = "bold",
        colour = figure_colours$text,
        size = figure_text$tag
      ),
      plot.margin = plot_margin
    )
}

theme_set(theme_manuscript())

model2_display <- "Monthly age-structured transmission bridge"
model2_short <- "dynamic bridge"
model2_pilot_display <- "nine-country dynamic bridge"

country_abbrev <- c(
  AUS = "Australia",
  BRA = "Brazil",
  CAN = "Canada",
  CHN = "China",
  FRA = "France",
  GBR = "United Kingdom",
  JPN = "Japan",
  NLD = "Netherlands",
  NZL = "New Zealand",
  SGP = "Singapore",
  SWE = "Sweden",
  THA = "Thailand",
  USA = "United States",
  ZAF = "South Africa"
)

country_abbrev_note <- paste(
  paste0(names(country_abbrev), ", ", unname(country_abbrev)),
  collapse = "; "
)

mechanism_palette <- c(
  low_circulation_missed_boosting = clr("blue"),
  contact_circulation_pathway = clr("blue"),
  covid_contact_shock = clr("blue"),
  routine_immunization_disruption = clr("orange"),
  observation_reporting_boundary = clr("dark_grey"),
  reporting_testing_shift = clr("dark_grey"),
  post_2023_transmission_shift = clr("red"),
  excess_epidemic_pressure = clr("purple"),
  outbreak_ignition_pulse = clr("orange"),
  post_pandemic_transmission_drift = clr("red"),
  vaccine_waning = clr("green"),
  vaccine_waning_booster_gap = clr("green"),
  birth_cohort_demography = clr("purple")
)

mechanism_labels <- c(
  low_circulation_missed_boosting = "Low circulation/\nmissed boosting",
  contact_circulation_pathway = "Reduced circulation/\nmissed exposure",
  covid_contact_shock = "Contact shock",
  routine_immunization_disruption = "Routine disruption",
  observation_reporting_boundary = "Observation boundary",
  reporting_testing_shift = "Reporting/\ntesting shift",
  post_2023_transmission_shift = "Post-2023\ntransmission shift",
  excess_epidemic_pressure = "Excess epidemic\npressure",
  outbreak_ignition_pulse = "Outbreak ignition\npulse",
  post_pandemic_transmission_drift = "Post-pandemic\ntransmission drift",
  vaccine_waning = "Waning",
  vaccine_waning_booster_gap = "Waning/\nbooster gap",
  birth_cohort_demography = "Birth-cohort\ndemography"
)

scenario2c_levels <- c(
  "full_model",
  "no_contact_suppression",
  "no_missed_exposure_boosting",
  "restored_circulation_boosting_path",
  "no_routine_immunization_disruption",
  "low_vaccine_waning",
  "restored_booster_reach",
  "fixed_birth_cohort_demography",
  "susceptibility_neutral_combined",
  "no_post_2023_transmission_context",
  "no_reporting_testing_shift"
)

scenario2c_labels <- c(
  full_model = "Full model",
  no_contact_suppression = "Restored contact path",
  no_missed_exposure_boosting = "Normal 2021-2024 boosting",
  restored_circulation_boosting_path = "Restored 2021-2024 circulation + boosting",
  no_routine_immunization_disruption = "Restored routine flow",
  low_vaccine_waning = "No vaccine waning",
  restored_booster_reach = "Restored booster reach",
  fixed_birth_cohort_demography = "Fixed age structure",
  susceptibility_neutral_combined = "All susceptibility switches neutral",
  no_post_2023_transmission_context = "Post-2023 context audit",
  no_reporting_testing_shift = "Reporting visibility audit"
)

scenario2c_palette <- c(
  full_model = clr("black"),
  no_contact_suppression = clr("blue"),
  no_missed_exposure_boosting = clr("teal"),
  restored_circulation_boosting_path = clr("sky"),
  no_routine_immunization_disruption = clr("orange"),
  low_vaccine_waning = clr("green"),
  restored_booster_reach = clr("gold"),
  fixed_birth_cohort_demography = clr("purple"),
  susceptibility_neutral_combined = clr("dark_grey"),
  no_post_2023_transmission_context = clr("red"),
  no_reporting_testing_shift = clr("brown")
)

preferred_factor_palette <- c(
  low_circulation_missed_boosting_weight_percent = clr("blue"),
  routine_immunization_disruption_weight_percent = clr("orange"),
  vaccine_waning_booster_gap_weight_percent = clr("green"),
  birth_cohort_demography_weight_percent = clr("purple")
)

preferred_factor_labels <- c(
  low_circulation_missed_boosting_weight_percent = "Low circulation/\nmissed boosting",
  routine_immunization_disruption_weight_percent = "Routine\nimmunisation",
  vaccine_waning_booster_gap_weight_percent = "Waning/\nbooster gap",
  birth_cohort_demography_weight_percent = "Birth-cohort\ndemography"
)

trend_palette <- c(
  below_recent_pre_pandemic_reference = clr("blue"),
  near_recent_pre_pandemic_reference = clr("green"),
  above_recent_pre_pandemic_reference = clr("red"),
  no_current_year_time_resolved_signal = clr("light_grey"),
  no_recent_pre_pandemic_reference = clr("light_grey")
)

trend_labels <- c(
  below_recent_pre_pandemic_reference = "Below 2018-2019",
  near_recent_pre_pandemic_reference = "Near 2018-2019",
  above_recent_pre_pandemic_reference = "Above 2018-2019",
  no_current_year_time_resolved_signal = "No 2026 signal",
  no_recent_pre_pandemic_reference = "No reference"
)

scope_palette <- c(
  peak_only = clr("blue"),
  qualified_current = clr("orange"),
  current_elevated = clr("red"),
  no_signal = clr("light_grey")
)

region_palette <- c(
  AFRO = clr("blue"),
  AMRO = clr("green"),
  EMRO = clr("orange"),
  EURO = clr("red"),
  SEARO = clr("teal"),
  WPRO = clr("gold")
)

evidence_palette <- c(
  supported = clr("green"),
  partial = clr("orange"),
  missing = clr("light_grey"),
  caution = clr("red"),
  selected = clr("blue"),
  excluded = clr("grey")
)

policy_labels <- c(
  P1_DTP_catch_up = "DTP catch-up",
  P2_maternal_Tdap_80 = "Maternal Tdap 80%",
  P3_maternal_Tdap_90 = "Maternal Tdap 90%",
  P4_preschool_booster = "Preschool booster",
  P5_adolescent_booster = "Adolescent booster",
  P6_maternal_plus_adolescent = "Maternal + adolescent",
  P7_targeted_adult_booster = "Targeted adult booster",
  P8_next_generation_vaccine = "Next-generation vaccine",
  P0_status_quo = "Status quo"
)

policy_palette <- setNames(
  rep(c(clr("green"), clr("blue"), clr("orange"), clr("red"), clr("teal"), clr("purple"), clr("gold")), length.out = length(policy_labels)),
  names(policy_labels)
)

age_order <- c(
  "infant_0_2m", "infant_3_11m", "child_1_4y", "child_5_9y",
  "adolescent_10_17y", "adult_18_39y", "adult_40_64y", "adult_65plus"
)

age_labels <- c(
  infant_0_2m = "<3 months",
  infant_3_11m = "3-11 months",
  child_1_4y = "1-4 years",
  child_5_9y = "5-9 years",
  adolescent_10_17y = "10-17 years",
  adult_18_39y = "18-39 years",
  adult_40_64y = "40-64 years",
  adult_65plus = "65+ years"
)

save_nature <- function(plot, filename, width_mm, height_mm, dpi = 320, output_dir = NULL) {
  if (width_mm > nature_full_width_mm || height_mm > nature_max_height_mm) {
    warning(
      "Figure size exceeds Nature full-width or maximum-height guidance: ",
      filename,
      " (",
      width_mm,
      " x ",
      height_mm,
      " mm).",
      call. = FALSE
    )
  }
  if (is.null(output_dir)) {
    output_dir <- if (startsWith(basename(filename), "supplement_")) supplement_fig_dir else fig_dir
  }
  output_path <- file.path(output_dir, filename)
  ggsave(
    filename = output_path,
    plot = plot,
    width = width_mm,
    height = height_mm,
    units = "mm",
    dpi = dpi,
    bg = "white"
  )
  message("Wrote ", output_path)
}

save_main_figure <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  for (extension in c("png", "pdf")) {
    save_nature(
      plot,
      paste0(stem, ".", extension),
      width_mm = width_mm,
      height_mm = height_mm,
      dpi = dpi
    )
  }
}

save_supplement_figure <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  save_nature(
    plot,
    paste0(stem, ".png"),
    width_mm = width_mm,
    height_mm = height_mm,
    dpi = dpi,
    output_dir = supplement_fig_dir
  )
}

panel_tags <- function(plot) {
  plot +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = figure_text$tag))
}

pilot_iso3 <- c("AUS", "BRA", "CHN", "GBR", "JPN", "NZL", "SWE", "THA", "USA")

pilot_order <- tibble(
  iso3 = pilot_iso3,
  pilot_rank = seq_along(pilot_iso3)
)

evidence_status <- function(value) {
  case_when(
    is.na(value) ~ "missing",
    is.logical(value) & value ~ "supported",
    is.logical(value) & !value ~ "missing",
    as.character(value) %in% c(
      "TRUE",
      "true",
      "supported",
      "fine_source_bins",
      "country_age_month_composite_likelihood_supported",
      "source_age_bin_likelihood_supported"
    ) ~ "supported",
    str_detect(as.character(value), regex("fragile|limited|composite|partial|available", ignore_case = TRUE)) ~ "partial",
    str_detect(as.character(value), regex("missing|unavailable|no_", ignore_case = TRUE)) ~ "missing",
    TRUE ~ "partial"
  )
}

as_plot_date <- function(x) {
  as.Date(substr(as.character(x), 1, 10))
}

short_country <- function(country) {
  dplyr::case_when(
    str_detect(country, "United Kingdom") ~ "United Kingdom",
    str_detect(country, "United States") ~ "United States",
    TRUE ~ country
  )
}

country_display_name <- function(country) {
  dplyr::recode(
    country,
    "United Kingdom of Great Britain and Northern Ireland" = "United Kingdom",
    "United States of America" = "United States",
    "Iran (Islamic Republic of)" = "Iran",
    .default = country
  )
}

format_country_panel <- function(iso3, country) {
  iso3
}

format_country_inline <- function(iso3, country) {
  iso3
}
