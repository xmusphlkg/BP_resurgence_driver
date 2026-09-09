#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

primary_dtp_coverage_type <- "WHO/UNICEF Estimates of National Immunization Coverage"
primary_dtp_coverage_source_label <- "WHO/UNICEF estimates (WUENIC)"
primary_dtp_carry_forward_label <- "Carry-forward from WUENIC"

official_strategy <- read_csv_checked(
  file.path(root, "data", "raw", "immunization", "official_pertussis_schedule_strategy.csv"),
  c(
    "iso3", "country", "schedule_year", "target_pop", "target_group", "token",
    "vaccine_code", "vaccine_description", "vaccine_family", "source_name",
    "source_url", "special_conditional_route", "catch_up_route"
  )
)
for (optional_column in c(
  "target_description", "source_comment", "official_gestation_window",
  "maternal_vaccine_label", "maternal_vaccine_family", "event_window_label",
  "analysis_note"
)) {
  if (!optional_column %in% names(official_strategy)) {
    official_strategy[[optional_column]] <- ""
  }
}
official_strategy <- official_strategy %>%
  mutate(
    iso3 = str_to_upper(as.character(iso3)),
    country = as.character(country),
    schedule_year = suppressWarnings(as.integer(schedule_year)),
    target_pop = str_to_upper(coalesce(as.character(target_pop), "")),
    target_group = coalesce(as.character(target_group), ""),
    token = coalesce(as.character(token), ""),
    vaccine_code = str_to_upper(coalesce(as.character(vaccine_code), "")),
    vaccine_description = coalesce(as.character(vaccine_description), ""),
    vaccine_family = coalesce(as.character(vaccine_family), ""),
    source_comment = coalesce(as.character(source_comment), ""),
    source_name = coalesce(as.character(source_name), ""),
    source_url = coalesce(as.character(source_url), ""),
    official_gestation_window = coalesce(as.character(official_gestation_window), ""),
    maternal_vaccine_label = coalesce(as.character(maternal_vaccine_label), ""),
    maternal_vaccine_family = coalesce(as.character(maternal_vaccine_family), ""),
    event_window_label = coalesce(as.character(event_window_label), ""),
    analysis_note = coalesce(as.character(analysis_note), ""),
    special_conditional_route = coalesce(as.logical(special_conditional_route), FALSE),
    catch_up_route = coalesce(as.logical(catch_up_route), FALSE)
  )

collapse_tokens <- function(x) {
  x <- str_trim(na.omit(as.character(x)))
  x <- x[x != ""]
  paste(unique(x), collapse = ";")
}

collapse_semicolon <- function(x) {
  x <- str_trim(na.omit(as.character(x)))
  x <- x[x != ""]
  paste(unique(x), collapse = "; ")
}

family_to_platform <- function(x) {
  x <- str_to_lower(str_trim(coalesce(x, "")))
  case_when(
    x %in% c("dtwp", "dtpw", "wp", "whole-cell", "whole cell") ~ "wP",
    x %in% c("dtap", "tdap", "ap", "acellular") ~ "aP",
    TRUE ~ "unknown"
  )
}

schedule <- official_strategy %>%
  group_by(iso3, country) %>%
  summarise(
    year_start = suppressWarnings(min(schedule_year, na.rm = TRUE)),
    year_end = suppressWarnings(max(schedule_year, na.rm = TRUE)),
    primary_schedule_months = collapse_tokens(token[
      target_pop == "INFANTS" &
        target_group == "Routine" &
        !special_conditional_route &
        !catch_up_route
    ]),
    booster_ages = collapse_tokens(token[
      target_pop != "PW" &
        target_pop != "INFANTS" &
        !special_conditional_route &
        !catch_up_route
    ]),
    maternal_program = any(target_group == "Pregnancy" | target_pop == "PW", na.rm = TRUE),
    maternal_recommended_week = collapse_semicolon(official_gestation_window[target_group == "Pregnancy" | target_pop == "PW"]),
    vaccine_type_primary = {
      platforms <- unique(family_to_platform(vaccine_family[
        target_pop == "INFANTS" &
          target_group == "Routine" &
          !special_conditional_route &
          !catch_up_route
      ]))
      platforms <- platforms[platforms != "unknown"]
      case_when(
        length(platforms) == 1 ~ platforms[[1]],
        length(platforms) > 1 ~ "mixed",
        TRUE ~ "unknown"
      )
    },
    vaccine_type_booster = {
      platforms <- unique(family_to_platform(vaccine_family[
        target_pop != "PW" &
          target_pop != "INFANTS" &
          !special_conditional_route &
          !catch_up_route
      ]))
      platforms <- platforms[platforms != "unknown"]
      case_when(
        length(platforms) == 1 ~ platforms[[1]],
        length(platforms) > 1 ~ "mixed",
        TRUE ~ vaccine_type_primary
      )
    },
    source = collapse_semicolon(source_name),
    source_url = collapse_semicolon(source_url),
    notes = collapse_semicolon(analysis_note),
    .groups = "drop"
  ) %>%
  mutate(
    year_start = if_else(is.infinite(year_start), NA_integer_, as.integer(year_start)),
    year_end = if_else(is.infinite(year_end), NA_integer_, as.integer(year_end)),
    vaccine_type_primary = case_when(
      str_detect(str_to_lower(coalesce(vaccine_type_primary, "")), "mixed") ~ "mixed",
      str_detect(str_to_lower(coalesce(vaccine_type_primary, "")), "ap") ~ "aP",
      str_detect(str_to_lower(coalesce(vaccine_type_primary, "")), "wp") ~ "wP",
      TRUE ~ "unknown"
    ),
    maternal_program = coalesce(as.logical(maternal_program), FALSE),
    booster_text = coalesce(booster_ages, ""),
    adolescent_booster = str_detect(booster_text, regex("Y(10|11|12|13|14|15|16|17|18)|adolescent", ignore_case = TRUE)),
    primary_dose_count = str_count(coalesce(primary_schedule_months, ""), ";") +
      if_else(is.na(primary_schedule_months) | primary_schedule_months == "", 0L, 1L),
    primary_dose_count = pmin(primary_dose_count, 5L),
    booster_count = str_count(coalesce(booster_ages, ""), ";") +
      if_else(is.na(booster_ages) | booster_ages == "", 0L, 1L),
    booster_count = pmin(booster_count, 6L),
    vaccine_type_primary = factor(vaccine_type_primary, levels = c("wP", "mixed", "aP", "unknown"))
  )

write_csv(schedule, fd_path("figure_2_vaccine_schedule.csv"))

selection <- read_csv_checked(
  out_path("model2c_country_selection_audit.csv"),
  c(
    "iso3", "country", "recommended_tier", "tier_score", "post_to_baseline_rr",
    "latest_case_year", "latest_reported_cases", "selected_for_current_model2c",
    "eligible_with_current_local_high_resolution_data", "excluded_from_default_model2",
    "weak_resurgence_or_negative_control_candidate", "time_resolved_cases",
    "age_group_cases", "maternal_program", "adolescent_booster", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 2 Model 2C country selection")

metadata <- read_csv_checked(
  file.path(root, "data", "clean", "country_metadata.csv"),
  c(
    "iso3", "dtp1_latest_coverage", "dtp3_latest_coverage",
    "maternal_coverage_latest_percent", "maternal_coverage_year", "routine_dose_count",
    "population_age_available", "contact_matrix_available",
    "vaccine_schedule_available"
  )
) %>%
  select(
    iso3,
    dtp1_latest_coverage,
    dtp3_latest_coverage,
    maternal_coverage_latest_percent,
    maternal_coverage_year,
    routine_dose_count,
    population_age_available,
    contact_matrix_available,
    vaccine_schedule_available
  )

focus_rows <- selection %>%
  filter(
    selected_for_current_model2c |
      eligible_with_current_local_high_resolution_data |
      excluded_from_default_model2
  ) %>%
  left_join(pilot_order, by = "iso3") %>%
  left_join(metadata, by = "iso3") %>%
  left_join(
    schedule %>%
      transmute(
        iso3,
        vaccine_type_schedule = as.character(vaccine_type_primary),
        maternal_program_schedule = maternal_program,
        adolescent_booster_schedule = adolescent_booster,
        primary_dose_count,
        booster_count,
        primary_schedule_months,
        booster_ages
      ),
    by = "iso3"
  ) %>%
  mutate(
    pilot_rank = case_when(
      selected_for_current_model2c ~ coalesce(pilot_rank, 90L),
      excluded_from_default_model2 ~ 98L,
      TRUE ~ 99L
    ),
    # Reuse the Figure 3a display-name contract so country labels and ordering
    # remain identical across the two main-text figures.
    country_label = country_display_name(country),
    inclusion_status = case_when(
      selected_for_current_model2c ~ "Default Bayesian pilot",
      excluded_from_default_model2 ~ "Sensitivity only",
      TRUE ~ "Evidence only"
    ),
    dtp1_percent = dtp1_latest_coverage * 100,
    dtp3_percent = dtp3_latest_coverage * 100,
    maternal_coverage_percent = maternal_coverage_latest_percent,
    maternal_coverage_year = as.integer(maternal_coverage_year),
    maternal_program_context = coalesce(maternal_program_schedule, FALSE) | coalesce(maternal_program, FALSE),
    routine_dose_count = coalesce(routine_dose_count, as.numeric(primary_dose_count)),
    booster_count = coalesce(as.numeric(booster_count), 0)
  ) %>%
  arrange(pilot_rank, country)

country_levels <- rev(focus_rows$country_label)
coverage_plot_years <- 2010:2025

inclusion_palette <- c(
  "Default Bayesian pilot" = clr("blue"),
  "Sensitivity only" = clr("orange"),
  "Evidence only" = clr("light_grey")
)

focus_lookup <- focus_rows %>%
  transmute(
    iso3,
    country_label = factor(country_label, levels = country_levels),
    inclusion_status = factor(
      inclusion_status,
      levels = c("Default Bayesian pilot", "Sensitivity only", "Evidence only")
    )
  )

write_csv(
  official_strategy %>% filter(iso3 %in% focus_rows$iso3),
  fd_path("figure_2_official_schedule_strategy.csv")
)

coverage_yearly <- read_csv_checked(
  file.path(root, "data", "clean", "vaccine_coverage.csv"),
  c("iso3", "year", "antigen", "coverage", "coverage_type")
) %>%
  filter(
    iso3 %in% focus_rows$iso3,
    antigen %in% c("DTPCV1", "DTPCV3"),
    !is.na(coverage),
    coverage_type == primary_dtp_coverage_type
  ) %>%
  mutate(
    year = as.integer(year),
    coverage_percent = pmin(100, pmax(0, as.numeric(coverage) * 100)),
    antigen = recode(antigen, DTPCV1 = "DTP1", DTPCV3 = "DTP3"),
    coverage_source = "WUENIC"
  ) %>%
  arrange(iso3, antigen, year) %>%
  group_by(iso3, antigen, year) %>%
  slice(1) %>%
  ungroup() %>%
  left_join(focus_lookup, by = "iso3") %>%
  filter(year >= 2010, year <= 2025) %>%
  mutate(antigen = factor(antigen, levels = c("DTP1", "DTP3")))

coverage_check <- coverage_yearly %>%
  count(iso3, antigen) %>%
  complete(iso3 = focus_rows$iso3, antigen = factor(c("DTP1", "DTP3"), levels = c("DTP1", "DTP3")), fill = list(n = 0))
if (any(coverage_check$n == 0)) {
  missing_pairs <- coverage_check %>%
    filter(n == 0) %>%
    mutate(pair = paste(iso3, antigen, sep = ":")) %>%
    pull(pair)
  stop("Figure 2 coverage trajectories are missing required country-antigen series: ", paste(missing_pairs, collapse = ", "), call. = FALSE)
}

coverage_grid <- expand_grid(
  iso3 = focus_rows$iso3,
  antigen = factor(c("DTP1", "DTP3"), levels = c("DTP1", "DTP3")),
  year = coverage_plot_years
) %>%
  left_join(focus_lookup, by = "iso3") %>%
  left_join(
    coverage_yearly %>% select(iso3, antigen, year, coverage_percent, coverage_source),
    by = c("iso3", "antigen", "year")
  ) %>%
  mutate(
    country_label = factor(country_label, levels = country_levels),
    country_facet = factor(as.character(country_label), levels = focus_rows$country_label)
  ) %>%
  arrange(iso3, antigen, year) %>%
  group_by(iso3, antigen) %>%
  mutate(
    selected_coverage_percent = coverage_percent,
    selected_coverage_source = coverage_source
  ) %>%
  fill(coverage_percent, coverage_source, .direction = "down") %>%
  mutate(
    coverage_status = case_when(
      !is.na(selected_coverage_percent) ~ "Selected WUENIC row",
      !is.na(coverage_percent) ~ "Carry-forward",
      TRUE ~ "Missing"
    ),
    coverage_point_source = case_when(
      coverage_status == "Selected WUENIC row" ~ primary_dtp_coverage_source_label,
      coverage_status == "Carry-forward" ~ primary_dtp_carry_forward_label,
      TRUE ~ NA_character_
    )
  ) %>%
  ungroup() %>%
  mutate(
    coverage_status = factor(coverage_status, levels = c("Selected WUENIC row", "Carry-forward", "Missing")),
    coverage_point_source = factor(
      coverage_point_source,
      levels = c(primary_dtp_coverage_source_label, primary_dtp_carry_forward_label)
    )
  )

coverage_facet_band <- expand_grid(
  country_facet = factor(focus_rows$country_label, levels = focus_rows$country_label),
  xmin = 2019.5,
  xmax = 2022.5,
  ymin = -Inf,
  ymax = Inf
)

coverage_y_min <- 60
coverage_y_breaks <- seq(60, 100, 10)
coverage_dose_palette <- c(DTP1 = clr("blue"), DTP3 = clr("green"))
coverage_point_shapes <- c(16, 1)
names(coverage_point_shapes) <- c(primary_dtp_coverage_source_label, primary_dtp_carry_forward_label)
coverage_line_data <- coverage_grid %>%
  filter(!is.na(coverage_percent))
coverage_selected_source_points <- coverage_line_data %>%
  filter(coverage_status == "Selected WUENIC row")
coverage_inferred_points <- coverage_line_data %>%
  filter(coverage_status == "Carry-forward")

p_coverage_lines <- ggplot() +
  geom_rect(
    data = coverage_facet_band,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = clr("paper"),
    alpha = 0.95
  ) +
  geom_hline(yintercept = coverage_y_breaks, colour = pal["light_grey"], linewidth = 0.16) +
  geom_line(
    data = coverage_line_data,
    aes(year, coverage_percent, colour = antigen, linewidth = antigen, group = antigen),
    alpha = 0.70,
    lineend = "round",
    linetype = "solid"
  ) +
  geom_point(
    data = coverage_selected_source_points,
    aes(year, coverage_percent, colour = antigen, shape = coverage_point_source, size = antigen),
    alpha = 0.72,
    stroke = 0.20
  ) +
  geom_point(
    data = coverage_inferred_points,
    aes(year, coverage_percent, colour = antigen, shape = coverage_point_source, size = antigen),
    alpha = 0.72,
    stroke = 0.38
  ) +
  facet_wrap(~country_facet, nrow = 2) +
  scale_colour_manual(values = coverage_dose_palette, name = "Dose", drop = FALSE) +
  scale_shape_manual(
    values = coverage_point_shapes,
    breaks = names(coverage_point_shapes),
    name = "Coverage source",
    drop = FALSE
  ) +
  scale_linewidth_manual(values = c(DTP1 = 0.54, DTP3 = 0.38), guide = "none") +
  scale_size_manual(values = c(DTP1 = 1.00, DTP3 = 0.76), guide = "none") +
  scale_x_continuous(
    breaks = figure_breaks_year(2010, 2025, 5),
    expand = expansion(mult = 0, add = 0.72)
  ) +
  scale_y_continuous(
    breaks = coverage_y_breaks,
    labels = label_number(accuracy = 1),
    expand = expansion(mult = 0, add = 0)
  ) +
  coord_cartesian(ylim = c(coverage_y_min, 100), clip = "off") +
  labs(x = NULL, y = "Coverage (%)") +
  guides(
    colour = guide_legend(
      order = 1,
      override.aes = list(linewidth = 0.68, size = 1.6, alpha = 0.72, linetype = "solid")
    ),
    shape = guide_legend(
      order = 2,
      override.aes = list(colour = pal["dark_grey"], size = 1.7, alpha = 0.72, stroke = 0.42)
    )
  ) +
  theme_manuscript() +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text_dense),
    panel.spacing.x = unit(3.0, "mm"),
    panel.spacing.y = unit(1.2, "mm"),
    strip.text = element_text(size = figure_text$strip_dense, face = "bold"),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.direction = "horizontal",
    legend.key.size = unit(2.4, "mm"),
    legend.spacing.x = unit(1.2, "mm"),
    legend.spacing.y = unit(0, "mm"),
    legend.margin = margin(0, 0, 0, 0)
  )

dtp3_change <- coverage_yearly %>%
  filter(antigen == "DTP3") %>%
  group_by(iso3, country_label, inclusion_status) %>%
  summarise(
    baseline_year = {
      candidate <- year[year <= 2019 & !is.na(coverage_percent)]
      if (length(candidate) == 0) NA_integer_ else max(candidate)
    },
    latest_year = max(year, na.rm = TRUE),
    baseline = coverage_percent[year == baseline_year][1],
    latest = coverage_percent[year == latest_year][1],
    .groups = "drop"
  ) %>%
  mutate(
    change_pp = latest - baseline,
    change_direction = case_when(
      is.na(change_pp) ~ "Missing",
      change_pp <= -2 ~ "Lower",
      change_pp >= 2 ~ "Higher",
      TRUE ~ "Stable"
    ),
    label = if_else(
      is.na(change_pp),
      "NA",
      paste0(if_else(change_pp > 0, "+", ""), number(change_pp, accuracy = 0.1), " pp")
    )
  )

dtp3_nadir <- coverage_yearly %>%
  filter(antigen == "DTP3", year >= 2020, year <= 2024) %>%
  left_join(
    dtp3_change %>% select(iso3, baseline, baseline_year),
    by = "iso3"
  ) %>%
  group_by(iso3, country_label, inclusion_status) %>%
  arrange(coverage_percent, year, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    nadir_shortfall_pp = coverage_percent - baseline,
    nadir_status = case_when(
      is.na(nadir_shortfall_pp) ~ "Missing",
      nadir_shortfall_pp <= -2 ~ "Shortfall",
      nadir_shortfall_pp >= 2 ~ "Higher",
      TRUE ~ "Stable"
    ),
    label = if_else(
      is.na(nadir_shortfall_pp),
      "NA",
      paste0(
        if_else(nadir_shortfall_pp > 0, "+", ""),
        number(nadir_shortfall_pp, accuracy = 0.1),
        " (",
        year,
        ")"
      )
    ),
    label_x = 0.45,
    label_hjust = 0
  )

nadir_abs <- max(abs(dtp3_nadir$nadir_shortfall_pp), na.rm = TRUE)
nadir_limit <- if_else(is.finite(nadir_abs), ceiling(nadir_abs + 2), 5)

p_dtp3_nadir <- ggplot(dtp3_nadir, aes(nadir_shortfall_pp, country_label)) +
  geom_vline(xintercept = 0, colour = pal["dark_grey"], linewidth = 0.28) +
  geom_segment(
    aes(x = 0, xend = nadir_shortfall_pp, yend = country_label),
    colour = pal["light_grey"],
    linewidth = 0.58
  ) +
  geom_point(aes(fill = nadir_status), shape = 21, size = 2.2, colour = pal["white"], stroke = 0.34) +
  geom_text(
    aes(x = label_x, label = label, hjust = label_hjust),
    size = figure_text$annotation,
    colour = pal["black"]
  ) +
  scale_fill_manual(
    values = c(Shortfall = clr("red"), Stable = clr("light_grey"), Higher = clr("green"), Missing = clr("light_grey")),
    guide = "none"
  ) +
  scale_y_discrete(drop = FALSE, limits = country_levels) +
  scale_x_continuous(breaks = figure_breaks_linear(4)) +
  coord_cartesian(xlim = c(-nadir_limit, nadir_limit), clip = "off") +
  labs(x = "Lowest 2020-2024 DTP3 vs baseline (pp)", y = NULL) +
  theme_manuscript(grid = "x")

maternal_plot <- focus_rows %>%
  transmute(
    iso3,
    country_label = factor(country_label, levels = country_levels),
    maternal_program_context,
    maternal_coverage_percent = pmin(100, pmax(0, as.numeric(maternal_coverage_percent))),
    maternal_coverage_year
  ) %>%
  mutate(
    maternal_value = if_else(is.na(maternal_coverage_percent), 0, maternal_coverage_percent),
    maternal_status = case_when(
      !maternal_program_context ~ "No programme",
      is.na(maternal_coverage_percent) ~ "No uptake series",
      maternal_value == 0 ~ "Zero or unreported",
      TRUE ~ "Measured uptake"
    ),
    label = case_when(
      !maternal_program_context ~ "No programme",
      is.na(maternal_coverage_percent) ~ "No uptake data",
      TRUE ~ paste0(
        number(maternal_value, accuracy = 1),
        "%",
        if_else(is.na(maternal_coverage_year), "", paste0(" (", maternal_coverage_year, ")"))
      )
    )
  )

p_maternal <- ggplot(maternal_plot, aes(maternal_value, country_label)) +
  geom_col(aes(fill = maternal_status), width = 0.68) +
  geom_text(
    aes(x = pmin(maternal_value + 2.5, 98), label = label),
    hjust = 0,
    size = figure_text$annotation,
    colour = pal["black"]
  ) +
  scale_fill_manual(
    values = c(
      "Measured uptake" = clr("green"),
      "Zero or unreported" = clr("orange"),
      "No uptake series" = clr("light_grey"),
      "No programme" = clr("paper")
    ),
    guide = "none"
  ) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = figure_breaks$percent_coarse,
    expand = expansion(mult = c(0, 0.08))
  ) +
  scale_y_discrete(drop = FALSE, limits = country_levels) +
  labs(x = "Latest maternal pertussis uptake (%)", y = NULL) +
  theme_manuscript(grid = "x") +
  theme(
    legend.position = "bottom"
  )

parse_schedule_age_year <- function(token) {
  token <- str_trim(coalesce(token, ""))
  if (token == "" || str_detect(token, regex("contact", ignore_case = TRUE))) {
    return(NA_real_)
  }
  token <- str_remove(token, "^\\+")
  token <- str_remove(token, "^>")
  unit <- str_sub(token, 1, 1)
  values <- str_extract_all(token, "\\d+(?:\\.\\d+)?")[[1]]
  if (length(values) == 0) {
    return(NA_real_)
  }
  value <- mean(as.numeric(values), na.rm = TRUE)
  if (!is.finite(value)) {
    return(NA_real_)
  }
  case_when(
    unit == "W" ~ value / 52,
    unit == "M" ~ value / 12,
    unit == "Y" ~ value,
    TRUE ~ NA_real_
  )
}

parse_gestation_week_window <- function(text) {
  text <- str_squish(str_to_lower(coalesce(text, "")))
  if (!str_detect(text, "week|wk|gestation|pregnan|maternal|trimester|antenatal")) {
    return(c(NA_real_, NA_real_))
  }

  numbers <- as.numeric(str_extract_all(text, "\\d+(?:\\.\\d+)?")[[1]])
  numbers <- numbers[numbers >= 4 & numbers <= 45]
  if (length(numbers) >= 2) {
    if (numbers[[2]] < numbers[[1]]) {
      return(c(numbers[[1]], 40))
    }
    return(c(numbers[[1]], numbers[[2]]))
  }
  if (length(numbers) == 1 && str_detect(text, "from|after|>=|at least")) {
    return(c(numbers[[1]], 40))
  }
  if (length(numbers) == 1) {
    return(c(numbers[[1]], numbers[[1]]))
  }
  c(NA_real_, NA_real_)
}

gestation_week_to_plot <- function(week) {
  week <- pmin(40, pmax(8, week))
  -0.34 + ((week - 8) / 32) * 0.24
}

pregnancy_marker_age <- gestation_week_to_plot(28)

strategy_country_y <- tibble(
  country_label = factor(country_levels, levels = country_levels),
  country_y = seq_along(country_levels)
)

vaccine_path_palette <- c(
  "DTaP" = clr("blue"),
  "DTwP" = clr("green"),
  "Tdap" = clr("purple"),
  "aP" = clr("gold")
)

target_shapes <- c(
  "Routine" = 21,
  "Pregnancy" = 23,
  "Adult" = 24,
  "Risk group" = 22,
  "Target/trigger" = 25,
  "Special/conditional" = 10
)

maternal_vaccine_source <- focus_rows %>%
  select(iso3) %>%
  left_join(
    official_strategy %>%
      filter(target_group == "Pregnancy" | target_pop == "PW") %>%
      transmute(
        iso3,
        maternal_vaccine_label = na_if(maternal_vaccine_label, ""),
        maternal_vaccine_family = na_if(maternal_vaccine_family, ""),
        official_gestation_window = na_if(official_gestation_window, ""),
        maternal_vaccine_source = na_if(source_name, ""),
        maternal_vaccine_url = na_if(source_url, ""),
        maternal_vaccine_note = na_if(source_comment, "")
      ) %>%
      distinct(iso3, .keep_all = TRUE),
    by = "iso3"
  ) %>%
  mutate(
    maternal_vaccine_note = coalesce(
      maternal_vaccine_note,
      "No routine pregnancy pertussis programme marker in the official schedule strategy table."
    )
  )

write_csv(maternal_vaccine_source, fd_path("figure_2_maternal_vaccine_sources.csv"))

raw_schedule <- official_strategy %>%
  mutate(
    iso3 = str_to_upper(as.character(iso3)),
    schedule_year = suppressWarnings(as.integer(schedule_year)),
    vaccine_code = str_to_upper(as.character(vaccine_code)),
    vaccine_description = as.character(vaccine_description),
    target_pop = str_to_upper(coalesce(as.character(target_pop), "")),
    target_description = coalesce(as.character(target_description), as.character(target_group), ""),
    token = str_trim(coalesce(as.character(token), "")),
    source_comment = coalesce(as.character(source_comment), ""),
    target_text = str_squish(paste(target_pop, target_description, source_comment)),
    pertussis_product_text = str_to_upper(paste(vaccine_code, vaccine_description)),
    official_vaccine_family = coalesce(as.character(vaccine_family), ""),
    official_target_group = coalesce(as.character(target_group), ""),
    official_event_window_label = coalesce(as.character(event_window_label), ""),
    special_conditional_route_source = coalesce(as.logical(special_conditional_route), FALSE),
    catch_up_route_source = coalesce(as.logical(catch_up_route), FALSE)
  ) %>%
  filter(
    iso3 %in% focus_rows$iso3
  )

schedule_product_events <- raw_schedule %>%
  left_join(focus_lookup %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    country_label = factor(country_label, levels = country_levels),
    age_year = vapply(token, parse_schedule_age_year, numeric(1)),
    relative_interval = str_detect(token, "^\\+"),
    target_group = case_when(
      official_target_group %in% names(target_shapes) ~ official_target_group,
      target_pop == "PW" ~ "Pregnancy",
      target_pop == "ADULTS" ~ "Adult",
      target_pop == "RISKGROUPS" ~ "Risk group",
      str_detect(target_description, regex("pregnan|maternal", ignore_case = TRUE)) ~ "Pregnancy",
      str_detect(target_description, regex("\\badult", ignore_case = TRUE)) ~ "Adult",
      str_detect(target_description, regex("risk", ignore_case = TRUE)) ~ "Risk group",
      str_detect(source_comment, regex("risk|adverse|hsct|chemotherapy|patholog|immuno", ignore_case = TRUE)) ~ "Risk group",
      str_detect(token, regex("contact", ignore_case = TRUE)) ~ "Target/trigger",
      TRUE ~ "Routine"
    ),
    target_group = factor(target_group, levels = names(target_shapes)),
    vaccine_family = case_when(
      official_vaccine_family %in% names(vaccine_path_palette) ~ official_vaccine_family,
      vaccine_code == "AP" ~ "aP",
      str_detect(vaccine_code, "TDAP") ~ "Tdap",
      str_detect(vaccine_code, "DTWP|DTPW") ~ "DTwP",
      str_detect(vaccine_code, "DTAP") ~ "DTaP",
      TRUE ~ NA_character_
    ),
    vaccine_family = factor(vaccine_family, levels = names(vaccine_path_palette)),
    vaccine_path_label = if_else(
      as.character(target_group) == "Routine",
      vaccine_code,
      paste(vaccine_code, as.character(target_group), sep = " / ")
    ),
    gestation_start_week = vapply(target_text, function(x) parse_gestation_week_window(x)[[1]], numeric(1)),
    gestation_end_week = vapply(target_text, function(x) parse_gestation_week_window(x)[[2]], numeric(1)),
    gestation_point_week = case_when(
      !is.na(gestation_start_week) & !is.na(gestation_end_week) ~ (gestation_start_week + gestation_end_week) / 2,
      TRUE ~ NA_real_
    ),
    event_start_age = case_when(
      target_group == "Pregnancy" ~ pregnancy_marker_age,
      target_group == "Adult" & relative_interval ~ 40,
      target_group == "Adult" & is.na(age_year) ~ 21,
      target_group == "Risk group" & is.na(age_year) ~ 72,
      target_group == "Target/trigger" & is.na(age_year) ~ 72,
      relative_interval ~ 40,
      !is.na(age_year) ~ age_year,
      TRUE ~ NA_real_
    ),
    event_end_age = case_when(
      target_group == "Pregnancy" ~ pregnancy_marker_age,
      target_group == "Adult" & relative_interval ~ 40,
      target_group == "Adult" & is.na(age_year) ~ 21,
      target_group == "Risk group" & is.na(age_year) ~ 72,
      target_group == "Target/trigger" & is.na(age_year) ~ 72,
      relative_interval ~ 40,
      !is.na(age_year) ~ age_year,
      TRUE ~ NA_real_
    ),
    plot_age_year = (event_start_age + event_end_age) / 2,
    event_is_window = FALSE,
    catch_up_route = str_detect(
      target_text,
      regex("catch|unvaccinated|incomplete|missed|not fully immun", ignore_case = TRUE)
    ) | catch_up_route_source,
    special_conditional_route = special_conditional_route_source |
      target_group %in% c("Risk group", "Target/trigger") |
      str_detect(
        target_text,
        regex(
          "special|risk|revaccination|post\\s*hsct|chemotherapy|adverse|meningococcal|immunosupp|transplant|splenectomy",
          ignore_case = TRUE
        )
      ),
    route_annotation_label = case_when(
      catch_up_route ~ "Catch-up/incomplete",
      special_conditional_route ~ "Special/conditional",
      TRUE ~ NA_character_
    ),
    event_window_label = case_when(
      official_event_window_label != "" ~ official_event_window_label,
      target_group == "Pregnancy" & !is.na(gestation_start_week) & !is.na(gestation_end_week) ~ paste0(
        "gestation ",
        number(gestation_start_week, accuracy = 1),
        "-",
        number(gestation_end_week, accuracy = 1),
        "w"
      ),
      target_group == "Pregnancy" ~ "pregnancy window",
      target_group == "Adult" & relative_interval ~ "adult recurring interval",
      target_group == "Adult" & is.na(age_year) ~ "adult window",
      target_group %in% c("Risk group", "Target/trigger") & is.na(age_year) ~ "target-trigger window",
      TRUE ~ token
    )
  ) %>%
  filter(!is.na(plot_age_year)) %>%
  left_join(strategy_country_y, by = "country_label") %>%
  arrange(country_label, plot_age_year, vaccine_family, vaccine_path_label, token)

schedule_product_events <- schedule_product_events %>%
  left_join(maternal_vaccine_source, by = "iso3", suffix = c("", "_lookup")) %>%
  mutate(
    maternal_vaccine_label = coalesce(na_if(maternal_vaccine_label, ""), maternal_vaccine_label_lookup),
    maternal_vaccine_family = coalesce(na_if(maternal_vaccine_family, ""), maternal_vaccine_family_lookup),
    official_gestation_window = coalesce(na_if(official_gestation_window, ""), official_gestation_window_lookup),
    maternal_vaccine_source = coalesce(na_if(source_name, ""), maternal_vaccine_source),
    maternal_vaccine_url = coalesce(na_if(source_url, ""), maternal_vaccine_url),
    maternal_vaccine_note = coalesce(na_if(source_comment, ""), maternal_vaccine_note)
  ) %>%
  select(-ends_with("_lookup")) %>%
  arrange(country_label, plot_age_year, vaccine_family, vaccine_path_label, token)

is_adolescent_booster <- function(target_pop, target_group, age) {
  target_pop == "B_ADO_W" | (target_group == "Routine" & !is.na(age) & age >= 9 & age < 18)
}

is_child_booster <- function(target_pop, target_group, age) {
  target_pop %in% c("B_2YL_W", "B_2YL_F", "B_CHILD_W") |
    (target_group == "Routine" & !is.na(age) & age >= 1 & age < 9)
}

strategy_stage_levels <- c(
  "Pregnancy",
  "Infant primary path",
  "Special/conditional route",
  "Child booster",
  "Adolescent booster",
  "Adult booster",
  "Risk-group booster",
  "Other"
)

strategy_events <- schedule_product_events %>%
  mutate(
    is_adolescent_booster = is_adolescent_booster(target_pop, as.character(target_group), plot_age_year),
    is_child_booster = is_child_booster(target_pop, as.character(target_group), plot_age_year),
    strategy_stage = case_when(
      target_group == "Pregnancy" ~ "Pregnancy",
      target_group == "Adult" ~ "Adult booster",
      target_group == "Risk group" & plot_age_year > 6 ~ "Risk-group booster",
      target_group %in% c("Risk group", "Target/trigger") ~ "Special/conditional route",
      is_adolescent_booster ~ "Adolescent booster",
      is_child_booster ~ "Child booster",
      target_group == "Routine" ~ "Infant primary path",
      TRUE ~ "Other"
    ),
    strategy_stage = factor(strategy_stage, levels = strategy_stage_levels),
    maternal_vaccine_label_display = if_else(
      target_group == "Pregnancy",
      coalesce(maternal_vaccine_label, as.character(vaccine_family)),
      NA_character_
    )
  )

collapse_unique <- function(x) {
  x <- unique(na.omit(as.character(x)))
  if (length(x) == 0) {
    return(NA_character_)
  }
  paste(x, collapse = "; ")
}

strategy_plot_events <- strategy_events %>%
  mutate(
    display_product_alternative = !is.na(vaccine_family) &
      target_group == "Routine" &
      strategy_stage %in% c("Infant primary path", "Child booster"),
    display_vaccine_family = as.character(vaccine_family),
    display_vaccine_path_label = if_else(
      display_product_alternative,
      paste(as.character(vaccine_family), "platform"),
      vaccine_path_label
    ),
    display_note = if_else(
      display_product_alternative,
      "Product alternatives within the same pertussis-family schedule collapsed for display",
      NA_character_
    )
  ) %>%
  group_by(
    iso3,
    country_label,
    country_y,
    schedule_year,
    target_pop,
    target_group,
    strategy_stage,
    plot_age_year,
    event_window_label,
    route_annotation_label,
    catch_up_route,
    special_conditional_route,
    display_vaccine_path_label,
    display_vaccine_family
  ) %>%
  summarise(
    vaccine_code = collapse_unique(vaccine_code),
    vaccine_description = collapse_unique(vaccine_description),
    target_description = collapse_unique(target_description),
    token = collapse_unique(token),
    source_comment = collapse_unique(source_comment),
    target_text = collapse_unique(target_text),
    age_year = first(age_year),
    relative_interval = any(relative_interval, na.rm = TRUE),
    gestation_start_week = first(gestation_start_week),
    gestation_end_week = first(gestation_end_week),
    gestation_point_week = first(gestation_point_week),
    event_start_age = first(event_start_age),
    event_end_age = first(event_end_age),
    event_is_window = any(event_is_window, na.rm = TRUE),
    vaccine_path_label = first(display_vaccine_path_label),
    vaccine_family = factor(first(display_vaccine_family), levels = names(vaccine_path_palette)),
    maternal_vaccine_label_display = collapse_unique(maternal_vaccine_label_display),
    maternal_vaccine_source = collapse_unique(maternal_vaccine_source),
    maternal_vaccine_url = collapse_unique(maternal_vaccine_url),
    display_note = collapse_unique(display_note),
    .groups = "drop"
  ) %>%
  arrange(country_label, plot_age_year, vaccine_family, vaccine_path_label, token)

primary_path_events <- strategy_plot_events %>%
  filter(strategy_stage == "Infant primary path") %>%
  group_by(country_label, vaccine_path_label, vaccine_family) %>%
  mutate(path_dose_count = n()) %>%
  ungroup()

primary_path_order <- primary_path_events %>%
  group_by(country_label, country_y, vaccine_path_label, vaccine_family) %>%
  summarise(
    path_start = min(plot_age_year, na.rm = TRUE),
    path_end = max(plot_age_year, na.rm = TRUE),
    dose_count = n(),
    .groups = "drop"
  ) %>%
  group_by(country_label) %>%
  arrange(desc(dose_count), path_start, path_end, vaccine_family, vaccine_path_label, .by_group = TRUE) %>%
  mutate(
    path_n = n(),
    path_index = row_number(),
    path_span = case_when(
      path_n <= 1L ~ 0,
      path_n == 2L ~ 0.30,
      path_n == 3L ~ 0.42,
      TRUE ~ 0.64
    ),
    path_offset = if_else(path_n == 1L, 0, ((path_index - (path_n + 1) / 2) / pmax(path_n - 1L, 1L)) * path_span)
  ) %>%
  ungroup()

primary_path_events <- primary_path_events %>%
  left_join(
    primary_path_order %>% select(country_label, vaccine_path_label, vaccine_family, path_offset, path_n),
    by = c("country_label", "vaccine_path_label", "vaccine_family")
  ) %>%
  mutate(path_y = country_y + coalesce(path_offset, 0))

primary_path_range <- primary_path_events %>%
  group_by(country_label, country_y, vaccine_path_label, vaccine_family, path_y) %>%
  summarise(
    path_start = min(plot_age_year, na.rm = TRUE),
    path_end = max(plot_age_year, na.rm = TRUE),
    .groups = "drop"
  )

strategy_flow <- strategy_country_y %>%
  left_join(
    primary_path_range %>%
      group_by(country_label, country_y) %>%
      summarise(
        primary_start = min(path_start, na.rm = TRUE),
        primary_end = max(path_end, na.rm = TRUE),
        .groups = "drop"
      ),
    by = c("country_label", "country_y")
  ) %>%
  mutate(
    branch_start_age = 0,
    branch_end_age = pmax(coalesce(primary_end, 0.5) + 0.35, 0.85)
  )

pregnancy_points <- strategy_plot_events %>%
  filter(strategy_stage == "Pregnancy") %>%
  group_by(country_label) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(path_y = country_y)

booster_points <- strategy_plot_events %>%
  filter(
    strategy_stage %in% c("Child booster", "Adolescent booster", "Adult booster", "Risk-group booster"),
    !special_conditional_route
  ) %>%
  mutate(path_y = country_y)

conditional_points <- strategy_plot_events %>%
  filter(!is.na(route_annotation_label)) %>%
  group_by(country_label, plot_age_year) %>%
  arrange(vaccine_family, vaccine_path_label, .by_group = TRUE) %>%
  mutate(
    conditional_n_at_age = n(),
    conditional_index = row_number(),
    conditional_y_offset = case_when(
      strategy_stage == "Special/conditional route" ~ -0.24,
      plot_age_year >= 60 & conditional_n_at_age > 1 ~
        (conditional_index - (conditional_n_at_age + 1) / 2) * 0.28,
      TRUE ~ 0.22
    ),
    path_y = country_y + conditional_y_offset,
    plot_age_display = case_when(
      plot_age_year >= 60 & conditional_n_at_age > 1 ~
        plot_age_year + (conditional_index - (conditional_n_at_age + 1) / 2) * 1.65,
      TRUE ~ plot_age_year
    ),
    route_annotation_label = factor(route_annotation_label, levels = c("Special/conditional", "Catch-up/incomplete"))
  ) %>%
  ungroup()

conditional_point_connectors <- conditional_points %>%
  transmute(
    x = plot_age_display,
    xend = plot_age_display,
    y = country_y,
    yend = path_y
  )

booster_flow <- booster_points %>%
  group_by(country_label, country_y) %>%
  summarise(booster_end_age = max(plot_age_year, na.rm = TRUE), .groups = "drop")

strategy_flow <- strategy_flow %>%
  left_join(booster_flow, by = c("country_label", "country_y")) %>%
  mutate(flow_end_age = pmax(branch_end_age, coalesce(booster_end_age, branch_end_age), na.rm = TRUE))

primary_path_range <- primary_path_range %>%
  left_join(strategy_flow %>% select(country_label, branch_start_age, branch_end_age), by = "country_label")

primary_entry_segments <- primary_path_range %>%
  transmute(
    connector_id = paste(country_label, vaccine_path_label, "entry", sep = "_"),
    vaccine_family,
    x = branch_start_age,
    xend = path_start,
    y = country_y,
    yend = path_y
  )

primary_exit_segments <- primary_path_range %>%
  transmute(
    connector_id = paste(country_label, vaccine_path_label, "exit", sep = "_"),
    vaccine_family,
    x = path_end,
    xend = branch_end_age,
    y = path_y,
    yend = country_y
  )

make_flow_curve <- function(df, n_points = 16L) {
  if (nrow(df) == 0) {
    return(tibble(connector_id = character(), vaccine_family = factor(levels = names(vaccine_path_palette)), x = numeric(), y = numeric()))
  }
  bind_rows(lapply(seq_len(nrow(df)), function(i) {
    row <- df[i, ]
    t <- seq(0, 1, length.out = n_points)
    dx <- row$xend - row$x
    tibble(
      connector_id = row$connector_id,
      vaccine_family = row$vaccine_family,
      x = ((1 - t)^3 * row$x) +
        (3 * (1 - t)^2 * t * (row$x + dx * 0.28)) +
        (3 * (1 - t) * t^2 * (row$x + dx * 0.72)) +
        (t^3 * row$xend),
      y = ((1 - t)^3 * row$y) +
        (3 * (1 - t)^2 * t * row$y) +
        (3 * (1 - t) * t^2 * row$yend) +
        (t^3 * row$yend)
    )
  }))
}

primary_connector_curves <- bind_rows(
  make_flow_curve(primary_entry_segments),
  make_flow_curve(primary_exit_segments)
)

schedule_strategy_audit <- strategy_events %>%
  group_by(iso3, country_label, strategy_stage) %>%
  summarise(
    vaccine_codes = collapse_unique(vaccine_path_label),
    vaccine_families = collapse_unique(vaccine_family),
    timings = collapse_unique(event_window_label),
    special_or_conditional_route = collapse_unique(if_else(special_conditional_route, vaccine_path_label, NA_character_)),
    catch_up_or_incomplete_route = collapse_unique(if_else(catch_up_route, vaccine_path_label, NA_character_)),
    pregnancy_vaccine = collapse_unique(maternal_vaccine_label_display),
    pregnancy_source = collapse_unique(if_else(target_group == "Pregnancy", maternal_vaccine_source, NA_character_)),
    pregnancy_url = collapse_unique(if_else(target_group == "Pregnancy", maternal_vaccine_url, NA_character_)),
    source_names = collapse_unique(source_name),
    source_urls = collapse_unique(source_url),
    analysis_notes = collapse_unique(analysis_note),
    .groups = "drop"
  ) %>%
  arrange(country_label, strategy_stage)

write_csv(schedule_strategy_audit, fd_path("figure_2_schedule_strategy_audit.csv"))

maternal_to_primary_segments <- pregnancy_points %>%
  left_join(strategy_flow %>% select(country_label, branch_start_age), by = "country_label") %>%
  transmute(
    x = plot_age_year,
    xend = branch_start_age,
    y = country_y,
    yend = country_y
  )

post_primary_segments <- strategy_flow %>%
  filter(flow_end_age > branch_end_age) %>%
  transmute(
    x = branch_end_age,
    xend = flow_end_age,
    y = country_y,
    yend = country_y
  )

strategy_country_bands <- strategy_country_y %>%
  mutate(
    ymin = country_y - 0.50,
    ymax = country_y + 0.50,
    show_band = row_number() %% 2 == 1
  ) %>%
  filter(show_band)

p_schedule <- ggplot() +
  geom_rect(
    data = strategy_country_bands,
    aes(xmin = -0.35, xmax = 75, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = clr("paper"),
    alpha = 0.55
  ) +
  geom_segment(
    data = maternal_to_primary_segments,
    aes(x = x, xend = xend, y = y, yend = yend),
    colour = pal["light_grey"],
    linewidth = 0.34,
    lineend = "round"
  ) +
  geom_path(
    data = primary_connector_curves,
    aes(x, y, group = connector_id, colour = vaccine_family),
    linewidth = 0.32,
    alpha = 0.50,
    lineend = "round"
  ) +
  geom_segment(
    data = primary_path_range,
    aes(x = path_start, xend = path_end, y = path_y, yend = path_y, colour = vaccine_family),
    linewidth = 0.52,
    alpha = 0.82,
    lineend = "round"
  ) +
  geom_segment(
    data = post_primary_segments,
    aes(x = x, xend = xend, y = y, yend = yend),
    colour = pal["light_grey"],
    linewidth = 0.34,
    lineend = "round"
  ) +
  geom_segment(
    data = conditional_point_connectors,
    aes(x = x, xend = xend, y = y, yend = yend),
    inherit.aes = FALSE,
    colour = pal["light_grey"],
    linewidth = 0.30,
    alpha = 0.95,
    lineend = "round"
  ) +
  geom_point(
    data = primary_path_events,
    aes(plot_age_year, path_y, fill = vaccine_family, shape = target_group),
    size = 1.78,
    colour = "white",
    stroke = 0.24
  ) +
  geom_point(
    data = booster_points,
    aes(plot_age_year, path_y, fill = vaccine_family, shape = target_group),
    size = 1.88,
    colour = "white",
    stroke = 0.26
  ) +
  geom_point(
    data = conditional_points,
    aes(plot_age_display, path_y, fill = vaccine_family, shape = target_group),
    size = 1.72,
    colour = pal["black"],
    stroke = 0.26
  ) +
  geom_point(
    data = pregnancy_points,
    aes(plot_age_year, path_y, fill = vaccine_family, shape = target_group),
    size = 2.05,
    colour = pal["dark_grey"],
    stroke = 0.42
  ) +
  geom_point(
    data = conditional_points,
    aes(plot_age_display, path_y, shape = route_annotation_label),
    inherit.aes = FALSE,
    size = 2.45,
    colour = pal["black"],
    stroke = 0.42
  ) +
  scale_colour_manual(values = vaccine_path_palette, name = "Vaccine path", drop = TRUE) +
  scale_fill_manual(values = vaccine_path_palette, name = "Vaccine path", drop = TRUE) +
  scale_shape_manual(
    values = target_shapes,
    name = "Target/route",
    labels = function(x) recode(x, "Special/conditional" = "Conditional"),
    drop = TRUE
  ) +
  guides(
    colour = "none",
    fill = guide_legend(
      order = 2,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(shape = 21, size = 2.1, colour = "white")
    ),
    shape = guide_legend(
      order = 3,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(fill = pal["grey"], colour = pal["dark_grey"], size = 2.0)
    )
  ) +
  scale_x_continuous(
    trans = pseudo_log_trans(sigma = 0.08, base = 10),
    limits = c(-0.35, 75),
    breaks = c(gestation_week_to_plot(28), 0, 2 / 12, 4 / 12, 0.5, 1, 5, 12, 18, 40, 68.5),
    labels = c("PW", "0", "2m", "4m", "6m", "1y", "5y", "12y", "18y", "40y", "65+/risk"),
    expand = expansion(mult = c(0.005, 0.03)),
    guide = guide_axis(n.dodge = 1)
  ) +
  scale_y_continuous(
    breaks = strategy_country_y$country_y,
    labels = as.character(strategy_country_y$country_label),
    expand = expansion(add = c(0.55, 0.55))
  ) +
  coord_cartesian(clip = "off") +
  labs(x = "Timing along maternal, childhood and booster strategy paths", y = NULL) +
  theme_manuscript(grid = "x") +
  theme(
    axis.text.x = element_text(size = figure_text$axis_text_dense),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.direction = "horizontal",
    legend.key.size = unit(2.0, "mm"),
    legend.spacing.x = unit(0.25, "mm"),
    legend.spacing.y = unit(2.2, "mm"),
    legend.key.spacing.x = unit(0.25, "mm"),
    legend.key.spacing.y = unit(0.9, "mm"),
    legend.text = element_text(size = figure_text$legend_text_dense - 0.2),
    legend.title = element_text(size = figure_text$legend_title - 0.2, face = "bold")
  )

figure_2_rendered_source <- bind_rows(
  coverage_grid %>%
    transmute(
      panel = "a_coverage_lines",
      iso3,
      country_label = as.character(country_label),
      metric = paste0(
        as.character(antigen),
        " annual coverage",
        if_else(coverage_status == "Carry-forward", " (carry-forward)", "")
      ),
      year = as.integer(year),
      value = coverage_percent,
      source = as.character(coverage_point_source),
      label = case_when(
        is.na(coverage_percent) ~ NA_character_,
        coverage_status == "Carry-forward" ~ paste0(number(coverage_percent, accuracy = 0.1), " cf"),
        TRUE ~ number(coverage_percent, accuracy = 0.1)
      )
    ),
  dtp3_nadir %>%
    transmute(
      panel = "b_dtp3_post2019_nadir",
      iso3,
      country_label = as.character(country_label),
      metric = "DTP3 lowest 2020-2024 value vs baseline, pp",
      year = as.integer(year),
      value = nadir_shortfall_pp,
      label
    ),
  maternal_plot %>%
    transmute(
      panel = "c_maternal_coverage",
      iso3,
      country_label = as.character(country_label),
      metric = "Latest maternal pertussis uptake",
      year = as.integer(maternal_coverage_year),
      value = maternal_coverage_percent,
      label
    ),
  strategy_plot_events %>%
    transmute(
      panel = "d_schedule_age_strategy",
      iso3,
      country_label = as.character(country_label),
      metric = paste(as.character(vaccine_family), strategy_stage, sep = " "),
      year = as.integer(schedule_year),
      value = plot_age_year,
      label = paste(
        strategy_stage,
        vaccine_path_label,
        as.character(target_group),
        event_window_label,
        coalesce(maternal_vaccine_label_display, ""),
        coalesce(route_annotation_label, ""),
        coalesce(display_note, ""),
        sep = " | "
      )
    )
)

write_csv(figure_2_rendered_source, fd_path("figure_2_programme_context_rendered.csv"))

figure_layout <- "
AA
BD
CD
"

figure <- panel_tags(
  free(p_coverage_lines) + p_dtp3_nadir + p_maternal + free(p_schedule) +
    plot_layout(
      design = figure_layout,
      heights = c(1.35, 0.82, 0.82),
      widths = c(0.92, 1.38),
      guides = "keep"
    )
) &
  theme(
    legend.direction = "horizontal",
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.spacing.x = unit(1.0, "mm")
  )

save_supplement_figure(figure, "figure_2_schedule_clusters", nature_full_width_mm, 170)
