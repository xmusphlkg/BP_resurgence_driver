#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

scenario_monthly <- read_csv_first_available(
  c(
    fd_path("figure_4_model2c_counterfactual_scenarios_full.csv"),
    fd_path("figure_4_model2c_counterfactual_scenarios.csv")
  ),
  c(
    "iso3", "country", "scenario_id", "scenario_label", "scenario_family",
    "neutralized_mechanisms", "year", "month", "date", "fit_role",
    "observed_cases", "reported_cases_median", "reported_cases_lower_95",
    "reported_cases_upper_95", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 4 Model 2C scenario trajectories")

scenario_summary <- read_csv_first_available(
  c(
    fd_path("figure_4_model2c_scenario_contrasts_full.csv"),
    fd_path("figure_4_model2c_scenario_contrasts.csv")
  ),
  c(
    "iso3", "country", "scenario_id", "scenario_label", "scenario_family",
    "neutralized_mechanisms", "period_start", "period_end", "outcome_target",
    "full_total_median", "full_total_lower_95", "full_total_upper_95",
    "scenario_total_median", "full_minus_scenario_median",
    "full_minus_scenario_lower_95", "full_minus_scenario_upper_95",
    "scenario_to_full_ratio_median",
    "scenario_to_full_ratio_lower_95", "scenario_to_full_ratio_upper_95",
    "full_peak_median", "scenario_peak_median",
    "peak_ratio_median", "full_peak_month", "scenario_peak_month",
    "scenario_interpretation_note", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 4 Model 2C scenario contrasts")

mechanism_bridge <- read_csv_checked(
  fd_path("model2c_mechanism_bridge.csv"),
  c(
    "iso3", "country", "mechanism_id", "mechanism_label", "scenario_id",
    "input_effect_pct", "state_effect_pct",
    "effective_susceptible_under5_reduction_pct",
    "effective_susceptible_school_age_reduction_pct",
    "effective_susceptible_adult_reduction_pct",
    "under5_natural_immunity_effect_pct",
    "school_age_natural_immunity_effect_pct",
    "adult_natural_immunity_effect_pct",
    "routine_delay_debt_pct_2024_births",
    "all_age_true_reduction_pct", "all_age_symptomatic_reduction_pct",
    "all_age_reported_reduction_pct",
    "under5_true_reduction_pct", "under5_reported_reduction_pct",
    "school_age_true_reduction_pct", "school_age_reported_reduction_pct",
    "adult_true_reduction_pct", "adult_reported_reduction_pct",
    "closure_status", "closure_summary", "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Figure 4 Model 2C mechanism bridge")

pilot <- scenario_monthly %>%
  filter(iso3 %in% pilot_iso3) %>%
  distinct(iso3, country) %>%
  left_join(pilot_order, by = "iso3") %>%
  mutate(
    pilot_rank = coalesce(pilot_rank, 999L),
    country_label = country_display_name(country)
  ) %>%
  arrange(pilot_rank, country)

scenario_curve_levels <- c(
  "full_model",
  "no_contact_suppression",
  "no_missed_exposure_boosting",
  "restored_circulation_boosting_path",
  "no_routine_immunization_disruption",
  "low_vaccine_waning",
  "restored_booster_reach",
  "fixed_birth_cohort_demography",
  "susceptibility_neutral_combined"
)

contrast_levels <- c(
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

susceptibility_levels <- c(
  "no_missed_exposure_boosting",
  "no_routine_immunization_disruption",
  "low_vaccine_waning",
  "restored_booster_reach",
  "fixed_birth_cohort_demography",
  "susceptibility_neutral_combined"
)

susceptibility_display_levels <- c(
  "low_vaccine_waning",
  "susceptibility_neutral_combined",
  "no_missed_exposure_boosting",
  "fixed_birth_cohort_demography",
  "no_routine_immunization_disruption",
  "restored_booster_reach"
)

susceptibility_summary_levels <- c(
  "low_vaccine_waning",
  "susceptibility_neutral_combined",
  "no_missed_exposure_boosting",
  "fixed_birth_cohort_demography"
)

audit_levels <- c(
  "no_post_2023_transmission_context",
  "no_reporting_testing_shift"
)

scenario2c_short_labels <- c(
  no_contact_suppression = "Contact\npath",
  no_missed_exposure_boosting = "Normal\nboosting",
  restored_circulation_boosting_path = "Joint\ndiagnostic",
  no_routine_immunization_disruption = "Routine\nflow",
  low_vaccine_waning = "No\nwaning",
  restored_booster_reach = "Booster\nreach",
  fixed_birth_cohort_demography = "Age\nstructure",
  susceptibility_neutral_combined = "All\nsusceptibility",
  no_post_2023_transmission_context = "Context audit",
  no_reporting_testing_shift = "Reporting audit"
)

outcome_levels <- c("true_infections", "symptomatic_infections", "reported_cases")
outcome_labels <- c(
  true_infections = "Total infections",
  symptomatic_infections = "Symptomatic",
  reported_cases = "Reported"
)

label_signed_pct <- function(x, accuracy = 1) {
  case_when(
    is.na(x) ~ NA_character_,
    abs(x) < accuracy / 2 ~ "0%",
    x > 0 ~ paste0("+", number(x, accuracy = accuracy), "%"),
    TRUE ~ paste0(number(x, accuracy = accuracy), "%")
  )
}

label_signed_axis <- function(x, accuracy = 1) {
  case_when(
    is.na(x) ~ NA_character_,
    abs(x) < accuracy / 2 ~ "0",
    x > 0 ~ paste0("+", number(x, accuracy = accuracy)),
    TRUE ~ number(x, accuracy = accuracy)
  )
}

label_effect_pct <- function(x) {
  case_when(
    is.na(x) ~ NA_character_,
    abs(x) < 0.05 ~ "0",
    abs(x) < 10 ~ label_signed_pct(x, accuracy = 0.1),
    TRUE ~ label_signed_pct(x, accuracy = 1)
  )
}

add_reduction_columns <- function(df) {
  df %>%
    mutate(
      reduction_pct = 100 * (1 - scenario_to_full_ratio_median),
      reduction_lower_pct = 100 * (1 - scenario_to_full_ratio_upper_95),
      reduction_upper_pct = 100 * (1 - scenario_to_full_ratio_lower_95),
      peak_reduction_pct = 100 * (1 - peak_ratio_median),
      interval_excludes_zero = reduction_lower_pct > 0 | reduction_upper_pct < 0,
      support_status = case_when(
        reduction_lower_pct > 0 & abs(reduction_pct) >= 4 ~ "Supported reduction",
        reduction_upper_pct < 0 & abs(reduction_pct) >= 4 ~ "Supported increase",
        TRUE ~ "No clear signal"
      )
    )
}

scenario_curve <- scenario_monthly %>%
  filter(
    iso3 %in% pilot_iso3,
    scenario_id %in% scenario_curve_levels,
    year >= 2023,
    year <= 2025
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    date = as.Date(coalesce(as.character(date), sprintf("%04d-%02d-15", as.integer(year), as.integer(month)))),
    observed_cases = as.numeric(observed_cases),
    reported_cases_median = as.numeric(reported_cases_median),
    reported_cases_lower_95 = as.numeric(reported_cases_lower_95),
    reported_cases_upper_95 = as.numeric(reported_cases_upper_95),
    scenario_id = factor(scenario_id, levels = scenario2c_levels),
    scenario_label_plot = factor(recode(as.character(scenario_id), !!!scenario2c_labels), levels = scenario2c_labels)
  )

if (nrow(scenario_curve) == 0) {
  stop("Figure 4 Model 2C scenario data has no pilot-country rows in 2023-2025.", call. = FALSE)
}

reported_contrasts <- scenario_summary %>%
  filter(
    iso3 %in% pilot_iso3,
    outcome_target == "reported_cases",
    scenario_id %in% contrast_levels,
    period_start == 2023,
    period_end == 2025
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    across(
      c(
        full_total_median,
        scenario_total_median,
        full_minus_scenario_median,
        full_minus_scenario_lower_95,
        full_minus_scenario_upper_95,
        scenario_to_full_ratio_median,
        scenario_to_full_ratio_lower_95,
        scenario_to_full_ratio_upper_95,
        full_peak_median,
        scenario_peak_median,
        peak_ratio_median
      ),
      as.numeric
    ),
    scenario_id = factor(scenario_id, levels = contrast_levels),
    scenario_label_plot = factor(recode(as.character(scenario_id), !!!scenario2c_labels), levels = scenario2c_labels[contrast_levels])
  ) %>%
  add_reduction_columns()

if (nrow(reported_contrasts) == 0) {
  stop("Figure 4 Model 2C reported-case contrast data is empty.", call. = FALSE)
}

waning_for_order <- reported_contrasts %>%
  filter(as.character(scenario_id) == "low_vaccine_waning") %>%
  arrange(desc(reduction_pct), desc(full_minus_scenario_median), country_label)

if (nrow(waning_for_order) == 0) {
  stop("Figure 4 lower-waning contrast data is empty.", call. = FALSE)
}

country_levels <- as.character(waning_for_order$country_label)
country_factor_levels <- rev(country_levels)

reported_reductions <- reported_contrasts %>%
  mutate(country_label = factor(as.character(country_label), levels = country_factor_levels))

country_axis <- tibble(
  country_label = factor(country_factor_levels, levels = country_factor_levels),
  country_y = seq_along(country_factor_levels),
  country_label_text = as.character(country_label)
)

state_contrasts <- scenario_summary %>%
  filter(
    iso3 %in% pilot_iso3,
    outcome_target == "effective_susceptible_start_2025",
    scenario_id %in% contrast_levels,
    period_start == 2025
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    across(
      c(
        full_total_median,
        scenario_total_median,
        full_minus_scenario_median,
        full_minus_scenario_lower_95,
        full_minus_scenario_upper_95,
        scenario_to_full_ratio_median,
        scenario_to_full_ratio_lower_95,
        scenario_to_full_ratio_upper_95
      ),
      as.numeric
    ),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    scenario_id = factor(scenario_id, levels = contrast_levels),
    scenario_label_plot = factor(recode(as.character(scenario_id), !!!scenario2c_labels), levels = scenario2c_labels[contrast_levels])
  ) %>%
  add_reduction_columns()

if (nrow(state_contrasts) == 0) {
  stop("Figure 4 Model 2C immune-state contrast data is empty.", call. = FALSE)
}

outcome_ratio_source <- scenario_summary %>%
  filter(
    iso3 %in% pilot_iso3,
    scenario_id %in% contrast_levels,
    outcome_target %in% outcome_levels,
    period_start == 2023,
    period_end == 2025
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    across(
      c(
        scenario_to_full_ratio_median,
        scenario_to_full_ratio_lower_95,
        scenario_to_full_ratio_upper_95,
        full_minus_scenario_median,
        full_minus_scenario_lower_95,
        full_minus_scenario_upper_95,
        peak_ratio_median
      ),
      as.numeric
    ),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    outcome_target = factor(outcome_target, levels = outcome_levels, labels = outcome_labels[outcome_levels]),
    scenario_id = factor(scenario_id, levels = contrast_levels),
    scenario_label_plot = factor(recode(as.character(scenario_id), !!!scenario2c_labels), levels = scenario2c_labels[contrast_levels])
  ) %>%
  add_reduction_columns()

if (nrow(outcome_ratio_source) == 0) {
  stop("Figure 4 Model 2C outcome-ratio source data is empty.", call. = FALSE)
}

outcome_offsets <- tibble(
  outcome_target = factor(outcome_labels[outcome_levels], levels = outcome_labels[outcome_levels]),
  outcome_offset = c(-0.18, 0, 0.18)
)

outcome_palette <- c(
  "Total infections" = "#6A51A3",
  "Symptomatic" = "#C05A9D",
  "Reported" = clr("black")
)

waning_focus <- reported_reductions %>%
  filter(as.character(scenario_id) == "low_vaccine_waning") %>%
  mutate(
    country_x = factor(as.character(country_label), levels = country_factor_levels),
    absolute_case_label = label_short(full_minus_scenario_median),
    supported_reduction = reduction_lower_pct > 0 & reduction_pct >= 5,
    difference_plot = pmax(full_minus_scenario_median, 0),
    difference_label = if_else(full_minus_scenario_median <= 0, "0", label_short(full_minus_scenario_median))
  )

waning_outcome_ratios <- outcome_ratio_source %>%
  filter(as.character(scenario_id) == "low_vaccine_waning") %>%
  left_join(outcome_offsets, by = "outcome_target") %>%
  mutate(
    country_x = factor(as.character(country_label), levels = country_factor_levels),
    country_y = as.integer(country_label) + outcome_offset,
    ratio_label = if_else(
      outcome_target == "Reported" & abs(reduction_pct) >= 5,
      label_signed_pct(reduction_pct),
      ""
    )
  )

waning_reduction_range <- range(
  c(
    waning_focus$reduction_lower_pct,
    waning_focus$reduction_upper_pct,
    waning_outcome_ratios$reduction_lower_pct,
    waning_outcome_ratios$reduction_upper_pct
  ),
  na.rm = TRUE
)
waning_reduction_limits <- c(
  min(-5, floor(waning_reduction_range[[1]] / 5) * 5),
  max(55, ceiling(waning_reduction_range[[2]] / 5) * 5)
)

waning_focus <- waning_focus %>%
  mutate(
    label_x = pmin(
      pmax(reduction_pct, reduction_upper_pct) + diff(waning_reduction_limits) * 0.025,
      waning_reduction_limits[[2]] - diff(waning_reduction_limits) * 0.02
    )
  )

waning_outcome_ratios <- waning_outcome_ratios %>%
  mutate(
    label_x = pmin(
      pmax(reduction_pct, reduction_upper_pct) + diff(waning_reduction_limits) * 0.025,
      waning_reduction_limits[[2]] - diff(waning_reduction_limits) * 0.02
    )
  )

natural_bridge <- mechanism_bridge %>%
  filter(
    iso3 %in% pilot_iso3,
    mechanism_id == "natural_exposure_boosting"
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    across(
      c(
        input_effect_pct,
        state_effect_pct,
        effective_susceptible_under5_reduction_pct,
        effective_susceptible_school_age_reduction_pct,
        effective_susceptible_adult_reduction_pct,
        under5_natural_immunity_effect_pct,
        school_age_natural_immunity_effect_pct,
        adult_natural_immunity_effect_pct,
        all_age_true_reduction_pct,
        all_age_symptomatic_reduction_pct,
        all_age_reported_reduction_pct,
        under5_true_reduction_pct,
        under5_reported_reduction_pct,
        school_age_true_reduction_pct,
        school_age_reported_reduction_pct,
        adult_true_reduction_pct,
        adult_reported_reduction_pct
      ),
      as.numeric
    ),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    country_x = factor(as.character(country_label), levels = country_factor_levels),
    stock_label = label_signed_pct(state_effect_pct)
  ) %>%
  arrange(country_x)

if (nrow(natural_bridge) == 0) {
  stop("Figure 4 Model 2C natural mechanism bridge data is empty.", call. = FALSE)
}

immune_debt <- natural_bridge %>%
  mutate(
    scenario_id = factor(scenario_id, levels = contrast_levels),
    reduction_pct = state_effect_pct,
    reduction_lower_pct = state_effect_pct,
    reduction_upper_pct = state_effect_pct
  )

natural_age_bridge <- bind_rows(
  natural_bridge %>%
    transmute(
      iso3,
      country,
      country_label,
      country_x,
      scenario_id,
      age_layer = "All ages",
      layer_order = 1L,
      stock_effect_pct = state_effect_pct,
      susceptible_reduction_pct = NA_real_,
      true_reduction_pct = all_age_true_reduction_pct,
      reported_reduction_pct = all_age_reported_reduction_pct
    ),
  natural_bridge %>%
    transmute(
      iso3,
      country,
      country_label,
      country_x,
      scenario_id,
      age_layer = "Under 5",
      layer_order = 2L,
      stock_effect_pct = under5_natural_immunity_effect_pct,
      susceptible_reduction_pct = effective_susceptible_under5_reduction_pct,
      true_reduction_pct = under5_true_reduction_pct,
      reported_reduction_pct = under5_reported_reduction_pct
    ),
  natural_bridge %>%
    transmute(
      iso3,
      country,
      country_label,
      country_x,
      scenario_id,
      age_layer = "5-17 years",
      layer_order = 3L,
      stock_effect_pct = school_age_natural_immunity_effect_pct,
      susceptible_reduction_pct = effective_susceptible_school_age_reduction_pct,
      true_reduction_pct = school_age_true_reduction_pct,
      reported_reduction_pct = school_age_reported_reduction_pct
    ),
  natural_bridge %>%
    transmute(
      iso3,
      country,
      country_label,
      country_x,
      scenario_id,
      age_layer = "18+ years",
      layer_order = 4L,
      stock_effect_pct = adult_natural_immunity_effect_pct,
      susceptible_reduction_pct = effective_susceptible_adult_reduction_pct,
      true_reduction_pct = adult_true_reduction_pct,
      reported_reduction_pct = adult_reported_reduction_pct
    )
) %>%
  mutate(
    age_layer = factor(age_layer, levels = c("All ages", "Under 5", "5-17 years", "18+ years")),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    country_x = factor(as.character(country_x), levels = country_factor_levels)
  ) %>%
  arrange(country_x, layer_order)

natural_age_palette <- c(
  "All ages" = unname(clr("blue")),
  "Under 5" = unname(clr("orange")),
  "5-17 years" = unname(clr("teal")),
  "18+ years" = unname(pal["black"])
)

natural_age_shapes <- c(
  "All ages" = 21,
  "Under 5" = 22,
  "5-17 years" = 24,
  "18+ years" = 23
)

natural_age_summary <- natural_age_bridge %>%
  group_by(age_layer) %>%
  summarise(
    stock_median = median(stock_effect_pct, na.rm = TRUE),
    stock_lower = quantile(stock_effect_pct, 0.25, na.rm = TRUE),
    stock_upper = quantile(stock_effect_pct, 0.75, na.rm = TRUE),
    reported_median = median(reported_reduction_pct, na.rm = TRUE),
    reported_lower = quantile(reported_reduction_pct, 0.25, na.rm = TRUE),
    reported_upper = quantile(reported_reduction_pct, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(
    c(stock_median, reported_median),
    names_to = "metric",
    values_to = "median"
  ) %>%
  mutate(
    lower = if_else(metric == "stock_median", stock_lower, reported_lower),
    upper = if_else(metric == "stock_median", stock_upper, reported_upper),
    metric = recode(
      metric,
      stock_median = "Natural-immunity state",
      reported_median = "Reported cases"
    ),
    metric = factor(metric, levels = c("Natural-immunity state", "Reported cases")),
    age_layer = factor(age_layer, levels = rev(levels(natural_age_bridge$age_layer)))
  )

natural_metric_palette <- c(
  "Natural-immunity state" = "#2E7D32",
  "Reported cases" = "#D95F02"
)

natural_summary_limits <- c(0, max(45, ceiling(max(natural_age_summary$upper, na.rm = TRUE) / 5) * 5))

immune_reduction_limits <- c(
  min(0, floor(min(natural_age_bridge$stock_effect_pct, na.rm = TRUE))),
  max(25, ceiling(max(natural_age_bridge$stock_effect_pct, na.rm = TRUE) / 5) * 5)
)
immune_break_width <- case_when(
  immune_reduction_limits[[2]] <= 3 ~ 0.5,
  immune_reduction_limits[[2]] <= 40 ~ 10,
  TRUE ~ 20
)
immune_translation_breaks <- if (immune_reduction_limits[[2]] > 25) {
  seq(0, ceiling(immune_reduction_limits[[2]] / 10) * 10, by = 10)
} else {
  breaks_width(immune_break_width)
}

routine_bridge <- mechanism_bridge %>%
  filter(
    iso3 %in% pilot_iso3,
    mechanism_id == "routine_dtp_delay"
  ) %>%
  left_join(pilot %>% select(iso3, country_label), by = "iso3") %>%
  mutate(
    across(
      c(
        input_effect_pct,
        state_effect_pct,
        effective_susceptible_under5_reduction_pct,
        effective_susceptible_school_age_reduction_pct,
        effective_susceptible_adult_reduction_pct,
        under5_natural_immunity_effect_pct,
        school_age_natural_immunity_effect_pct,
        adult_natural_immunity_effect_pct,
        routine_delay_debt_pct_2024_births,
        all_age_true_reduction_pct,
        all_age_symptomatic_reduction_pct,
        all_age_reported_reduction_pct,
        under5_true_reduction_pct,
        under5_reported_reduction_pct,
        school_age_true_reduction_pct,
        school_age_reported_reduction_pct,
        adult_true_reduction_pct,
        adult_reported_reduction_pct
      ),
      as.numeric
    ),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    country_x = factor(as.character(country_label), levels = country_factor_levels),
    debt_pct_annual_births = routine_delay_debt_pct_2024_births,
    debt_pct_annual_births_lower = routine_delay_debt_pct_2024_births,
    debt_pct_annual_births_upper = routine_delay_debt_pct_2024_births
  )

routine_debt <- routine_bridge

if (nrow(routine_debt) == 0) {
  stop("Figure 4 Model 2C routine mechanism bridge data is empty.", call. = FALSE)
}

routine_bridge_long <- routine_bridge %>%
  transmute(
    iso3,
    country,
    country_label,
    country_x,
    debt_pct_annual_births,
    under5_reported_reduction_pct
  ) %>%
  pivot_longer(
    c(debt_pct_annual_births, under5_reported_reduction_pct),
    names_to = "bridge_metric",
    values_to = "bridge_value"
  ) %>%
  mutate(
    bridge_metric = recode(
      bridge_metric,
      debt_pct_annual_births = "Susceptibility",
      under5_reported_reduction_pct = "Cases"
    ),
    bridge_metric = factor(
      bridge_metric,
      levels = c("Susceptibility", "Cases")
    ),
    bridge_lower = bridge_value,
    bridge_upper = bridge_value
  )

routine_debt_limits <- c(
  0,
  max(110, ceiling(max(routine_debt$debt_pct_annual_births_upper, na.rm = TRUE) / 10) * 10)
)

routine_bridge_palette <- c(
  "Susceptibility" = "#8C6D31",
  "Cases" = "#B2182B"
)

factor_levels <- c(
  "Natural boost\n2025 stock",
  "Waning\nprotection",
  "Routine DTP\n2025 debt",
  "Booster\nreach",
  "Age\nstructure"
)

factor_matrix <- bind_rows(
  immune_debt %>%
    transmute(
      iso3,
      country,
      country_label,
      factor_label = "Natural boost\n2025 stock",
      scenario_id = as.character(scenario_id),
      metric = "effective_susceptible_start_2025",
      reduction_pct,
      reduction_lower_pct,
      reduction_upper_pct
    ),
  routine_debt %>%
    transmute(
      iso3,
      country,
      country_label,
      factor_label = "Routine DTP\n2025 debt",
      scenario_id = as.character(scenario_id),
      metric = "routine_delay_debt_pct_annual_births_start_2025",
      reduction_pct = debt_pct_annual_births,
      reduction_lower_pct = debt_pct_annual_births_lower,
      reduction_upper_pct = debt_pct_annual_births_upper
    ),
  reported_reductions %>%
    filter(
      as.character(scenario_id) %in% c(
        "low_vaccine_waning",
        "restored_booster_reach",
        "fixed_birth_cohort_demography"
      )
    ) %>%
    transmute(
      iso3,
      country,
      country_label,
      factor_label = recode(
        as.character(scenario_id),
        low_vaccine_waning = "Waning\nprotection",
        restored_booster_reach = "Booster\nreach",
        fixed_birth_cohort_demography = "Age\nstructure"
      ),
      scenario_id = as.character(scenario_id),
      metric = "reported_cases_2023_2025",
      reduction_pct,
      reduction_lower_pct,
      reduction_upper_pct
    )
) %>%
  mutate(
    factor_label = factor(factor_label, levels = factor_levels),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    cell_label = label_effect_pct(reduction_pct),
    factor_status = case_when(
      factor_label == "Routine DTP\n2025 debt" & reduction_pct >= 20 ~ "Strong",
      factor_label == "Routine DTP\n2025 debt" & reduction_pct >= 5 ~ "Moderate",
      factor_label == "Routine DTP\n2025 debt" & reduction_pct >= 1 ~ "Small",
      factor_label == "Natural boost\n2025 stock" & reduction_lower_pct > 0 & reduction_pct >= 10 ~ "Strong",
      factor_label == "Natural boost\n2025 stock" & reduction_lower_pct > 0 & reduction_pct >= 4 ~ "Moderate",
      factor_label == "Natural boost\n2025 stock" & reduction_lower_pct > 0 & reduction_pct >= 1 ~ "Small",
      factor_label == "Waning\nprotection" & reduction_lower_pct > 0 & reduction_pct >= 20 ~ "Strong",
      factor_label == "Waning\nprotection" & reduction_lower_pct > 0 & reduction_pct >= 5 ~ "Moderate",
      factor_label == "Waning\nprotection" & reduction_lower_pct > 0 & reduction_pct >= 1 ~ "Small",
      reduction_lower_pct > 0 & reduction_pct >= 10 ~ "Strong",
      reduction_lower_pct > 0 & reduction_pct >= 4 ~ "Moderate",
      reduction_pct > 0 & reduction_pct >= 1 ~ "Small",
      reduction_pct <= -1 ~ "Opposite",
      TRUE ~ "Limited"
    ),
    factor_status = factor(
      factor_status,
      levels = c("Strong", "Moderate", "Small", "Limited", "Opposite")
    )
  )

factor_counts <- factor_matrix %>%
  distinct(iso3, factor_label, factor_status) %>%
  count(factor_label, factor_status, name = "n_countries") %>%
  complete(
    factor_label = factor(factor_levels, levels = factor_levels),
    factor_status = factor(
      c("Strong", "Moderate", "Small", "Limited", "Opposite"),
      levels = c("Strong", "Moderate", "Small", "Limited", "Opposite")
    ),
    fill = list(n_countries = 0)
  ) %>%
  mutate(
    factor_label = factor(factor_label, levels = rev(factor_levels)),
    count_label = if_else(n_countries > 0, as.character(n_countries), "")
  )

heat_limits <- c(
  min(-15, floor(min(factor_matrix$reduction_pct, na.rm = TRUE) / 5) * 5),
  max(45, ceiling(max(factor_matrix$reduction_pct, na.rm = TRUE) / 5) * 5)
)
heat_breaks <- c(-20, 0, 20, 40)

reduction_fill_palette <- c(
  low = clr("orange"),
  mid = unname(pal["white"]),
  high = clr("blue")
)

factor_status_palette <- c(
  "Strong" = clr("blue"),
  "Moderate" = "#86BCE3",
  "Small" = "#D8DADC",
  "Limited" = "#F2F3F3",
  "Opposite" = "#F2B287"
)

contribution_levels <- c(
  "low_vaccine_waning",
  "no_missed_exposure_boosting",
  "no_routine_immunization_disruption",
  "restored_booster_reach",
  "fixed_birth_cohort_demography"
)

contribution_labels <- c(
  low_vaccine_waning = "No further waning",
  no_missed_exposure_boosting = "Exposure boost restored",
  no_routine_immunization_disruption = "DTP flow restored",
  restored_booster_reach = "Booster reach restored",
  fixed_birth_cohort_demography = "Fixed cohort demography"
)

contribution_palette <- c(
  "No further waning" = clr("blue"),
  "Exposure boost restored" = clr("teal"),
  "DTP flow restored" = clr("orange"),
  "Booster reach restored" = "#7E8B8C",
  "Fixed cohort demography" = "#B65D2E"
)

reported_factor_contrasts <- reported_reductions %>%
  filter(as.character(scenario_id) %in% contribution_levels) %>%
  mutate(
    factor_label = factor(
      recode(as.character(scenario_id), !!!contribution_labels),
      levels = contribution_labels[contribution_levels]
    ),
    country_label = factor(as.character(country_label), levels = country_factor_levels),
    contribution_label = label_effect_pct(reduction_pct),
    interval_status = if_else(
      reduction_lower_pct <= 0 & reduction_upper_pct >= 0,
      "95% UI includes zero",
      "95% UI excludes zero"
    ),
    interval_status = factor(
      interval_status,
      levels = c("95% UI excludes zero", "95% UI includes zero")
    ),
    interval_support = case_when(
      reduction_lower_pct > 0 ~ "Reduction supported",
      reduction_upper_pct < 0 ~ "Opposite direction",
      TRUE ~ "95% UI includes zero"
    ),
    interval_support = factor(
      interval_support,
      levels = c("Reduction supported", "95% UI includes zero", "Opposite direction")
    ),
    contribution_status = case_when(
      reduction_pct >= 20 ~ "Strong",
      reduction_pct >= 5 ~ "Moderate",
      reduction_pct >= 1 ~ "Small",
      reduction_pct <= -1 ~ "Opposite",
      TRUE ~ "Limited"
    )
  )

contribution_summary <- reported_factor_contrasts %>%
  group_by(factor_label) %>%
  summarise(
    median = median(reduction_pct, na.rm = TRUE),
    lower = quantile(reduction_pct, 0.25, na.rm = TRUE),
    upper = quantile(reduction_pct, 0.75, na.rm = TRUE),
    countries_positive = sum(reduction_pct > 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    factor_label = factor(as.character(factor_label), levels = rev(contribution_labels[contribution_levels])),
    support_label = paste0(countries_positive, "/9")
  )

outcome_factor_summary <- outcome_ratio_source %>%
  filter(
    as.character(scenario_id) %in% contribution_levels,
    as.character(outcome_target) %in% outcome_labels[outcome_levels]
  ) %>%
  mutate(
    factor_label = factor(
      recode(as.character(scenario_id), !!!contribution_labels),
      levels = rev(contribution_labels[contribution_levels])
    ),
    outcome_target = factor(as.character(outcome_target), levels = outcome_labels[outcome_levels])
  ) %>%
  group_by(factor_label, outcome_target) %>%
  summarise(
    median = median(reduction_pct, na.rm = TRUE),
    lower = quantile(reduction_pct, 0.25, na.rm = TRUE),
    upper = quantile(reduction_pct, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

contribution_limits <- c(
  min(-10, floor(min(reported_factor_contrasts$reduction_lower_pct, na.rm = TRUE) / 5) * 5),
  max(60, ceiling(max(reported_factor_contrasts$reduction_upper_pct, na.rm = TRUE) / 5) * 5)
)

reported_factor_contrasts <- reported_factor_contrasts %>%
  mutate(
    label_x = pmin(
      pmax(reduction_pct, reduction_upper_pct, na.rm = TRUE) + diff(contribution_limits) * 0.035,
      contribution_limits[[2]] - diff(contribution_limits) * 0.04
    )
  )

p_factor_uncertainty <- ggplot(reported_factor_contrasts, aes(y = country_label)) +
  geom_vline(xintercept = 0, colour = pal["grey"], linewidth = 0.24, linetype = "22") +
  geom_segment(
    aes(
      x = reduction_lower_pct,
      xend = reduction_upper_pct,
      yend = country_label,
      colour = factor_label
    ),
    linewidth = 0.52,
    lineend = "round",
    alpha = 0.92
  ) +
  geom_point(
    aes(x = reduction_pct, colour = factor_label, fill = factor_label),
    shape = 21,
    stroke = 0.25,
    size = 1.55,
    alpha = 0.92
  ) +
  geom_text(
    aes(x = label_x, label = contribution_label),
    hjust = 0,
    size = figure_text$annotation_dense,
    lineheight = 0.90,
    fontface = "bold",
    colour = pal["black"],
    na.rm = TRUE
  ) +
  facet_grid(. ~ factor_label) +
  scale_colour_manual(values = contribution_palette, guide = "none") +
  scale_fill_manual(values = contribution_palette, guide = "none") +
  scale_x_continuous(
    limits = contribution_limits,
    breaks = c(0, 20, 40, 60),
    labels = label_signed_pct,
    expand = expansion(mult = c(0.01, 0.12))
  ) +
  labs(
    x = "Conditional reported reduction (%)\nmedian (95% UI)",
    y = NULL
  ) +
  theme_manuscript(grid = "x") +
  theme(
    panel.spacing.x = unit(2.0, "mm"),
    strip.text.x = element_text(
      size = figure_text$strip_dense,
      lineheight = 0.90,
      margin = margin(1, 1, 1, 1)
    ),
    strip.clip = "off",
    legend.position = "none"
  )

p_outcome_factor <- ggplot(outcome_factor_summary, aes(median, factor_label, colour = outcome_target)) +
  geom_vline(xintercept = 0, colour = pal["grey"], linewidth = 0.26, linetype = "22") +
  geom_pointrange(
    aes(xmin = lower, xmax = upper),
    position = position_dodge(width = 0.52),
    linewidth = 0.48,
    size = 0.39,
    alpha = 0.92
  ) +
  scale_colour_manual(
    values = outcome_palette,
    breaks = names(outcome_palette),
    drop = FALSE,
    guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 0.30, linewidth = 0.58))
  ) +
  scale_x_continuous(
    limits = contribution_limits,
    breaks = breaks_width(20),
    labels = label_signed_pct,
    expand = expansion(mult = c(0.01, 0.05))
  ) +
  labs(
    x = "Conditional reduction by outcome\nacross-country median (IQR; %)",
    y = NULL,
    colour = NULL
  ) +
  theme_manuscript(grid = "x") +
  theme(
    axis.text.y = element_text(lineheight = 0.86),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.key.height = unit(2.0, "mm"),
    legend.key.width = unit(5.0, "mm"),
    legend.text = element_text(size = figure_text$legend_text_dense),
    legend.margin = margin(0, 0, 0, 0)
  )

p_age_summary <- ggplot(natural_age_summary, aes(median, age_layer, colour = metric)) +
  geom_vline(xintercept = 0, colour = pal["grey"], linewidth = 0.26, linetype = "22") +
  geom_pointrange(
    aes(xmin = lower, xmax = upper),
    position = position_dodge(width = 0.48),
    linewidth = 0.56,
    size = 0.39,
    alpha = 0.94
  ) +
  scale_colour_manual(
    values = natural_metric_palette,
    guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 0.35, linewidth = 0.58))
  ) +
  scale_x_continuous(
    limits = natural_summary_limits,
    breaks = c(0, 40, 80),
    labels = label_signed_pct,
    expand = expansion(mult = c(0.01, 0.04))
  ) +
  labs(
    x = "Across-country median (IQR; %)",
    y = NULL,
    colour = NULL
  ) +
  theme_manuscript(grid = "x") +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.key.height = unit(2.0, "mm"),
    legend.key.width = unit(5.0, "mm"),
    legend.text = element_text(size = figure_text$legend_text_dense),
    legend.margin = margin(0, 0, 0, 0)
  )

p_immune_debt <- ggplot(natural_age_bridge, aes(country_x, stock_effect_pct)) +
  geom_hline(yintercept = 0, colour = pal["grey"], linewidth = 0.28, linetype = "22") +
  geom_point(
    aes(colour = age_layer, fill = age_layer, shape = age_layer),
    position = position_dodge(width = 0.58),
    size = 1.95,
    stroke = 0.35,
    alpha = 0.94
  ) +
  scale_colour_manual(
    values = natural_age_palette,
    guide = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  scale_fill_manual(values = natural_age_palette, guide = "none") +
  scale_shape_manual(values = natural_age_shapes, guide = "none") +
  scale_y_continuous(
    limits = immune_reduction_limits,
    breaks = breaks_width(immune_break_width),
    labels = function(x) label_signed_pct(x, accuracy = 0.5),
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  coord_flip(clip = "off") +
  labs(
    x = NULL,
    y = "Natural immunity stock higher (%)",
    colour = NULL
  ) +
  theme_manuscript(grid = "x") +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.67, 0.12),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = pal["white"], colour = NA),
    legend.key.height = unit(2.1, "mm"),
    legend.key.width = unit(4.4, "mm"),
    legend.text = element_text(size = figure_text$legend_text_dense),
    legend.margin = margin(0, 0, 0, 0)
  )

natural_translation_limits <- list(
  x = c(0, max(45, ceiling(max(natural_age_bridge$stock_effect_pct, na.rm = TRUE) / 5) * 5)),
  y = c(0, max(50, ceiling(max(natural_age_bridge$reported_reduction_pct, na.rm = TRUE) / 10) * 10))
)

p_burden <- ggplot(natural_age_bridge, aes(stock_effect_pct, reported_reduction_pct)) +
  geom_vline(xintercept = 0, colour = pal["grey"], linewidth = 0.24, linetype = "22") +
  geom_hline(yintercept = 0, colour = pal["grey"], linewidth = 0.24, linetype = "22") +
  geom_point(
    aes(colour = age_layer, fill = age_layer, shape = age_layer),
    stroke = 0.32,
    size = 1.65,
    alpha = 0.94
  ) +
  scale_colour_manual(values = natural_age_palette, guide = "none") +
  scale_fill_manual(values = natural_age_palette, guide = "none") +
  scale_shape_manual(values = natural_age_shapes, guide = "none") +
  scale_x_continuous(
    limits = natural_translation_limits$x,
    breaks = breaks_width(20),
    labels = label_signed_axis,
    expand = expansion(mult = c(0.02, 0.06))
  ) +
  scale_y_continuous(
    limits = natural_translation_limits$y,
    breaks = breaks_width(10),
    labels = label_signed_pct,
    expand = expansion(mult = c(0.02, 0.06))
  ) +
  labs(
    x = "Natural stock higher (%)",
    y = "Reported reduction (%)"
  ) +
  theme_manuscript(grid = "x")

p_factor_matrix <- ggplot(factor_matrix, aes(factor_label, country_label)) +
  geom_tile(aes(fill = factor_status), colour = pal["white"], linewidth = 0.42, width = 0.94, height = 0.88) +
  geom_vline(xintercept = 3.5, colour = pal["light_grey"], linewidth = 0.26) +
  geom_text(
    aes(label = cell_label),
    size = figure_text$annotation_dense,
    lineheight = 0.90,
    fontface = "bold",
    colour = pal["black"],
    na.rm = TRUE
  ) +
  scale_fill_manual(
    values = factor_status_palette,
    limits = c("Strong", "Moderate", "Small", "Limited", "Opposite"),
    drop = FALSE,
    guide = "none"
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = NULL
  ) +
  theme_manuscript(grid = "x") +
  theme(
    axis.text.x = element_text(lineheight = 0.86),
    panel.grid = element_blank(),
    plot.title = element_blank(),
    legend.position = "none"
  )

p_factor_counts <- ggplot(factor_counts, aes(n_countries, factor_label, fill = factor_status)) +
  geom_col(width = 0.72, colour = pal["white"], linewidth = 0.25) +
  geom_text(
    aes(label = count_label),
    position = position_stack(vjust = 0.5),
    size = figure_text$annotation_dense,
    colour = pal["black"]
  ) +
  scale_fill_manual(
    values = factor_status_palette,
    limits = c("Strong", "Moderate", "Small", "Limited", "Opposite"),
    breaks = c("Strong", "Moderate", "Small", "Limited", "Opposite"),
    labels = c("Strong", "Moderate", "Small", "Limited", "Opposite"),
    drop = FALSE,
    guide = guide_legend(title.position = "top", nrow = 2, byrow = TRUE)
  ) +
  scale_x_continuous(
    limits = c(0, 9),
    breaks = c(0, 3, 6, 9),
    expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    x = "Countries",
    y = NULL,
    fill = "Evidence type"
  ) +
  theme_manuscript(grid = "x") +
  theme(
    axis.text.y = element_text(size = figure_text$axis_text_dense, lineheight = 0.85),
    legend.position = "bottom",
    legend.title = element_text(size = figure_text$legend_title, face = "bold"),
    legend.text = element_text(size = figure_text$legend_text),
    legend.key.size = unit(2.0, "mm"),
    legend.margin = margin(0, 0, 0, 0)
  )

p_routine_debt <- ggplot(routine_bridge_long, aes(country_x, bridge_value, colour = bridge_metric)) +
  geom_hline(yintercept = 0, colour = pal["grey"], linewidth = 0.24, linetype = "22") +
  geom_pointrange(
    aes(ymin = bridge_lower, ymax = bridge_upper),
    position = position_dodge(width = 0.52),
    shape = 21,
    fill = pal["white"],
    linewidth = 0.58,
    size = 0.39,
    stroke = 0.30,
    alpha = 0.92
  ) +
  scale_colour_manual(
    values = routine_bridge_palette,
    guide = guide_legend(
      title.position = "left",
      title.vjust = 0.5,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(size = 0.32, linewidth = 0.55)
    )
  ) +
  scale_y_continuous(
    limits = routine_debt_limits,
    breaks = figure_breaks$percent_coarse,
    labels = function(x) paste0(number(x, accuracy = 1), "%"),
    expand = expansion(mult = c(0.01, 0.04))
  ) +
  coord_flip(clip = "off") +
  labs(
    x = NULL,
    y = "Conditional change (%)",
    colour = "Under-5"
  ) +
  theme_manuscript(
    grid = "x"
  ) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.key.height = unit(2.0, "mm"),
    legend.key.width = unit(5.0, "mm"),
    legend.text = element_text(size = figure_text$legend_text_dense),
    legend.margin = margin(0, 0, 0, 0),
    plot.title = element_blank()
  )

outcome_dodge <- position_dodge(width = 0.62)

p_waning_outcomes <- ggplot(waning_outcome_ratios, aes(country_x, reduction_pct, colour = outcome_target)) +
  geom_hline(yintercept = 0, colour = pal["grey"], linewidth = 0.28, linetype = "22") +
  geom_pointrange(
    aes(ymin = reduction_lower_pct, ymax = reduction_upper_pct),
    position = outcome_dodge,
    linewidth = 0.48,
    size = 0.48,
    alpha = 0.90
  ) +
  scale_colour_manual(
    values = outcome_palette,
    breaks = names(outcome_palette),
    drop = FALSE,
    guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 0.24, linewidth = 0.60, alpha = 1))
  ) +
  scale_y_continuous(
    limits = waning_reduction_limits,
    breaks = breaks_width(10),
    labels = label_signed_pct,
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  coord_flip(clip = "off") +
  labs(
    x = NULL,
    y = "Outcome reduction (%)",
    colour = NULL
  ) +
  theme_manuscript(grid = "x") +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.5, 1),
    legend.justification = c(0.5, 1),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = pal["white"], colour = NA),
    legend.key.height = unit(2.6, "mm"),
    legend.key.width = unit(7.4, "mm"),
    legend.text = element_text(size = figure_text$legend_text),
    legend.margin = margin(0, 0, 0, 0)
  )

figure_layout <- "
AAAAAAAAA
BBBCCCDDD
"

figure <- free(p_factor_uncertainty) +
  p_outcome_factor +
  p_age_summary +
  p_routine_debt +
  plot_layout(
    design = figure_layout,
    widths = rep(1, 9),
    heights = c(1, 1),
    guides = "keep"
  ) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(face = "bold", size = figure_text$tag)
  )

scenario_source <- scenario_curve %>%
  transmute(
    panel = "scenario_monthly_reported_cases",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = as.character(scenario_label_plot),
    outcome_target = "reported_cases",
    year,
    month,
    date,
    median = reported_cases_median,
    lower_95 = reported_cases_lower_95,
    upper_95 = reported_cases_upper_95,
    observed_cases,
    note = "Monthly reported-case trajectory from calibration-ensemble counterfactual replay."
  )

reported_contrast_source <- reported_reductions %>%
  transmute(
    panel = "reported_cumulative_contrast",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = as.character(scenario_label_plot),
    outcome_target = "reported_cases",
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = full_minus_scenario_median,
    lower_95 = full_minus_scenario_lower_95,
    upper_95 = full_minus_scenario_upper_95,
    observed_cases = NA_real_,
    note = "Full minus scenario 2023-2025 cumulative reported cases; not normalized to 100%."
  )

reported_reduction_render_source <- reported_reductions %>%
  transmute(
    panel = "reported_reduction_percent_display",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = as.character(scenario_label_plot),
    outcome_target = "reported_cases",
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = reduction_pct,
    lower_95 = reduction_lower_pct,
    upper_95 = reduction_upper_pct,
    observed_cases = NA_real_,
    note = "Displayed value: 100 * (1 - scenario/full cumulative reported-case ratio); positive means fewer reported cases under the scenario."
  )

factor_interval_render_source <- reported_factor_contrasts %>%
  transmute(
    panel = "factor_reported_reduction_interval_matrix",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = as.character(factor_label),
    outcome_target = "reported_cases",
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = reduction_pct,
    lower_95 = reduction_lower_pct,
    upper_95 = reduction_upper_pct,
    observed_cases = NA_real_,
    interval_label = "95% UI",
    ui_method = "calibration_ensemble_percentile_2.5_97.5",
    note = "Supplementary Fig. 19 panel a: primary-specification diagnostic counterfactual for 2023-2025 reported cases; the point is the median and lower_95/upper_95 form the 95% UI from calibration-ensemble percentiles."
  )

outcome_factor_render_source <- outcome_factor_summary %>%
  transmute(
    panel = "factor_outcome_reduction_summary",
    iso3 = NA_character_,
    country = NA_character_,
    scenario_id = NA_character_,
    scenario_label = as.character(factor_label),
    outcome_target = as.character(outcome_target),
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median,
    lower_95 = lower,
    upper_95 = upper,
    observed_cases = NA_real_,
    note = "Supplementary Fig. 19 panel b: median and interquartile range across the nine modelled countries by outcome target."
  )

natural_age_summary_render_source <- natural_age_summary %>%
  transmute(
    panel = "natural_age_layer_summary",
    iso3 = NA_character_,
    country = NA_character_,
    scenario_id = "no_missed_exposure_boosting",
    scenario_label = as.character(age_layer),
    outcome_target = as.character(metric),
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median,
    lower_95 = lower,
    upper_95 = upper,
    observed_cases = NA_real_,
    note = "Supplementary Fig. 19 panel c: median and interquartile range across the nine modelled countries by age layer."
  )

immune_state_render_source <- natural_age_bridge %>%
  transmute(
    panel = "natural_immunity_stock_start_2025",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = "Natural boosting 2025 stock",
    outcome_target = paste0("natural_immunity_stock_start_2025_", str_replace_all(str_to_lower(as.character(age_layer)), "[^a-z0-9]+", "_")),
    year = 2025L,
    month = 1L,
    date = as.Date("2025-01-15"),
    median = stock_effect_pct,
    lower_95 = stock_effect_pct,
    upper_95 = stock_effect_pct,
    observed_cases = NA_real_,
    note = paste0(
      "Displayed value for ", as.character(age_layer),
      ": 100 * (normal 2021-2024 exposure-boosting scenario / full corrected natural-immunity stock at 2025 entry - 1); waned natural immunity is not double counted."
    )
  )

natural_translation_render_source <- natural_age_bridge %>%
  transmute(
    panel = "natural_stock_to_reported_reduction",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = "Normal exposure boosting",
    outcome_target = paste0("reported_cases_", str_replace_all(str_to_lower(as.character(age_layer)), "[^a-z0-9]+", "_")),
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = reported_reduction_pct,
    lower_95 = NA_real_,
    upper_95 = NA_real_,
    observed_cases = NA_real_,
    note = paste0(
      "Natural bridge translation panel for ", as.character(age_layer),
      ": x is corrected 2025 natural-immunity stock increase and median is 2023-2025 reported-case reduction."
    )
  )

routine_bridge_render_source <- routine_bridge_long %>%
  transmute(
    panel = "routine_bridge_percent_display",
    iso3,
    country,
    scenario_id = "no_routine_immunization_disruption",
    scenario_label = "Restored routine flow",
    outcome_target = as.character(bridge_metric),
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = bridge_value,
    lower_95 = bridge_lower,
    upper_95 = bridge_upper,
    observed_cases = NA_real_,
    note = "Supplementary Fig. 19 panel d: the coverage-derived under-5 susceptible-equivalent state is expressed as percent of 2024 births and paired with the 2023-2025 under-5 reported-case reduction; the state is not a missed-dose count."
  )

factor_matrix_render_source <- factor_matrix %>%
  transmute(
    panel = "factor_split_matrix",
    iso3,
    country,
    scenario_id,
    scenario_label = as.character(factor_label),
    outcome_target = metric,
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = reduction_pct,
    lower_95 = reduction_lower_pct,
    upper_95 = reduction_upper_pct,
    observed_cases = NA_real_,
    note = "Factor split display value; natural uses 2025 natural-immunity stock, routine uses 2025 DTP delay debt as percent of 2024 births, other factors use 2023-2025 reported-case reduction."
  )

outcome_ratio_render_source <- outcome_ratio_source %>%
  transmute(
    panel = "outcome_ratio_source",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = as.character(scenario_label_plot),
    outcome_target = as.character(outcome_target),
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = scenario_to_full_ratio_median,
    lower_95 = scenario_to_full_ratio_lower_95,
    upper_95 = scenario_to_full_ratio_upper_95,
    observed_cases = NA_real_,
    note = "Scenario to full 2023-2025 cumulative outcome ratio."
  )

outcome_reduction_render_source <- outcome_ratio_source %>%
  transmute(
    panel = "outcome_reduction_percent_display",
    iso3,
    country,
    scenario_id = as.character(scenario_id),
    scenario_label = as.character(scenario_label_plot),
    outcome_target = as.character(outcome_target),
    year = NA_integer_,
    month = NA_integer_,
    date = as.Date(NA),
    median = reduction_pct,
    lower_95 = reduction_lower_pct,
    upper_95 = reduction_upper_pct,
    observed_cases = NA_real_,
    note = "Displayed value: 100 * (1 - scenario/full cumulative outcome ratio); positive means fewer outcomes under the scenario."
  )

readr::write_csv(
  bind_rows(
    factor_interval_render_source,
    outcome_factor_render_source,
    natural_age_summary_render_source,
    routine_bridge_render_source
  ),
  fd_path("figure_4_model2c_scenario_replay_rendered.csv")
)
save_supplement_figure(
  figure,
  "figure_4_model2c_scenario_replay",
  nature_full_width_mm,
  120
)
