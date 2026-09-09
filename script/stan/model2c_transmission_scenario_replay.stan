data {
  int<lower=1> N_months;
  int<lower=1> N_years;
  int<lower=1> N_age;
  array[N_months] int<lower=1900> year;
  array[N_months] int<lower=1, upper=12> month;
  array[N_months] int<lower=1, upper=N_years> year_index;
  vector<lower=0.01, upper=0.99>[N_months] dtp1;
  vector<lower=0.01, upper=0.99>[N_months] dtp3;
  real<lower=0.01, upper=0.99> pre_mean_dtp1;
  real<lower=0.01, upper=0.99> pre_mean_dtp3;
  real<lower=0.0> pre_mean_incidence_per_100k;
  real<lower=0.0, upper=0.05> silent_exposure_floor_monthly;
  real<lower=0.0, upper=10.0> natural_exposure_contact_sensitivity;
  vector<lower=0.0>[N_months] periodic_multiplier;
  vector<lower=0.01>[N_months] seasonal_multiplier;
  vector<lower=0.30, upper=5.0>[N_months] pathogen_pressure_prior;
  vector<lower=0.0, upper=100.0>[N_months] contact_intensity;
  matrix<lower=0.0>[N_months, N_age] age_population;
  matrix<lower=0.0>[N_months, N_age] age_population_ref;
  vector<lower=0.0>[N_months] monthly_births;
  vector<lower=0.0>[N_months] monthly_births_ref;
  vector<lower=0.0>[N_age] age_width_years;
  vector<lower=0.0, upper=0.95>[N_age] age_aging_rate;
  vector<lower=0.0>[N_age] age_rel_contact;
  matrix<lower=0.0>[N_age, N_age] age_contact_matrix;
  vector<lower=0.0, upper=1.0>[N_age] age_symptom_prob;
  vector<lower=0.0>[N_age] age_report_weight;
  vector<lower=0.0, upper=1.0>[N_age] age_program_reach;
  vector<lower=0.0, upper=1.0>[N_age] routine_dtp1_eligibility;
  vector<lower=0.0, upper=1.0>[N_age] routine_dtp3_eligibility;
  vector<lower=0.0, upper=3.0>[N_age] routine_delay_age_weight;
  real<lower=0.0, upper=15.0> routine_delay_sensitivity;
  vector<lower=0.0>[N_age] age_waning_years;
  vector<lower=0.0, upper=1.40>[N_age] booster_reach_modifier;
  vector<lower=0.35, upper=2.50>[N_age] vaccine_waning_age_modifier;
  real<lower=0.0, upper=1.0> base_ve_infection;
  real<lower=0.0, upper=1.0> base_ve_disease;

  int<lower=0> N_month_obs;
  array[N_month_obs] int<lower=1, upper=N_months> month_obs_idx;
  array[N_month_obs] int<lower=0> y_month;
  vector<lower=0.50, upper=6.00>[N_month_obs] month_obs_weight;
  int<lower=0> N_month_ppc_obs;
  array[N_month_ppc_obs] int<lower=1, upper=N_months> month_ppc_obs_idx;
  array[N_month_ppc_obs] int<lower=0> y_month_ppc;

  int<lower=0> N_age_obs;
  array[N_age_obs] int<lower=1, upper=N_years> age_obs_year_idx;
  array[N_age_obs] int<lower=1, upper=N_age> age_obs_age_idx;
  array[N_age_obs] int<lower=0> y_age;
  vector<lower=0.0, upper=1.0>[N_age_obs] age_obs_weight;

  real log_beta_prior_mean;
  real log_reporting_rate_prior_mean;
  real<lower=0.10, upper=1.50> log_reporting_rate_prior_sd;
  real<lower=0.0, upper=10.0> contact_sensitivity_upper;
  real<lower=0.0, upper=10.0> contact_sensitivity_prior_mean;
  real<lower=0.005, upper=0.40> waning_rate_prior_mean;
  real<lower=0.03, upper=0.12> waning_rate_prior_sd;
  real<lower=0.0, upper=0.50> init_natural_share_prior_mean;
  real<lower=0.05, upper=0.95> init_vaccine_recent_fraction_prior_mean;
  real<lower=0.05, upper=0.95> init_natural_recent_fraction_prior_mean;
  real log_post_transmission_prior_mean;
  real<lower=0.05, upper=1.0> log_post_transmission_prior_sd;
  real<lower=1.0, upper=48.0> rebound_peak_month_prior_mean;
  real<lower=0.0, upper=2.5> pathogen_pressure_scale_prior_mean;
  real<lower=0.05, upper=1.5> pathogen_pressure_scale_prior_sd;
  real<lower=0.0001, upper=0.30> outbreak_seeding_gain_prior_mean;
  real<lower=0.05, upper=2.0> outbreak_seeding_gain_prior_log_sd;
  real reporting_prior_log_mean;
  real<lower=0.05, upper=2.0> reporting_prior_log_sd;
  real<lower=1.0> nb_phi_prior_mean;
  real<lower=0.2> nb_phi_month_prior_mean;
  real<lower=0.2> nb_phi_age_prior_mean;
}

parameters {
  real<lower=-16.11809565, upper=-1.38629436> log_beta; // log(1e-7) to log(0.25)
  real<lower=-6.21460810, upper=0.40546511> log_reporting_rate; // log(0.002) to log(1.5)
  real<lower=0.0, upper=contact_sensitivity_upper> contact_sensitivity;
  real<lower=0.005, upper=0.40> waning_rate;
  real<lower=-0.69314718, upper=0.91629073> log_post_transmission_multiplier; // log(0.5) to log(2.5)
  real<lower=-1.38629436, upper=1.38629436> log_post_reporting_multiplier; // log(0.25) to log(4)
  real<lower=1.0, upper=48.0> rebound_peak_month;
  real<lower=0.75, upper=14.0> rebound_width;
  real<lower=-0.75, upper=0.75> rebound_asymmetry;
  real<lower=0.0, upper=4.0> rebound_release_gain;
  real<lower=2.0, upper=24.0> secondary_rebound_delay;
  real<lower=1.0, upper=16.0> secondary_rebound_width;
  real<lower=0.0, upper=3.0> secondary_rebound_release_gain;
  real<lower=0.0, upper=0.30> outbreak_seeding_gain;
  real<lower=0.0, upper=2.5> seasonal_strength;
  real<lower=0.0, upper=1.0> natural_boosting_efficiency;
  real<lower=0.0, upper=2.5> pathogen_pressure_scale;
  real<lower=0.0, upper=25.0> epidemic_feedback_gain;
  real<lower=0.0, upper=0.08> endemic_reseed_pressure;
  real<lower=0.15, upper=0.85> epidemic_pressure_retention;
  real<lower=0.0, upper=3.0> post_reporting_peak_gain;
  real<lower=-0.80, upper=0.80> post_reporting_trend;
  real<lower=0.2, upper=500.0> nb_phi_month;
  real<lower=0.2, upper=500.0> nb_phi_age;
  real<lower=0.0, upper=0.50> init_natural_share;
  real<lower=0.05, upper=0.95> init_vaccine_recent_fraction;
  real<lower=0.05, upper=0.95> init_natural_recent_fraction;
}

transformed parameters {
  vector<lower=0.0>[N_months] mu_month;
  vector<lower=0.0>[N_months] true_month;
  vector<lower=0.0>[N_months] symptomatic_month;
  matrix<lower=0.0>[N_years, N_age] mu_year_age;
  real<lower=0.0> missed_natural_boosting_2020_2022;
  real<lower=0.0> natural_boosting_flow_2020_2022;
  real<lower=0.0> natural_exposure_recharge_flow_2020_2022;
  real<lower=0.0> routine_dtp1_shortfall_2020_2022;
  real<lower=0.0> routine_dtp3_shortfall_2020_2022;
  real<lower=0.0> routine_vaccination_shortfall_2020_2022;
  real<lower=0.0> natural_waned_stock_2023_start;
  real<lower=0.0> net_demographic_adjustment_2015_2026;
  real beta = exp(log_beta);
  real reporting_rate = exp(log_reporting_rate);
  real post_transmission_multiplier = exp(log_post_transmission_multiplier);
  real post_reporting_multiplier = exp(log_post_reporting_multiplier);
  real natural_waning_month = 0.010;
  real deep_natural_waning_month = 0.003;
  real reported_baseline_monthly_risk = pre_mean_incidence_per_100k / 100000.0 / 12.0;
  real silent_exposure_multiplier = fmin(40.0, fmax(4.0, 1.0 / fmax(reporting_rate, 0.01)));
  real reported_exposure_floor = reported_baseline_monthly_risk * silent_exposure_multiplier * 10.0;
  real background_exposure_floor = fmin(0.05, fmax(silent_exposure_floor_monthly, reported_exposure_floor));
  real exposed_progression = fmin(0.95, fmax(0.55, 1.0 - 0.50 * epidemic_pressure_retention));
  real infectious_clearance = fmin(0.95, fmax(0.35, 1.0 - 0.65 * epidemic_pressure_retention));
  vector<lower=0.0>[N_age] susceptible_naive;
  vector<lower=0.0>[N_age] vaccine_recent;
  vector<lower=0.0>[N_age] vaccine_waned;
  vector<lower=0.0>[N_age] vaccine_partial_recent;
  vector<lower=0.0>[N_age] vaccine_partial_waned;
  vector<lower=0.0>[N_age] natural_recent;
  vector<lower=0.0>[N_age] natural_waned;
  vector<lower=0.0>[N_age] exposed;
  vector<lower=0.0>[N_age] infectious;
  vector<lower=0.0>[N_age] routine_delay_debt;

  missed_natural_boosting_2020_2022 = 0.0;
  natural_boosting_flow_2020_2022 = 0.0;
  natural_exposure_recharge_flow_2020_2022 = 0.0;
  routine_dtp1_shortfall_2020_2022 = 0.0;
  routine_dtp3_shortfall_2020_2022 = 0.0;
  routine_vaccination_shortfall_2020_2022 = 0.0;
  natural_waned_stock_2023_start = 0.0;
  net_demographic_adjustment_2015_2026 = 0.0;

  for (y in 1:N_years) {
    for (a in 1:N_age) {
      mu_year_age[y, a] = 0.0;
    }
  }
  for (a in 1:N_age) {
    real pop0 = fmax(age_population[1, a], 1.0);
    real dtp1_share = fmin(0.97, fmax(0.0, fmax(dtp1[1], dtp3[1]) * age_program_reach[a] * booster_reach_modifier[a]));
    real dtp3_share = fmin(0.92, fmax(0.0, dtp3[1] * age_program_reach[a] * booster_reach_modifier[a]));
    real partial_vaccine_share = fmax(0.0, dtp1_share - dtp3_share);
    real natural_share = fmin(0.50, fmax(0.0, init_natural_share));
    vaccine_recent[a] = pop0 * dtp3_share * init_vaccine_recent_fraction;
    vaccine_waned[a] = pop0 * dtp3_share * (1.0 - init_vaccine_recent_fraction);
    vaccine_partial_recent[a] = pop0 * partial_vaccine_share * init_vaccine_recent_fraction;
    vaccine_partial_waned[a] = pop0 * partial_vaccine_share * (1.0 - init_vaccine_recent_fraction);
    natural_recent[a] = pop0 * natural_share * init_natural_recent_fraction;
    natural_waned[a] = pop0 * natural_share * (1.0 - init_natural_recent_fraction);
    exposed[a] = pop0 * 1e-8;
    infectious[a] = pop0 * 1e-8;
    routine_delay_debt[a] = 0.0;
    susceptible_naive[a] = fmax(
      0.0,
      pop0 - vaccine_recent[a] - vaccine_waned[a] - vaccine_partial_recent[a] - vaccine_partial_waned[a]
        - natural_recent[a] - natural_waned[a] - exposed[a] - infectious[a]
    );
  }

  for (m in 1:N_months) {
    real contact_multiplier = 1.0;
    real exposure_boosting_contact_multiplier = 1.0;
    real primary_release_shape = 0.0;
    real secondary_release_shape = 0.0;
    real reporting_peak_shape = 0.0;
    real seasonal_force = pow(seasonal_multiplier[m], seasonal_strength);
    real excess_epidemic_pressure_multiplier = fmin(
      6.0,
      fmax(0.40, 1.0 + pathogen_pressure_scale * (pathogen_pressure_prior[m] - 1.0))
    );
    real contact_weight_sum = fmax(sum(age_rel_contact), 1e-9);
    real transmission_multiplier = year[m] >= 2023 ? post_transmission_multiplier : 1.0;
    real reporting_multiplier = year[m] >= 2023 ? post_reporting_multiplier : 1.0;
    vector[N_age] infectious_fraction;
    vector[N_age] age_infectious_pressure;
    vector[N_age] next_susceptible_naive = rep_vector(0.0, N_age);
    vector[N_age] next_vaccine_recent = rep_vector(0.0, N_age);
    vector[N_age] next_vaccine_waned = rep_vector(0.0, N_age);
    vector[N_age] next_vaccine_partial_recent = rep_vector(0.0, N_age);
    vector[N_age] next_vaccine_partial_waned = rep_vector(0.0, N_age);
    vector[N_age] next_natural_recent = rep_vector(0.0, N_age);
    vector[N_age] next_natural_waned = rep_vector(0.0, N_age);
    vector[N_age] next_exposed = rep_vector(0.0, N_age);
    vector[N_age] next_infectious = rep_vector(0.0, N_age);
    vector[N_age] next_routine_delay_debt = rep_vector(0.0, N_age);
    true_month[m] = 0.0;
    symptomatic_month[m] = 0.0;
    mu_month[m] = 0.0;

    if (contact_intensity[m] > 0.0) {
      contact_multiplier = 0.03 + 0.97 * exp(-contact_sensitivity * contact_intensity[m] / 100.0);
      exposure_boosting_contact_multiplier = 0.03
        + 0.97 * exp(-natural_exposure_contact_sensitivity * contact_intensity[m] / 100.0);
    }
    if (year[m] >= 2023) {
      real post_month = (year[m] - 2023) * 12 + month[m];
      real primary_width = post_month < rebound_peak_month
        ? rebound_width * exp(-0.5 * rebound_asymmetry)
        : rebound_width * exp(rebound_asymmetry);
      real secondary_peak_month = fmin(48.0, rebound_peak_month + secondary_rebound_delay);
      primary_release_shape = exp(-0.5 * square((post_month - rebound_peak_month) / primary_width));
      secondary_release_shape = exp(-0.5 * square((post_month - secondary_peak_month) / secondary_rebound_width));
      reporting_peak_shape = exp(-0.5 * square((post_month - rebound_peak_month) / fmax(0.75, 0.55 * rebound_width)));
      reporting_multiplier *= 1.0 + post_reporting_peak_gain * reporting_peak_shape;
      reporting_multiplier *= exp(post_reporting_trend * (post_month - 1.0) / 12.0);
    }

    for (a in 1:N_age) {
      real aging_out = age_aging_rate[a];
      real stay = 1.0 - aging_out;
      next_susceptible_naive[a] += susceptible_naive[a] * stay;
      next_vaccine_recent[a] += vaccine_recent[a] * stay;
      next_vaccine_waned[a] += vaccine_waned[a] * stay;
      next_vaccine_partial_recent[a] += vaccine_partial_recent[a] * stay;
      next_vaccine_partial_waned[a] += vaccine_partial_waned[a] * stay;
      next_natural_recent[a] += natural_recent[a] * stay;
      next_natural_waned[a] += natural_waned[a] * stay;
      next_exposed[a] += exposed[a] * stay;
      next_infectious[a] += infectious[a] * stay;
      next_routine_delay_debt[a] += routine_delay_debt[a] * stay;
      if (a < N_age) {
        next_susceptible_naive[a + 1] += susceptible_naive[a] * aging_out;
        next_vaccine_recent[a + 1] += vaccine_recent[a] * aging_out;
        next_vaccine_waned[a + 1] += vaccine_waned[a] * aging_out;
        next_vaccine_partial_recent[a + 1] += vaccine_partial_recent[a] * aging_out;
        next_vaccine_partial_waned[a + 1] += vaccine_partial_waned[a] * aging_out;
        next_natural_recent[a + 1] += natural_recent[a] * aging_out;
        next_natural_waned[a + 1] += natural_waned[a] * aging_out;
        next_exposed[a + 1] += exposed[a] * aging_out;
        next_infectious[a + 1] += infectious[a] * aging_out;
        next_routine_delay_debt[a + 1] += routine_delay_debt[a] * aging_out;
      }
    }
    next_susceptible_naive[1] += monthly_births[m];
    susceptible_naive = next_susceptible_naive;
    vaccine_recent = next_vaccine_recent;
    vaccine_waned = next_vaccine_waned;
    vaccine_partial_recent = next_vaccine_partial_recent;
    vaccine_partial_waned = next_vaccine_partial_waned;
    natural_recent = next_natural_recent;
    natural_waned = next_natural_waned;
    exposed = next_exposed;
    infectious = next_infectious;
    routine_delay_debt = next_routine_delay_debt;

    for (a in 1:N_age) {
      real target_population = fmax(age_population[m, a], 1.0);
      real state_population = fmax(
        susceptible_naive[a] + vaccine_recent[a] + vaccine_waned[a]
          + vaccine_partial_recent[a] + vaccine_partial_waned[a]
          + natural_recent[a] + natural_waned[a] + exposed[a] + infectious[a],
        1.0
      );
      real demographic_adjustment = target_population / state_population;
      net_demographic_adjustment_2015_2026 += abs(target_population - state_population);
      susceptible_naive[a] *= demographic_adjustment;
      vaccine_recent[a] *= demographic_adjustment;
      vaccine_waned[a] *= demographic_adjustment;
      vaccine_partial_recent[a] *= demographic_adjustment;
      vaccine_partial_waned[a] *= demographic_adjustment;
      natural_recent[a] *= demographic_adjustment;
      natural_waned[a] *= demographic_adjustment;
      exposed[a] *= demographic_adjustment;
      infectious[a] *= demographic_adjustment;
      routine_delay_debt[a] = fmin(0.50 * target_population, routine_delay_debt[a] * demographic_adjustment);
    }

    for (a in 1:N_age) {
      real state_population = fmax(
        susceptible_naive[a] + vaccine_recent[a] + vaccine_waned[a]
          + vaccine_partial_recent[a] + vaccine_partial_waned[a]
          + natural_recent[a] + natural_waned[a] + exposed[a] + infectious[a],
        1.0
      );
      infectious_fraction[a] = (infectious[a] + 0.35 * exposed[a]) / state_population;
    }
    for (a in 1:N_age) {
      age_infectious_pressure[a] = 0.0;
      for (b in 1:N_age) {
        age_infectious_pressure[a] += age_contact_matrix[a, b] * infectious_fraction[b];
      }
      age_infectious_pressure[a] = age_infectious_pressure[a] / contact_weight_sum;
    }

    for (a in 1:N_age) {
      real target_population = fmax(age_population[m, a], 1.0);
      real dtp1_effective = fmax(dtp1[m], dtp3[m]);
      real target_started_share = fmin(0.97, fmax(0.0, dtp1_effective * age_program_reach[a] * booster_reach_modifier[a]));
      real target_completed_share = fmin(0.95, fmax(0.0, dtp3[m] * age_program_reach[a] * booster_reach_modifier[a]));
      real reference_started_share = fmin(0.97, fmax(0.0, fmax(pre_mean_dtp1, pre_mean_dtp3) * age_program_reach[a] * booster_reach_modifier[a]));
      real reference_completed_share = fmin(0.95, fmax(0.0, pre_mean_dtp3 * age_program_reach[a] * booster_reach_modifier[a]));
      real target_started_count;
      real target_completed_count;
      real full_vaccine_count;
      real partial_vaccine_count;
      real initiation_flow;
      real completion_need;
      real completion_flow;
      real catchup_completion_flow;
      real vaccine_wane;
      real partial_vaccine_wane;
      real natural_wane;
      real deep_natural_wane;
      real age_waning_rate = waning_rate * vaccine_waning_age_modifier[a];
      real waning_multiplier = exp(-age_waning_rate * age_waning_years[a] / 8.0);
      real recent_ve_infection = fmin(0.95, base_ve_infection * waning_multiplier);
      real recent_ve_disease = fmin(0.98, base_ve_disease * waning_multiplier);
      real susc_vaccine_recent = fmin(1.0, fmax(0.05, 1.0 - recent_ve_infection));
      real susc_vaccine_waned = fmin(1.0, fmax(0.10, 1.0 - 0.45 * recent_ve_infection));
      real susc_partial_recent = fmin(1.0, fmax(0.20, 1.0 - 0.45 * recent_ve_infection));
      real susc_partial_waned = fmin(1.0, fmax(0.35, 1.0 - 0.20 * recent_ve_infection));
      real susc_natural_recent = 0.08;
      real susc_natural_waned = 0.35;
      real symptom_naive = age_symptom_prob[a];
      real symptom_vaccine_recent = age_symptom_prob[a] * fmin(1.0, fmax(0.12, 1.0 - 0.80 * recent_ve_disease));
      real symptom_vaccine_waned = age_symptom_prob[a] * fmin(1.0, fmax(0.25, 1.0 - 0.40 * recent_ve_disease));
      real symptom_partial_recent = age_symptom_prob[a] * fmin(1.0, fmax(0.40, 1.0 - 0.45 * recent_ve_disease));
      real symptom_partial_waned = age_symptom_prob[a] * fmin(1.0, fmax(0.55, 1.0 - 0.20 * recent_ve_disease));
      real symptom_natural_recent = age_symptom_prob[a] * 0.15;
      real symptom_natural_waned = age_symptom_prob[a] * 0.45;
      real outbreak_ignition_pressure = year[m] >= 2023
        ? outbreak_seeding_gain * (
          (1.0 + 0.35 * rebound_release_gain) * primary_release_shape
          + secondary_rebound_release_gain * secondary_release_shape
        )
        : 0.0;
      real transmission_pressure = endemic_reseed_pressure + outbreak_ignition_pressure
        + epidemic_feedback_gain * age_infectious_pressure[a];
      real raw_lambda_age = beta
        * periodic_multiplier[m]
        * seasonal_force
        * excess_epidemic_pressure_multiplier
        * transmission_multiplier
        * transmission_pressure;
      real lambda_age;
      real boosting_lambda_age;
      real reference_lambda_age;
      real dtp1_eligible_count;
      real dtp3_eligible_count;
      real routine_delay_fraction;
      real routine_delay_multiplier;
      real new_naive;
      real new_vaccine_recent;
      real new_vaccine_waned;
      real new_partial_recent;
      real new_partial_waned;
      real new_natural_recent;
      real new_natural_waned;
      real available_vaccine_waned;
      real available_partial_waned;
      real available_natural_waned;
      real boost_vaccine_waned;
      real boost_partial_waned;
      real boost_natural_waned;
      real reference_boost_natural_waned;
      real recharge_driver_lam;
      real recharge_lam;
      real remaining_naive;
      real remaining_vaccine_recent;
      real remaining_vaccine_waned;
      real remaining_partial_recent;
      real remaining_partial_waned;
      real remaining_natural_waned;
      real recharge_naive;
      real recharge_vaccine_recent;
      real recharge_vaccine_waned;
      real recharge_partial_recent;
      real recharge_partial_waned;
      real recharge_natural_waned;
      real natural_recharge;
      real infections;
      real symptomatic;
      real exposed_old;
      real infectious_old;
      real progressed_exposed;
      real cleared_infectious;
      real same_month_progressed;
      real reported;

      if (year[m] == 2023 && month[m] == 1) {
        natural_waned_stock_2023_start += natural_waned[a];
      }

      dtp1_eligible_count = monthly_births[m] * routine_dtp1_eligibility[a];
      dtp3_eligible_count = monthly_births[m] * routine_dtp3_eligibility[a];
      target_started_count = dtp1_effective * dtp1_eligible_count;
      target_completed_count = dtp3[m] * dtp3_eligible_count;
      full_vaccine_count = vaccine_recent[a] + vaccine_waned[a];
      partial_vaccine_count = vaccine_partial_recent[a] + vaccine_partial_waned[a];
      initiation_flow = fmin(
        susceptible_naive[a],
        target_started_count
      );
      susceptible_naive[a] -= initiation_flow;
      vaccine_partial_recent[a] += initiation_flow;
      partial_vaccine_count += initiation_flow;
      completion_need = target_completed_count;
      completion_flow = fmin(partial_vaccine_count, completion_need);
      if (completion_flow > 0.0 && partial_vaccine_count > 0.0) {
        real completion_from_recent = completion_flow * vaccine_partial_recent[a] / partial_vaccine_count;
        real completion_from_waned = completion_flow * vaccine_partial_waned[a] / partial_vaccine_count;
        vaccine_partial_recent[a] = fmax(0.0, vaccine_partial_recent[a] - completion_from_recent);
        vaccine_partial_waned[a] = fmax(0.0, vaccine_partial_waned[a] - completion_from_waned);
        vaccine_recent[a] += completion_flow;
        full_vaccine_count += completion_flow;
      }
      catchup_completion_flow = fmin(
        susceptible_naive[a],
        fmax(0.0, target_completed_count - completion_flow) * 0.50
      );
      susceptible_naive[a] -= catchup_completion_flow;
      vaccine_recent[a] += catchup_completion_flow;
      if (year[m] >= 2020 && year[m] <= 2024) {
        real reference_started_count = fmax(pre_mean_dtp1, pre_mean_dtp3) * dtp1_eligible_count;
        real reference_completed_count = pre_mean_dtp3 * dtp3_eligible_count;
        real dtp1_shortfall = fmax(0.0, reference_started_count - target_started_count);
        real dtp3_shortfall = fmax(0.0, reference_completed_count - target_completed_count);
        routine_dtp1_shortfall_2020_2022 += dtp1_shortfall;
        routine_dtp3_shortfall_2020_2022 += dtp3_shortfall;
        routine_vaccination_shortfall_2020_2022 += dtp1_shortfall + dtp3_shortfall;
        routine_delay_debt[a] = fmin(
          0.50 * target_population,
          routine_delay_debt[a] + 0.50 * dtp1_shortfall + dtp3_shortfall
        );
      }

      vaccine_wane = fmin(vaccine_recent[a], vaccine_recent[a] * age_waning_rate / 12.0);
      vaccine_recent[a] -= vaccine_wane;
      vaccine_waned[a] += vaccine_wane;
      partial_vaccine_wane = fmin(vaccine_partial_recent[a], vaccine_partial_recent[a] * age_waning_rate * 1.20 / 12.0);
      vaccine_partial_recent[a] -= partial_vaccine_wane;
      vaccine_partial_waned[a] += partial_vaccine_wane;
      natural_wane = fmin(natural_recent[a], natural_recent[a] * natural_waning_month);
      natural_recent[a] -= natural_wane;
      natural_waned[a] += natural_wane;
      deep_natural_wane = fmin(natural_waned[a], natural_waned[a] * deep_natural_waning_month);
      natural_waned[a] -= deep_natural_wane;
      susceptible_naive[a] += deep_natural_wane;

      lambda_age = fmin(1.25, fmax(0.0, raw_lambda_age * contact_multiplier));
      boosting_lambda_age = fmin(1.25, fmax(0.0, raw_lambda_age * exposure_boosting_contact_multiplier));
      reference_lambda_age = fmin(1.25, fmax(0.0, raw_lambda_age));
      routine_delay_fraction = fmin(0.40, routine_delay_debt[a] / target_population);
      routine_delay_multiplier = fmin(
        3.0,
        1.0 + routine_delay_sensitivity * routine_delay_age_weight[a] * routine_delay_fraction
      );
      new_naive = susceptible_naive[a] * (1.0 - exp(-lambda_age * routine_delay_multiplier));
      new_vaccine_recent = vaccine_recent[a] * (1.0 - exp(-lambda_age * susc_vaccine_recent));
      new_vaccine_waned = vaccine_waned[a] * (1.0 - exp(-lambda_age * susc_vaccine_waned));
      new_partial_recent = vaccine_partial_recent[a] * (1.0 - exp(-lambda_age * susc_partial_recent * routine_delay_multiplier));
      new_partial_waned = vaccine_partial_waned[a] * (1.0 - exp(-lambda_age * susc_partial_waned * routine_delay_multiplier));
      new_natural_recent = natural_recent[a] * (1.0 - exp(-lambda_age * susc_natural_recent));
      new_natural_waned = natural_waned[a] * (1.0 - exp(-lambda_age * susc_natural_waned));
      available_vaccine_waned = fmax(0.0, vaccine_waned[a] - new_vaccine_waned);
      available_partial_waned = fmax(0.0, vaccine_partial_waned[a] - new_partial_waned);
      available_natural_waned = fmax(0.0, natural_waned[a] - new_natural_waned);
      boost_vaccine_waned = fmin(
        available_vaccine_waned,
        natural_boosting_efficiency * boosting_lambda_age * available_vaccine_waned
      );
      boost_partial_waned = fmin(
        available_partial_waned,
        natural_boosting_efficiency * boosting_lambda_age * available_partial_waned
      );
      boost_natural_waned = fmin(
        available_natural_waned,
        natural_boosting_efficiency * boosting_lambda_age * available_natural_waned
      );
      reference_boost_natural_waned = fmin(
        available_natural_waned,
        natural_boosting_efficiency * reference_lambda_age * available_natural_waned
      );
      recharge_driver_lam = fmax(
        boosting_lambda_age * silent_exposure_multiplier,
        background_exposure_floor * exposure_boosting_contact_multiplier
      );
      recharge_lam = fmin(0.35, natural_boosting_efficiency * recharge_driver_lam);
      remaining_naive = fmax(0.0, susceptible_naive[a] - new_naive);
      remaining_vaccine_recent = fmax(0.0, vaccine_recent[a] - new_vaccine_recent);
      remaining_vaccine_waned = fmax(0.0, available_vaccine_waned - boost_vaccine_waned);
      remaining_partial_recent = fmax(0.0, vaccine_partial_recent[a] - new_partial_recent);
      remaining_partial_waned = fmax(0.0, available_partial_waned - boost_partial_waned);
      remaining_natural_waned = fmax(0.0, available_natural_waned - boost_natural_waned);
      recharge_naive = fmin(remaining_naive, 0.35 * recharge_lam * remaining_naive);
      recharge_vaccine_recent = fmin(remaining_vaccine_recent, 0.15 * recharge_lam * remaining_vaccine_recent);
      recharge_vaccine_waned = fmin(remaining_vaccine_waned, 0.70 * recharge_lam * remaining_vaccine_waned);
      recharge_partial_recent = fmin(remaining_partial_recent, 0.25 * recharge_lam * remaining_partial_recent);
      recharge_partial_waned = fmin(remaining_partial_waned, 0.80 * recharge_lam * remaining_partial_waned);
      recharge_natural_waned = fmin(remaining_natural_waned, recharge_lam * remaining_natural_waned);
      natural_recharge = recharge_naive + recharge_vaccine_recent + recharge_vaccine_waned
        + recharge_partial_recent + recharge_partial_waned + recharge_natural_waned;
      if (year[m] >= 2020 && year[m] <= 2022) {
        natural_boosting_flow_2020_2022 += boost_natural_waned;
        missed_natural_boosting_2020_2022 += fmax(0.0, reference_boost_natural_waned - boost_natural_waned);
        natural_exposure_recharge_flow_2020_2022 += natural_recharge;
      }

      infections = new_naive + new_vaccine_recent + new_vaccine_waned
        + new_partial_recent + new_partial_waned + new_natural_recent + new_natural_waned;
      symptomatic = new_naive * symptom_naive
        + new_vaccine_recent * symptom_vaccine_recent
        + new_vaccine_waned * symptom_vaccine_waned
        + new_partial_recent * symptom_partial_recent
        + new_partial_waned * symptom_partial_waned
        + new_natural_recent * symptom_natural_recent
        + new_natural_waned * symptom_natural_waned;
      reported = symptomatic * reporting_rate * reporting_multiplier * age_report_weight[a];
      true_month[m] += infections;
      symptomatic_month[m] += symptomatic;
      mu_month[m] += reported;
      mu_year_age[year_index[m], a] += reported;

      exposed_old = exposed[a];
      infectious_old = infectious[a];
      progressed_exposed = exposed_progression * exposed_old;
      cleared_infectious = infectious_clearance * infectious_old;
      same_month_progressed = 0.35 * infections;

      susceptible_naive[a] = fmax(0.0, susceptible_naive[a] - new_naive - recharge_naive);
      vaccine_recent[a] = fmax(0.0, vaccine_recent[a] - new_vaccine_recent - recharge_vaccine_recent);
      vaccine_waned[a] = fmax(0.0, vaccine_waned[a] - new_vaccine_waned - boost_vaccine_waned - recharge_vaccine_waned);
      vaccine_partial_recent[a] = fmax(0.0, vaccine_partial_recent[a] - new_partial_recent - recharge_partial_recent);
      vaccine_partial_waned[a] = fmax(
        0.0,
        vaccine_partial_waned[a] - new_partial_waned - boost_partial_waned - recharge_partial_waned
      );
      natural_recent[a] = fmax(
        0.0,
        natural_recent[a] - new_natural_recent + cleared_infectious
          + boost_vaccine_waned + boost_partial_waned + boost_natural_waned + natural_recharge
      );
      natural_waned[a] = fmax(0.0, natural_waned[a] - new_natural_waned - boost_natural_waned - recharge_natural_waned);
      exposed[a] = fmax(0.0, exposed_old - progressed_exposed + infections - same_month_progressed);
      infectious[a] = fmax(0.0, infectious_old - cleared_infectious + progressed_exposed + same_month_progressed);
    }
  }
}

model {
  log_beta ~ normal(log_beta_prior_mean, 1.0);
  log_reporting_rate ~ normal(log_reporting_rate_prior_mean, log_reporting_rate_prior_sd);
  contact_sensitivity ~ normal(contact_sensitivity_prior_mean, 0.75);
  waning_rate ~ normal(waning_rate_prior_mean, waning_rate_prior_sd);
  log_post_transmission_multiplier ~ normal(log_post_transmission_prior_mean, log_post_transmission_prior_sd);
  log_post_reporting_multiplier ~ normal(reporting_prior_log_mean, reporting_prior_log_sd);
  rebound_peak_month ~ normal(rebound_peak_month_prior_mean, 1.5);
  rebound_width ~ lognormal(log(3.0), 0.55);
  rebound_asymmetry ~ normal(0.0, 0.35);
  rebound_release_gain ~ lognormal(log(1.25), 0.45);
  secondary_rebound_delay ~ normal(9.0, 4.0);
  secondary_rebound_width ~ lognormal(log(5.0), 0.50);
  secondary_rebound_release_gain ~ lognormal(log(0.35), 0.70);
  outbreak_seeding_gain ~ lognormal(log(outbreak_seeding_gain_prior_mean), outbreak_seeding_gain_prior_log_sd);
  seasonal_strength ~ normal(1.0, 0.35);
  natural_boosting_efficiency ~ normal(0.60, 0.12);
  pathogen_pressure_scale ~ normal(pathogen_pressure_scale_prior_mean, pathogen_pressure_scale_prior_sd);
  epidemic_feedback_gain ~ lognormal(log(4.0), 0.70);
  endemic_reseed_pressure ~ lognormal(log(0.01), 0.65);
  epidemic_pressure_retention ~ normal(0.45, 0.15);
  post_reporting_peak_gain ~ normal(0.0, 0.45);
  post_reporting_trend ~ normal(0.0, 0.25);
  nb_phi_month ~ lognormal(log(nb_phi_month_prior_mean), 0.35);
  nb_phi_age ~ lognormal(log(nb_phi_age_prior_mean), 0.60);
  init_natural_share ~ normal(init_natural_share_prior_mean, 0.08);
  init_vaccine_recent_fraction ~ normal(init_vaccine_recent_fraction_prior_mean, 0.18);
  init_natural_recent_fraction ~ normal(init_natural_recent_fraction_prior_mean, 0.18);

  for (i in 1:N_month_obs) {
    target += month_obs_weight[i] * neg_binomial_2_lpmf(y_month[i] | mu_month[month_obs_idx[i]] + 1e-6, nb_phi_month);
  }
  for (i in 1:N_age_obs) {
    target += age_obs_weight[i]
      * neg_binomial_2_lpmf(y_age[i] | mu_year_age[age_obs_year_idx[i], age_obs_age_idx[i]] + 1e-6, nb_phi_age);
  }
}

generated quantities {
  real nb_phi = nb_phi_month;
  real endemic_background_force = endemic_reseed_pressure;
  real post_2023_transmission_shift_multiplier = post_transmission_multiplier;
  real excess_epidemic_pressure_scale = pathogen_pressure_scale;
  real outbreak_ignition_pulse_gain = outbreak_seeding_gain;
  real secondary_outbreak_ignition_pulse_gain = outbreak_seeding_gain * secondary_rebound_release_gain;
  real post_reporting_peak_multiplier = 1.0 + post_reporting_peak_gain;
  real post_reporting_trend_per_year = post_reporting_trend;
  real true_infections_2023_2025 = 0.0;
  real symptomatic_infections_2023_2025 = 0.0;
  real reported_cases_2023_2025 = 0.0;
  real reported_cases_2026 = 0.0;

  for (m in 1:N_months) {
    if (year[m] >= 2023 && year[m] <= 2025) {
      true_infections_2023_2025 += true_month[m];
      symptomatic_infections_2023_2025 += symptomatic_month[m];
      reported_cases_2023_2025 += mu_month[m];
    }
    if (year[m] == 2026) {
      reported_cases_2026 += mu_month[m];
    }
  }
}
