from __future__ import annotations

import numpy as np

MODEL2_EXCLUDED_MAIN_ANALYSIS = {
    "SGP": (
        "Primary nine-country set preserved the frozen source library; a separate Singapore sensitivity "
        "fit met the sampling criterion (maximum R-hat 1.008, minimum bulk ESS 759) and changed descriptive "
        "scenario reductions by at most 0.013 percentage points."
    ),
}

PRIMARY_DTP_COVERAGE_TYPE = "WHO/UNICEF Estimates of National Immunization Coverage"

PREFERRED_COVERAGE_TYPES = {
    PRIMARY_DTP_COVERAGE_TYPE: 0,
    "Official coverage": 1,
    "Administrative coverage": 2,
}

AGE_ORDER = [
    "infant_0_2m",
    "infant_3_11m",
    "child_1_4y",
    "child_5_9y",
    "adolescent_10_17y",
    "young_adult_18_39y",
    "middle_adult_40_64y",
    "elderly_65plus",
]

AGE_LABELS = {
    "infant_0_2m": "<3 months",
    "infant_3_11m": "3-11 months",
    "child_1_4y": "1-4 years",
    "child_5_9y": "5-9 years",
    "adolescent_10_17y": "10-17 years",
    "young_adult_18_39y": "18-39 years",
    "middle_adult_40_64y": "40-64 years",
    "elderly_65plus": "65+ years",
}

AGE_BOUNDS = {
    "infant_0_2m": (0.0, 0.25),
    "infant_3_11m": (0.25, 1.0),
    "child_1_4y": (1.0, 5.0),
    "child_5_9y": (5.0, 10.0),
    "adolescent_10_17y": (10.0, 18.0),
    "young_adult_18_39y": (18.0, 40.0),
    "middle_adult_40_64y": (40.0, 65.0),
    "elderly_65plus": (65.0, 100.0),
}

AGE_REL_CONTACT = {
    "infant_0_2m": 0.35,
    "infant_3_11m": 0.45,
    "child_1_4y": 1.10,
    "child_5_9y": 1.35,
    "adolescent_10_17y": 1.45,
    "young_adult_18_39y": 1.00,
    "middle_adult_40_64y": 0.78,
    "elderly_65plus": 0.50,
}

AGE_SYMPTOM_PROB = {
    "infant_0_2m": 0.85,
    "infant_3_11m": 0.70,
    "child_1_4y": 0.45,
    "child_5_9y": 0.35,
    "adolescent_10_17y": 0.28,
    "young_adult_18_39y": 0.20,
    "middle_adult_40_64y": 0.18,
    "elderly_65plus": 0.24,
}

AGE_REPORT_WEIGHT = {
    "infant_0_2m": 3.5,
    "infant_3_11m": 2.5,
    "child_1_4y": 1.4,
    "child_5_9y": 1.0,
    "adolescent_10_17y": 0.8,
    "young_adult_18_39y": 0.45,
    "middle_adult_40_64y": 0.40,
    "elderly_65plus": 0.65,
}

AGE_VACCINE_PROGRAM_REACH = {
    "infant_0_2m": 0.10,
    "infant_3_11m": 0.60,
    "child_1_4y": 0.92,
    "child_5_9y": 0.85,
    "adolescent_10_17y": 0.70,
    "young_adult_18_39y": 0.45,
    "middle_adult_40_64y": 0.30,
    "elderly_65plus": 0.20,
}

AGE_WANING_YEARS = {
    "infant_0_2m": 0.1,
    "infant_3_11m": 0.5,
    "child_1_4y": 2.5,
    "child_5_9y": 7.0,
    "adolescent_10_17y": 13.5,
    "young_adult_18_39y": 25.0,
    "middle_adult_40_64y": 45.0,
    "elderly_65plus": 70.0,
}

NB_DISPERSION = 25.0
BOOTSTRAP_SEED = 20260604
MONTHLY_ODE_AGE_OBJECTIVE_WEIGHT = 0.35
MONTHLY_ODE_JOINT_AGE_MONTH_OBJECTIVE_WEIGHT = 0.20
MONTHLY_ODE_PROFILE_FIXED_FACTORS = (0.85, 1.00, 1.18)
MONTHLY_ODE_PROFILE_NUISANCE_FACTORS = (1.00,)
MONTHLY_ODE_PROFILE_SUPPORT_95_1DF = 3.84
MONTHLY_ODE_OPTIMIZER_MAXITER = 48
MONTHLY_ODE_RESIDUAL_ADJUSTMENT_PENALTY_WEIGHT = 0.300
MONTHLY_ODE_TRANSITION_PENALTY_WEIGHT = 0.050
MONTHLY_ODE_RESIDUAL_ADJUSTMENT_FREE_LOG = np.log(2.0)
MONTHLY_ODE_TRANSITION_FREE_LOG = np.log(2.2)
MONTHLY_ODE_INITIAL_OBSERVATION_BLEND = 0.95
MONTHLY_ODE_SEARCH_PARAMETERS = [
    "ode_beta_scale",
    "ode_pandemic_beta_scale",
    "ode_importation_scale",
    "ode_post_beta_scale",
    "ode_pandemic_reporting_scale",
    "ode_post_reporting_scale",
    "ode_seasonal_strength",
    "ode_pandemic_seasonal_shift_months",
    "ode_post_seasonal_shift_months",
    "ode_initial_infectious_scale",
    "ode_pathogen_drift_scale",
]
MECHANISM_PROFILE_NUISANCE_FACTORS = (0.75, 1.00, 1.33)
MECHANISM_PROFILE_SUPPORT_95_1DF = 3.84

MONTHLY_ODE_PARAMETER_BOUNDS = {
    "ode_beta_scale": (0.05, 18.0),
    "ode_reporting_scale": (0.02, 250.0),
    "ode_pandemic_beta_scale": (0.05, 3.0),
    "ode_post_beta_scale": (0.15, 18.0),
    "ode_pandemic_reporting_scale": (0.05, 20.0),
    "ode_post_reporting_scale": (0.05, 80.0),
    "ode_importation_scale": (0.03, 80.0),
    "ode_seasonal_strength": (0.25, 2.25),
    "ode_pandemic_seasonal_shift_months": (-4.0, 4.0),
    "ode_post_seasonal_shift_months": (-4.0, 4.0),
    "ode_initial_infectious_scale": (0.05, 35.0),
    "ode_pathogen_drift_scale": (0.05, 2.50),
}

MONTHLY_ODE_PARAMETER_NAMES = list(MONTHLY_ODE_PARAMETER_BOUNDS)
MONTHLY_ODE_SIGNED_PARAMETERS = {
    "ode_pandemic_seasonal_shift_months",
    "ode_post_seasonal_shift_months",
}

PARAMETER_BOUNDS = {
    "beta": (1e-6, 0.20),
    "reporting_rate": (0.005, 0.50),
    "contact_sensitivity": (0.0, 1.5),
    "boosting_debt_weight": (0.0, 4.0),
    "coverage_disruption_weight": (0.0, 3.0),
    "post_reporting_multiplier": (0.25, 4.0),
    "post_transmission_multiplier": (0.25, 3.5),
    "waning_rate": (0.005, 0.40),
}

MODEL2_PARAMETER_NAMES = [
    "beta",
    "reporting_rate",
    "contact_sensitivity",
    "boosting_debt_weight",
    "coverage_disruption_weight",
    "post_reporting_multiplier",
    "post_transmission_multiplier",
    "waning_rate",
]

BOOTSTRAP_OUTCOME_METRICS = [
    "reported_cases_2023_2025",
    "true_infections_2023_2025",
    "reported_under_1y_2023_2025",
    "post_pandemic_rmse_log_cases",
]

VACCINE_ASSUMPTION_PROFILES = {
    "base": {
        "profile_label": "Base vaccine protection assumptions",
        "vaccine_infection_multiplier": 1.0,
        "vaccine_disease_multiplier": 1.0,
        "profile_note": "Uses the staged Model 2 vaccine protection assumptions.",
    },
    "disease_only_no_infection_blocking": {
        "profile_label": "Disease protection only",
        "vaccine_infection_multiplier": 0.0,
        "vaccine_disease_multiplier": 1.0,
        "profile_note": "Removes vaccine protection against infection while retaining protection against symptomatic disease.",
    },
    "weak_infection_blocking": {
        "profile_label": "Weak infection protection",
        "vaccine_infection_multiplier": 0.5,
        "vaccine_disease_multiplier": 1.0,
        "profile_note": "Halves vaccine protection against infection while retaining base disease protection.",
    },
    "strong_infection_blocking": {
        "profile_label": "Strong infection protection",
        "vaccine_infection_multiplier": 1.35,
        "vaccine_disease_multiplier": 1.0,
        "profile_note": "Increases vaccine protection against infection by 35%, clipped inside the simulator.",
    },
    "strong_disease_protection": {
        "profile_label": "Strong disease protection",
        "vaccine_infection_multiplier": 1.0,
        "vaccine_disease_multiplier": 1.10,
        "profile_note": "Increases vaccine protection against symptomatic disease by 10%, clipped inside the simulator.",
    },
}

ASYMPTOMATIC_INFECTIOUSNESS_PROFILES = {
    "low_asymptomatic_infectiousness": {
        "profile_label": "Low asymptomatic infectiousness",
        "asymptomatic_relative_infectiousness": 0.2,
        "profile_note": "Assumes mild or asymptomatic infections are substantially less infectious than symptomatic infections.",
    },
    "base": {
        "profile_label": "Base asymptomatic infectiousness",
        "asymptomatic_relative_infectiousness": 0.5,
        "profile_note": "Uses the staged Model 2 assumption that mild or asymptomatic infections are half as infectious as symptomatic infections.",
    },
    "high_asymptomatic_infectiousness": {
        "profile_label": "High asymptomatic infectiousness",
        "asymptomatic_relative_infectiousness": 0.8,
        "profile_note": "Assumes mild or asymptomatic infections transmit almost as efficiently as symptomatic infections.",
    },
    "symptomatic_equivalent_infectiousness": {
        "profile_label": "Symptomatic-equivalent infectiousness",
        "asymptomatic_relative_infectiousness": 1.0,
        "profile_note": "Stress test where mild or asymptomatic infections have the same infectiousness as symptomatic infections.",
    },
}

CONTACT_PROXY_PROFILES = {
    "stringency_index": {
        "profile_label": "OxCGRT stringency index",
        "contact_proxy_source": "OxCGRT stringency_index",
        "profile_note": "Base Model 2 proxy using the annual OxCGRT stringency index.",
    },
    "school_closure": {
        "profile_label": "School closure proxy",
        "contact_proxy_source": "OxCGRT school_closure scaled to 0-100",
        "profile_note": "Uses school-closure intensity as a contact-shock proxy, emphasizing child and adolescent contact disruption.",
    },
    "workplace_stay_home": {
        "profile_label": "Workplace plus stay-home proxy",
        "contact_proxy_source": "Mean of OxCGRT workplace_closure and stay_home_requirement scaled to 0-100",
        "profile_note": "Uses workplace and stay-home restrictions as an adult/general contact-shock proxy.",
    },
    "closure_composite": {
        "profile_label": "Closure composite proxy",
        "contact_proxy_source": "Mean of OxCGRT school_closure, workplace_closure, and stay_home_requirement scaled to 0-100",
        "profile_note": "Uses a simple closure-policy composite instead of the broader OxCGRT stringency index.",
    },
}
