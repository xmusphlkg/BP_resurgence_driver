from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    label: str
    core_column: str
    biological_role: str
    counterfactual_boundary: str
    claim_boundary: str


MECHANISM_SPECS = (
    MechanismSpec(
        mechanism_id="covid_contact_shock",
        label="COVID-era contact reduction and natural boosting interruption",
        core_column="natural_boosting_interruption",
        biological_role="transmission_immunity",
        counterfactual_boundary="Switch off the COVID-era contact-shock term and its natural-boosting interruption effect.",
        claim_boundary="Uses OxCGRT/contact proxy data rather than direct contact-survey validation.",
    ),
    MechanismSpec(
        mechanism_id="routine_immunization_disruption",
        label="2020-2025 DTP coverage disruption relative to pre-pandemic coverage",
        core_column="coverage_decline",
        biological_role="immunization_program",
        counterfactual_boundary="Hold routine immunization disruption at the pre-pandemic coverage baseline.",
        claim_boundary="Coverage disruption is source-qualified and does not measure timeliness or catch-up directly.",
    ),
    MechanismSpec(
        mechanism_id="observation_reporting_boundary",
        label="Post-pandemic observation residual",
        core_column="reporting_change",
        biological_role="observation_process",
        counterfactual_boundary=(
            "Remove the fitted post-pandemic observation multiplier from reported cases; "
            "this residual can reflect reporting/testing change or unmeasured transmission drift."
        ),
        claim_boundary=(
            "A fitted observation-process residual, not confirmed reporting/testing change, "
        "unless independent testing/reporting covariates are available."
        ),
    ),
    MechanismSpec(
        mechanism_id="post_pandemic_transmission_drift",
        label="Post-pandemic transmission drift",
        core_column="post_pandemic_transmission_drift",
        biological_role="transmission_dynamics",
        counterfactual_boundary=(
            "Hold post-2023 transmission multiplier at one while keeping reporting terms estimated from external diagnostic priors."
        ),
        claim_boundary=(
            "This term represents unmodeled post-2023 transmission pressure (contact behavior, pathogen drift, or other mechanisms not explicit in the reduced model). "
            "It is distinct from observation/reporting process changes when this mechanism is explicitly controlled in scenario runs."
        ),
    ),
    MechanismSpec(
        mechanism_id="vaccine_waning",
        label="Shorter vaccine-derived protection / immune waning",
        core_column="ap_waning",
        biological_role="immunity_waning",
        counterfactual_boundary="Replace the fitted waning behavior with the long-lasting vaccine-immunity counterfactual.",
        claim_boundary="Waning attribution is conditional on fixed vaccine-protection assumptions and available surveillance data.",
    ),
    MechanismSpec(
        mechanism_id="population_age_structure",
        label="Age-structure rescaling",
        core_column="population_age_structure",
        biological_role="demography",
        counterfactual_boundary="Hold age structure at the pre-pandemic proxy distribution.",
        claim_boundary="Current age layer is monthly age-group rescaling, not a birth-death-aging cohort process.",
    ),
)

MECHANISMS = {spec.mechanism_id: spec.label for spec in MECHANISM_SPECS}
CORE_MECHANISM_COLUMNS = {spec.mechanism_id: spec.core_column for spec in MECHANISM_SPECS}


def mechanism_ids() -> list[str]:
    return [spec.mechanism_id for spec in MECHANISM_SPECS]


def mechanism_overrides(subset: set[str]) -> dict[str, Any]:
    return {
        "include_contact_shock": "covid_contact_shock" in subset,
        "include_immunization_disruption": "routine_immunization_disruption" in subset,
        "include_reporting_change": "observation_reporting_boundary" in subset,
        "include_post_pandemic_transmission_drift": "post_pandemic_transmission_drift" in subset,
        "long_lasting_vaccine_immunity": "vaccine_waning" not in subset,
        "include_population_age_structure_change": "population_age_structure" in subset,
    }
