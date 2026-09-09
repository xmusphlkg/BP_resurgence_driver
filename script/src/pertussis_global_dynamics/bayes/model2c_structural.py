from __future__ import annotations

from pathlib import Path
from typing import Any


STRUCTURAL_FLAG_DEFAULTS = {
    "use_exposure_interruption": 1,
    "use_post_2023_waning": 1,
    "use_reporting_shift": 1,
    "use_flexible_outbreak_seeding": 0,
}


def augment_structural_data(data: dict[str, Any], overrides: dict[str, int] | None = None) -> dict[str, Any]:
    """Add explicit, auditable structure switches to one Model 2C data set."""
    flags = dict(STRUCTURAL_FLAG_DEFAULTS)
    flags.update(overrides or {})
    unknown = set(flags) - set(STRUCTURAL_FLAG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown Model 2C structure flag(s): {', '.join(sorted(unknown))}")
    invalid = {name: value for name, value in flags.items() if value not in (0, 1)}
    if invalid:
        raise ValueError(f"Structural flags must be binary: {invalid}")
    return {**data, **flags}


def derive_structural_stan(base_source: str) -> str:
    """Generate the isolated structural-contrast Stan program from the frozen base source."""
    replacements = [
        (
            "data {\n",
            "// GENERATED structural-refit model; do not edit by hand.\n"
            "// The manuscript-default Stan program and outputs remain untouched.\n"
            "data {\n"
            "  int<lower=0, upper=1> use_exposure_interruption;\n"
            "  int<lower=0, upper=1> use_post_2023_waning;\n"
            "  int<lower=0, upper=1> use_reporting_shift;\n"
            "  int<lower=0, upper=1> use_flexible_outbreak_seeding;\n",
        ),
        (
            "  real<lower=0.0, upper=0.30> outbreak_seeding_gain;\n",
            "  real<lower=0.0, upper=0.30> outbreak_seeding_gain;\n"
            "  vector<lower=0.0, upper=0.30>[4] outbreak_seed_basis_gain;\n",
        ),
        (
            "    real reporting_multiplier = year[m] >= 2023 ? post_reporting_multiplier : 1.0;\n",
            "    real reporting_multiplier = (use_reporting_shift == 1 && year[m] >= 2023)\n"
            "      ? post_reporting_multiplier : 1.0;\n",
        ),
        (
            "    if (contact_intensity[m] > 0.0) {\n"
            "      contact_multiplier = 0.03 + 0.97 * exp(-contact_sensitivity * contact_intensity[m] / 100.0);\n"
            "      exposure_boosting_contact_multiplier = 0.03\n"
            "        + 0.97 * exp(-natural_exposure_contact_sensitivity * contact_intensity[m] / 100.0);\n"
            "    }\n",
            "    if (use_exposure_interruption == 1 && contact_intensity[m] > 0.0) {\n"
            "      contact_multiplier = 0.03 + 0.97 * exp(-contact_sensitivity * contact_intensity[m] / 100.0);\n"
            "      exposure_boosting_contact_multiplier = 0.03\n"
            "        + 0.97 * exp(-natural_exposure_contact_sensitivity * contact_intensity[m] / 100.0);\n"
            "    }\n",
        ),
        (
            "      reporting_multiplier *= 1.0 + post_reporting_peak_gain * reporting_peak_shape;\n"
            "      reporting_multiplier *= exp(post_reporting_trend * (post_month - 1.0) / 12.0);\n",
            "      if (use_reporting_shift == 1) {\n"
            "        reporting_multiplier *= 1.0 + post_reporting_peak_gain * reporting_peak_shape;\n"
            "        reporting_multiplier *= exp(post_reporting_trend * (post_month - 1.0) / 12.0);\n"
            "      }\n",
        ),
        (
            "      real outbreak_ignition_pressure = year[m] >= 2023\n"
            "        ? outbreak_seeding_gain * (\n"
            "          (1.0 + 0.35 * rebound_release_gain) * primary_release_shape\n"
            "          + secondary_rebound_release_gain * secondary_release_shape\n"
            "        )\n"
            "        : 0.0;\n",
            "      real outbreak_ignition_pressure = 0.0;\n"
            "      if (year[m] >= 2023) {\n"
            "        real post_month = (year[m] - 2023) * 12 + month[m];\n"
            "        if (use_flexible_outbreak_seeding == 1) {\n"
            "          for (k in 1:4) {\n"
            "            real basis_center = 6.0 + 12.0 * (k - 1);\n"
            "            outbreak_ignition_pressure += outbreak_seed_basis_gain[k]\n"
            "              * exp(-0.5 * square((post_month - basis_center) / 4.0));\n"
            "          }\n"
            "        } else {\n"
            "          outbreak_ignition_pressure = outbreak_seeding_gain * (\n"
            "            (1.0 + 0.35 * rebound_release_gain) * primary_release_shape\n"
            "            + secondary_rebound_release_gain * secondary_release_shape\n"
            "          );\n"
            "        }\n"
            "      }\n",
        ),
        (
            "      vaccine_wane = fmin(vaccine_recent[a], vaccine_recent[a] * age_waning_rate / 12.0);\n"
            "      vaccine_recent[a] -= vaccine_wane;\n"
            "      vaccine_waned[a] += vaccine_wane;\n"
            "      partial_vaccine_wane = fmin(vaccine_partial_recent[a], vaccine_partial_recent[a] * age_waning_rate * 1.20 / 12.0);\n"
            "      vaccine_partial_recent[a] -= partial_vaccine_wane;\n"
            "      vaccine_partial_waned[a] += partial_vaccine_wane;\n"
            "      natural_wane = fmin(natural_recent[a], natural_recent[a] * natural_waning_month);\n"
            "      natural_recent[a] -= natural_wane;\n"
            "      natural_waned[a] += natural_wane;\n"
            "      deep_natural_wane = fmin(natural_waned[a], natural_waned[a] * deep_natural_waning_month);\n"
            "      natural_waned[a] -= deep_natural_wane;\n"
            "      susceptible_naive[a] += deep_natural_wane;\n",
            "      if (use_post_2023_waning == 1 || year[m] < 2023) {\n"
            "        vaccine_wane = fmin(vaccine_recent[a], vaccine_recent[a] * age_waning_rate / 12.0);\n"
            "        vaccine_recent[a] -= vaccine_wane;\n"
            "        vaccine_waned[a] += vaccine_wane;\n"
            "        partial_vaccine_wane = fmin(vaccine_partial_recent[a], vaccine_partial_recent[a] * age_waning_rate * 1.20 / 12.0);\n"
            "        vaccine_partial_recent[a] -= partial_vaccine_wane;\n"
            "        vaccine_partial_waned[a] += partial_vaccine_wane;\n"
            "        natural_wane = fmin(natural_recent[a], natural_recent[a] * natural_waning_month);\n"
            "        natural_recent[a] -= natural_wane;\n"
            "        natural_waned[a] += natural_wane;\n"
            "        deep_natural_wane = fmin(natural_waned[a], natural_waned[a] * deep_natural_waning_month);\n"
            "        natural_waned[a] -= deep_natural_wane;\n"
            "        susceptible_naive[a] += deep_natural_wane;\n"
            "      } else {\n"
            "        vaccine_wane = 0.0;\n"
            "        partial_vaccine_wane = 0.0;\n"
            "        natural_wane = 0.0;\n"
            "        deep_natural_wane = 0.0;\n"
            "      }\n",
        ),
        (
            "  outbreak_seeding_gain ~ lognormal(log(outbreak_seeding_gain_prior_mean), outbreak_seeding_gain_prior_log_sd);\n",
            "  outbreak_seeding_gain ~ lognormal(log(outbreak_seeding_gain_prior_mean), outbreak_seeding_gain_prior_log_sd);\n"
            "  outbreak_seed_basis_gain ~ lognormal(log(outbreak_seeding_gain_prior_mean / 4.0), 0.85);\n",
        ),
        (
            "  real outbreak_ignition_pulse_gain = outbreak_seeding_gain;\n",
            "  real outbreak_ignition_pulse_gain = outbreak_seeding_gain;\n"
            "  real flexible_outbreak_seeding_total_gain = sum(outbreak_seed_basis_gain);\n",
        ),
    ]
    derived = base_source
    for old, new in replacements:
        if derived.count(old) != 1:
            raise ValueError("Base Stan structure changed; refusing ambiguous structural rewrite: " + old.splitlines()[0])
        derived = derived.replace(old, new, 1)
    return derived


def write_structural_stan(base_path: Path, output_path: Path) -> Path:
    source = derive_structural_stan(base_path.read_text(encoding="utf-8"))
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != source:
        output_path.write_text(source, encoding="utf-8")
    return output_path
