#!/usr/bin/env python3
"""Build an observed-stream episode universe candidate table.

This script is intentionally source-agnostic: it records the official/public
source screen completed for the Nature Communications observed-stream upgrade
package, rather than scraping live dashboards whose exports are heterogeneous.
"""

from __future__ import annotations

import csv
from pathlib import Path


ACCESS_DATE = "2026-08-13"

OUT = Path("manuscript/tables/observed_episode_universe_2026-08-13.csv")

FIELDNAMES = [
    "country",
    "period",
    "episode_type",
    "testing_denominator_available",
    "positivity_available",
    "severe_outcome_available",
    "likely_role",
    "source_scope",
    "source_url",
    "access_date",
    "limitations",
]


ROWS = [
    {
        "country": "EU/EEA aggregate",
        "period": "2023-01 to 2024-03; 2024 annual report",
        "episode_type": "regional_resurgence_universe_anchor",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "screen_frame_not_country_episode",
        "source_scope": "ECDC EU/EEA risk assessment and annual epidemiological report; aggregate reported cases and deaths",
        "source_url": "https://www.ecdc.europa.eu/en/publications-data/increase-pertussis-cases-eueea; https://www.ecdc.europa.eu/en/publications-data/pertussis-annual-epidemiological-report-2024",
        "access_date": ACCESS_DATE,
        "limitations": "Regional aggregate identifies the wider official universe but cannot serve as an independent country-level episode; no testing denominator or positivity stream.",
    },
    {
        "country": "Denmark",
        "period": "2022-2024; epidemic peak 2023-11",
        "episode_type": "laboratory_confirmed_resurgence",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "burden_linked_resurgence_candidate",
        "source_scope": "SSI national laboratory-confirmed pertussis annual report, diagnoses by culture, PCR or serology, with death summary",
        "source_url": "https://en.ssi.dk/surveillance-and-preparedness/surveillance-in-denmark/annual-reports-on-disease-incidence/w/whooping-cough---2022-2024",
        "access_date": ACCESS_DATE,
        "limitations": "Laboratory-confirmed numerator only; no national all-test denominator or positivity series was located; hospitalisation series not in the public episode source.",
    },
    {
        "country": "Netherlands",
        "period": "2023-06 to 2025 low-activity follow-up",
        "episode_type": "reported_case_resurgence_with_hospital_admission_and_death_tracking",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "burden_linked_resurgence_candidate",
        "source_scope": "RIVM national whooping-cough figures: reported cases, infant cases, hospital admissions and deaths",
        "source_url": "https://www.rivm.nl/en/whooping-cough/current-figures-on-whooping-cough-in-netherlands",
        "access_date": ACCESS_DATE,
        "limitations": "RIVM notes under-reporting because not everyone with symptoms is tested; no public national testing denominator or positivity series was located.",
    },
    {
        "country": "New Zealand",
        "period": "2024-10-19 to 2026-01-09",
        "episode_type": "national_epidemic_with_case_definition_change",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "burden_linked_resurgence_candidate_with_reporting_rule_caveat",
        "source_scope": "PHF Science national epidemic-to-date report using EpiSurv confirmed, probable and suspect notifications, hospitalisations and deaths",
        "source_url": "https://www.phfscience.nz/media/yrvjjtfw/2026-01-14-pertussis-report.pdf",
        "access_date": ACCESS_DATE,
        "limitations": "Suspect notifications before the mid-December 2024 definition change are included; the public report gives hospitalisations but not testing denominators or positivity.",
    },
    {
        "country": "United States",
        "period": "2024-01-01 to 2025-12-31",
        "episode_type": "national_reported_case_resurgence_with_enhanced_surveillance_reports",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "burden_linked_resurgence_candidate_testing_unresolved",
        "source_scope": "CDC NNDSS pertussis surveillance reports and weekly data page; case counts, age-specific summaries and deaths",
        "source_url": "https://www.cdc.gov/pertussis/php/surveillance/index.html",
        "access_date": ACCESS_DATE,
        "limitations": "No official national testing denominator or positivity stream in the surveillance report; 2024-2025 reports are provisional and affected by the post-2020 case-definition framework.",
    },
    {
        "country": "Japan",
        "period": "2025 weeks 1-21; ARI sentinel change from week 15",
        "episode_type": "rapid_resurgence_with_reporting_facility_change",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "detection_change_comparator_not_negative_control",
        "source_scope": "JIHS national IDWR featured report; cumulative notified cases, diagnosis method mix, ARI sentinel caveat and infant death note",
        "source_url": "https://id-info.jihs.go.jp/en/surveillance/idwr/featured/2025/22/index.html",
        "access_date": ACCESS_DATE,
        "limitations": "The report states ARI sentinel reporting began in week 15 and may have increased reporting facilities, but judged the rise not fully explained by that change; no all-test denominator or positivity series.",
    },
    {
        "country": "Germany",
        "period": "2024-01-01 to 2024-11-19",
        "episode_type": "partial_year_national_resurgence_with_deaths",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "burden_linked_resurgence_candidate_testing_unresolved",
        "source_scope": "Robert Koch Institute Epidemiologisches Bulletin 48/2024; national notified cases, age incidence and deaths through 19 Nov 2024",
        "source_url": "https://edoc.rki.de/bitstream/handle/176904/12324/EB-48-2024_10-25646-12936.pdf?isAllowed=y&sequence=1",
        "access_date": ACCESS_DATE,
        "limitations": "Partial-year official summary; no public testing denominator or positivity stream identified in the bulletin.",
    },
    {
        "country": "Sweden",
        "period": "2024-01-01 to 2025 follow-up",
        "episode_type": "national_resurgence_with_improved_diagnostics_note",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "mixed_burden_and_detection_candidate_not_negative_control",
        "source_scope": "Public Health Agency of Sweden pertussis statistics and vaccination information pages; annual cases, infant death and interpretation note",
        "source_url": "https://www.folkhalsomyndigheten.se/statistik-och-data/hitta-statistik-och-data/kikhosta-statistik/; https://www.folkhalsomyndigheten.se/vara-amnesomraden/vaccinationer/vaccinationer-a-o/vaccination-mot-kikhosta/",
        "access_date": ACCESS_DATE,
        "limitations": "Official page attributes the 2024 increase to lower post-pandemic immunity and improved diagnostics; no public 2024 all-test denominator, positivity, or hospitalisation table was located.",
    },
    {
        "country": "Australia",
        "period": "2024-01-01 to 2025-12-31",
        "episode_type": "national_reported_case_resurgence_with_lagged_severe_outcome_release",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "PARTIAL_LAGGED",
        "likely_role": "prospective_negative_control_candidate_requires_2024_2025_severe_extraction",
        "source_scope": "Australian CDC topic summary and AIHW vaccine-preventable disease summary; notifications through 2024 and latest public hospital/death summary to 2022",
        "source_url": "https://www.cdc.gov.au/diseases/whooping-cough-pertussis; https://www.aihw.gov.au/reports/immunisation/vaccine-preventable-diseases/contents/summary",
        "access_date": ACCESS_DATE,
        "limitations": "Large notification surge is official, but national hospitalisation/death data in the public AIHW summary lag to 2022; cannot classify detection-change dominance until 2024-2025 severe outcomes are extracted or released.",
    },
    {
        "country": "Thailand",
        "period": "2023-01-01 to 2024-01-31",
        "episode_type": "outbreak_with_lab_submission_snapshot_and_infant_deaths",
        "testing_denominator_available": "TRUE",
        "positivity_available": "TRUE",
        "severe_outcome_available": "TRUE",
        "likely_role": "burden_linked_resurgence_with_testing_snapshot",
        "source_scope": "Thailand DDC situation report and WHO SEARO bulletin; suspected cases, laboratory submissions, results and infant deaths",
        "source_url": "https://ddc.moph.go.th/uploads/ckeditor2//files/%E0%B8%A3%E0%B8%B2%E0%B8%A2%E0%B8%87%E0%B8%B2%E0%B8%99%E0%B8%AA%E0%B8%96%E0%B8%B2%E0%B8%99%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%93%E0%B9%8C%E0%B9%82%E0%B8%A3%E0%B8%84%E0%B9%84%E0%B8%AD%E0%B8%81%E0%B8%A3%E0%B8%99-2.pdf; https://cdn.who.int/media/docs/default-source/searo/whe/wherepib/2024_3_searo_epi_bulletin.pdf",
        "access_date": ACCESS_DATE,
        "limitations": "Testing denominator is laboratory submissions among suspected outbreak reports, not a continuous community or routine national testing series; source language and PDF extraction require human verification.",
    },
    {
        "country": "Canada",
        "period": "2024-01-01 to 2025-12-31",
        "episode_type": "national_high_case_count_signal_with_lab_method_context",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "FALSE",
        "likely_role": "eligible_resurgence_requires_provincial_or_authorised_severe_extraction",
        "source_scope": "PHAC health-professional pertussis page and PHAC laboratory-methods survey; national statement on high case counts and diagnostic-method context",
        "source_url": "https://www.canada.ca/en/public-health/services/diseases/whooping-cough-pertussis/health-professionals.html; https://www.canada.ca/en/public-health/services/reports-publications/canada-communicable-disease-report-ccdr/monthly-issue/2026-52/issue-5-may-2026/laboratory-diagnosis-pertussis-survey-provincial-public-health-laboratory-methods.html",
        "access_date": ACCESS_DATE,
        "limitations": "Public national page states high case counts in parts of Canada but does not provide a national episode denominator, testing volume, positivity, or severe-outcome series; provincial extraction is needed.",
    },
    {
        "country": "United States",
        "period": "2020 case-definition change affecting 2023-2025 surveillance baseline",
        "episode_type": "case_definition_sensitivity_control_outside_primary_episode_window",
        "testing_denominator_available": "FALSE",
        "positivity_available": "FALSE",
        "severe_outcome_available": "TRUE",
        "likely_role": "off_window_detection_change_control_for_sensitivity_only",
        "source_scope": "CDC surveillance manual and annual surveillance reports under the 2020 CSTE definition",
        "source_url": "https://www.cdc.gov/surv-manual/php/table-of-contents/chapter-10-pertussis.html; https://www.cdc.gov/pertussis/php/surveillance/index.html",
        "access_date": ACCESS_DATE,
        "limitations": "Not a 2023-2025 change event; useful only as an off-window sensitivity/control because the revised reporting definition shapes later observed case counts.",
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"wrote {OUT} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
