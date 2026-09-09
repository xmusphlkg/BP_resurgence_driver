# Model 2C recovery sampling diagnostics

Recovery tag: `posterior_truth_nine_country_r5`.

A `*.complete.json` marker means only that the fitting workflow finished and wrote its artifacts; it is not evidence of convergence. The independent diagnostic gate requires all 27 core parameters (`log_beta` through `init_natural_recent_fraction`) to have finite diagnostics, maximum R-hat < 1.01, and minimum bulk ESS >= 400.

## Overall status

Expected fits: 180; workflow-complete fits: 180; diagnostically evaluable fits: 180; gate-passing fits: 2.

## By country and truth scenario

| aggregation_level | iso3 | scenario_id | n_expected_fits | n_fit_complete | n_diagnostics_evaluable | n_diagnostic_gate_pass | diagnostic_gate_pass_rate_among_evaluable | worst_max_core_rhat_among_evaluable | worst_min_core_ess_bulk_among_evaluable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| country_scenario | AUS | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.06233 | 49.1094 |
| country_scenario | AUS | null_both | 5 | 5 | 5 | 0 | 0 | 1.02398 | 124.676 |
| country_scenario | AUS | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.02537 | 220.063 |
| country_scenario | AUS | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.54944 | 7.3439 |
| country_scenario | BRA | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.7529 | 6.20089 |
| country_scenario | BRA | null_both | 5 | 5 | 5 | 0 | 0 | 1.02102 | 315.466 |
| country_scenario | BRA | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.27024 | 11.6812 |
| country_scenario | BRA | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.03141 | 143.893 |
| country_scenario | CHN | multi_mechanism | 5 | 5 | 5 | 1 | 0.2 | 1.5378 | 7.33017 |
| country_scenario | CHN | null_both | 5 | 5 | 5 | 0 | 0 | 1.53518 | 7.40434 |
| country_scenario | CHN | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.53354 | 7.40218 |
| country_scenario | CHN | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.05898 | 54.0675 |
| country_scenario | GBR | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.5397 | 7.38054 |
| country_scenario | GBR | null_both | 5 | 5 | 5 | 0 | 0 | 1.5553 | 7.29854 |
| country_scenario | GBR | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.75928 | 6.19749 |
| country_scenario | GBR | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.75459 | 6.28141 |
| country_scenario | JPN | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.02399 | 207.09 |
| country_scenario | JPN | null_both | 5 | 5 | 5 | 0 | 0 | 1.0194 | 355.602 |
| country_scenario | JPN | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.55685 | 7.2915 |
| country_scenario | JPN | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.02759 | 211.356 |
| country_scenario | NZL | multi_mechanism | 5 | 5 | 5 | 1 | 0.2 | 1.01887 | 309.283 |
| country_scenario | NZL | null_both | 5 | 5 | 5 | 0 | 0 | 1.05097 | 167.838 |
| country_scenario | NZL | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.0354 | 120.967 |
| country_scenario | NZL | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.0197 | 265.257 |
| country_scenario | SWE | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.73901 | 6.29317 |
| country_scenario | SWE | null_both | 5 | 5 | 5 | 0 | 0 | 1.05507 | 51.5918 |
| country_scenario | SWE | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.02506 | 328.338 |
| country_scenario | SWE | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.57756 | 7.04005 |
| country_scenario | THA | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.22442 | 13.8135 |
| country_scenario | THA | null_both | 5 | 5 | 5 | 0 | 0 | 1.03006 | 115.098 |
| country_scenario | THA | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.03232 | 279.016 |
| country_scenario | THA | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.0418 | 62.094 |
| country_scenario | USA | multi_mechanism | 5 | 5 | 5 | 0 | 0 | 1.54183 | 7.31707 |
| country_scenario | USA | null_both | 5 | 5 | 5 | 0 | 0 | 1.54059 | 7.3506 |
| country_scenario | USA | reporting_only | 5 | 5 | 5 | 0 | 0 | 1.77122 | 6.19642 |
| country_scenario | USA | transmission_only | 5 | 5 | 5 | 0 | 0 | 1.53671 | 7.38606 |

## Diagnostic failures

| run_id | max_core_rhat | min_core_ess_bulk |
| --- | --- | --- |
| AUS_multi_mechanism_r001 | 1.01171 | 447.521 |
| AUS_multi_mechanism_r002 | 1.0179 | 436.474 |
| AUS_multi_mechanism_r003 | 1.01849 | 407.269 |
| AUS_multi_mechanism_r004 | 1.06233 | 49.1094 |
| AUS_multi_mechanism_r005 | 1.01478 | 365.502 |
| AUS_null_both_r001 | 1.02398 | 124.676 |
| AUS_null_both_r002 | 1.01817 | 447.427 |
| AUS_null_both_r003 | 1.01431 | 230.581 |
| AUS_null_both_r004 | 1.02003 | 499.398 |
| AUS_null_both_r005 | 1.02065 | 410.437 |
| AUS_reporting_only_r001 | 1.01308 | 432.253 |
| AUS_reporting_only_r002 | 1.02409 | 574.244 |
| AUS_reporting_only_r003 | 1.01898 | 406.241 |
| AUS_reporting_only_r004 | 1.02537 | 220.063 |
| AUS_reporting_only_r005 | 1.0175 | 283.898 |
| AUS_transmission_only_r001 | 1.03158 | 107.012 |
| AUS_transmission_only_r002 | 1.01196 | 497.77 |
| AUS_transmission_only_r003 | 1.02026 | 263.644 |
| AUS_transmission_only_r004 | 1.54944 | 7.3439 |
| AUS_transmission_only_r005 | 1.01611 | 318.686 |
| BRA_multi_mechanism_r001 | 1.0252 | 116.122 |
| BRA_multi_mechanism_r002 | 1.01713 | 210.178 |
| BRA_multi_mechanism_r003 | 1.7529 | 6.20089 |
| BRA_multi_mechanism_r004 | 1.03899 | 97.8486 |
| BRA_multi_mechanism_r005 | 1.74545 | 6.30956 |
| BRA_null_both_r001 | 1.02102 | 485.842 |
| BRA_null_both_r002 | 1.01595 | 561.026 |
| BRA_null_both_r003 | 1.01724 | 427.375 |
| BRA_null_both_r004 | 1.01611 | 315.466 |
| BRA_null_both_r005 | 1.01025 | 589.81 |
| BRA_reporting_only_r001 | 1.01818 | 317.536 |
| BRA_reporting_only_r002 | 1.01224 | 443.179 |
| BRA_reporting_only_r003 | 1.01933 | 428.373 |
| BRA_reporting_only_r004 | 1.0178 | 407.68 |
| BRA_reporting_only_r005 | 1.27024 | 11.6812 |
| BRA_transmission_only_r001 | 1.0143 | 532.676 |
| BRA_transmission_only_r002 | 1.03141 | 168.904 |
| BRA_transmission_only_r003 | 1.01565 | 356.477 |
| BRA_transmission_only_r004 | 1.02954 | 143.893 |
| BRA_transmission_only_r005 | 1.02539 | 187.37 |
| CHN_multi_mechanism_r001 | 1.01519 | 272.202 |
| CHN_multi_mechanism_r002 | 1.5378 | 7.33017 |
| CHN_multi_mechanism_r003 | 1.02086 | 194.2 |
| CHN_multi_mechanism_r005 | 1.02613 | 347.89 |
| CHN_null_both_r001 | 1.04906 | 67.2063 |
| CHN_null_both_r002 | 1.0355 | 147.864 |
| CHN_null_both_r003 | 1.53518 | 7.40434 |
| CHN_null_both_r004 | 1.02038 | 454.229 |
| CHN_null_both_r005 | 1.01281 | 590.799 |
| CHN_reporting_only_r001 | 1.02531 | 347.985 |
| CHN_reporting_only_r002 | 1.52844 | 7.40218 |
| CHN_reporting_only_r003 | 1.53354 | 7.49608 |
| CHN_reporting_only_r004 | 1.01285 | 501.864 |
| CHN_reporting_only_r005 | 1.01508 | 467.43 |
| CHN_transmission_only_r001 | 1.01571 | 318.547 |
| CHN_transmission_only_r002 | 1.05898 | 54.0675 |
| CHN_transmission_only_r003 | 1.02816 | 479.121 |
| CHN_transmission_only_r004 | 1.01581 | 293.762 |
| CHN_transmission_only_r005 | 1.02706 | 451.957 |
| GBR_multi_mechanism_r001 | 1.05596 | 62.5555 |
| GBR_multi_mechanism_r002 | 1.01553 | 253.721 |
| GBR_multi_mechanism_r003 | 1.01193 | 383.352 |
| GBR_multi_mechanism_r004 | 1.01659 | 257.052 |
| GBR_multi_mechanism_r005 | 1.5397 | 7.38054 |
| GBR_null_both_r001 | 1.01381 | 384.286 |
| GBR_null_both_r002 | 1.01512 | 413.757 |
| GBR_null_both_r003 | 1.01029 | 510.915 |
| GBR_null_both_r004 | 1.5553 | 7.29854 |
| GBR_null_both_r005 | 1.01235 | 407.57 |
| GBR_reporting_only_r001 | 1.55357 | 7.33497 |
| GBR_reporting_only_r002 | 1.01496 | 376.465 |
| GBR_reporting_only_r003 | 1.02643 | 530.916 |
| GBR_reporting_only_r004 | 1.75928 | 6.19749 |
| GBR_reporting_only_r005 | 1.02636 | 416.134 |
| GBR_transmission_only_r001 | 1.01869 | 170.63 |
| GBR_transmission_only_r002 | 1.01378 | 514.001 |
| GBR_transmission_only_r003 | 1.0152 | 312.561 |
| GBR_transmission_only_r004 | 1.022 | 414.793 |
| GBR_transmission_only_r005 | 1.75459 | 6.28141 |
| JPN_multi_mechanism_r001 | 1.02276 | 337.869 |
| JPN_multi_mechanism_r002 | 1.02399 | 323.648 |
| JPN_multi_mechanism_r003 | 1.01218 | 429.208 |
| JPN_multi_mechanism_r004 | 1.0107 | 207.09 |
| JPN_multi_mechanism_r005 | 1.01235 | 404.661 |
| JPN_null_both_r001 | 1.01912 | 410.229 |
| JPN_null_both_r002 | 1.01656 | 432.945 |
| JPN_null_both_r003 | 1.01794 | 355.602 |
| JPN_null_both_r004 | 1.014 | 455.244 |
| JPN_null_both_r005 | 1.0194 | 393.66 |
| JPN_reporting_only_r001 | 1.04031 | 143.726 |
| JPN_reporting_only_r002 | 1.01982 | 454.558 |
| JPN_reporting_only_r003 | 1.01827 | 520.219 |
| JPN_reporting_only_r004 | 1.01679 | 361.139 |
| JPN_reporting_only_r005 | 1.55685 | 7.2915 |
| JPN_transmission_only_r001 | 1.01767 | 288.577 |
| JPN_transmission_only_r002 | 1.01719 | 616.798 |
| JPN_transmission_only_r003 | 1.01383 | 355.516 |
| JPN_transmission_only_r004 | 1.02759 | 211.356 |
| JPN_transmission_only_r005 | 1.01334 | 447.531 |
| NZL_multi_mechanism_r001 | 1.01887 | 309.283 |
| NZL_multi_mechanism_r002 | 1.01647 | 320.069 |
| NZL_multi_mechanism_r003 | 1.01563 | 447.531 |
| NZL_multi_mechanism_r004 | 1.01514 | 469.596 |
| NZL_null_both_r001 | 1.05097 | 167.838 |
| NZL_null_both_r002 | 1.01404 | 466.192 |
| NZL_null_both_r003 | 1.02902 | 386.272 |
| NZL_null_both_r004 | 1.01314 | 460.245 |
| NZL_null_both_r005 | 1.00909 | 199.806 |
| NZL_reporting_only_r001 | 1.01476 | 303.979 |
| NZL_reporting_only_r002 | 1.02083 | 306.673 |
| NZL_reporting_only_r003 | 1.01334 | 487.325 |
| NZL_reporting_only_r004 | 1.0354 | 120.967 |
| NZL_reporting_only_r005 | 1.01685 | 365.652 |
| NZL_transmission_only_r001 | 1.01686 | 265.257 |
| NZL_transmission_only_r002 | 1.0122 | 459.108 |
| NZL_transmission_only_r003 | 1.01954 | 373.758 |
| NZL_transmission_only_r004 | 1.0197 | 332.605 |
| NZL_transmission_only_r005 | 1.01338 | 447.261 |
| SWE_multi_mechanism_r001 | 1.01308 | 276.861 |
| SWE_multi_mechanism_r002 | 1.00891 | 308.304 |
| SWE_multi_mechanism_r003 | 1.02037 | 355.721 |
| SWE_multi_mechanism_r004 | 1.01191 | 188.294 |
| SWE_multi_mechanism_r005 | 1.73901 | 6.29317 |
| SWE_null_both_r001 | 1.02248 | 339.08 |
| SWE_null_both_r002 | 1.03268 | 537.389 |
| SWE_null_both_r003 | 1.01356 | 447.144 |
| SWE_null_both_r004 | 1.01096 | 460.66 |
| SWE_null_both_r005 | 1.05507 | 51.5918 |
| SWE_reporting_only_r001 | 1.02506 | 347.687 |
| SWE_reporting_only_r002 | 1.02153 | 630.975 |
| SWE_reporting_only_r003 | 1.01179 | 328.338 |
| SWE_reporting_only_r004 | 1.01287 | 466.765 |
| SWE_reporting_only_r005 | 1.01291 | 405.967 |
| SWE_transmission_only_r001 | 1.01832 | 345.861 |
| SWE_transmission_only_r002 | 1.01607 | 474.948 |
| SWE_transmission_only_r003 | 1.01803 | 310.229 |
| SWE_transmission_only_r004 | 1.57756 | 7.04005 |
| SWE_transmission_only_r005 | 1.15744 | 17.6823 |
| THA_multi_mechanism_r001 | 1.01555 | 318.735 |
| THA_multi_mechanism_r002 | 1.22442 | 13.8135 |
| THA_multi_mechanism_r003 | 1.02049 | 151.785 |
| THA_multi_mechanism_r004 | 1.01729 | 313.801 |
| THA_multi_mechanism_r005 | 1.01547 | 595.468 |
| THA_null_both_r001 | 1.03006 | 115.098 |
| THA_null_both_r002 | 1.01457 | 556.189 |
| THA_null_both_r003 | 1.01808 | 460.829 |
| THA_null_both_r004 | 1.02192 | 289.666 |
| THA_null_both_r005 | 1.01278 | 420.695 |
| THA_reporting_only_r001 | 1.03232 | 279.016 |
| THA_reporting_only_r002 | 1.01263 | 461.792 |
| THA_reporting_only_r003 | 1.01292 | 521.889 |
| THA_reporting_only_r004 | 1.01242 | 353.564 |
| THA_reporting_only_r005 | 1.02088 | 559.175 |
| THA_transmission_only_r001 | 1.02191 | 607.343 |
| THA_transmission_only_r002 | 1.02067 | 462.622 |
| THA_transmission_only_r003 | 1.0418 | 62.094 |
| THA_transmission_only_r004 | 1.01829 | 454.702 |
| THA_transmission_only_r005 | 1.01345 | 396.111 |
| USA_multi_mechanism_r001 | 1.54183 | 7.31707 |
| USA_multi_mechanism_r002 | 1.01271 | 437.692 |
| USA_multi_mechanism_r003 | 1.02198 | 397.957 |
| USA_multi_mechanism_r004 | 1.01559 | 351.198 |
| USA_multi_mechanism_r005 | 1.01491 | 364.218 |
| USA_null_both_r001 | 1.53342 | 7.39312 |
| USA_null_both_r002 | 1.01831 | 411.164 |
| USA_null_both_r003 | 1.54059 | 7.3506 |
| USA_null_both_r004 | 1.063 | 46.9485 |
| USA_null_both_r005 | 1.01412 | 381.12 |
| USA_reporting_only_r001 | 1.01392 | 406.503 |
| USA_reporting_only_r002 | 1.77122 | 6.19642 |
| USA_reporting_only_r003 | 1.53243 | 7.37447 |
| USA_reporting_only_r004 | 1.53492 | 7.36916 |
| USA_reporting_only_r005 | 1.53745 | 7.41172 |
| USA_transmission_only_r001 | 1.03207 | 413.026 |
| USA_transmission_only_r002 | 1.01214 | 455.508 |
| USA_transmission_only_r003 | 1.53671 | 7.38606 |
| USA_transmission_only_r004 | 1.01247 | 313.626 |
| USA_transmission_only_r005 | 1.03346 | 79.0488 |

## Incomplete or invalid artifacts

All expected fits have valid completion markers.

These diagnostics qualify recovery estimates; they do not turn simulation recovery into external validation.
