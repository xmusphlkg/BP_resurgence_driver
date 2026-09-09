from __future__ import annotations

import numpy as np
import pandas as pd

from script.run_model2c_surveillance_design import _match_notification_trajectory
from script.run_model2c_surveillance_precision_frontier import _frontier_summary, _settings

from pertussis_global_dynamics.bayes.model2c_surveillance_design import (
    MEASUREMENT_DESIGNS,
    OPTIONAL_STREAMS,
    assign_group_folds,
    cross_validated_predictions,
    discrimination_metrics,
    feature_columns_for_package,
    leave_one_country_out_predictions,
    minimum_successful_packages,
    package_id,
    shapley_stream_gains,
    simulate_observation_features,
    structural_contrasts,
    surveillance_packages,
)


def _latent_row() -> dict[str, object]:
    row: dict[str, object] = {
        "iso3": "AUS",
        "draw_id": "1:1:0",
        "draw_rank": 0,
        "scenario_id": "full_model",
        "group_id": "AUS:1:1:0",
        "nb_phi_month": 30.0,
        "nb_phi_age": 40.0,
        "latent_severe_source_baseline": 500.0,
    }
    for month in range(1, 13):
        row[f"latent_notification_baseline_m{month:02d}"] = 10.0 + month
    for year in (2023, 2024, 2025):
        for month in range(1, 13):
            row[f"latent_notification_mean_{year}_m{month:02d}"] = 20.0 + month
        row[f"latent_age_share_under5_{year}"] = 0.20
        row[f"latent_age_share_school_age_{year}"] = 0.35
        row[f"latent_age_share_adult_{year}"] = 0.45
        row[f"latent_testing_log_index_{year}"] = 0.25
        row[f"latent_severe_source_{year}"] = 750.0
    for year in (2023, 2025):
        row[f"latent_serology_probability_under5_{year}"] = 0.08
        row[f"latent_serology_probability_school_age_{year}"] = 0.15
        row[f"latent_serology_probability_adult_{year}"] = 0.05
        row[f"latent_serology_immunity_probability_under5_{year}"] = 0.30
        row[f"latent_serology_immunity_probability_school_age_{year}"] = 0.45
        row[f"latent_serology_immunity_probability_adult_{year}"] = 0.35
    return row


def test_all_optional_stream_combinations_are_prespecified() -> None:
    packages = surveillance_packages()
    assert len(packages) == 16
    assert len(set(packages)) == 16
    assert packages[0] == ()
    assert tuple(OPTIONAL_STREAMS) in packages
    assert package_id(()) == "notifications"
    assert package_id(("serology", "age")) == "notifications+age+serology"
    assert structural_contrasts()["transmission_vs_reporting"] == (
        "matched_transmission_truth",
        "matched_reporting_truth",
    )


def test_observation_features_are_reproducible_and_complete() -> None:
    first = simulate_observation_features(
        _latent_row(),
        design=MEASUREMENT_DESIGNS["main"],
        seed=41,
        observation_replicate=1,
    )
    second = simulate_observation_features(
        _latent_row(),
        design=MEASUREMENT_DESIGNS["main"],
        seed=41,
        observation_replicate=1,
    )
    assert first == second
    frame = pd.DataFrame([first])
    assert len(feature_columns_for_package(frame, ())) == 36
    assert len(feature_columns_for_package(frame, OPTIONAL_STREAMS)) == 60
    assert np.isfinite(frame.filter(like="feature_").to_numpy(dtype=float)).all()


def test_group_folds_keep_paired_structures_together() -> None:
    groups = ["AUS:1", "AUS:1", "BRA:2", "BRA:2", "CHN:3"]
    folds = assign_group_folds(groups, n_folds=3)
    assert set(folds) == {"AUS:1", "BRA:2", "CHN:3"}
    assert len(set(folds.values())) == 3


def test_matched_structure_preserves_latent_states_and_notification_target() -> None:
    target = pd.DataFrame(
        {
            "reported_cases_predicted": [100.0, 200.0],
            "reported_cases_under5_predicted": [20.0, 40.0],
            "true_infections_predicted": [1000.0, 2000.0],
        }
    )
    source = pd.DataFrame(
        {
            "reported_cases_predicted": [50.0, 50.0],
            "reported_cases_under5_predicted": [5.0, 10.0],
            "true_infections_predicted": [400.0, 500.0],
        }
    )
    matched = _match_notification_trajectory(target, source)
    np.testing.assert_allclose(matched["reported_cases_predicted"], target["reported_cases_predicted"])
    np.testing.assert_allclose(matched["reported_cases_under5_predicted"], [10.0, 40.0])
    np.testing.assert_allclose(matched["true_infections_predicted"], source["true_infections_predicted"])


def test_cross_validated_discrimination_recovers_separated_signal() -> None:
    rows = []
    rng = np.random.default_rng(7)
    for group_index in range(30):
        for truth_active, scenario_id in ((0, "neutral"), (1, "full_model")):
            for replicate in range(2):
                rows.append(
                    {
                        "iso3": ["AUS", "BRA", "CHN"][group_index % 3],
                        "draw_id": str(group_index),
                        "draw_rank": group_index,
                        "group_id": f"group-{group_index}",
                        "scenario_id": scenario_id,
                        "observation_replicate": replicate,
                        "truth_active": truth_active,
                        "feature_notifications_signal": rng.normal(3.0 * truth_active, 0.25),
                    }
                )
    frame = pd.DataFrame(rows)
    predictions = cross_validated_predictions(
        frame,
        ["feature_notifications_signal"],
        n_folds=5,
    )
    metrics = discrimination_metrics(
        predictions["truth_active"],
        predictions["probability_active"],
    )
    assert metrics["auc"] > 0.99
    assert metrics["balanced_accuracy"] > 0.95
    assert predictions.groupby("group_id")["fold"].nunique().eq(1).all()


def test_leave_one_country_out_predictions_exclude_test_country() -> None:
    rows = []
    rng = np.random.default_rng(17)
    for country_index, iso3 in enumerate(("AUS", "BRA", "CHN", "GBR")):
        for group_index in range(8):
            for truth_active, scenario_id in ((0, "neutral"), (1, "active")):
                rows.append(
                    {
                        "iso3": iso3,
                        "draw_id": str(group_index),
                        "draw_rank": group_index,
                        "group_id": f"{iso3}:{group_index}",
                        "scenario_id": scenario_id,
                        "observation_replicate": 0,
                        "truth_active": truth_active,
                        "feature_notifications_signal": rng.normal(
                            2.0 * truth_active + 0.1 * country_index,
                            0.20,
                        ),
                    }
                )
    predictions = leave_one_country_out_predictions(
        pd.DataFrame(rows),
        ["feature_notifications_signal"],
    )
    assert predictions["iso3"].eq(predictions["holdout_country"]).all()
    assert predictions.groupby("iso3")["fold"].nunique().eq(1).all()
    metrics = discrimination_metrics(
        predictions["truth_active"],
        predictions["probability_active"],
    )
    assert metrics["auc"] > 0.95


def test_shapley_gains_recover_additive_package_values() -> None:
    weights = {"age": 0.03, "testing": 0.07, "severe": 0.05, "serology": 0.10}
    rows = []
    for streams in surveillance_packages():
        rows.append(
            {
                "mechanism": "example",
                "analysis_population": "all_nine",
                "measurement_design": "main",
                "package_streams": "|".join(streams),
                "auc": 0.50 + sum(weights[stream] for stream in streams),
            }
        )
    result = shapley_stream_gains(pd.DataFrame(rows)).set_index("stream")
    for stream, expected in weights.items():
        assert np.isclose(result.loc[stream, "shapley_auc_gain"], expected)


def test_minimum_package_applies_pooled_and_country_rules() -> None:
    package = pd.DataFrame(
        [
            {
                "mechanism": "example",
                "analysis_population": "all_nine",
                "measurement_design": "main",
                "package_id": "notifications",
                "package_streams": "",
                "n_optional_streams": 0,
                "auc": 0.70,
                "auc_lower_95": 0.65,
                "auc_upper_95": 0.75,
                "log_loss": 0.60,
            },
            {
                "mechanism": "example",
                "analysis_population": "all_nine",
                "measurement_design": "main",
                "package_id": "notifications+testing",
                "package_streams": "testing",
                "n_optional_streams": 1,
                "auc": 0.85,
                "auc_lower_95": 0.80,
                "auc_upper_95": 0.90,
                "log_loss": 0.40,
            },
        ]
    )
    country_rows = []
    for package_id_value, success_count in (("notifications", 1), ("notifications+testing", 7)):
        for index in range(9):
            country_rows.append(
                {
                    "mechanism": "example",
                    "analysis_population": "all_nine",
                    "measurement_design": "main",
                    "package_id": package_id_value,
                    "auc": 0.75 if index < success_count else 0.60,
                }
            )
    result = minimum_successful_packages(package, pd.DataFrame(country_rows))
    assert result.loc[0, "recommendation_status"] == "success_rule_met"
    assert result.loc[0, "recommended_package_id"] == "notifications+testing"


def test_precision_frontier_grid_is_complete_and_one_parameter_at_a_time() -> None:
    settings = pd.DataFrame(_settings())
    assert len(settings) == 29
    assert settings.groupby("frontier_type").size().to_dict() == {
        "testing_log_sd": 10,
        "severe_baseline_events": 11,
        "severe_annual_log_sd": 8,
    }
    testing = settings.loc[settings["frontier_type"].eq("testing_log_sd")]
    assert testing["severe_baseline_events"].eq(50.0).all()
    assert testing["severe_log_sd"].eq(0.20).all()
    severe_events = settings.loc[
        settings["frontier_type"].eq("severe_baseline_events")
    ]
    assert severe_events["testing_log_sd"].eq(0.20).all()
    assert severe_events["severe_log_sd"].eq(0.20).all()


def test_precision_frontier_summary_uses_directional_thresholds() -> None:
    rows = []
    for frontier_type, values, successes in (
        ("testing_log_sd", (0.5, 1.0, 1.5), (True, True, False)),
        ("severe_baseline_events", (0.5, 1.0, 2.0), (False, True, True)),
        ("severe_annual_log_sd", (0.5, 1.0, 1.5), (True, True, False)),
    ):
        for value, success in zip(values, successes):
            rows.append(
                {
                    "frontier_type": frontier_type,
                    "parameter_value": value,
                    "parameter_units": "example",
                    "operational_success": success,
                    "conservative_success": success,
                }
            )
    summary = _frontier_summary(pd.DataFrame(rows)).set_index("frontier_type")
    assert summary.loc["testing_log_sd", "conservative_threshold"] == 1.0
    assert summary.loc["severe_baseline_events", "conservative_threshold"] == 1.0
    assert summary.loc["severe_annual_log_sd", "conservative_threshold"] == 1.0
