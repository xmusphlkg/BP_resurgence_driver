from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DATA = ROOT / "manuscript" / "figure_data"
FIGURES = ROOT / "manuscript" / "figures"
SURVEILLANCE_DESIGN_VERSION = "model2c_surveillance_design_v5_20260726"


def test_figure4_matched_path_is_exact_for_both_classes() -> None:
    path = pd.read_csv(FIGURE_DATA / "figure_4_representative_matched_path.csv")
    assert len(path) == 72
    assert path["group_id"].nunique() == 1
    assert path["trajectory_class"].nunique() == 2
    paired = path.pivot(
        index="date",
        columns="trajectory_class",
        values="expected_notifications",
    )
    assert len(paired) == 36
    np.testing.assert_allclose(
        paired["Transmission-dominant"],
        paired["Reporting-dominant"],
        rtol=0,
        atol=1e-10,
    )
    assert (
        path["max_absolute_notification_difference_all_720_pairs"].max() == 0
    )


def test_figure4_latent_ratios_match_manuscript_values() -> None:
    ratios = pd.read_csv(FIGURE_DATA / "figure_4_matched_latent_ratios.csv")
    assert len(ratios) == 720
    assert ratios["group_id"].nunique() == 720
    assert np.isclose(
        ratios["under5_symptomatic_burden_ratio_2024"].median(),
        8.08863419737042,
    )
    assert np.isclose(
        ratios["relative_ascertainment_ratio_2024"].median(),
        0.135287417987238,
    )
    assert set(ratios["surveillance_design_version"]) == {
        SURVEILLANCE_DESIGN_VERSION
    }


def test_figure4_roc_curves_reproduce_frozen_auc_results() -> None:
    roc = pd.read_csv(FIGURE_DATA / "figure_4_grouped_cv_roc.csv")
    package_auc = pd.read_csv(
        FIGURE_DATA / "surveillance_design_package_auc.csv"
    ).set_index("package_id")["auc"]
    expected_packages = set(package_auc.index)
    assert set(roc["package_id"]) == expected_packages
    assert set(roc["validation_scheme"]) == {"grouped_five_fold_oof"}
    assert set(roc["surveillance_design_version"]) == {
        SURVEILLANCE_DESIGN_VERSION
    }
    for package_id, curve in roc.groupby("package_id"):
        curve = curve.sort_values("point_order")
        fpr = curve["false_positive_rate"].to_numpy()
        tpr = curve["true_positive_rate"].to_numpy()
        assert np.all(np.diff(fpr) >= 0)
        assert np.all(np.diff(tpr) >= 0)
        np.testing.assert_allclose([fpr[0], tpr[0]], [0, 0], atol=0, rtol=0)
        np.testing.assert_allclose([fpr[-1], tpr[-1]], [1, 1], atol=0, rtol=0)
        integrated_auc = np.sum(
            np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2
        )
        assert np.isclose(
            integrated_auc,
            package_auc.loc[package_id],
            atol=1e-12,
            rtol=0,
        )
        assert curve["n_country_draw_groups"].eq(720).all()
        assert curve["n_observations"].eq(4320).all()


def test_figure4_package_matrix_encodes_all_stream_combinations() -> None:
    performance = pd.read_csv(
        FIGURE_DATA / "figure_4_package_performance.csv"
    )
    membership_columns = [
        "includes_notifications",
        "includes_age",
        "includes_testing",
        "includes_severe",
        "includes_serology",
    ]
    grouped_cv = performance[
        performance["validation"].eq("Grouped 5-fold CV")
    ]

    assert len(grouped_cv) == 16
    assert len(performance) == 16
    assert grouped_cv["package_id"].nunique() == 16
    assert set(grouped_cv["panel"]) == {"a"}
    assert grouped_cv["includes_notifications"].all()
    assert not grouped_cv[membership_columns].duplicated().any()
    assert (
        grouped_cv["auc_lower_95"]
        <= grouped_cv["auc"]
    ).all()
    assert (
        grouped_cv["auc"]
        <= grouped_cv["auc_upper_95"]
    ).all()


def test_figure4_stream_gains_match_frozen_analysis() -> None:
    gains = pd.read_csv(
        FIGURE_DATA / "surveillance_design_shapley_auc_gain.csv"
    ).set_index("stream")
    assert set(gains.index) == {"age", "testing", "severe", "serology"}
    assert np.isclose(gains.loc["severe", "shapley_auc_gain"], 0.2091464298982624)
    assert np.isclose(gains.loc["testing", "shapley_auc_gain"], 0.2071436506915866)
    assert np.isclose(gains.loc["serology", "shapley_auc_gain"], 0.0699181420038865)
    assert np.isclose(gains.loc["age", "shapley_auc_gain"], 0.0103669231538637)
    assert set(gains["surveillance_design_version"]) == {
        SURVEILLANCE_DESIGN_VERSION
    }


def test_figure4_country_holdout_grid_is_complete() -> None:
    holdout = pd.read_csv(FIGURE_DATA / "figure_4_country_holdout_auc.csv")
    assert len(holdout) == 54
    assert set(holdout["panel"]) == {"b"}
    assert holdout["iso3"].nunique() == 9
    assert holdout["package_id"].nunique() == 6
    assert not holdout.duplicated(["iso3", "package_id"]).any()
    selected = holdout[
        holdout["package_id"].isin(
            ["notifications+testing", "notifications+severe"]
        )
    ]
    assert selected.groupby("package_id")["auc"].min().ge(0.70).all()
    assert set(holdout["surveillance_design_version"]) == {
        SURVEILLANCE_DESIGN_VERSION
    }


def test_figure4_precision_robustness_is_complete() -> None:
    precision = pd.read_csv(
        FIGURE_DATA / "figure_4_precision_robustness_auc.csv"
    )
    assert len(precision) == 6
    assert set(precision["measurement_design"]) == {
        "low_precision",
        "main",
        "high_precision",
    }
    assert set(precision["package_id"]) == {
        "notifications+testing",
        "notifications+severe",
    }
    low = precision[precision["measurement_design"].eq("low_precision")].set_index(
        "package_id"
    )
    assert np.isclose(low.loc["notifications+testing", "auc"], 0.9776410322)
    assert np.isclose(low.loc["notifications+severe", "auc"], 0.9686469059)


def test_figure4_cross_loading_surface_is_complete() -> None:
    surface = pd.read_csv(
        FIGURE_DATA / "figure_4_cross_loading_surface.csv"
    )
    expected_levels = {0.0, 0.25, 0.50, 0.75, 1.0}
    assert len(surface) == 25
    assert set(surface["panel"]) == {"e_surface"}
    assert set(surface["testing_cross_loading_fraction"]) == expected_levels
    assert set(surface["severe_cross_loading_fraction"]) == expected_levels
    assert not surface.duplicated(
        [
            "testing_cross_loading_fraction",
            "severe_cross_loading_fraction",
        ]
    ).any()
    assert surface["package_id"].eq("notifications+testing+severe").all()
    indexed = surface.set_index(
        [
            "testing_cross_loading_fraction",
            "severe_cross_loading_fraction",
        ]
    )
    assert indexed.loc[(0.0, 0.0), "grouped_cv_auc"] > 0.99
    assert indexed.loc[(1.0, 0.0), "grouped_cv_auc"] > 0.98
    assert indexed.loc[(0.0, 1.0), "grouped_cv_auc"] > 0.98
    assert 0.45 <= indexed.loc[(1.0, 1.0), "grouped_cv_auc"] <= 0.55
    assert (
        surface["grouped_cv_auc_lower_95"]
        <= surface["grouped_cv_auc"]
    ).all()
    assert (
        surface["grouped_cv_auc"]
        <= surface["grouped_cv_auc_upper_95"]
    ).all()


def test_figure4_export_bundle_exists() -> None:
    for suffix in (".png", ".pdf"):
        path = FIGURES / f"figure_4_surveillance_auc{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1000
