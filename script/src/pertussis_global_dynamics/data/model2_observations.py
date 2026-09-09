from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pertussis_global_dynamics.data.context import CountryContext
from pertussis_global_dynamics.model2c_windows import (
    MODEL2C_OBSERVATION_END_YEAR,
    MODEL2C_OBSERVATION_START_YEAR,
)


def clean_text_value(value: Any, default: str = "not_available") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "nat"}:
        return default
    return text


def default_detection_prior_row() -> dict[str, Any]:
    return {
        "ode_relative_detection_prior_mean": 1.0,
        "ode_relative_detection_prior_lower": 0.75,
        "ode_relative_detection_prior_upper": 1.35,
        "ode_testing_denominator_series_relative_detection_prior_factor": 1.0,
        "ode_testing_denominator_series_relative_detection_prior_factor_lower": 0.85,
        "ode_testing_denominator_series_relative_detection_prior_factor_upper": 1.15,
        "ode_testing_denominator_series_years_available": np.nan,
        "ode_testing_denominator_series_first_year": np.nan,
        "ode_testing_denominator_series_last_year": np.nan,
        "ode_testing_denominator_series_support_level": "not_available",
        "ode_testing_denominator_series_source_ids": "not_available",
        "ode_testing_denominator_series_source_type": "not_available",
        "ode_detection_effect_direction": "no_external_period_match",
        "ode_detection_evidence_strength": "none",
        "ode_surveillance_regime": "not_available",
        "ode_primary_diagnostic_methods": "not_available",
        "ode_case_definition_or_reporting_change": "No country-month diagnostic timeline row matched; relative detection prior defaults to 1.0.",
        "ode_detection_source_ids": "not_available",
        "ode_detection_prior_source": "not_available",
        "ode_detection_prior_source_type": "not_available",
        "ode_detection_prior_notes": "No matched observation-process prior.",
        "ode_testing_denominator_years_available": np.nan,
        "ode_testing_denominator_first_year": np.nan,
        "ode_testing_denominator_last_year": np.nan,
        "ode_testing_denominator_support_level": "not_available",
        "ode_testing_denominator_source_ids": "not_available",
        "ode_surveillance_outcome_source_type": "not_available",
        "ode_surveillance_outcome_source_ids": "not_available",
        "ode_surveillance_outcome_support_level": "not_available",
        "ode_surveillance_outcome_notes": "No matched official surveillance outcome summary.",
    }


def _testing_series_priority(row: pd.Series) -> int:
    text = " ".join(
        [
            clean_text_value(row.get("source_type", "not_available"), default=""),
            clean_text_value(row.get("notes", "not_available"), default=""),
            clean_text_value(row.get("source", "not_available"), default=""),
            clean_text_value(row.get("source_ids", "not_available"), default=""),
        ]
    ).casefold()
    if "single-study" in text or "single study" in text:
        return 0
    if "multi-year" in text or "annual report" in text or "denominator-data project" in text or "series" in text:
        return 2
    return 1


def direct_testing_denominator_series_table(surveillance: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "testing_denominator",
        "test_positivity_percent",
        "direct_testing_series_priority",
        "direct_testing_series_years_available",
        "direct_testing_series_first_year",
        "direct_testing_series_last_year",
        "direct_testing_series_support_level",
        "direct_testing_series_source_ids",
        "direct_testing_series_source_type",
        "direct_testing_series_notes",
        "direct_testing_series_log_testing_denominator",
        "direct_testing_series_centered_log_testing_denominator",
        "direct_testing_series_relative_detection_prior_factor",
        "direct_testing_series_relative_detection_prior_factor_lower",
        "direct_testing_series_relative_detection_prior_factor_upper",
    ]
    if surveillance.empty or "testing_denominator" not in surveillance.columns:
        return pd.DataFrame(columns=columns)

    frame = surveillance.copy()
    frame["iso3"] = frame.get("iso3", pd.Series(dtype=str)).astype(str).str.upper()
    frame["country"] = frame.get("country", frame["iso3"]).fillna(frame["iso3"]).astype(str).str.strip()
    frame["year"] = pd.to_numeric(frame.get("year"), errors="coerce")
    frame["testing_denominator"] = pd.to_numeric(frame.get("testing_denominator"), errors="coerce")
    frame["test_positivity_percent"] = pd.to_numeric(frame.get("test_positivity_percent"), errors="coerce")
    frame = frame.loc[frame["year"].notna() & frame["testing_denominator"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    for column in ["source_type", "notes", "source_ids"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("not_available").astype(str)
        else:
            frame[column] = "not_available"
    frame["direct_testing_series_priority"] = frame.apply(_testing_series_priority, axis=1)
    frame["direct_testing_series_log_testing_denominator"] = np.log1p(frame["testing_denominator"].clip(lower=0.0))

    selected = (
        frame.sort_values(
            by=["iso3", "year", "direct_testing_series_priority", "testing_denominator", "source_ids"],
            ascending=[True, True, False, False, True],
            kind="stable",
        )
        .drop_duplicates(subset=["iso3", "year"], keep="first")
        .reset_index(drop=True)
    )
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["direct_testing_series_source_ids"] = selected["source_ids"]
    selected["direct_testing_series_source_type"] = selected["source_type"]
    selected["direct_testing_series_notes"] = selected["notes"]

    rows: list[dict[str, Any]] = []
    for (iso3, country), group in selected.groupby(["iso3", "country"], sort=True):
        group = group.sort_values("year").copy()
        years_available = int(group["year"].nunique())
        if years_available > 1:
            signal_center = float(group["direct_testing_series_log_testing_denominator"].median())
            centered = group["direct_testing_series_log_testing_denominator"] - signal_center
            factor = np.exp(0.30 * centered.to_numpy(dtype=float))
        else:
            signal_center = np.nan
            factor = np.ones(len(group), dtype=float)
        group["direct_testing_series_centered_log_testing_denominator"] = (
            group["direct_testing_series_log_testing_denominator"] - signal_center if np.isfinite(signal_center) else np.nan
        )
        group["direct_testing_series_relative_detection_prior_factor"] = np.clip(factor, 0.50, 2.50)
        group["direct_testing_series_relative_detection_prior_factor_lower"] = np.clip(
            group["direct_testing_series_relative_detection_prior_factor"] * np.exp(-0.15), 0.40, 3.00
        )
        group["direct_testing_series_relative_detection_prior_factor_upper"] = np.clip(
            group["direct_testing_series_relative_detection_prior_factor"] * np.exp(0.15), 0.50, 3.50
        )
        group["direct_testing_series_support_level"] = (
            "multi_year_direct_testing_denominator_series_support"
            if years_available > 1
            else "single_year_direct_testing_denominator_summary"
        )
        group["direct_testing_series_years_available"] = years_available
        group["direct_testing_series_first_year"] = int(group["year"].min())
        group["direct_testing_series_last_year"] = int(group["year"].max())
        rows.extend(
            group.loc[:, columns].assign(
                direct_testing_series_priority=group["direct_testing_series_priority"].astype(int),
                direct_testing_series_years_available=group["direct_testing_series_years_available"].astype(int),
                direct_testing_series_first_year=group["direct_testing_series_first_year"].astype(int),
                direct_testing_series_last_year=group["direct_testing_series_last_year"].astype(int),
            ).to_dict("records")
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["iso3", "year"]).reset_index(drop=True)


def summarize_direct_testing_denominator_series_table(series: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "direct_testing_series_rows",
        "direct_testing_series_years_available",
        "direct_testing_series_first_year",
        "direct_testing_series_last_year",
        "direct_testing_series_support_level",
        "direct_testing_series_source_ids",
        "direct_testing_series_source_type",
        "direct_testing_series_testing_denominator_median",
        "direct_testing_series_testing_denominator_min",
        "direct_testing_series_testing_denominator_max",
        "direct_testing_series_relative_detection_prior_factor_median",
        "direct_testing_series_relative_detection_prior_factor_min",
        "direct_testing_series_relative_detection_prior_factor_max",
        "direct_testing_series_test_positivity_percent_median",
        "direct_testing_series_test_positivity_percent_min",
        "direct_testing_series_test_positivity_percent_max",
    ]
    if series.empty:
        return pd.DataFrame(columns=columns)

    frame = series.copy()
    for column in [
        "year",
        "testing_denominator",
        "test_positivity_percent",
        "direct_testing_series_priority",
        "direct_testing_series_years_available",
        "direct_testing_series_first_year",
        "direct_testing_series_last_year",
        "direct_testing_series_relative_detection_prior_factor",
        "direct_testing_series_relative_detection_prior_factor_lower",
        "direct_testing_series_relative_detection_prior_factor_upper",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    def _collapse(series: pd.Series) -> str:
        values = [value for value in series.dropna().astype(str) if value and value != "not_available"]
        return "; ".join(dict.fromkeys(values)) if values else "not_available"

    rows: list[dict[str, Any]] = []
    for (iso3, country), group in frame.groupby(["iso3", "country"], sort=True):
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "direct_testing_series_rows": int(len(group)),
                "direct_testing_series_years_available": int(group["year"].nunique()),
                "direct_testing_series_first_year": int(group["year"].min()),
                "direct_testing_series_last_year": int(group["year"].max()),
                "direct_testing_series_support_level": _collapse(group["direct_testing_series_support_level"]),
                "direct_testing_series_source_ids": _collapse(group["direct_testing_series_source_ids"]),
                "direct_testing_series_source_type": _collapse(group["direct_testing_series_source_type"]),
                "direct_testing_series_testing_denominator_median": float(group["testing_denominator"].median()),
                "direct_testing_series_testing_denominator_min": float(group["testing_denominator"].min()),
                "direct_testing_series_testing_denominator_max": float(group["testing_denominator"].max()),
                "direct_testing_series_relative_detection_prior_factor_median": float(
                    group["direct_testing_series_relative_detection_prior_factor"].median()
                ),
                "direct_testing_series_relative_detection_prior_factor_min": float(
                    group["direct_testing_series_relative_detection_prior_factor"].min()
                ),
                "direct_testing_series_relative_detection_prior_factor_max": float(
                    group["direct_testing_series_relative_detection_prior_factor"].max()
                ),
                "direct_testing_series_test_positivity_percent_median": float(group["test_positivity_percent"].median())
                if group["test_positivity_percent"].notna().any()
                else np.nan,
                "direct_testing_series_test_positivity_percent_min": float(group["test_positivity_percent"].min())
                if group["test_positivity_percent"].notna().any()
                else np.nan,
                "direct_testing_series_test_positivity_percent_max": float(group["test_positivity_percent"].max())
                if group["test_positivity_percent"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["iso3"], na_position="last").reset_index(drop=True)


def monthly_detection_prior_table(
    context: CountryContext,
    start_year: int = MODEL2C_OBSERVATION_START_YEAR,
    end_year: int = MODEL2C_OBSERVATION_END_YEAR,
    include_testing_denominator: bool = True,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "month",
        *default_detection_prior_row().keys(),
        "ode_observation_prior_method",
    ]
    rows: list[dict[str, Any]] = []
    timeline = context.diagnostic_timeline.copy()
    surveillance = context.surveillance_outcomes.copy()
    if not timeline.empty:
        timeline["period_start"] = pd.to_datetime(timeline["period_start"], errors="coerce")
        timeline["period_end"] = pd.to_datetime(timeline["period_end"], errors="coerce")
        for column in [
            "relative_detection_prior_mean",
            "relative_detection_prior_lower",
            "relative_detection_prior_upper",
        ]:
            timeline[column] = pd.to_numeric(timeline.get(column, np.nan), errors="coerce")
        timeline = timeline.dropna(subset=["period_start", "period_end"]).sort_values("period_start")

    testing_summary: dict[str, Any] = default_detection_prior_row()
    testing_series = pd.DataFrame()
    if not surveillance.empty and "iso3" in surveillance:
        surveillance = surveillance.copy()
        surveillance["year"] = pd.to_numeric(surveillance.get("year"), errors="coerce")
        for column in [
            "case_count",
            "hospitalizations",
            "total_deaths",
            "infant_deaths",
            "testing_denominator",
            "test_positivity_percent",
            "under6m_hospitalization_percent",
            "total_hospitalization_percent",
        ]:
            if column in surveillance:
                surveillance[column] = pd.to_numeric(surveillance[column], errors="coerce")
            else:
                surveillance[column] = np.nan
        surveillance["has_testing_denominator"] = surveillance[["testing_denominator"]].notna().any(axis=1)
        surveillance["has_severe_outcome"] = surveillance[["hospitalizations", "total_deaths", "infant_deaths"]].notna().any(axis=1)
        severe_source_ids = ";".join(
            sorted(
                {
                    text
                    for text in surveillance.get("source_ids", pd.Series(dtype=str)).dropna().astype(str)
                    if text and text != "nan"
                }
            )
        )
        if severe_source_ids:
            testing_summary["ode_surveillance_outcome_source_type"] = ";".join(
                sorted(
                    {
                        text
                        for text in surveillance.get("source_type", pd.Series(dtype=str)).dropna().astype(str)
                        if text and text != "nan"
                    }
                )
            ) or "not_available"
            testing_summary["ode_surveillance_outcome_support_level"] = "official_surveillance_outcome_support"
            testing_summary["ode_surveillance_outcome_source_ids"] = severe_source_ids
            testing_summary["ode_surveillance_outcome_notes"] = (
                "Official surveillance outcome summaries are available; direct testing-denominator series are "
                + ("present" if include_testing_denominator else "not used in this prior mode")
                + "."
            )
        if include_testing_denominator:
            testing_series = direct_testing_denominator_series_table(surveillance)
            if not testing_series.empty:
                testing_years = testing_series.loc[testing_series["year"].between(start_year, end_year)].copy()
                if not testing_years.empty:
                    years_available = int(testing_years["year"].nunique())
                    source_ids = ";".join(
                        sorted(
                            {
                                text
                                for text in testing_years.get("direct_testing_series_source_ids", pd.Series(dtype=str))
                                .dropna()
                                .astype(str)
                                if text and text != "nan"
                            }
                        )
                    ) or "not_available"
                    testing_summary.update(
                        {
                            "ode_testing_denominator_years_available": years_available,
                            "ode_testing_denominator_first_year": int(testing_years["year"].min()),
                            "ode_testing_denominator_last_year": int(testing_years["year"].max()),
                            "ode_testing_denominator_support_level": (
                                "multi_year_direct_testing_denominator_series_support"
                                if years_available > 1
                                else "single_year_direct_testing_denominator_summary"
                            ),
                            "ode_testing_denominator_source_ids": source_ids,
                        }
                    )

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            prior = default_detection_prior_row()
            testing_factor = 1.0
            testing_factor_lower = 0.85
            testing_factor_upper = 1.15
            testing_series_years_available = np.nan
            testing_series_first_year = np.nan
            testing_series_last_year = np.nan
            testing_series_support_level = "not_available"
            testing_series_source_ids = "not_available"
            testing_series_source_type = "not_available"
            prior.update(
                {
                    "ode_testing_denominator_years_available": testing_summary["ode_testing_denominator_years_available"],
                    "ode_testing_denominator_first_year": testing_summary["ode_testing_denominator_first_year"],
                    "ode_testing_denominator_last_year": testing_summary["ode_testing_denominator_last_year"],
                    "ode_testing_denominator_support_level": testing_summary["ode_testing_denominator_support_level"],
                    "ode_testing_denominator_source_ids": testing_summary["ode_testing_denominator_source_ids"],
                    "ode_surveillance_outcome_source_type": testing_summary["ode_surveillance_outcome_source_type"],
                    "ode_surveillance_outcome_source_ids": testing_summary["ode_surveillance_outcome_source_ids"],
                    "ode_surveillance_outcome_support_level": testing_summary["ode_surveillance_outcome_support_level"],
                    "ode_surveillance_outcome_notes": testing_summary["ode_surveillance_outcome_notes"],
                }
            )
            if not timeline.empty:
                date = pd.Timestamp(year=year, month=month, day=1)
                matched = timeline.loc[(timeline["period_start"].le(date)) & (timeline["period_end"].ge(date))]
                if not matched.empty:
                    item = matched.sort_values("period_start").iloc[-1]
                    mean = float(item.get("relative_detection_prior_mean", np.nan))
                    lower = float(item.get("relative_detection_prior_lower", np.nan))
                    upper = float(item.get("relative_detection_prior_upper", np.nan))
                    prior.update(
                        {
                            "ode_relative_detection_prior_mean": float(np.clip(mean if np.isfinite(mean) else 1.0, 0.05, 12.0)),
                            "ode_relative_detection_prior_lower": float(np.clip(lower if np.isfinite(lower) else 0.75, 0.02, 12.0)),
                            "ode_relative_detection_prior_upper": float(np.clip(upper if np.isfinite(upper) else 1.35, 0.02, 15.0)),
                            "ode_detection_effect_direction": clean_text_value(item.get("effect_direction", "not_available")),
                            "ode_detection_evidence_strength": clean_text_value(item.get("evidence_strength", "not_available")),
                            "ode_surveillance_regime": clean_text_value(item.get("surveillance_regime", "not_available")),
                            "ode_primary_diagnostic_methods": clean_text_value(item.get("primary_diagnostic_methods", "not_available")),
                            "ode_case_definition_or_reporting_change": clean_text_value(item.get("case_definition_or_reporting_change", "not_available")),
                            "ode_detection_source_ids": clean_text_value(item.get("source_ids", "not_available")),
                            "ode_detection_prior_source": clean_text_value(item.get("source", "not_available")),
                            "ode_detection_prior_source_type": clean_text_value(item.get("source_type", "not_available")),
                            "ode_detection_prior_notes": clean_text_value(item.get("notes", "not_available")),
                        }
                    )
            if include_testing_denominator and not testing_series.empty:
                matched_testing = testing_series.loc[testing_series["year"].eq(year)].copy()
                if not matched_testing.empty:
                    matched_testing = matched_testing.sort_values(
                        ["direct_testing_series_priority", "testing_denominator", "direct_testing_series_source_ids"],
                        ascending=[False, False, True],
                        kind="stable",
                    ).iloc[0]
                    testing_factor = float(
                        np.clip(
                            matched_testing["direct_testing_series_relative_detection_prior_factor"],
                            0.50,
                            2.50,
                        )
                    )
                    testing_factor_lower = float(
                        np.clip(
                            matched_testing["direct_testing_series_relative_detection_prior_factor_lower"],
                            0.40,
                            3.00,
                        )
                    )
                    testing_factor_upper = float(
                        np.clip(
                            matched_testing["direct_testing_series_relative_detection_prior_factor_upper"],
                            0.50,
                            3.50,
                        )
                    )
                    testing_series_years_available = float(matched_testing["direct_testing_series_years_available"])
                    testing_series_first_year = float(matched_testing["direct_testing_series_first_year"])
                    testing_series_last_year = float(matched_testing["direct_testing_series_last_year"])
                    testing_series_support_level = clean_text_value(
                        matched_testing.get("direct_testing_series_support_level"), default="not_available"
                    )
                    testing_series_source_ids = clean_text_value(
                        matched_testing.get("direct_testing_series_source_ids"), default="not_available"
                    )
                    testing_series_source_type = clean_text_value(
                        matched_testing.get("direct_testing_series_source_type"), default="not_available"
                    )
            prior["ode_testing_denominator_series_relative_detection_prior_factor"] = float(
                np.clip(testing_factor, 0.50, 2.50)
            )
            prior["ode_testing_denominator_series_relative_detection_prior_factor_lower"] = float(
                np.clip(testing_factor_lower, 0.40, 3.00)
            )
            prior["ode_testing_denominator_series_relative_detection_prior_factor_upper"] = float(
                np.clip(testing_factor_upper, 0.50, 3.50)
            )
            prior["ode_testing_denominator_series_years_available"] = testing_series_years_available
            prior["ode_testing_denominator_series_first_year"] = testing_series_first_year
            prior["ode_testing_denominator_series_last_year"] = testing_series_last_year
            prior["ode_testing_denominator_series_support_level"] = testing_series_support_level
            prior["ode_testing_denominator_series_source_ids"] = testing_series_source_ids
            prior["ode_testing_denominator_series_source_type"] = testing_series_source_type
            prior["ode_relative_detection_prior_mean"] = float(
                np.clip(prior["ode_relative_detection_prior_mean"] * testing_factor, 0.05, 12.0)
            )
            prior["ode_relative_detection_prior_lower"] = float(
                np.clip(prior["ode_relative_detection_prior_lower"] * testing_factor, 0.02, 12.0)
            )
            prior["ode_relative_detection_prior_upper"] = float(
                np.clip(prior["ode_relative_detection_prior_upper"] * testing_factor, 0.02, 15.0)
            )
            prior["ode_observation_prior_method"] = (
                "Country-month observation-process prior from the curated surveillance diagnostic timeline"
                + (
                    " and official surveillance outcome summaries where available"
                    if testing_summary["ode_surveillance_outcome_support_level"] != "not_available"
                    else ""
                )
                + (
                    "; selected direct testing-denominator series modulate the year-specific detection prior"
                    if testing_series_support_level != "not_available"
                    else ""
                )
                + "; "
                + "the prior multiplies the ODE-reported cases before age and residual month-level reporting calibration."
            )
            rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "year": year,
                    "month": month,
                    **prior,
                    "ode_observation_prior_method": prior["ode_observation_prior_method"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def summarize_monthly_detection_prior_table(prior: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "months_covered",
        "years_covered",
        "year_min",
        "year_max",
        "matched_months",
        "matched_month_fraction",
        "detection_prior_mean_median",
        "detection_prior_mean_min",
        "detection_prior_mean_max",
        "detection_prior_lower_median",
        "detection_prior_upper_median",
        "surveillance_regime_count",
        "surveillance_regime_summary",
        "effect_direction_summary",
        "evidence_strength_summary",
        "testing_denominator_support_level",
        "testing_denominator_series_rows",
        "testing_denominator_years_available",
        "testing_denominator_first_year",
        "testing_denominator_last_year",
        "testing_denominator_series_support_level",
        "testing_denominator_series_years_available",
        "testing_denominator_series_first_year",
        "testing_denominator_series_last_year",
        "testing_denominator_series_source_ids",
        "testing_denominator_series_source_type",
        "testing_denominator_series_relative_detection_prior_factor_median",
        "testing_denominator_series_relative_detection_prior_factor_min",
        "testing_denominator_series_relative_detection_prior_factor_max",
        "surveillance_outcome_support_level",
        "observation_prior_method",
    ]
    if prior.empty:
        return pd.DataFrame(columns=columns)
    frame = prior.copy()
    for column in [
        "year",
        "month",
        "ode_relative_detection_prior_mean",
        "ode_relative_detection_prior_lower",
        "ode_relative_detection_prior_upper",
        "ode_testing_denominator_years_available",
        "ode_testing_denominator_first_year",
        "ode_testing_denominator_last_year",
        "ode_testing_denominator_series_relative_detection_prior_factor",
        "ode_testing_denominator_series_relative_detection_prior_factor_lower",
        "ode_testing_denominator_series_relative_detection_prior_factor_upper",
        "ode_testing_denominator_series_years_available",
        "ode_testing_denominator_series_first_year",
        "ode_testing_denominator_series_last_year",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    text_columns = [
        "ode_detection_effect_direction",
        "ode_detection_evidence_strength",
        "ode_surveillance_regime",
        "ode_observation_prior_method",
        "ode_testing_denominator_support_level",
        "ode_testing_denominator_series_support_level",
        "ode_surveillance_outcome_support_level",
        "ode_detection_source_ids",
        "ode_detection_prior_source_type",
        "ode_testing_denominator_series_source_ids",
        "ode_testing_denominator_series_source_type",
    ]
    for column in text_columns:
        frame[column] = frame.get(column, pd.Series(dtype=str)).fillna("not_available").astype(str).str.strip()
    matched = frame.loc[
        ~frame["ode_detection_source_ids"].eq("not_available") | ~frame["ode_detection_prior_source_type"].eq("not_available")
    ].copy()

    def _collapse(series: pd.Series) -> str:
        values = [value for value in series.dropna().astype(str) if value and value != "not_available"]
        return "; ".join(dict.fromkeys(values)) if values else "not_available"

    rows: list[dict[str, object]] = []
    for (iso3, country), group in frame.groupby(["iso3", "country"], dropna=False):
        group = group.copy()
        matched_group = matched.loc[matched["iso3"].eq(iso3) & matched["country"].eq(country)].copy()
        year = pd.to_numeric(group["year"], errors="coerce").dropna()
        months_covered = int(group.shape[0])
        matched_months = int(matched_group.shape[0])
        rows.append(
            {
                "iso3": iso3,
                "country": country,
                "months_covered": months_covered,
                "years_covered": int(year.nunique()) if not year.empty else 0,
                "year_min": int(year.min()) if not year.empty else pd.NA,
                "year_max": int(year.max()) if not year.empty else pd.NA,
                "matched_months": matched_months,
                "matched_month_fraction": float(matched_months / months_covered) if months_covered else np.nan,
                "detection_prior_mean_median": float(group["ode_relative_detection_prior_mean"].median()),
                "detection_prior_mean_min": float(group["ode_relative_detection_prior_mean"].min()),
                "detection_prior_mean_max": float(group["ode_relative_detection_prior_mean"].max()),
                "detection_prior_lower_median": float(group["ode_relative_detection_prior_lower"].median()),
                "detection_prior_upper_median": float(group["ode_relative_detection_prior_upper"].median()),
                "surveillance_regime_count": int(group["ode_surveillance_regime"].replace("not_available", pd.NA).dropna().nunique()),
                "surveillance_regime_summary": _collapse(group["ode_surveillance_regime"]),
                "effect_direction_summary": _collapse(group["ode_detection_effect_direction"]),
                "evidence_strength_summary": _collapse(group["ode_detection_evidence_strength"]),
                "testing_denominator_support_level": _collapse(group["ode_testing_denominator_support_level"]),
                "testing_denominator_series_rows": int(group["ode_testing_denominator_series_years_available"].notna().sum()),
                "testing_denominator_years_available": int(group["ode_testing_denominator_years_available"].dropna().max())
                if group["ode_testing_denominator_years_available"].notna().any()
                else pd.NA,
                "testing_denominator_first_year": int(group["ode_testing_denominator_first_year"].dropna().min())
                if group["ode_testing_denominator_first_year"].notna().any()
                else pd.NA,
                "testing_denominator_last_year": int(group["ode_testing_denominator_last_year"].dropna().max())
                if group["ode_testing_denominator_last_year"].notna().any()
                else pd.NA,
                "testing_denominator_series_support_level": _collapse(group["ode_testing_denominator_series_support_level"]),
                "testing_denominator_series_years_available": int(group["ode_testing_denominator_series_years_available"].dropna().max())
                if group["ode_testing_denominator_series_years_available"].notna().any()
                else pd.NA,
                "testing_denominator_series_first_year": int(group["ode_testing_denominator_series_first_year"].dropna().min())
                if group["ode_testing_denominator_series_first_year"].notna().any()
                else pd.NA,
                "testing_denominator_series_last_year": int(group["ode_testing_denominator_series_last_year"].dropna().max())
                if group["ode_testing_denominator_series_last_year"].notna().any()
                else pd.NA,
                "testing_denominator_series_source_ids": _collapse(group["ode_testing_denominator_series_source_ids"]),
                "testing_denominator_series_source_type": _collapse(group.get("ode_testing_denominator_series_source_type", pd.Series(dtype=str))),
                "testing_denominator_series_relative_detection_prior_factor_median": float(
                    group["ode_testing_denominator_series_relative_detection_prior_factor"].median()
                ),
                "testing_denominator_series_relative_detection_prior_factor_min": float(
                    group["ode_testing_denominator_series_relative_detection_prior_factor"].min()
                ),
                "testing_denominator_series_relative_detection_prior_factor_max": float(
                    group["ode_testing_denominator_series_relative_detection_prior_factor"].max()
                ),
                "surveillance_outcome_support_level": _collapse(group["ode_surveillance_outcome_support_level"]),
                "observation_prior_method": _collapse(group["ode_observation_prior_method"]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["iso3"], na_position="last").reset_index(drop=True)


def default_pathogen_prior_row() -> dict[str, Any]:
    return {
        "ode_pathogen_pressure_prior_mean": 1.0,
        "ode_pathogen_pressure_prior_lower": 0.85,
        "ode_pathogen_pressure_prior_upper": 1.20,
        "ode_pathogen_regime": "no_external_period_match",
        "ode_pathogen_features": "not_available",
        "ode_macrolide_resistance_level": "not_available",
        "ode_vaccine_escape_level": "not_available",
        "ode_pathogen_evidence_strength": "none",
        "ode_pathogen_source_ids": "not_available",
        "ode_pathogen_prior_source": "not_available",
        "ode_pathogen_prior_source_type": "not_available",
        "ode_pathogen_prior_notes": "No matched pathogen strain or AMR prior.",
    }


def monthly_pathogen_prior_table(
    context: CountryContext,
    start_year: int = MODEL2C_OBSERVATION_START_YEAR,
    end_year: int = MODEL2C_OBSERVATION_END_YEAR,
) -> pd.DataFrame:
    columns = [
        "iso3",
        "country",
        "year",
        "month",
        *default_pathogen_prior_row().keys(),
        "ode_pathogen_prior_method",
    ]
    rows: list[dict[str, Any]] = []
    timeline = context.pathogen_timeline.copy()
    if not timeline.empty:
        timeline["period_start"] = pd.to_datetime(timeline["period_start"], errors="coerce")
        timeline["period_end"] = pd.to_datetime(timeline["period_end"], errors="coerce")
        for column in [
            "transmission_prior_mean",
            "transmission_prior_lower",
            "transmission_prior_upper",
        ]:
            timeline[column] = pd.to_numeric(timeline.get(column, np.nan), errors="coerce")
        timeline = timeline.dropna(subset=["period_start", "period_end"]).sort_values("period_start")

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            prior = default_pathogen_prior_row()
            if not timeline.empty:
                date = pd.Timestamp(year=year, month=month, day=1)
                matched = timeline.loc[(timeline["period_start"].le(date)) & (timeline["period_end"].ge(date))]
                if not matched.empty:
                    item = matched.sort_values("period_start").iloc[-1]
                    mean = float(item.get("transmission_prior_mean", np.nan))
                    lower = float(item.get("transmission_prior_lower", np.nan))
                    upper = float(item.get("transmission_prior_upper", np.nan))
                    prior = {
                        "ode_pathogen_pressure_prior_mean": float(np.clip(mean if np.isfinite(mean) else 1.0, 0.50, 4.00)),
                        "ode_pathogen_pressure_prior_lower": float(np.clip(lower if np.isfinite(lower) else 0.85, 0.30, 4.00)),
                        "ode_pathogen_pressure_prior_upper": float(np.clip(upper if np.isfinite(upper) else 1.20, 0.30, 5.00)),
                        "ode_pathogen_regime": clean_text_value(item.get("pathogen_regime", "not_available")),
                        "ode_pathogen_features": clean_text_value(item.get("antigenic_or_resistance_features", "not_available")),
                        "ode_macrolide_resistance_level": clean_text_value(item.get("macrolide_resistance_level", "not_available")),
                        "ode_vaccine_escape_level": clean_text_value(item.get("vaccine_escape_level", "not_available")),
                        "ode_pathogen_evidence_strength": clean_text_value(item.get("evidence_strength", "not_available")),
                        "ode_pathogen_source_ids": clean_text_value(item.get("source_ids", "not_available")),
                        "ode_pathogen_prior_source": clean_text_value(item.get("source", "not_available")),
                        "ode_pathogen_prior_source_type": clean_text_value(item.get("source_type", "not_available")),
                        "ode_pathogen_prior_notes": clean_text_value(item.get("notes", "not_available")),
                    }
            rows.append(
                {
                    "iso3": context.iso3,
                    "country": context.country,
                    "year": year,
                    "month": month,
                    **prior,
                    "ode_pathogen_prior_method": (
                        "Country-month pathogen pressure prior from curated strain/AMR evidence; "
                        "the fitted ode_pathogen_drift_scale attenuates or amplifies this prior in the ODE transmission mapping."
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def monthly_observed_cases(context: CountryContext) -> pd.DataFrame:
    observed = context.time_resolved.copy()
    if observed.empty:
        return pd.DataFrame(columns=["iso3", "country", "year", "month", "observed_cases"])
    observed["date_start"] = pd.to_datetime(observed["date_start"], errors="coerce")
    observed["year"] = pd.to_numeric(observed["year"], errors="coerce").astype("Int64")
    observed["month"] = pd.to_numeric(observed["month"], errors="coerce").fillna(observed["date_start"].dt.month)
    observed["cases_total"] = pd.to_numeric(observed["cases_total"], errors="coerce")
    observed = observed.dropna(subset=["year", "month", "cases_total"]).copy()
    if observed.empty:
        return pd.DataFrame(columns=["iso3", "country", "year", "month", "observed_cases"])
    out = observed.groupby(["iso3", "country", "year", "month"], as_index=False).agg(observed_cases=("cases_total", "sum"))
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out


def allocated_age_observations(inputs: dict[str, pd.DataFrame], iso3: str) -> pd.DataFrame:
    observed = inputs["age_cases"].loc[inputs["age_cases"]["iso3"].eq(iso3)].copy()
    columns = [
        "iso3",
        "country",
        "year",
        "model_age_group",
        "observed_cases",
        "age_allocation_weight",
        "source_age_group_count",
        "max_source_split_count",
        "min_allocation_fraction",
        "age_allocation_note",
    ]
    if observed.empty:
        return pd.DataFrame(columns=columns)

    observed["year"] = pd.to_numeric(observed["year"], errors="coerce").astype("Int64")
    observed["cases_total"] = pd.to_numeric(observed["cases_total"], errors="coerce").fillna(0.0)
    if "allocation_fraction" in observed:
        observed["allocation_fraction"] = pd.to_numeric(observed["allocation_fraction"], errors="coerce").fillna(1.0)
    else:
        observed["allocation_fraction"] = 1.0
    if "source_age_groups" not in observed:
        observed["source_age_groups"] = observed["model_age_group"]
    observed["source_age_groups"] = observed["source_age_groups"].astype(str)
    observed = observed.loc[
        observed["year"].between(MODEL2C_OBSERVATION_START_YEAR, MODEL2C_OBSERVATION_END_YEAR)
    ].copy()
    if observed.empty:
        return pd.DataFrame(columns=columns)

    observed["source_split_count"] = observed.groupby(["year", "source_age_groups"], dropna=False)["model_age_group"].transform("nunique")
    observed["allocation_weighted_cases"] = observed["cases_total"] * observed["allocation_fraction"]
    grouped = observed.groupby(["iso3", "country", "year", "model_age_group"], as_index=False).agg(
        observed_cases=("cases_total", "sum"),
        allocation_weighted_cases=("allocation_weighted_cases", "sum"),
        source_age_group_count=("source_age_groups", "nunique"),
        max_source_split_count=("source_split_count", "max"),
        min_allocation_fraction=("allocation_fraction", "min"),
    )
    grouped["age_allocation_weight"] = grouped["allocation_weighted_cases"] / grouped["observed_cases"].replace({0: np.nan})
    grouped["age_allocation_weight"] = grouped["age_allocation_weight"].fillna(1.0).clip(lower=0.05, upper=1.0)
    grouped["age_allocation_note"] = np.where(
        grouped["max_source_split_count"].gt(1),
        "Allocated from source age bins split across multiple model age groups; down-weighted in age observation diagnostics.",
        "Source age bin maps directly or near-directly to this model age group.",
    )
    return grouped.loc[:, columns]
