from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CountryContext:
    iso3: str
    country: str
    annual: pd.DataFrame
    time_resolved: pd.DataFrame
    population_age: pd.DataFrame
    metadata: dict[str, Any]
    pre_mean_dtp3: float
    pre_mean_incidence: float
    baseline_note: str
    monthly_covid: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostic_timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    pathogen_timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    surveillance_outcomes: pd.DataFrame = field(default_factory=pd.DataFrame)
    pre_mean_dtp1: float = 0.90
