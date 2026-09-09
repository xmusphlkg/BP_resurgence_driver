#!/usr/bin/env python3
"""Extract public age-time pertussis severe-outcome evidence into one ledger."""

from __future__ import annotations

import csv
import html
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_SEVERE_OUTCOMES = ROOT / "data" / "raw" / "severe_outcomes"
OUTPUT_PATH = RAW_SEVERE_OUTCOMES / "severe_outcome_age_time_evidence.csv"
SOURCES_PATH = RAW_SEVERE_OUTCOMES / "severe_outcome_age_time_sources.csv"

EXTRACTION_DATE = date.today().isoformat()

LEDGER_COLUMNS = [
    "iso3",
    "country",
    "source_id",
    "geographic_scope",
    "outcome_type",
    "year",
    "month",
    "age_group",
    "count",
    "count_qualifier",
    "coverage_note",
    "source_url",
    "extraction_date",
    "notes",
]

SOURCE_COLUMNS = [
    "source_id",
    "iso3",
    "country",
    "source_url",
    "extraction_date",
    "extraction_status",
    "notes",
]

BRAZIL_SINAN_FILES = [f"coqbr{year:02d}.dbf" for year in range(10, 25)]
BRAZIL_SIM_FILES = [f"obtuf{year:02d}.dbf" for year in range(10, 25)]
BRAZIL_SIH_FILES = [f"nruf{year:02d}{month:02d}.dbf" for year in range(10, 25) for month in range(1, 13)]

JAPAN_ESTAT_TABLES = {
    2010: "000011816897",
    2011: "000014888136",
    2012: "000022220103",
    2013: "000027231570",
    2014: "000031288487",
    2015: "000032020051",
    2016: "000032020287",
    2017: "000032020166",
    2018: "000031883971",
    2019: "000031982788",
    2020: "000032119361",
    2021: "000032235980",
    2022: "000040098337",
    2023: "000040206174",
    2024: "000040316542",
}

NOMIS_AGE_CODES = {
    0: "all_ages",
    1: "<1 year",
    2: "1-4 years",
    3: "5-9 years",
    4: "10-14 years",
    5: "15-19 years",
    6: "20-24 years",
    7: "25-29 years",
    8: "30-34 years",
    9: "35-39 years",
    10: "40-44 years",
    11: "45-49 years",
    12: "50-54 years",
    13: "55-59 years",
    14: "60-64 years",
    15: "65-69 years",
    16: "70-74 years",
    17: "75-79 years",
    18: "80-84 years",
    19: "85-89 years",
    20: "90+ years",
}


def clean_count(value: object) -> int | None:
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", ".", "nan", "None"}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_brazil_age(label: str) -> str:
    text = label.strip()
    mapping = {
        "Total": "all_ages",
        "Ign/Branco": "unknown",
        "Menor 1 ano": "<1 year",
        "< 1 ano": "<1 year",
        "1 a 4 anos": "1-4 years",
        "5 a 9 anos": "5-9 years",
        "10 a 14 anos": "10-14 years",
        "15 a 19 anos": "15-19 years",
        "20 a 29 anos": "20-29 years",
        "30 a 39 anos": "30-39 years",
        "40 a 49 anos": "40-49 years",
        "50 a 59 anos": "50-59 years",
        "60 a 69 anos": "60-69 years",
        "70 a 79 anos": "70-79 years",
        "80 anos e mais": "80+ years",
    }
    return mapping.get(text, text)


def normalize_japan_age(label: str) -> str:
    text = label.strip()
    if text == "Total":
        return "all_ages"
    match = re.fullmatch(r"(\d+)Year", text)
    if match:
        return f"{match.group(1)} years"
    match = re.fullmatch(r"(\d+)-(\d+)Years", text)
    if match:
        return f"{match.group(1)}-{match.group(2)} years"
    match = re.fullmatch(r"(\d+)Years and over", text)
    if match:
        return f"{match.group(1)}+ years"
    return text.replace("Years", " years").replace("Year", " years")


def request_text(
    url: str,
    *,
    data: bytes | None = None,
    timeout: int = 180,
    encoding: str | None = None,
    attempts: int = 3,
) -> str:
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "pertussis-global-dynamics evidence-ledger/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except Exception as exc:  # pragma: no cover - network hardening path
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(float(attempt))
    else:  # pragma: no cover - defensive, loop either breaks or raises.
        raise RuntimeError(f"Failed to fetch {url}") from last_error
    if encoding is not None:
        return raw.decode(encoding, errors="replace")
    for candidate in ("utf-8-sig", "cp932", "latin-1"):
        try:
            return raw.decode(candidate)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def post_tabnet(def_path: str, params: dict[str, object], *, timeout: int = 180) -> str:
    url = f"http://tabnet.datasus.gov.br/cgi/tabcgi.exe?{def_path}"
    payload = urllib.parse.urlencode(params, doseq=True, encoding="latin-1").encode("latin-1")
    return request_text(url, data=payload, timeout=timeout, encoding="latin-1")


def existing_rows_for_source(source_id: str) -> list[dict[str, object]]:
    if not OUTPUT_PATH.exists():
        return []
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("source_id") == source_id]


def replace_rows_from_existing(
    rows: list[dict[str, object]],
    sources: list[dict[str, str]],
    *,
    source_id: str,
    iso3: str,
    country: str,
    source_url: str,
    error: Exception,
) -> None:
    fallback_rows = existing_rows_for_source(source_id)
    if not fallback_rows:
        raise RuntimeError(f"No existing fallback rows available for {source_id}") from error
    rows[:] = [row for row in rows if row.get("source_id") != source_id]
    rows.extend(fallback_rows)
    add_source(
        sources,
        source_id=source_id,
        iso3=iso3,
        country=country,
        source_url=source_url,
        extraction_status="reused_existing_after_fetch_error",
        notes=f"Reused existing ledger rows after public-source fetch failed: {type(error).__name__}.",
    )


def parse_tabnet_prn(page: str) -> list[list[str]]:
    match = re.search(r"<pre[^>]*>(.*?)</pre>", page, flags=re.IGNORECASE | re.DOTALL)
    body = html.unescape(match.group(1) if match else page)
    lines = [line.strip() for line in body.splitlines() if line.strip().startswith('"')]
    return list(csv.reader(lines, delimiter=";", quotechar='"'))


def append_ledger_row(
    rows: list[dict[str, object]],
    *,
    iso3: str,
    country: str,
    source_id: str,
    geographic_scope: str,
    outcome_type: str,
    year: int,
    age_group: str,
    count: int,
    count_qualifier: str,
    coverage_note: str,
    source_url: str,
    notes: str,
    month: int | str = "",
) -> None:
    rows.append(
        {
            "iso3": iso3,
            "country": country,
            "source_id": source_id,
            "geographic_scope": geographic_scope,
            "outcome_type": outcome_type,
            "year": int(year),
            "month": month,
            "age_group": age_group,
            "count": int(count),
            "count_qualifier": count_qualifier,
            "coverage_note": coverage_note,
            "source_url": source_url,
            "extraction_date": EXTRACTION_DATE,
            "notes": notes,
        }
    )


def add_source(
    sources: list[dict[str, str]],
    *,
    source_id: str,
    iso3: str,
    country: str,
    source_url: str,
    extraction_status: str,
    notes: str,
) -> None:
    sources.append(
        {
            "source_id": source_id,
            "iso3": iso3,
            "country": country,
            "source_url": source_url,
            "extraction_date": EXTRACTION_DATE,
            "extraction_status": extraction_status,
            "notes": notes,
        }
    )


def add_brazil_table_rows(
    rows: list[dict[str, object]],
    table: list[list[str]],
    *,
    source_id: str,
    outcome_type: str,
    geographic_scope: str,
    source_url: str,
    coverage_note: str,
    notes: str,
) -> None:
    if not table:
        return
    header = table[0]
    for cells in table[1:]:
        if not cells or cells[0].strip().lower() == "total":
            continue
        year = clean_count(cells[0])
        if year is None or year < 2010 or year > 2024:
            continue
        for label, value in zip(header[1:], cells[1:]):
            age_group = normalize_brazil_age(label)
            count = clean_count(value)
            if count is None or count <= 0:
                continue
            append_ledger_row(
                rows,
                iso3="BRA",
                country="Brazil",
                source_id=source_id,
                geographic_scope=geographic_scope,
                outcome_type=outcome_type,
                year=year,
                age_group=age_group,
                count=count,
                count_qualifier="annual_total" if age_group == "all_ages" else "annual_age_group_count",
                coverage_note=coverage_note,
                source_url=source_url,
                notes=notes,
            )


def fetch_brazil(rows: list[dict[str, object]], sources: list[dict[str, str]]) -> None:
    cases_source = "BRA_SINAN_TABNET_COQUELUCHE_CASES_2010_2024"
    cases_url = "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sinannet/cnv/coquebr.def"
    hosp_source = "BRA_SIH_TABNET_COQUELUCHE_HOSPITALIZATIONS_2010_2024"
    hosp_url = "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sih/cnv/nruf.def"
    death_source = "BRA_SIM_TABNET_A37_DEATHS_2010_2024"
    death_url = "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt10uf.def"
    try:
        cases_page = post_tabnet(
            "sinannet/cnv/coquebr.def",
            {
                "Linha": "Ano_1º_Sintoma(s)",
                "Coluna": "Faixa_Etária",
                "Incremento": "Casos_confirmados",
                "Arquivos": BRAZIL_SINAN_FILES,
                "formato": "prn",
                "mostre": "Mostra",
            },
        )
        add_brazil_table_rows(
            rows,
            parse_tabnet_prn(cases_page),
            source_id=cases_source,
            outcome_type="reported_cases",
            geographic_scope="Brazil national notifications",
            source_url=cases_url,
            coverage_note="SINAN confirmed pertussis notifications by symptom year and age group.",
            notes="DATASUS TabNet PRN extraction; nonzero rows are displayed by TabNet and annual totals are retained.",
        )
        add_source(
            sources,
            source_id=cases_source,
            iso3="BRA",
            country="Brazil",
            source_url=cases_url,
            extraction_status="extracted",
            notes="SINAN confirmed cases by age and symptom year.",
        )

        hosp_page = post_tabnet(
            "sih/cnv/nruf.def",
            {
                "Linha": "Ano_atendimento",
                "Coluna": "Faixa_Etária_1",
                "Incremento": "Internações",
                "Arquivos": BRAZIL_SIH_FILES,
                "SLista_Morb__CID-10": "22",
                "formato": "prn",
                "mostre": "Mostra",
            },
            timeout=240,
        )
        add_brazil_table_rows(
            rows,
            parse_tabnet_prn(hosp_page),
            source_id=hosp_source,
            outcome_type="hospitalizations",
            geographic_scope="Brazil national SIH/SUS admissions",
            source_url=hosp_url,
            coverage_note="SIH/SUS hospital admissions for coqueluche by attendance year and age group.",
            notes="DATASUS TabNet PRN extraction from SIH reduced files; public SIH/SUS admissions are not a private-sector all-hospital census.",
        )
        add_source(
            sources,
            source_id=hosp_source,
            iso3="BRA",
            country="Brazil",
            source_url=hosp_url,
            extraction_status="extracted",
            notes="SIH/SUS hospital admissions by age and attendance year.",
        )

        death_page = post_tabnet(
            "sim/cnv/obt10uf.def",
            {
                "Linha": "Ano_do_Óbito",
                "Coluna": "Faixa_Etária",
                "Incremento": "Óbitos_p/Residênc",
                "Arquivos": BRAZIL_SIM_FILES,
                "SCategoria_CID-10": "32",
                "formato": "prn",
                "mostre": "Mostra",
            },
        )
        add_brazil_table_rows(
            rows,
            parse_tabnet_prn(death_page),
            source_id=death_source,
            outcome_type="deaths",
            geographic_scope="Brazil national resident deaths",
            source_url=death_url,
            coverage_note="SIM resident deaths for ICD-10 A37/coqueluche by death year and age group.",
            notes="DATASUS TabNet PRN extraction; zero years may be omitted by the default nonzero display.",
        )
        add_source(
            sources,
            source_id=death_source,
            iso3="BRA",
            country="Brazil",
            source_url=death_url,
            extraction_status="extracted",
            notes="SIM resident deaths for ICD-10 A37/coqueluche by age and death year.",
        )
    except Exception as exc:  # pragma: no cover - network hardening path
        for source_id, source_url in [
            (cases_source, cases_url),
            (hosp_source, hosp_url),
            (death_source, death_url),
        ]:
            replace_rows_from_existing(
                rows,
                sources,
                source_id=source_id,
                iso3="BRA",
                country="Brazil",
                source_url=source_url,
                error=exc,
            )


def fetch_gbr_nomis(rows: list[dict[str, object]], sources: list[dict[str, str]]) -> None:
    source_id = "GBR_NOMIS_A37_DEATHS_AGE_2013_2024"
    base = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_161_1/geography/2092957703/"
        "cause_of_death/10137/gender/0/age/{age_code}/measure/1/measures/20100.data.csv"
    )
    try:
        for age_code, age_group in NOMIS_AGE_CODES.items():
            url = base.format(age_code=age_code)
            text = request_text(url, timeout=120)
            reader = csv.DictReader(StringIO(text))
            for item in reader:
                year_value = (
                    item.get("DATE")
                    or item.get("date")
                    or item.get("DATE_NAME")
                    or item.get("date_name")
                    or item.get("TIME")
                    or item.get("time")
                )
                count_value = (
                    item.get("OBS_VALUE")
                    or item.get("obs_value")
                    or item.get("VALUE")
                    or item.get("value")
                    or item.get("OBS")
                )
                year = clean_count(year_value)
                count = clean_count(count_value)
                if year is None or count is None:
                    continue
                append_ledger_row(
                    rows,
                    iso3="GBR",
                    country="United Kingdom",
                    source_id=source_id,
                    geographic_scope="England and Wales",
                    outcome_type="deaths",
                    year=year,
                    age_group=age_group,
                    count=count,
                    count_qualifier=f"Nomis age code {age_code}",
                    coverage_note="Nomis mortality statistics for ICD-10 A37 underlying-cause deaths; geography is England and Wales.",
                    source_url=url,
                    notes="GBR model label is not a complete UK severe-outcome geography for this source.",
                )
        add_source(
            sources,
            source_id=source_id,
            iso3="GBR",
            country="United Kingdom",
            source_url=base.format(age_code=0),
            extraction_status="extracted",
            notes="Nomis A37 deaths by age band; geography retained as England and Wales.",
        )
    except Exception as exc:  # pragma: no cover - network hardening path
        replace_rows_from_existing(
            rows,
            sources,
            source_id=source_id,
            iso3="GBR",
            country="United Kingdom",
            source_url=base.format(age_code=0),
            error=exc,
        )


def fetch_japan_estat(rows: list[dict[str, object]], sources: list[dict[str, str]]) -> None:
    source_id = "JPN_ESTAT_VITAL_A37_DEATHS_2010_2024"
    try:
        for year, stat_id in JAPAN_ESTAT_TABLES.items():
            url = f"https://www.e-stat.go.jp/en/stat-search/file-download?statInfId={stat_id}&fileKind=1"
            text = request_text(url, timeout=90)
            table = list(csv.reader(StringIO(text)))
            header: list[str] | None = None
            a37_row: list[str] | None = None
            for line in table:
                if "Total" in line and any("Year" in item for item in line):
                    header = line
                if len(line) >= 3 and line[0].strip() == "A37" and line[1].strip() == "Total":
                    a37_row = line
                    break
            if header is None or a37_row is None:
                continue
            for label, value in zip(header[2:], a37_row[2:]):
                count = clean_count(value)
                if count is None or count <= 0:
                    continue
                age_group = normalize_japan_age(label)
                append_ledger_row(
                    rows,
                    iso3="JPN",
                    country="Japan",
                    source_id=source_id,
                    geographic_scope="Japan national vital statistics",
                    outcome_type="deaths",
                    year=year,
                    age_group=age_group,
                    count=count,
                    count_qualifier="e-Stat ICD-10 A37 age column",
                    coverage_note="Japan Vital Statistics Table 1-1 (1), ICD-10 A37 deaths by age column.",
                    source_url=url,
                    notes="e-Stat age columns include overlapping detail and aggregate bins; use all_ages for annual totals.",
                )
        add_source(
            sources,
            source_id=source_id,
            iso3="JPN",
            country="Japan",
            source_url="https://www.e-stat.go.jp/en/stat-search/files",
            extraction_status="extracted",
            notes="Fixed statInfId list for Vital Statistics Table 1-1 (1), ICD-10 A37.",
        )
    except Exception as exc:  # pragma: no cover - network hardening path
        replace_rows_from_existing(
            rows,
            sources,
            source_id=source_id,
            iso3="JPN",
            country="Japan",
            source_url="https://www.e-stat.go.jp/en/stat-search/files",
            error=exc,
        )


def add_usa_cdc(rows: list[dict[str, object]], sources: list[dict[str, str]]) -> None:
    source_url = "https://www.cdc.gov/pertussis/php/surveillance/index.html"
    seen_summary_source_ids: set[str] = set()
    summary_path = ROOT / "outputs" / "usa_annual_age_specific_surveillance_summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                if item.get("evidence_family") != "official_severe_outcome_age_signature":
                    continue
                year = clean_count(item.get("year"))
                if year is None:
                    continue
                source_id = item.get("source_ids") or "US_CDC_ANNUAL_PERTUSSIS_SURVEILLANCE"
                seen_summary_source_ids.add(str(source_id))
                scope = item.get("geographic_scope") or "national"
                rowspecs = [
                    ("reported_cases", "all_ages", item.get("reported_case_total"), "CDC annual reported case total"),
                    ("hospitalizations", "all_ages", item.get("hospitalizations"), "CDC annual hospitalization count"),
                    ("deaths", "all_ages", item.get("total_deaths"), "CDC annual total deaths"),
                    ("deaths", "<1 year", item.get("infant_deaths"), "CDC annual infant deaths"),
                ]
                for outcome_type, age_group, value, qualifier in rowspecs:
                    count = clean_count(value)
                    if count is None or count <= 0:
                        continue
                    append_ledger_row(
                        rows,
                        iso3="USA",
                        country="United States of America",
                        source_id=str(source_id),
                        geographic_scope=str(scope),
                        outcome_type=outcome_type,
                        year=year,
                        age_group=age_group,
                        count=count,
                        count_qualifier=qualifier,
                        coverage_note="CDC annual pertussis surveillance severe-outcome summary.",
                        source_url=source_url,
                        notes=item.get("notes") or "CDC annual surveillance summary.",
                    )

    for source_id in sorted(seen_summary_source_ids):
        add_source(
            sources,
            source_id=source_id,
            iso3="USA",
            country="United States of America",
            source_url=source_url,
            extraction_status="extracted_from_existing_project_clean_table",
            notes="Annual CDC/NCHS surveillance severe-outcome row normalized from the project clean summary.",
        )

    age_path = ROOT / "data" / "clean" / "cases_age_specific.csv"
    if age_path.exists():
        with age_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                if str(item.get("iso3", "")).upper() != "USA":
                    continue
                year = clean_count(item.get("year"))
                count = clean_count(item.get("cases_total"))
                if year is None or count is None or count <= 0:
                    continue
                append_ledger_row(
                    rows,
                    iso3="USA",
                    country="United States of America",
                    source_id="USA_PERTUSSISINCIDENCE_AGE_CASES_2015_2025",
                    geographic_scope="United States national age-specific reported cases",
                    outcome_type="reported_cases",
                    year=year,
                    age_group=str(item.get("age_group") or "unknown"),
                    count=count,
                    count_qualifier="curated annual source age group count",
                    coverage_note="Curated annual age-specific reported-case rows used as external age-time evidence.",
                        source_url="https://github.com/xmusphlkg/PertussisIncidence",
                        notes="Reported cases by source age group; not a severe endpoint and not promoted to likelihood in this plan.",
                )

    add_source(
        sources,
        source_id="USA_PERTUSSISINCIDENCE_AGE_CASES_2015_2025",
        iso3="USA",
        country="United States of America",
        source_url="https://github.com/xmusphlkg/PertussisIncidence",
        extraction_status="extracted_from_existing_project_clean_table",
        notes="Curated annual USA reported-case age rows normalized as external age-time evidence.",
    )
    add_source(
        sources,
        source_id="USA_WONDER_A37_DEATHS",
        iso3="USA",
        country="United States of America",
        source_url="https://wonder.cdc.gov/",
        extraction_status="queued_access_limited_or_interactive",
        notes="CDC WONDER A37 direct query remains queued; CDC annual surveillance deaths are used as current public severe-outcome evidence.",
    )


def add_queued_sources(sources: list[dict[str, str]]) -> None:
    add_source(
        sources,
        source_id="SWE_SOCIALSTYRELSEN_A37_ENDPOINT",
        iso3="SWE",
        country="Sweden",
        source_url="https://sdb.socialstyrelsen.se/",
        extraction_status="queued_api_stability_check",
        notes="Swedish severe-outcome endpoint remains queued and is not required for primary results.",
    )


def deduplicate_sources(sources: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for source in sources:
        source_id = str(source["source_id"])
        by_id.setdefault(source_id, source)
    return [by_id[source_id] for source_id in sorted(by_id)]


def validate_rows_and_sources(rows: list[dict[str, object]], sources: list[dict[str, str]]) -> None:
    source_ids = {str(source["source_id"]) for source in sources}
    row_source_ids = {str(row["source_id"]) for row in rows}
    missing_sources = sorted(row_source_ids.difference(source_ids))
    if missing_sources:
        raise SystemExit("Missing severe-outcome source metadata for: " + ", ".join(missing_sources))

    seen_keys: set[tuple[str, str, str, int, str, str]] = set()
    for row in rows:
        key = (
            str(row["source_id"]),
            str(row["geographic_scope"]),
            str(row["outcome_type"]),
            int(row["year"]),
            str(row["month"]),
            str(row["age_group"]),
        )
        if key in seen_keys:
            raise SystemExit("Duplicate severe-outcome age-time ledger key: " + repr(key))
        seen_keys.add(key)

    required_counts = {
        ("GBR_NOMIS_A37_DEATHS_AGE_2013_2024", "deaths", 2024, "all_ages"): 13,
        ("JPN_ESTAT_VITAL_A37_DEATHS_2010_2024", "deaths", 2024, "all_ages"): 1,
        ("BRA_SIM_TABNET_A37_DEATHS_2010_2024", "deaths", 2024, "all_ages"): 22,
        ("BRA_SIH_TABNET_COQUELUCHE_HOSPITALIZATIONS_2010_2024", "hospitalizations", 2024, "all_ages"): 730,
        ("BRA_SINAN_TABNET_COQUELUCHE_CASES_2010_2024", "reported_cases", 2024, "all_ages"): 7204,
    }
    lookup = {
        (str(row["source_id"]), str(row["outcome_type"]), int(row["year"]), str(row["age_group"])): int(row["count"])
        for row in rows
    }
    for key, expected in required_counts.items():
        observed = lookup.get(key)
        if observed != expected:
            raise SystemExit(f"Known-count check failed for {key}: observed {observed}, expected {expected}")


def write_csv(path: Path, rows: Iterable[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    fetch_brazil(rows, sources)
    fetch_gbr_nomis(rows, sources)
    fetch_japan_estat(rows, sources)
    add_usa_cdc(rows, sources)
    add_queued_sources(sources)
    sources = deduplicate_sources(sources)

    rows = sorted(
        rows,
        key=lambda item: (
            str(item["iso3"]),
            str(item["source_id"]),
            int(item["year"]),
            str(item["outcome_type"]),
            str(item["age_group"]),
        ),
    )
    validate_rows_and_sources(rows, sources)
    write_csv(OUTPUT_PATH, rows, LEDGER_COLUMNS)
    write_csv(SOURCES_PATH, sources, SOURCE_COLUMNS)
    print(f"Wrote {len(rows)} severe-outcome age-time ledger rows to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
