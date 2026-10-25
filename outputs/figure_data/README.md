# Figure 1 public source data

These frozen CSV exports are the machine-readable inputs required by
`script/figures/figure_1_global_trends.R`:

- `figure_1_context_monthly.csv` contains 1,188 country-month rows for the
  nine primary model backgrounds from January 2015 through December 2025.
  Missing observations are retained as explicit rows and marked in
  `observation_available`.
- `figure_1_context_baselines.csv` contains the 27 comparator rows used for the
  three Figure 1 baseline estimates per country.
- `figure_1_context_regional.csv`, `figure_1_context_selection.csv`,
  `figure_1_context_income.csv`, `figure_1_context_coverage.csv`,
  `figure_1_context_temporal_coverage.csv` and
  `figure_1_context_vaccination.csv` preserve the accompanying regional,
  selection, income, coverage, temporal-check and vaccination summaries from
  the same frozen build.
- `figure_1_context.metadata.json` records the builder version and SHA-256
  checksums of the generated exports.

The exports are frozen aggregate model summaries. They do not contain raw
third-party snapshots or participant-level records. The complete upstream
regeneration inputs remain outside this selective public release; use the
recorded Git commit and the metadata checksums when citing this snapshot.
