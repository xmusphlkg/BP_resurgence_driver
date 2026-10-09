# BP resurgence driver

Public analysis code and shareable aggregate data for pertussis-notification dynamics.

## Scope

This repository contains custom Python, Stan and R code, processed non-identifiable inputs, a sanitized provenance registry, and compact model-generated outputs. It deliberately excludes manuscripts, supplementary materials, figures, tables, peer-review records, submission material, and internal release checks.

Third-party raw snapshots, logs, compiled executables, and oversized posterior-predictive artefacts are also excluded. See [DATA_ACCESS.md](DATA_ACCESS.md) and [LICENSING.md](LICENSING.md).

## Layout

| Path | Contents |
|---|---|
| `script/` | Analysis code, Stan models, and tests |
| `data/clean/` | Processed aggregate metadata |
| `data/provenance/` | Sanitized source registry |
| `outputs/` | Compact model-generated results |

## Setup

```bash
git clone https://github.com/xmusphlkg/BP_resurgence_driver.git
cd BP_resurgence_driver
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Some workflows require R, CmdStan, third-party source retrieval, and substantial compute. Reused data remain subject to their original terms.

Citation metadata are available in [CITATION.cff](CITATION.cff).

## Current analytical snapshot

The October 2026 update includes the current observation-map summaries,
country-macro results, burden-stratified comparisons, regenerated notification
null results and observation-noise stability results. Numerical outputs are
under `outputs/current_observation_map_benchmark/`,
`outputs/ijid_revision_audits_20261008/` and
`outputs/results_stability_20261009/`. Dated directory names identify analytical
runs. `PUBLIC_FILES.sha256` records checksums of the distributed files; cite the
Git commit for the exact public version. No archival DOI has been assigned.

This is a selective code and aggregate-data snapshot, not a self-contained
end-to-end reproduction bundle. Full regeneration requires separately retrieved
third-party data and preparation steps. Some historical figure and manuscript
build scripts refer to assets deliberately omitted from this repository.
