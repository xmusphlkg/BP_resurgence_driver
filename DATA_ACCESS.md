# Data access and provenance

## Public content

This repository includes custom analysis code, processed aggregate inputs, sanitized provenance metadata, and compact model-generated outputs. It does not contain identifiable participant-level records.

## Exclusions

The public release deliberately excludes manuscript drafts and supplements, peer-review and submission material, internal audit records, raw third-party snapshots, local logs, compiled executables, and large draw-level posterior-predictive artefacts.

## Reused sources

`data/provenance/source_registry.csv` records the source name, URL, access date and recorded terms for reused inputs. It contains no local filesystem paths or credentials. Source providers' terms govern retrieval and reuse; raw source files are not redistributed here.

## Current derived results

Selected numerical benchmark and sensitivity results are distributed under
`outputs/`. The matched latent-ratio input is in
`data/derived/matched_latent_ratios.csv`. These are aggregate or simulated
analytical records. Internal run manifests containing working-tree state,
review correspondence and submission records are not distributed.

The public snapshot does not contain every input required for full regeneration.
Consult source URLs and provider terms for third-party retrieval. Public versions
are identified by Git commit and `PUBLIC_FILES.sha256`; no archival DOI has been
assigned.
