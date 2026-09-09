# Public data layer

This public repository uses two data tiers:

- `provenance/source_registry.csv` is a sanitized catalogue of reused sources. It preserves source names, URLs, access dates, original byte sizes, SHA-256 digests, recorded terms and analytical notes, but removes local paths and acquisition errors.
- `clean/country_metadata.csv` is the small processed input required to rerender main Figure 1.

The broader analysis-ready evidence is released next to the claim it supports:

- figure source data: `manuscript/figure_data/`;
- machine-readable tables: `manuscript/tables/`;
- generated model and design results: `outputs/`.

Third-party raw snapshots are not included. See `../DATA_ACCESS.md` for the access and redistribution boundary.
