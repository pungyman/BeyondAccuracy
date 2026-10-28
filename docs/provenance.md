# Release provenance

This is a curated release from the authors' LRD research workspace, prepared for **Beyond Accuracy: What Language-Subspace Interventions Capture, Remove, and Improve** (Multi-LLL 2026).

The selected source files and their original SHA-256 digests are listed in [release-manifest.json](release-manifest.json). These hashes describe the inputs to curation; the public configuration, packaging, and documentation were then adapted as described below. The manifest records the source Git commit; local file contents at curation time are identified by their digests.

## Included

- Core estimators, controls, geometry, probing, activation hooks, operators, dose calibration, evaluation, and model policies.
- E3 geometry, E10 probe-erasure, E9 intervention-grid, and E8 prompt-control/joint-scoring entry points.
- Offline tests relevant to those components and optional-engine checks.
- Selected aggregate CSVs and numerical JSON snapshots used to reproduce the accepted paper's figures/tables.
- Original figure generators and PNG figures from the workshop paper.

## Public-release adaptations

- Replaced the old reimplementation-focused README with the accepted paper's scope, findings, attribution, and reproduction instructions.
- Renamed the package distribution to `beyondaccuracy`, retaining flat-module imports and historical experiment IDs.
- Removed default dependence on private data/activation repositories. Local inputs are the default; explicit environment overrides permit user-owned caches.
- Added a pinned public-data fetcher, a local activation-extraction entry point, and a standard-library result viewer.
- Added a citation file, release documentation, and offline-test CI.

These setup changes do not alter the estimators, projection/erasure operators, or scoring definitions. The released summaries are original research outputs; offline validation is distinct from rerunning GPU inference.

## Excluded

Research history, Overleaf synchronization, manuscript drafts, private service configuration, cluster launchers, model weights, activations, raw datasets, per-response generations, and later E11 experiments are outside this release. The camera-ready manuscript was read for context; it is not uploaded here while the proceedings link is pending.

## Licensing

No new license is assigned by this curation. See [NOTICE](../NOTICE.md) for upstream attribution and dataset/model terms. A repository-wide license can be added by the rights holders independently of the paper citation.
