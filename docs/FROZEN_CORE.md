# Frozen source compatibility

All 21 source paths in the historical V1/V2 `code_sha256` manifests are preserved
byte-for-byte. The loaders still check the original complete hash mappings and
checkpoint/calibration locks. No `check_code=False` override is used.

The original V1 loader hashes every Python file in `src/omega_dl/models`.
Consequently `autoencoder.py` and `heads.py` remain as compatibility assets even
though the primary verification CLI does not select an autoencoder or use the
old identification head. Removing them would invalidate the historical core
mapping. They do not add a separate advertised workflow.

The single V1 reference source remains because `data/filters.py` reads its BANDS
definition. Its original bytes are checked by the new publication registry in
`artifacts/FROZEN_CORE.json`. The old two-reference registry remains in the archive;
it is not rewritten to pretend the historical experiment used a different set.

All retained scientific modules, including scoring, calibration, preprocessing
and reporting calculations, are copied unchanged. Publication-only changes affect
CLI startup, tests, docs and artifact preparation tools. CLI stages are explicit;
`--help` creates no runtime files. New provenance snapshots describe this new
source set. They are never substituted for old training provenance.
