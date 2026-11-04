# External artifacts and restoration

Nothing is downloaded automatically. Use the existing authorized local archive
`../../deep_learning_extension` and dataset `../../omega_student_broadband_T260_v1`
relative to this document. Public acquisition links are not available in this
source subset. Without authorized inputs, unit tests remain usable, but real replay,
historical validation and inference cannot be claimed complete.

`tools/prepare_saved_demo.py` imports the original V1 HTML/CSV after fixed hash,
complete-grid and decision checks. These individual records remain Git-ignored.

`tools/restore_results.py` imports the pinned V1/V2 protocols, original environments,
checkpoints, calibrations, saved scores, embeddings and training summaries.
Only the listed files in `artifacts/frozen_runs.json` are copied; the default
excludes feature caches. `--include-example-cache` adds three recording caches
for fold 0 genuine/impostor examples and their existing graph caches. Cache bytes
are checked during copy and by the ordinary runtime cache validators. It is not
a full dataset/feature-cache replica. No symlinks or junctions into the archive
are created. Existing different files are refused, and historical files are never
overwritten. Import receipts go to `local_artifacts/imports` separately.

The MEG export contains 120 float32 arrays of shape (260,33280), a shared sensor
order and metadata. Frozen loaders check source metadata and original code hashes;
V1 inference also checks raw-recording hashes. Keep the sibling dataset location
or use a new config/output for a new experiment, without editing old manifests.

`tools/verify_results.py` calls the unmodified V1/V2 validators, recalculates saved
metrics and replays all V2 streams from saved embeddings. It checks three aggregate
tables at absolute tolerance 1e-12. It does not generate fresh neural predictions.
The historical V2 run lacks phase DEV score tables. Its phase thresholds remain
hash-locked and phase TEST decisions/metrics are reproduced; full phase DEV
reconstruction remains in the separate historical archive, outside this subset.

V1 and V2 inference CLI perform real encoder computation. V2 uses verified cached
windows; missing caches can be generated inside the restored output, never by
writing into the source archive. GPU/CPU behavior and numerical agreement still
need checking on each target machine. A source-PC venv is not a portable artifact.

The bundled `demo/index.html` now works without these private artifacts: it uses public aggregates and explicitly synthetic examples. The original individual-score replay and the trained-weight release candidate remain Git-ignored pending the distribution review in [PUBLIC_ARTIFACTS.md](PUBLIC_ARTIFACTS.md).
