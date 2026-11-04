# Public demo and trained-weight review

Checked 2026-11-04T13:59:45.874196+00:00.

## Completed

- A bundled offline `demo/index.html` is generated exclusively from the three already tracked aggregate CSVs. It contains **75 aggregate selector combinations** ({'v1': 9, 'v2_static': 30, 'v2_stream': 36}) and four explicitly synthetic decision examples. It has no participant IDs, individual score records, raw MEG, actual DEV thresholds or neural weights.
- No external scripts, images, fonts, frames or network data loads are required. Python/GPU/private archive are not required to open the public HTML.
- Node controller unit tests pass for all **75 aggregate choices**, genuine/impostor/false-accept/false-reject controls, equality at the threshold and non-finite rejection. These are logic tests with a DOM fixture, not a visual browser test. No browser restriction was bypassed.
- Full current V1/V2 fixture suite: **70 passed**, including new aggregate-only/privacy and build reproducibility tests. The existing source-PC environment and temporary installed SciPy 1.15.3 copy on C: were reused; no dependencies were installed and no fresh venv validation is claimed.
- **18 trained best checkpoints**, **3,300,786 bytes** total, have pinned source hashes, finite tensor-only content, no textual ID/path pattern hits, and strict loading into the unchanged matching architectures. They are copied byte-for-byte only into Git-ignored `local_artifacts/release_candidates/trained_weights/`. No optimizer states or source signals are copied.
- All **21 frozen core hashes**, all **31 retained scientific source modules**, all **18 original trained checkpoints** and the **original full replay** remain unchanged. No training, new seeds, threshold tuning or new scientific evaluation occurred.
- README contains an intentionally empty **Google Drive URL placeholder**, access/source/release/citation placeholders and the required sibling dataset layout. No Drive file, link or sharing permission was created or modified.

## Pending, not claimed complete

The historical replay has 60 participant pseudonyms and 14,400 individual method decisions. It remains private and unchanged. The public demo does not imitate those records with renamed subjects.

Redistribution permission for trained weights and individual score matrices has not been established from the local export. The educational purpose does not resolve this question by itself. The official OMEGA portal describes controlled access and data redistribution restrictions; the exact upstream release and applicable local export agreement remain unconfirmed. See [PUBLIC_ARTIFACTS.md](PUBLIC_ARTIFACTS.md). Weights and the full replay must not be staged/committed while this question is pending. The original data are excluded regardless.

No external GitHub publication, Pages deployment, fresh-environment installation or visual browser PASS is claimed. The code/reference license choice remains outstanding. The weight audit is not a proof against model memorization.

Local receipts: `.qa/public-demo-tests.xml`, `.qa/weight-candidates.log`, the ignored weight candidate `manifest.json`. New demo code and safe metadata can be reviewed independently of permission to distribute the binary candidates.

## Clean source export

All 90 staged source files round-tripped byte-for-byte into a checkout with no private archive, data, weights or individual replay. The complete Python suite passed **69 tests, 1 expected external-run skip**, and the Node controller checks passed all 75 aggregate choices and four synthetic examples. This is not a fresh dependency installation or visual browser test.

The current README no longer contains the two real pseudonymous ID examples; it obtains example IDs from the private local protocol. Existing Git history was not rewritten: the prior local root commit still contains those two README examples. A source-only snapshot ZIP without `.git` is prepared under ignored `local_artifacts/release_candidates/public_source/`. Use a fresh repository from this verified file snapshot if distribution must exclude those historic examples. No learned binaries or individual records are included in that ZIP.

## Update after user authorship clarification

The preceding pending-weight statements describe the earlier review state. The user confirmed that weights were trained within this project and explicitly requested their inclusion. All 18 tensor state dictionaries are now copied byte-for-byte to weights/ and linked to original frozen SHA-256 in weights/manifest.json. No raw MEG, individual scores, participant embeddings or personal templates are included. Checkpoint files form complete folds 00-05 for V1, EEGNet and Graph; frozen core remains unchanged. No further training or lengthy computation was run for this final inclusion. See weights/README.md and PUBLIC_ARTIFACTS.md for the current scope.
