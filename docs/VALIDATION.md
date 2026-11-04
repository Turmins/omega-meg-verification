# Validation of the V1/V2 publication subset

Prepared 2026-11-03T14:39:32.935025+00:00.

- **68 unit/fixture tests passed**, no skips with restored artifacts. Includes V1/V2 frozen guards, invocation provenance, future initial/phase development outputs, importer safety and explicit CLI stages.
- **21 frozen core hashes match**; all **31 retained scientific source modules** are byte-identical to the preceding source snapshot. Historical loaders and numerical formulas were not modified.
- Saved results reproduce: **21,600 V1 score rows / 54 metric rows**, **180 V2 static metric rows**, **129,600 V2 sequential decisions**, three aggregate tables at absolute tolerance **1e-12**. No neural inference is used for this saved-results check.
- Six separate real CUDA inference examples passed: genuine and impostor for V1, EEGNet and Graph. Decisions match saved TEST rows; maximum score difference **3.47522443045e-08**, below **5e-6**. These examples are not a new accuracy study.
- V1 saved replay: **7,200 selections /14,400 decisions** match restored score tables exactly. Inline JavaScript passes Node syntax checking; referenced local resources exist; no external network resource was found. **No visual browser PASS is claimed**; the previous local-file browser restriction was respected.
- **239 imported run files** match the pinned archive hashes, including original protocols, environments, weights, calibrations and score tables. Three example caches were imported separately, without junctions or source writes.
- **425 selected original inputs/outputs rehashed unchanged**. The original full scientific archive and preceding Git repository remain separate. This is not a full rehash of every GiB in the archive.

## Actual environment and limitations

The initial run failed before test collection with a Windows page error while importing a SciPy DLL from the source-PC venv on F:. Already installed SciPy **1.15.3** on C: imported successfully. Validation used a temporary C: copy of that package with the existing Python **3.10.11** venv and its other installed dependencies. The checked fblas DLL matches the original byte-for-byte. No dependencies or external skills were installed; the original environments were not repaired or altered. Raw logs and receipts are Git-ignored under `.qa/`.

This temporary validation overlay is not a portable runtime dependency. A clean Python environment from `requirements-dev.txt` must still be installed and checked on the target computer. No laptop verification or standalone wheel-runtime certification is claimed.

Historical V2 phase DEV score reconstruction is excluded from this subset: phase thresholds remain hash-locked and phase TEST decisions/metrics are verified, but full phase DEV calibration replay is an archived workflow. Hybrid and E/L/F results/reproduction, legacy identification and old presentations are outside the public subset. Historical manifests were not rewritten.

No MEG training, new seeds, threshold tuning or external publication was performed. Dataset access and code/reference redistribution terms still require the owner's decisions; no license or public download URL has been invented.

## Machine-readable evidence

Tracked registries: `artifacts/FROZEN_CORE.json`, `artifacts/frozen_runs.json`, `artifacts/SOURCE_ORIGIN.json`. Local evidence: `.qa/unit-tests.xml`, `.qa/real-results.json`, `.qa/inference.json`, `.qa/validation.json`, `.qa/protected-inputs-before.json`. The protected-input snapshot SHA-256 is `6f7191d24153e00f815c9fb93438877c9c057e892032806cc81e1ccf8d30c342`.

## Standalone Git export

The 82 staged files round-tripped through `git checkout-index` byte-for-byte. A separate checkout on C: with no adjacent dataset or historical archive passed **67 tests, 1 expected external-artifact skip** in 17.31 seconds. It used the same documented validation environment/temporary SciPy overlay, not a newly installed environment. The first export harness attempt omitted the explicit basetemp parent; after fixing that harness setup, the complete suite passed. Public source did not require a change for this.

The staged artifact scan found no forbidden arrays/weights, participant-level CSVs, user-machine paths or credential-pattern hits. This is a pattern/allowlist check, not an exhaustive proof against every possible encoded secret.

## Public demo update

See [PUBLIC_DEMO_VALIDATION.md](PUBLIC_DEMO_VALIDATION.md) for the later bundled public demo, aggregate-only privacy tests and trained-weight release review. Earlier checks above refer to the private historical replay and remain preserved as historical evidence.

## Update after user authorship clarification

The preceding pending-weight statements describe the earlier review state. The user confirmed that weights were trained within this project and explicitly requested their inclusion. All 18 tensor state dictionaries are now copied byte-for-byte to weights/ and linked to original frozen SHA-256 in weights/manifest.json. No raw MEG, individual scores, participant embeddings or personal templates are included. Checkpoint files form complete folds 00-05 for V1, EEGNet and Graph; frozen core remains unchanged. No further training or lengthy computation was run for this final inclusion. See weights/README.md and PUBLIC_ARTIFACTS.md for the current scope.

## Final launch smoke test after bundling project-trained weights

The current 70-test suite passed in 20.21 seconds. V1, EEGNet and Graph each completed a real 30-second-query CUDA inference smoke test on existing authorized local inputs (7.76 /4.81 /4.98 seconds). V1 verified its saved decisions internally; the two V2 outputs were also compared with the saved static score rows and matched thresholds/decisions within 5e-6. The public demo controller passed 75 aggregate choices and four synthetic examples. These checks used the documented existing Python3.10 environment/SciPy workaround and local private runs, not a new dependency installation. No training was run. The English README was checked against actual CLI options, CSV results and available file paths; its Google Drive field remains intentionally empty.
