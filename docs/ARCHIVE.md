# Publication boundary and unchanged historical archive

This source subset supports V1 Siamese/phase verification and V2 EEGNet/graph
static/stream verification. It includes their aggregate results only.

The original complete scientific archive remains at `../../deep_learning_extension`
relative to this document. The previous source snapshot remains at
`../../omega-meg-verification`. Neither is rewritten by preparing this subset.
Their full original instructions and manifests, including excluded tests and
reference checksums, remain there. No private correspondence is copied into Git.

Excluded from this publication subset:

- identification CLI 02-09 and its configs, training/evaluation/visualization modules;
- old feasibility CLI 10, delivery validator 15, migration validator 16;
- phase DEV recovery 19, hybrid 22/23, E/L/F 24 and additional-seed runtime;
- hybrid and E/L/F aggregate tables, old educational plans and presentation builders;
- fusion reference CLI and its original two-file checksum registry.

The exact exclusions and original source hashes are in `artifacts/SOURCE_ORIGIN.json`.
Hybrid reused TEST; E/L/F reused DEV. This subset does not claim to reproduce either
series. Those results require the separate original code, frozen inputs and
review/admission packages. Guards must not be bypassed or manifests rewritten.
If these become central publication claims, supply a separate reviewed supplement.

The replacement integrity command protects explicitly selected inputs. It does not
recreate historical whole-workspace snapshots or claim that a snapshot taken after
an experiment proves earlier immutability.

No license is invented for the project, reference asset or private dataset.
Before external release, the owner must determine redistribution terms and how
authorized reviewers obtain the external artifacts. This is a local preparation,
not a public data/weights release.

The bundled `demo/index.html` now works without these private artifacts: it uses public aggregates and explicitly synthetic examples. The original individual-score replay and the trained-weight release candidate remain Git-ignored pending the distribution review in [PUBLIC_ARTIFACTS.md](PUBLIC_ARTIFACTS.md).
