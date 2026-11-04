# Trained weights: local candidate, publication decision pending

The 18 trained best checkpoints are **not bundled in Git yet**. They have been
prepared locally at `local_artifacts/release_candidates/trained_weights/`:

| Model | Folds | Bytes per checkpoint | Total |
|---|---:|---:|---:|
| V1 Siamese CNN | 6 | 419,559 | 2,517,354 |
| EEGNet | 6 | 61,068 | 366,408 |
| Graph | 6 | 69,504 | 417,024 |
| All | 18 | | 3,300,786 |

Each file matches its frozen archive hash. The local `manifest.json` records the
tensor audit and strict architecture loading. There are no text participant-ID
or absolute-path patterns in the inspected checkpoint metadata. This does not
prove that learned representations contain no participant information or establish
redistribution rights. The checkpoints are trained on the supplied OMEGA export.

For an authorized local copy, recreate the candidate with:

```powershell
.\.venv\Scripts\python.exe -B tools\prepare_weight_candidates.py
```

The tool uses `torch.load(..., weights_only=True, map_location='cpu')`, checks that
all entries are finite tensors, and uses strict model-state loading. It does not
train or run inference. It refuses an occupied candidate output; use a new ignored
`--out local_artifacts/release_candidates/ANOTHER_NAME` if another inspection is needed.
The immutable historical archive is never edited.

The exact permission question and public demo scope are in
[PUBLIC_ARTIFACTS.md](../docs/PUBLIC_ARTIFACTS.md). Do not force-add the private local
candidate or recordings before resolving that question.
