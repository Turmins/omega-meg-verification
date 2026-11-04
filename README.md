# MEG Verification — V1 / V2

**An educational study of cross-session identity verification from short MEG recordings.**

Code, **18 project-trained checkpoints**, an **offline interactive demo**, and saved aggregate results are included. Raw MEG recordings and participant-level score records are excluded.

[Offline demo](#offline-demo) · [Quick start](#quick-start) · [Results](#results) · [Dataset](#dataset-and-google-drive-access) · [Reproduction](#reproducing-the-saved-experiments)

## The question

**Does this short recording match the claimed person's reference recording?**

A shared encoder learns from training participants. At verification time, a reference template is compared with a query from another recording session; the system accepts or rejects the identity claim. No separate neural network is trained for each enrolled person.

The experiments compare:

- **V1:** a phase/coherency reference and a compact Siamese CNN.
- **V2:** the phase reference, an EEGNet-inspired encoder and a sensor graph encoder, with static and adaptive reference policies.

The supplied dataset contains **60 participants with two recordings each**. Six subject-disjoint folds use **40 TRAIN / 10 DEV / 10 TEST** participants. Both sessions stay in the same partition. Checkpoints and decision thresholds are selected on DEV and frozen before TEST evaluation of each run.

## Offline demo

Download or clone the repository and open **[`demo/index.html`](demo/index.html)** in a browser. On Windows PowerShell, from the repository root:

```powershell
Invoke-Item .\demo\index.html
```

The Russian-language interface works **without Python, GPU, model weights, a dataset download or network access**. It provides:

- **Real saved aggregates:** V1/V2, model, query duration, enrollment duration and update-policy selectors.
- **Synthetic decision examples:** genuine acceptance, impostor rejection, false acceptance and false rejection.

The example scores and thresholds are clearly illustrative. **The browser does not run a neural network, replay individual TEST decisions or display actual DEV thresholds.** Aggregate results and synthetic examples are separate parts of the page.

GitHub's source-file view does not execute HTML: open the downloaded file locally. A hosted GitHub Pages demo has not been deployed.

## Quick start

Use **64-bit Python 3.10**, and run commands from the repository root. The documented host environment used Python 3.10.11 and PyTorch 2.5.1+cu121. Create a fresh environment rather than copying another computer's `.venv`.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Environment activation is optional because the commands use its Python explicitly. Ordinary tests use synthetic fixtures; an external-artifact integration test is skipped when historical runs are absent. Tests do not retrain models on MEG.

This is a **source-checkout workflow**: keep `src/`, `scripts/`, `configs/`, `weights/` and the reference asset together. Standalone wheel execution without those repository files is not the supported workflow.

## Included trained weights

These are **our project's trained encoders**, not downloaded pretrained models. Only the selected `best.pt` state dictionaries are bundled, renamed without changing their bytes:

| Model | Files | Total size |
|---|---|---:|
| Siamese CNN, V1 | `weights/v1/fold_00.pt` … `fold_05.pt` | 2,517,354 bytes |
| EEGNet, V2 | `weights/eegnet/fold_00.pt` … `fold_05.pt` | 366,408 bytes |
| Graph, V2 | `weights/graph/fold_00.pt` … `fold_05.pt` | 417,024 bytes |
| **Total** | **18 checkpoints** | **3.30 MB** |

[`weights/manifest.json`](weights/manifest.json) binds every file to its historical SHA-256. All 18 checkpoints passed finite-tensor checks and strict loading into the matching architecture. They contain model parameters and ordinary buffers, not participant tables, personal templates or saved signal embeddings. The phase reference has no learned checkpoint.

Read a state dictionary with the pinned PyTorch version:

```python
import torch

state = torch.load(
    "weights/eegnet/fold_00.pt",
    map_location="cpu",
    weights_only=True,
)
```

Weights alone are not a complete calibrated verification package: real MEG inference also needs compatible preprocessing, enrollment inputs, protocols and saved DEV thresholds. See [weight details](weights/README.md) and [external artifacts](docs/EXTERNAL_ARTIFACTS.md).

## Results

**Static verification, 30-second queries. HTER (%) — lower is better.**

| Method | V1: 240 s enrollment | V2: 30 s enrollment | V2: 240 s enrollment |
|---|---:|---:|---:|
| Phase / coherency | **6.23** | **10.94** | **6.83** |
| Siamese CNN | 12.80 | — | — |
| EEGNet-inspired encoder | — | 13.95 | 15.06 |
| Graph encoder | — | 21.33 | 18.52 |

**Compare methods within a column.** V1 uses four fixed query starts; V2 uses every nonoverlapping interval. Their percentages are not a matched comparison across protocols. Stream results are a separate set of attempts and are available in [`v2_stream_summary.csv`](results/v2_stream_summary.csv).

FAR measures accepted impostor attempts; FRR measures rejected genuine attempts. **HTER = (FAR + FRR) / 2.** Decisions use each fold's DEV-selected threshold. ROC-AUC and diagnostic approximate EER are computed within folds and then averaged; TEST EER does not select the operating threshold.

The phase reference outperformed the tested trained encoders in these static settings. This is an observed result, not a claim that deep learning must generally perform worse.

These are **exploratory results on a previously studied dataset**. Attempts share participants and signal intervals; TRAIN folds overlap. They do not establish independent external validation, long-term adaptation, resistance to unseen attacks or performance on other populations/devices.

Sources: [V1 aggregates](results/v1_summary.csv), [V2 static aggregates](results/v2_static_summary.csv), [protocol](docs/PROTOCOL.md).

## Dataset and Google Drive access

**User-provided Google Drive link: `[ADD_GOOGLE_DRIVE_DATASET_URL]`**

- Access instructions and applicable conditions: `[ADD_ACCESS_INSTRUCTIONS]`
- Original source and exact release, when confirmed: `[ADD_SOURCE_AND_RELEASE]`
- Dataset citation / DOI: `[ADD_DATASET_CITATION]`

These fields are intentionally unfilled. No Drive link or sharing permission is created by this repository. Dataset access conditions remain separate from authorship of the trained model weights.

Extract the authorized dataset beside the repository, without an extra nested directory:

```text
parent/
  omega-meg-verification-publication/
  omega_student_broadband_T260_v1/
    cache/                   # 120 source NPY arrays
    recording_index.csv
    cache_manifest.csv
    channels.txt
    protocol_FROZEN.json
    README.txt
  deep_learning_extension/   # optional private historical run archive
```

The local export has 120 float32 arrays of shape `(260, 33280)`: 260 seconds at 128 Hz, including context around the central 240 seconds. Its exact upstream release is not documented locally. A different download/release is not guaranteed to match the frozen hashes. Do not commit recordings, participant/session maps or individual scores.

**The bundled offline demo and ordinary unit tests do not require this dataset.**

## Reproducing the saved experiments

For authorized access to the historical archive, restore the pinned run artifacts and three example feature caches. The tool can show sizes before copying:

```powershell
.\.venv\Scripts\python.exe -B tools\restore_results.py --include-example-cache --plan
.\.venv\Scripts\python.exe -B tools\restore_results.py --include-example-cache
.\.venv\Scripts\python.exe -B tools\verify_results.py
```

The selected package is approximately 285 MB, not the full scientific archive. Historical files are copied byte-for-byte; differing occupied files are refused. Saved metrics and V2 streams are recalculated from saved scores/embeddings, without generating fresh neural predictions.

For **real inference**, use the restored run and authorized source data. Obtain example identities from the private protocol:

```powershell
$ids = (Get-Content .\outputs\verification_v1\protocol.json -Raw | ConvertFrom-Json).folds[0].test_subjects
$claimed = $ids[0]
$source = $ids[0]  # use $ids[1] for an impostor query

.\.venv\Scripts\python.exe -B scripts\14_verify_claim.py --fold 0 --claimed $claimed --source $source --duration 30 --start 0
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model eegnet --fold 0 --claimed $claimed --source $source --enrollment 30 --duration 30 --start 0 --policy static
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model graph --fold 0 --claimed $claimed --source $source --enrollment 30 --duration 30 --start 0 --policy static
```

V1 processes allowed windows from source recordings. V2 runs its encoder on verified cached windows. The three example caches do not cover all participants. `static` does not update the reference; no profile is persisted without `--profile`.

### Optional new training

Training is separate from the demo, tests and saved-result checks. Stages must be explicit, and new experiments need a new output directory:

```powershell
.\.venv\Scripts\python.exe -B scripts\11_run_verification.py --out outputs\v1_new --stage prepare
# Only when new training is intended:
.\.venv\Scripts\python.exe -B scripts\11_run_verification.py --out outputs\v1_new --stage train
```

V2 additionally needs the complete compatible V1 feature cache named by `base_run`; three example caches are insufficient. Never edit historical manifests or disable hash/freeze checks to load a changed experiment. Script numbers do not define a mandatory sequential pipeline.

## Verification status and boundaries

- **70 tests passed** in the available host environment.
- The offline demo controller passed **75 aggregate choices and four synthetic examples**.
- V1, EEGNet and Graph completed real CUDA inference smoke tests on available authorized inputs.
- All **21 frozen core hashes** remain unchanged; all **18 bundled weight hashes** match their historical checkpoints.
- Saved verification reproduced 54 V1 metric rows, 180 V2 static metric rows and 129,600 V2 stream decisions.

Fresh-environment installation on another computer and visual browser verification are not claimed. Host validation used an existing Python environment with the documented SciPy workaround. See [validation evidence](docs/VALIDATION.md).

Hybrid, E/L/F, legacy identification, individual-score replay and old presentation/migration tools are outside this public subset. Their original materials remain in the separate unchanged [historical archive](docs/ARCHIVE.md). Full phase DEV reconstruction for the historical V2 run remains an archived workflow; its phase thresholds and TEST decisions are still checked.

## Repository guide

| Path | Purpose |
|---|---|
| `demo/` | Self-contained aggregate/synthetic demonstration |
| `weights/` | 18 project-trained encoders and checksum manifest |
| `src/omega_dl/` | V1/V2 scientific implementation |
| `configs/` | V1/V2 configurations |
| `results/` | Saved aggregate CSVs |
| `scripts/` | Experiment, reporting and inference entry points |
| `tools/` | Artifact preparation and integrity checks |
| `tests/`, `docs/` | Tests, protocol and scope evidence |

The compatibility assets `models/autoencoder.py` and `models/heads.py` remain because the original V1 loader hashes the complete models directory. They are not additional advertised experiments. See [frozen-core compatibility](docs/FROZEN_CORE.md).

## License

The project owner still needs to select the code/model license and confirm the reference-code redistribution terms before external release. No license is invented here. Including our own model weights does not grant permission to redistribute the source dataset or participant-level records.
