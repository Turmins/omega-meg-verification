# MEG Verification — V1 / V2

**An educational study of cross-session identity verification from short MEG recordings.**

Code, **18 project-trained checkpoints**, an **offline interactive demo**, and saved aggregate results are included. Raw MEG recordings and participant-level score records are excluded.

[Offline demo](#offline-demo) · [Quick start](#quick-start) · [Results](#results) · [Dataset](#dataset-access) · [Reproduction](#reproducing-the-saved-experiments)

## The question

**Does this short recording match the claimed person's reference recording?**

A shared encoder learns from training participants. At verification time, a reference template is compared with a query from another recording session; the system accepts or rejects the identity claim. No separate neural network is trained for each enrolled person.

The experiments compare:

- **V1:** a phase/coherency reference and a compact Siamese CNN.
- **V2:** the phase reference, an EEGNet-inspired encoder and a sensor graph encoder, with static and adaptive reference policies.

The supplied dataset contains **60 participants with two recordings each**. Six subject-disjoint folds use **40 TRAIN / 10 DEV / 10 TEST** participants. Both sessions stay in the same partition. Checkpoints and decision thresholds are selected on DEV and frozen before TEST evaluation of each run.

## Offline demo

**The fastest first run:** obtain the complete project folder and open **[`demo/index.html`](demo/index.html)** in a browser. No installation is needed. In Windows PowerShell, replace the example folder below with your actual project folder:

```powershell
Set-Location 'C:\work\omega-meg-verification'
Invoke-Item .\demo\index.html
```

The Russian-language interface works **without Python, GPU, model weights, a dataset download or network access**. It provides:

- **Real saved aggregates:** V1/V2, model, query duration, enrollment duration and update-policy selectors.
- **Synthetic decision examples:** genuine acceptance, impostor rejection, false acceptance and false rejection.

The example scores and thresholds are clearly illustrative. **The browser does not run a neural network, replay individual TEST decisions or display actual DEV thresholds.** Aggregate results and synthetic examples are separate parts of the page.

GitHub's source-file view does not execute HTML: open the downloaded file locally. A hosted GitHub Pages demo has not been deployed.

## Quick start

These instructions target **Windows with PowerShell**. Start with the offline demo above if you only want to view the results.

| Goal | What you need | Next step |
|---|---|---|
| View the interactive demo | Complete project folder and a browser | [Offline demo](#offline-demo) |
| Run unit tests | Project folder and Python environment | Steps 1-4 below |
| Run real inference / verify saved results | Environment, compatible dataset and historical run archive | Then [restore the artifacts](#reproducing-the-saved-experiments) |

### 1. Open the project folder

Extract the complete source folder first. Open PowerShell and replace this example path with the folder containing `README.md`:

```powershell
Set-Location 'C:\work\omega-meg-verification'
Get-Item .\README.md, .\requirements-dev.txt, .\demo\index.html
```

All following commands run from this folder. You may use another drive or folder name. This is a **source-checkout workflow**: keep `src/`, `scripts/`, `configs/`, `weights/` and the reference asset together. Standalone wheel execution is not the supported workflow.

### 2. Create a Python environment

Install **64-bit Python 3.10** from the [official Python 3.10.11 release](https://www.python.org/downloads/release/python-31011/) (Windows installer, 64-bit), including the Python launcher. Reopen PowerShell after installation. Python 3.10.11 is the historical compatibility version used for host validation; newer Python versions have not been validated for this project.

```powershell
py -3.10 --version
py -3.10 -c "import struct; print('Python bits:', struct.calcsize('P') * 8)"
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Expect `Python 3.10.x` and `Python bits: 64`. If `py` is not found, check that Python and its launcher were installed, then reopen PowerShell. Create a fresh `.venv`; an environment copied from another computer is not portable. Activation is unnecessary because every command uses the environment's Python explicitly.

### 3. Install dependencies: choose CUDA or CPU

**NVIDIA GPU / CUDA route:** confirm that the installed NVIDIA driver sees a CUDA-compatible GPU, then install the CUDA 12.1 build used by the project:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**CPU alternative:** in a fresh environment, use this command instead of the CUDA install above:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

After **one** of those choices, install the remaining pinned packages and check them:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('Wheel CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The CUDA route should report `2.5.1+cu121`, `12.1`, `True` and the NVIDIA GPU name. If CUDA is unavailable, inspect the driver and installed wheel before proceeding with GPU inference. The CPU route should report `CUDA available: False` and `Device: CPU`. The inference scripts select CUDA automatically when available and otherwise use CPU; they have no `--device` option. CPU execution and numerical agreement on another machine are not yet validated.

The wheel choices follow the [official PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251). This project does not require `torchvision` or `torchaudio`. Initial package installation requires an internet connection and downloads a large PyTorch wheel.

### 4. Run the tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Ordinary tests use synthetic fixtures and do not retrain models on MEG. A source-only export previously passed **69 tests with one expected external-artifact skip**; the restored local project passed all **70**. A passing unit suite alone does not establish that real-data inference is ready. Continue below only if you need the dataset and historical runs.

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

## Dataset access

**The link to the dataset will be sent by email.**

- Access instructions and applicable conditions: `[ADD_ACCESS_INSTRUCTIONS]`
- Original source and exact release, when confirmed: `[ADD_SOURCE_AND_RELEASE]`
- Dataset citation / DOI: `[ADD_DATASET_CITATION]`

The remaining metadata fields are intentionally unfilled. Dataset access conditions remain separate from authorship of the trained model weights.

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
  deep_learning_extension/   # required for artifact restoration / inference examples
```

The local export has 120 float32 arrays of shape `(260, 33280)`: 260 seconds at 128 Hz, including context around the central 240 seconds. Its exact upstream release is not documented locally. A different download/release is not guaranteed to match the frozen hashes. Do not commit recordings, participant/session maps or individual scores.

**The bundled offline demo and ordinary unit tests do not require this dataset.**

For the documented real-data examples, obtain **both** the compatible dataset above and the private historical `deep_learning_extension` archive. The dataset link will be sent by email; no archive download link is supplied. A new user must obtain both inputs from the project owner before proceeding. The bundled `weights/` directory does not replace the protocols, DEV thresholds, saved scores or feature caches in that archive.

## Reproducing the saved experiments

### 5. Restore the run artifacts

Prerequisites: complete Steps 1-4, place the compatible dataset beside the project as shown above, and make the historical archive available. Check the default locations:

```powershell
Test-Path ..\omega_student_broadband_T260_v1\recording_index.csv
Test-Path ..\deep_learning_extension\outputs\verification_v1\protocol.json
Test-Path ..\deep_learning_extension\outputs\adaptive_v2\protocol.json
```

All three should return `True`. Restore the pinned run artifacts and three example feature caches. Run the plan first to see file count and bytes without copying:

```powershell
.\.venv\Scripts\python.exe -B tools\restore_results.py --include-example-cache --plan
.\.venv\Scripts\python.exe -B tools\restore_results.py --include-example-cache
```

The default archive is `..\deep_learning_extension`. If it is elsewhere, add `--archive-root 'D:\your-archive\deep_learning_extension'` to **both** restore commands, replacing that example with its actual location. A successful copy prints a JSON `receipt`; it creates `outputs/verification_v1` and `outputs/adaptive_v2` inside this checkout. The selected package is approximately 285 MB. Differing occupied files are refused; historical files are copied byte-for-byte.

### 6. Verify the saved results

```powershell
.\.venv\Scripts\python.exe -B tools\verify_results.py
```

Expect JSON with `"status": "passed"`, `"aggregate_tables_reproduced": 3`, `"training": false` and `"neural_inference": false`. This recalculates saved metrics and V2 streams from saved scores/embeddings. It does not generate fresh neural predictions.

### 7. Run real inference on an existing held-out example

These commands use `best.pt`, DEV thresholds and protocols from the **restored outputs**, rather than reading `weights/` directly. V1 also checks the original recording and saved evaluation row. Keep the dataset at the documented sibling location; changing historical manifests or disabling hash checks is not a supported relocation method.

Obtain the first fold's example identities from the private protocol, then run one genuine claim for each model:

```powershell
$ids = (Get-Content .\outputs\verification_v1\protocol.json -Raw | ConvertFrom-Json).folds[0].test_subjects
$claimed = $ids[0]
$source = $ids[0]  # use $ids[1] for an impostor query

.\.venv\Scripts\python.exe -B scripts\14_verify_claim.py --fold 0 --claimed $claimed --source $source --duration 30 --start 0
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model eegnet --fold 0 --claimed $claimed --source $source --enrollment 30 --duration 30 --start 0 --policy static
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model graph --fold 0 --claimed $claimed --source $source --enrollment 30 --duration 30 --start 0 --policy static
```

Each command prints JSON including `device`, score, DEV threshold and accept/reject decision. V1 reports `matches_saved_decisions: true` after checking its phase/CNN predictions against the saved row; V2 reports `score`, `threshold`, `accepted` and `updated`. To try an impostor claim, set `$source = $ids[1]` and repeat the same three commands. These two IDs are covered by the restored example caches; the caches do not cover all participants.

V1 processes allowed windows from source recordings. V2 runs its encoder on verified cached windows. `static` does not update the reference; no profile is persisted without `--profile`. This is actual encoder computation, unlike the browser demo.

If a file is missing, check the dataset/archive layout and restore step. A hash/frozen-input mismatch requires the matching original input, not edits to an old manifest or threshold.

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

**Test machine used for the 2026-11-04 checks:** Windows, Python 3.10.11 x64 and an NVIDIA RTX 3060 Ti (8 GB), driver 560.94, with PyTorch 2.5.1+cu121. This records the validation environment; that GPU model and memory size are not project requirements. V1/EEGNet/Graph inference and a simple tensor operation explicitly ran on CUDA. The 70-test suite, `pip check` and artifact restore plan passed on the same host, without a claim that every test used the GPU; the plan found 260 files (284,778,686 bytes). A fresh venv with pip was also created, and all 12 PowerShell command blocks parsed successfully.

**Not yet verified:** installing all dependencies into that fresh venv, the CPU installation/inference route, another computer, and visual browser behavior. The practical model/test checks reused the existing Python environment with the documented temporary SciPy workaround; no fresh dependency-installation success is claimed. See [validation evidence](docs/VALIDATION.md).

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
