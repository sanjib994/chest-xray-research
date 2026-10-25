# VinBigData_XAI

A product-oriented explainable chest X-ray analysis system derived from
the completed VinBigData research project.

## Setup

Use Python 3.10 or newer. From a fresh clone, create an environment and
install the dependencies. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The two model checkpoints are configured for Git LFS. Install Git LFS and
run `git lfs install` before cloning this repository. If you already
cloned it, run `git lfs pull` from the project directory.

The application code does not require the separate research dataset.
Inference requires the model assets listed below. The checkpoint files are
stored in Git LFS; the prototype evidence manifest and class-specific
thresholds are included in Git:

| Feature | Required asset | Default location |
|---|---|---|
| Classification and post-hoc XAI | ConvNeXt-Tiny checkpoint | `checkpoints/convnext_tiny/convnext_tiny_weak_classes_best.pth` |
| Classification | Class-specific thresholds | `configs/thresholds/best_validation_thresholds_weak_tuned.csv` |
| Intrinsic prototype explanation | DenseNet-backed prototype checkpoint | `checkpoints/prototype/prototype_network_full_training_final.pt` |
| Intrinsic prototype explanation | Prototype evidence manifest | `configs/prototype/P4_v2_prototype_evidence_manifest.csv` |

The `.gitattributes` file configures Git LFS for both checkpoints. If
running from a separately obtained checkpoint, set the corresponding
`VINBIGDATA_*` variables in `.env` to absolute paths or paths relative to
this project. Never commit the research dataset or untrusted
PyTorch/pickle checkpoints.

The ConvNeXt classifier and its post-hoc XAI do not require the prototype
assets. Intrinsic prototype endpoints require both the DenseNet-backed
prototype checkpoint and its evidence manifest. If the threshold file is
unavailable, predictions use the existing `0.5` threshold fallback.
The published prototype manifest omits source-image IDs and local paths;
training X-ray images are not included in this repository. To display
prototype reference images, obtain approved evidence images separately and
place each PNG in `assets/prototype_evidence` using the filenames
`prototype_000_full_evidence.png`, `prototype_000_query_patch.png`, and
corresponding three-digit prototype indices. The image folder is ignored by
Git. You can also set `VINBIGDATA_PROTOTYPE_IMAGE_DIR` in `.env` to another
folder. Never put patient images in a public repository unless you have
verified the applicable consent, privacy, and redistribution requirements.

If you have authorized local access to the original research evidence
manifest and its referenced images, prepare the app's local image folder with:

```powershell
python scripts/prepare_prototype_evidence.py `
  --manifest "PATH_TO_ORIGINAL_PROTOTYPE_EVIDENCE_MANIFEST.csv" `
  --source-root "PATH_TO_AUTHORIZED_RESEARCH_DATA"
```

The script copies only the referenced evidence images, renames them to the
portable prototype-index filenames, and strips embedded image metadata. It
does not copy the dataset, source IDs, or research manifest. The image
content itself may still be sensitive; do not redistribute it unless you
have verified permission and privacy requirements. Other clone users can
provide their own approved image files with the same filenames.

## Run

Start the API and UI in separate terminals:

```powershell
python scripts/run_backend.py
```

```powershell
python scripts/run_ui.py
```

The API is available at `http://127.0.0.1:8765/docs`; the Streamlit UI
normally runs at `http://localhost:8501`.

## Optional research workflows

Calibration and evaluation require the separately maintained research
dataset and its split files. They are not needed to run the basic app.
For calibration, set `VINBIGDATA_RESEARCH_ROOT` to the research dataset
directory before running `python scripts/run_calibration.py`. Do not add
the dataset to this repository.
