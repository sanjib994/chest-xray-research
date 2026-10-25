# VinBigData_XAI Production UI

This directory contains the production Streamlit user interface.

## Runtime architecture

Streamlit UI -> FastAPI -> production inference pipeline.

The UI does not load the research checkpoint directly.

The UI communicates with:

`POST /predict`

provided by the validated FastAPI production API.

## Supported input

- JPG
- JPEG
- PNG

Maximum upload size:

10 MB

## Output

The UI displays:

- ConvNeXt-Tiny disease probabilities
- Grad-CAM
- Grad-CAM++
- Integrated Gradients
- faithfulness evaluation (deletion perturbation)
- stability evaluation (attribution similarity)
- localization status (requires a linked lesion mask)
- uncertainty information
- robustness information
- inference metadata

## Safety

The UI does not:

- train models
- update weights
- tune thresholds
- run the official test set
- modify research results
- modify model checkpoints
- scan the research dataset

This is a research/decision-support prototype and is not a standalone medical diagnosis.