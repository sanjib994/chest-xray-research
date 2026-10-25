import os
from pathlib import Path

import pandas as pd

from app.settings import configured_path


def load_threshold_map():
    csv_path = configured_path(
        "VINBIGDATA_THRESHOLDS",
        "configs/thresholds/best_validation_thresholds_weak_tuned.csv",
    )
    if not csv_path.exists():
        return {disease.casefold(): 0.5 for disease in DISEASES}

    df = pd.read_csv(csv_path)
    return {
        str(row["class_name"]).strip().casefold(): float(row["best_threshold"])
        for _, row in df.iterrows()
    }


from src.xai.production_visuals import get_visual_xai_artifacts, visual_xai_available

import tempfile
import uuid

import numpy as np
import torch

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.data.preprocessing import preprocess_image
from src.models.model_loader import load_model
from src.inference.predictor import ModelPredictor
from src.reliability.uncertainty import (
    uncertainty_summary,
    confidence_uncertainty,
    binary_entropy,
    probability_margin,
)
from src.reliability.robustness import (
    add_gaussian_noise,
    prediction_shift,
    prediction_agreement,
)


router = APIRouter()


DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
]


MODEL_NAME = os.environ.get("VINBIGDATA_MODEL", "convnext_tiny")
CHECKPOINT = configured_path(
    "VINBIGDATA_CHECKPOINT",
    str(
        Path("checkpoints")
        / (
            "convnext_tiny"
            if MODEL_NAME.lower().startswith("convnext")
            else "densenet121"
        )
        / (
            "convnext_tiny_weak_classes_best.pth"
            if MODEL_NAME.lower().startswith("convnext")
            else "densenet121_final_best.pth"
        )
    )
)


_model = None
_predictor = None


def _get_predictor():

    global _model
    global _predictor

    if _predictor is None:

        _model = load_model(
    checkpoint_path=str(CHECKPOINT),
    num_classes=7,
    device="cpu",
    model_name=MODEL_NAME,
)

        if hasattr(
            _model,
            "eval"
        ):

            _model.eval()

        _predictor = ModelPredictor(
            model=_model,
            diseases=DISEASES,
            device="cpu",
        )

    return _predictor


def _prediction_vector(
    prediction
):

    if isinstance(
        prediction,
        dict,
    ):

        values = []

        for disease in DISEASES:

            value = prediction[
                disease
            ]

            if isinstance(
                value,
                dict,
            ):

                value = value.get(
                    "probability"
                )

            values.append(
                float(value)
            )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    if isinstance(
        prediction,
        torch.Tensor,
    ):

        prediction = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        prediction,
        dtype=np.float32
    ).reshape(-1)


def _probability_dict(
    vector
):

    return {
        disease: float(
            value
        )
        for disease, value in zip(
            DISEASES,
            vector
        )
    }


def _target_layer(
    model
):

    if hasattr(
        model,
        "features"
    ):

        features = model.features

        if hasattr(
            features,
            "denseblock4"
        ):

            return features.denseblock4

    convolution_layers = [

        module

        for module in model.modules()

        if isinstance(
            module,
            torch.nn.Conv2d
        )
    ]

    if convolution_layers:

        return convolution_layers[-1]

    return None


def _run_xai(
    class_name,
    tensor
):

    if class_name == "GradCAM":

        from src.xai.gradcam import GradCAM

        constructor = GradCAM

    else:

        from src.xai.gradcampp import GradCAMPlusPlus

        constructor = GradCAMPlusPlus

    predictor = _get_predictor()

    model = predictor.model

    layer = _target_layer(
        model
    )

    if layer is None:

        raise RuntimeError(
            f"{MODEL_NAME} target layer unavailable."
        )

    explainer = constructor(
        model,
        layer
    )

    if callable(
        explainer
    ):

        return explainer(
            tensor
        )

    if hasattr(
        explainer,
        "generate"
    ):

        return explainer.generate(
            tensor
        )

    if hasattr(
        explainer,
        "compute"
    ):

        return explainer.compute(
            tensor
        )

    raise RuntimeError(
        f"{class_name} execution interface unavailable."
    )


def _json_safe(
    value
):

    if value is None:
        return None

    if isinstance(
        value,
        torch.Tensor,
    ):

        return value.detach().cpu().numpy().tolist()

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        dict,
    ):

        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            _json_safe(v)
            for v in value
        ]

    try:

        return value.item()

    except Exception:

        return value


@router.get(
    "/health"
)
def health():

    return {

        "status":
            "ok",

        "service":
            "VinBigData_XAI",

        "version":
            "1.0.0",
    }


@router.get(
    "/info"
)
def info():

    return {

        "service":
            "VinBigData_XAI",

        "version":
            "1.0.0",

        "model": {
    "name": MODEL_NAME.upper().replace("_", "-"),
    "type": "multilabel_classifier",
},

        "diseases":
            DISEASES,

        "official_test_inference_enabled":
            False,

        "training_enabled":
            False,

        "real_production_inference":
            True,
    }


@router.post(
    "/predict"
)
async def predict(
    file: UploadFile = File(...)
):

    filename = (
        file.filename
        or ""
    )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )

    if len(image_bytes) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB limit."
        )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as handle:

            handle.write(
                image_bytes
            )

            temporary_path = Path(
                handle.name
            )

        predictor = _get_predictor()

        processed = preprocess_image(
            str(
                temporary_path
            )
        )

        if not isinstance(
            processed,
            torch.Tensor,
        ):

            processed = torch.as_tensor(
                processed,
                dtype=torch.float32,
            )

        processed = (
            processed
            .detach()
            .clone()
            .float()
        )

        if processed.ndim == 3:

            processed = (
                processed.unsqueeze(0)
            )

        if processed.ndim != 4:

            raise RuntimeError(
                "Invalid production tensor shape."
            )

        with torch.no_grad():

            raw_prediction = (
                predictor.predict_tensor(
                    processed
                )
            )

        probabilities = (
            _prediction_vector(
                raw_prediction
            )
        )

        prediction_payload = {}
        threshold_map = load_threshold_map()
        for disease, probability in zip(
            DISEASES,
            probabilities
        ):

            value = float(
                probability
            )
            threshold = float(threshold_map.get(disease.casefold(), 0.5))
            prediction_payload[
                disease
            ] = {

                "probability":
                    value,

                "threshold":
                    threshold,

                "predicted":
                    value >= threshold,
            }

        gradcam_output = None
        gradcampp_output = None

        try:

            gradcam_output = _run_xai(
                "GradCAM",
                processed
            )

        except Exception as exc:

            gradcam_output = {

                "error":
                    f"{type(exc).__name__}: {exc}"
            }

        try:

            gradcampp_output = _run_xai(
                "GradCAMPlusPlus",
                processed
            )

        except Exception as exc:

            gradcampp_output = {

                "error":
                    f"{type(exc).__name__}: {exc}"
            }

        reliability = {}

        try:

            reliability[
                "uncertainty_summary"
            ] = _json_safe(
                uncertainty_summary(
                    probabilities
                )
            )

            reliability[
                "confidence_uncertainty"
            ] = _json_safe(
                confidence_uncertainty(
                    probabilities
                )
            )

            reliability[
                "binary_entropy"
            ] = _json_safe(
                binary_entropy(
                    probabilities
                )
            )

            reliability[
                "probability_margin"
            ] = _json_safe(
                probability_margin(
                    probabilities
                )
            )

        except Exception as exc:

            reliability[
                "uncertainty_error"
            ] = (
                f"{type(exc).__name__}: {exc}"
            )

        try:

            perturbed = add_gaussian_noise(
                processed.clone(),
                sigma=0.02,
                seed=42,
            )

            if perturbed.ndim == 3:

                perturbed = (
                    perturbed.unsqueeze(0)
                )

            with torch.no_grad():

                perturbed_raw = (
                    predictor.predict_tensor(
                        perturbed
                    )
                )

            perturbed_probabilities = (
                _prediction_vector(
                    perturbed_raw
                )
            )

            reliability[
                "robustness"
            ] = {

                "perturbed_probabilities":
                    _probability_dict(
                        perturbed_probabilities
                    ),

                "prediction_shift":
                    _json_safe(
                        prediction_shift(
                            probabilities,
                            perturbed_probabilities
                        )
                    ),

                "prediction_agreement":
                    _json_safe(
                        prediction_agreement(
                            probabilities,
                            perturbed_probabilities
                        )
                    ),
            }

        except Exception as exc:

            reliability[
                "robustness_error"
            ] = (
                f"{type(exc).__name__}: {exc}"
            )

        return {

            "schema_version":
                "1.0",

            "image_id":
                Path(filename).stem,

            "model": {

                "name":
                    (
                        "ConvNeXt-Tiny"
                        if MODEL_NAME.lower().startswith("convnext")
                        else "DenseNet121"
                    ),

                "type":
                    "multilabel_classifier",
            },

            "predictions":
                prediction_payload,

            "explanation": {

                "available":
                    (
                        gradcam_output is not None
                        or gradcampp_output is not None
                    ),

                "method":
                    "Grad-CAM + Grad-CAM++",

                "details": {

                    "gradcam":
                        _json_safe(
                            gradcam_output
                        ),

                    "gradcampp":
                        _json_safe(
                            gradcampp_output
                        ),
                },
            },

            "reliability": {

                "available":
                    len(
                        reliability
                    ) > 0,

                "details":
                    _json_safe(
                        reliability
                    ),
            },

            "metadata": {

    "synthetic":
        False,

    "official_test":
        False,

    "production_checkpoint":
        True,

    "training":
        False,

    "threshold_tuning":
        True,

    "model_name":
        MODEL_NAME,

    "threshold_source":
        "validation_f1_optimized",
},
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Production inference failed: "
                + str(exc)
            ),
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:

                temporary_path.unlink()

            except Exception:

                pass


# ---------------------------------------------------------------------------
# STEP 10.13B — PERMANENT VISUAL XAI HELPERS
# ---------------------------------------------------------------------------

def _attach_visual_xai_response(payload):
    try:
        artifacts = get_visual_xai_artifacts()

        payload["visual_xai"] = {
            "available": visual_xai_available(),
            "gradcam": artifacts.get("gradcam"),
            "gradcam_plus_plus":
                artifacts.get("gradcam_plus_plus"),
            "gradcam_attention_box":
                artifacts.get("gradcam_attention_box"),
            "gradcam_plus_plus_attention_box":
                artifacts.get(
                    "gradcam_plus_plus_attention_box"
                ),
            "attention_region_label":
                "MODEL ATTENTION REGION",
            "medical_lesion_boundary":
                False,
        }

    except Exception:
        payload["visual_xai"] = {
            "available": False,
            "gradcam": None,
            "gradcam_plus_plus": None,
            "gradcam_attention_box": None,
            "gradcam_plus_plus_attention_box": None,
            "attention_region_label":
                "MODEL ATTENTION REGION",
            "medical_lesion_boundary":
                False,
        }

    return payload


# STEP_10_13B_VISUAL_XAI_INTEGRATION
