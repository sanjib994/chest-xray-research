"""
VinBigData_XAI — Production Result Formatter

This module converts model/XAI/reliability outputs into a stable,
JSON-serializable product response.

No model inference occurs here.
"""

from typing import Any, Dict, Optional


DEFAULT_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
]


def _safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_prediction(
    disease,
    probability,
    threshold=0.5,
):
    """
    Format one disease prediction.
    """

    probability = _safe_float(
        probability
    )

    threshold = _safe_float(
        threshold
    )

    result = {
        "probability": probability,
        "threshold": threshold,
        "predicted": None,
    }

    if (
        probability is not None
        and threshold is not None
    ):
        result["predicted"] = bool(
            probability >= threshold
        )

    return result


def format_result(
    predictions,
    image_id=None,
    model_name="ConvNeXt-Tiny",
    xai=None,
    reliability=None,
    metadata=None,
):
    """
    Build the product response.

    Missing evidence is represented as None rather than fabricated.
    """

    formatted_predictions = {}

    for disease in DEFAULT_DISEASES:

        value = predictions.get(
            disease
        )

        if isinstance(value, dict):

            probability = value.get(
                "probability"
            )

            threshold = value.get(
                "threshold",
                0.5
            )

        else:

            probability = value
            threshold = 0.5

        formatted_predictions[
            disease
        ] = format_prediction(
            disease,
            probability,
            threshold,
        )

    result = {
        "schema_version": "1.0",
        "image_id": image_id,
        "model": {
            "name": model_name,
            "type": "multilabel_classifier",
        },
        "predictions": formatted_predictions,
        "explanation": (
            xai
            if xai is not None
            else {
                "available": False,
                "method": None,
            }
        ),
        "reliability": (
            reliability
            if reliability is not None
            else {
                "available": False,
            }
        ),
        "metadata": (
            metadata
            if metadata is not None
            else {}
        ),
    }

    return result


def validate_result(
    result,
):
    """
    Validate the minimum product response contract.
    """

    required = [
        "schema_version",
        "model",
        "predictions",
        "explanation",
        "reliability",
    ]

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:

        raise ValueError(
            "Missing result fields: "
            + ", ".join(missing)
        )

    return True
