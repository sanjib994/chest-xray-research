"""Paired explanation comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_explanation_summary(
    posthoc_metrics: dict[str, Any],
    prototype_metrics: dict[str, Any],
    posthoc_attribution,
    prototype_attribution,
) -> dict[str, Any]:
    """Combine metrics computed on the same image and target class."""
    posthoc = np.asarray(posthoc_attribution, dtype=np.float64).reshape(-1)
    prototype = np.asarray(prototype_attribution, dtype=np.float64).reshape(-1)
    if posthoc.size != prototype.size:
        raise ValueError("Paired attributions must have the same spatial size.")
    denominator = np.linalg.norm(posthoc) * np.linalg.norm(prototype)
    agreement = (
        float(np.dot(posthoc, prototype) / denominator) if denominator else 0.0
    )
    return {
        "protocol": "same_image_same_target_class",
        "posthoc_model": posthoc_metrics,
        "prototype": prototype_metrics,
        "attribution_cosine_similarity": agreement,
    }


def reliability_score(
    *,
    faithfulness: float | None = None,
    stability: float | None = None,
    prediction_agreement: float | None = None,
    mean_confidence_uncertainty: float | None = None,
    localization: float | None = None,
    calibration: float | None = None,
) -> dict[str, Any]:
    """Aggregate available reliability evidence into a transparent 0-100 score.

    Each supplied component is normalized to [0, 1]. Lower uncertainty is
    better, while all other components are higher-is-better. Missing evidence
    is excluded from both the numerator and denominator.
    """
    values = {}
    if faithfulness is not None:
        values["faithfulness"] = float(np.clip(faithfulness, 0.0, 1.0))
    if stability is not None:
        values["stability"] = float(np.clip(stability, 0.0, 1.0))
    if prediction_agreement is not None:
        values["robustness"] = float(np.clip(prediction_agreement, 0.0, 1.0))
    if mean_confidence_uncertainty is not None:
        values["certainty"] = float(
            1.0 - np.clip(mean_confidence_uncertainty, 0.0, 1.0)
        )
    if localization is not None:
        values["localization"] = float(np.clip(localization, 0.0, 1.0))
    if calibration is not None:
        values["calibration"] = float(np.clip(calibration, 0.0, 1.0))
    if not values:
        return {
            "available": False,
            "score": None,
            "components": {},
            "coverage": 0.0,
            "formula": "100 * mean(available normalized components)",
        }
    return {
        "available": True,
        "score": float(100.0 * np.mean(list(values.values()))),
        "components": values,
        "coverage": float(len(values) / 6.0),
        "formula": "100 * mean(available normalized components)",
    }
