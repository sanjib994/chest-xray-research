"""
VinBigData_XAI — Uncertainty Utilities

Production-safe uncertainty utilities.

These utilities calculate uncertainty from already-produced
probabilities. They do not retrain models, fit calibration parameters,
or modify research results.
"""

from typing import Dict

import numpy as np


def binary_entropy(
    probabilities,
):
    """
    Calculate binary entropy for each probability.

    Entropy is normalized by log(2), producing values in [0, 1].
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0 - 1e-8
    )

    entropy = -(
        probabilities
        * np.log(probabilities)
        + (
            1.0 - probabilities
        )
        * np.log(
            1.0 - probabilities
        )
    )

    return entropy / np.log(2.0)


def confidence_uncertainty(
    probabilities,
):
    """
    Convert binary confidence into uncertainty.

    For binary probabilities, uncertainty is:

        1 - max(p, 1-p)
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    return 1.0 - confidence


def probability_margin(
    probabilities,
):
    """
    Calculate binary probability margin.

    Larger values indicate greater separation from the decision
    boundary at 0.5.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    return np.abs(
        probabilities - 0.5
    )


def multiclass_entropy(
    probabilities,
    normalized: bool = True,
):
    """
    Calculate entropy for multiclass probability vectors.

    Parameters
    ----------
    probabilities:
        Array shaped (..., number_of_classes).

    normalized:
        Normalize by log(number_of_classes).
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if probabilities.ndim < 2:

        raise ValueError(
            "Multiclass probabilities must have "
            "at least two dimensions."
        )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    entropy = -np.sum(
        probabilities
        * np.log(probabilities),
        axis=-1
    )

    if not normalized:

        return entropy

    classes = probabilities.shape[-1]

    if classes <= 1:

        return np.zeros_like(
            entropy
        )

    return (
        entropy
        / np.log(classes)
    )


def uncertainty_summary(
    probabilities,
) -> Dict[str, float]:
    """
    Produce a compact binary uncertainty summary.
    """

    entropy = binary_entropy(
        probabilities
    )

    uncertainty = confidence_uncertainty(
        probabilities
    )

    margin = probability_margin(
        probabilities
    )

    return {
        "mean_entropy":
            float(np.mean(entropy)),

        "median_entropy":
            float(np.median(entropy)),

        "mean_confidence_uncertainty":
            float(np.mean(uncertainty)),

        "median_confidence_uncertainty":
            float(np.median(uncertainty)),

        "mean_probability_margin":
            float(np.mean(margin)),

        "median_probability_margin":
            float(np.median(margin)),
    }
