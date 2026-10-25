"""
VinBigData_XAI — Robustness Utilities

Production-safe robustness helpers.

This module provides deterministic image perturbation utilities and
prediction-difference measurements. It does not automatically run
official-test robustness experiments.
"""

from typing import Dict

import numpy as np


def add_gaussian_noise(
    image,
    sigma: float = 0.02,
    seed: int = 42,
):
    """
    Add deterministic Gaussian noise to an image.

    The image is expected to be represented in [0, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if sigma < 0:

        raise ValueError(
            "sigma must be non-negative."
        )

    rng = np.random.default_rng(
        seed
    )

    noise = rng.normal(
        loc=0.0,
        scale=float(sigma),
        size=image.shape
    )

    perturbed = (
        image + noise
    )

    return np.clip(
        perturbed,
        0.0,
        1.0
    )


def adjust_brightness(
    image,
    factor: float = 1.0,
):
    """
    Deterministically adjust image brightness.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if factor < 0:

        raise ValueError(
            "factor must be non-negative."
        )

    return np.clip(
        image * float(factor),
        0.0,
        1.0
    )


def adjust_contrast(
    image,
    factor: float = 1.0,
):
    """
    Deterministically adjust image contrast around its mean.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if factor < 0:

        raise ValueError(
            "factor must be non-negative."
        )

    mean = float(
        image.mean()
    )

    result = (
        mean
        + float(factor)
        * (
            image - mean
        )
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


def prediction_shift(
    original_probabilities,
    perturbed_probabilities,
) -> Dict[str, float]:
    """
    Quantify prediction change caused by a perturbation.
    """

    original = np.asarray(
        original_probabilities,
        dtype=np.float64
    )

    perturbed = np.asarray(
        perturbed_probabilities,
        dtype=np.float64
    )

    if original.shape != perturbed.shape:

        raise ValueError(
            "Original and perturbed predictions "
            "must have identical shapes."
        )

    difference = np.abs(
        perturbed - original
    )

    return {
        "mean_absolute_shift":
            float(
                np.mean(difference)
            ),

        "maximum_absolute_shift":
            float(
                np.max(difference)
            ),

        "median_absolute_shift":
            float(
                np.median(difference)
            ),
    }


def prediction_agreement(
    original_probabilities,
    perturbed_probabilities,
    threshold: float = 0.5,
):
    """
    Calculate binary prediction agreement.
    """

    original = np.asarray(
        original_probabilities,
        dtype=np.float64
    )

    perturbed = np.asarray(
        perturbed_probabilities,
        dtype=np.float64
    )

    original_labels = (
        original >= threshold
    )

    perturbed_labels = (
        perturbed >= threshold
    )

    return float(
        np.mean(
            original_labels
            == perturbed_labels
        )
    )
