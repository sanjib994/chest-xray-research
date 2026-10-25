"""
VinBigData_XAI — Calibration Utilities

Production-safe calibration utilities.

This module provides reusable mathematical utilities for calibration
analysis. It does not fit a calibration model automatically and does
not modify research calibration results.
"""

from typing import Optional, Tuple

import numpy as np


def expected_calibration_error(
    probabilities,
    targets,
    n_bins: int = 10,
):
    """
    Calculate binary Expected Calibration Error (ECE).

    Parameters
    ----------
    probabilities:
        Predicted probabilities.

    targets:
        Binary ground-truth labels.

    n_bins:
        Number of equal-width confidence bins.

    Returns
    -------
    float
        Expected Calibration Error.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    if probabilities.shape != targets.shape:

        raise ValueError(
            "probabilities and targets must "
            "have identical shapes."
        )

    if probabilities.size == 0:

        return 0.0

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    n_bins = int(n_bins)

    if n_bins <= 0:

        raise ValueError(
            "n_bins must be positive."
        )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    total = probabilities.size

    for index in range(
        n_bins
    ):

        lower = bin_edges[index]
        upper = bin_edges[index + 1]

        if index == n_bins - 1:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):

            continue

        confidence = float(
            probabilities[mask].mean()
        )

        accuracy = float(
            targets[mask].mean()
        )

        weight = (
            float(mask.sum())
            / float(total)
        )

        ece += (
            weight
            * abs(
                accuracy - confidence
            )
        )

    return float(ece)


def brier_score(
    probabilities,
    targets,
):
    """
    Calculate the binary Brier score.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    if probabilities.shape != targets.shape:

        raise ValueError(
            "probabilities and targets must "
            "have identical shapes."
        )

    if probabilities.size == 0:

        return 0.0

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    return float(
        np.mean(
            (
                probabilities
                - targets
            ) ** 2
        )
    )


def reliability_bins(
    probabilities,
    targets,
    n_bins: int = 10,
):
    """
    Return reliability-bin statistics.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    if probabilities.shape != targets.shape:

        raise ValueError(
            "probabilities and targets must "
            "have identical shapes."
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    edges = np.linspace(
        0.0,
        1.0,
        int(n_bins) + 1
    )

    rows = []

    for index in range(
        int(n_bins)
    ):

        lower = edges[index]
        upper = edges[index + 1]

        if index == int(n_bins) - 1:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            rows.append(
                {
                    "bin":
                        index,
                    "count":
                        0,
                    "confidence":
                        None,
                    "accuracy":
                        None,
                }
            )

            continue

        rows.append(
            {
                "bin":
                    index,

                "count":
                    count,

                "confidence":
                    float(
                        probabilities[
                            mask
                        ].mean()
                    ),

                "accuracy":
                    float(
                        targets[
                            mask
                        ].mean()
                    ),
            }
        )

    return rows
