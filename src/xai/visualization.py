"""
VinBigData_XAI — Explanation Visualization Utilities

Visualization helpers only. They do not read or modify research
explanation artifacts.
"""

from pathlib import Path

import numpy as np


def normalize_heatmap(heatmap):

    if hasattr(
        heatmap,
        "detach"
    ):

        heatmap = (
            heatmap.detach()
            .cpu()
            .numpy()
        )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    minimum = float(
        np.min(heatmap)
    )

    maximum = float(
        np.max(heatmap)
    )

    if maximum - minimum < 1e-8:

        return np.zeros_like(
            heatmap
        )

    return (
        heatmap - minimum
    ) / (
        maximum - minimum
    )


def save_heatmap_array(
    heatmap,
    output_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    normalized = normalize_heatmap(
        heatmap
    )

    np.save(
        output_path,
        normalized
    )

    return output_path
