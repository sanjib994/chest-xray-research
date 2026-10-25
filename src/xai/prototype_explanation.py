"""
VinBigData_XAI — Prototype Explanation

Production interface for intrinsic prototype explanations.

This module does not claim that a prototype explanation is available
for a model unless the supplied model exposes prototype information.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import csv

import torch


class PrototypeExplanation:

    def __init__(
        self,
        model,
    ):

        self.model = model

    def is_supported(self) -> bool:
        """
        Determine whether the supplied model exposes prototype data.
        """

        attributes = [
            "prototypes",
            "prototype_vectors",
            "prototype_layer",
            "prototype_activations",
        ]

        return any(
            hasattr(
                self.model,
                attribute
            )
            for attribute in attributes
        )

    def explain(
        self,
        model_output: Optional[Any] = None,
    ) -> Dict[str, Any]:

        if not self.is_supported():

            raise RuntimeError(
                "The supplied model does not expose "
                "a recognized prototype representation."
            )

        result = {
            "type":
                "prototype_intrinsic",

            "supported":
                True,
        }

        if model_output is not None:

            result[
                "model_output"
            ] = model_output

        prototype_attributes = [
            "prototypes",
            "prototype_vectors",
            "prototype_activations",
        ]

        for attribute in prototype_attributes:

            if hasattr(
                self.model,
                attribute
            ):

                value = getattr(
                    self.model,
                    attribute
                )

                if isinstance(
                    value,
                    torch.Tensor
                ):

                    result[
                        attribute
                    ] = value.detach().cpu()

                else:

                    result[
                        attribute
                    ] = value

        return result


def load_prototype_evidence_index(
    manifest_path: str | Path,
) -> Dict[int, Dict[str, str]]:
    """Load durable training-image evidence records keyed by prototype index."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prototype evidence manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"prototype_index", "prototype_id", "prototype_class",
                    "full_evidence_image", "query_patch_image"}
        if not required.issubset(rows.fieldnames or set()):
            raise RuntimeError("Prototype evidence manifest has an invalid schema.")
        return {int(row["prototype_index"]): row for row in rows}


def rank_prototypes(
    model_output: Dict[str, torch.Tensor],
    evidence_index: Dict[int, Dict[str, str]],
    top_k: int = 5,
) -> list[Dict[str, Any]]:
    """Return real prototype similarity reasoning for one model output."""
    similarities = model_output["similarities"][0].detach().cpu()
    distances = model_output["min_distances"][0].detach().cpu()
    indices = torch.argsort(similarities, descending=True)[:top_k].tolist()
    result = []
    for rank, index in enumerate(indices, start=1):
        record = evidence_index.get(int(index))
        if record is None:
            raise RuntimeError(f"No durable evidence record for prototype {index}.")
        item = {
            "rank": rank,
            "prototype_index": int(index),
            "prototype_id": record["prototype_id"],
            "prototype_class": record["prototype_class"],
            "similarity": float(similarities[index]),
            "distance": float(distances[index]),
            "prototype_evidence_image": record["full_evidence_image"],
            "prototype_query_patch": record["query_patch_image"],
        }
        if record.get("source_training_image_id"):
            item["source_training_image_id"] = record["source_training_image_id"]
        result.append(item)
    return result
