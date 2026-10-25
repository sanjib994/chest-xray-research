"""
VinBigData_XAI — Production Predictor

The predictor is deliberately model-agnostic at the product boundary.

It can accept an already-loaded PyTorch model and perform inference
on an already-preprocessed tensor.

No checkpoint is loaded automatically by this module.
No training occurs here.
"""

from typing import Dict, Iterable, Optional

import numpy as np


DEFAULT_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
]


class ModelPredictor:
    """
    Thin production inference wrapper around a loaded model.
    """

    def __init__(
        self,
        model,
        diseases=None,
        device=None,
    ):

        if model is None:

            raise ValueError(
                "model must not be None."
            )

        self.model = model

        self.diseases = list(
            diseases
            if diseases is not None
            else DEFAULT_DISEASES
        )

        self.device = device

        if hasattr(
            self.model,
            "eval"
        ):

            self.model.eval()

    def _prepare_input(
        self,
        image_tensor,
    ):

        try:

            import torch

        except ImportError as exc:

            raise RuntimeError(
                "PyTorch is required for "
                "model inference."
            ) from exc

        if not torch.is_tensor(
            image_tensor
        ):

            image_tensor = torch.as_tensor(
                image_tensor,
                dtype=torch.float32
            )

        if image_tensor.ndim == 3:

            image_tensor = (
                image_tensor
                .unsqueeze(0)
            )

        if image_tensor.ndim != 4:

            raise ValueError(
                "Expected input with shape "
                "[B,C,H,W] or [C,H,W]."
            )

        if self.device is not None:

            image_tensor = (
                image_tensor.to(
                    self.device
                )
            )

        return image_tensor

    def predict_tensor(
        self,
        image_tensor,
    ) -> Dict[str, float]:
        """
        Run deterministic inference on an already-preprocessed tensor.
        """

        try:

            import torch

        except ImportError as exc:

            raise RuntimeError(
                "PyTorch is required for "
                "model inference."
            ) from exc

        image_tensor = (
            self._prepare_input(
                image_tensor
            )
        )

        with torch.no_grad():

            output = self.model(
                image_tensor
            )

        if isinstance(
            output,
            (tuple, list)
        ):

            output = output[0]

        if not torch.is_tensor(
            output
        ):

            output = torch.as_tensor(
                output
            )

        if output.ndim == 1:

            output = output.unsqueeze(
                0
            )

        probabilities = torch.sigmoid(
            output
        )

        values = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

        if len(values) != len(
            self.diseases
        ):

            raise ValueError(
                "Model output dimension "
                "does not match disease list."
            )

        return {
            disease:
                float(value)
            for disease, value
            in zip(
                self.diseases,
                values
            )
        }


DenseNetPredictor = ModelPredictor
