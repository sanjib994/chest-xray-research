"""
VinBigData_XAI — Grad-CAM

Production-safe Grad-CAM implementation.

This module generates explanations only when explicitly called by
the product inference layer. It does not automatically run inference,
save research artifacts, or modify research results.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM explanation generator.

    Parameters
    ----------
    model:
        PyTorch classification model.
    target_layer:
        Layer whose activations are used for Grad-CAM.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self._backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        if grad_output:

            self.gradients = (
                grad_output[0]
            )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ):

        if input_tensor.ndim == 3:

            input_tensor = (
                input_tensor.unsqueeze(0)
            )

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            input_tensor
        )

        if target_class is None:

            target_class = int(
                output.argmax(
                    dim=1
                ).item()
            )

        score = output[
            0,
            target_class
        ]

        score.backward(
            retain_graph=True
        )

        if (
            self.activations is None
            or self.gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "activations and gradients."
            )

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_min = cam.amin(
            dim=(2, 3),
            keepdim=True
        )

        cam_max = cam.amax(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return {
            "class_index": target_class,
            "score": float(
                score.detach().cpu().item()
            ),
            "heatmap": cam[
                0,
                0
            ].detach().cpu(),
        }

    def close(self):

        self._forward_handle.remove()
        self._backward_handle.remove()
