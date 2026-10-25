"""
VinBigData_XAI — Grad-CAM++

Production implementation wrapper.

The implementation is intentionally explicit: it performs Grad-CAM++
when requested by the production pipeline and does not generate
research artifacts automatically.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    """
    Grad-CAM++ explanation generator.
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
                "Grad-CAM++ hooks did not capture "
                "activations and gradients."
            )

        activations = self.activations
        gradients = self.gradients

        gradients_squared = (
            gradients ** 2
        )

        gradients_cubed = (
            gradients ** 3
        )

        denominator = (
            2.0 * gradients_squared
            + (
                gradients_cubed
                * activations.sum(
                    dim=(2, 3),
                    keepdim=True
                )
            )
        )

        alpha = (
            gradients_squared
            / (
                denominator + 1e-8
            )
        )

        positive_gradients = F.relu(
            gradients
        )

        weights = (
            alpha
            * positive_gradients
        ).sum(
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
