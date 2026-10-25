"""Integrated Gradients explanations for PyTorch classifiers."""

from typing import Optional

import torch


class IntegratedGradients:
    """Compute path-integrated input attributions for a model output."""

    def __init__(self, model):
        if model is None:
            raise ValueError("model must not be None.")
        self.model = model

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ):
        if steps <= 0:
            raise ValueError("steps must be positive.")

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.ndim != 4:
            raise ValueError(
                "Expected input with shape [B,C,H,W] or [C,H,W]."
            )

        input_tensor = input_tensor.detach()
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            if baseline.ndim == 3:
                baseline = baseline.unsqueeze(0)
            if baseline.shape != input_tensor.shape:
                raise ValueError(
                    "baseline and input_tensor must have identical shapes."
                )
            baseline = baseline.detach().to(input_tensor)

        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if target_class is None:
                    target_class = int(output[0].argmax().item())
                if target_class < 0 or target_class >= output.shape[1]:
                    raise ValueError("target_class is outside model output range.")
                input_score = output[0, target_class].sum().detach()
                baseline_output = self.model(baseline)
                if isinstance(baseline_output, (tuple, list)):
                    baseline_output = baseline_output[0]
                baseline_score = baseline_output[0, target_class].sum().detach()

            total_gradients = torch.zeros_like(input_tensor)
            for index in range(1, steps + 1):
                alpha = index / steps
                interpolated = (
                    baseline + alpha * (input_tensor - baseline)
                ).detach().requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                step_output = self.model(interpolated)
                if isinstance(step_output, (tuple, list)):
                    step_output = step_output[0]
                step_output[0, target_class].sum().backward()
                if interpolated.grad is None:
                    raise RuntimeError(
                        "Integrated Gradients did not capture input gradients."
                    )
                total_gradients += interpolated.grad.detach()

            attributions = (input_tensor - baseline) * total_gradients / steps
        finally:
            self.model.train(was_training)

        return {
            "class_index": target_class,
            "input_score": float(input_score.cpu().item()),
            "baseline_score": float(baseline_score.cpu().item()),
            "attributions": attributions[0].cpu(),
            "convergence_delta": float(
                (
                    attributions.sum()
                    - (input_score - baseline_score).cpu()
                ).abs().item()
            ),
        }
