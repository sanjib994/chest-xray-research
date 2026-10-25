"""Metrics for evaluating post-hoc explanations.

The functions in this module operate on supplied predictions, masks, and
attribution generators. They do not load checkpoints or mutate research data.
"""

from typing import Callable, Iterable, Optional

import numpy as np
import torch


def _normalise(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Explanation must not be empty.")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum <= 1e-12:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def localization_metrics(
    attribution,
    ground_truth_mask,
    threshold: float = 0.5,
):
    """Measure overlap between an explanation and a binary lesion mask."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    heatmap = _normalise(attribution)
    mask = np.asarray(ground_truth_mask).astype(bool)
    if heatmap.shape != mask.shape:
        raise ValueError("attribution and ground_truth_mask must match.")

    predicted = heatmap >= threshold
    intersection = int(np.logical_and(predicted, mask).sum())
    union = int(np.logical_or(predicted, mask).sum())
    mask_pixels = int(mask.sum())
    return {
        "iou": float(intersection / union) if union else 0.0,
        "energy_coverage": float(heatmap[mask].sum() / heatmap.sum())
        if mask_pixels and heatmap.sum() > 0
        else 0.0,
        "pointing_game": bool(mask.flat[int(np.argmax(heatmap))])
        if mask_pixels
        else False,
        "predicted_positive_fraction": float(predicted.mean()),
    }


def _target_probability(model, tensor: torch.Tensor, target_class: int) -> float:
    output = model(tensor)
    if isinstance(output, dict):
        output = output.get("logits")
        if output is None:
            raise ValueError("Model output dictionary does not contain logits.")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 1:
        output = output.unsqueeze(0)
    if not 0 <= target_class < output.shape[1]:
        raise ValueError("target_class is outside model output range.")
    return float(torch.sigmoid(output[0, target_class]).detach().cpu().item())


def faithfulness_deletion(
    model,
    input_tensor: torch.Tensor,
    attribution,
    target_class: int,
    fractions: Iterable[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    baseline: Optional[torch.Tensor] = None,
):
    """Evaluate prediction decrease as the most important pixels are removed."""
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    if input_tensor.ndim != 4:
        raise ValueError("input_tensor must have shape [B,C,H,W] or [C,H,W].")
    if input_tensor.shape[0] != 1:
        raise ValueError("faithfulness_deletion currently supports one image.")

    heatmap = _normalise(attribution)
    if heatmap.shape != tuple(input_tensor.shape[-2:]):
        raise ValueError("attribution must match input spatial dimensions.")
    values = tuple(float(fraction) for fraction in fractions)
    if not values or any(fraction < 0.0 or fraction > 1.0 for fraction in values):
        raise ValueError("fractions must be non-empty values between 0 and 1.")

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    if baseline.ndim == 3:
        baseline = baseline.unsqueeze(0)
    if baseline.shape != input_tensor.shape:
        raise ValueError("baseline and input_tensor must have identical shapes.")

    flat_order = np.argsort(heatmap.reshape(-1))[::-1]
    total_pixels = flat_order.size
    scores = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            original_score = _target_probability(
                model,
                input_tensor,
                target_class,
            )
            for fraction in values:
                count = int(round(fraction * total_pixels))
                perturbed = input_tensor.clone()
                if count:
                    indices = torch.as_tensor(
                        flat_order[:count].copy(),
                        dtype=torch.long,
                    )
                    rows = torch.div(indices, heatmap.shape[1], rounding_mode="floor")
                    columns = indices.remainder(heatmap.shape[1])
                    perturbed[0, :, rows, columns] = baseline[0, :, rows, columns]
                scores.append(
                    _target_probability(
                        model,
                        perturbed,
                        target_class,
                    )
                )
    finally:
        model.train(was_training)

    drops = [original_score - score for score in scores]
    area = float(np.trapezoid(scores, values)) if len(values) > 1 else 0.0
    faithfulness_score = max(
        0.0,
        min(
            1.0,
            1.0 - area / max(original_score, 1e-12),
        ),
    )
    return {
        "fractions": list(values),
        "scores": scores,
        "original_score": original_score,
        "final_score": scores[-1],
        "mean_drop": float(np.mean(drops)),
        "deletion_auc": area,
        "normalized_deletion_auc": float(
            area / max(original_score, 1e-12)
        ),
        "faithfulness_score": float(faithfulness_score),
        "score_type": "sigmoid_probability",
    }


def explanation_stability(
    input_tensor: torch.Tensor,
    attribution_fn: Callable[[torch.Tensor], object],
    perturbation_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    repeats: int = 5,
):
    """Measure attribution similarity under repeated deterministic perturbations."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2.")
    if perturbation_fn is None:
        perturbation_fn = lambda tensor, seed: tensor + torch.randn(
            tensor.shape, generator=torch.Generator(device=tensor.device).manual_seed(seed),
            device=tensor.device,
        ) * 0.01

    reference = _normalise(attribution_fn(input_tensor))
    similarities = []
    for seed in range(repeats):
        perturbed = perturbation_fn(input_tensor, seed)
        current = _normalise(attribution_fn(perturbed))
        if current.shape != reference.shape:
            raise ValueError("Attributions must have identical shapes.")
        reference_flat = reference.reshape(-1)
        current_flat = current.reshape(-1)
        denominator = np.linalg.norm(reference_flat) * np.linalg.norm(current_flat)
        similarities.append(
            float(np.dot(reference_flat, current_flat) / denominator)
            if denominator > 1e-12
            else 1.0
        )

    return {
        "repeats": repeats,
        "cosine_similarities": similarities,
        "mean_similarity": float(np.mean(similarities)),
        "minimum_similarity": float(np.min(similarities)),
    }
