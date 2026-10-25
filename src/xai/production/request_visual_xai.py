
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ...evaluation.explanation_metrics import (
    explanation_stability,
    faithfulness_deletion,
)

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
]


def _checkpoint_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    configured = os.environ.get("VINBIGDATA_CHECKPOINT")
    checkpoint = (
        Path(configured)
        if configured
        else project_root
        / "checkpoints"
        / "convnext_tiny"
        / "convnext_tiny_weak_classes_best.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    return checkpoint


def _load_production_components():
    from ...data.preprocessing import preprocess_image
    from ...models.model_loader import load_model
    from ...inference.predictor import ModelPredictor

    return preprocess_image, load_model, ModelPredictor


def _load_model():
    _, load_model, _ = _load_production_components()

    checkpoint = _checkpoint_path()
    model_name = os.environ.get("VINBIGDATA_MODEL", "convnext_tiny")

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Production checkpoint not found: {checkpoint}"
        )

    model = load_model(
        str(checkpoint),
        num_classes=7,
        device="cpu",
        model_name=model_name,
    )

    model.eval()
    return model


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalise_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cam -= cam.min()

    maximum = float(cam.max())

    if maximum > 0.0:
        cam /= maximum

    return np.clip(cam, 0.0, 1.0)


def _make_overlay(
    original: Image.Image,
    cam: np.ndarray,
) -> Image.Image:

    import matplotlib.cm as cm

    original = original.convert("RGB")

    width, height = original.size

    cam_image = Image.fromarray(
        np.uint8(_normalise_cam(cam) * 255.0),
        mode="L",
    )

    cam_image = cam_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    heat = cm.jet(
        np.asarray(cam_image, dtype=np.float32) / 255.0
    )

    heat_rgb = Image.fromarray(
        np.uint8(heat[:, :, :3] * 255.0),
        mode="RGB",
    )

    return Image.blend(
        original,
        heat_rgb,
        alpha=0.45,
    )


def _make_attention_box(
    original: Image.Image,
    cam: np.ndarray,
) -> Image.Image:

    from PIL import ImageDraw

    original = original.convert("RGB")

    width, height = original.size

    cam_norm = _normalise_cam(cam)

    threshold = float(np.percentile(cam_norm, 85.0))

    ys, xs = np.where(cam_norm >= threshold)

    result = original.copy()
    draw = ImageDraw.Draw(result)

    if len(xs) == 0 or len(ys) == 0:
        return result
    x1 = int(xs.min() / cam_norm.shape[1] * width)
    x2 = int((xs.max() + 1) / cam_norm.shape[1] * width)

    y1 = int(ys.min() / cam_norm.shape[0] * height)
    y2 = int((ys.max() + 1) / cam_norm.shape[0] * height)

    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))

    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=(255, 0, 0),
        width=max(3, width // 256),
    )

    draw.text(
        (x1 + 8, y1 + 8),
        "MODEL ATTENTION REGION",
        fill=(255, 0, 0),
    )

    return result


def _capture_integrated_gradients(
    model,
    tensor,
    class_index,
):
    from ..integrated_gradients import IntegratedGradients

    result = IntegratedGradients(model).generate(
        tensor,
        target_class=class_index,
        steps=32,
    )

    attributions = result["attributions"]
    heatmap = attributions.abs().sum(dim=0).numpy()
    return heatmap, result["convergence_delta"]


def _capture_cam(model, tensor, class_index):
    """
    Safe Grad-CAM implementation.

    We use a forward hook that RETURNS a cloned tensor.
    This prevents the inplace/view autograd error encountered by
    the previous implementation.
    """

    if hasattr(model.features, "norm5"):
        target_layer = model.features.norm5
    else:
        target_layer = model.features[-1]

    captured = {
        "activation": None,
    }

    def forward_hook(module, inputs, output):
        cloned = output.clone()

        captured["activation"] = cloned

        cloned.retain_grad()

        return cloned

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        output = model(tensor)

        if output.ndim == 1:
            output = output.unsqueeze(0)

        if class_index < 0 or class_index >= output.shape[1]:
            raise ValueError(
                f"Invalid target class index: {class_index}"
            )

        score = output[:, class_index].sum()

        score.backward()

        activation = captured["activation"]

        if activation is None:
            raise RuntimeError(
                "Target activation was not captured."
            )

        gradient = activation.grad

        if gradient is None:
            raise RuntimeError(
                "Target activation gradient was not captured."
            )

        activation_data = activation.detach()
        gradient_data = gradient.detach()

        if activation_data.ndim != 4:
            raise RuntimeError(
                f"Unexpected target activation shape: "
                f"{tuple(activation_data.shape)}"
            )

        weights = gradient_data.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activation_data
        ).sum(dim=1)[0]

        cam = torch.relu(cam)

        return cam.cpu().numpy()

    finally:
        handle.remove()


def _capture_gradcam_plus_plus(
    model,
    tensor,
    class_index,
):
    """
    Gradient-based Grad-CAM++ approximation using the
    same safe cloned forward activation.
    """

    if hasattr(model.features, "norm5"):
        target_layer = model.features.norm5
    else:
        target_layer = model.features[-1]

    captured = {
        "activation": None,
    }

    def forward_hook(module, inputs, output):
        cloned = output.clone()

        captured["activation"] = cloned

        cloned.retain_grad()

        return cloned

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        output = model(tensor)

        if output.ndim == 1:
            output = output.unsqueeze(0)

        score = output[:, class_index].sum()

        gradients = torch.autograd.grad(
            score,
            captured["activation"],
            create_graph=True,
            retain_graph=True,
        )[0]

        activation = captured["activation"]

        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)

        summed_activation = activation.sum(
            dim=(2, 3),
            keepdim=True,
        )

        denominator = (
            2.0 * grad2
            + grad3 * summed_activation
            + 1e-8
        )

        alpha = grad2 / denominator

        positive_gradients = torch.relu(
            gradients
        )

        weights = (
            alpha * positive_gradients
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activation
        ).sum(dim=1)[0]

        cam = torch.relu(cam)

        return cam.detach().cpu().numpy()

    finally:
        handle.remove()


def generate_request_visual_xai(
    image_bytes: bytes,
    filename: str = "uploaded_xray.png",
    target_class: Optional[int] = None,
) -> Dict[str, Any]:

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError(
            "Uploaded image exceeds the 10 MB production limit."
        )

    original = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    with tempfile.NamedTemporaryFile(
        suffix=Path(filename).suffix or ".png",
        delete=False,
    ) as temporary:

        temporary.write(image_bytes)
        temporary_path = Path(temporary.name)

    try:
        preprocess_image, _, ModelPredictor = (
            _load_production_components()
        )

        model = _load_model()

        tensor = preprocess_image(
            str(temporary_path)
        )

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(
                tensor,
                dtype=torch.float32,
            )

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            dtype=torch.float32,
            device="cpu",
        )

        predictor = ModelPredictor(
            model=model,
            diseases=DISEASES,
            device="cpu",
        )

        with torch.no_grad():
            prediction = predictor.predict_tensor(
                tensor
            )

        if not isinstance(prediction, dict):
            raise RuntimeError(
                "Production predictor did not return a dictionary."
            )

        probabilities = []

        for disease in DISEASES:
            value = prediction.get(disease)

            if value is None:
                raise RuntimeError(
                    f"Missing disease probability: {disease}"
                )

            probabilities.append(float(value))

        if target_class is None:
            target_index = int(
                np.argmax(probabilities)
            )
        else:
            target_index = int(target_class)

        target_index = max(
            0,
            min(
                len(DISEASES) - 1,
                target_index,
            ),
        )

        target_disease = DISEASES[target_index]

        # Grad-CAM and Grad-CAM++ require gradients.
        cam = _capture_cam(
            model,
            tensor,
            target_index,
        )

        cam_pp = _capture_gradcam_plus_plus(
            model,
            tensor,
            target_index,
        )

        integrated_gradients, ig_delta = (
            _capture_integrated_gradients(
                model,
                tensor,
                target_index,
            )
        )

        cam = _normalise_cam(cam)
        cam_pp = _normalise_cam(cam_pp)
        integrated_gradients = _normalise_cam(
            integrated_gradients
        )

        gradcam_overlay = _make_overlay(
            original,
            cam,
        )

        gradcam_pp_overlay = _make_overlay(
            original,
            cam_pp,
        )

        gradcam_box = _make_attention_box(
            original,
            cam,
        )

        gradcam_pp_box = _make_attention_box(
            original,
            cam_pp,
        )

        integrated_gradients_overlay = _make_overlay(
            original,
            integrated_gradients,
        )

        faithfulness = faithfulness_deletion(
            model,
            tensor,
            integrated_gradients,
            target_index,
            fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        )

        def integrated_gradients_attribution(input_value):
            result = _capture_integrated_gradients(
                model,
                input_value,
                target_index,
            )[0]
            return result

        stability = explanation_stability(
            tensor,
            integrated_gradients_attribution,
            repeats=3,
        )

        return {
            "prediction": {
                disease: probabilities[index]
                for index, disease in enumerate(DISEASES)
            },
            "target": {
                "class_index": target_index,
                "disease": target_disease,
                "probability": probabilities[target_index],
            },
            "visual_xai": {
                "gradcam": _encode_png(
                    gradcam_overlay
                ),
                "gradcam_plus_plus": _encode_png(
                    gradcam_pp_overlay
                ),
                "gradcam_attention_box": _encode_png(
                    gradcam_box
                ),
                "gradcam_plus_plus_attention_box": _encode_png(
                    gradcam_pp_box
                ),
                "integrated_gradients": _encode_png(
                    integrated_gradients_overlay
                ),
            },
            "metadata": {
                "synthetic": False,
                "official_test": False,
                "production_checkpoint": True,
                "training": False,
                "threshold_tuning": False,
                "request_specific": True,
                "static_artifact_used": False,
                "medical_lesion_boundary": False,
                "attention_label": (
                    "MODEL ATTENTION REGION"
                ),
                "integrated_gradients": {
                    "enabled": True,
                    "steps": 32,
                    "convergence_delta": ig_delta,
                },
                "evaluation": {
                    "localization": {
                        "available": False,
                        "reason": (
                            "A ground-truth lesion mask was not supplied "
                            "with this uploaded image."
                        ),
                    },
                    "faithfulness": faithfulness,
                    "stability": stability,
                },
            },
        }

    finally:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass
