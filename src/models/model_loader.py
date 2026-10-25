"""
VinBigData_XAI production model loader.

Inference-only loading of supported project classifier checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .densenet121 import (
    create_densenet121,
    load_densenet121_checkpoint,
)

MODEL_NAME_ALIASES = {
    "densenet": "densenet121",
    "densenet121": "densenet121",
    "convnext": "convnext_tiny",
    "convnext_tiny": "convnext_tiny",
    "convnexttiny": "convnext_tiny",
}

def create_convnext_tiny(num_classes: int = 7):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def load_convnext_tiny_checkpoint(
    checkpoint_path,
    num_classes: int = 7,
    device: str = "cpu",
    strict: bool = True,
):
    checkpoint_path = Path(checkpoint_path)
    model = create_convnext_tiny(num_classes=num_classes)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights", "network", "net"):
            if isinstance(checkpoint.get(key), dict):
                state_dict = checkpoint[key]
                break

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid state dictionary.")

    cleaned = {}
    for key, value in state_dict.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=strict)
    model.to(device)
    model.eval()
    return model

def resolve_model_name(model_name: str | None) -> str:
    key = (model_name or "densenet121").lower()
    return MODEL_NAME_ALIASES.get(key, "densenet121")

def load_model(
    checkpoint_path,
    num_classes: int = 7,
    device: str = "cpu",
    model_name: str = "densenet121",
    strict: bool = True,
):
    model_key = resolve_model_name(model_name)

    if model_key == "convnext_tiny":
        return load_convnext_tiny_checkpoint(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            device=device,
            strict=strict,
        )

    return load_densenet121_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
        strict=strict,
    )

def build_model(
    num_classes: int = 7,
    device: str = "cpu",
    model_name: str = "densenet121",
):
    model_key = resolve_model_name(model_name)

    if model_key == "convnext_tiny":
        model = create_convnext_tiny(num_classes=num_classes)
    else:
        model = create_densenet121(num_classes=num_classes)

    model.to(device)
    model.eval()
    return model
