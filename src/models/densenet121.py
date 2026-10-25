"""
VinBigData_XAI production DenseNet121 model interface.

This module contains inference-only model construction.
It does not train or modify research checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


DEFAULT_NUM_CLASSES = 7


def create_densenet121(
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> nn.Module:
    """
    Create the DenseNet121 architecture used for production inference.

    No pretrained weights are requested here. The caller loads the verified
    research checkpoint explicitly.
    """

    model = models.densenet121(
        weights=None
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        int(num_classes),
    )

    return model


def load_densenet121_checkpoint(
    checkpoint_path,
    num_classes: int = DEFAULT_NUM_CLASSES,
    device: str = "cpu",
    strict: bool = True,
):
    """
    Construct DenseNet121 and load an existing checkpoint into memory.

    The checkpoint file is opened read-only and is never written.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    model = create_densenet121(
        num_classes=num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint

    if isinstance(
        checkpoint,
        dict
    ):

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "weights",
            "network",
            "net",
        ):

            candidate = checkpoint.get(
                key
            )

            if isinstance(
                candidate,
                dict
            ):

                state_dict = candidate
                break

    if not isinstance(
        state_dict,
        dict
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "a valid state dictionary."
        )

    cleaned = {}

    for key, value in state_dict.items():

        key = str(key)

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=strict,
    )

    model.to(
        device
    )

    model.eval()

    return model
