"""Inference interface for the validated prototype research checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


DEFAULT_NUM_CLASSES = 7
DEFAULT_PROTOTYPE_DIMENSION = 128
DEFAULT_PROTOTYPES_PER_CLASS = 10
BACKBONE_FEATURE_DIMENSION = 1024


class GenuinePrototypeNetwork(nn.Module):
    """DenseNet feature maps followed by learnable prototype reasoning."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = DEFAULT_NUM_CLASSES,
        prototype_dimension: int = DEFAULT_PROTOTYPE_DIMENSION,
        prototypes_per_class: int = DEFAULT_PROTOTYPES_PER_CLASS,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.prototype_dimension = int(prototype_dimension)
        self.prototypes_per_class = int(prototypes_per_class)
        self.num_prototypes = self.num_classes * self.prototypes_per_class
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(
                BACKBONE_FEATURE_DIMENSION,
                self.prototype_dimension,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.prototype_vectors = nn.Parameter(
            torch.empty(
                self.num_prototypes,
                self.prototype_dimension,
                1,
                1,
            )
        )
        identity = torch.zeros(
            self.num_prototypes,
            self.num_classes,
            dtype=torch.float32,
        )
        for index in range(self.num_prototypes):
            identity[index, index // self.prototypes_per_class] = 1.0
        self.register_buffer("prototype_class_identity", identity)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes)

    def feature_maps(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.backbone.features(inputs), inplace=False)
        return self.add_on_layers(features)

    def prototype_distances(self, projected: torch.Tensor) -> torch.Tensor:
        x_squared = projected.pow(2).sum(dim=1, keepdim=True)
        p_squared = self.prototype_vectors.pow(2).sum(dim=1).view(
            1, self.num_prototypes, 1, 1
        )
        cross = F.conv2d(projected, self.prototype_vectors)
        return torch.clamp(x_squared + p_squared - 2.0 * cross, min=0.0)

    def similarities_from_distances(
        self, distances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        minimum = distances.flatten(start_dim=2).min(dim=2).values
        return 1.0 / (1.0 + minimum), minimum

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.feature_maps(inputs)
        distances = self.prototype_distances(projected)
        similarities, minimum = self.similarities_from_distances(distances)
        return {
            "logits": self.last_layer(similarities),
            "similarities": similarities,
            "min_distances": minimum,
            "projected_features": projected,
            "distances": distances,
        }


def create_prototype_network(num_classes: int = DEFAULT_NUM_CLASSES) -> GenuinePrototypeNetwork:
    backbone = models.densenet121(weights=None)
    backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
    return GenuinePrototypeNetwork(backbone, num_classes=num_classes)


def load_prototype_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int = DEFAULT_NUM_CLASSES,
    device: str = "cpu",
) -> GenuinePrototypeNetwork:
    """Load only a genuine prototype checkpoint; never writes to it."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prototype checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise RuntimeError("Checkpoint does not contain a prototype state_dict.")

    state_dict = checkpoint["state_dict"]
    required = {
        "prototype_vectors",
        "prototype_class_identity",
        "add_on_layers.0.weight",
        "last_layer.weight",
    }
    missing = required.difference(state_dict)
    if missing:
        raise RuntimeError(
            "Checkpoint is not a genuine prototype checkpoint; missing "
            + ", ".join(sorted(missing))
        )

    model = create_prototype_network(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
