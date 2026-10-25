"""
VinBigData_XAI — Production Dataset Layer

This module reads research CSV definitions without modifying them.

No images are copied or moved by this module.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    load_image,
    build_inference_transform,
)


CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
]


class ChestXrayDataset(Dataset):
    """
    Read-only chest X-ray dataset wrapper.
    """

    def __init__(
        self,
        csv_path,
        image_root=None,
        transform=None,
        id_column="image_id",
    ):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}"
            )

        self.df = pd.read_csv(
            self.csv_path
        )

        self.image_root = (
            Path(image_root)
            if image_root is not None
            else None
        )

        self.id_column = id_column

        if self.id_column not in self.df.columns:

            raise ValueError(
                f"ID column '{self.id_column}' "
                f"not found in {self.csv_path}"
            )

        column_lookup = {
            str(column).strip().lower(): column
            for column in self.df.columns
        }
        missing_classes = [
            name for name in CLASS_NAMES
            if name.lower() not in column_lookup
        ]
        if missing_classes:
            raise ValueError(
                f"Missing class columns in {self.csv_path}: {missing_classes}"
            )
        self.class_columns = [
            column_lookup[name.lower()] for name in CLASS_NAMES
        ]

        self.transform = (
            transform
            if transform is not None
            else build_inference_transform()
        )

    def __len__(self):

        return len(self.df)

    def _resolve_image_path(self, image_id):

        image_id = str(image_id)

        candidates = []

        if self.image_root is not None:

            candidates.extend([
                self.image_root / image_id,
                self.image_root / f"{image_id}.jpg",
                self.image_root / f"{image_id}.jpeg",
                self.image_root / f"{image_id}.png",
            ])

        for path in candidates:

            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not resolve image ID "
            f"'{image_id}' from image root "
            f"'{self.image_root}'."
        )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = str(
            row[self.id_column]
        )

        image_path = (
            self._resolve_image_path(
                image_id
            )
        )

        image = load_image(
            image_path
        )

        image_tensor = self.transform(
            image
        )

        target = torch.tensor(
            [
                float(row[column])
                for column in self.class_columns
            ],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "target": target,
            "image_id": image_id,
            "image_path": str(image_path),
        }
