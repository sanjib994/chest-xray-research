"""Provenance-verified temperature scaling for the two production models.

Calibration parameters are fit on ``final_val_split.csv`` only.  Metrics in
the report are computed once on the untouched ``final_test_split.csv``.
The report contains SHA-256 hashes for every input artifact and is rejected by
the API if any of those inputs changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES
from src.data.preprocessing import build_inference_transform
from src.models.model_loader import load_model
from src.models.prototype_network import load_prototype_checkpoint
from .calibration import expected_calibration_error, brier_score


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _columns(frame: pd.DataFrame) -> list[str]:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    missing = [name for name in CLASS_NAMES if name.lower() not in lookup]
    # The source files use "Pleural effusion" while the product name is
    # title-cased; accept that documented spelling without guessing labels.
    if "Pleural Effusion" in missing and "pleural effusion" in lookup:
        missing.remove("Pleural Effusion")
        lookup["pleural effusion"] = lookup["pleural effusion"]
    if missing:
        raise ValueError(f"Split is missing class columns: {missing}")
    return [lookup[name.lower()] for name in CLASS_NAMES]


def _dataset(csv_path: Path, image_root: Path):
    from src.data.dataset import ChestXrayDataset

    return ChestXrayDataset(
        csv_path=csv_path,
        image_root=image_root,
        transform=build_inference_transform(),
    )


def _logits(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    outputs, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))
            if isinstance(output, dict):
                output = output["logits"]
            outputs.append(output.detach().cpu().numpy())
            targets.append(batch["target"].numpy())
    if not outputs:
        raise ValueError("Split contains no rows.")
    return np.concatenate(outputs), np.concatenate(targets)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit one positive sigmoid temperature per label using validation data."""
    values = []
    for index in range(logits.shape[1]):
        parameter = torch.nn.Parameter(torch.zeros(1))
        x = torch.as_tensor(logits[:, index], dtype=torch.float64)
        y = torch.as_tensor(targets[:, index], dtype=torch.float64)
        optimizer = torch.optim.LBFGS([parameter], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(parameter.clamp(-10.0, 10.0)).clamp_min(1e-3)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x / temperature, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        values.append(float(torch.exp(parameter.clamp(-10.0, 10.0)).clamp_min(1e-3).item()))
    return np.asarray(values, dtype=np.float64)


def _metrics(logits: np.ndarray, targets: np.ndarray, temperatures: np.ndarray, n_bins: int) -> dict[str, Any]:
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperatures))
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "ece": expected_calibration_error(probabilities[:, index], targets[:, index], n_bins),
            "brier": brier_score(probabilities[:, index], targets[:, index]),
            "count": int(len(targets)),
        }
    return {
        "per_class": per_class,
        "macro_ece": float(np.mean([row["ece"] for row in per_class.values()])),
        "macro_brier": float(np.mean([row["brier"] for row in per_class.values()])),
        "n_samples": int(len(targets)),
        "n_bins": int(n_bins),
    }


def run_calibration(
    *,
    val_csv: str | Path,
    test_csv: str | Path,
    image_root: str | Path,
    densenet_checkpoint: str | Path,
    prototype_checkpoint: str | Path,
    output: str | Path,
    batch_size: int = 16,
    device: str = "cpu",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run both models and write a self-describing, provenance-bound report."""
    val_csv, test_csv, image_root = Path(val_csv), Path(test_csv), Path(image_root)
    dense_path, proto_path, output = Path(densenet_checkpoint), Path(prototype_checkpoint), Path(output)
    val_frame, test_frame = pd.read_csv(val_csv), pd.read_csv(test_csv)
    val_columns, test_columns = _columns(val_frame), _columns(test_frame)
    if val_columns != test_columns:
        raise ValueError("Validation and test class-column order differs.")

    def loader(csv_path: Path):
        dataset = _dataset(csv_path, image_root)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    val_loader, test_loader = loader(val_csv), loader(test_csv)
    models = {
        "densenet121": load_model(dense_path, num_classes=len(CLASS_NAMES), device=device),
        "prototype": load_prototype_checkpoint(proto_path, num_classes=len(CLASS_NAMES), device=device),
    }
    reports = {}
    for name, model in models.items():
        val_logits, val_targets = _logits(model, val_loader, device)
        test_logits, test_targets = _logits(model, test_loader, device)
        temperatures = fit_temperature(val_logits, val_targets)
        reports[name] = {
            "temperature": temperatures.tolist(),
            "validation": _metrics(val_logits, val_targets, temperatures, n_bins),
            "test": _metrics(test_logits, test_targets, temperatures, n_bins),
        }

    artifact = {
        "schema_version": "calibration.v1",
        "method": "per-class sigmoid temperature scaling",
        "fit_split": str(val_csv),
        "evaluation_split": str(test_csv),
        "image_root": str(image_root),
        "densenet_checkpoint": str(dense_path),
        "prototype_checkpoint": str(proto_path),
        "class_names": list(CLASS_NAMES),
        "provenance": {
            "validation_csv_sha256": sha256_file(val_csv),
            "test_csv_sha256": sha256_file(test_csv),
            "densenet_checkpoint_sha256": sha256_file(dense_path),
            "prototype_checkpoint_sha256": sha256_file(proto_path),
            "validation_rows": int(len(val_frame)),
            "test_rows": int(len(test_frame)),
            "image_root_exists": image_root.is_dir(),
        },
        "models": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def load_verified_report(path: str | Path) -> dict[str, Any] | None:
    """Load only a report whose recorded files and hashes still match."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        provenance = report["provenance"]
        checks = (
            ("validation_csv", provenance["validation_csv_sha256"]),
            ("test_csv", provenance["test_csv_sha256"]),
            ("densenet_checkpoint", provenance["densenet_checkpoint_sha256"]),
            ("prototype_checkpoint", provenance["prototype_checkpoint_sha256"]),
        )
        roots = {
            "validation_csv": report["fit_split"],
            "test_csv": report["evaluation_split"],
            "densenet_checkpoint": report.get("densenet_checkpoint", ""),
            "prototype_checkpoint": report.get("prototype_checkpoint", ""),
        }
        # Older/incomplete artifacts are intentionally unavailable.
        if not all(roots[key] for key, _ in checks):
            return None
        if not all(sha256_file(roots[key]) == expected for key, expected in checks):
            return None
        if not Path(report["image_root"]).is_dir():
            return None
        return report
    except (KeyError, OSError, ValueError, TypeError):
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--densenet-checkpoint", required=True)
    parser.add_argument("--prototype-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = run_calibration(**vars(args))
    print(json.dumps({"output": args.output, "models": list(report["models"])}, indent=2))


if __name__ == "__main__":
    main()
