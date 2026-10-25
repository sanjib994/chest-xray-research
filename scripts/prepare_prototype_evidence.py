"""Prepare local prototype evidence images without copying source identifiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "prototype_evidence"


def _source_image(value: str, source_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source_root / path
    path = path.resolve()
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Evidence image is outside source root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Evidence image not found: {path}")
    return path


def prepare_evidence_images(
    manifest_path: Path,
    source_root: Path,
    output_dir: Path = DEFAULT_OUTPUT,
) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Research manifest not found: {manifest_path}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Research source directory not found: {source_root}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"prototype_index", "full_evidence_image", "query_patch_image"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Research manifest does not contain required image columns.")
        rows = list(reader)

    if not rows:
        raise ValueError("Research manifest contains no prototype records.")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        index = int(row["prototype_index"])
        for field, suffix in (
            ("full_evidence_image", "full_evidence"),
            ("query_patch_image", "query_patch"),
        ):
            source_value = row[field].strip()
            if not source_value:
                raise ValueError(
                    f"Prototype {index} has no source image in {field}."
                )
            source = _source_image(source_value, source_root)
            destination = output_dir / f"prototype_{index:03d}_{suffix}.png"
            with Image.open(source) as image:
                image.convert("RGB").save(destination, format="PNG", optimize=True)
            copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy only prototype evidence images from an authorized local "
            "research dataset into the app's ignored evidence folder."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Original research prototype evidence manifest.",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Root directory that contains all source images referenced by the manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output folder (default: {DEFAULT_OUTPUT}).",
    )
    arguments = parser.parse_args()
    copied = prepare_evidence_images(
        manifest_path=arguments.manifest,
        source_root=arguments.source_root,
        output_dir=arguments.output_dir,
    )
    print(f"Prepared {copied} prototype evidence images in {arguments.output_dir}.")
    print("Review privacy and redistribution permissions before sharing these files.")


if __name__ == "__main__":
    main()
