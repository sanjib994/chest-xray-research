"""Generate provenance-verified calibration for DenseNet121 and Prototype."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.calibrated_evaluation import run_calibration


if __name__ == "__main__":
    research_root = os.environ.get("VINBIGDATA_RESEARCH_ROOT")
    if not research_root:
        raise SystemExit(
            "Set VINBIGDATA_RESEARCH_ROOT to the research dataset directory."
        )

    dataset = Path(research_root).expanduser()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Research dataset directory not found: {dataset}")

    report = run_calibration(
        val_csv=dataset / "processed" / "final_val_split.csv",
        test_csv=dataset / "processed" / "final_test_split.csv",
        image_root=dataset / "train",
        densenet_checkpoint=ROOT / "checkpoints" / "densenet121" / "densenet121_final_best.pth",
        prototype_checkpoint=ROOT / "checkpoints" / "prototype"
        / "prototype_network_full_training_final.pt",
        output=ROOT / "runtime" / "calibration" / "calibration_report.json",
    )
    for name, metrics in report["models"].items():
        print(name, metrics["test"]["macro_ece"], metrics["test"]["macro_brier"])
