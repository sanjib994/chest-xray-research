
from pathlib import Path
from typing import Dict, Optional


PRODUCT_ROOT = Path(__file__).resolve().parents[2]

VISUAL_ROOT = (
    PRODUCT_ROOT
    / "runtime"
    / "visual_xai"
)


ARTIFACTS = {
    "gradcam": "gradcam_overlay.png",
    "gradcam_plus_plus": "gradcam_plus_plus_overlay.png",
    "gradcam_attention_box": "gradcam_attention_box.png",
    "gradcam_plus_plus_attention_box":
        "gradcam_plus_plus_attention_box.png",
}


def get_visual_xai_artifacts() -> Dict[str, Optional[str]]:
    result = {}

    for name, filename in ARTIFACTS.items():

        path = VISUAL_ROOT / filename

        if path.exists():
            result[name] = str(path)
        else:
            result[name] = None

    return result


def visual_xai_available() -> bool:
    artifacts = get_visual_xai_artifacts()

    return (
        artifacts["gradcam"] is not None
        and
        artifacts["gradcam_plus_plus"] is not None
    )
