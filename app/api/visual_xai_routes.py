import fastapi
import os
import re
import tempfile


from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi import HTTPException
import torch
import torch.nn.functional as F

from app.settings import PROJECT_ROOT, configured_path


router = APIRouter(
    prefix="/visual-xai",
    tags=["visual-xai"]
)


DISEASES = (
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Lung Opacity", "Nodule/Mass", "Pleural Effusion",
    "Pneumothorax",
)
PROTOTYPE_CHECKPOINT = configured_path(
    "VINBIGDATA_PROTOTYPE_CHECKPOINT",
    "checkpoints/prototype/prototype_network_full_training_final.pt",
)
PROTOTYPE_MANIFEST = configured_path(
    "VINBIGDATA_PROTOTYPE_MANIFEST",
    "configs/prototype/P4_v2_prototype_evidence_manifest.csv",
)
PROTOTYPE_IMAGE_ROOT = configured_path(
    "VINBIGDATA_PROTOTYPE_IMAGE_DIR",
    "assets/prototype_evidence",
)
CALIBRATION_REPORT = configured_path(
    "VINBIGDATA_CALIBRATION_REPORT",
    "runtime/calibration/calibration_report.json",
)

VISUAL_ROOT = (
    PROJECT_ROOT / "runtime" / "visual_xai"
)


ARTIFACTS = {
    "gradcam": "gradcam_overlay.png",
    "gradcam-plus-plus":
        "gradcam_plus_plus_overlay.png",
    "gradcam-attention-box":
        "gradcam_attention_box.png",
    "gradcam-plus-plus-attention-box":
        "gradcam_plus_plus_attention_box.png",
}


@router.get("/status")
def visual_xai_status():

    result = {}

    for name, filename in ARTIFACTS.items():

        path = VISUAL_ROOT / filename

        result[name] = {
            "available": path.exists(),
            "path": str(path) if path.exists() else None,
        }

    return {
        "visual_xai": result,
        "medical_lesion_boundary": False,
        "attention_label":
            "MODEL ATTENTION REGION",
    }


@router.get("/prototype-assets/{filename}")
def get_prototype_asset(filename: str):
    """Serve only explicitly named prototype reference images."""
    if not re.fullmatch(r"prototype_\d{3}_(full_evidence|query_patch)\.png", filename):
        raise HTTPException(status_code=404, detail="Prototype evidence image not found.")

    path = PROTOTYPE_IMAGE_ROOT / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Prototype evidence image not found.")
    return FileResponse(path, media_type="image/png")


@router.post("/prototype")
async def generate_prototype_explanation(
    image: fastapi.UploadFile = fastapi.File(...),
):
    """Run the validated prototype checkpoint on the uploaded X-ray."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    from src.data.preprocessing import preprocess_image
    from src.models.prototype_network import load_prototype_checkpoint
    from src.xai.prototype_explanation import (
        load_prototype_evidence_index,
        rank_prototypes,
    )
    from src.evaluation.explanation_metrics import (
        explanation_stability,
        faithfulness_deletion,
    )

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(image_bytes)
            temporary_path = Path(handle.name)
        tensor = preprocess_image(str(temporary_path)).unsqueeze(0)
        model = load_prototype_checkpoint(PROTOTYPE_CHECKPOINT)
        with torch.no_grad():
            output = model(tensor)
        probabilities = torch.sigmoid(output["logits"])[0].cpu().tolist()
        evidence = load_prototype_evidence_index(PROTOTYPE_MANIFEST)
        top_prototypes = rank_prototypes(output, evidence)
        for item in top_prototypes:
            index = item["prototype_index"]
            evidence_filename = f"prototype_{index:03d}_full_evidence.png"
            patch_filename = f"prototype_{index:03d}_query_patch.png"
            item["prototype_evidence_image"] = (
                f"/visual-xai/prototype-assets/{evidence_filename}"
                if (PROTOTYPE_IMAGE_ROOT / evidence_filename).is_file()
                else None
            )
            item["prototype_query_patch"] = (
                f"/visual-xai/prototype-assets/{patch_filename}"
                if (PROTOTYPE_IMAGE_ROOT / patch_filename).is_file()
                else None
            )
        selected_index = top_prototypes[0]["prototype_index"]
        selected_class = int(torch.argmax(output["logits"][0]).item())

        def prototype_attribution(input_tensor: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                current = model(input_tensor)
                distance_map = current["distances"][:, selected_index]
                attribution = 1.0 / (1.0 + distance_map)
                return F.interpolate(
                    attribution.unsqueeze(1),
                    size=tuple(input_tensor.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu()

        attribution = prototype_attribution(tensor)
        faithfulness = faithfulness_deletion(
            model=model,
            input_tensor=tensor,
            attribution=attribution,
            target_class=selected_class,
        )
        stability = explanation_stability(
            input_tensor=tensor,
            attribution_fn=prototype_attribution,
            repeats=5,
        )
        return {
            "schema_version": "1.0",
            "model": {
                "name": "GenuinePrototypeNetwork",
                "backbone": "DenseNet121",
                "checkpoint": str(PROTOTYPE_CHECKPOINT),
            },
            "predictions": {
                disease: {
                    "probability": float(probability),
                    "threshold": 0.5,
                    "predicted": float(probability) >= 0.5,
                }
                for disease, probability in zip(
                    ("Atelectasis", "Cardiomegaly", "Consolidation",
                     "Lung Opacity", "Nodule/Mass", "Pleural Effusion",
                     "Pneumothorax"),
                    probabilities,
                )
            },
            "prototype_explanation": {
                "available": True,
                "top_prototypes": top_prototypes,
                "target": {
                    "disease": (
                        "Atelectasis", "Cardiomegaly", "Consolidation",
                        "Lung Opacity", "Nodule/Mass", "Pleural Effusion",
                        "Pneumothorax",
                    )[selected_class],
                    "class_index": selected_class,
                    "probability": float(probabilities[selected_class]),
                },
                "evaluation": {
                    "localization": {
                        "available": False,
                        "reason": (
                            "Ground-truth lesion mask was not supplied; "
                            "prototype attribution is available but IoU is not computed."
                        ),
                    },
                    "faithfulness": faithfulness,
                    "stability": stability,
                },
            },
            "metadata": {
                "synthetic": False,
                "training": False,
                "official_test": False,
                "prototype_projection": False,
            },
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@router.post("/paired")
async def generate_paired_explanations(
    image: fastapi.UploadFile = fastapi.File(...),
):
    """Compare ConvNeXt post-hoc and intrinsic prototype explanations."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    from src.data.preprocessing import preprocess_image
    from src.models.prototype_network import load_prototype_checkpoint
    from src.xai.production.request_visual_xai import (
        _capture_integrated_gradients,
        _load_model,
    )
    from src.evaluation.comparison import paired_explanation_summary, reliability_score
    from src.evaluation.explanation_metrics import (
        explanation_stability,
        faithfulness_deletion,
    )

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(image_bytes)
            temporary_path = Path(handle.name)
        tensor = preprocess_image(str(temporary_path)).unsqueeze(0)
        posthoc_model = _load_model()
        prototype_model = load_prototype_checkpoint(PROTOTYPE_CHECKPOINT)

        with torch.no_grad():
            posthoc_logits = posthoc_model(tensor)
        target_class = int(torch.argmax(posthoc_logits[0]).item())
        posthoc_attribution = _capture_integrated_gradients(
            posthoc_model, tensor, target_class
        )[0]

        with torch.no_grad():
            prototype_output = prototype_model(tensor)
        prototype_index = int(
            torch.argmax(prototype_output["similarities"][0]).item()
        )

        def prototype_attribution(input_tensor):
            with torch.no_grad():
                output = prototype_model(input_tensor)
                distances = output["distances"][:, prototype_index]
                attribution = 1.0 / (1.0 + distances)
                return F.interpolate(
                    attribution.unsqueeze(1),
                    size=tuple(input_tensor.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu()

        posthoc_metrics = {
            "faithfulness": faithfulness_deletion(
                posthoc_model, tensor, posthoc_attribution, target_class
            ),
            "stability": explanation_stability(
                tensor,
                lambda value: _capture_integrated_gradients(
                    posthoc_model, value, target_class
                )[0],
                repeats=3,
            ),
            "localization": {
                "available": False,
                "reason": "Ground-truth lesion mask was not supplied.",
            },
        }
        prototype_attr = prototype_attribution(tensor)
        prototype_metrics = {
            "faithfulness": faithfulness_deletion(
                prototype_model, tensor, prototype_attr, target_class
            ),
            "stability": explanation_stability(
                tensor, prototype_attribution, repeats=3
            ),
            "localization": {
                "available": False,
                "reason": "Ground-truth lesion mask was not supplied.",
            },
        }
        posthoc_metrics["reliability_score"] = reliability_score(
            faithfulness=posthoc_metrics["faithfulness"]["faithfulness_score"],
            stability=posthoc_metrics["stability"]["mean_similarity"],
        )
        prototype_metrics["reliability_score"] = reliability_score(
            faithfulness=prototype_metrics["faithfulness"]["faithfulness_score"],
            stability=prototype_metrics["stability"]["mean_similarity"],
        )
        return paired_explanation_summary(
            posthoc_metrics,
            prototype_metrics,
            posthoc_attribution,
            prototype_attr,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@router.post("/reliability-comparison")
async def generate_reliability_comparison(
    image: fastapi.UploadFile = fastapi.File(...),
):
    """Compare uncertainty and deterministic noise robustness for both models."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    from src.data.preprocessing import preprocess_image
    from src.models.prototype_network import load_prototype_checkpoint
    from src.models.model_loader import resolve_model_name
    from src.xai.production.request_visual_xai import _load_model
    from src.reliability.robustness import prediction_agreement, prediction_shift
    from src.reliability.uncertainty import uncertainty_summary
    from src.evaluation.comparison import reliability_score

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(image_bytes)
            temporary_path = Path(handle.name)
        tensor = preprocess_image(str(temporary_path)).unsqueeze(0)
        active_model_name = resolve_model_name(
            os.environ.get("VINBIGDATA_MODEL", "convnext_tiny")
        )
        models_by_name = {
            active_model_name: _load_model(),
            "prototype": load_prototype_checkpoint(PROTOTYPE_CHECKPOINT),
        }
        result = {
            "protocol": "same_image_deterministic_noise_sigma_0.02_seed_42",
            "models": {},
        }
        from src.reliability.calibrated_evaluation import load_verified_report
        verified_report = load_verified_report(CALIBRATION_REPORT)
        result["verified_calibration"] = (
            {"available": True, "report": verified_report}
            if verified_report is not None
            else {
                "available": False,
                "reason": "No provenance-verified calibration artifact is available.",
            }
        )
        generator = torch.Generator(device=tensor.device).manual_seed(42)
        perturbed = tensor + torch.randn(
            tensor.shape, generator=generator, device=tensor.device
        ) * 0.02
        for name, model in models_by_name.items():
            with torch.no_grad():
                original_output = model(tensor)
                perturbed_output = model(perturbed)
                if isinstance(original_output, dict):
                    original_output = original_output["logits"]
                    perturbed_output = perturbed_output["logits"]
                original_probabilities = torch.sigmoid(original_output[0]).cpu().numpy()
                perturbed_probabilities = torch.sigmoid(perturbed_output[0]).cpu().numpy()
            result["models"][name] = {
                "uncertainty": uncertainty_summary(original_probabilities),
                "robustness": {
                    "prediction_shift": prediction_shift(
                        original_probabilities, perturbed_probabilities
                    ),
                    "prediction_agreement": prediction_agreement(
                        original_probabilities, perturbed_probabilities
                    ),
                    "perturbation": {"sigma": 0.02, "seed": 42},
                },
            }
            if verified_report is not None and name in verified_report["models"]:
                calibration = verified_report["models"][name]["test"]
                result["models"][name]["calibration"] = calibration
            else:
                calibration = None
            result["models"][name]["reliability_score"] = reliability_score(
                prediction_agreement=result["models"][name]["robustness"][
                    "prediction_agreement"
                ],
                mean_confidence_uncertainty=result["models"][name][
                    "uncertainty"
                ]["mean_confidence_uncertainty"],
                calibration=(
                    1.0 - float(calibration["macro_ece"])
                    if calibration is not None
                    else None
                ),
            )
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@router.get("/calibration")
def get_verified_calibration():
    """Return calibration only when all recorded input hashes still match."""
    from src.reliability.calibrated_evaluation import load_verified_report

    report = load_verified_report(CALIBRATION_REPORT)
    if report is None:
        return {
            "available": False,
            "reason": "No provenance-verified calibration artifact is available.",
        }
    return {"available": True, "report": report}


def _file_response(name):

    filename = ARTIFACTS.get(name)

    if filename is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown visual XAI artifact."
        )

    path = VISUAL_ROOT / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Visual XAI artifact unavailable."
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=filename,
    )


@router.get("/gradcam")
def get_gradcam():

    return _file_response(
        "gradcam"
    )


@router.get("/gradcam-plus-plus")
def get_gradcam_plus_plus():

    return _file_response(
        "gradcam-plus-plus"
    )


@router.get("/gradcam-attention-box")
def get_gradcam_attention_box():

    return _file_response(
        "gradcam-attention-box"
    )


@router.get("/gradcam-plus-plus-attention-box")
def get_gradcam_plus_plus_attention_box():

    return _file_response(
        "gradcam-plus-plus-attention-box"
    )


# ==============================================================================
# STEP 10.13C — REQUEST-SPECIFIC VISUAL XAI
# ==============================================================================

@router.post("/generate")
async def generate_request_specific_visual_xai(
    image: fastapi.UploadFile = fastapi.File(...),
    target_class: int | None = fastapi.Form(default=None),
):
    """
    Generate Visual XAI for the image in THIS request.

    This endpoint intentionally does not retrieve the static runtime PNGs.
    """

    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise fastapi.HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        from src.xai.production.request_visual_xai import (
            generate_request_visual_xai,
        )

        result = generate_request_visual_xai(
            image_bytes=image_bytes,
            filename=image.filename or "uploaded_xray.png",
            target_class=target_class,
        )

        return fastapi.responses.JSONResponse(
            content=result
        )

    except fastapi.HTTPException:
        raise

    except Exception as exc:
        raise fastapi.HTTPException(
            status_code=500,
            detail=(
                "Request-specific Visual XAI generation failed: "
                + str(exc)
            ),
        )
