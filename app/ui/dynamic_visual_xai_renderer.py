
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image


def _decode_image(value: str) -> Image.Image:
    raw = base64.b64decode(value)
    return Image.open(
        io.BytesIO(raw)
    ).convert("RGB")


def render_prototype_explanation(api_url: str, image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Render prototype IDs, similarities, and durable evidence images."""
    response = requests.post(
        api_url.rstrip("/") + "/visual-xai/prototype",
        files={"image": (filename, image_bytes, "image/jpeg")},
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    explanation = result.get("prototype_explanation", {})
    if explanation.get("available") is not True:
        raise RuntimeError("Prototype explanation was not available.")

    st.subheader("Intrinsic Prototype Explanation")
    st.caption(
        "These are real prototype similarities from the validated prototype "
        "checkpoint and training-image evidence manifest."
    )
    for item in explanation.get("top_prototypes", []):
        evidence_column, details_column = st.columns([1, 1])
        with evidence_column:
            evidence_path = item.get("prototype_evidence_image")
            if evidence_path and Path(evidence_path).is_file():
                st.image(
                    Image.open(evidence_path).convert("RGB"),
                    caption=f"{item['prototype_id']} evidence",
                    use_container_width=True,
                )
            else:
                st.caption("Evidence image unavailable.")
        with details_column:
            st.markdown(f"**Rank {item['rank']}: {item['prototype_id']}**")
            st.write(f"Prototype class: {item['prototype_class']}")
            st.metric("Similarity", f"{float(item['similarity']):.4f}")
            st.write(f"Distance: {float(item['distance']):.4f}")
            if item.get("source_training_image_id"):
                st.caption(
                    "Source training image: "
                    f"{item['source_training_image_id']}"
                )
            patch_path = item.get("prototype_query_patch")
            if patch_path and Path(patch_path).is_file():
                st.image(
                    Image.open(patch_path).convert("RGB"),
                    caption="Query patch",
                    use_container_width=True,
                )

    evaluation = explanation.get("evaluation", {})
    st.markdown("### Prototype Explanation Evaluation")
    metric_columns = st.columns(3)
    localization = evaluation.get("localization", {})
    with metric_columns[0]:
        st.markdown("**Localization**")
        st.caption(
            localization.get("reason", "Not available for this upload.")
        )
    faithfulness = evaluation.get("faithfulness", {})
    with metric_columns[1]:
        st.markdown("**Faithfulness**")
        if faithfulness:
            st.metric(
                "Normalized score",
                f"{float(faithfulness.get('faithfulness_score', 0.0)):.3f}",
            )
            st.caption(
                f"Mean probability drop: "
                f"{float(faithfulness.get('mean_drop', 0.0)):.3f}"
            )
        else:
            st.caption("Not available.")
    stability = evaluation.get("stability", {})
    with metric_columns[2]:
        st.markdown("**Stability**")
        if stability:
            st.metric(
                "Mean similarity",
                f"{float(stability.get('mean_similarity', 0.0)):.3f}",
            )
            st.caption(
                f"Minimum: "
                f"{float(stability.get('minimum_similarity', 0.0)):.3f}"
            )
        else:
            st.caption("Not available.")

    with st.expander("Prototype technical metadata"):
        st.json(
            {
                "model": result.get("model"),
                "target": explanation.get("target"),
                "metadata": result.get("metadata"),
            }
        )
    return result


def render_paired_explanation_comparison(
    api_url: str,
    image_bytes: bytes,
    filename: str,
    model_name: str = "ConvNeXt-Tiny",
) -> Dict[str, Any]:
    """Render metrics computed for both models on the same upload and target."""
    response = requests.post(
        api_url.rstrip("/") + "/visual-xai/paired",
        files={"image": (filename, image_bytes, "image/jpeg")},
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    st.subheader("Paired Explanation Comparison")
    st.caption("Both methods used the same uploaded image and target class.")
    posthoc = result.get("posthoc_model", {})
    prototype = result.get("prototype", {})
    rows = []
    for name, metrics in ((f"{model_name} + Integrated Gradients", posthoc),
                          ("Prototype similarity", prototype)):
        faithfulness = metrics.get("faithfulness", {})
        stability = metrics.get("stability", {})
        rows.append({
            "Method": name,
            "Faithfulness": float(
                faithfulness.get("faithfulness_score", 0.0)
            ),
            "Stability": float(stability.get("mean_similarity", 0.0)),
            "Localization": "Unavailable (no lesion mask)",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.metric(
        "Attribution agreement",
        f"{float(result.get('attribution_cosine_similarity', 0.0)):.3f}",
    )
    return result


def render_reliability_comparison(
    api_url: str,
    image_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """Render uncertainty and robustness comparison for both models."""
    response = requests.post(
        api_url.rstrip("/") + "/visual-xai/reliability-comparison",
        files={"image": (filename, image_bytes, "image/jpeg")},
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    st.subheader("Reliability Comparison")
    rows = []
    for name, metrics in result.get("models", {}).items():
        uncertainty = metrics.get("uncertainty", {})
        robustness = metrics.get("robustness", {})
        shift = robustness.get("prediction_shift", {})
        rows.append({
            "Model": name,
            "Mean entropy": float(uncertainty.get("mean_entropy", 0.0)),
            "Mean uncertainty": float(
                uncertainty.get("mean_confidence_uncertainty", 0.0)
            ),
            "Prediction agreement": float(
                robustness.get("prediction_agreement", 0.0)
            ),
            "Mean absolute shift": float(
                shift.get("mean_absolute_shift", 0.0)
            ),
            "Reliability score": (
                float(metrics.get("reliability_score", {}).get("score"))
                if metrics.get("reliability_score", {}).get("score") is not None
                else None
            ),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    calibration = result.get("verified_calibration", {})
    if calibration.get("available") is True:
        st.subheader("Previously verified calibration (untouched test split)")
        calibration_rows = []
        report_models = calibration.get("report", {}).get("models", {})
        for name, model_report in report_models.items():
            report = model_report.get("test", {})
            if report:
                calibration_rows.append({
                    "Model": name,
                    "Macro ECE": report.get("macro_ece"),
                    "Macro Brier": report.get("macro_brier"),
                    "Test samples": report.get("n_samples"),
                })
        if calibration_rows:
            st.dataframe(calibration_rows, use_container_width=True, hide_index=True)
        st.caption("These are the existing hash-verified DenseNet/prototype calibration results, not ConvNeXt calibration. Calibration parameters were fit on validation data only; values are bound to checkpoint and split hashes.")
    else:
        st.info("Verified calibration unavailable; run the calibration script to generate a hash-bound report.")
    st.caption(
        "Score formula: 100 × mean of available normalized components. "
        "Missing localization/calibration evidence is excluded, not treated as zero."
    )
    return result


def render_final_report(
    prediction_result: Dict[str, Any],
    prototype_result: Dict[str, Any],
    paired_result: Dict[str, Any],
    reliability_result: Dict[str, Any],
    model_name: str = "ConvNeXt-Tiny",
) -> None:
    """Render the consolidated report without inventing unavailable evidence."""
    st.header("Final Explainable CXR Report")
    st.caption(
        "Research/decision-support output. Metrics are calculated from the "
        "uploaded image; this is not a medical diagnosis."
    )

    predictions = prediction_result.get("predictions", {})
    if predictions:
        st.subheader("Disease Predictions")
        st.dataframe(
            [
                {
                    "Disease": disease,
                    "Probability": float(details.get("probability", 0.0)),
                    "Predicted": bool(details.get("predicted", False)),
                }
                for disease, details in predictions.items()
            ],
            use_container_width=True,
            hide_index=True,
        )

    prototype = prototype_result.get("prototype_explanation", {})
    top_prototypes = prototype.get("top_prototypes", [])
    paired_rows = []
    for label, metrics in (
        (
            f"{model_name} + Integrated Gradients",
            paired_result.get("posthoc_model", {}),
        ),
        ("Prototype similarity", paired_result.get("prototype", {})),
    ):
        faithfulness = metrics.get("faithfulness", {})
        stability = metrics.get("stability", {})
        score = metrics.get("reliability_score", {}).get("score")
        paired_rows.append(
            {
                "Method": label,
                "Faithfulness": float(
                    faithfulness.get("faithfulness_score", 0.0)
                ),
                "Stability": float(stability.get("mean_similarity", 0.0)),
                "Reliability score": float(score) if score is not None else None,
            }
        )
    if paired_rows:
        st.subheader("Explanation Method Comparison")
        st.dataframe(paired_rows, use_container_width=True, hide_index=True)

    if top_prototypes:
        best = top_prototypes[0]
        st.subheader("Leading Prototype Reasoning")
        st.write(
            f"**{best['prototype_id']}** — {best['prototype_class']} | "
            f"similarity {float(best['similarity']):.4f} | "
            f"distance {float(best['distance']):.4f}"
        )

    model_rows = []
    for name, metrics in reliability_result.get("models", {}).items():
        score = metrics.get("reliability_score", {}).get("score")
        model_rows.append(
            {
                "Model": name,
                "Reliability score": float(score) if score is not None else None,
                "Prediction agreement": float(
                    metrics.get("robustness", {}).get(
                        "prediction_agreement", 0.0
                    )
                ),
                "Mean uncertainty": float(
                    metrics.get("uncertainty", {}).get(
                        "mean_confidence_uncertainty", 0.0
                    )
                ),
                "Macro ECE": (
                    metrics.get("calibration", {}).get("macro_ece")
                    if metrics.get("calibration")
                    else None
                ),
                "Macro Brier": (
                    metrics.get("calibration", {}).get("macro_brier")
                    if metrics.get("calibration")
                    else None
                ),
            }
        )
    if model_rows:
        st.subheader("Model Reliability")
        st.dataframe(model_rows, use_container_width=True, hide_index=True)

    st.subheader("Limitations")
    st.markdown(
        "- Localization IoU is unavailable without a linked lesion annotation mask.\n"
        "- Similarity identifies the closest learned prototype; it does not prove clinical plausibility.\n"
        "- Calibration is not reported unless calibration provenance is verified.\n"
        "- This research/decision-support output requires qualified clinical review."
    )


def render_request_visual_xai(
    api_url: str,
    image_bytes: bytes,
    filename: str,
    target_class: Optional[int] = None,
) -> Dict[str, Any]:

    files = {
        "image": (
            filename,
            image_bytes,
            "image/jpeg",
        )
    }

    data = {}

    if target_class is not None:
        data["target_class"] = str(
            int(target_class)
        )

    response = requests.post(
        api_url.rstrip("/")
        + "/visual-xai/generate",
        files=files,
        data=data,
        timeout=180,
    )

    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        raise RuntimeError(
            "Dynamic Visual XAI API failed: "
            f"HTTP {response.status_code}: {detail}"
        )

    result = response.json()

    metadata = result.get(
        "metadata",
        {},
    )

    if metadata.get("request_specific") is not True:
        raise RuntimeError(
            "API did not identify the result as request-specific."
        )

    if metadata.get("static_artifact_used") is True:
        raise RuntimeError(
            "API reported use of a static XAI artifact."
        )

    visual = result.get(
        "visual_xai",
        {},
    )

    required = [
        "gradcam",
        "gradcam_plus_plus",
        "integrated_gradients",
        "gradcam_attention_box",
        "gradcam_plus_plus_attention_box",
    ]

    missing = [
        name
        for name in required
        if not visual.get(name)
    ]

    if missing:
        raise RuntimeError(
            "Dynamic Visual XAI response is missing: "
            + ", ".join(missing)
        )

    st.subheader(
        "Visual XAI — This Uploaded X-ray"
    )

    st.caption(
        "These visualizations were generated from "
        "the X-ray uploaded in this request."
    )

    target = result.get(
        "target",
        {},
    )

    if target:
        st.write(
            "Explanation target:",
            target.get("disease"),
            f"({float(target.get('probability', 0.0)):.4f})",
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Grad-CAM")
        st.image(
            _decode_image(
                visual["gradcam"]
            ),
            caption=(
                "Grad-CAM — generated from current upload"
            ),
            use_container_width=True,
        )

    with col2:
        st.markdown("### Grad-CAM++")
        st.image(
            _decode_image(
                visual["gradcam_plus_plus"]
            ),
            caption=(
                "Grad-CAM++ — generated from current upload"
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Integrated Gradients")
        st.image(
            _decode_image(
                visual["integrated_gradients"]
            ),
            caption=(
                "Integrated Gradients — generated from current upload"
            ),
            use_container_width=True,
        )

    with col4:
        st.markdown(
            "### Model Attention Region"
        )
        st.image(
            _decode_image(
                visual["gradcam_attention_box"]
            ),
            caption=(
                "Model attention region — Grad-CAM"
            ),
            use_container_width=True,
        )

    col5, col6 = st.columns(2)

    with col5:
        st.markdown(
            "### Model Attention Region — Grad-CAM++"
        )
        st.image(
            _decode_image(
                visual["gradcam_plus_plus_attention_box"]
            ),
            caption=(
                "Model attention region — Grad-CAM++"
            ),
            use_container_width=True,
        )

    with col6:
        st.markdown("### Explanation methods")
        st.caption(
            "Grad-CAM and Grad-CAM++ use convolutional feature maps. "
            "Integrated Gradients attributes the prediction along a "
            "baseline-to-input path."
        )

    evaluation = metadata.get("evaluation", {})
    st.subheader("Explanation Evaluation")
    metric_columns = st.columns(3)

    localization = evaluation.get("localization", {})
    with metric_columns[0]:
        st.markdown("**Localization**")
        if localization.get("available"):
            st.metric(
                "IoU",
                f"{float(localization.get('iou', 0.0)):.3f}",
            )
        else:
            st.caption(
                localization.get(
                    "reason",
                    "Ground-truth lesion mask required.",
                )
            )

    faithfulness = evaluation.get("faithfulness", {})
    with metric_columns[1]:
        st.markdown("**Faithfulness**")
        if faithfulness:
            st.metric(
                "Mean probability drop",
                f"{float(faithfulness.get('mean_drop', 0.0)):.3f}",
            )
            st.caption(
                f"Deletion AUC: "
                f"{float(faithfulness.get('deletion_auc', 0.0)):.3f}"
            )
            st.caption(
                f"Normalized faithfulness: "
                f"{float(faithfulness.get('faithfulness_score', 0.0)):.3f}"
            )
        else:
            st.caption("Not available for this response.")

    stability = evaluation.get("stability", {})
    with metric_columns[2]:
        st.markdown("**Stability**")
        if stability:
            st.metric(
                "Mean attribution similarity",
                f"{float(stability.get('mean_similarity', 0.0)):.3f}",
            )
            st.caption(
                f"Minimum: "
                f"{float(stability.get('minimum_similarity', 0.0)):.3f}"
            )
        else:
            st.caption("Not available for this response.")

    st.warning(
        "The displayed attention regions represent "
        "model attention and are NOT medically validated "
        "lesion boundaries."
    )

    with st.expander(
        "Visual XAI technical metadata"
    ):
        st.json(
            {
                "target": target,
                "metadata": metadata,
            }
        )

    return result
