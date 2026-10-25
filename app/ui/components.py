import os
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use absolute imports from project root
from app.ui.visual_xai_renderer import display_visual_xai


DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
]


def render_prediction_table(
    predictions,
):

    rows = []

    for disease in DISEASES:

        entry = predictions.get(
            disease,
            {}
        )

        if isinstance(
            entry,
            dict
        ):

            probability = float(
                entry.get(
                    "probability",
                    0.0
                )
            )

            threshold = float(
                entry.get(
                    "threshold",
                    0.5
                )
            )

            predicted = bool(
                entry.get(
                    "predicted",
                    probability >= threshold
                )
            )

        else:

            probability = float(
                entry
            )

            threshold = 0.5

            predicted = (
                probability
                >= threshold
            )

        rows.append(
            {
                "Disease":
                    disease,

                "Probability":
                    f"{probability * 100:.2f}%",

                "Threshold":
                    f"{threshold:.2f}",

                "Status":
                    (
                        "Positive"
                        if predicted
                        else "Negative"
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_reliability(
    reliability,
):

    if not reliability:

        st.info(
            "Reliability information unavailable."
        )

        return

    details = reliability.get(
        "details",
        {}
    )

    if not details:

        st.info(
            "Reliability information unavailable."
        )

        return

    st.subheader(
        "Uncertainty"
    )

    uncertainty = details.get(
        "uncertainty_summary"
    )

    if uncertainty:

        if isinstance(
            uncertainty,
            dict
        ):

            rows = [

                {
                    "Metric":
                        key,

                    "Value":
                        value,
                }

                for key, value
                in uncertainty.items()
            ]

            st.dataframe(
                pd.DataFrame(
                    rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.json(
                uncertainty
            )

    robustness = details.get(
        "robustness"
    )

    st.subheader(
        "Robustness"
    )

    if robustness:

        agreement = (
            robustness.get(
                "prediction_agreement"
            )
        )

        if agreement is not None:

            st.metric(
                "Prediction agreement",
                f"{float(agreement) * 100:.2f}%"
            )

        shift = robustness.get(
            "prediction_shift"
        )

        if shift:

            st.json(
                shift
            )

    else:

        st.info(
            "Robustness information unavailable."
        )


def _render_xai_json_original(
    explanation,
):

    st.subheader(
        "Explainability"
    )

    if not explanation:

        st.info(
            "Explanation unavailable."
        )

        return

    available = explanation.get(
        "available",
        False
    )

    if not available:

        st.warning(
            "XAI explanation unavailable."
        )

        return

    st.success(
        "XAI explanation available."
    )

    details = explanation.get(
        "details",
        {}
    )

    gradcam = details.get(
        "gradcam"
    )

    gradcampp = details.get(
        "gradcampp"
    )

    st.write(
        "Grad-CAM"
    )

    if isinstance(
        gradcam,
        dict
    ) and "error" in gradcam:

        st.warning(
            gradcam["error"]
        )

    elif gradcam is not None:

        st.json(
            gradcam
        )

    else:

        st.info(
            "Grad-CAM output unavailable."
        )

    st.write(
        "Grad-CAM++"
    )

    if isinstance(
        gradcampp,
        dict
    ) and "error" in gradcampp:

        st.warning(
            gradcampp["error"]
        )

    elif gradcampp is not None:

        st.json(
            gradcampp
        )

    else:

        st.info(
            "Grad-CAM++ output unavailable."
        )




def render_metadata(
    metadata,
):

    st.subheader(
        "Inference metadata"
    )

    st.json(
        metadata
        if metadata
        else {}
    )


# =============================================================================
# STEP 10.13B — VISUAL XAI DISPLAY
# =============================================================================

def display_visual_xai(
    visual_xai,
    api_base_url="http://127.0.0.1:8765",
):
    import streamlit as st

    st.subheader(
        "Visual XAI"
    )

    if not visual_xai:
        st.info(
            "Visual XAI is not available."
        )
        return

    if not visual_xai.get(
        "available",
        False
    ):
        st.warning(
            "Visual XAI artifacts are unavailable."
        )
        return

    gradcam = visual_xai.get(
        "gradcam"
    )

    gradcampp = visual_xai.get(
        "gradcam_plus_plus"
    )

    box = visual_xai.get(
        "gradcam_attention_box"
    )

    box_pp = visual_xai.get(
        "gradcam_plus_plus_attention_box"
    )

    columns = st.columns(2)

    if gradcam:
        with columns[0]:
            st.markdown(
                "### Grad-CAM"
            )
            st.image(
                gradcam,
                use_container_width=True,
            )

    if gradcampp:
        with columns[1]:
            st.markdown(
                "### Grad-CAM++"
            )
            st.image(
                gradcampp,
                use_container_width=True,
            )

    st.markdown(
        "### Attention Regions"
    )

    box_columns = st.columns(2)

    if box:
        with box_columns[0]:
            st.image(
                box,
                caption="Grad-CAM — Model Attention Region",
                use_container_width=True,
            )

    if box_pp:
        with box_columns[1]:
            st.image(
                box_pp,
                caption="Grad-CAM++ — Model Attention Region",
                use_container_width=True,
            )

    st.caption(
        "These regions represent model attention and "
        "are not confirmed medical lesion boundaries."
    )


# =============================================================================
# STEP 10.13B-2R VISUAL XAI WRAPPER
# =============================================================================

def render_xai(*args, **kwargs):
    """
    Backward-compatible XAI renderer.

    1. Runs the original JSON/text XAI renderer.
    2. Adds the real visual Grad-CAM / Grad-CAM++ artifacts.

    Existing callers do not need to change.
    """

    original_result = _render_xai_json_original(
        *args,
        **kwargs,
    )

    api_url = os.environ.get(
        "VINBIGDATA_XAI_API_URL",
        "http://127.0.0.1:8765",
    )

    try:

        visual_result = display_visual_xai(
            api_url=api_url,
        )

    except Exception as exc:

        st.error(
            "Visual XAI rendering error: "
            + str(exc)
        )

        visual_result = {
            "rendered": False,
            "error": str(exc),
        }

    return {
        "original": original_result,
        "visual": visual_result,
    }


# =============================================================================
# STEP 10.13B-2R3
# PERMANENT VISUAL XAI COMPATIBILITY WRAPPER
# =============================================================================

from app.ui.visual_xai_renderer import (
    display_visual_xai as _production_visual_xai_renderer,
)


def display_visual_xai(
    api_url=None,
    *args,
    **kwargs,
):
    """
    Compatibility wrapper for the production Streamlit Visual XAI renderer.

    Supports both:

        display_visual_xai()

    and:

        display_visual_xai(api_url=...)

    The wrapper delegates all actual image rendering to
    visual_xai_renderer.py.
    """

    if api_url is None:

        api_url = kwargs.get(
            "api_url"
        )

    return _production_visual_xai_renderer(
        api_url=api_url,
        *args,
        **kwargs,
    )

# =============================================================================
# END STEP 10.13B-2R3
# =============================================================================



# ==============================================================================
# STEP 10.13C — REQUEST-SPECIFIC VISUAL XAI
# ==============================================================================

def display_request_visual_xai(
    api_url,
    image_bytes,
    filename="uploaded_xray.png",
    target_class=None,
):
    from app.ui.dynamic_visual_xai_renderer import (
        render_request_visual_xai,
    )

    return render_request_visual_xai(
        api_url=api_url,
        image_bytes=image_bytes,
        filename=filename,
        target_class=target_class,
    )


def render_prototype_explanation(
    api_url,
    image_bytes,
    filename="uploaded_xray.png",
):
    from app.ui.dynamic_visual_xai_renderer import (
        render_prototype_explanation as render_dynamic_prototype_explanation,
    )

    return render_dynamic_prototype_explanation(
        api_url=api_url,
        image_bytes=image_bytes,
        filename=filename,
    )


def render_paired_explanation_comparison(
    api_url,
    image_bytes,
    filename="uploaded_xray.png",
    model_name="ConvNeXt-Tiny",
):
    from app.ui.dynamic_visual_xai_renderer import (
        render_paired_explanation_comparison as render_dynamic_paired_comparison,
    )

    return render_dynamic_paired_comparison(
        api_url=api_url,
        image_bytes=image_bytes,
        filename=filename,
        model_name=model_name,
    )


def render_reliability_comparison(
    api_url,
    image_bytes,
    filename="uploaded_xray.png",
):
    from app.ui.dynamic_visual_xai_renderer import (
        render_reliability_comparison as render_dynamic_reliability_comparison,
    )

    return render_dynamic_reliability_comparison(
        api_url=api_url,
        image_bytes=image_bytes,
        filename=filename,
    )


def render_final_report(
    prediction_result,
    prototype_result,
    paired_result,
    reliability_result,
    model_name="ConvNeXt-Tiny",
):
    from app.ui.dynamic_visual_xai_renderer import (
        render_final_report as render_dynamic_final_report,
    )

    return render_dynamic_final_report(
        prediction_result=prediction_result,
        prototype_result=prototype_result,
        paired_result=paired_result,
        reliability_result=reliability_result,
        model_name=model_name,
    )
