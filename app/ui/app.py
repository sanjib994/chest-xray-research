from pathlib import Path
import tempfile
import sys
import os

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Use absolute imports from project root
from app.ui.api_client import ProductionAPIClient
from app.ui.components import (
    render_prediction_table,
    render_xai,
    render_reliability,
    render_metadata,
    display_visual_xai,
    display_request_visual_xai,
    render_prototype_explanation,
    render_paired_explanation_comparison,
    render_reliability_comparison,
    render_final_report,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = (
    "http://127.0.0.1:8765"
)

MAX_FILE_SIZE = (
    10 * 1024 * 1024
)


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def _display_step_10_13b_visual_xai(api_response, uploaded_file):
    """
    Display DYNAMIC, request-specific visual XAI.

    This generates fresh Grad-CAM heatmaps for EACH uploaded image,
    not static/cached visualizations.

    Parameters:
    - api_response: Result from the API predict endpoint
    - uploaded_file: The actual file object uploaded by the user
    """
    try:
        uploaded_bytes = (
            uploaded_file.getvalue()
        )

        st.header(
            "Visual Explainability"
        )

        st.info(
            "Generating dynamic Grad-CAM visualizations for your uploaded X-ray..."
        )

        # Generate and display DYNAMIC heatmaps for this specific image
        visual_result = display_request_visual_xai(
            api_url=API_URL,
            image_bytes=uploaded_bytes,
            filename=uploaded_file.name,
        )

        model_name = api_response.get("model", {}).get("name", "ConvNeXt-Tiny")
        model_name = {
            "CONVNEXT-TINY": "ConvNeXt-Tiny",
            "DENSENET121": "DenseNet121",
        }.get(str(model_name).upper(), model_name)

        st.info(
            "Generating intrinsic prototype similarity reasoning..."
        )
        prototype_result = render_prototype_explanation(
            api_url=API_URL,
            image_bytes=uploaded_bytes,
            filename=uploaded_file.name,
        )
        paired_result = render_paired_explanation_comparison(
            api_url=API_URL,
            image_bytes=uploaded_bytes,
            filename=uploaded_file.name,
            model_name=model_name,
        )
        reliability_result = render_reliability_comparison(
            api_url=API_URL,
            image_bytes=uploaded_bytes,
            filename=uploaded_file.name,
        )
        render_final_report(
            prediction_result=api_response,
            prototype_result=prototype_result,
            paired_result=paired_result,
            reliability_result=reliability_result,
            model_name=model_name,
        )

    except Exception as exc:
        st.error(
            "Visual XAI generation failed"
        )
        st.exception(exc)
        st.write(
            "**Error details:** Make sure the FastAPI backend is running on port 8765"
        )


# =============================================================================
# PAGE
# =============================================================================

st.set_page_config(
    page_title="VinBigData XAI",
    page_icon="🩻",
    layout="wide",
)


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "VinBigData XAI"
)

st.caption(
    "Chest X-ray explainable AI "
    "decision-support prototype"
)

st.warning(
    "Research/decision-support prototype. "
    "This application is not a standalone "
    "medical diagnosis."
)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header(
    "Production API"
)

st.sidebar.code(
    API_URL
)

st.sidebar.caption(
    "The UI communicates with the "
    "validated FastAPI production service."
)


# =============================================================================
# API HEALTH
# =============================================================================

client = ProductionAPIClient(
    base_url=API_URL,
    timeout=300,
)

try:

    health = client.health()

    if health.get(
        "status"
    ) == "ok":

        st.sidebar.success(
            "API: online"
        )

    else:

        st.sidebar.warning(
            "API: unexpected response"
        )

except Exception as exc:

    st.sidebar.error(
        "API unavailable"
    )

    st.info(
        "Start the production API before "
        "submitting an X-ray."
    )

    st.stop()


# =============================================================================
# MODEL INFORMATION
# =============================================================================

try:

    info = client.info()

    st.sidebar.write(
        f"Model: {info.get('model', 'Unknown')}"
    )

    st.sidebar.write(
        "Official-test inference: "
        + (
            "Enabled"
            if info.get(
                "official_test_inference_enabled"
            )
            else "Disabled"
        )
    )

except Exception:

    info = {}


# =============================================================================
# UPLOAD
# =============================================================================

st.header(
    "Upload Chest X-ray"
)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    help=(
        "Upload one chest X-ray for "
        "production inference."
    ),
)


# =============================================================================
# PROCESS
# =============================================================================

if uploaded_file is not None:

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "File exceeds the 10 MB limit."
        )

        st.stop()

    st.subheader(
        "Uploaded image"
    )

    st.image(
        uploaded_file,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    if st.button(
        "Analyze X-ray",
        type="primary",
        use_container_width=True,
    ):

        suffix = (
            Path(
                uploaded_file.name
            )
            .suffix
            .lower()
        )

        if suffix not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:

            st.error(
                "Unsupported image format."
            )

            st.stop()

        temporary_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as handle:

                handle.write(
                    uploaded_file.getbuffer()
                )

                temporary_path = Path(
                    handle.name
                )

            with st.spinner(
                "Running production inference..."
            ):

                result = client.predict(
                    temporary_path
                )

            st.success(
                "Production inference completed."
            )

            # -------------------------------------------------------------
            # SAFETY
            # -------------------------------------------------------------

            metadata = result.get(
                "metadata",
                {}
            )

            if metadata.get(
                "synthetic"
            ) is True:

                st.error(
                    "Synthetic response detected. "
                    "Result rejected."
                )

                st.stop()

            # -------------------------------------------------------------
            # PREDICTIONS
            # -------------------------------------------------------------

            st.header(
                "Prediction"
            )

            predictions = result.get(
                "predictions",
                {}
            )

            render_prediction_table(
                predictions
            )

            # ===== DYNAMIC VISUAL XAI GENERATION =====
            # Generate fresh Grad-CAM heatmaps for THIS uploaded image

            _display_step_10_13b_visual_xai(result, uploaded_file)

            # ===== END VISUAL XAI =====

            # ----- Optional: Technical Explanation (JSON) -----

            with st.expander(
                "Technical explanation details"
            ):
                render_xai(
                    result.get(
                        "explanation"
                    )
                )

            # -------------------------------------------------------------
            # RELIABILITY
            # -------------------------------------------------------------

            st.header(
                "Reliability"
            )

            render_reliability(
                result.get(
                    "reliability"
                )
            )

            # -------------------------------------------------------------
            # METADATA
            # -------------------------------------------------------------

            with st.expander(
                "Technical inference metadata"
            ):

                render_metadata(
                    metadata
                )

        except Exception as exc:

            st.error(
                "Production API request failed."
            )

            st.exception(
                exc
            )

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                try:

                    temporary_path.unlink()

                except Exception:

                    pass

else:

    st.info(
        "Upload a chest X-ray to begin."
    )
