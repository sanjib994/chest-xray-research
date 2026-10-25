"""
Step 10.8 — API smoke tests.

These tests are intended for synthetic/mock inference only.
They do not load the research checkpoint and do not access
the official test population.
"""

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import set_inference_handler


def synthetic_handler(
    image_bytes,
    filename,
):

    return {
        "schema_version": "1.0",
        "image_id": "SYNTHETIC_API_TEST",
        "model": {
            "name": "DenseNet121",
            "type": "multilabel_classifier",
        },
        "predictions": {
            "Atelectasis": {
                "probability": 0.10,
                "threshold": 0.50,
                "predicted": False,
            },
            "Cardiomegaly": {
                "probability": 0.20,
                "threshold": 0.50,
                "predicted": False,
            },
            "Consolidation": {
                "probability": 0.70,
                "threshold": 0.50,
                "predicted": True,
            },
            "Lung Opacity": {
                "probability": 0.30,
                "threshold": 0.50,
                "predicted": False,
            },
            "Nodule/Mass": {
                "probability": 0.10,
                "threshold": 0.50,
                "predicted": False,
            },
            "Pleural Effusion": {
                "probability": 0.60,
                "threshold": 0.50,
                "predicted": True,
            },
            "Pneumothorax": {
                "probability": 0.05,
                "threshold": 0.50,
                "predicted": False,
            },
        },
        "explanation": {
            "available": False,
            "method": None,
            "details": None,
        },
        "reliability": {
            "available": False,
            "details": {
                "reason":
                    "Synthetic API smoke test",
            },
        },
        "metadata": {
            "synthetic": True,
        },
    }


set_inference_handler(
    synthetic_handler
)

client = TestClient(
    app
)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"


def test_info():

    response = client.get(
        "/info"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model"] == "CONVNEXT-TINY"

    assert (
        payload[
            "official_test_inference_enabled"
        ]
        is False
    )


def test_root():

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"


def test_predict_synthetic():

    response = client.post(
        "/predict",
        files={
            "file": (
                "synthetic.png",
                b"synthetic-image-data",
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["model"]["name"]
        == "DenseNet121"
    )

    assert (
        len(payload["predictions"])
        == 7
    )


def test_predict_invalid_type():

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"not-an-image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_predict_empty_file():

    response = client.post(
        "/predict",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
