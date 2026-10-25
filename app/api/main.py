"""
VinBigData_XAI — FastAPI application.

This module creates the API application without automatically
loading the research checkpoint.

Model initialization is intentionally separated from API construction.
"""
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

from fastapi import FastAPI

from .routes import router
from .visual_xai_routes import router as visual_xai_router


APP_NAME = "VinBigData_XAI"
APP_VERSION = "1.0.0"


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Production API for the VinBigData XAI inference pipeline."
    ),
)

app.include_router(
    router
)

app.include_router(
    visual_xai_router
)


@app.get(
    "/",
    include_in_schema=False,
)
def root():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
    }
