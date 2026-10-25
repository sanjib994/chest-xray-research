"""
VinBigData_XAI — API schemas.

Pydantic schemas define the stable public API contract.

Unavailable evidence must remain unavailable.
The API never fabricates explanation or reliability values.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class InfoResponse(BaseModel):
    service: str
    version: str
    model: str
    diseases: list[str]
    official_test_inference_enabled: bool
    training_enabled: bool


class DiseasePrediction(BaseModel):
    probability: Optional[float] = None
    threshold: Optional[float] = None
    predicted: Optional[bool] = None


class ExplanationResponse(BaseModel):
    available: bool
    method: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ReliabilityResponse(BaseModel):
    available: bool
    details: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    schema_version: str
    image_id: Optional[str] = None
    model: Dict[str, Any]
    predictions: Dict[str, DiseasePrediction]
    explanation: ExplanationResponse
    reliability: ReliabilityResponse
    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
