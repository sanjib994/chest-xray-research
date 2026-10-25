"""
VinBigData_XAI — Reliability package.
"""

from .calibration import (
    expected_calibration_error,
    brier_score,
    reliability_bins,
)
from .calibrated_evaluation import (
    fit_temperature,
    run_calibration,
    load_verified_report,
)

from .uncertainty import (
    binary_entropy,
    confidence_uncertainty,
    probability_margin,
    multiclass_entropy,
    uncertainty_summary,
)

from .robustness import (
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    prediction_shift,
    prediction_agreement,
)

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "reliability_bins",
    "fit_temperature",
    "run_calibration",
    "load_verified_report",
    "binary_entropy",
    "confidence_uncertainty",
    "probability_margin",
    "multiclass_entropy",
    "uncertainty_summary",
    "add_gaussian_noise",
    "adjust_brightness",
    "adjust_contrast",
    "prediction_shift",
    "prediction_agreement",
]
