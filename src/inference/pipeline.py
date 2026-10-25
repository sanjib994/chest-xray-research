"""
VinBigData_XAI — Production Inference Pipeline

Orchestrates:

    preprocessing
        ->
    ConvNeXt prediction
        ->
    optional XAI
        ->
    reliability
        ->
    result formatting

The pipeline never assumes that unavailable evidence exists.
"""

from typing import Any, Callable, Dict, Optional

from .result_formatter import (
    format_result,
    validate_result,
)


class InferencePipeline:
    """
    Production inference orchestrator.
    """

    def __init__(
        self,
        predictor,
        preprocess_fn=None,
        xai_fn=None,
        reliability_fn=None,
        formatter_fn=None,
    ):

        if predictor is None:

            raise ValueError(
                "predictor must not be None."
            )

        self.predictor = predictor

        self.preprocess_fn = (
            preprocess_fn
        )

        self.xai_fn = xai_fn

        self.reliability_fn = (
            reliability_fn
        )

        self.formatter_fn = (
            formatter_fn
            if formatter_fn is not None
            else format_result
        )

    def preprocess(
        self,
        image,
    ):

        if self.preprocess_fn is None:

            return image

        return self.preprocess_fn(
            image
        )

    def predict(
        self,
        image,
        image_id=None,
        generate_xai=False,
        calculate_reliability=False,
    ):

        processed = self.preprocess(
            image
        )

        predictions = (
            self.predictor
            .predict_tensor(
                processed
            )
        )

        xai_result = None

        if (
            generate_xai
            and self.xai_fn is not None
        ):

            xai_result = self.xai_fn(
                processed,
                predictions,
            )

        reliability_result = None

        if (
            calculate_reliability
            and self.reliability_fn is not None
        ):

            reliability_result = (
                self.reliability_fn(
                    predictions
                )
            )

        result = self.formatter_fn(
            predictions=predictions,
            image_id=image_id,
            xai=xai_result,
            reliability=reliability_result,
        )

        validate_result(
            result
        )

        return result
