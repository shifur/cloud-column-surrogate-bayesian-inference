"""Validated adapter for a controlled DNN checkpoint."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cloud_column_reproduction.domain import ParameterBox

from .base import (
    EvaluationBatch,
    EvaluatorMetadata,
    require_physical_batch,
    require_prediction_batch,
)


class DNNEvaluator:
    def __init__(
        self,
        parameter_box: ParameterBox,
        metadata: EvaluatorMetadata,
        predict_canonical: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        if metadata.method != "dnn":
            raise ValueError("DNN evaluator metadata must declare method='dnn'")
        if metadata.parameter_names != parameter_box.names:
            raise ValueError("DNN parameter order does not match the common box")
        if metadata.parameter_box_fingerprint != parameter_box.fingerprint:
            raise ValueError("DNN checkpoint parameter-box fingerprint mismatch")
        self.parameter_box = parameter_box
        self.metadata = metadata
        self._predict_canonical = predict_canonical

    def evaluate(self, physical_parameters: np.ndarray) -> EvaluationBatch:
        physical = require_physical_batch(self.parameter_box, physical_parameters)
        canonical = self.parameter_box.to_canonical(physical)
        predictions = require_prediction_batch(
            self._predict_canonical(canonical), physical.shape[0]
        )
        return EvaluationBatch(
            predictions=predictions,
            status=np.zeros(physical.shape[0], dtype=np.int8),
            messages=("",) * physical.shape[0],
        )
