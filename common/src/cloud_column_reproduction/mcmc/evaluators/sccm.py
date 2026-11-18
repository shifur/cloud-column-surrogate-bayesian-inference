"""Process-parallel scalar SCCM adapter with stable row ordering."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor

import numpy as np

from cloud_column_reproduction.domain import ParameterBox

from .base import EvaluationBatch, EvaluatorMetadata, require_physical_batch


class SCCMEvaluator:
    def __init__(
        self,
        parameter_box: ParameterBox,
        metadata: EvaluatorMetadata,
        evaluate_one: Callable[[np.ndarray], np.ndarray],
        executor: Executor | None = None,
    ) -> None:
        if metadata.method != "sccm":
            raise ValueError("SCCM evaluator metadata must declare method='sccm'")
        if metadata.parameter_names != parameter_box.names:
            raise ValueError("SCCM parameter order does not match the common box")
        if metadata.parameter_box_fingerprint != parameter_box.fingerprint:
            raise ValueError("SCCM parameter-box fingerprint mismatch")
        self.parameter_box = parameter_box
        self.metadata = metadata
        self._evaluate_one = evaluate_one
        self._executor = executor

    def evaluate(self, physical_parameters: np.ndarray) -> EvaluationBatch:
        physical = require_physical_batch(self.parameter_box, physical_parameters)
        if self._executor is None:
            records = [self._safe_one(row) for row in physical]
        else:
            records = list(self._executor.map(self._safe_one, physical))
        predictions = np.stack([record[0] for record in records])
        status = np.asarray([record[1] for record in records], dtype=np.int8)
        messages = tuple(record[2] for record in records)
        return EvaluationBatch(predictions, status, messages)

    def _safe_one(self, row: np.ndarray) -> tuple[np.ndarray, int, str]:
        try:
            prediction = np.asarray(self._evaluate_one(row), dtype=np.float64)
            if prediction.shape != (6,) or not np.all(np.isfinite(prediction)):
                raise ValueError("SCCM row did not return six finite outputs")
            return prediction, 0, ""
        except Exception as error:  # evaluator failures are data, not silent retries
            return np.full(6, np.nan), 1, f"{type(error).__name__}: {error}"
