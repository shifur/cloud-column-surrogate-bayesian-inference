"""Evaluator protocol and common physical-input/output validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from cloud_column_reproduction.domain import ParameterBox


@dataclass(frozen=True)
class EvaluatorMetadata:
    method: str
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    parameter_box_fingerprint: str
    artifact_sha256: str
    evaluation_mode: str
    device: str
    dtype: str
    batch_size: int | None
    workers: int | None


@dataclass(frozen=True)
class EvaluationBatch:
    predictions: np.ndarray
    status: np.ndarray
    messages: tuple[str, ...]

    def __post_init__(self) -> None:
        predictions = np.asarray(self.predictions)
        status = np.asarray(self.status)
        if predictions.ndim != 2 or predictions.shape[1] != 6:
            raise ValueError("Evaluator predictions must have shape (n, 6)")
        if status.shape != (predictions.shape[0],):
            raise ValueError("Evaluator status must have shape (n,)")
        if len(self.messages) != predictions.shape[0]:
            raise ValueError("Evaluator messages must contain one entry per row")


@runtime_checkable
class ForwardEvaluator(Protocol):
    parameter_box: ParameterBox
    metadata: EvaluatorMetadata

    def evaluate(self, physical_parameters: np.ndarray) -> EvaluationBatch:
        """Evaluate a physical batch with shape (n, 11), returning (n, 6)."""


def require_physical_batch(parameter_box: ParameterBox, values: np.ndarray) -> np.ndarray:
    batch = np.asarray(values, dtype=np.float64)
    if batch.ndim == 1:
        batch = batch[None, :]
    if batch.ndim != 2 or batch.shape[1] != parameter_box.dimension:
        raise ValueError(f"Expected physical batch shape (n, {parameter_box.dimension})")
    parameter_box.require_physical(batch)
    return batch


def require_prediction_batch(values: np.ndarray, row_count: int) -> np.ndarray:
    predictions = np.asarray(values)
    if predictions.shape != (row_count, 6):
        raise ValueError(f"Evaluator must return shape ({row_count}, 6)")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Evaluator returned non-finite predictions")
    return predictions
