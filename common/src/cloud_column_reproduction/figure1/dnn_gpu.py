"""Persistent CUDA inference candidates for the frozen Figure 1 DNN.

The baseline scientific checkpoint is unchanged.  These wrappers only alter the
online execution path and expose synchronized NumPy predictions so timing
includes all host/device transfers used by the current emcee integration.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class PersistentCUDADNNPredictor:
    """Persistent eager or compiled CUDA execution for a fixed PyTorch model."""

    def __init__(
        self,
        model: Any,
        output_minimum: np.ndarray,
        output_span: np.ndarray,
        *,
        backend: str,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for CUDA DNN inference") from error
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access a CUDA device")
        if backend not in {"torch-persistent-cuda", "torch-compile-cuda"}:
            raise ValueError(f"Unsupported CUDA DNN backend: {backend}")
        self.torch = torch
        self.backend = backend
        self.device = torch.device("cuda:0")
        self.device_name = str(torch.cuda.get_device_name(self.device))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        self.model = model.to(self.device).eval()
        if backend == "torch-compile-cuda":
            if not hasattr(torch, "compile"):
                raise RuntimeError("This PyTorch build does not provide torch.compile")
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )
        self.output_minimum = np.asarray(output_minimum, dtype=np.float64)
        self.output_span = np.asarray(output_span, dtype=np.float64)
        if self.output_minimum.shape != (6,) or self.output_span.shape != (6,):
            raise ValueError("DNN output scaling must contain six values")
        self._input_buffers: dict[int, Any] = {}
        self._compiled_batches: set[int] = set()
        implementation = (
            "torch-compile-reduce-overhead-fullgraph-v1"
            if backend == "torch-compile-cuda"
            else "torch-persistent-inference-mode-v1"
        )
        self.implementation_sha256 = hashlib.sha256(
            implementation.encode("ascii")
        ).hexdigest()

    def _validate(self, canonical_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(canonical_inputs, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 11:
            raise ValueError("DNN CUDA inputs must have shape (n, 11)")
        if not np.all(np.isfinite(values)):
            raise ValueError("DNN CUDA inputs must be finite")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("DNN CUDA inputs must lie inside [-1, 1]")
        return np.ascontiguousarray(values)

    def _input_buffer(self, rows: int):
        if rows not in self._input_buffers:
            self._input_buffers[rows] = self.torch.empty(
                (rows, 11), dtype=self.torch.float32, device=self.device
            )
        return self._input_buffers[rows]

    def raw_device_prediction(self, canonical_inputs: np.ndarray):
        values = self._validate(canonical_inputs)
        if len(values) == 0:
            return self.torch.empty((0, 6), dtype=self.torch.float32, device=self.device)
        inputs = self._input_buffer(len(values))
        inputs.copy_(self.torch.from_numpy(values))
        with self.torch.inference_mode():
            prediction = self.model(inputs)
        if prediction.shape != (len(values), 6):
            raise RuntimeError("CUDA DNN predictor returned the wrong shape")
        self._compiled_batches.add(len(values))
        return prediction

    def __call__(self, canonical_inputs: np.ndarray) -> np.ndarray:
        raw = self.raw_device_prediction(canonical_inputs).cpu().numpy()
        # Preserve the established output-scaling order and float64 return type.
        return (raw + 1.0) * 0.5 * self.output_span + self.output_minimum

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "model_dtype": "float32",
            "output_dtype": "float64",
            "persistent_input_buffers": "true",
            "inference_mode": "true",
            "tf32": "false",
            "implementation_sha256": self.implementation_sha256,
            "compiled_batch_sizes": ",".join(
                str(value) for value in sorted(self._compiled_batches)
            ),
        }

    def close(self) -> None:
        self.torch.cuda.synchronize(self.device)


def build_cuda_dnn_predictor(
    model: Any,
    output_minimum: np.ndarray,
    output_span: np.ndarray,
    *,
    backend: str,
) -> PersistentCUDADNNPredictor:
    return PersistentCUDADNNPredictor(
        model,
        output_minimum,
        output_span,
        backend=backend,
    )

