"""Persistent CUDA inference backends for frozen polynomial surrogates.

The PDR solve is not repeated here.  These classes evaluate an already fitted
Legendre expansion while keeping all immutable model arrays on the GPU.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

# Every selectable CUDA backend records its exact runtime-kernel source hash in
# the validation artifact.


FUSED_LEGENDRE_KERNEL_SOURCE = r"""
extern "C" __global__
void legendre_tensor_design(
    const double* points,
    const long long* indices,
    double* design,
    const int rows,
    const int basis,
    const int dimension)
{
    const long long flat = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long count = (long long)rows * basis;
    if (flat >= count) return;
    const int row = (int)(flat / basis);
    const int column = (int)(flat - (long long)row * basis);
    double product = 1.0;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        const int degree = (int)indices[(long long)column * dimension + coordinate];
        const double x = points[(long long)row * dimension + coordinate];
        double value = 1.0;
        if (degree > 0) {
            double previous = 1.0;
            double current = x;
            if (degree == 1) {
                value = current;
            } else {
                for (int n = 2; n <= degree; ++n) {
                    const double next = ((2.0 * n - 1.0) * x * current
                                         - (n - 1.0) * previous) / n;
                    previous = current;
                    current = next;
                }
                value = current;
            }
        }
        product *= sqrt(2.0 * degree + 1.0) * value;
    }
    design[flat] = product;
}
"""


ZERO_SKIP_LEGENDRE_KERNEL_SOURCE = r"""
extern "C" __global__
void legendre_tensor_design_zero_skip(
    const double* points,
    const unsigned char* indices,
    const double* scales,
    double* design,
    const int rows,
    const int basis,
    const int dimension)
{
    const long long flat = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long count = (long long)rows * basis;
    if (flat >= count) return;
    const int row = (int)(flat / basis);
    const int column = (int)(flat - (long long)row * basis);
    double product = 1.0;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        const int degree = (int)indices[(long long)column * dimension + coordinate];
        if (degree == 0) continue;
        const double x = points[(long long)row * dimension + coordinate];
        double value = x;
        if (degree > 1) {
            double previous = 1.0;
            double current = x;
            for (int n = 2; n <= degree; ++n) {
                const double next = ((2.0 * n - 1.0) * x * current
                                     - (n - 1.0) * previous) / n;
                previous = current;
                current = next;
            }
            value = current;
        }
        product *= scales[degree] * value;
    }
    design[flat] = product;
}
"""


CACHED_SPARSE_LEGENDRE_KERNEL_SOURCE = r"""
extern "C" __global__
void build_normalized_legendre_tables(
    const double* points,
    const unsigned char* maximum_degrees,
    const double* scales,
    double* tables,
    const int rows,
    const int dimension,
    const int degree_count)
{
    const int flat = blockDim.x * blockIdx.x + threadIdx.x;
    const int count = rows * dimension;
    if (flat >= count) return;
    const int row = flat / dimension;
    const int coordinate = flat - row * dimension;
    const int maximum = (int)maximum_degrees[coordinate];
    const double x = points[(long long)row * dimension + coordinate];
    double* table = tables + ((long long)row * dimension + coordinate) * degree_count;
    table[0] = 1.0;
    if (maximum == 0) return;
    double previous = 1.0;
    double current = x;
    table[1] = scales[1] * current;
    for (int degree = 2; degree <= maximum; ++degree) {
        const double next = ((2.0 * degree - 1.0) * x * current
                             - (degree - 1.0) * previous) / degree;
        previous = current;
        current = next;
        table[degree] = scales[degree] * current;
    }
}

extern "C" __global__
void sparse_tensor_output_partials(
    const double* tables,
    const int* multi_offsets,
    const unsigned char* multi_dimensions,
    const unsigned char* multi_degrees,
    const int* coefficient_offsets,
    const unsigned char* coefficient_outputs,
    const double* coefficient_values,
    double* partials,
    const int rows,
    const int basis,
    const int dimension,
    const int degree_count,
    const int tiles)
{
    const int row = blockIdx.x / tiles;
    const int tile = blockIdx.x - row * tiles;
    if (row >= rows) return;
    const int basis_row = tile * blockDim.x + threadIdx.x;
    double accum0 = 0.0;
    double accum1 = 0.0;
    double accum2 = 0.0;
    double accum3 = 0.0;
    double accum4 = 0.0;
    double accum5 = 0.0;
    if (basis_row < basis) {
        double product = 1.0;
        const int multi_start = multi_offsets[basis_row];
        const int multi_stop = multi_offsets[basis_row + 1];
        for (int position = multi_start; position < multi_stop; ++position) {
            const int coordinate = (int)multi_dimensions[position];
            const int degree = (int)multi_degrees[position];
            product *= tables[
                ((long long)row * dimension + coordinate) * degree_count + degree
            ];
        }
        const int coefficient_start = coefficient_offsets[basis_row];
        const int coefficient_stop = coefficient_offsets[basis_row + 1];
        for (int position = coefficient_start; position < coefficient_stop; ++position) {
            const double contribution = product * coefficient_values[position];
            switch ((int)coefficient_outputs[position]) {
                case 0: accum0 += contribution; break;
                case 1: accum1 += contribution; break;
                case 2: accum2 += contribution; break;
                case 3: accum3 += contribution; break;
                case 4: accum4 += contribution; break;
                case 5: accum5 += contribution; break;
            }
        }
    }
    extern __shared__ double shared[];
    const int lane = threadIdx.x;
    shared[0 * blockDim.x + lane] = accum0;
    shared[1 * blockDim.x + lane] = accum1;
    shared[2 * blockDim.x + lane] = accum2;
    shared[3 * blockDim.x + lane] = accum3;
    shared[4 * blockDim.x + lane] = accum4;
    shared[5 * blockDim.x + lane] = accum5;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            #pragma unroll
            for (int output = 0; output < 6; ++output) {
                shared[output * blockDim.x + lane] +=
                    shared[output * blockDim.x + lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        const long long destination = ((long long)row * tiles + tile) * 6;
        #pragma unroll
        for (int output = 0; output < 6; ++output) {
            partials[destination + output] = shared[output * blockDim.x];
        }
    }
}

extern "C" __global__
void reduce_sparse_output_partials(
    const double* partials,
    double* prediction,
    const int rows,
    const int tiles)
{
    const int row = blockIdx.x;
    const int output = threadIdx.x;
    if (row >= rows || output >= 6) return;
    double total = 0.0;
    for (int tile = 0; tile < tiles; ++tile) {
        total += partials[((long long)row * tiles + tile) * 6 + output];
    }
    prediction[(long long)row * 6 + output] = total;
}
"""



def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def active_polynomial_support(
    coefficients: np.ndarray,
    indices: np.ndarray,
    *,
    prune_exact_zeros: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    index_values = np.asarray(indices, dtype=np.int64)
    if coefficient_values.ndim != 2 or index_values.ndim != 2:
        raise ValueError("PDR coefficients and indices must be matrices")
    if len(coefficient_values) != len(index_values):
        raise ValueError("PDR coefficients and indices must have the same basis size")
    active = (
        np.any(coefficient_values != 0.0, axis=1)
        if prune_exact_zeros
        else np.ones(len(coefficient_values), dtype=bool)
    )
    return coefficient_values[active], index_values[active], active


class _PersistentPDRBase:
    def __init__(
        self,
        coefficients: np.ndarray,
        indices: np.ndarray,
        *,
        prune_exact_zeros: bool,
    ) -> None:
        coefficient_values = np.asarray(coefficients, dtype=np.float64)
        self.dense_basis_size = int(len(coefficient_values))
        selected_coefficients, selected_indices, active = active_polynomial_support(
            coefficient_values, indices, prune_exact_zeros=prune_exact_zeros
        )
        if len(selected_indices) == 0:
            raise ValueError("Frozen PDR surrogate has no active polynomial terms")
        self.coefficients_host = np.ascontiguousarray(selected_coefficients)
        self.indices_host = np.ascontiguousarray(selected_indices)
        self.active_mask = np.ascontiguousarray(active)
        self.active_basis_size = int(len(selected_indices))
        self.output_count = int(selected_coefficients.shape[1])
        self.dimension = int(selected_indices.shape[1])
        self.prune_exact_zeros = bool(prune_exact_zeros)
        self.active_support_sha256 = _sha256_arrays(
            self.active_mask, self.indices_host, self.coefficients_host
        )

    def _validate(self, canonical_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(canonical_inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"PDR CUDA input must have shape (n, {self.dimension})"
            )
        if not np.all(np.isfinite(values)) or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("PDR CUDA inputs must be finite and inside [-1, 1]")
        return np.ascontiguousarray(values)

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "dtype": "float64",
            "dense_basis_size": str(self.dense_basis_size),
            "active_basis_size": str(self.active_basis_size),
            "prune_exact_zeros": str(self.prune_exact_zeros).lower(),
            "active_support_sha256": self.active_support_sha256,
            "kernel_source_sha256": self.kernel_source_sha256,
        }

    def close(self) -> None:
        return None


class TorchCUDAPDRPredictor(_PersistentPDRBase):
    """Cached PyTorch-CUDA implementation used as a benchmark candidate."""

    backend = "torch-cuda-cached-legendre"
    kernel_source_sha256 = hashlib.sha256(
        b"torch-vectorized-legendre-recurrence-v1"
    ).hexdigest()

    def __init__(self, coefficients: np.ndarray, indices: np.ndarray, *,
                 prune_exact_zeros: bool = True) -> None:
        super().__init__(coefficients, indices, prune_exact_zeros=prune_exact_zeros)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for torch CUDA PDR inference") from error
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA for PDR inference")
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.device_name = str(torch.cuda.get_device_name(self.device))
        self.coefficients = torch.as_tensor(
            self.coefficients_host, dtype=torch.float64, device=self.device
        )
        self.degrees = tuple(
            torch.as_tensor(
                self.indices_host[:, column], dtype=torch.long, device=self.device
            )
            for column in range(self.dimension)
        )
        self.maximum_degrees = tuple(
            int(np.max(self.indices_host[:, column], initial=0))
            for column in range(self.dimension)
        )
        self.scales = tuple(
            torch.sqrt(
                2.0 * torch.arange(maximum + 1, dtype=torch.float64, device=self.device)
                + 1.0
            )
            for maximum in self.maximum_degrees
        )

    def __call__(self, canonical_inputs: np.ndarray) -> np.ndarray:
        values = self._validate(canonical_inputs)
        torch = self.torch
        points = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        design = torch.ones(
            (len(values), self.active_basis_size),
            dtype=torch.float64,
            device=self.device,
        )
        with torch.no_grad():
            for column, maximum in enumerate(self.maximum_degrees):
                coordinate = points[:, column]
                polynomials = [torch.ones_like(coordinate)]
                if maximum >= 1:
                    polynomials.append(coordinate)
                for degree in range(2, maximum + 1):
                    polynomials.append(
                        ((2 * degree - 1) * coordinate * polynomials[-1]
                         - (degree - 1) * polynomials[-2]) / degree
                    )
                table = torch.stack(polynomials, dim=1)
                degrees = self.degrees[column]
                design *= table[:, degrees] * self.scales[column][degrees]
            prediction = design @ self.coefficients
        return prediction.cpu().numpy()

    def close(self) -> None:
        self.torch.cuda.synchronize(self.device)


class CuPyFusedPDRPredictor(_PersistentPDRBase):
    """Fused tensor-Legendre design kernel followed by a CUDA matrix product."""

    backend = "cupy-cuda-fused-legendre"
    kernel_source_sha256 = hashlib.sha256(
        FUSED_LEGENDRE_KERNEL_SOURCE.encode("utf-8")
    ).hexdigest()

    def __init__(self, coefficients: np.ndarray, indices: np.ndarray, *,
                 prune_exact_zeros: bool = True) -> None:
        super().__init__(coefficients, indices, prune_exact_zeros=prune_exact_zeros)
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError("CuPy is required for fused CUDA PDR inference") from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy cannot access CUDA for PDR inference")
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties.get("name", "unknown CUDA device")
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)
        self.coefficients = cp.asarray(self.coefficients_host, dtype=cp.float64)
        self.indices = cp.asarray(self.indices_host, dtype=cp.int64)
        self.kernel = cp.RawKernel(
            FUSED_LEGENDRE_KERNEL_SOURCE, "legendre_tensor_design"
        )
        self._buffers: dict[int, tuple[Any, Any, Any]] = {}

    def _buffers_for(self, rows: int) -> tuple[Any, Any, Any]:
        if rows not in self._buffers:
            cp = self.cp
            self._buffers[rows] = (
                cp.empty((rows, self.dimension), dtype=cp.float64),
                cp.empty((rows, self.active_basis_size), dtype=cp.float64),
                cp.empty((rows, self.output_count), dtype=cp.float64),
            )
        return self._buffers[rows]

    def __call__(self, canonical_inputs: np.ndarray) -> np.ndarray:
        values = self._validate(canonical_inputs)
        if len(values) == 0:
            return np.empty((0, self.output_count), dtype=np.float64)
        points, design, prediction = self._buffers_for(len(values))
        points.set(values)
        threads = 256
        count = len(values) * self.active_basis_size
        blocks = (count + threads - 1) // threads
        self.kernel(
            (blocks,),
            (threads,),
            (
                points,
                self.indices,
                design,
                np.int32(len(values)),
                np.int32(self.active_basis_size),
                np.int32(self.dimension),
            ),
        )
        self.cp.matmul(design, self.coefficients, out=prediction)
        return prediction.get()

    def close(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()


class CuPyZeroSkipPDRPredictor(_PersistentPDRBase):
    """Current dense-design path with zero-degree and scale-table optimizations."""

    backend = "cupy-cuda-zero-skip-legendre"
    kernel_source_sha256 = hashlib.sha256(
        ZERO_SKIP_LEGENDRE_KERNEL_SOURCE.encode("utf-8")
    ).hexdigest()

    def __init__(self, coefficients: np.ndarray, indices: np.ndarray, *,
                 prune_exact_zeros: bool = True) -> None:
        super().__init__(coefficients, indices, prune_exact_zeros=prune_exact_zeros)
        if np.min(self.indices_host) < 0 or np.max(self.indices_host) > 255:
            raise ValueError("Compact CUDA degree indices require values in [0, 255]")
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError("CuPy is required for optimized CUDA PDR inference") from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy cannot access a CUDA device")
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties.get("name", "unknown CUDA device")
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)
        self.coefficients = cp.asarray(self.coefficients_host, dtype=cp.float64)
        self.indices = cp.asarray(self.indices_host, dtype=cp.uint8)
        maximum = int(np.max(self.indices_host, initial=0))
        self.scales = cp.sqrt(
            2.0 * cp.arange(maximum + 1, dtype=cp.float64) + 1.0
        )
        self.kernel = cp.RawKernel(
            ZERO_SKIP_LEGENDRE_KERNEL_SOURCE,
            "legendre_tensor_design_zero_skip",
        )
        self._buffers: dict[int, tuple[Any, Any, Any]] = {}

    def _buffers_for(self, rows: int) -> tuple[Any, Any, Any]:
        if rows not in self._buffers:
            cp = self.cp
            self._buffers[rows] = (
                cp.empty((rows, self.dimension), dtype=cp.float64),
                cp.empty((rows, self.active_basis_size), dtype=cp.float64),
                cp.empty((rows, self.output_count), dtype=cp.float64),
            )
        return self._buffers[rows]

    def __call__(self, canonical_inputs: np.ndarray) -> np.ndarray:
        values = self._validate(canonical_inputs)
        if len(values) == 0:
            return np.empty((0, self.output_count), dtype=np.float64)
        points, design, prediction = self._buffers_for(len(values))
        points.set(values)
        threads = 256
        count = len(values) * self.active_basis_size
        blocks = (count + threads - 1) // threads
        self.kernel(
            (blocks,),
            (threads,),
            (
                points,
                self.indices,
                self.scales,
                design,
                np.int32(len(values)),
                np.int32(self.active_basis_size),
                np.int32(self.dimension),
            ),
        )
        self.cp.matmul(design, self.coefficients, out=prediction)
        return prediction.get()

    def close(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()


class CuPyCachedSparsePDRPredictor(_PersistentPDRBase):
    """Cached Legendre tables with sparse multi-index and output accumulation."""

    backend = "cupy-cuda-cached-sparse-legendre"
    kernel_source_sha256 = hashlib.sha256(
        CACHED_SPARSE_LEGENDRE_KERNEL_SOURCE.encode("utf-8")
    ).hexdigest()
    threads = 256

    def __init__(self, coefficients: np.ndarray, indices: np.ndarray, *,
                 prune_exact_zeros: bool = True, inference_dtype: str = "float64") -> None:
        if inference_dtype not in {"float32", "float64"}:
            raise ValueError("Cached CS dtype must be float32 or float64")
        self.inference_dtype = np.dtype(inference_dtype)
        source = CACHED_SPARSE_LEGENDRE_KERNEL_SOURCE
        if inference_dtype == "float32":
            source = re.sub(r"\bdouble\b", "float", source)
            source = re.sub(r"(?<![\w.])(\d+\.\d+)(?![\w.])", r"\1f", source)
            self.backend = "cupy-cuda-cached-sparse-legendre-float32"
        self.kernel_source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        super().__init__(coefficients, indices, prune_exact_zeros=prune_exact_zeros)
        if self.output_count != 6:
            raise ValueError("Cached sparse CUDA PDR currently requires six outputs")
        if np.min(self.indices_host) < 0 or np.max(self.indices_host) > 255:
            raise ValueError("Compact CUDA degree indices require values in [0, 255]")
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError("CuPy is required for cached sparse CUDA PDR inference") from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy cannot access a CUDA device")
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties.get("name", "unknown CUDA device")
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)

        multi_offsets = [0]
        multi_dimensions: list[int] = []
        multi_degrees: list[int] = []
        coefficient_offsets = [0]
        coefficient_outputs: list[int] = []
        coefficient_values: list[float] = []
        for index_row, coefficient_row in zip(
            self.indices_host, self.coefficients_host, strict=True
        ):
            nonzero_dimensions = np.flatnonzero(index_row)
            multi_dimensions.extend(int(value) for value in nonzero_dimensions)
            multi_degrees.extend(int(index_row[value]) for value in nonzero_dimensions)
            multi_offsets.append(len(multi_dimensions))
            nonzero_outputs = np.flatnonzero(coefficient_row != 0.0)
            coefficient_outputs.extend(int(value) for value in nonzero_outputs)
            coefficient_values.extend(float(coefficient_row[value]) for value in nonzero_outputs)
            coefficient_offsets.append(len(coefficient_outputs))

        self.maximum_degrees_host = np.max(self.indices_host, axis=0).astype(
            np.uint8, copy=False
        )
        self.degree_count = int(np.max(self.maximum_degrees_host)) + 1
        self.tiles = (
            self.active_basis_size + self.threads - 1
        ) // self.threads
        self.multi_nonzero_count = len(multi_dimensions)
        self.coefficient_nonzero_count = len(coefficient_values)
        self.maximum_degrees = cp.asarray(self.maximum_degrees_host, dtype=cp.uint8)
        self.scales = cp.sqrt(
            2.0 * cp.arange(self.degree_count, dtype=self.inference_dtype) + 1.0
        )
        self.multi_offsets = cp.asarray(multi_offsets, dtype=cp.int32)
        self.multi_dimensions = cp.asarray(multi_dimensions, dtype=cp.uint8)
        self.multi_degrees = cp.asarray(multi_degrees, dtype=cp.uint8)
        self.coefficient_offsets = cp.asarray(coefficient_offsets, dtype=cp.int32)
        self.coefficient_outputs = cp.asarray(coefficient_outputs, dtype=cp.uint8)
        self.coefficient_values = cp.asarray(coefficient_values, dtype=self.inference_dtype)
        module = cp.RawModule(
            code=source,
            name_expressions=(
                "build_normalized_legendre_tables",
                "sparse_tensor_output_partials",
                "reduce_sparse_output_partials",
            ),
        )
        self.table_kernel = module.get_function("build_normalized_legendre_tables")
        self.partial_kernel = module.get_function("sparse_tensor_output_partials")
        self.reduction_kernel = module.get_function("reduce_sparse_output_partials")
        self._module = module
        self._buffers: dict[int, tuple[Any, Any, Any, Any]] = {}

    @property
    def metadata(self) -> dict[str, str]:
        return {
            **super().metadata,
            "dtype": str(self.inference_dtype),
            "output_dtype": "float64",
            "fast_math": "false",
            "multi_index_nonzeros": str(self.multi_nonzero_count),
            "coefficient_nonzeros": str(self.coefficient_nonzero_count),
            "degree_table_width": str(self.degree_count),
            "basis_tiles": str(self.tiles),
            "intermediate_design_matrix": "false",
            "deterministic_fixed_order_reduction": "true",
        }

    def _buffers_for(self, rows: int) -> tuple[Any, Any, Any, Any]:
        if rows not in self._buffers:
            cp = self.cp
            self._buffers[rows] = (
                cp.empty((rows, self.dimension), dtype=self.inference_dtype),
                cp.empty(
                    (rows, self.dimension, self.degree_count),
                    dtype=self.inference_dtype,
                ),
                cp.empty((rows, self.tiles, self.output_count), dtype=self.inference_dtype),
                cp.empty((rows, self.output_count), dtype=self.inference_dtype),
            )
        return self._buffers[rows]

    def __call__(self, canonical_inputs: np.ndarray) -> np.ndarray:
        values = self._validate(canonical_inputs)
        if len(values) == 0:
            return np.empty((0, self.output_count), dtype=np.float64)
        points, tables, partials, prediction = self._buffers_for(len(values))
        points.set(values.astype(self.inference_dtype, copy=False))
        table_count = len(values) * self.dimension
        table_blocks = (table_count + self.threads - 1) // self.threads
        self.table_kernel(
            (table_blocks,),
            (self.threads,),
            (
                points,
                self.maximum_degrees,
                self.scales,
                tables,
                np.int32(len(values)),
                np.int32(self.dimension),
                np.int32(self.degree_count),
            ),
        )
        self.partial_kernel(
            (len(values) * self.tiles,),
            (self.threads,),
            (
                tables,
                self.multi_offsets,
                self.multi_dimensions,
                self.multi_degrees,
                self.coefficient_offsets,
                self.coefficient_outputs,
                self.coefficient_values,
                partials,
                np.int32(len(values)),
                np.int32(self.active_basis_size),
                np.int32(self.dimension),
                np.int32(self.degree_count),
                np.int32(self.tiles),
            ),
            shared_mem=self.threads * self.output_count * self.inference_dtype.itemsize,
        )
        self.reduction_kernel(
            (len(values),),
            (32,),
            (
                partials,
                prediction,
                np.int32(len(values)),
                np.int32(self.tiles),
            ),
        )
        return prediction.get().astype(np.float64, copy=False)

    def close(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()



def build_cuda_pdr_predictor(
    coefficients: np.ndarray,
    indices: np.ndarray,
    *,
    backend: str,
    prune_exact_zeros: bool,
) -> _PersistentPDRBase:
    if backend == "torch-cuda":
        return TorchCUDAPDRPredictor(
            coefficients, indices, prune_exact_zeros=prune_exact_zeros
        )
    if backend == "cupy-fused-cuda":
        return CuPyFusedPDRPredictor(
            coefficients, indices, prune_exact_zeros=prune_exact_zeros
        )
    if backend == "cupy-zero-skip-cuda":
        return CuPyZeroSkipPDRPredictor(
            coefficients, indices, prune_exact_zeros=prune_exact_zeros
        )
    if backend == "cupy-cached-sparse-float32-cuda":
        return CuPyCachedSparsePDRPredictor(
            coefficients, indices, prune_exact_zeros=prune_exact_zeros,
            inference_dtype="float32",
        )
    if backend == "cupy-cached-sparse-cuda":
        return CuPyCachedSparsePDRPredictor(
            coefficients, indices, prune_exact_zeros=prune_exact_zeros
        )
    raise ValueError(f"Unsupported CUDA PDR inference backend: {backend}")
