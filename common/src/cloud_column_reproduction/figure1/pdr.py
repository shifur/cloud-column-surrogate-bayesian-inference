"""Corrected decoupled restarted primal-dual polynomial surrogate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .polynomials import (
    intrinsic_legendre_weights,
    legendre_design_matrix,
    polynomial_indices,
)


PDR_ALGORITHM_REVISION = "mems-book-configurable-norm-and-index-set-v6"
PDR_BUDGET_POLICY = "ceil-full-restarts"
PDR_RESTART_OUTPUT = "ergodic-average"
PDR_DEFAULT_SPECTRAL_NORM_METHOD = "power-iteration"
PDR_DEFAULT_SPECTRAL_NORM_SAFETY_FACTOR = 1.01
PDR_DEFAULT_SPECTRAL_NORM_SEED = 8675309
PDR_DEFAULT_POWER_ITERATIONS = 50
PDR_DEFAULT_RANDOMIZED_OVERSAMPLES = 10
PDR_DEFAULT_RANDOMIZED_POWER_ITERATIONS = 5
PDR_RANDOMIZED_COMPONENTS = 1
PDR_RANDOMIZED_POWER_NORMALIZER = "auto"
PDR_PROGRESS_INTERVAL = 1000


@dataclass(frozen=True)
class PDRHyperparameters:
    algorithm_revision: str
    lambda_value: float
    tau: float
    sigma: float
    contraction: float
    additive_tolerance: float
    inner_iterations: int
    restart_scale: float
    target_iterations: int
    planned_restarts: int
    planned_iterations: int
    iteration_budget_policy: str
    restart_output: str
    spectral_norm: float
    spectral_norm_raw_estimate: float
    spectral_norm_method: str
    spectral_norm_safety_factor: float
    spectral_norm_seed: int
    spectral_norm_power_iterations: int
    spectral_norm_randomized_components: int
    spectral_norm_randomized_oversamples: int
    spectral_norm_randomized_power_iterations: int
    spectral_norm_randomized_power_normalizer: str


@dataclass(frozen=True)
class PDRDiagnostics:
    realized_iterations: int
    completed_restarts: int
    restart_lengths: tuple[int, ...]
    residual_norms: tuple[float, ...]
    weighted_l1_norms: tuple[float, ...]
    normalized_objectives: tuple[float, ...]
    relative_primal_dual_gaps: tuple[float, ...]
    residual_dual_objectives: tuple[float, ...]
    relative_residual_dual_gaps: tuple[float, ...]
    relative_training_residuals: tuple[float, ...]
    relative_coefficient_changes: tuple[float, ...]
    wall_seconds: float
    total_fit_wall_seconds: float
    index_and_design_wall_seconds: float
    spectral_norm_wall_seconds: float
    spectral_norm_source: str
    backend_transfer_setup_wall_seconds: float
    checkpoint_wall_seconds: float
    design_source: str
    backend: str
    dtype: str


@dataclass(frozen=True)
class PDRConvergenceHistory:
    restarts: tuple[int, ...]
    realized_iterations: tuple[int, ...]
    primal_objectives: tuple[tuple[float, ...], ...]
    normalized_objectives: tuple[tuple[float, ...], ...]
    dual_objectives: tuple[tuple[float, ...], ...]
    relative_primal_dual_gaps: tuple[tuple[float, ...], ...]
    residual_dual_objectives: tuple[tuple[float, ...], ...]
    relative_residual_dual_gaps: tuple[tuple[float, ...], ...]
    relative_training_residuals: tuple[tuple[float, ...], ...]
    relative_coefficient_changes: tuple[tuple[float, ...], ...]


def exact_spectral_norm(matrix: np.ndarray) -> float:
    """Compute ``||A||_2`` by a full double-precision singular-value decomposition."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("spectral-norm input must be a finite matrix")
    norm = float(np.linalg.norm(values, ord=2))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("PDR design matrix has invalid or zero spectral norm")
    return norm


def power_iteration_spectral_norm(
    matrix: np.ndarray,
    *,
    iterations: int = PDR_DEFAULT_POWER_ITERATIONS,
    seed: int = PDR_DEFAULT_SPECTRAL_NORM_SEED,
) -> float:
    """Estimate the largest singular value by power iteration on ``A.T @ A``."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("spectral-norm input must be a finite matrix")
    if iterations < 1:
        raise ValueError("power-iteration count must be positive")
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=values.shape[1])
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        updated = values.T @ (values @ vector)
        updated_norm = float(np.linalg.norm(updated))
        if not np.isfinite(updated_norm) or updated_norm == 0.0:
            raise ValueError("power iteration encountered an invalid zero iterate")
        vector = updated / updated_norm
    estimate = float(np.linalg.norm(values @ vector))
    if not np.isfinite(estimate) or estimate == 0.0:
        raise ValueError("power iteration produced an invalid spectral norm")
    return estimate


def randomized_svd_spectral_norm(
    matrix: np.ndarray,
    *,
    oversamples: int = PDR_DEFAULT_RANDOMIZED_OVERSAMPLES,
    power_iterations: int = PDR_DEFAULT_RANDOMIZED_POWER_ITERATIONS,
    seed: int = PDR_DEFAULT_SPECTRAL_NORM_SEED,
) -> float:
    """Estimate the largest singular value with a rank-one randomized SVD."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("spectral-norm input must be a finite matrix")
    if oversamples < 1 or power_iterations < 0:
        raise ValueError("randomized-SVD settings are invalid")
    try:
        from sklearn.utils.extmath import randomized_svd
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for randomized spectral-norm estimation"
        ) from error
    _, singular_values, _ = randomized_svd(
        values,
        n_components=PDR_RANDOMIZED_COMPONENTS,
        n_oversamples=oversamples,
        n_iter=power_iterations,
        power_iteration_normalizer=PDR_RANDOMIZED_POWER_NORMALIZER,
        transpose="auto",
        random_state=seed,
    )
    estimate = float(singular_values[0])
    if not np.isfinite(estimate) or estimate == 0.0:
        raise ValueError("randomized SVD produced an invalid spectral norm")
    return estimate


def spectral_norm_estimate(
    matrix: np.ndarray,
    *,
    method: str = PDR_DEFAULT_SPECTRAL_NORM_METHOD,
    safety_factor: float = PDR_DEFAULT_SPECTRAL_NORM_SAFETY_FACTOR,
    seed: int = PDR_DEFAULT_SPECTRAL_NORM_SEED,
    power_iterations: int = PDR_DEFAULT_POWER_ITERATIONS,
    randomized_oversamples: int = PDR_DEFAULT_RANDOMIZED_OVERSAMPLES,
    randomized_power_iterations: int = PDR_DEFAULT_RANDOMIZED_POWER_ITERATIONS,
) -> tuple[float, float]:
    """Return the inflated norm used by PDR and its uninflated estimate."""
    if safety_factor < 1.0 or not np.isfinite(safety_factor):
        raise ValueError("spectral-norm safety factor must be finite and at least one")
    if method == "exact-svd":
        raw = exact_spectral_norm(matrix)
    elif method == "power-iteration":
        raw = power_iteration_spectral_norm(
            matrix, iterations=power_iterations, seed=seed
        )
    elif method == "randomized-svd":
        raw = randomized_svd_spectral_norm(
            matrix,
            oversamples=randomized_oversamples,
            power_iterations=randomized_power_iterations,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported spectral-norm method: {method}")
    return safety_factor * raw, raw


def table51_hyperparameters(
    scaled_design: np.ndarray,
    sample_count: int,
    target_iterations: int,
    contraction: float,
    additive_tolerance: float,
    spectral_norm_method: str = PDR_DEFAULT_SPECTRAL_NORM_METHOD,
    spectral_norm_safety_factor: float = PDR_DEFAULT_SPECTRAL_NORM_SAFETY_FACTOR,
    spectral_norm_seed: int = PDR_DEFAULT_SPECTRAL_NORM_SEED,
    spectral_norm_power_iterations: int = PDR_DEFAULT_POWER_ITERATIONS,
    spectral_norm_randomized_oversamples: int = PDR_DEFAULT_RANDOMIZED_OVERSAMPLES,
    spectral_norm_randomized_power_iterations: int = (
        PDR_DEFAULT_RANDOMIZED_POWER_ITERATIONS
    ),
    precomputed_spectral_norm: tuple[float, float] | None = None,
) -> PDRHyperparameters:
    if target_iterations < 1:
        raise ValueError("PDR target iteration budget must be positive")
    if not 0.0 < contraction < 1.0:
        raise ValueError("PDR restart contraction must lie strictly between zero and one")
    if additive_tolerance <= 0.0:
        raise ValueError("PDR restart additive tolerance must be positive")
    if precomputed_spectral_norm is None:
        norm, raw_norm = spectral_norm_estimate(
            scaled_design,
            method=spectral_norm_method,
            safety_factor=spectral_norm_safety_factor,
            seed=spectral_norm_seed,
            power_iterations=spectral_norm_power_iterations,
            randomized_oversamples=spectral_norm_randomized_oversamples,
            randomized_power_iterations=spectral_norm_randomized_power_iterations,
        )
    else:
        norm, raw_norm = (float(value) for value in precomputed_spectral_norm)
        if (
            not np.isfinite(norm)
            or not np.isfinite(raw_norm)
            or raw_norm <= 0.0
            or norm != spectral_norm_safety_factor * raw_norm
        ):
            raise ValueError("Precomputed spectral norm is inconsistent with the safety factor")
    tau = 1.0 / norm
    sigma = 1.0 / norm
    inner = max(
        1,
        int(np.ceil(2.0 / (contraction * np.sqrt(tau * sigma)))),
    )
    restarts = int(math.ceil(target_iterations / inner))
    return PDRHyperparameters(
        algorithm_revision=PDR_ALGORITHM_REVISION,
        lambda_value=1.0 / np.sqrt(25.0 * sample_count),
        tau=tau,
        sigma=sigma,
        contraction=contraction,
        additive_tolerance=additive_tolerance,
        inner_iterations=inner,
        restart_scale=inner / (2.0 * norm),
        target_iterations=target_iterations,
        planned_restarts=restarts,
        planned_iterations=restarts * inner,
        iteration_budget_policy=PDR_BUDGET_POLICY,
        restart_output=PDR_RESTART_OUTPUT,
        spectral_norm=norm,
        spectral_norm_raw_estimate=raw_norm,
        spectral_norm_method=spectral_norm_method,
        spectral_norm_safety_factor=spectral_norm_safety_factor,
        spectral_norm_seed=spectral_norm_seed,
        spectral_norm_power_iterations=spectral_norm_power_iterations,
        spectral_norm_randomized_components=PDR_RANDOMIZED_COMPONENTS,
        spectral_norm_randomized_oversamples=spectral_norm_randomized_oversamples,
        spectral_norm_randomized_power_iterations=(
            spectral_norm_randomized_power_iterations
        ),
        spectral_norm_randomized_power_normalizer=(
            PDR_RANDOMIZED_POWER_NORMALIZER
        ),
    )


class _TorchLinalg:
    @staticmethod
    def norm(array: Any, axis: int):
        import torch
        return torch.linalg.vector_norm(array, dim=axis)


class _TorchCUDA:
    """Minimal NumPy-like facade for the PDR operations used on CUDA."""

    def __init__(self) -> None:
        import torch
        self.torch = torch
        self.float32 = torch.float32
        self.float64 = torch.float64
        self.linalg = _TorchLinalg()

    def asarray(self, values: Any, dtype: Any = None):
        return self.torch.as_tensor(values, dtype=dtype, device="cuda:0")

    def zeros(self, shape: Any, dtype: Any = None):
        return self.torch.zeros(shape, dtype=dtype, device="cuda:0")

    def zeros_like(self, values: Any):
        return self.torch.zeros_like(values)

    def sign(self, values: Any):
        return self.torch.sign(values)

    def abs(self, values: Any):
        return self.torch.abs(values)

    def maximum(self, left: Any, right: Any):
        if not self.torch.is_tensor(left):
            left = self.torch.as_tensor(left, dtype=right.dtype, device=right.device)
        if not self.torch.is_tensor(right):
            right = self.torch.as_tensor(right, dtype=left.dtype, device=left.device)
        return self.torch.maximum(left, right)

    def sum(self, values: Any, axis: int):
        return self.torch.sum(values, dim=axis)

    def max(self, values: Any, axis: int):
        return self.torch.amax(values, dim=axis)


def _backend(use_gpu: bool):
    if not use_gpu:
        return np, "numpy-cpu"
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for GPU PDR execution") from error
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA device for PDR")
    return _TorchCUDA(), "torch-cuda"


def _to_numpy(array: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(array):
        return array.detach().cpu().numpy()
    try:
        import cupy as cp
    except ImportError:
        cp = None
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _array_sha256(array: np.ndarray) -> str:
    """Fingerprint an array's exact shape, dtype, and C-order bytes."""
    values = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _progress_report_due(
    previous_realized: int,
    realized: int,
    planned_iterations: int,
) -> bool:
    """Report once per 1,000-iteration boundary and once at completion."""
    crossed_boundary = (
        realized // PDR_PROGRESS_INTERVAL
        > previous_realized // PDR_PROGRESS_INTERVAL
    )
    return crossed_boundary or realized == planned_iterations


def fit_pdr(
    canonical_inputs: np.ndarray,
    targets: np.ndarray,
    *,
    order: int,
    index_set: str = "hyperbolic-cross",
    target_iterations: int,
    additive_tolerance: float,
    contraction: float,
    use_gpu: bool,
    dtype: np.dtype | type = np.float64,
    checkpoint_path: str | Path | None = None,
    spectral_norm_method: str = PDR_DEFAULT_SPECTRAL_NORM_METHOD,
    spectral_norm_safety_factor: float = PDR_DEFAULT_SPECTRAL_NORM_SAFETY_FACTOR,
    spectral_norm_seed: int = PDR_DEFAULT_SPECTRAL_NORM_SEED,
    spectral_norm_power_iterations: int = PDR_DEFAULT_POWER_ITERATIONS,
    spectral_norm_randomized_oversamples: int = PDR_DEFAULT_RANDOMIZED_OVERSAMPLES,
    spectral_norm_randomized_power_iterations: int = (
        PDR_DEFAULT_RANDOMIZED_POWER_ITERATIONS
    ),
    precomputed_unscaled_design: np.ndarray | None = None,
    precomputed_indices: np.ndarray | None = None,
    precomputed_intrinsic_weights: np.ndarray | None = None,
    precomputed_spectral_norm: tuple[float, float] | None = None,
    precomputed_design_source: str = "checksum-validated-float64-prefix-cache",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PDRHyperparameters,
    PDRDiagnostics,
    PDRConvergenceHistory,
]:
    """Fit all outputs together while preserving independent restart scales.

    Columns share matrix multiplications only. Each output has its own residual
    norm, restart scale, dual unit-ball projection, and coefficients. The dual
    variable is intentionally reset to zero at every restart.
    """
    inputs = np.asarray(canonical_inputs, dtype=np.float64)
    responses = np.asarray(targets, dtype=np.float64)
    if inputs.ndim != 2 or responses.ndim != 2 or len(inputs) != len(responses):
        raise ValueError("PDR inputs and targets must be aligned matrices")
    fit_started = time.perf_counter()
    sample_count = len(inputs)
    design_started = time.perf_counter()
    expected_indices = polynomial_indices(inputs.shape[1], index_set, order)
    supplied = (
        precomputed_unscaled_design, precomputed_indices,
        precomputed_intrinsic_weights,
    )
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError("Precomputed PDR design, indices, and weights are all required")
        indices = np.asarray(precomputed_indices, dtype=np.int64)
        weights = np.asarray(precomputed_intrinsic_weights, dtype=np.float64)
        unscaled = np.asarray(precomputed_unscaled_design, dtype=np.float64)
        if not np.array_equal(indices, expected_indices):
            raise ValueError("Precomputed PDR indices do not match the requested space")
        if not np.array_equal(weights, intrinsic_legendre_weights(expected_indices)):
            raise ValueError("Precomputed PDR intrinsic weights do not match the basis")
        if unscaled.shape != (sample_count, len(indices)) or not np.all(np.isfinite(unscaled)):
            raise ValueError("Precomputed PDR design has the wrong shape or non-finite values")
        design_source = str(precomputed_design_source)
        if not design_source:
            raise ValueError("Precomputed PDR design source must be nonempty")
    else:
        indices = expected_indices
        weights = intrinsic_legendre_weights(indices)
        unscaled = legendre_design_matrix(inputs, indices, dtype=np.float64)
        design_source = "computed-in-fit"
    root_m = np.sqrt(float(sample_count))
    design = unscaled / root_m
    right_hand_side = responses / root_m
    index_and_design_wall_seconds = time.perf_counter() - design_started
    spectral_norm_started = time.perf_counter()
    hyperparameters = table51_hyperparameters(
        design,
        sample_count,
        target_iterations,
        contraction,
        additive_tolerance,
        spectral_norm_method=spectral_norm_method,
        spectral_norm_safety_factor=spectral_norm_safety_factor,
        spectral_norm_seed=spectral_norm_seed,
        spectral_norm_power_iterations=spectral_norm_power_iterations,
        spectral_norm_randomized_oversamples=(
            spectral_norm_randomized_oversamples
        ),
        spectral_norm_randomized_power_iterations=(
            spectral_norm_randomized_power_iterations
        ),
        precomputed_spectral_norm=precomputed_spectral_norm,
    )
    spectral_norm_wall_seconds = time.perf_counter() - spectral_norm_started
    design_sha256 = _array_sha256(design)
    right_hand_side_sha256 = _array_sha256(right_hand_side)

    backend_setup_started = time.perf_counter()
    xp, backend_name = _backend(use_gpu)
    backend_dtype = xp.float64 if np.dtype(dtype) == np.float64 else xp.float32
    A = xp.asarray(design, dtype=backend_dtype)
    B = xp.asarray(right_hand_side, dtype=backend_dtype)
    W = xp.asarray(weights, dtype=backend_dtype).reshape(-1, 1)
    coefficient_count = A.shape[1]
    output_count = B.shape[1]
    coefficients = xp.zeros((coefficient_count, output_count), dtype=backend_dtype)
    epsilons = xp.linalg.norm(B, axis=0)
    if use_gpu:
        if backend_name == "torch-cuda":
            xp.torch.cuda.synchronize()
        else:
            xp.cuda.Stream.null.synchronize()
    backend_transfer_setup_wall_seconds = time.perf_counter() - backend_setup_started
    realized = 0
    restart_lengths: list[int] = []
    primal_objective_history: list[np.ndarray] = []
    normalized_objective_history: list[np.ndarray] = []
    dual_objective_history: list[np.ndarray] = []
    relative_gap_history: list[np.ndarray] = []
    residual_dual_objective_history: list[np.ndarray] = []
    relative_residual_dual_gap_history: list[np.ndarray] = []
    relative_residual_history: list[np.ndarray] = []
    relative_coefficient_change_history: list[np.ndarray] = []

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint is not None and checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as saved:
            required = {
                "algorithm_revision",
                "iteration_budget_policy",
                "restart_output",
                "index_set",
                "index_level",
                "target_iterations",
                "planned_restarts",
                "planned_iterations",
                "inner_iterations",
                "additive_tolerance",
                "contraction",
                "spectral_norm",
                "spectral_norm_raw_estimate",
                "spectral_norm_method",
                "spectral_norm_safety_factor",
                "spectral_norm_seed",
                "spectral_norm_power_iterations",
                "spectral_norm_randomized_components",
                "spectral_norm_randomized_oversamples",
                "spectral_norm_randomized_power_iterations",
                "spectral_norm_randomized_power_normalizer",
                "design_sha256",
                "right_hand_side_sha256",
                "backend_dtype",
                "indices",
                "coefficients",
                "epsilons",
                "realized_iterations",
                "restart_lengths",
                "primal_objective_history",
                "normalized_objective_history",
                "dual_objective_history",
                "relative_gap_history",
                "residual_dual_objective_history",
                "relative_residual_dual_gap_history",
                "relative_residual_history",
                "relative_coefficient_change_history",
            }
            missing = required.difference(saved.files)
            if missing:
                raise ValueError(
                    "PDR checkpoint predates the MEMS-book schedule and cannot be "
                    f"resumed; missing metadata: {sorted(missing)}"
                )
            expected_scalars = {
                "algorithm_revision": PDR_ALGORITHM_REVISION,
                "iteration_budget_policy": PDR_BUDGET_POLICY,
                "restart_output": PDR_RESTART_OUTPUT,
                "index_set": index_set,
                "index_level": order,
                "target_iterations": target_iterations,
                "planned_restarts": hyperparameters.planned_restarts,
                "planned_iterations": hyperparameters.planned_iterations,
                "inner_iterations": hyperparameters.inner_iterations,
                "additive_tolerance": additive_tolerance,
                "contraction": contraction,
                "spectral_norm": hyperparameters.spectral_norm,
                "spectral_norm_raw_estimate": (
                    hyperparameters.spectral_norm_raw_estimate
                ),
                "spectral_norm_method": hyperparameters.spectral_norm_method,
                "spectral_norm_safety_factor": (
                    hyperparameters.spectral_norm_safety_factor
                ),
                "spectral_norm_seed": hyperparameters.spectral_norm_seed,
                "spectral_norm_power_iterations": (
                    hyperparameters.spectral_norm_power_iterations
                ),
                "spectral_norm_randomized_components": (
                    hyperparameters.spectral_norm_randomized_components
                ),
                "spectral_norm_randomized_oversamples": (
                    hyperparameters.spectral_norm_randomized_oversamples
                ),
                "spectral_norm_randomized_power_iterations": (
                    hyperparameters.spectral_norm_randomized_power_iterations
                ),
                "spectral_norm_randomized_power_normalizer": (
                    hyperparameters.spectral_norm_randomized_power_normalizer
                ),
                "design_sha256": design_sha256,
                "right_hand_side_sha256": right_hand_side_sha256,
                "backend_dtype": np.dtype(dtype).name,
            }
            for name, expected in expected_scalars.items():
                actual = saved[name].item()
                if actual != expected:
                    raise ValueError(
                        f"PDR checkpoint {name} does not match the requested run: "
                        f"{actual!r} != {expected!r}"
                    )
            if not np.array_equal(saved["indices"], indices):
                raise ValueError("PDR checkpoint basis does not match the run")
            coefficients = xp.asarray(saved["coefficients"], dtype=backend_dtype)
            epsilons = xp.asarray(saved["epsilons"], dtype=backend_dtype)
            realized = int(saved["realized_iterations"])
            restart_lengths = [int(value) for value in saved["restart_lengths"]]
            history_targets = {
                "primal_objective_history": primal_objective_history,
                "normalized_objective_history": normalized_objective_history,
                "dual_objective_history": dual_objective_history,
                "relative_gap_history": relative_gap_history,
                "residual_dual_objective_history": residual_dual_objective_history,
                "relative_residual_dual_gap_history": (
                    relative_residual_dual_gap_history
                ),
                "relative_residual_history": relative_residual_history,
                "relative_coefficient_change_history": (
                    relative_coefficient_change_history
                ),
            }
            expected_history_shape = (len(restart_lengths), int(output_count))
            for name, destination in history_targets.items():
                values = np.asarray(saved[name], dtype=np.float64)
                if values.shape != expected_history_shape:
                    raise ValueError(
                        f"PDR checkpoint {name} has shape {values.shape}, expected "
                        f"{expected_history_shape}"
                    )
                destination.extend(values)
        if any(length != hyperparameters.inner_iterations for length in restart_lengths):
            raise ValueError("PDR checkpoint contains a truncated restart")
        if len(restart_lengths) > hyperparameters.planned_restarts:
            raise ValueError("PDR checkpoint contains too many restarts")
        if realized != len(restart_lengths) * hyperparameters.inner_iterations:
            raise ValueError("PDR checkpoint realized count is inconsistent")

    checkpoint_wall_seconds = 0.0
    solve_started = time.perf_counter()
    while len(restart_lengths) < hyperparameters.planned_restarts:
        length = hyperparameters.inner_iterations
        previous_coefficients = coefficients
        epsilons = contraction * (epsilons + additive_tolerance)
        scales = hyperparameters.restart_scale * epsilons
        primal = coefficients / scales.reshape(1, -1)
        scaled_B = B / scales.reshape(1, -1)
        dual = xp.zeros((sample_count, output_count), dtype=backend_dtype)
        average = xp.zeros_like(primal)
        dual_average = xp.zeros_like(dual)
        threshold = (
            hyperparameters.tau * hyperparameters.lambda_value * W
        )
        transpose = A.T
        for iteration in range(length):
            proposal = primal - hyperparameters.tau * (transpose @ dual)
            primal_next = xp.sign(proposal) * xp.maximum(
                xp.abs(proposal) - threshold, 0.0
            )
            dual_proposal = (
                dual
                + hyperparameters.sigma * (A @ (2.0 * primal_next - primal))
                - hyperparameters.sigma * scaled_B
            )
            dual_norms = xp.linalg.norm(dual_proposal, axis=0)
            dual_next = dual_proposal / xp.maximum(1.0, dual_norms).reshape(1, -1)
            average += (primal_next - average) / float(iteration + 1)
            dual_average += (dual_next - dual_average) / float(iteration + 1)
            primal, dual = primal_next, dual_next
        coefficients = average * scales.reshape(1, -1)
        realized += length
        restart_lengths.append(length)

        residual = A @ coefficients - B
        residual_norms = xp.linalg.norm(residual, axis=0)
        right_hand_side_norms = xp.linalg.norm(B, axis=0)
        weighted_l1_norms = xp.sum(W * xp.abs(coefficients), axis=0)
        primal_objectives = (
            hyperparameters.lambda_value * weighted_l1_norms + residual_norms
        )

        dual_constraint_ratios = xp.max(
            xp.abs(transpose @ dual_average) / (hyperparameters.lambda_value * W),
            axis=0,
        )
        dual_norms = xp.linalg.norm(dual_average, axis=0)
        dual_scales = xp.maximum(
            1.0, xp.maximum(dual_norms, dual_constraint_ratios)
        )
        feasible_dual = dual_average / dual_scales.reshape(1, -1)
        dual_objectives = -xp.sum(B * feasible_dual, axis=0)

        numerical_floor = xp.asarray(
            np.finfo(np.dtype(dtype)).eps, dtype=backend_dtype
        )
        primal_dual_gaps = xp.maximum(primal_objectives - dual_objectives, 0.0)
        relative_gaps = primal_dual_gaps / xp.maximum(
            xp.maximum(xp.abs(primal_objectives), xp.abs(dual_objectives)),
            numerical_floor,
        )

        residual_dual_candidate = residual / xp.maximum(
            residual_norms, numerical_floor
        ).reshape(1, -1)
        residual_dual_constraint_ratios = xp.max(
            xp.abs(transpose @ residual_dual_candidate)
            / (hyperparameters.lambda_value * W),
            axis=0,
        )
        residual_dual_scales = xp.maximum(
            1.0,
            xp.maximum(
                xp.linalg.norm(residual_dual_candidate, axis=0),
                residual_dual_constraint_ratios,
            ),
        )
        feasible_residual_dual = residual_dual_candidate / (
            residual_dual_scales.reshape(1, -1)
        )
        residual_dual_objectives = -xp.sum(B * feasible_residual_dual, axis=0)
        residual_dual_gaps = xp.maximum(
            primal_objectives - residual_dual_objectives, 0.0
        )
        relative_residual_dual_gaps = residual_dual_gaps / xp.maximum(
            xp.maximum(
                xp.abs(primal_objectives), xp.abs(residual_dual_objectives)
            ),
            numerical_floor,
        )
        relative_residuals = residual_norms / xp.maximum(
            right_hand_side_norms, numerical_floor
        )
        normalized_objectives = primal_objectives / xp.maximum(
            right_hand_side_norms, numerical_floor
        )
        coefficient_changes = xp.linalg.norm(
            coefficients - previous_coefficients, axis=0
        ) / xp.maximum(xp.linalg.norm(coefficients, axis=0), numerical_floor)

        metric_rows = (
            (primal_objective_history, primal_objectives),
            (normalized_objective_history, normalized_objectives),
            (dual_objective_history, dual_objectives),
            (relative_gap_history, relative_gaps),
            (residual_dual_objective_history, residual_dual_objectives),
            (relative_residual_dual_gap_history, relative_residual_dual_gaps),
            (relative_residual_history, relative_residuals),
            (relative_coefficient_change_history, coefficient_changes),
        )
        for destination, values in metric_rows:
            destination.append(
                _to_numpy(values).astype(np.float64, copy=False).reshape(-1)
            )
        gap_values = relative_gap_history[-1]
        residual_gap_values = relative_residual_dual_gap_history[-1]
        residual_values = relative_residual_history[-1]
        change_values = relative_coefficient_change_history[-1]
        previous_realized = realized - length
        report_due = _progress_report_due(
            previous_realized,
            realized,
            hyperparameters.planned_iterations,
        )
        if report_due:
            print(
                f"PDR iterations: {realized}/{hyperparameters.planned_iterations} "
                f"(nominal budget {target_iterations}; restart "
                f"{len(restart_lengths)}/{hyperparameters.planned_restarts}) | "
                f"rel_gap median={np.median(gap_values):.3e} "
                f"max={np.max(gap_values):.3e} | "
                f"resid_gap median={np.median(residual_gap_values):.3e} "
                f"max={np.max(residual_gap_values):.3e} | "
                f"rel_resid median={np.median(residual_values):.3e} "
                f"max={np.max(residual_values):.3e} | "
                f"rel_dc max={np.max(change_values):.3e}",
                flush=True,
            )
        if checkpoint is not None and report_due:
            checkpoint_started = time.perf_counter()
            _atomic_npz(
                checkpoint,
                coefficients=_to_numpy(coefficients),
                epsilons=_to_numpy(epsilons),
                realized_iterations=np.asarray(realized),
                restart_lengths=np.asarray(restart_lengths, dtype=np.int64),
                primal_objective_history=np.asarray(primal_objective_history),
                normalized_objective_history=np.asarray(normalized_objective_history),
                dual_objective_history=np.asarray(dual_objective_history),
                relative_gap_history=np.asarray(relative_gap_history),
                residual_dual_objective_history=np.asarray(
                    residual_dual_objective_history
                ),
                relative_residual_dual_gap_history=np.asarray(
                    relative_residual_dual_gap_history
                ),
                relative_residual_history=np.asarray(relative_residual_history),
                relative_coefficient_change_history=np.asarray(
                    relative_coefficient_change_history
                ),
                algorithm_revision=np.asarray(PDR_ALGORITHM_REVISION),
                iteration_budget_policy=np.asarray(PDR_BUDGET_POLICY),
                restart_output=np.asarray(PDR_RESTART_OUTPUT),
                index_set=np.asarray(index_set),
                index_level=np.asarray(order),
                target_iterations=np.asarray(target_iterations),
                planned_restarts=np.asarray(hyperparameters.planned_restarts),
                planned_iterations=np.asarray(hyperparameters.planned_iterations),
                inner_iterations=np.asarray(hyperparameters.inner_iterations),
                additive_tolerance=np.asarray(additive_tolerance),
                contraction=np.asarray(contraction),
                spectral_norm=np.asarray(hyperparameters.spectral_norm),
                spectral_norm_raw_estimate=np.asarray(
                    hyperparameters.spectral_norm_raw_estimate
                ),
                spectral_norm_method=np.asarray(
                    hyperparameters.spectral_norm_method
                ),
                spectral_norm_safety_factor=np.asarray(
                    hyperparameters.spectral_norm_safety_factor
                ),
                spectral_norm_seed=np.asarray(hyperparameters.spectral_norm_seed),
                spectral_norm_power_iterations=np.asarray(
                    hyperparameters.spectral_norm_power_iterations
                ),
                spectral_norm_randomized_components=np.asarray(
                    hyperparameters.spectral_norm_randomized_components
                ),
                spectral_norm_randomized_oversamples=np.asarray(
                    hyperparameters.spectral_norm_randomized_oversamples
                ),
                spectral_norm_randomized_power_iterations=np.asarray(
                    hyperparameters.spectral_norm_randomized_power_iterations
                ),
                spectral_norm_randomized_power_normalizer=np.asarray(
                    hyperparameters.spectral_norm_randomized_power_normalizer
                ),
                design_sha256=np.asarray(design_sha256),
                right_hand_side_sha256=np.asarray(right_hand_side_sha256),
                backend_dtype=np.asarray(np.dtype(dtype).name),
                indices=indices,
            )
            checkpoint_wall_seconds += time.perf_counter() - checkpoint_started
    if use_gpu:
        if backend_name == "torch-cuda":
            xp.torch.cuda.synchronize()
        else:
            xp.cuda.Stream.null.synchronize()
    wall_seconds = time.perf_counter() - solve_started
    total_fit_wall_seconds = time.perf_counter() - fit_started
    coefficient_array = _to_numpy(coefficients).astype(np.float64, copy=False)
    residual = design @ coefficient_array - right_hand_side
    diagnostics = PDRDiagnostics(
        realized_iterations=realized,
        completed_restarts=len(restart_lengths),
        restart_lengths=tuple(restart_lengths),
        residual_norms=tuple(float(value) for value in np.linalg.norm(residual, axis=0)),
        weighted_l1_norms=tuple(
            float(value)
            for value in np.sum(weights[:, None] * np.abs(coefficient_array), axis=0)
        ),
        normalized_objectives=tuple(float(value) for value in normalized_objective_history[-1]),
        relative_primal_dual_gaps=tuple(float(value) for value in relative_gap_history[-1]),
        residual_dual_objectives=tuple(
            float(value) for value in residual_dual_objective_history[-1]
        ),
        relative_residual_dual_gaps=tuple(
            float(value) for value in relative_residual_dual_gap_history[-1]
        ),
        relative_training_residuals=tuple(
            float(value) for value in relative_residual_history[-1]
        ),
        relative_coefficient_changes=tuple(
            float(value) for value in relative_coefficient_change_history[-1]
        ),
        wall_seconds=wall_seconds,
        total_fit_wall_seconds=total_fit_wall_seconds,
        index_and_design_wall_seconds=index_and_design_wall_seconds,
        spectral_norm_wall_seconds=spectral_norm_wall_seconds,
        spectral_norm_source=(
            "checksum-validated-prefix-cache"
            if precomputed_spectral_norm is not None
            else "computed-in-fit"
        ),
        backend_transfer_setup_wall_seconds=backend_transfer_setup_wall_seconds,
        checkpoint_wall_seconds=checkpoint_wall_seconds,
        design_source=design_source,
        backend=backend_name,
        dtype=np.dtype(dtype).name,
    )
    convergence = PDRConvergenceHistory(
        restarts=tuple(range(1, len(restart_lengths) + 1)),
        realized_iterations=tuple(
            hyperparameters.inner_iterations * restart
            for restart in range(1, len(restart_lengths) + 1)
        ),
        primal_objectives=tuple(tuple(float(value) for value in row) for row in primal_objective_history),
        normalized_objectives=tuple(
            tuple(float(value) for value in row) for row in normalized_objective_history
        ),
        dual_objectives=tuple(tuple(float(value) for value in row) for row in dual_objective_history),
        relative_primal_dual_gaps=tuple(
            tuple(float(value) for value in row) for row in relative_gap_history
        ),
        residual_dual_objectives=tuple(
            tuple(float(value) for value in row)
            for row in residual_dual_objective_history
        ),
        relative_residual_dual_gaps=tuple(
            tuple(float(value) for value in row)
            for row in relative_residual_dual_gap_history
        ),
        relative_training_residuals=tuple(
            tuple(float(value) for value in row) for row in relative_residual_history
        ),
        relative_coefficient_changes=tuple(
            tuple(float(value) for value in row)
            for row in relative_coefficient_change_history
        ),
    )
    return coefficient_array, indices, weights, hyperparameters, diagnostics, convergence


def predict_pdr(
    canonical_inputs: np.ndarray,
    coefficients: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = 512,
    workers: int = 1,
    prune_exact_zeros: bool = False,
) -> np.ndarray:
    """Evaluate a fitted polynomial surrogate in bounded-memory chunks.

    Exact-zero support pruning evaluates only basis rows retained by the fitted
    q=23 CS model. The option is recorded in the artifact and runtime manifest.
    """
    inputs = np.asarray(canonical_inputs, dtype=np.float64)
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    index_values = np.asarray(indices, dtype=np.int64)
    if coefficient_values.ndim != 2 or index_values.ndim != 2:
        raise ValueError("PDR coefficients and indices must be two-dimensional")
    if len(coefficient_values) != len(index_values):
        raise ValueError("PDR coefficients and indices must have the same basis size")
    if chunk_size < 1 or workers < 1:
        raise ValueError("PDR prediction chunk size and worker count must be positive")
    if prune_exact_zeros:
        active = np.any(coefficient_values != 0.0, axis=1)
        coefficient_values = coefficient_values[active]
        index_values = index_values[active]
    result = np.zeros((len(inputs), coefficient_values.shape[1]), dtype=np.float64)
    if len(index_values) == 0:
        return result

    chunks = tuple(
        (start, min(start + chunk_size, len(inputs)))
        for start in range(0, len(inputs), chunk_size)
    )

    def evaluate_chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
        start, stop = bounds
        design = legendre_design_matrix(
            inputs[start:stop], index_values, dtype=np.float64
        )
        return start, design @ coefficient_values

    if workers == 1:
        evaluated = map(evaluate_chunk, chunks)
        for start, prediction in evaluated:
            result[start : start + len(prediction)] = prediction
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start, prediction in pool.map(evaluate_chunk, chunks):
            result[start : start + len(prediction)] = prediction
    return result


def hyperparameters_dict(value: PDRHyperparameters) -> dict[str, Any]:
    return asdict(value)


def diagnostics_dict(value: PDRDiagnostics) -> dict[str, Any]:
    return asdict(value)
