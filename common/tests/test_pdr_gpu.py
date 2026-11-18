import numpy as np
import pytest

from cloud_column_reproduction.figure1.pdr import predict_pdr
from cloud_column_reproduction.figure1.pdr_gpu import (
    active_polynomial_support,
    build_cuda_pdr_predictor,
)
from cloud_column_reproduction.figure1.polynomials import hyperbolic_cross_indices


def _problem():
    indices = hyperbolic_cross_indices(3, 5)
    generator = np.random.default_rng(81)
    coefficients = generator.normal(size=(len(indices), 6))
    coefficients[1::3] = 0.0
    points = generator.uniform(-1.0, 1.0, size=(16, 3))
    return points, coefficients, indices


def test_exact_zero_support_is_union_across_outputs() -> None:
    _, coefficients, indices = _problem()
    selected_coefficients, selected_indices, active = active_polynomial_support(
        coefficients, indices, prune_exact_zeros=True
    )
    np.testing.assert_array_equal(active, np.any(coefficients != 0.0, axis=1))
    np.testing.assert_array_equal(selected_coefficients, coefficients[active])
    np.testing.assert_array_equal(selected_indices, indices[active])


def test_pruned_cpu_prediction_matches_dense_prediction() -> None:
    points, coefficients, indices = _problem()
    dense = predict_pdr(points, coefficients, indices, prune_exact_zeros=False)
    pruned = predict_pdr(points, coefficients, indices, prune_exact_zeros=True)
    np.testing.assert_allclose(pruned, dense, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.parametrize(
    "backend",
    (
        "torch-cuda", "cupy-fused-cuda", "cupy-zero-skip-cuda",
        "cupy-cached-sparse-cuda",
    ),
)
def test_cuda_predictor_matches_dense_cpu_when_cuda_available(backend: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if backend.startswith("cupy-"):
        pytest.importorskip("cupy")
    points, coefficients, indices = _problem()
    reference = predict_pdr(points, coefficients, indices, prune_exact_zeros=False)
    predictor = build_cuda_pdr_predictor(
        coefficients, indices, backend=backend, prune_exact_zeros=True
    )
    try:
        first = predictor(points)
        second = predictor(points)
    finally:
        predictor.close()
    np.testing.assert_allclose(first, reference, rtol=5.0e-12, atol=5.0e-12)
    np.testing.assert_array_equal(first, second)


def test_cuda_backend_name_must_be_explicit() -> None:
    _, coefficients, indices = _problem()
    with pytest.raises(ValueError, match="Unsupported CUDA"):
        build_cuda_pdr_predictor(
            coefficients, indices, backend="automatic", prune_exact_zeros=True
        )


def test_float32_cs_matches_cpu_and_preserves_frozen_support_hash():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("cupy")
    points, coefficients, indices = _problem()
    reference = predict_pdr(points, coefficients, indices, prune_exact_zeros=False)
    full = build_cuda_pdr_predictor(coefficients, indices,
        backend="cupy-cached-sparse-cuda", prune_exact_zeros=True)
    single = build_cuda_pdr_predictor(coefficients, indices,
        backend="cupy-cached-sparse-float32-cuda", prune_exact_zeros=True)
    try:
        result = single(points)
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)
        assert result.dtype == np.float64
        assert single.coefficient_values.dtype == np.float32
        assert single.active_support_sha256 == full.active_support_sha256
        assert single.kernel_source_sha256 != full.kernel_source_sha256
        assert single.metadata["dtype"] == "float32"
    finally:
        single.close()
        full.close()
