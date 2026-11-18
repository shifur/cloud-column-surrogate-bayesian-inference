from copy import deepcopy

import numpy as np
import pytest

from cloud_column_reproduction.figure1.dnn import build_network, predict_dnn
from cloud_column_reproduction.figure1.dnn_gpu import build_cuda_dnn_predictor


@pytest.mark.parametrize(
    "backend", ("torch-persistent-cuda", "torch-compile-cuda")
)
def test_cuda_dnn_candidates_match_frozen_execution_when_available(
    backend: str,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(912)
    model = build_network(11, 6, 8, 50)
    generator = np.random.default_rng(913)
    points = generator.uniform(-1.0, 1.0, size=(16, 11))
    minimum = np.linspace(-2.0, 3.0, 6)
    span = np.linspace(1.0, 4.0, 6)
    reference = predict_dnn(
        deepcopy(model), points, minimum, span, batch_size=len(points)
    )
    predictor = build_cuda_dnn_predictor(
        deepcopy(model), minimum, span, backend=backend
    )
    try:
        first = predictor(points)
        second = predictor(points)
    finally:
        predictor.close()
    np.testing.assert_allclose(first, reference, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(first, second)


def test_cuda_dnn_backend_name_must_be_explicit() -> None:
    torch = pytest.importorskip("torch")
    model = build_network(11, 6, 8, 50)
    with pytest.raises((ValueError, RuntimeError)):
        build_cuda_dnn_predictor(
            model, np.zeros(6), np.ones(6), backend="automatic"
        )

