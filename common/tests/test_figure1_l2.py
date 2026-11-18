import numpy as np
import pytest

from cloud_column_reproduction.figure1.l2 import (
    equal_weight_relative_l2,
    raw_signed_quadrature_terms,
    replicated_rqmc_relative_l2,
    signed_quadrature_relative_l2,
    signed_quadrature_terms,
)


def test_equal_weight_relative_l2() -> None:
    true = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    predicted = 0.5 * true
    np.testing.assert_allclose(equal_weight_relative_l2(true, predicted), [0.5, 0.5])


def test_signed_quadrature_weights_are_not_silently_absolutized() -> None:
    true = np.asarray([1.0, 1.0])
    predicted = np.asarray([1.0, 0.0])
    weights = np.asarray([2.0, -1.0])
    with pytest.raises(ValueError, match="Do not replace weights with absolute values"):
        signed_quadrature_relative_l2(true, predicted, weights)
    numerator, denominator = raw_signed_quadrature_terms(true, predicted, weights)
    np.testing.assert_allclose(numerator, [-1.0])
    np.testing.assert_allclose(denominator, [1.0])


def test_signed_quadrature_exposes_raw_terms() -> None:
    true = np.asarray([[1.0, 2.0], [1.0, 1.0]])
    predicted = np.asarray([[0.5, 1.5], [0.75, 0.75]])
    weights = np.asarray([2.0, -0.5])
    numerator, denominator = signed_quadrature_terms(true, predicted, weights)
    np.testing.assert_allclose(numerator, [0.46875, 0.46875])
    np.testing.assert_allclose(denominator, [1.5, 7.5])


def test_replicated_rqmc_uses_positive_equal_means_and_paired_uncertainty() -> None:
    true = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
            [[1.5, 2.5], [3.5, 4.5]],
        ]
    )
    predicted = 0.5 * true
    result = replicated_rqmc_relative_l2(true, predicted)
    np.testing.assert_allclose(result.relative_l2, [0.5, 0.5])
    np.testing.assert_allclose(result.standard_error, [0.0, 0.0], atol=1e-15)
    assert result.squared_error_by_replication.shape == (3, 2)
    assert result.squared_truth_by_replication.shape == (3, 2)
    np.testing.assert_allclose(result.confidence_lower, [0.5, 0.5])
    np.testing.assert_allclose(result.confidence_upper, [0.5, 0.5])


def test_rqmc_requires_independent_replications() -> None:
    with pytest.raises(ValueError, match="at least two"):
        replicated_rqmc_relative_l2(
            np.ones((1, 8, 1)), np.zeros((1, 8, 1))
        )
