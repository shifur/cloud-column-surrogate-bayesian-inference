"""Explicit Figure 1 error measures; absolute quadrature weights are forbidden."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class RQMCRatioEstimate:
    """Replicate-based estimate and uncertainty for relative L2 error."""

    relative_l2: np.ndarray
    standard_error: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    squared_error_by_replication: np.ndarray
    squared_truth_by_replication: np.ndarray
    confidence_level: float


def _rqmc_prediction_arrays(
    true: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(true, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if true_values.shape != predicted_values.shape or true_values.ndim not in (2, 3):
        raise ValueError(
            "RQMC truth and predictions must have matching (replication, point) "
            "or (replication, point, output) shapes"
        )
    if true_values.shape[0] < 2 or true_values.shape[1] < 1:
        raise ValueError("RQMC requires at least two nonempty replications")
    if not np.all(np.isfinite(true_values)) or not np.all(
        np.isfinite(predicted_values)
    ):
        raise ValueError("RQMC relative L2 inputs must be finite")
    return true_values, predicted_values


def replicated_rqmc_relative_l2(
    true: np.ndarray,
    predicted: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> RQMCRatioEstimate:
    """Estimate relative L2 from independent randomized-QMC replications.

    Each scramble has equal positive point weights.  Numerator and denominator
    integrals are averaged across scrambles before their ratio is formed.  The
    reported uncertainty is a paired delete-one-replication jackknife with a
    Student-t multiplier; pairing retains their covariance.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    true_values, predicted_values = _rqmc_prediction_arrays(true, predicted)
    point_axis = 1
    squared_error = np.mean((true_values - predicted_values) ** 2, axis=point_axis)
    squared_truth = np.mean(true_values**2, axis=point_axis)
    if np.any(squared_truth <= 0.0):
        raise ValueError("Every RQMC replicate truth norm must be positive")

    replication_count = true_values.shape[0]
    mean_error = np.mean(squared_error, axis=0)
    mean_truth = np.mean(squared_truth, axis=0)
    estimate = np.sqrt(mean_error / mean_truth)

    leave_error = (
        replication_count * mean_error - squared_error
    ) / (replication_count - 1)
    leave_truth = (
        replication_count * mean_truth - squared_truth
    ) / (replication_count - 1)
    if np.any(leave_error < 0.0) or np.any(leave_truth <= 0.0):
        raise ValueError("RQMC delete-one terms do not define a relative L2 norm")
    leave_estimate = np.sqrt(leave_error / leave_truth)
    pseudovalues = (
        replication_count * estimate
        - (replication_count - 1) * leave_estimate
    )
    standard_error = np.std(pseudovalues, axis=0, ddof=1) / np.sqrt(
        replication_count
    )
    critical = float(
        student_t.ppf(
            0.5 + confidence_level / 2.0,
            df=replication_count - 1,
        )
    )
    lower = np.maximum(0.0, estimate - critical * standard_error)
    upper = estimate + critical * standard_error
    return RQMCRatioEstimate(
        relative_l2=np.atleast_1d(estimate),
        standard_error=np.atleast_1d(standard_error),
        confidence_lower=np.atleast_1d(lower),
        confidence_upper=np.atleast_1d(upper),
        squared_error_by_replication=np.asarray(squared_error),
        squared_truth_by_replication=np.asarray(squared_truth),
        confidence_level=confidence_level,
    )


def _prediction_arrays(true: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(true, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if true_values.shape != predicted_values.shape or true_values.ndim not in (1, 2):
        raise ValueError("True and predicted arrays must have the same one- or two-dimensional shape")
    if not np.all(np.isfinite(true_values)) or not np.all(np.isfinite(predicted_values)):
        raise ValueError("Relative L2 inputs must be finite")
    return true_values, predicted_values


def equal_weight_relative_l2(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Secondary Monte Carlo estimate of prior-box relative L2."""
    true_values, predicted_values = _prediction_arrays(true, predicted)
    numerator = np.sum((true_values - predicted_values) ** 2, axis=0)
    denominator = np.sum(true_values**2, axis=0)
    if np.any(denominator <= 0.0):
        raise ValueError("Relative L2 denominator must be positive")
    return np.sqrt(numerator / denominator)


def signed_quadrature_relative_l2(
    true: np.ndarray, predicted: np.ndarray, signed_weights: np.ndarray
) -> np.ndarray:
    """Primary Figure 1 error using Tasmanian quadrature weights unchanged."""
    numerator, denominator = signed_quadrature_terms(
        true, predicted, signed_weights
    )
    return np.sqrt(numerator / denominator)


def signed_quadrature_terms(
    true: np.ndarray, predicted: np.ndarray, signed_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed squared-error and squared-truth quadrature estimates.

    The raw terms are exposed for manifests and convergence audits.  A negative
    numerator or nonpositive denominator is an explicit quadrature failure; the
    implementation never takes absolute values or silently changes the measure.
    """
    numerator, denominator = raw_signed_quadrature_terms(
        true, predicted, signed_weights
    )
    if np.any(numerator < 0.0) or np.any(denominator <= 0.0):
        raise ValueError(
            "Signed sparse-grid quadrature produced a negative squared-error "
            f"estimate or nonpositive truth norm: numerator={numerator.tolist()}, "
            f"denominator={denominator.tolist()}. Do not replace weights with "
            "absolute values."
        )
    return numerator, denominator


def raw_signed_quadrature_terms(
    true: np.ndarray, predicted: np.ndarray, signed_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed terms without asserting that they define a norm.

    This audit interface preserves negative estimates so sparse-grid level
    inadequacy can be diagnosed and recorded. Production metrics must use
    :func:`signed_quadrature_terms`, which rejects invalid terms.
    """
    true_values, predicted_values = _prediction_arrays(true, predicted)
    weights = np.asarray(signed_weights, dtype=float)
    if weights.shape != (true_values.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("Signed quadrature weights must match the sample dimension")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("Signed quadrature weights must have positive total mass")
    squared_error = (true_values - predicted_values) ** 2
    squared_true = true_values**2
    numerator = np.atleast_1d(np.tensordot(weights, squared_error, axes=(0, 0)))
    denominator = np.atleast_1d(np.tensordot(weights, squared_true, axes=(0, 0)))
    return numerator, denominator
