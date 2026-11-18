"""Polynomial-index utilities needed by the Figure 1 CS surrogate."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json

import numpy as np


@lru_cache(maxsize=None)
def _factor_tuples(dimension: int, product_budget: int) -> tuple[tuple[int, ...], ...]:
    if dimension == 1:
        return tuple((value,) for value in range(1, product_budget + 1))

    rows: list[tuple[int, ...]] = []
    for first in range(1, product_budget + 1):
        remaining_budget = product_budget // first
        rows.extend(
            (first, *tail)
            for tail in _factor_tuples(dimension - 1, remaining_budget)
        )
    return tuple(rows)


def hyperbolic_cross_indices(dimension: int, order: int) -> np.ndarray:
    """Return ``nu`` satisfying ``prod(nu + 1) <= order + 1``."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if order < 0:
        raise ValueError("order must be nonnegative")
    factors = _factor_tuples(dimension, order + 1)
    return np.asarray(factors, dtype=np.int64) - 1


def hyperbolic_cross_size(dimension: int, order: int) -> int:
    return len(_factor_tuples(dimension, order + 1))


def polynomial_basis_fingerprint(dimension: int, index_set: str, level: int) -> str:
    """Fingerprint ordered multi-indices using a platform-independent encoding."""
    indices = polynomial_indices(dimension, index_set, level)
    canonical = np.ascontiguousarray(indices, dtype=np.dtype("<i8"))
    metadata = {
        "dimension": int(dimension),
        "dtype": "little-endian-int64",
        "index_set": str(index_set),
        "level": int(level),
        "shape": list(canonical.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def polynomial_indices(dimension: int, index_set: str, level: int) -> np.ndarray:
    """Build the production hyperbolic-cross index family."""
    if index_set == "hyperbolic-cross":
        return hyperbolic_cross_indices(dimension, level)
    raise ValueError(f"Unsupported polynomial index set: {index_set}")


def polynomial_index_size(dimension: int, index_set: str, level: int) -> int:
    if index_set == "hyperbolic-cross":
        return hyperbolic_cross_size(dimension, level)
    raise ValueError(f"Unsupported polynomial index set: {index_set}")


def intrinsic_legendre_weights(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 2 or np.any(indices < 0):
        raise ValueError("indices must be a nonnegative two-dimensional array")
    return np.prod(np.sqrt(2.0 * indices + 1.0), axis=1)


def legendre_design_matrix(
    canonical_points: np.ndarray,
    indices: np.ndarray,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Evaluate the probability-orthonormal tensor Legendre basis.

    The one-dimensional factors are ``sqrt(2*n + 1) * P_n(x)``, matching the
    archived implementation and orthonormality under the uniform probability
    measure on ``[-1, 1]``.
    """
    points = np.asarray(canonical_points, dtype=dtype)
    multi_indices = np.asarray(indices, dtype=np.int64)
    if points.ndim != 2 or multi_indices.ndim != 2:
        raise ValueError("points and indices must both be two-dimensional")
    if points.shape[1] != multi_indices.shape[1]:
        raise ValueError("point dimension does not match the polynomial indices")
    if np.any(points < -1.0) or np.any(points > 1.0):
        raise ValueError("canonical points must lie in [-1, 1]")
    if np.any(multi_indices < 0):
        raise ValueError("polynomial indices must be nonnegative")

    row_count = points.shape[0]
    basis_count = multi_indices.shape[0]
    design = np.ones((row_count, basis_count), dtype=dtype)
    for dimension in range(points.shape[1]):
        degrees = multi_indices[:, dimension]
        maximum = int(degrees.max(initial=0))
        values = np.empty((row_count, maximum + 1), dtype=dtype)
        values[:, 0] = 1.0
        if maximum >= 1:
            values[:, 1] = points[:, dimension]
        for degree in range(2, maximum + 1):
            values[:, degree] = (
                (2 * degree - 1) * points[:, dimension] * values[:, degree - 1]
                - (degree - 1) * values[:, degree - 2]
            ) / degree
        values *= np.sqrt(2.0 * np.arange(maximum + 1) + 1.0)
        design *= values[:, degrees]
    return design
