"""Coordinate transformations for bounded-prior MCMC."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.special import expit


def physical_to_logit(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    unit = (np.asarray(values, dtype=np.float64) - lower) / (upper - lower)
    if np.any(unit <= 0.0) or np.any(unit >= 1.0):
        raise ValueError("Box-logit coordinates require points strictly inside the prior box")
    return np.log(unit) - np.log1p(-unit)


def logit_to_physical(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    unit = expit(np.asarray(values, dtype=np.float64))
    return lower + unit * (upper - lower)


def log_abs_det_jacobian(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    terms = np.log(upper - lower) - np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)
    return np.sum(terms, axis=-1)


def transformed_log_probability(
    function: Callable[[np.ndarray], Any], lower: np.ndarray, upper: np.ndarray
) -> Callable[[np.ndarray], Any]:
    """Return the exact density in box-logit coordinates, including Jacobian."""
    def evaluate(values: np.ndarray) -> Any:
        z = np.asarray(values, dtype=np.float64)
        physical = logit_to_physical(z, lower, upper)
        rows = function(physical)
        row_list = rows if z.ndim == 2 else [rows]
        jacobian = np.atleast_1d(log_abs_det_jacobian(z, lower, upper))
        adjusted = [
            (
                float(row[0]) + float(log_j),
                float(row[1]) + float(log_j),
                float(row[2]),
                int(row[3]),
                *(
                    ()
                    if len(row) == 4
                    else (np.asarray(row[4], dtype=np.float64),)
                ),
            )
            for row, log_j in zip(row_list, jacobian, strict=True)
        ]
        return adjusted if z.ndim == 2 else adjusted[0]
    return evaluate
