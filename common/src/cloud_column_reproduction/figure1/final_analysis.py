"""Statistical summaries for the completed Figure 1 reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class SampleSummary:
    count: int
    mean: float
    sample_standard_deviation: float
    median: float
    first_quartile: float
    third_quartile: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class PairedLogRatio:
    count: int
    geometric_mean_ratio: float
    confidence_lower: float
    confidence_upper: float
    arithmetic_mean_difference: float
    sample_standard_deviation_difference: float
    numerator_better_count: int
    denominator_better_count: int
    tied_count: int


def sample_summary(values: np.ndarray) -> SampleSummary:
    """Return a reproducible descriptive summary using linear quartiles."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("A sample summary requires at least two finite values")
    return SampleSummary(
        count=len(array),
        mean=float(np.mean(array)),
        sample_standard_deviation=float(np.std(array, ddof=1)),
        median=float(np.median(array)),
        first_quartile=float(np.quantile(array, 0.25, method="linear")),
        third_quartile=float(np.quantile(array, 0.75, method="linear")),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )


def paired_log_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> PairedLogRatio:
    """Summarize a paired positive ratio with a Student-t interval in log space.

    A ratio below one favors the numerator. The interval is pointwise and is
    not adjusted for inspecting multiple training sizes or outputs.
    """
    top = np.asarray(numerator, dtype=np.float64).reshape(-1)
    bottom = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if top.shape != bottom.shape or len(top) < 2:
        raise ValueError("Paired ratio inputs must have equal length of at least two")
    if (
        not np.all(np.isfinite(top))
        or not np.all(np.isfinite(bottom))
        or np.any(top <= 0.0)
        or np.any(bottom <= 0.0)
    ):
        raise ValueError("Paired ratio inputs must be finite and strictly positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must lie strictly between zero and one")

    logs = np.log(top / bottom)
    mean_log = float(np.mean(logs))
    standard_error = float(np.std(logs, ddof=1) / np.sqrt(len(logs)))
    critical = float(t.ppf(0.5 + confidence_level / 2.0, df=len(logs) - 1))
    differences = top - bottom
    return PairedLogRatio(
        count=len(logs),
        geometric_mean_ratio=float(np.exp(mean_log)),
        confidence_lower=float(np.exp(mean_log - critical * standard_error)),
        confidence_upper=float(np.exp(mean_log + critical * standard_error)),
        arithmetic_mean_difference=float(np.mean(differences)),
        sample_standard_deviation_difference=float(np.std(differences, ddof=1)),
        numerator_better_count=int(np.count_nonzero(top < bottom)),
        denominator_better_count=int(np.count_nonzero(top > bottom)),
        tied_count=int(np.count_nonzero(top == bottom)),
    )
