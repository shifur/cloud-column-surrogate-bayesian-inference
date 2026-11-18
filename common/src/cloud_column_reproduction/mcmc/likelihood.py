"""Single Gaussian likelihood and posterior implementation used by every evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluators.base import ForwardEvaluator
from .problem import InverseProblem


@dataclass(frozen=True)
class PosteriorEvaluation:
    log_prior: np.ndarray
    log_likelihood: np.ndarray
    log_posterior: np.ndarray
    predictions: np.ndarray
    status: np.ndarray
    messages: tuple[str, ...]


class GaussianLikelihood:
    """Independent Gaussian errors whose configured entries are standard deviations."""

    def __init__(self, problem: InverseProblem) -> None:
        self.problem = problem
        self._selected = problem.likelihood_case.active_indices
        sigma = np.asarray(problem.sigma, dtype=np.float64)[self._selected]
        self._variance = sigma * sigma  # the only squaring in the MCMC pipeline
        self._observation = np.asarray(problem.observation, dtype=np.float64)[self._selected]
        self._normalizer = -0.5 * (
            self._selected.size * np.log(2.0 * np.pi) + np.sum(np.log(self._variance))
        )

    @property
    def selected_variance(self) -> np.ndarray:
        return self._variance.copy()

    def log_likelihood(self, predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("Likelihood predictions must have shape (n, 6)")
        selected = values[:, self._selected]
        if not np.all(np.isfinite(selected)):
            raise ValueError("Active evaluator predictions must be finite")
        residual = selected - self._observation[None, :]
        return self._normalizer - 0.5 * np.sum(
            residual * residual / self._variance[None, :], axis=1
        )


def evaluate_posterior(
    physical_parameters: np.ndarray,
    evaluator: ForwardEvaluator,
    likelihood: GaussianLikelihood,
) -> PosteriorEvaluation:
    values = np.asarray(physical_parameters, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != likelihood.problem.parameter_box.dimension:
        raise ValueError("Posterior input must have shape (n, 11)")
    count = values.shape[0]
    log_prior = np.full(count, -np.inf, dtype=np.float64)
    log_likelihood = np.full(count, -np.inf, dtype=np.float64)
    predictions = np.full((count, 6), np.nan, dtype=np.float64)
    status = np.full(count, 2, dtype=np.int8)  # 2 = rejected by prior
    messages = ["outside authoritative parameter box"] * count

    finite = np.all(np.isfinite(values), axis=1)
    in_bounds = np.zeros(count, dtype=bool)
    if np.any(finite):
        in_bounds[finite] = likelihood.problem.parameter_box.contains(values[finite])
    if np.any(in_bounds):
        batch = evaluator.evaluate(values[in_bounds])
        target = np.flatnonzero(in_bounds)
        predictions[target] = batch.predictions
        status[target] = batch.status
        for index, message in zip(target, batch.messages, strict=True):
            messages[index] = message
        successful = batch.status == 0
        if np.any(successful):
            successful_target = target[successful]
            log_prior[successful_target] = 0.0
            log_likelihood[successful_target] = likelihood.log_likelihood(
                batch.predictions[successful]
            )

    return PosteriorEvaluation(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_posterior=log_prior + log_likelihood,
        predictions=predictions,
        status=status,
        messages=tuple(messages),
    )
