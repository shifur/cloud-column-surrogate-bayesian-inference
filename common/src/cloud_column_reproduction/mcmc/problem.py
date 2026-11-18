"""Immutable representation of the common controlled inverse problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloud_column_reproduction.domain import ParameterBox


@dataclass(frozen=True)
class LikelihoodCase:
    name: str
    mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.mask) != 6 or any(value not in (0, 1) for value in self.mask):
            raise ValueError("Likelihood mask must contain six binary entries")
        if not any(self.mask):
            raise ValueError("Likelihood mask must select at least one output")

    @property
    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.mask, dtype=bool))


@dataclass(frozen=True)
class InverseProblem:
    parameter_box: ParameterBox
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    output_units: tuple[str, ...]
    truth: tuple[float, ...]
    observation: tuple[float, ...]
    sigma: tuple[float, ...]
    observation_time_minutes: int
    likelihood_case: LikelihoodCase

    def __post_init__(self) -> None:
        if self.parameter_names != self.parameter_box.names:
            raise ValueError("Problem parameter names must match the parameter box")
        if len(self.output_names) != 6 or len(self.output_units) != 6:
            raise ValueError("Problem must define six ordered outputs and units")
        if self.observation_time_minutes != 120:
            raise ValueError("Controlled observation time must be 120 minutes")
        self.parameter_box.require_physical(self.truth)
        observation = np.asarray(self.observation, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if observation.shape != (6,) or not np.all(np.isfinite(observation)):
            raise ValueError("Observation must contain six finite values")
        if sigma.shape != (6,) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("Sigma must contain six positive finite standard deviations")
