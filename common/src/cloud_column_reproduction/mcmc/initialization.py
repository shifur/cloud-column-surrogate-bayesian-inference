"""Deterministic initialization for genuinely independent emcee ensembles."""

from __future__ import annotations

import hashlib

import numpy as np

from cloud_column_reproduction.domain import ParameterBox


def initialize_walkers(
    parameter_box: ParameterBox,
    truth: np.ndarray,
    scales: np.ndarray,
    *,
    walkers: int,
    seed: int,
    maximum_redraw_rounds: int,
) -> np.ndarray:
    truth = parameter_box.require_physical(truth)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (parameter_box.dimension,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Initialization scales must be positive and match the parameter dimension")
    if walkers < 2 or maximum_redraw_rounds < 1:
        raise ValueError("Initialization requires at least two walkers and a positive redraw guard")
    rng = np.random.default_rng(seed)
    positions = truth[None, :] + rng.normal(size=(walkers, parameter_box.dimension)) * scales
    lower = np.asarray(parameter_box.lower)
    upper = np.asarray(parameter_box.upper)
    for column in range(parameter_box.dimension):
        bad = (positions[:, column] < lower[column]) | (positions[:, column] > upper[column])
        rounds = 0
        while np.any(bad):
            rounds += 1
            if rounds > maximum_redraw_rounds:
                raise RuntimeError(
                    f"Initialization redraw guard exceeded for parameter {parameter_box.names[column]}"
                )
            positions[bad, column] = truth[column] + rng.normal(
                size=int(np.sum(bad))
            ) * scales[column]
            bad = (positions[:, column] < lower[column]) | (positions[:, column] > upper[column])
    parameter_box.require_physical(positions)
    return positions


def initialize_walkers_latin_hypercube(
    parameter_box: ParameterBox,
    *,
    walkers: int,
    seed: int,
) -> np.ndarray:
    """Draw a dispersed Latin-hypercube ensemble over the complete prior box.

    Every parameter is stratified into ``walkers`` equal-probability intervals,
    with an independently shuffled stratum assignment.  Distinct calls are
    independent only when they use distinct seeds.  No truth-centered location
    enters this initialization, so a collection of ensembles can expose
    initialization-dependent mode discovery instead of sharing one local bias.
    """
    if walkers < 2:
        raise ValueError("Initialization requires at least two walkers")
    rng = np.random.default_rng(seed)
    dimension = parameter_box.dimension
    unit = np.empty((walkers, dimension), dtype=np.float64)
    for column in range(dimension):
        strata = rng.permutation(walkers)
        unit[:, column] = (strata + rng.random(walkers)) / float(walkers)
    lower = np.asarray(parameter_box.lower, dtype=np.float64)
    upper = np.asarray(parameter_box.upper, dtype=np.float64)
    positions = lower + unit * (upper - lower)
    parameter_box.require_physical(positions)
    if np.any(positions == lower) or np.any(positions == upper):
        raise RuntimeError("Latin-hypercube initialization unexpectedly hit a boundary")
    return positions


def initialize_independent_ensemble(
    parameter_box: ParameterBox,
    truth: np.ndarray,
    scales: np.ndarray,
    *,
    walkers: int,
    seed: int,
    maximum_redraw_rounds: int,
    law: str,
) -> np.ndarray:
    """Dispatch one recorded initialization law without silently substituting.

    ``latin-hypercube-prior`` is the production candidate because it probes the
    full prior and separated posterior modes.  The historical truth-centered
    law remains available only for controlled comparisons and canaries.
    """
    if law == "latin-hypercube-prior":
        return initialize_walkers_latin_hypercube(
            parameter_box, walkers=walkers, seed=seed
        )
    if law == "componentwise-truncated-gaussian-around-synthetic-truth":
        return initialize_walkers(
            parameter_box,
            truth,
            scales,
            walkers=walkers,
            seed=seed,
            maximum_redraw_rounds=maximum_redraw_rounds,
        )
    raise ValueError(f"Unsupported MCMC initialization law: {law!r}")


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
