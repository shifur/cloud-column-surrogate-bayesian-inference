"""Canary-only emcee orchestration preserving independent-ensemble identity.

Production sampling uses :mod:`mcmc.resumable`; this in-memory helper must not
be used for manuscript chains because it has no atomic continuation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .resumable import build_emcee_move


@dataclass(frozen=True)
class EnsembleResult:
    positions: np.ndarray
    log_posterior: np.ndarray
    log_prior: np.ndarray
    log_likelihood: np.ndarray
    status: np.ndarray
    accepted: np.ndarray
    initial_positions: np.ndarray
    post_burn_positions: np.ndarray
    proposal_seed: int
    random_state_before: str
    random_state_after: str


def run_emcee_ensemble(
    log_probability: Any,
    initial_positions: np.ndarray,
    *,
    burn_in_steps: int,
    production_steps: int,
    proposal_seed: int,
    vectorize: bool,
    move_name: str = "emcee.moves.WalkMove",
    move_parameters: dict[str, Any] | None = None,
) -> EnsembleResult:
    """Run one configured ensemble for deterministic software canaries only."""
    try:
        import emcee
    except ImportError as error:  # keep base package importable without MCMC extras
        raise RuntimeError("The mcmc optional dependencies are required") from error
    initial = np.asarray(initial_positions, dtype=np.float64)
    if initial.ndim != 2:
        raise ValueError("Initial positions must have shape (walkers, parameters)")
    walkers, dimensions = initial.shape
    if walkers != 32:
        raise ValueError("Controlled emcee runs require exactly 32 walkers")
    if burn_in_steps < 0 or production_steps < 1:
        raise ValueError("Sampler lengths must be nonnegative burn-in and positive production")

    sampler = emcee.EnsembleSampler(
        walkers,
        dimensions,
        log_probability,
        moves=build_emcee_move(move_name, move_parameters),
        vectorize=vectorize,
        blobs_dtype=[("log_prior", "f8"), ("log_likelihood", "f8"), ("status", "i1")],
    )
    random_state = np.random.RandomState(proposal_seed).get_state()
    sampler.random_state = random_state
    before = repr(sampler.random_state)
    state = sampler.run_mcmc(initial, burn_in_steps, progress=False) if burn_in_steps else initial
    post_burn = np.asarray(state.coords if hasattr(state, "coords") else state).copy()
    sampler.reset()
    sampler.run_mcmc(state, production_steps, progress=False)

    positions = sampler.get_chain()
    log_posterior = sampler.get_log_prob()
    blobs = sampler.get_blobs()
    if blobs is None or blobs.dtype.names is None:
        raise RuntimeError("Controlled log probability must return structured audit blobs")
    accepted = np.zeros((production_steps, walkers), dtype=np.int8)
    accepted[0] = np.any(positions[0] != post_burn, axis=1)
    accepted[1:] = np.any(positions[1:] != positions[:-1], axis=2)
    return EnsembleResult(
        positions=positions,
        log_posterior=log_posterior,
        log_prior=np.asarray(blobs["log_prior"], dtype=np.float64),
        log_likelihood=np.asarray(blobs["log_likelihood"], dtype=np.float64),
        status=np.asarray(blobs["status"], dtype=np.int8),
        accepted=accepted,
        initial_positions=initial.copy(),
        post_burn_positions=post_burn,
        proposal_seed=proposal_seed,
        random_state_before=before,
        random_state_after=repr(sampler.random_state),
    )
