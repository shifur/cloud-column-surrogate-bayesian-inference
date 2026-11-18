import numpy as np
from pathlib import Path

from cloud_column_reproduction.mcmc.diagnostics import (
    candidate_cluster_aware_rank_diagnostics,
    independent_ensemble_diagnostics_from_blocks,
)


def _iid(seed: int, transitions: int = 2000, walkers: int = 16) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(2, transitions, walkers, 1))


def _ar1(seed: int, rho: float, transitions: int = 2000, walkers: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed); noise = rng.normal(size=(2, transitions, walkers, 1))
    values = np.empty_like(noise); values[:, 0] = noise[:, 0]
    scale = np.sqrt(1.0 - rho * rho)
    for draw in range(1, transitions): values[:, draw] = rho * values[:, draw - 1] + scale * noise[:, draw]
    return values


def test_candidate_detects_location_mismatch_and_stuck_modes():
    values = _iid(101); values[0] -= 1.5; values[1] += 1.5
    result = candidate_cluster_aware_rank_diagnostics(values)
    assert result.rhat_rank[0] > 1.05 and result.rhat_final[0] > 1.05


def test_candidate_detects_scale_mismatch_with_cluster_folded_rhat():
    values = _iid(102); values[1] *= 3.0
    result = candidate_cluster_aware_rank_diagnostics(values)
    assert result.rhat_folded[0] > 1.05 and result.rhat_final[0] > 1.05


def test_candidate_detects_slow_mixing_with_bulk_ess():
    result = candidate_cluster_aware_rank_diagnostics(_ar1(103, 0.995))
    assert result.bulk_ess[0] < 400


def test_candidate_detects_tail_stickiness():
    rng = np.random.default_rng(104)
    values = rng.normal(size=(2, 4000, 32, 1))
    # A sticky 5%-occupancy upper-tail state affects only a walker subset, so
    # bulk location still mixes while the posterior-tail mass does not.
    for ensemble in range(2):
        state = False
        for transition in range(4000):
            state = rng.random() < 0.98 if state else rng.random() < (0.05 * 0.02 / 0.95)
            if state: values[ensemble, transition, :8, 0] = 8.0
    result = candidate_cluster_aware_rank_diagnostics(values)
    assert result.rhat_final[0] < 1.05 and result.bulk_ess[0] >= 400
    assert result.tail_ess[0] < 400


def test_candidate_detects_cross_walker_dependence_without_counting_walkers_as_chains():
    rng = np.random.default_rng(105)
    common = rng.normal(size=(2, 2000, 1, 1))
    values = np.repeat(common, 16, axis=2) + 1e-5 * rng.normal(size=(2, 2000, 16, 1))
    result = candidate_cluster_aware_rank_diagnostics(values)
    assert result.mean_absolute_cross_walker_correlation[0] > 0.99
    assert result.ensemble_count == 2 and result.transition_count == 2000
    assert result.bulk_ess[0] <= 2 * result.transition_count * 1.01


def test_well_mixed_control_passes_candidate_rhat_and_ess_thresholds():
    result = candidate_cluster_aware_rank_diagnostics(_iid(106))
    assert result.rhat_final[0] < 1.05
    assert result.bulk_ess[0] >= 400 and result.tail_ess[0] >= 400


def test_blockwise_diagnostics_match_in_memory_construction(tmp_path: Path):
    values = np.random.default_rng(107).normal(size=(2, 24, 4, 2))
    paths = []
    for ensemble in range(2):
        ensemble_paths = []
        for start in range(0, 24, 6):
            path = tmp_path / f"e{ensemble}_{start}.npz"
            np.savez_compressed(path, position=values[ensemble, start : start + 6])
            ensemble_paths.append(path)
        paths.append(tuple(ensemble_paths))
    expected = candidate_cluster_aware_rank_diagnostics(values)
    actual = independent_ensemble_diagnostics_from_blocks(tuple(paths), prefix=24)
    for field in (
        "rhat_rank",
        "rhat_folded",
        "bulk_ess",
        "tail_ess",
        "maximum_quantile_bin_occupancy_difference",
    ):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field))
