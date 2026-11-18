"""Cluster-aware diagnostics for independently initialized emcee ensembles.

The independent sampling unit is an entire emcee ensemble.  The interacting
walkers within one ensemble are preserved as a dependent cluster and are never
promoted to the chain dimension used by R-hat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloud_column_reproduction.provenance import sha256_file, write_manifest


@dataclass(frozen=True)
class BasicDiagnostics:
    acceptance_fraction: np.ndarray
    boundary_fraction: np.ndarray
    between_ensemble_mean_range: np.ndarray
    scientific_convergence_claim: bool
    rhat_ess_status: str


@dataclass(frozen=True)
class ClusterAwareRankDiagnostics:
    """Marginal diagnostics whose only chain axis is independent ensemble runs.

    For each walker label and parameter, the ArviZ chain axis contains the two
    independently initialized ensembles and the draw axis contains transitions.
    Walkers are never put on the chain axis. Reported R-hat is the maximum and
    reported ESS the minimum across the 32 matched-walker calculations. This is
    conservative and directly concerns marginal walker trajectories, unlike an
    R-hat or ESS of the ensemble's walker mean.
    """

    rhat_rank: np.ndarray
    rhat_folded: np.ndarray
    rhat_final: np.ndarray
    bulk_ess: np.ndarray
    tail_ess: np.ndarray
    lower_tail_ess: np.ndarray
    upper_tail_ess: np.ndarray
    walker_rhat_rank: np.ndarray
    walker_rhat_folded: np.ndarray
    walker_bulk_ess: np.ndarray
    walker_lower_tail_ess: np.ndarray
    walker_upper_tail_ess: np.ndarray
    walker_tail_ess: np.ndarray
    mean_absolute_cross_walker_correlation: np.ndarray
    maximum_absolute_cross_walker_correlation: np.ndarray
    maximum_quantile_bin_occupancy_difference: np.ndarray
    ensemble_count: int
    transition_count: int
    walker_count: int
    effective_sample_unit: str


def basic_independent_ensemble_diagnostics(
    positions: np.ndarray,
    accepted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    boundary_fraction_of_range: float = 0.01,
) -> BasicDiagnostics:
    """Compute diagnostics that do not pretend interacting walkers are chains.

    ``positions`` has shape (ensemble, transition, walker, parameter). These
    inexpensive summaries are descriptive only. Formal diagnostics for every
    method require at least two independently initialized ensembles.
    """
    values = np.asarray(positions, dtype=np.float64)
    accepted = np.asarray(accepted)
    if values.ndim != 4 or values.shape[0] < 1:
        raise ValueError("Diagnostics require arrays shaped ensemble, draw, walker, parameter")
    if accepted.shape != values.shape[:3]:
        raise ValueError("Accepted flags must match ensemble, transition, walker dimensions")
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    width = upper - lower
    near_lower = values <= lower[None, None, None, :] + boundary_fraction_of_range * width
    near_upper = values >= upper[None, None, None, :] - boundary_fraction_of_range * width
    boundary = np.mean(near_lower | near_upper, axis=(1, 2))
    ensemble_means = np.mean(values, axis=(1, 2))
    return BasicDiagnostics(
        acceptance_fraction=np.mean(accepted, axis=(1, 2)),
        boundary_fraction=boundary,
        between_ensemble_mean_range=np.ptp(ensemble_means, axis=0),
        scientific_convergence_claim=False,
        rhat_ess_status="not-computed-by-basic-summary;use-independent-ensemble-rank-diagnostics",
    )


def require_approved_rank_diagnostic_contract(array_construction: str) -> None:
    if array_construction.startswith("UNRESOLVED"):
        raise RuntimeError(
            "DNN/PDR rank-normalized/folded R-hat and bulk/tail ESS are blocked until "
            "the independent-ensemble array construction is approved"
        )


def require_independent_ensemble_provenance(
    positions: np.ndarray,
    *,
    chain_axis_source: str,
    initialization_seeds: tuple[int, ...] | list[int],
) -> None:
    """Fail closed if interacting walkers have been disguised as chains."""
    values = np.asarray(positions)
    if values.ndim != 4:
        raise ValueError(
            "Formal diagnostics require shape "
            "(independent_ensemble, transition, walker, parameter); a 3-D "
            "walker-split array is forbidden"
        )
    if chain_axis_source != "independently-initialized-emcee-ensemble":
        raise ValueError(
            "Diagnostic chain axis must contain independently initialized "
            "emcee ensembles, never walkers or walker halves"
        )
    seeds = tuple(int(value) for value in initialization_seeds)
    if len(seeds) != values.shape[0] or len(set(seeds)) != len(seeds):
        raise ValueError(
            "Every independent ensemble must have one distinct initialization seed"
        )
    if values.shape[0] < 2:
        raise ValueError("Formal between-run diagnostics require at least two ensembles")


def independent_ensemble_rank_diagnostics(positions: np.ndarray) -> ClusterAwareRankDiagnostics:
    """Compute conservative marginal diagnostics from independent ensembles."""
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 2 or values.shape[1] < 8 or values.shape[2] < 2:
        raise ValueError(
            "Diagnostics require shape "
            "(independent_ensemble>=2, transition>=8, walker>=2, parameter)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Candidate diagnostics require finite positions")
    import arviz as az

    ensembles, transitions, walkers, parameters = values.shape
    walker_rank = np.empty((walkers, parameters))
    walker_folded = np.empty((walkers, parameters))
    walker_bulk = np.empty((walkers, parameters))
    walker_lower = np.empty((walkers, parameters))
    walker_upper = np.empty((walkers, parameters))
    walker_tail = np.empty((walkers, parameters))
    for parameter in range(parameters):
        for walker_index in range(walkers):
            trajectory = values[:, :, walker_index, parameter]
            walker_rank[walker_index, parameter] = float(
                az.rhat(trajectory, method="rank")
            )
            walker_folded[walker_index, parameter] = float(
                az.rhat(trajectory, method="folded")
            )
            walker_bulk[walker_index, parameter] = float(
                az.ess(trajectory, method="bulk")
            )
            walker_lower[walker_index, parameter] = float(
                az.ess(trajectory, method="quantile", prob=0.05)
            )
            walker_upper[walker_index, parameter] = float(
                az.ess(trajectory, method="quantile", prob=0.95)
            )
            walker_tail[walker_index, parameter] = min(
                walker_lower[walker_index, parameter],
                walker_upper[walker_index, parameter],
            )

    rhat_rank = np.max(walker_rank, axis=0)
    rhat_folded = np.max(walker_folded, axis=0)
    bulk = np.min(walker_bulk, axis=0)
    tail = np.min(walker_tail, axis=0)

    mean_correlation = np.empty(parameters); maximum_correlation = np.empty(parameters)
    mask = ~np.eye(walkers, dtype=bool)
    for parameter in range(parameters):
        correlations = []
        for ensemble in range(ensembles):
            series = values[ensemble, :, :, parameter]
            centered = series - np.mean(series, axis=0, keepdims=True)
            scale = np.sqrt(np.sum(centered * centered, axis=0))
            denominator = scale[:, None] * scale[None, :]
            numerator = centered.T @ centered
            matrix = np.full((walkers, walkers), np.nan, dtype=np.float64)
            np.divide(numerator, denominator, out=matrix, where=denominator > 0)
            correlations.extend(np.abs(matrix[mask]).tolist())
        finite = np.asarray(correlations)[np.isfinite(correlations)]
        mean_correlation[parameter] = np.mean(finite) if len(finite) else np.nan
        maximum_correlation[parameter] = np.max(finite) if len(finite) else np.nan

    # Compare posterior mass across independent runs.  Pooled decile edges make
    # this diagnostic invariant to parameter units while retaining separated
    # mode information that a location-only R-hat can miss.
    occupancy_difference = np.empty(parameters)
    for parameter in range(parameters):
        flattened = values[..., parameter].reshape(-1)
        edges = np.unique(np.quantile(flattened, np.linspace(0.0, 1.0, 11)))
        if len(edges) < 3:
            occupancy_difference[parameter] = np.inf
            continue
        occupancies = []
        for ensemble in range(ensembles):
            counts, _ = np.histogram(values[ensemble, ..., parameter], bins=edges)
            occupancies.append(counts / np.sum(counts))
        occupancy = np.asarray(occupancies)
        occupancy_difference[parameter] = float(
            np.max(np.ptp(occupancy, axis=0))
        )
    return ClusterAwareRankDiagnostics(
        rhat_rank=rhat_rank, rhat_folded=rhat_folded,
        rhat_final=np.maximum(rhat_rank, rhat_folded), bulk_ess=bulk,
        tail_ess=tail,
        lower_tail_ess=np.min(walker_lower, axis=0),
        upper_tail_ess=np.min(walker_upper, axis=0),
        walker_rhat_rank=walker_rank,
        walker_rhat_folded=walker_folded,
        walker_bulk_ess=walker_bulk,
        walker_lower_tail_ess=walker_lower,
        walker_upper_tail_ess=walker_upper,
        walker_tail_ess=walker_tail,
        mean_absolute_cross_walker_correlation=mean_correlation,
        maximum_absolute_cross_walker_correlation=maximum_correlation,
        maximum_quantile_bin_occupancy_difference=occupancy_difference,
        ensemble_count=ensembles, transition_count=transitions, walker_count=walkers,
        effective_sample_unit=(
            "minimum-matched-walker-trajectory-ess-across-two-independent-ensembles"
        ),
    )


def candidate_cluster_aware_rank_diagnostics(positions: np.ndarray) -> ClusterAwareRankDiagnostics:
    """Backward-compatible name for the now-approved array construction."""
    return independent_ensemble_rank_diagnostics(positions)


def independent_ensemble_diagnostics_from_blocks(
    block_paths_by_ensemble: tuple[tuple[Path, ...], ...],
    *,
    prefix: int,
    sampler_coordinate_system: str = "physical",
    parameter_lower: np.ndarray | None = None,
    parameter_upper: np.ndarray | None = None,
) -> ClusterAwareRankDiagnostics:
    """Compute exact prefix diagnostics with bounded peak position memory.

    Only one parameter's ``ensemble x transition x walker`` values are held at
    once.  The result is numerically the same construction used by
    :func:`independent_ensemble_rank_diagnostics`, apart from roundoff due to
    the different concatenation order.
    """
    if len(block_paths_by_ensemble) < 2 or prefix < 8:
        raise ValueError("Block diagnostics require at least two ensembles and prefix >= 8")
    first_path = block_paths_by_ensemble[0][0]
    with np.load(first_path, allow_pickle=False) as first:
        first_position = np.asarray(first["position"])
        if first_position.ndim != 3:
            raise ValueError("Production blocks require position=(transition,walker,parameter)")
        walkers, parameters = first_position.shape[1:]
    if sampler_coordinate_system not in {"physical", "box-logit"}:
        raise ValueError("Unknown sampler coordinate system")
    if sampler_coordinate_system == "box-logit":
        lower = np.asarray(parameter_lower, dtype=np.float64)
        upper = np.asarray(parameter_upper, dtype=np.float64)
        if lower.shape != (parameters,) or upper.shape != (parameters,):
            raise ValueError("Box-logit diagnostics require complete parameter bounds")
        from scipy.special import expit
    import arviz as az

    count = len(block_paths_by_ensemble)
    walker_rank = np.empty((walkers, parameters))
    walker_folded = np.empty((walkers, parameters))
    walker_bulk = np.empty((walkers, parameters))
    walker_lower = np.empty((walkers, parameters))
    walker_upper = np.empty((walkers, parameters))
    walker_tail = np.empty((walkers, parameters))
    mean_correlation = np.empty(parameters); maximum_correlation = np.empty(parameters)
    occupancy_difference = np.empty(parameters)
    mask = ~np.eye(walkers, dtype=bool)

    for parameter in range(parameters):
        values = np.empty((count, prefix, walkers), dtype=np.float64)
        for ensemble, paths in enumerate(block_paths_by_ensemble):
            offset = 0
            for path in paths:
                if offset >= prefix:
                    break
                with np.load(path, allow_pickle=False) as archive:
                    block = np.asarray(archive["position"], dtype=np.float64)
                take = min(len(block), prefix - offset)
                values[ensemble, offset : offset + take] = block[:take, :, parameter]
                offset += take
            if offset != prefix:
                raise ValueError(
                    f"Ensemble {ensemble} has {offset} committed production transitions; "
                    f"diagnostic prefix {prefix} is unavailable"
                )
        if sampler_coordinate_system == "box-logit":
            values = lower[parameter] + expit(values) * (
                upper[parameter] - lower[parameter]
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Formal diagnostics require finite positions")
        for walker_index in range(walkers):
            trajectory = values[:, :, walker_index]
            walker_rank[walker_index, parameter] = float(
                az.rhat(trajectory, method="rank")
            )
            walker_folded[walker_index, parameter] = float(
                az.rhat(trajectory, method="folded")
            )
            walker_bulk[walker_index, parameter] = float(
                az.ess(trajectory, method="bulk")
            )
            walker_lower[walker_index, parameter] = float(
                az.ess(trajectory, method="quantile", prob=0.05)
            )
            walker_upper[walker_index, parameter] = float(
                az.ess(trajectory, method="quantile", prob=0.95)
            )
            walker_tail[walker_index, parameter] = min(
                walker_lower[walker_index, parameter],
                walker_upper[walker_index, parameter],
            )
        correlations = []
        for ensemble in range(count):
            series = values[ensemble]
            centered = series - np.mean(series, axis=0, keepdims=True)
            scale = np.sqrt(np.sum(centered * centered, axis=0))
            denominator = scale[:, None] * scale[None, :]
            matrix = np.full((walkers, walkers), np.nan, dtype=np.float64)
            np.divide(centered.T @ centered, denominator, out=matrix, where=denominator > 0)
            correlations.extend(np.abs(matrix[mask]).tolist())
        finite = np.asarray(correlations)[np.isfinite(correlations)]
        mean_correlation[parameter] = np.mean(finite) if len(finite) else np.nan
        maximum_correlation[parameter] = np.max(finite) if len(finite) else np.nan
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)))
        if len(edges) < 3:
            occupancy_difference[parameter] = np.inf
        else:
            occupancy = []
            for ensemble in range(count):
                counts, _ = np.histogram(values[ensemble], bins=edges)
                occupancy.append(counts / np.sum(counts))
            occupancy_difference[parameter] = float(
                np.max(np.ptp(np.asarray(occupancy), axis=0))
            )

    return ClusterAwareRankDiagnostics(
        rhat_rank=np.max(walker_rank, axis=0),
        rhat_folded=np.max(walker_folded, axis=0),
        rhat_final=np.maximum(
            np.max(walker_rank, axis=0), np.max(walker_folded, axis=0)
        ),
        bulk_ess=np.min(walker_bulk, axis=0),
        tail_ess=np.min(walker_tail, axis=0),
        lower_tail_ess=np.min(walker_lower, axis=0),
        upper_tail_ess=np.min(walker_upper, axis=0),
        walker_rhat_rank=walker_rank,
        walker_rhat_folded=walker_folded,
        walker_bulk_ess=walker_bulk,
        walker_lower_tail_ess=walker_lower,
        walker_upper_tail_ess=walker_upper,
        walker_tail_ess=walker_tail,
        mean_absolute_cross_walker_correlation=mean_correlation,
        maximum_absolute_cross_walker_correlation=maximum_correlation,
        maximum_quantile_bin_occupancy_difference=occupancy_difference,
        ensemble_count=count,
        transition_count=prefix,
        walker_count=walkers,
        effective_sample_unit=(
            "minimum-matched-walker-trajectory-ess-across-two-independent-ensembles"
        ),
    )


def write_cluster_diagnostic_artifact(
    path: str | Path, positions: np.ndarray, *, repository: str | Path,
    config_path: str | Path, method: str, likelihood_case: str,
    chain_sha256: list[str], parameter_names: tuple[str, ...],
) -> tuple[Path, Path]:
    """Write approved diagnostic arrays and a schema-v2 provenance manifest."""
    repository = Path(repository).resolve(); path = Path(path).resolve()
    result = independent_ensemble_rank_diagnostics(positions)
    if len(chain_sha256) != result.ensemble_count or len(set(chain_sha256)) != len(chain_sha256):
        raise ValueError("Formal diagnostics require one distinct hash per independent ensemble")
    if method not in ("sccm", "dnn", "pdr"):
        raise ValueError("Unknown controlled MCMC method")
    arrays = {
        "rhat_rank": result.rhat_rank, "rhat_folded": result.rhat_folded,
        "rhat_final": result.rhat_final, "bulk_ess": result.bulk_ess,
        "tail_ess": result.tail_ess, "lower_tail_ess": result.lower_tail_ess,
        "upper_tail_ess": result.upper_tail_ess,
        "walker_rhat_rank": result.walker_rhat_rank,
        "walker_rhat_folded": result.walker_rhat_folded,
        "walker_bulk_ess": result.walker_bulk_ess,
        "walker_lower_tail_ess": result.walker_lower_tail_ess,
        "walker_upper_tail_ess": result.walker_upper_tail_ess,
        "walker_tail_ess": result.walker_tail_ess,
        "mean_absolute_cross_walker_correlation": result.mean_absolute_cross_walker_correlation,
        "maximum_absolute_cross_walker_correlation": result.maximum_absolute_cross_walker_correlation,
        "maximum_quantile_bin_occupancy_difference": result.maximum_quantile_bin_occupancy_difference,
        "parameter_names": np.asarray(parameter_names),
    }
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path); manifest = path.with_suffix(".manifest.json")
    config_path = Path(config_path).resolve()
    write_manifest(manifest, {
        "artifact": "mcmc-cluster-aware-diagnostics", "artifact_role": "development-verification-not-manuscript-evidence",
        "method": method, "likelihood_case": likelihood_case, "data_path": str(path.relative_to(repository)),
        "data_sha256": sha256_file(path), "config": str(config_path.relative_to(repository)),
        "config_sha256": sha256_file(config_path), "chain_sha256": chain_sha256,
        "array_construction": "per-walker-marginal-trajectories-with-only-independent-emcee-ensembles-on-chain-axis;max-rhat-and-min-ess-across-walkers",
        "walker_split_rhat_forbidden": True,
        "shape": list(np.asarray(positions).shape), "rhat_threshold_strictly_less_than": 1.01,
        "bulk_ess_threshold": 400, "tail_ess_threshold": 400,
        "maximum_quantile_bin_occupancy_difference_threshold": 0.10,
        "cross_walker_dependence_is_pass_fail": False,
        "pass": bool(np.all(result.rhat_final < 1.01)
                     and np.all(result.bulk_ess >= 400)
                     and np.all(result.tail_ess >= 400)
                     and np.all(result.maximum_quantile_bin_occupancy_difference < 0.10)),
    }, repository=repository, environment_specification="environment/derecho.yml")
    return path, manifest
