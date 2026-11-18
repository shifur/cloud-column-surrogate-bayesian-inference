"""Production-unit planning, resumable execution, and finalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import time
from typing import Any
from uuid import uuid4

import numpy as np

from cloud_column_reproduction.provenance import (
    scientific_source_snapshot,
    sha256_file,
    write_manifest,
)
from cloud_column_reproduction.sccm import runner_from_repository

from .chain import write_chain_netcdf_from_blocks
from .config import ControlledMCMCConfig
from .initialization import array_sha256, initialize_independent_ensemble
from .diagnostics import independent_ensemble_diagnostics_from_blocks
from .production_gate import resumable_unit_preflight
from .resumable import AtomicBlockStore, TimedLogProbability, run_resumable_emcee
from .runtime import build_runtime_evaluator
from .transforms import logit_to_physical, physical_to_logit, transformed_log_probability


METHODS = ("sccm", "dnn", "pdr")


def _relative_artifact_path(path: Path, repository: Path) -> str:
    """Return a logical repository-relative path for Scratch-backed artifacts."""
    try:
        return str(path.absolute().relative_to(repository))
    except ValueError:
        pass

    absolute = path.absolute()
    for logical_name in ("data", "outputs", "figures"):
        physical_root = (repository / logical_name).resolve()
        try:
            suffix = absolute.relative_to(physical_root)
        except ValueError:
            continue
        return str(Path(logical_name) / suffix)

    raise ValueError(
        f"{absolute} is neither inside the repository nor one of its "
        "Scratch-backed artifact directories"
    )


@dataclass(frozen=True)
class ProductionUnit:
    likelihood_case: str
    ensemble_id: int
    method: str

    @property
    def relative_directory(self) -> Path:
        return Path(self.likelihood_case) / self.method / f"ensemble_{self.ensemble_id:03d}"


def planned_units(config: ControlledMCMCConfig) -> tuple[ProductionUnit, ...]:
    """Return matched case/ensemble/method order for progressive comparison."""
    units = []
    cases = list(config.execution_cases())
    preferred = config.raw.get("study", {}).get("initial_likelihood_case")
    if preferred in cases:
        cases = [preferred, *(case for case in cases if case != preferred)]
    for case in cases:
        methods = config.execution_methods()
        maximum = max(config.ensemble_count(method) for method in methods)
        for ensemble_id in range(maximum):
            for method in methods:
                if ensemble_id < config.ensemble_count(method):
                    units.append(ProductionUnit(case, ensemble_id, method))
    return tuple(units)


def workload_summary(config: ControlledMCMCConfig) -> dict[str, Any]:
    sampler = config.raw["sampler"]
    per_ensemble = int(sampler["walkers_per_ensemble"]) * (
        int(sampler["burn_in_steps"]) + int(sampler["production_steps"])
    )
    return {
        "study_role": config.study_role,
        "likelihood_cases": list(config.execution_cases()),
        "initial_likelihood_case": config.raw.get("study", {}).get(
            "initial_likelihood_case", list(config.execution_cases())[0]
        ),
        "initial_retained_target": config.raw.get("study", {}).get(
            "initial_retained_target", int(sampler["production_steps"])
        ),
        "methods": list(config.execution_methods()),
        "execution_order": "likelihood-case/ensemble/method",
        "ensemble_counts": {method: config.ensemble_count(method) for method in config.execution_methods()},
        "unit_count": len(planned_units(config)),
        "transition_proposals_per_full_ensemble": per_ensemble,
        "posterior_evaluation_upper_bound": {
            method: len(config.execution_cases()) * config.ensemble_count(method) * (per_ensemble + 32)
            for method in config.execution_methods()
        },
        "forward_evaluation_note": "prior-rejected proposals do not call the forward evaluator",
        "diagnostic_prefixes": sampler["checkpointing"]["diagnostic_prefixes"],
        "resumable": True,
    }


def production_status(
    config: ControlledMCMCConfig, *, output_root: str | Path | None = None
) -> dict[str, Any]:
    """Report progress, productive timing, interruption cost, and rough ETA.

    ETA is a within-unit extrapolation from checksum-committed sampler time. It
    is deliberately omitted before any transitions commit and is not treated as
    a hardware-portable benchmark.
    """
    root = Path(output_root).resolve() if output_root is not None else (
        config.path.parent.parent / config.raw["artifacts"]["root"]
    )
    sampler = config.raw["sampler"]
    target_burn = int(sampler["burn_in_steps"])
    target_production = int(sampler["production_steps"])
    rows = []
    for unit in planned_units(config):
        directory = root / unit.relative_directory
        checkpoint = directory / "checkpoint.json"
        final_manifest = directory / "chain.manifest.json"
        if checkpoint.is_file():
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            burn = int(record.get("completed_burn_in", 0))
            production = int(record.get("completed_production", 0))
            timing = dict(record.get("timing", {}))
            committed = burn + production
            remaining = (target_burn - burn) + (target_production - production)
            productive = float(timing.get("committed_sampler_wall_seconds", 0.0))
            eta = productive * remaining / committed if committed > 0 else None
            status = "finalized" if final_manifest.is_file() else record.get("status", "running")
        else:
            burn = production = 0
            timing = {}
            eta = None
            status = "not-started"
        rows.append(
            {
                "likelihood_case": unit.likelihood_case,
                "ensemble_id": unit.ensemble_id,
                "method": unit.method,
                "status": status,
                "completed_burn_in": burn,
                "completed_production": production,
                "productive_sampler_wall_seconds": timing.get(
                    "committed_sampler_wall_seconds", 0.0
                ),
                "evaluator_wall_seconds": timing.get(
                    "committed_log_probability_wall_seconds", 0.0
                ),
                "successful_evaluations": timing.get(
                    "committed_successful_evaluations", 0
                ),
                "evaluator_failures": timing.get("committed_evaluator_failures", 0),
                "prior_rejections": timing.get("committed_prior_rejections", 0),
                "checkpoint_io_wall_seconds": timing.get(
                    "checkpoint_io_wall_seconds", 0.0
                ),
                "session_setup_wall_seconds": timing.get(
                    "session_setup_wall_seconds", 0.0
                ),
                "caught_interruption_wall_seconds": timing.get(
                    "observed_interrupted_sampler_wall_seconds", 0.0
                ),
                "hard_loss_wall_seconds_upper_bound": timing.get(
                    "interrupted_wall_seconds_upper_bound", 0.0
                ),
                "estimated_remaining_seconds_at_this_unit_rate": eta,
            }
        )
    return {
        "output_root": str(root),
        "timing_interpretation": (
            "ETA uses only committed productive time; caught interruption and "
            "hard-loss upper-bound time remain separate"
        ),
        "units": rows,
    }


def run_prefix_diagnostics(
    config: ControlledMCMCConfig,
    *,
    likelihood_case: str,
    method: str,
    prefix: int,
    output_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Diagnose a common committed prefix of independent ensemble blocks."""
    if method not in METHODS or likelihood_case not in config.masks:
        raise ValueError("Unknown controlled MCMC case or method")
    allowed = tuple(int(value) for value in config.raw["sampler"]["checkpointing"]["diagnostic_prefixes"])
    if prefix not in allowed:
        raise ValueError(f"Diagnostic prefix must be one of {allowed}")
    root = Path(output_root).resolve() if output_root is not None else (
        config.path.parent.parent / config.raw["artifacts"]["root"]
    )
    block_groups = []
    checkpoint_hashes = []
    block_hashes = []
    for ensemble_id in range(config.ensemble_count(method)):
        unit_root = root / ProductionUnit(likelihood_case, ensemble_id, method).relative_directory
        checkpoint = unit_root / "checkpoint.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing ensemble checkpoint: {checkpoint}")
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        if int(record.get("completed_production", 0)) < prefix:
            raise RuntimeError(f"Ensemble {ensemble_id} has not reached prefix {prefix}")
        paths = []
        covered = 0
        hashes = []
        for block in record["blocks"]:
            if block["phase"] != "production" or covered >= prefix:
                continue
            path = unit_root / block["path"]
            if not path.is_file() or sha256_file(path) != block["sha256"]:
                raise RuntimeError(f"Diagnostic source block failed checksum: {path}")
            paths.append(path)
            hashes.append(block["sha256"])
            covered = min(prefix, int(block["stop"]))
        if covered < prefix:
            raise RuntimeError(f"Committed blocks do not cover prefix {prefix}")
        block_groups.append(tuple(paths))
        block_hashes.append(hashes)
        checkpoint_hashes.append(sha256_file(checkpoint))
    coordinate_system = str(config.raw["sampler"].get("coordinate_system", "physical"))
    result = independent_ensemble_diagnostics_from_blocks(
        tuple(block_groups),
        prefix=prefix,
        sampler_coordinate_system=coordinate_system,
        parameter_lower=np.asarray(config.parameter_box.lower, dtype=np.float64),
        parameter_upper=np.asarray(config.parameter_box.upper, dtype=np.float64),
    )
    diagnostics_root = root / "diagnostics" / likelihood_case / method
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    data_path = diagnostics_root / f"prefix_{prefix:09d}.npz"
    manifest_path = data_path.with_suffix(".manifest.json")
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a prefix diagnostic artifact")
    temporary = data_path.with_name(f".{data_path.name}.{uuid4().hex}.tmp")
    arrays = {
        "rhat_rank": result.rhat_rank,
        "rhat_folded": result.rhat_folded,
        "rhat_final": result.rhat_final,
        "bulk_ess": result.bulk_ess,
        "tail_ess": result.tail_ess,
        "lower_tail_ess": result.lower_tail_ess,
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
        "parameter_names": np.asarray(config.parameter_box.names),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, data_path)
    passed = bool(
        np.all(result.rhat_final < 1.01)
        and np.all(result.bulk_ess >= 400)
        and np.all(result.tail_ess >= 400)
        and np.all(result.maximum_quantile_bin_occupancy_difference < 0.10)
    )
    write_manifest(
        manifest_path,
        {
            "artifact": "mcmc-independent-ensemble-prefix-diagnostics",
            "artifact_role": "production-convergence-diagnostic",
            "method": method,
            "likelihood_case": likelihood_case,
            "prefix": prefix,
            "data_path": _relative_artifact_path(data_path, config.path.parent.parent),
            "data_sha256": sha256_file(data_path),
            "config_sha256": sha256_file(config.path),
            "source_checkpoint_sha256": checkpoint_hashes,
            "source_block_sha256": block_hashes,
            "initialization_seeds": list(config.initialization_seeds(method)),
            "chain_axis_source": "independently-initialized-emcee-ensemble",
            "walker_split_rhat_forbidden": True,
            "walker_aggregation": "maximum-rhat-and-minimum-ess-across-matched-walker-trajectories",
            "rhat_threshold_strictly_less_than": 1.01,
            "bulk_ess_threshold": 400,
            "tail_ess_threshold": 400,
            "maximum_quantile_bin_occupancy_difference_threshold": 0.10,
            "pass": passed,
        },
        repository=config.path.parent.parent,
        environment_specification="environment/derecho.yml",
    )
    return data_path, manifest_path


def _unit_bindings(
    config: ControlledMCMCConfig,
    unit: ProductionUnit,
    runtime: Any,
) -> dict[str, str]:
    repository = config.path.parent.parent
    runtime_source_roots = (
        "scripts/mcmc.py",
        "scripts/run_mcmc_surrogates.py",
        "src/cloud_column_reproduction/domain.py",
        "src/cloud_column_reproduction/provenance.py",
        "src/cloud_column_reproduction/sccm.py",
        "src/cloud_column_reproduction/mcmc",
        "src/cloud_column_reproduction/figure1/dnn.py",
        "src/cloud_column_reproduction/figure1/dnn_gpu.py",
        "src/cloud_column_reproduction/figure1/pdr.py",
        "src/cloud_column_reproduction/figure1/pdr_gpu.py",
        "src/cloud_column_reproduction/figure1/polynomials.py",
    )
    bindings = {
        "config_sha256": sha256_file(config.path),
        "observation_sha256": config.raw["observations"]["artifact_sha256"],
        "parameter_box_sha256": config.parameter_box.fingerprint,
        "evaluator_sha256": runtime.metadata.artifact_sha256,
        "mcmc_runtime_source_sha256": scientific_source_snapshot(
            repository, roots=runtime_source_roots
        )["sha256"],
        "environment_specification_sha256": sha256_file(
            repository / "environment" / "derecho.yml"
        ),
        "likelihood_case": unit.likelihood_case,
        "method": unit.method,
        "ensemble_id": str(unit.ensemble_id),
        "sampler_move": str(config.raw["sampler"]["move"]),
        "sampler_move_parameters": json.dumps(
            config.raw["sampler"].get("move_parameters", {}), sort_keys=True
        ),
        "sampler_coordinate_system": str(
            config.raw["sampler"].get("coordinate_system", "physical")
        ),
        "blob_policy": (
            "with-six-output-prediction"
            if runtime.include_prediction_blobs
            else "compact-surrogate-log-prior-likelihood-status"
        ),
    }
    for key, value in runtime.binding_hashes.items():
        bindings[f"evaluator_{key}"] = str(value)
    return bindings


def _post_burn_position(
    store: AtomicBlockStore,
    *,
    coordinate_system: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    paths = store.block_paths("burn-in")
    if not paths:
        raise RuntimeError("Cannot finalize a chain without a completed burn-in state")
    with np.load(paths[-1], allow_pickle=False) as archive:
        values = np.asarray(archive["state_coords"], dtype=np.float64)
    if coordinate_system == "physical":
        return values
    if coordinate_system == "box-logit":
        return logit_to_physical(values, lower, upper)
    raise ValueError("Unknown sampler coordinate system")


def finalize_unit(
    config: ControlledMCMCConfig,
    unit: ProductionUnit,
    store: AtomicBlockStore,
    *,
    initial_positions: np.ndarray,
    evaluator_metadata: Any,
    artifact_role: str,
) -> tuple[Path, Path]:
    target = int(config.raw["sampler"]["production_steps"])
    if int(store.record["completed_production"]) != target:
        raise RuntimeError("Refusing to finalize an incomplete production chain")
    failures = int(store.record["timing"].get("committed_evaluator_failures", 0))
    if failures:
        raise RuntimeError(
            "Refusing to finalize a production chain containing "
            f"{failures} evaluator failures"
        )
    root = store.root
    chain_path = root / "chain.nc"
    manifest_path = root / "chain.manifest.json"
    if chain_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a finalized MCMC unit")
    include_predictions = unit.method == "sccm"
    coordinate_system = str(config.raw["sampler"].get("coordinate_system", "physical"))
    lower = np.asarray(config.parameter_box.lower, dtype=np.float64)
    upper = np.asarray(config.parameter_box.upper, dtype=np.float64)
    metadata = {
        "artifact_role": artifact_role,
        "case": unit.likelihood_case,
        "method": unit.method,
        "ensemble_id": unit.ensemble_id,
        "chain_axis_source": "independently-initialized-emcee-ensemble",
        "walker_dependence": "interacting-cluster-never-promoted-to-chain-axis",
        "initialization_seed": config.initialization_seeds(unit.method)[unit.ensemble_id],
        "proposal_seed": config.proposal_seeds(unit.method)[unit.ensemble_id],
        "sampler_move": config.raw["sampler"]["move"],
        "sampler_move_parameters": config.raw["sampler"].get("move_parameters", {}),
        "sampler_coordinate_system": coordinate_system,
        "stored_coordinate_system": "physical",
        "sampling_density": (
            "physical-posterior-times-absolute-coordinate-Jacobian"
            if coordinate_system == "box-logit" else "physical-posterior"
        ),
        "initial_position_sha256": array_sha256(initial_positions),
        "burn_in_steps": int(config.raw["sampler"]["burn_in_steps"]),
        "production_steps": target,
        "thinning": "none",
        "timing": store.timing_summary(),
    }
    record = write_chain_netcdf_from_blocks(
        chain_path,
        store.block_paths("production"),
        parameter_names=config.parameter_box.names,
        initial_positions=initial_positions,
        post_burn_positions=_post_burn_position(
            store, coordinate_system=coordinate_system, lower=lower, upper=upper
        ),
        metadata=metadata,
        include_sccm_predictions=include_predictions,
        sampler_coordinate_system=coordinate_system,
        parameter_lower=lower,
        parameter_upper=upper,
    )
    prediction_binding = None
    sccm_payload: dict[str, Any] = {}
    evaluator_payload: dict[str, Any] = dict(evaluator_metadata.__dict__)
    if include_predictions:
        runner = runner_from_repository(
            config.path.parent.parent, parameter_box=config.parameter_box
        )
        prediction_binding = {
            "chain_config_sha256": sha256_file(config.path),
            "evaluator_sha256": evaluator_metadata.artifact_sha256,
            "observation_sha256": config.raw["observations"]["artifact_sha256"],
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "prediction_source": "retained-production-state-direct-sccm-evaluation",
        }
        sccm_payload = {
            "sccm_executable": str(runner.executable),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
        }
        evaluator_payload.update(
            kind="direct-sccm",
            prediction_source="direct-sccm-production",
            sccm_executable_sha256=sccm_payload["sccm_executable_sha256"],
            sccm_namelist_sha256=sccm_payload["sccm_namelist_sha256"],
            sccm_rendered_namelist_sha256=sccm_payload["sccm_rendered_namelist_sha256"],
            sccm_radiation_table_sha256=sccm_payload["sccm_radiation_table_sha256"],
        )
    elif unit.method == "dnn":
        evaluator_payload.update(
            kind="frozen-dnn-surrogate",
            model_sha256=config.raw["surrogates"]["dnn"]["model_sha256"],
        )
    else:
        pdr = config.raw["surrogates"]["pdr"]
        evaluator_payload.update(
            kind="frozen-pdr-surrogate",
            model_sha256=pdr["model_sha256"],
            q=pdr["q"],
            basis_size=pdr["basis_size"],
            basis_fingerprint_sha256=pdr["basis_fingerprint_sha256"],
        )
    write_manifest(
        manifest_path,
        {
            "artifact": "mcmc-chain",
            "artifact_role": artifact_role,
            "input_status": "production" if artifact_role == "production" else artifact_role,
            "synthetic": False,
            "method": unit.method,
            "likelihood_case": unit.likelihood_case,
            "likelihood_mask": config.masks[unit.likelihood_case].tolist(),
            "ensemble_id": unit.ensemble_id,
            "ensemble_index": unit.ensemble_id,
            "ensemble_count": config.ensemble_count(unit.method),
            "chain_path": _relative_artifact_path(chain_path, config.path.parent.parent),
            "chain_sha256": record["sha256"],
            "config": str(config.path.relative_to(config.path.parent.parent)),
            "config_sha256": sha256_file(config.path),
            "observation_sha256": config.raw["observations"]["artifact_sha256"],
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "evaluator": evaluator_payload,
            "parameter_order": list(config.parameter_box.names),
            "walker_count": int(config.raw["sampler"]["walkers_per_ensemble"]),
            "retained_transition_count": int(config.raw["sampler"]["production_steps"]),
            "burn_in_transitions": int(config.raw["sampler"]["burn_in_steps"]),
            "burn_in_discarded": True,
            "thinning": 1,
            "initialization_seed": config.initialization_seeds(unit.method)[unit.ensemble_id],
            "proposal_seed": config.proposal_seeds(unit.method)[unit.ensemble_id],
            "sampler_move": config.raw["sampler"]["move"],
            "sampler_move_parameters": config.raw["sampler"].get("move_parameters", {}),
            "sampler_coordinate_system": coordinate_system,
            "stored_coordinate_system": "physical",
            "coordinate_jacobian_in_sampler_target": coordinate_system == "box-logit",
            "initial_position_sha256": array_sha256(initial_positions),
            "chain_axis_source": "independently-initialized-emcee-ensemble",
            "walker_split_rhat_forbidden": True,
            "prediction_cache_present": include_predictions,
            "prediction_cache_array_sha256": record["prediction_cache_array_sha256"],
            "prediction_cache_binding": prediction_binding,
            "shape": record["shape"],
            "source_block_count": record["source_block_count"],
            "timing": store.timing_summary(),
            "checkpoint_sha256": sha256_file(store.checkpoint_path),
            **sccm_payload,
        },
        repository=config.path.parent.parent,
        environment_specification="environment/derecho.yml",
    )
    return chain_path, manifest_path


def run_unit(
    config: ControlledMCMCConfig,
    unit: ProductionUnit,
    *,
    output_root: str | Path | None = None,
    dnn_device: str = "cpu",
    dnn_backend: str | None = None,
    dnn_validation_audit: str | Path | None = None,
    pdr_workers: int = 1,
    pdr_device: str | None = None,
    pdr_backend: str | None = None,
    pdr_prune_exact_zeros: bool | None = None,
    pdr_validation_audit: str | Path | None = None,
    sccm_workers: int = 32,
    prebuilt_sccm: bool = False,
    maximum_blocks: int | None = None,
    production_authorized: bool = False,
    progress_every_transitions: int | None = 100,
    runtime_override: Any | None = None,
    defer_finalization: bool = False,
) -> dict[str, Any]:
    """Advance exactly one configured production unit to a block boundary."""
    if unit not in planned_units(config):
        raise ValueError("Requested MCMC unit is not in the configured schedule")
    if production_authorized:
        config.require_production_ready()
    elif maximum_blocks is None:
        raise RuntimeError(
            "A non-production run must set maximum_blocks; refusing an uncapped "
            "development invocation of the production sampler"
        )
    setup_started = time.perf_counter()
    runtime_config = config.raw.get("runtime", {})
    dnn_runtime = runtime_config.get("dnn", {})
    resolved_dnn_backend = dnn_backend or str(
        dnn_runtime.get(
            "backend",
            "torch-baseline-cuda" if str(dnn_device).startswith("cuda")
            else "torch-baseline",
        )
    )
    resolved_dnn_audit = (
        dnn_validation_audit or dnn_runtime.get("validation_audit")
    )
    pdr_runtime = runtime_config.get("pdr", {})
    resolved_pdr_device = pdr_device or str(pdr_runtime.get("device", "cpu"))
    resolved_pdr_backend = pdr_backend or str(
        pdr_runtime.get(
            "backend", "numpy-cpu" if resolved_pdr_device == "cpu" else "cupy-fused-cuda"
        )
    )
    resolved_pdr_pruning = (
        bool(pdr_runtime.get("prune_exact_zeros", False))
        if pdr_prune_exact_zeros is None else pdr_prune_exact_zeros
    )
    resolved_pdr_audit = pdr_validation_audit or pdr_runtime.get("validation_audit")
    owns_runtime = runtime_override is None
    runtime = runtime_override
    if runtime is None:
        runtime = build_runtime_evaluator(
            config,
            case_name=unit.likelihood_case,
            method=unit.method,
            dnn_device=dnn_device,
            dnn_backend=resolved_dnn_backend,
            dnn_validation_audit=resolved_dnn_audit,
            pdr_workers=pdr_workers,
            pdr_device=resolved_pdr_device,
            pdr_backend=resolved_pdr_backend,
            pdr_prune_exact_zeros=resolved_pdr_pruning,
            pdr_validation_audit=resolved_pdr_audit,
            sccm_workers=sccm_workers,
            prebuilt_sccm=prebuilt_sccm,
        )
    elif (
        runtime.metadata.method != unit.method
        or runtime.likelihood.problem.likelihood_case.name
        != unit.likelihood_case
    ):
        raise ValueError("Shared runtime does not match the requested MCMC unit")

    def _close_owned_runtime() -> None:
        if owns_runtime:
            runtime.close()
    initialization = config.raw["sampler"]["initialization"]
    initial = initialize_independent_ensemble(
        config.parameter_box,
        config.truth,
        config.initialization_scales,
        walkers=int(config.raw["sampler"]["walkers_per_ensemble"]),
        seed=config.initialization_seeds(unit.method)[unit.ensemble_id],
        maximum_redraw_rounds=int(initialization["maximum_redraw_rounds"]),
        law=initialization["law"],
    )
    expected_hash = config.raw["sampler"]["seeds"][unit.method]["initial_position_sha256"][unit.ensemble_id]
    if array_sha256(initial) != expected_hash:
        _close_owned_runtime()
        raise RuntimeError("Initial-position hash does not match the controlled schedule")
    initial_physical = initial
    coordinate_system = str(config.raw["sampler"].get("coordinate_system", "physical"))
    log_probability_function = runtime.log_probability
    if coordinate_system == "box-logit":
        lower = np.asarray(config.parameter_box.lower, dtype=np.float64)
        upper = np.asarray(config.parameter_box.upper, dtype=np.float64)
        initial = physical_to_logit(initial_physical, lower, upper)
        log_probability_function = transformed_log_probability(
            log_probability_function, lower, upper
        )
    elif coordinate_system != "physical":
        _close_owned_runtime()
        raise ValueError("Unsupported sampler coordinate system")
    root = Path(output_root).resolve() if output_root is not None else (
        config.path.parent.parent / config.raw["artifacts"]["root"]
    )
    unit_root = root / unit.relative_directory
    bindings = _unit_bindings(config, unit, runtime)
    if (unit_root / "chain.nc").exists() or (unit_root / "chain.manifest.json").exists():
        completed = resumable_unit_preflight(
            unit_root, expected_bindings=bindings
        )
        _close_owned_runtime()
        return {
            **completed,
            "likelihood_case": unit.likelihood_case,
            "method": unit.method,
            "sampler_coordinate_system": coordinate_system,
            "ensemble_id": unit.ensemble_id,
        }
    store = AtomicBlockStore(unit_root, bindings=bindings)
    setup_wall = time.perf_counter() - setup_started
    store.start_session(
        setup_wall_seconds=setup_wall,
        metadata={
            "method": unit.method,
            "sampler_coordinate_system": coordinate_system,
            "device": runtime.metadata.device,
            "dtype": runtime.metadata.dtype,
            "workers": runtime.metadata.workers,
            "logical_cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "pbs_queue": os.environ.get("PBS_QUEUE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    timed = TimedLogProbability(log_probability_function)
    try:
        result = run_resumable_emcee(
            store,
            timed,
            initial,
            burn_in_steps=int(config.raw["sampler"]["burn_in_steps"]),
            production_steps=int(config.raw["sampler"]["production_steps"]),
            block_transitions=int(config.raw["sampler"]["checkpointing"]["block_transitions"]),
            proposal_seed=config.proposal_seeds(unit.method)[unit.ensemble_id],
            vectorize=runtime.vectorize,
            blobs_dtype=runtime.blobs_dtype,
            store_predictions=unit.method == "sccm",
            maximum_blocks=maximum_blocks,
            progress_every_transitions=progress_every_transitions,
            move_name=config.raw["sampler"]["move"],
            move_parameters=config.raw["sampler"].get("move_parameters", {}),
        )
        store.finish_session(status="complete")
    except BaseException:
        store.finish_session(status="failed-or-interrupted-after-catch")
        raise
    finally:
        _close_owned_runtime()
    if result["status"] == "sampling-complete" and production_authorized and not defer_finalization:
        chain, manifest = finalize_unit(
            config,
            unit,
            store,
            initial_positions=initial_physical,
            evaluator_metadata=runtime.metadata,
            artifact_role="production",
        )
        result.update(chain=str(chain), manifest=str(manifest))
    result.update(
        likelihood_case=unit.likelihood_case,
        method=unit.method,
        ensemble_id=unit.ensemble_id,
        sampler_coordinate_system=coordinate_system,
        checkpoint=str(store.checkpoint_path),
    )
    return result
