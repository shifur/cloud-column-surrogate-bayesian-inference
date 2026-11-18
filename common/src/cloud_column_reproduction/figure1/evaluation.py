"""Exact Tasmanian grid construction and SCCM evaluation for Figure 1."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cloud_column_reproduction.provenance import (
    sha256_file,
    write_manifest,
)
from cloud_column_reproduction.invariants import audit_manifests
from cloud_column_reproduction.sccm import runner_from_repository

from .config import Figure1Config
from .data import derived_seed, evaluate_inputs


def installed_tasmanian_version() -> str:
    try:
        return version("Tasmanian")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Tasmanian is not installed. Activate the environment documented in "
            "environment/README.md before generating the evaluation grid."
        ) from error


def installed_qmcpy_version() -> str:
    try:
        return version("qmcpy")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "QMCPy is not installed. Install the pinned Figure 1 dependency "
            "qmcpy==2.3 before generating the primary RQMC evaluation set."
        ) from error


def construct_canonical_grid(config: Figure1Config) -> tuple[np.ndarray, np.ndarray]:
    installed = installed_tasmanian_version()
    sparse_grid = config.sparse_grid_reference
    required = str(sparse_grid["tasmanian_version"])
    if installed != required:
        raise RuntimeError(f"Tasmanian {required} is required; found {installed}")

    import Tasmanian

    grid = Tasmanian.SparseGrid()
    grid.makeGlobalGrid(
        int(sparse_grid["dimension"]),
        0,
        int(sparse_grid["level"]),
        str(sparse_grid["depth_type"]),
        str(sparse_grid["rule"]),
    )
    points = np.asarray(grid.getPoints(), dtype=float)
    weights = np.asarray(grid.getQuadratureWeights(), dtype=float).reshape(-1)
    expected_shape = (
        int(sparse_grid["expected_points"]),
        int(sparse_grid["dimension"]),
    )
    if points.shape != expected_shape:
        raise RuntimeError(f"Tasmanian grid shape is {points.shape}, expected {expected_shape}")
    if weights.shape != (expected_shape[0],):
        raise RuntimeError("Tasmanian quadrature-weight count does not match its points")
    if np.any(points < -1.0) or np.any(points > 1.0):
        raise RuntimeError("Tasmanian returned points outside the canonical [-1, 1] box")
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("Tasmanian returned non-finite quadrature weights")
    return points, weights


def canonical_to_physical(config: Figure1Config, points: np.ndarray) -> np.ndarray:
    return config.parameter_box.to_physical(points)


def _point_key(point: np.ndarray) -> bytes:
    values = np.array(point, dtype=np.float64, copy=True)
    values[values == 0.0] = 0.0  # normalize the IEEE sign bit on nested zero nodes
    return np.ascontiguousarray(values).tobytes()


def _initialize_nested_outputs(
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_outputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Place reusable outputs at exactly matching nested nodes."""
    target = np.asarray(target_points, dtype=np.float64)
    source = np.asarray(source_points, dtype=np.float64)
    values = np.asarray(source_outputs, dtype=np.float64)
    if target.ndim != 2 or source.ndim != 2 or target.shape[1:] != source.shape[1:]:
        raise ValueError("Nested source and target point dimensions do not match")
    if values.shape != (len(source), 36):
        raise ValueError("Nested source SCCM outputs have the wrong shape")
    source_lookup = {_point_key(point): index for index, point in enumerate(source)}
    if len(source_lookup) != len(source):
        raise ValueError("Nested reuse source contains duplicate grid points")
    outputs = np.full((len(target), 36), np.nan, dtype=np.float64)
    missing: list[int] = []
    reused = 0
    for index, point in enumerate(target):
        source_index = source_lookup.get(_point_key(point))
        if source_index is None:
            missing.append(index)
        else:
            outputs[index] = values[source_index]
            reused += 1
    if reused != len(source):
        raise ValueError("The configured reuse grid is not fully nested in the target grid")
    return outputs, np.asarray(missing, dtype=np.int64), reused


def _load_nested_evaluation_source(
    config: Figure1Config,
    repository: Path,
) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, object]] | None:
    source_name = config.evaluation.get("nested_reuse_data")
    if source_name is None:
        return None
    source_path = (repository / str(source_name)).resolve()
    try:
        source_path.relative_to(repository)
    except ValueError as error:
        raise ValueError("Nested reuse artifact must remain inside the repository") from error
    manifest_path = source_path.with_name(f"{source_path.stem}.manifest.json")
    if not source_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Nested sparse-grid source or manifest is missing: {source_path}"
        )
    audit_manifests(
        [manifest_path], config.parameter_box, config.parameter_box_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_sha256") != sha256_file(source_path):
        raise ValueError("Nested sparse-grid source checksum mismatch")
    if manifest.get("artifact") not in {
        "figure1-primary-sparse-grid-evaluation-data",
        "figure1-secondary-sparse-grid-evaluation-data",
    }:
        raise ValueError("Nested reuse source is not registered sparse-grid truth")
    if manifest.get("corrected_weight_policy") != "signed-quadrature":
        raise ValueError("Nested reuse source did not preserve signed weights")
    if not manifest.get("sccm_evaluated"):
        raise ValueError("Nested reuse source does not contain SCCM evaluations")
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "canonical_points",
        "physical_parameters",
        "model_outputs",
        "diagnostics_120min",
    }
    if not required.issubset(arrays):
        raise ValueError("Nested reuse source is missing required arrays")
    if arrays["canonical_points"].ndim != 2 or arrays["canonical_points"].shape[1] != 11:
        raise ValueError("Nested reuse source has the wrong canonical-point shape")
    source_count = len(arrays["canonical_points"])
    if arrays["physical_parameters"].shape != (source_count, 11):
        raise ValueError("Nested reuse source has the wrong physical-point shape")
    if arrays["model_outputs"].shape != (source_count, 36):
        raise ValueError("Nested reuse source has the wrong SCCM-output shape")
    expected_physical = canonical_to_physical(config, arrays["canonical_points"])
    if not np.array_equal(arrays["physical_parameters"], expected_physical):
        raise ValueError("Nested reuse source does not use the authoritative box mapping")
    return source_path, manifest_path, arrays, manifest


def sample_uniform_parameters(
    config: Figure1Config, section_name: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample a deterministic, role-specific uniform dataset in the shared box."""
    if section_name == "validation":
        section = config.validation
    elif section_name == "monte_carlo_diagnostic":
        section = config.monte_carlo_diagnostic
    elif section_name == "evaluation" and config.evaluation.get("kind") == (
        "independent-uniform-monte-carlo"
    ):
        # Backward-compatible access for the superseded versioned config.
        section = config.evaluation
    else:
        raise ValueError(
            "Uniform dataset role must be validation or monte_carlo_diagnostic"
        )
    point_count = int(section["expected_points"])
    seed = derived_seed(
        int(config.experiment["base_seed"]), str(section["seed_stage"]), 0
    )
    generator = np.random.default_rng(seed)
    parameter_box = config.parameter_box
    physical_parameters = generator.uniform(
        np.asarray(parameter_box.lower),
        np.asarray(parameter_box.upper),
        size=(point_count, parameter_box.dimension),
    )
    parameter_box.require_physical(physical_parameters)
    return physical_parameters, parameter_box.to_canonical(physical_parameters), seed


def generate_monte_carlo_diagnostic_data(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    workers: int,
    prebuilt: bool = False,
    inputs_only: bool = False,
) -> tuple[Path, Path]:
    """Generate the optional equal-weight Monte Carlo robustness dataset."""
    repository = Path(repository).resolve()
    destination_dir = Path(output_directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    monte_carlo = config.monte_carlo_diagnostic
    point_count = int(monte_carlo["expected_points"])
    physical_parameters, canonical_points, seed = sample_uniform_parameters(
        config, "monte_carlo_diagnostic"
    )

    outputs: np.ndarray | None = None
    evaluation_wall_seconds = 0.0
    if not inputs_only:
        outputs = np.empty((point_count, 36), dtype=float)
        evaluation_started = time.perf_counter()
        for index, model_output in enumerate(
            evaluate_inputs(
                physical_parameters,
                repository,
                config.parameter_box,
                workers=workers,
                prebuilt=prebuilt,
            )
        ):
            outputs[index] = model_output
            completed = index + 1
            if completed == point_count or completed % 100 == 0:
                print(f"SCCM Monte Carlo diagnostic calls: {completed}/{point_count}")
        evaluation_wall_seconds = time.perf_counter() - evaluation_started

    stem = f"uniform_mc_n{point_count}"
    if inputs_only:
        stem += "_inputs_only"
    data_path = destination_dir / f"{stem}.npz"
    temporary_path = destination_dir / f".{stem}.npz.tmp"
    arrays = {
        "canonical_points": canonical_points,
        "physical_parameters": physical_parameters,
        "equal_weights": np.full(point_count, 1.0 / point_count),
        "parameter_names": np.asarray(config.parameters["names"]),
        "seed": np.asarray(seed, dtype=np.uint64),
    }
    if outputs is not None:
        arrays.update(
            model_outputs=outputs,
            diagnostics_120min=outputs[:, 18:24],
            diagnostic_names=np.asarray(config.outputs),
        )
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_path.replace(data_path)

    runner = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=config.parameter_box
    )
    manifest_path = destination_dir / f"{stem}.manifest.json"
    write_manifest(
        manifest_path,
        {
            "artifact": (
                "figure1-monte-carlo-diagnostic-inputs"
                if inputs_only
                else "figure1-monte-carlo-diagnostic-data"
            ),
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"monte_carlo_diagnostic": seed},
            "random_number_generator": "numpy.random.PCG64",
            "evaluation_kind": monte_carlo["kind"],
            "metric": monte_carlo["metric"],
            "metric_status": monte_carlo.get(
                "metric_status", "superseded-equal-weight-contract"
            ),
            "point_count": point_count,
            "seed": seed,
            "used_for_model_selection": bool(monte_carlo["used_for_model_selection"]),
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "sccm_evaluated": outputs is not None,
            "forward_evaluations": 0 if outputs is None else point_count,
            "workers": workers,
            "wall_seconds": evaluation_wall_seconds,
            "used_prebuilt_sccm": prebuilt,
            "model_output_shape": None if outputs is None else [point_count, 36],
            "array_metadata": {
                "canonical_points": {"shape": [point_count, 11], "dtype": "float64"},
                "physical_parameters": {"shape": [point_count, 11], "dtype": "float64"},
                "equal_weights": {"shape": [point_count], "dtype": "float64"},
                "model_outputs": None
                if outputs is None
                else {"shape": [point_count, 36], "dtype": "float64"},
                "diagnostics_120min": None
                if outputs is None
                else {"shape": [point_count, 6], "dtype": "float64"},
            },
            "diagnostics_120min_columns": None if outputs is None else [18, 24],
            "sccm_executable": str(runner.executable.relative_to(repository)),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist": str(runner.namelist.relative_to(repository)),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table": str(runner.radiation_table.relative_to(repository)),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "sccm_namelist_bounds_source": "configs/parameter_box.toml",
            "data_file": data_path.name,
            "data_format": "numpy-npz-compressed",
            "data_sha256": sha256_file(data_path),
        },
        repository=repository,
    )
    return data_path, manifest_path


def generate_primary_evaluation_data(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    workers: int,
    prebuilt: bool = False,
    inputs_only: bool = False,
) -> tuple[Path, Path]:
    """Generate the config's primary evaluation, retaining old-config support."""
    if config.is_rqmc_primary:
        return generate_rqmc_evaluation_data(
            config,
            repository,
            output_directory,
            workers=workers,
            prebuilt=prebuilt,
            inputs_only=inputs_only,
        )
    if config.is_tasmanian_evaluation:
        return generate_evaluation_data(
            config,
            repository,
            output_directory,
            workers=workers,
            prebuilt=prebuilt,
            grid_only=inputs_only,
        )
    return generate_monte_carlo_diagnostic_data(
        config,
        repository,
        output_directory,
        workers=workers,
        prebuilt=prebuilt,
        inputs_only=inputs_only,
    )


def _construct_rqmc_points(
    config: Figure1Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct independent Owen-scrambled Sobol replications in canonical order."""
    evaluation = config.evaluation
    required = str(evaluation["generator_version"])
    installed = installed_qmcpy_version()
    if installed != required:
        raise RuntimeError(f"QMCPy {required} is required; found {installed}")

    from qmcpy import DigitalNetB2

    replications = int(evaluation["replications"])
    point_count = int(evaluation["points_per_replication"])
    dimension = int(evaluation["dimension"])
    points = np.empty((replications, point_count, dimension), dtype=np.float64)
    seeds = np.empty(replications, dtype=np.uint64)
    for replication in range(replications):
        seed = derived_seed(
            int(config.experiment["base_seed"]),
            str(evaluation["seed_stage"]),
            replication,
        )
        seeds[replication] = seed
        distribution = DigitalNetB2(
            dimension=dimension,
            seed=seed,
            randomize=str(evaluation["randomization"]),
            generating_matrices=str(evaluation["generating_matrices"]),
            order=str(evaluation["order"]),
            t=int(evaluation["bits"]),
            alpha=int(evaluation["alpha"]),
        )
        unit_points = np.asarray(distribution.gen_samples(n=point_count), dtype=np.float64)
        if unit_points.shape != (point_count, dimension):
            raise RuntimeError(
                f"QMCPy replication {replication} returned {unit_points.shape}, "
                f"expected {(point_count, dimension)}"
            )
        if np.any(unit_points < 0.0) or np.any(unit_points >= 1.0):
            raise RuntimeError("QMCPy returned a point outside [0, 1)")
        points[replication] = 2.0 * unit_points - 1.0
    return points, seeds


def generate_rqmc_evaluation_data(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    workers: int,
    prebuilt: bool = False,
    inputs_only: bool = False,
) -> tuple[Path, Path]:
    """Generate the immutable replicated scrambled-Sobol primary test set."""
    if not config.is_rqmc_primary:
        raise ValueError("The selected config does not declare an RQMC primary target")
    repository = Path(repository).resolve()
    destination_dir = Path(output_directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    evaluation = config.evaluation
    canonical_points, seeds = _construct_rqmc_points(config)
    physical_parameters = canonical_to_physical(config, canonical_points)
    config.parameter_box.require_physical(physical_parameters.reshape(-1, 11))
    replications, points_per_replication, dimension = canonical_points.shape
    total_points = replications * points_per_replication
    stem = f"scrambled_sobol_r{replications}_n{points_per_replication}"
    runner = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=config.parameter_box
    )

    outputs: np.ndarray | None = None
    evaluation_wall_seconds = 0.0
    completed_before_resume = 0
    if not inputs_only:
        progress_path = destination_dir / f".{stem}.outputs.progress.npy"
        progress_metadata_path = progress_path.with_suffix(".json")
        progress_contract = {
            "config_sha256": sha256_file(config.path),
            "canonical_points_sha256": hashlib.sha256(
                np.ascontiguousarray(canonical_points).tobytes()
            ).hexdigest(),
            "replication_seeds": [int(seed) for seed in seeds],
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "shape": [total_points, 36],
            "dtype": "float64",
        }
        if progress_path.is_file() or progress_metadata_path.is_file():
            if not progress_path.is_file() or not progress_metadata_path.is_file():
                raise RuntimeError("Incomplete RQMC progress checkpoint pair")
            recorded_contract = json.loads(
                progress_metadata_path.read_text(encoding="utf-8")
            )
            if recorded_contract != progress_contract:
                raise RuntimeError("RQMC progress checkpoint contract mismatch")
            flat_outputs = np.lib.format.open_memmap(
                progress_path, mode="r+", dtype=np.float64, shape=(total_points, 36)
            )
        else:
            flat_outputs = np.lib.format.open_memmap(
                progress_path, mode="w+", dtype=np.float64, shape=(total_points, 36)
            )
            flat_outputs[:] = np.nan
            flat_outputs.flush()
            progress_metadata_path.write_text(
                json.dumps(progress_contract, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        completed_rows = np.all(np.isfinite(flat_outputs), axis=1)
        incomplete = np.flatnonzero(~completed_rows)
        completed_before_resume = total_points if not len(incomplete) else int(incomplete[0])
        if np.any(completed_rows[completed_before_resume:]):
            raise RuntimeError("RQMC progress checkpoint is not a contiguous prefix")
        if completed_before_resume:
            print(
                f"Resuming SCCM RQMC evaluation at "
                f"{completed_before_resume}/{total_points}",
                flush=True,
            )
        evaluation_started = time.perf_counter()
        for index, model_output in enumerate(
            evaluate_inputs(
                physical_parameters.reshape(total_points, dimension)[
                    completed_before_resume:
                ],
                repository,
                config.parameter_box,
                workers=workers,
                prebuilt=prebuilt,
            ),
            start=completed_before_resume,
        ):
            flat_outputs[index] = model_output
            completed = index + 1
            if completed == total_points or completed % 100 == 0:
                flat_outputs.flush()
                print(f"SCCM RQMC evaluation calls: {completed}/{total_points}", flush=True)
        evaluation_wall_seconds = time.perf_counter() - evaluation_started
        flat_outputs.flush()
        if not np.all(np.isfinite(flat_outputs)):
            raise RuntimeError("RQMC SCCM progress contains incomplete or non-finite rows")
        outputs = np.array(flat_outputs, copy=True).reshape(
            replications, points_per_replication, 36
        )

    if inputs_only:
        stem += "_inputs_only"
    data_path = destination_dir / f"{stem}.npz"
    temporary_path = destination_dir / f".{stem}.npz.tmp"
    arrays = {
        "canonical_points": canonical_points,
        "physical_parameters": physical_parameters,
        "equal_weights": np.full(
            (replications, points_per_replication),
            1.0 / points_per_replication,
            dtype=np.float64,
        ),
        "replication_seeds": seeds,
        "prefix_log2": np.asarray(evaluation["prefix_log2"], dtype=np.int64),
        "parameter_names": np.asarray(config.parameters["names"]),
    }
    if outputs is not None:
        arrays.update(
            model_outputs=outputs,
            diagnostics_120min=outputs[:, :, 18:24],
            diagnostic_names=np.asarray(config.outputs),
        )
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_path.replace(data_path)
    if outputs is not None:
        del flat_outputs
        progress_path.unlink()
        progress_metadata_path.unlink()

    manifest_path = destination_dir / f"{stem}.manifest.json"
    array_metadata = {
        "canonical_points": {
            "shape": [replications, points_per_replication, dimension],
            "dtype": "float64",
        },
        "physical_parameters": {
            "shape": [replications, points_per_replication, dimension],
            "dtype": "float64",
        },
        "equal_weights": {
            "shape": [replications, points_per_replication],
            "dtype": "float64",
        },
        "replication_seeds": {"shape": [replications], "dtype": "uint64"},
        "model_outputs": None
        if outputs is None
        else {
            "shape": [replications, points_per_replication, 36],
            "dtype": "float64",
        },
        "diagnostics_120min": None
        if outputs is None
        else {
            "shape": [replications, points_per_replication, 6],
            "dtype": "float64",
        },
    }
    write_manifest(
        manifest_path,
        {
            "artifact": (
                "figure1-primary-rqmc-inputs"
                if inputs_only
                else "figure1-primary-rqmc-evaluation-data"
            ),
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"rqmc_replications": [int(seed) for seed in seeds]},
            "random_number_generator": "QMCPy.DigitalNetB2",
            "qmcpy_version": installed_qmcpy_version(),
            "evaluation_kind": evaluation["kind"],
            "rqmc_construction": {
                "generator_class": evaluation["generator_class"],
                "randomization": evaluation["randomization"],
                "generating_matrices": evaluation["generating_matrices"],
                "order": evaluation["order"],
                "bits": int(evaluation["bits"]),
                "alpha": int(evaluation["alpha"]),
                "dimension": dimension,
                "replications": replications,
                "points_per_replication": points_per_replication,
                "prefix_log2": [int(value) for value in evaluation["prefix_log2"]],
            },
            "point_count": total_points,
            "replications": replications,
            "points_per_replication": points_per_replication,
            "used_for_model_selection": False,
            "metric": evaluation["metric"],
            "metric_status": evaluation["metric_status"],
            "weight_policy": evaluation["weight_policy"],
            "uncertainty_method": evaluation["uncertainty_method"],
            "confidence_level": float(evaluation["confidence_level"]),
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "sccm_evaluated": outputs is not None,
            "forward_evaluations": 0
            if outputs is None
            else total_points - completed_before_resume,
            "total_evaluation_points": total_points,
            "resumed_completed_evaluations": completed_before_resume,
            "workers": workers,
            "wall_seconds": evaluation_wall_seconds,
            "used_prebuilt_sccm": prebuilt,
            "model_output_shape": None
            if outputs is None
            else [replications, points_per_replication, 36],
            "array_metadata": array_metadata,
            "diagnostics_120min_columns": None if outputs is None else [18, 24],
            "sccm_executable": str(runner.executable.relative_to(repository)),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist": str(runner.namelist.relative_to(repository)),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table": str(runner.radiation_table.relative_to(repository)),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "sccm_namelist_bounds_source": "configs/parameter_box.toml",
            "data_file": data_path.name,
            "data_format": "numpy-npz-compressed",
            "data_sha256": sha256_file(data_path),
        },
        repository=repository,
    )
    return data_path, manifest_path


def generate_validation_data(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    workers: int,
    prebuilt: bool = False,
    inputs_only: bool = False,
) -> tuple[Path, Path]:
    """Generate the fixed DNN model-selection set, separate from test data."""
    repository = Path(repository).resolve()
    destination_dir = Path(output_directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    point_count = int(config.validation["expected_points"])
    physical_parameters, canonical_points, seed = sample_uniform_parameters(
        config, "validation"
    )

    outputs: np.ndarray | None = None
    evaluation_wall_seconds = 0.0
    if not inputs_only:
        outputs = np.empty((point_count, 36), dtype=float)
        evaluation_started = time.perf_counter()
        for index, model_output in enumerate(
            evaluate_inputs(
                physical_parameters,
                repository,
                config.parameter_box,
                workers=workers,
                prebuilt=prebuilt,
            )
        ):
            outputs[index] = model_output
            completed = index + 1
            if completed == point_count or completed % 100 == 0:
                print(f"SCCM validation calls: {completed}/{point_count}")
        evaluation_wall_seconds = time.perf_counter() - evaluation_started

    stem = f"uniform_mc_n{point_count}"
    if inputs_only:
        stem += "_inputs_only"
    data_path = destination_dir / f"{stem}.npz"
    temporary_path = destination_dir / f".{stem}.npz.tmp"
    arrays = {
        "canonical_points": canonical_points,
        "physical_parameters": physical_parameters,
        "parameter_names": np.asarray(config.parameters["names"]),
        "seed": np.asarray(seed, dtype=np.uint64),
    }
    if outputs is not None:
        arrays.update(
            model_outputs=outputs,
            diagnostics_120min=outputs[:, 18:24],
            diagnostic_names=np.asarray(config.outputs),
        )
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_path.replace(data_path)

    runner = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=config.parameter_box
    )
    manifest_path = destination_dir / f"{stem}.manifest.json"
    write_manifest(
        manifest_path,
        {
            "artifact": (
                "figure1-validation-inputs"
                if inputs_only
                else "figure1-validation-data"
            ),
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"validation": seed},
            "random_number_generator": "numpy.random.PCG64",
            "validation_kind": config.validation["kind"],
            "point_count": point_count,
            "seed": seed,
            "used_for_model_selection": bool(
                config.validation["used_for_model_selection"]
            ),
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "sccm_evaluated": outputs is not None,
            "forward_evaluations": 0 if outputs is None else point_count,
            "workers": workers,
            "wall_seconds": evaluation_wall_seconds,
            "used_prebuilt_sccm": prebuilt,
            "model_output_shape": None if outputs is None else [point_count, 36],
            "array_metadata": {
                "canonical_points": {"shape": [point_count, 11], "dtype": "float64"},
                "physical_parameters": {
                    "shape": [point_count, 11],
                    "dtype": "float64",
                },
                "model_outputs": None
                if outputs is None
                else {"shape": [point_count, 36], "dtype": "float64"},
                "diagnostics_120min": None
                if outputs is None
                else {"shape": [point_count, 6], "dtype": "float64"},
            },
            "diagnostics_120min_columns": None if outputs is None else [18, 24],
            "sccm_executable": str(runner.executable.relative_to(repository)),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist": str(runner.namelist.relative_to(repository)),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table": str(runner.radiation_table.relative_to(repository)),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "sccm_namelist_bounds_source": "configs/parameter_box.toml",
            "data_file": data_path.name,
            "data_format": "numpy-npz-compressed",
            "data_sha256": sha256_file(data_path),
        },
        repository=repository,
    )
    return data_path, manifest_path


def generate_evaluation_data(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    workers: int,
    prebuilt: bool = False,
    grid_only: bool = False,
) -> tuple[Path, Path]:
    repository = Path(repository).resolve()
    destination_dir = Path(output_directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    canonical_points, weights = construct_canonical_grid(config)
    physical_parameters = canonical_to_physical(config, canonical_points)
    point_count = len(canonical_points)
    level = int(config.sparse_grid_reference["level"])
    rule_tag = str(config.sparse_grid_reference["rule"]).replace("-", "_")
    base_stem = f"tasmanian_l{level}_{rule_tag}"
    runner = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=config.parameter_box
    )

    outputs: np.ndarray | None = None
    reused_evaluations = 0
    new_forward_evaluations = 0
    completed_before_resume = 0
    reuse_source_path: Path | None = None
    reuse_manifest_path: Path | None = None
    evaluation_wall_seconds = 0.0
    if not grid_only:
        output_template = np.full((point_count, 36), np.nan, dtype=np.float64)
        missing_indices = np.arange(point_count, dtype=np.int64)
        reuse_source = _load_nested_evaluation_source(config, repository)
        if reuse_source is not None:
            (
                reuse_source_path,
                reuse_manifest_path,
                reuse_arrays,
                _,
            ) = reuse_source
            source_points = np.asarray(
                reuse_arrays["canonical_points"], dtype=np.float64
            )
            source_outputs = np.asarray(reuse_arrays["model_outputs"], dtype=np.float64)
            output_template, missing_indices, reused_evaluations = _initialize_nested_outputs(
                canonical_points, source_points, source_outputs
            )
            print(
                f"Reused nested SCCM evaluations: {reused_evaluations}/{point_count}; "
                f"new evaluations required: {len(missing_indices)}",
                flush=True,
            )
        new_forward_evaluations = len(missing_indices)
        progress_path = destination_dir / f".{base_stem}.outputs.progress.npy"
        progress_metadata_path = progress_path.with_suffix(".json")
        progress_contract = {
            "config_sha256": sha256_file(config.path),
            "canonical_points_sha256": hashlib.sha256(
                np.ascontiguousarray(canonical_points).tobytes()
            ).hexdigest(),
            "nested_reuse_data_sha256": None
            if reuse_source_path is None
            else sha256_file(reuse_source_path),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "shape": [point_count, 36],
            "dtype": "float64",
        }
        if progress_path.is_file() or progress_metadata_path.is_file():
            if not progress_path.is_file() or not progress_metadata_path.is_file():
                raise RuntimeError("Incomplete sparse-grid progress checkpoint pair")
            recorded_contract = json.loads(
                progress_metadata_path.read_text(encoding="utf-8")
            )
            if recorded_contract != progress_contract:
                raise RuntimeError("Sparse-grid progress checkpoint contract mismatch")
            progress_outputs = np.lib.format.open_memmap(
                progress_path,
                mode="r+",
                dtype=np.float64,
                shape=(point_count, 36),
            )
            reused_indices = np.flatnonzero(np.all(np.isfinite(output_template), axis=1))
            if reused_indices.size and not np.array_equal(
                progress_outputs[reused_indices], output_template[reused_indices]
            ):
                raise RuntimeError("Sparse-grid checkpoint disagrees with nested truth")
        else:
            progress_outputs = np.lib.format.open_memmap(
                progress_path,
                mode="w+",
                dtype=np.float64,
                shape=(point_count, 36),
            )
            progress_outputs[:] = output_template
            progress_outputs.flush()
            progress_metadata_path.write_text(
                json.dumps(progress_contract, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        completed_new = np.all(np.isfinite(progress_outputs[missing_indices]), axis=1)
        incomplete_positions = np.flatnonzero(~completed_new)
        completed_before_resume = (
            new_forward_evaluations
            if not len(incomplete_positions)
            else int(incomplete_positions[0])
        )
        if np.any(completed_new[completed_before_resume:]):
            raise RuntimeError("Sparse-grid progress is not a contiguous new-point prefix")
        if completed_before_resume:
            print(
                "Resuming new SCCM evaluation-grid calls at "
                f"{completed_before_resume}/{new_forward_evaluations}",
                flush=True,
            )
        evaluation_started = time.perf_counter()
        for completed_index, model_output in enumerate(
            evaluate_inputs(
                physical_parameters[missing_indices[completed_before_resume:]],
                repository,
                config.parameter_box,
                workers=workers,
                prebuilt=prebuilt,
            ),
            start=completed_before_resume + 1,
        ):
            target_index = int(missing_indices[completed_index - 1])
            progress_outputs[target_index] = model_output
            if completed_index == new_forward_evaluations or completed_index % 100 == 0:
                progress_outputs.flush()
                print(
                    "New SCCM evaluation-grid calls: "
                    f"{completed_index}/{new_forward_evaluations}",
                    flush=True,
                )
        evaluation_wall_seconds = time.perf_counter() - evaluation_started
        progress_outputs.flush()
        if not np.all(np.isfinite(progress_outputs)):
            raise RuntimeError("Combined nested sparse-grid outputs are not finite")
        outputs = np.array(progress_outputs, copy=True)

    stem = base_stem
    if grid_only:
        stem += "_grid_only"
    data_path = destination_dir / f"{stem}.npz"
    temporary_path = destination_dir / f".{stem}.npz.tmp"
    arrays = {
        "canonical_points": canonical_points,
        "physical_parameters": physical_parameters,
        "quadrature_weights": weights,
        "parameter_names": np.asarray(config.parameters["names"]),
    }
    if outputs is not None:
        arrays.update(
            model_outputs=outputs,
            diagnostics_120min=outputs[:, 18:24],
            diagnostic_names=np.asarray(config.outputs),
        )
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_path.replace(data_path)
    if outputs is not None:
        del progress_outputs
        progress_path.unlink()
        progress_metadata_path.unlink()

    manifest_path = destination_dir / f"{stem}.manifest.json"
    metric_status = str(config.sparse_grid_reference["metric_status"])
    primary_grid = metric_status == "primary-figure1-error-metric"
    write_manifest(
        manifest_path,
        {
            "artifact": (
                (
                    "figure1-primary-sparse-grid-inputs"
                    if primary_grid
                    else "figure1-secondary-sparse-grid-inputs"
                )
                if grid_only
                else (
                    "figure1-primary-sparse-grid-evaluation-data"
                    if primary_grid
                    else "figure1-secondary-sparse-grid-evaluation-data"
                )
            ),
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {},
            "random_number_generator": None,
            "tasmanian_version": installed_tasmanian_version(),
            "tasmanian_construction": {
                "kind": config.sparse_grid_reference["kind"],
                "dimension": int(config.sparse_grid_reference["dimension"]),
                "outputs": 0,
                "level": int(config.sparse_grid_reference["level"]),
                "depth_type": config.sparse_grid_reference["depth_type"],
                "rule": config.sparse_grid_reference["rule"],
            },
            "point_count": point_count,
            "used_for_model_selection": bool(
                config.sparse_grid_reference["used_for_model_selection"]
            ),
            "metric_status": config.sparse_grid_reference["metric_status"],
            "metric": config.sparse_grid_reference.get(
                "metric", "signed-quadrature-relative-l2"
            ),
            "historical_weight_policy": config.sparse_grid_reference[
                "historical_weight_policy"
            ],
            "corrected_weight_policy": config.sparse_grid_reference[
                "corrected_weight_policy"
            ],
            "quadrature_weight_sum": float(weights.sum()),
            "quadrature_absolute_weight_sum": float(np.abs(weights).sum()),
            "quadrature_negative_weight_count": int(np.count_nonzero(weights < 0.0)),
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "sccm_evaluated": outputs is not None,
            "forward_evaluations": 0
            if outputs is None
            else new_forward_evaluations - completed_before_resume,
            "total_grid_evaluations": 0 if outputs is None else point_count,
            "nested_reused_evaluations": reused_evaluations,
            "new_forward_evaluations": 0 if outputs is None else new_forward_evaluations,
            "resumed_completed_evaluations": completed_before_resume,
            "nested_reuse_data": None
            if reuse_source_path is None
            else str(reuse_source_path.relative_to(repository)),
            "nested_reuse_data_sha256": None
            if reuse_source_path is None
            else sha256_file(reuse_source_path),
            "nested_reuse_manifest": None
            if reuse_manifest_path is None
            else str(reuse_manifest_path.relative_to(repository)),
            "nested_reuse_manifest_sha256": None
            if reuse_manifest_path is None
            else sha256_file(reuse_manifest_path),
            "workers": workers,
            "wall_seconds": evaluation_wall_seconds,
            "used_prebuilt_sccm": prebuilt,
            "model_output_shape": None if outputs is None else [point_count, 36],
            "array_metadata": {
                "canonical_points": {"shape": [point_count, 11], "dtype": "float64"},
                "physical_parameters": {"shape": [point_count, 11], "dtype": "float64"},
                "quadrature_weights": {"shape": [point_count], "dtype": "float64"},
                "model_outputs": None
                if outputs is None
                else {"shape": [point_count, 36], "dtype": "float64"},
                "diagnostics_120min": None
                if outputs is None
                else {"shape": [point_count, 6], "dtype": "float64"},
            },
            "diagnostics_120min_columns": None if outputs is None else [18, 24],
            "sccm_executable": str(runner.executable.relative_to(repository)),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist": str(runner.namelist.relative_to(repository)),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table": str(runner.radiation_table.relative_to(repository)),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "sccm_namelist_bounds_source": "configs/parameter_box.toml",
            "data_file": data_path.name,
            "data_format": "numpy-npz-compressed",
            "data_sha256": sha256_file(data_path),
        },
        repository=repository,
    )
    return data_path, manifest_path
