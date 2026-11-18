"""Deterministic, paired Figure 1 training-data generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Iterator

import numpy as np

from cloud_column_reproduction.invariants import audit_manifests
from cloud_column_reproduction.provenance import (
    sha256_file,
    write_manifest,
)
from cloud_column_reproduction.sccm import SCCMRunner, runner_from_repository

from .config import Figure1Config


_WORKER_RUNNER: SCCMRunner | None = None


def derived_seed(base_seed: int, stage: str, index: int) -> int:
    payload = f"cloud-column-reproduction:{base_seed}:{stage}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sample_training_inputs(
    config: Figure1Config, trial: int, sample_count: int
) -> tuple[np.ndarray, int]:
    trials = int(config.experiment["trials"])
    maximum = int(config.experiment["train_size_max"])
    if trial not in range(trials):
        raise ValueError(f"trial must be in [0, {trials - 1}]")
    if sample_count < 1 or sample_count > maximum:
        raise ValueError(f"sample_count must be in [1, {maximum}]")
    seed = derived_seed(int(config.experiment["base_seed"]), "figure1-training", trial)
    generator = np.random.default_rng(seed)
    parameter_box = config.parameter_box
    lower = np.asarray(parameter_box.lower)
    upper = np.asarray(parameter_box.upper)
    inputs = generator.uniform(lower, upper, size=(sample_count, parameter_box.dimension))
    parameter_box.require_physical(inputs)
    return inputs, seed


def _initialize_worker(repository: str, prebuilt: bool, parameter_box) -> None:
    global _WORKER_RUNNER
    _WORKER_RUNNER = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=parameter_box
    )


def _evaluate_worker(parameters: np.ndarray) -> np.ndarray:
    if _WORKER_RUNNER is None:
        raise RuntimeError("SCCM worker was not initialized")
    return _WORKER_RUNNER.run(parameters)


def evaluate_inputs(
    inputs: np.ndarray,
    repository: str | Path,
    parameter_box,
    *,
    workers: int,
    prebuilt: bool,
) -> Iterator[np.ndarray]:
    if workers < 1:
        raise ValueError("workers must be positive")
    repository = str(Path(repository).resolve())
    if workers == 1:
        runner = runner_from_repository(
            repository, prebuilt=prebuilt, parameter_box=parameter_box
        )
        for parameters in inputs:
            yield runner.run(parameters)
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(repository, prebuilt, parameter_box),
    ) as executor:
        yield from executor.map(_evaluate_worker, inputs, chunksize=1)


def generate_training_pool(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    trial: int,
    sample_count: int,
    workers: int,
    prebuilt: bool = False,
) -> tuple[Path, Path]:
    repository = Path(repository).resolve()
    destination_dir = Path(output_directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    inputs, seed = sample_training_inputs(config, trial, sample_count)

    outputs = np.empty((sample_count, 36), dtype=float)
    progress_interval = max(1, min(100, sample_count // 10))
    evaluation_started = time.perf_counter()
    for index, model_output in enumerate(
        evaluate_inputs(
            inputs,
            repository,
            config.parameter_box,
            workers=workers,
            prebuilt=prebuilt,
        )
    ):
        outputs[index] = model_output
        completed = index + 1
        if completed == sample_count or completed % progress_interval == 0:
            print(f"SCCM training evaluations: {completed}/{sample_count}")
    evaluation_wall_seconds = time.perf_counter() - evaluation_started

    stem = f"trial_{trial:03d}_n{sample_count:05d}"
    data_path = destination_dir / f"{stem}.npz"
    temporary_path = destination_dir / f".{stem}.npz.tmp"
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            parameters=inputs,
            model_outputs=outputs,
            diagnostics_120min=outputs[:, 18:24],
            parameter_names=np.asarray(config.parameters["names"]),
            diagnostic_names=np.asarray(config.outputs),
            trial=np.asarray(trial),
            seed=np.asarray(seed, dtype=np.uint64),
        )
    temporary_path.replace(data_path)

    runner = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=config.parameter_box
    )
    manifest_path = destination_dir / f"{stem}.manifest.json"
    write_manifest(
        manifest_path,
        {
            "artifact": "figure1-training-pool",
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"training_pool": seed},
            "random_number_generator": "numpy.random.PCG64",
            "trial": trial,
            "seed": seed,
            "sample_count": sample_count,
            "forward_evaluations": sample_count,
            "workers": workers,
            "wall_seconds": evaluation_wall_seconds,
            "used_prebuilt_sccm": prebuilt,
            "nested_pool": bool(config.experiment["nested_training_subsets"]),
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "model_output_shape": [sample_count, 36],
            "array_metadata": {
                "parameters": {"shape": [sample_count, 11], "dtype": "float64"},
                "model_outputs": {"shape": [sample_count, 36], "dtype": "float64"},
                "diagnostics_120min": {"shape": [sample_count, 6], "dtype": "float64"},
            },
            "diagnostics_120min_columns": [18, 24],
            "diagnostic_names": list(config.outputs),
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


def extend_training_pool(
    config: Figure1Config,
    repository: str | Path,
    output_directory: str | Path,
    *,
    prefix_path: str | Path,
    trial: int,
    sample_count: int,
    workers: int,
    prebuilt: bool = False,
) -> tuple[Path, Path]:
    """Extend a registered nested pool without recomputing its verified prefix."""
    repository = Path(repository).resolve()
    destination_dir = Path(output_directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = Path(prefix_path).resolve()
    prefix_manifest_path = prefix_path.with_name(
        f"{prefix_path.stem}.manifest.json"
    )
    if not prefix_path.is_file() or not prefix_manifest_path.is_file():
        raise FileNotFoundError(
            f"Registered prefix data or manifest is missing for {prefix_path}"
        )
    audit_manifests(
        [prefix_manifest_path],
        config.parameter_box,
        config.parameter_box_path,
        allow_legacy_schema_v1=True,
    )
    prefix_manifest = json.loads(
        prefix_manifest_path.read_text(encoding="utf-8")
    )
    if prefix_manifest.get("artifact") != "figure1-training-pool":
        raise ValueError("Training extension prefix has the wrong artifact role")
    if int(prefix_manifest.get("trial", -1)) != trial:
        raise ValueError("Training extension prefix belongs to another trial")
    if prefix_manifest.get("data_sha256") != sha256_file(prefix_path):
        raise ValueError("Training extension prefix checksum mismatch")
    prefix_count = int(prefix_manifest.get("sample_count", -1))
    if prefix_count < 1 or prefix_count >= sample_count:
        raise ValueError("Training extension requires a nonempty shorter prefix")

    inputs, seed = sample_training_inputs(config, trial, sample_count)
    with np.load(prefix_path, allow_pickle=False) as archive:
        prefix_parameters = np.asarray(archive["parameters"], dtype=np.float64)
        prefix_outputs = np.asarray(archive["model_outputs"], dtype=np.float64)
        prefix_diagnostics = np.asarray(
            archive["diagnostics_120min"], dtype=np.float64
        )
        prefix_names = np.asarray(archive["parameter_names"])
        prefix_diagnostic_names = np.asarray(archive["diagnostic_names"])
        prefix_seed = int(np.asarray(archive["seed"]).item())
    if prefix_parameters.shape != (prefix_count, config.parameter_box.dimension):
        raise ValueError("Training extension prefix parameter shape is inconsistent")
    if prefix_outputs.shape != (prefix_count, 36):
        raise ValueError("Training extension prefix SCCM output shape is inconsistent")
    if prefix_diagnostics.shape != (prefix_count, len(config.outputs)):
        raise ValueError("Training extension prefix diagnostic shape is inconsistent")
    if not np.array_equal(prefix_diagnostics, prefix_outputs[:, 18:24]):
        raise ValueError("Training extension prefix diagnostics do not match SCCM output")
    if prefix_seed != seed or int(prefix_manifest.get("seed", -1)) != seed:
        raise ValueError("Training extension would change the registered trial seed")
    if not np.array_equal(prefix_parameters, inputs[:prefix_count]):
        raise ValueError("Training extension is not an exact nested continuation")
    if not np.array_equal(prefix_names, np.asarray(config.parameters["names"])):
        raise ValueError("Training extension prefix parameter names are inconsistent")
    if not np.array_equal(prefix_diagnostic_names, np.asarray(config.outputs)):
        raise ValueError("Training extension prefix diagnostic names are inconsistent")

    outputs = np.empty((sample_count, 36), dtype=np.float64)
    outputs[:prefix_count] = prefix_outputs
    extension = inputs[prefix_count:]
    progress_interval = max(1, min(100, len(extension) // 10))
    evaluation_started = time.perf_counter()
    for index, model_output in enumerate(
        evaluate_inputs(
            extension,
            repository,
            config.parameter_box,
            workers=workers,
            prebuilt=prebuilt,
        )
    ):
        outputs[prefix_count + index] = model_output
        completed = index + 1
        if completed == len(extension) or completed % progress_interval == 0:
            print(
                "SCCM training extension: "
                f"{completed}/{len(extension)} new "
                f"({prefix_count + completed}/{sample_count} total)",
                flush=True,
            )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started

    stem = f"trial_{trial:03d}_n{sample_count:05d}"
    data_path = destination_dir / f"{stem}.npz"
    temporary_path = destination_dir / f".{stem}.npz.tmp"
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            parameters=inputs,
            model_outputs=outputs,
            diagnostics_120min=outputs[:, 18:24],
            parameter_names=np.asarray(config.parameters["names"]),
            diagnostic_names=np.asarray(config.outputs),
            trial=np.asarray(trial),
            seed=np.asarray(seed, dtype=np.uint64),
        )
    temporary_path.replace(data_path)

    runner = runner_from_repository(
        repository, prebuilt=prebuilt, parameter_box=config.parameter_box
    )
    manifest_path = destination_dir / f"{stem}.manifest.json"
    write_manifest(
        manifest_path,
        {
            "artifact": "figure1-training-pool",
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"training_pool": seed},
            "random_number_generator": "numpy.random.PCG64",
            "trial": trial,
            "seed": seed,
            "sample_count": sample_count,
            "forward_evaluations": len(extension),
            "reused_forward_evaluations": prefix_count,
            "new_forward_evaluations": len(extension),
            "workers": workers,
            "wall_seconds": evaluation_wall_seconds,
            "used_prebuilt_sccm": prebuilt,
            "nested_pool": True,
            "extension_policy": "checksum-verified-exact-rng-prefix-reuse",
            "prefix_data": str(prefix_path.relative_to(repository)),
            "prefix_data_sha256": sha256_file(prefix_path),
            "prefix_manifest": str(prefix_manifest_path.relative_to(repository)),
            "prefix_manifest_sha256": sha256_file(prefix_manifest_path),
            "prefix_sample_count": prefix_count,
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(
                config.parameter_box_path.relative_to(repository)
            ),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "model_output_shape": [sample_count, 36],
            "array_metadata": {
                "parameters": {
                    "shape": [sample_count, 11],
                    "dtype": "float64",
                },
                "model_outputs": {
                    "shape": [sample_count, 36],
                    "dtype": "float64",
                },
                "diagnostics_120min": {
                    "shape": [sample_count, 6],
                    "dtype": "float64",
                },
            },
            "diagnostics_120min_columns": [18, 24],
            "diagnostic_names": list(config.outputs),
            "sccm_executable": str(runner.executable.relative_to(repository)),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist": str(runner.namelist.relative_to(repository)),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table": str(
                runner.radiation_table.relative_to(repository)
            ),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
            "sccm_namelist_bounds_source": "configs/parameter_box.toml",
            "data_file": data_path.name,
            "data_format": "numpy-npz-compressed",
            "data_sha256": sha256_file(data_path),
        },
        repository=repository,
    )
    return data_path, manifest_path
