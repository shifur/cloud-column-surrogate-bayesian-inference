"""Checksum-bound exact float64 q=23 design caches for sample-major Figure 1."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from cloud_column_reproduction.provenance import sha256_file, write_manifest

from .config import Figure1Config, pdr_spectral_norm_options
from .fitting import _load_registered_data
from .pdr import spectral_norm_estimate

from .polynomials import (
    hyperbolic_cross_indices,
    intrinsic_legendre_weights,
    legendre_design_matrix,
    polynomial_basis_fingerprint,
)


def cache_paths(directory: str | Path, trial: int) -> tuple[Path, Path]:
    # Preserve the logical repository path when data/ is a symlink to Scratch.
    root = Path(directory).absolute()
    data = root / f"trial_{trial:03d}_q23_n05000.npy"
    return data, data.with_suffix(".manifest.json")


def build_design_cache(
    config: Figure1Config,
    repository: str | Path,
    training_path: str | Path,
    output_directory: str | Path,
    *,
    trial: int,
    blas_threads: int = 1,
) -> tuple[Path, Path]:
    repository = Path(repository).resolve()
    if blas_threads < 1:
        raise ValueError("Design-cache BLAS thread count must be positive")
    overall_started = time.perf_counter()
    data_path, manifest_path = cache_paths(output_directory, trial)
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError("Figure 1 design-cache artifacts are immutable")
    order = int(config.cs["hyperbolic_cross_order"])
    expected_basis_size = int(config.cs["expected_basis_size"])
    dimension = config.parameter_box.dimension
    if order != 23 or expected_basis_size != 7569 or dimension != 11:
        raise ValueError("Current Figure 1 design cache is locked to d=11, q=23, N=7,569")
    data_load_started = time.perf_counter()
    training, manifest, _ = _load_registered_data(training_path, config)
    data_load_wall = time.perf_counter() - data_load_started
    if int(manifest["trial"]) != trial:
        raise ValueError("Training trial does not match design-cache trial")
    maximum = int(config.experiment["train_size_max"])
    canonical_started = time.perf_counter()
    physical = np.asarray(training["parameters"][:maximum], dtype=np.float64)
    canonical = config.parameter_box.to_canonical(physical)
    canonical_wall = time.perf_counter() - canonical_started
    indices = hyperbolic_cross_indices(dimension, order)
    if len(indices) != expected_basis_size:
        raise RuntimeError("q=23 basis cardinality changed")
    started = time.perf_counter()
    design = legendre_design_matrix(canonical, indices, dtype=np.float64)
    construction_wall = time.perf_counter() - started
    norm_options = pdr_spectral_norm_options(config)
    norm_started = time.perf_counter()
    spectral_norms = []
    for position, sample_count in enumerate(config.train_sizes, start=1):
        normalized_design = design[:sample_count] / np.sqrt(float(sample_count))
        inflated, raw = spectral_norm_estimate(
            normalized_design,
            method=norm_options["spectral_norm_method"],
            safety_factor=norm_options["spectral_norm_safety_factor"],
            seed=norm_options["spectral_norm_seed"],
            power_iterations=norm_options["spectral_norm_power_iterations"],
            randomized_oversamples=norm_options["spectral_norm_randomized_oversamples"],
            randomized_power_iterations=norm_options["spectral_norm_randomized_power_iterations"],
        )
        spectral_norms.append({
            "sample_count": sample_count,
            "spectral_norm": inflated,
            "spectral_norm_raw_estimate": raw,
        })
        if position % 10 == 0 or position == len(config.train_sizes):
            print(
                f"q=23 cache trial={trial:03d}: spectral norms "
                f"{position}/{len(config.train_sizes)}",
                flush=True,
            )
    norm_wall = time.perf_counter() - norm_started
    write_started = time.perf_counter()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_name(f".{data_path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, design, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, data_path)
    write_wall = time.perf_counter() - write_started
    checksum_started = time.perf_counter()
    data_sha256 = sha256_file(data_path)
    checksum_wall = time.perf_counter() - checksum_started
    overall_wall = time.perf_counter() - overall_started
    write_manifest(
        manifest_path,
        {
            "artifact": "figure1-cs-design-cache",
            "artifact_role": "regenerable-performance-cache-not-scientific-result",
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "training_data": str(Path(training_path).resolve()),
            "training_data_sha256": sha256_file(training_path),
            "trial": trial,
            "sample_count": maximum,
            "order": order,
            "basis_size": expected_basis_size,
            "basis_fingerprint_sha256": polynomial_basis_fingerprint(
                dimension, "hyperbolic-cross", order
            ),
            "array_shape": list(design.shape),
            "array_dtype": str(design.dtype),
            "data": str(data_path.relative_to(repository)),
            "data_sha256": data_sha256,
            "stage_timings": {
                "registered_training_load": data_load_wall,
                "canonicalization": canonical_wall,
                "polynomial_design_construction": construction_wall,
                "spectral_norm_prefix_cache": norm_wall,
                "data_write_and_fsync": write_wall,
                "data_sha256": checksum_wall,
            },
            "construction_wall_seconds": construction_wall,
            "spectral_norm_cache": {
                **norm_options,
                "entries": spectral_norms,
            },
            "random_seeds": {},
            "process_workers": 1,
            "blas_threads": blas_threads,
            "workers": blas_threads,
            "wall_seconds": overall_wall,
        },
        repository=repository,
        environment_specification="environment/derecho.yml",
    )
    return data_path, manifest_path


def load_design_cache(
    config: Figure1Config,
    repository: str | Path,
    training_path: str | Path,
    directory: str | Path,
    *,
    trial: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[float, float]], dict]:
    repository = Path(repository).resolve()
    data_path, manifest_path = cache_paths(directory, trial)
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing q=23 design cache for trial {trial:03d}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "artifact": "figure1-cs-design-cache",
        "config_sha256": sha256_file(config.path),
        "parameter_box_sha256": config.parameter_box.fingerprint,
        "training_data_sha256": sha256_file(training_path),
        "trial": trial,
        "sample_count": int(config.experiment["train_size_max"]),
        "order": int(config.cs["hyperbolic_cross_order"]),
        "basis_size": int(config.cs["expected_basis_size"]),
        "array_shape": [
            int(config.experiment["train_size_max"]),
            int(config.cs["expected_basis_size"]),
        ],
        "array_dtype": "float64",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Figure 1 design-cache field {key!r} is stale")
    if manifest.get("data_sha256") != sha256_file(data_path):
        raise ValueError("Figure 1 design-cache checksum mismatch")
    dimension = config.parameter_box.dimension
    order = int(config.cs["hyperbolic_cross_order"])
    basis_size = int(config.cs["expected_basis_size"])
    indices = hyperbolic_cross_indices(dimension, order)
    expected_fingerprint = polynomial_basis_fingerprint(
        dimension, "hyperbolic-cross", order
    )
    if manifest.get("basis_fingerprint_sha256") != expected_fingerprint:
        raise ValueError("Figure 1 design-cache basis fingerprint mismatch")
    design = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if design.shape != (int(config.experiment["train_size_max"]), basis_size):
        raise ValueError("Figure 1 design-cache shape mismatch")
    weights = intrinsic_legendre_weights(indices)
    norm_options = pdr_spectral_norm_options(config)
    cached_norms = manifest.get("spectral_norm_cache", {})
    for key, expected in norm_options.items():
        if cached_norms.get(key) != expected:
            raise ValueError(f"Figure 1 design-cache norm setting {key!r} is stale")
    norm_entries = cached_norms.get("entries", [])
    if [int(entry.get("sample_count", -1)) for entry in norm_entries] != list(config.train_sizes):
        raise ValueError("Figure 1 design-cache spectral-norm prefixes are incomplete")
    norms = {
        int(entry["sample_count"]): (
            float(entry["spectral_norm"]),
            float(entry["spectral_norm_raw_estimate"]),
        )
        for entry in norm_entries
    }
    safety = norm_options["spectral_norm_safety_factor"]
    if any(
        not np.isfinite(value) or not np.isfinite(raw)
        or value <= 0.0 or raw <= 0.0 or value != safety * raw
        for value, raw in norms.values()
    ):
        raise ValueError("Figure 1 design-cache spectral norms are invalid")
    return design, indices, weights, norms, manifest
