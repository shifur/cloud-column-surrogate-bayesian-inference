#!/usr/bin/env python3
"""Plan or build exact q=23 Figure 1 design caches in parallel across trials."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import os
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.figure1.config import load_config
from cloud_column_reproduction.figure1.design_cache import build_design_cache, cache_paths
from cloud_column_reproduction.figure1.fitting import _load_registered_data
from cloud_column_reproduction.provenance import sha256_file, write_manifest


def _build_trial(
    config_path: Path, training_directory: Path, output_directory: Path,
    trial: int, blas_threads: int,
) -> tuple[str, str]:
    from threadpoolctl import threadpool_limits

    config = load_config(config_path)
    maximum = int(config.experiment["train_size_max"])
    training = training_directory / f"trial_{trial:03d}_n{maximum:05d}.npz"
    with threadpool_limits(limits=blas_threads):
        data, manifest = build_design_cache(
            config, REPOSITORY, training, output_directory, trial=trial,
            blas_threads=blas_threads,
        )
    return str(data), str(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "validate", "build"))
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/figure1.toml")
    parser.add_argument("--training-directory", type=Path, default=REPOSITORY / "data/figure1/production/training")
    parser.add_argument("--output-directory", type=Path, default=REPOSITORY / "data/figure1/production/design_cache_q23_v1")
    parser.add_argument("--trial", type=int)
    parser.add_argument(
        "--workers", type=int,
        help="independent trial caches to construct concurrently (default: config policy)",
    )
    parser.add_argument(
        "--blas-threads-per-worker", type=int,
        help="BLAS threads in each process (default: CPU count divided by workers)",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    cache_policy = config.cs.get("offline_cache", {})
    if cache_policy.get("enabled") is not True:
        raise ValueError("This builder requires the authoritative q=23 offline-cache contract")
    if arguments.workers is None:
        arguments.workers = int(cache_policy["default_processes"])
    if arguments.workers < 1:
        raise ValueError("Design-cache workers must be positive")
    blas_threads = (
        max(1, (os.cpu_count() or 1) // arguments.workers)
        if arguments.blas_threads_per_worker is None
        else arguments.blas_threads_per_worker
    )
    if blas_threads < 1:
        raise ValueError("BLAS threads per cache worker must be positive")
    trials = range(int(config.experiment["trials"])) if arguments.trial is None else (arguments.trial,)
    if any(trial < 0 or trial >= int(config.experiment["trials"]) for trial in trials):
        raise ValueError("Design-cache trial is outside the configured range")
    trials = tuple(trials)
    bytes_per_trial = int(config.experiment["train_size_max"]) * int(config.cs["expected_basis_size"]) * 8
    print(json.dumps({
        "order": int(config.cs["hyperbolic_cross_order"]),
        "basis_size": int(config.cs["expected_basis_size"]),
        "rows_per_trial": int(config.experiment["train_size_max"]),
        "spectral_norm_prefixes_per_trial": len(config.train_sizes),
        "trials": list(trials),
        "bytes_per_trial": bytes_per_trial,
        "total_bytes": bytes_per_trial * len(trials),
        "total_gib": bytes_per_trial * len(trials) / 2**30,
        "requested_parallel_workers": arguments.workers,
        "blas_threads_per_worker": blas_threads,
        "planned_total_blas_threads": arguments.workers * blas_threads,
        "maximum_concurrent_output_gib": bytes_per_trial * min(arguments.workers, len(trials)) / 2**30,
        "output_directory": str(arguments.output_directory.resolve()),
    }, indent=2), flush=True)
    if arguments.action == "plan":
        return 0
    # Keep the logical common/data path so provenance remains repository-relative.
    output_directory = arguments.output_directory.absolute()
    training_directory = arguments.training_directory.resolve()
    pending = []
    for trial in trials:
        data, manifest = cache_paths(output_directory, trial)
        if data.is_file() and manifest.is_file():
            print(f"existing design cache trial={trial:03d}: {data}", flush=True)
        elif data.exists() or manifest.exists():
            raise FileExistsError(f"Incomplete immutable cache pair for trial {trial:03d}")
        else:
            pending.append(trial)
    missing_inputs = []
    maximum = int(config.experiment["train_size_max"])
    for trial in pending:
        training = training_directory / f"trial_{trial:03d}_n{maximum:05d}.npz"
        sidecar = training.with_name(f"{training.stem}.manifest.json")
        if not training.is_file() or not sidecar.is_file():
            missing_inputs.append(str(training))
    if missing_inputs:
        rendered = "\n".join(f"  - {path}" for path in missing_inputs)
        raise FileNotFoundError(
            "Registered Figure 1 training pools must be materialized before "
            "parallel cache construction. Missing:\n" + rendered
        )
    for trial in pending:
        training = training_directory / f"trial_{trial:03d}_n{maximum:05d}.npz"
        arrays, manifest, _ = _load_registered_data(training, config)
        if int(manifest.get("trial", -1)) != trial:
            raise ValueError(
                f"Registered training trial identity mismatch for {training}"
            )
        if arrays.get("parameters", np.empty((0, 0))).shape != (maximum, 11):
            raise ValueError(f"Registered training shape mismatch for {training}")
    if pending:
        print(
            f"validated {len(pending)} registered training pools before worker launch",
            flush=True,
        )
    if arguments.action == "validate":
        print("design-cache inputs are ready", flush=True)
        return 0
    started = time.perf_counter()
    worker_count = min(arguments.workers, len(pending))
    if pending:
        print(f"building {len(pending)} q=23 trial caches with {worker_count} processes", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _build_trial, arguments.config.resolve(), training_directory,
                    output_directory, trial, blas_threads,
                ): trial
                for trial in pending
            }
            for future in as_completed(futures):
                trial = futures[future]
                data, manifest = future.result()
                print(f"completed design cache trial={trial:03d}: {data}", flush=True)
                print(manifest, flush=True)
    orchestration_wall = time.perf_counter() - started
    set_manifest = output_directory / (
        "design_cache_set.manifest.json"
        if len(trials) == int(config.experiment["trials"])
        else f"trial_{trials[0]:03d}.cache_set.manifest.json"
    )
    sources = []
    for trial in trials:
        data, manifest = cache_paths(output_directory, trial)
        if not data.is_file() or not manifest.is_file():
            raise RuntimeError(f"Design cache trial {trial:03d} did not complete")
        source_payload = json.loads(manifest.read_text(encoding="utf-8"))
        sources.append({
            "trial": trial,
            "data": str(data.relative_to(REPOSITORY)),
            "data_sha256": sha256_file(data),
            "manifest": str(manifest.relative_to(REPOSITORY)),
            "manifest_sha256": sha256_file(manifest),
            "trial_wall_seconds": float(source_payload["wall_seconds"]),
        })
    if not set_manifest.exists():
        write_manifest(
            set_manifest,
            {
                "artifact": "figure1-cs-design-cache-set",
                "artifact_role": "regenerable-parallel-performance-cache-not-scientific-result",
                "config": str(config.path.relative_to(REPOSITORY)),
                "config_sha256": sha256_file(config.path),
                "parameter_box_sha256": config.parameter_box.fingerprint,
                "parameter_box_config": str(config.parameter_box_path.relative_to(REPOSITORY)),
                "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
                "order": int(config.cs["hyperbolic_cross_order"]),
                "basis_size": int(config.cs["expected_basis_size"]),
                "trial_count": len(trials),
                "parallel_processes_used": worker_count,
                "parallel_processes_requested": arguments.workers,
                "blas_threads_per_process": blas_threads,
                "planned_total_threads": worker_count * blas_threads,
                "orchestration_wall_seconds": orchestration_wall,
                "summed_trial_wall_seconds": sum(
                    source["trial_wall_seconds"] for source in sources
                ),
                "sources": sources,
                "random_seeds": {},
                "workers": max(1, worker_count * blas_threads),
                "wall_seconds": orchestration_wall,
            },
            repository=REPOSITORY,
            environment_specification="environment/derecho.yml",
        )
    print(f"design cache set complete: {set_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
