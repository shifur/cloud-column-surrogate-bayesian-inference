#!/usr/bin/env python3
"""Run the optimized two-method GPU study with paired ensemble microbatches."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.mcmc.production import (
    ProductionUnit,
    run_prefix_diagnostics,
    run_unit,
)
from cloud_column_reproduction.mcmc.runtime import build_runtime_evaluator
from cloud_column_reproduction.mcmc.matched_timing import MatchedTiming, exclusive_case, timing_summary
from cloud_column_reproduction.provenance import sha256_file, scientific_source_snapshot


DEFAULT_CONFIG = REPOSITORY / "configs/mcmc.toml"
METHODS = ("dnn", "pdr")


def _counts(root: Path, unit: ProductionUnit) -> tuple[int, int]:
    path = root / unit.relative_directory / "checkpoint.json"
    if not path.is_file():
        return 0, 0
    record = json.loads(path.read_text(encoding="utf-8"))
    return int(record["completed_burn_in"]), int(record["completed_production"])


def _shared_runtime(config, case: str, method: str, window_seconds: float):
    if method == "dnn":
        section = config.raw["runtime"]["dnn"]
        return build_runtime_evaluator(
            config,
            case_name=case,
            method=method,
            dnn_device=section["device"],
            dnn_backend=section["backend"],
            dnn_validation_audit=None,
            gpu_microbatch_requests=2,
            gpu_microbatch_window_seconds=window_seconds,
            include_prediction_blobs=False,
        )
    section = config.raw["runtime"]["pdr"]
    return build_runtime_evaluator(
        config,
        case_name=case,
        method=method,
        pdr_device=section["device"],
        pdr_backend=section["backend"],
        pdr_prune_exact_zeros=bool(section["prune_exact_zeros"]),
        pdr_validation_audit=None,
        gpu_microbatch_requests=2,
        gpu_microbatch_window_seconds=window_seconds,
        include_prediction_blobs=False,
    )


def _advance_pair(
    config,
    root: Path,
    case: str,
    method: str,
    runtime,
    *,
    progress_every: int,
    maximum_blocks: dict[int, int],
) -> None:
    units = tuple(ProductionUnit(case, ensemble, method) for ensemble in (0, 1))
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix=f"{method}-ensemble"
    ) as executor:
        futures = [
            executor.submit(
                run_unit,
                config,
                unit,
                output_root=root,
                dnn_device="cuda",
                maximum_blocks=maximum_blocks[unit.ensemble_id],
                production_authorized=True,
                progress_every_transitions=progress_every,
                runtime_override=runtime,
                defer_finalization=True,
            )
            for unit in units
        ]
        results = [future.result() for future in futures]
    print(json.dumps({
        "method": method,
        "paired_ensemble_block_results": results,
        "microbatch": next(
            (
                resource.metadata
                for resource in runtime.resources
                if hasattr(resource, "metadata")
                and "microbatch_policy" in resource.metadata
            ),
            {},
        ),
    }, indent=2, sort_keys=True), flush=True)


def _advance_single(
    config,
    root: Path,
    unit: ProductionUnit,
    runtime,
    *,
    progress_every: int,
    maximum_blocks: int,
) -> None:
    run_unit(
        config,
        unit,
        output_root=root,
        dnn_device="cuda",
        maximum_blocks=maximum_blocks,
        production_authorized=True,
        progress_every_transitions=progress_every,
        runtime_override=runtime,
        defer_finalization=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="override the fresh chain root (intended for isolated smoke/comparison runs)",
    )
    parser.add_argument(
        "--case",
        choices=("full_except_acc", "radiation_only"),
        default="full_except_acc",
    )
    parser.add_argument("--target-burn-in", type=int, default=30_000)
    parser.add_argument("--target-production", type=int, default=300_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--microbatch-window-us", type=float, default=2_000.0)
    parser.add_argument(
        "--blocks-per-unit-advance",
        type=int,
        help=(
            "number of consecutive 1,000-transition atomic blocks processed in "
            "one run_unit session (defaults to the configured 50-block policy)"
        ),
    )
    parser.add_argument("--maximum-block-rounds", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true",
                        help="Keep isolated timing blocks; skip posterior export and diagnostics")
    arguments = parser.parse_args()

    config = load_config(arguments.config, validate_registered_artifacts=True)
    if config.raw["study"].get("optimized_gpu_30k_300k") is not True:
        raise ValueError("This driver requires the optimized GPU study contract")
    if config.raw["runtime"]["pdr"]["backend"] != "cupy-cached-sparse-float32-cuda":
        raise ValueError("Matched timing requires upgrade-runtime and a matched FP32 config")
    block = int(config.raw["sampler"]["checkpointing"]["block_transitions"])
    configured_burn = int(config.raw["sampler"]["burn_in_steps"])
    configured_production = int(config.raw["sampler"]["production_steps"])
    if not 0 <= arguments.target_burn_in <= configured_burn:
        raise ValueError("Target burn-in exceeds the configured 30k")
    if not 0 <= arguments.target_production <= configured_production:
        raise ValueError("Target production exceeds the configured 300k")
    if (
        arguments.target_burn_in % block
        or arguments.target_production % block
        or arguments.progress_every < 1
    ):
        raise ValueError("Targets must be complete blocks and progress must be positive")
    expected_window = float(
        config.raw["runtime"]["execution"]["microbatch_window_us"]
    )
    if arguments.microbatch_window_us != expected_window:
        raise ValueError(
            "Microbatch window differs from the measured config policy"
        )
    if arguments.maximum_block_rounds is not None and arguments.maximum_block_rounds < 1:
        raise ValueError("Maximum block rounds must be positive")
    configured_blocks_per_advance = int(
        config.raw["runtime"]["execution"].get("blocks_per_unit_advance", 1)
    )
    blocks_per_advance = (
        configured_blocks_per_advance
        if arguments.blocks_per_unit_advance is None
        else arguments.blocks_per_unit_advance
    )
    if blocks_per_advance < 1:
        raise ValueError("Blocks per unit advance must be positive")

    root = (
        arguments.output_directory.resolve()
        if arguments.output_directory is not None
        else (config.path.parent.parent / config.raw["artifacts"]["root"]).resolve()
    )
    units = tuple(
        ProductionUnit(arguments.case, ensemble, method)
        for method in METHODS for ensemble in (0, 1)
    )
    print(json.dumps({
        "status": "optimized-surrogate-gpu-study-ready",
        "config": str(config.path),
        "config_sha256": sha256_file(config.path),
        "output_root": str(root),
        "case": arguments.case,
        "methods": list(METHODS),
        "ensemble_count_per_method": 2,
        "walkers_per_ensemble": 32,
        "target_burn_in": arguments.target_burn_in,
        "target_production": arguments.target_production,
        "schedule":
            "alternating method order by round; both methods use concurrent paired ensembles",
        "execution_contract": {
            "dnn": "two concurrent ensembles; one persistent compiled CUDA runtime",
            "pdr": "two concurrent ensembles; one persistent cached-sparse CUDA runtime",
        },
        "surrogate_arithmetic": "float32 for both; float64 likelihood and sampler coordinates",
        "microbatch_window_us": arguments.microbatch_window_us,
        "atomic_checkpoint_transitions": block,
        "blocks_per_unit_advance": blocks_per_advance,
        "transitions_per_unit_session": block * blocks_per_advance,
        "backends": {
            method: config.raw["runtime"][method]["backend"]
            for method in METHODS
        },
        "existing_progress": {
            str(unit.relative_directory): {
                "burn_in": _counts(root, unit)[0],
                "production": _counts(root, unit)[1],
            }
            for unit in units
        },
    }, indent=2, sort_keys=True), flush=True)
    if arguments.plan_only:
        return 0

    if arguments.target_production and arguments.target_burn_in != configured_burn:
        raise ValueError("Production targets require all configured burn-in transitions")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Optimized surrogate study requires CUDA")
    with exclusive_case(root, arguments.case):
        return _execute(config, root, units, arguments, block, blocks_per_advance)


def _execute(config, root, units, arguments, block, blocks_per_advance):
    construction_started = time.perf_counter()
    shared_runtimes = {
        method: _shared_runtime(
            config,
            arguments.case,
            method,
            arguments.microbatch_window_us * 1.0e-6,
        )
        for method in METHODS
    }
    construction_seconds = time.perf_counter() - construction_started
    for runtime in shared_runtimes.values():
        atexit.register(runtime.close)

    # Precompile every possible DE/Snooker singleton/paired/startup shape.
    # No extra MCMC transitions are discarded; the configured burn-in is timed.
    import platform
    import emcee
    import cupy
    import torch
    setup = {"runtime_construction_seconds_both_methods": construction_seconds,
             "python": sys.version, "platform": platform.platform(),
             "torch": torch.__version__, "cupy": cupy.__version__,
             "emcee": emcee.__version__, "numpy": np.__version__,
             "cuda_runtime": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(0), "argv": sys.argv}
    for method, runtime in shared_runtimes.items():
        started = time.perf_counter()
        predictor = runtime.evaluator._predict_canonical
        for rows in (8, 16, 24, 32, 40, 48, 64):
            predictor(np.zeros((rows, config.parameter_box.dimension)))
        setup[method] = {
            "prewarm_seconds": time.perf_counter() - started,
            "batch_sizes": [8, 16, 24, 32, 40, 48, 64],
            "bindings": runtime.binding_hashes,
            "metadata": vars(runtime.metadata),
            "resources": [r.metadata for r in runtime.resources if hasattr(r, "metadata")],
        }
    setup_path = root / arguments.case / "matched_timing" / f"setup_{time.time_ns()}.json"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(json.dumps(setup, indent=2, sort_keys=True) + "\n")
    clocks = {
        method: MatchedTiming(root, arguments.case, method,
                              config_sha256=sha256_file(config.path),
                              runtime={
                                  "bindings": runtime.binding_hashes,
                                  "torch": torch.__version__, "cupy": cupy.__version__,
                                  "emcee": emcee.__version__, "numpy": np.__version__,
                                  "gpu": torch.cuda.get_device_name(0),
                                  "runtime_source_sha256": scientific_source_snapshot(
                                      REPOSITORY, roots=("src", "scripts/run_mcmc_surrogates.py")
                                  )["sha256"],
                              },
                              advance_blocks=blocks_per_advance)
        for method, runtime in shared_runtimes.items()
    }
    round_number = 0
    while True:
        pending = [
            unit for unit in units
            if (
                _counts(root, unit)[0] < arguments.target_burn_in
                or _counts(root, unit)[1] < arguments.target_production
            )
        ]
        if not pending:
            break
        if (
            arguments.maximum_block_rounds is not None
            and round_number >= arguments.maximum_block_rounds
        ):
            break
        round_number += 1
        print(f"GPU block round {round_number}: {len(pending)} units pending", flush=True)
        for method in (METHODS if round_number % 2 else METHODS[::-1]):
            method_pending = tuple(unit for unit in pending if unit.method == method)
            block_limits = {}
            for unit in method_pending:
                completed_burn, completed_production = _counts(root, unit)
                remaining = (
                    max(0, arguments.target_burn_in - completed_burn)
                    + max(0, arguments.target_production - completed_production)
                )
                if remaining % block:
                    raise RuntimeError("Remaining target is not block aligned")
                block_limits[unit.ensemble_id] = min(
                    blocks_per_advance, remaining // block
                )
            if not method_pending:
                continue
            print(f"Advancing {method.upper()} with concurrent ensemble policy", flush=True)
            with clocks[method].measure():
                if len(method_pending) == 2:
                    _advance_pair(
                        config, root, arguments.case, method, shared_runtimes[method],
                        progress_every=arguments.progress_every,
                        maximum_blocks=block_limits,
                    )
                else:
                    _advance_single(
                        config, root, method_pending[0], shared_runtimes[method],
                        progress_every=arguments.progress_every,
                        maximum_blocks=block_limits[method_pending[0].ensemble_id],
                    )

    for method, runtime in shared_runtimes.items():
        setup[method]["resources_after_sampling"] = [
            r.metadata for r in runtime.resources if hasattr(r, "metadata")]
    setup_path.write_text(json.dumps(setup, indent=2, sort_keys=True) + "\n")

    # NetCDF export and diagnostics are outside every sampling elapsed clock.
    for unit in units:
        if not arguments.timing_only and _counts(root, unit) == (30000, 300000):
            run_unit(config, unit, output_root=root, maximum_blocks=0,
                     production_authorized=True,
                     runtime_override=shared_runtimes[unit.method])
    summaries = {method: timing_summary(root, arguments.case, method) for method in METHODS}
    summary_path = root / arguments.case / "matched_timing" / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matched_timing": summaries}, indent=2, sort_keys=True), flush=True)

    for runtime in shared_runtimes.values():
        runtime.close()

    complete = all(
        _counts(root, unit)[0] >= arguments.target_burn_in
        and _counts(root, unit)[1] >= arguments.target_production
        for unit in units
    )
    diagnostics = {}
    if not arguments.timing_only and complete and arguments.target_production in config.raw["sampler"]["checkpointing"]["diagnostic_prefixes"]:
        for method in METHODS:
            paths = run_prefix_diagnostics(
                config,
                likelihood_case=arguments.case,
                method=method,
                prefix=arguments.target_production,
                output_root=root,
            )
            diagnostics[method] = [str(path) for path in paths]
    print(json.dumps({
        "status": "target-complete" if complete else "bounded-rounds-complete",
        "completed": {
            str(unit.relative_directory): {
                "burn_in": _counts(root, unit)[0],
                "production": _counts(root, unit)[1],
            }
            for unit in units
        },
        "diagnostics": diagnostics,
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
