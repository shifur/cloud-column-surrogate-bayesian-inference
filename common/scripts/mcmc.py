#!/usr/bin/env python3
"""Validate, run, resume, and diagnose the controlled production MCMC study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.mcmc.production import (
    ProductionUnit,
    planned_units,
    production_status,
    run_prefix_diagnostics,
    run_unit,
    workload_summary,
)


DEFAULT_CONFIG = REPOSITORY / "configs" / "mcmc.toml"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the integration draft and list blockers")
    commands.add_parser("plan", help="list the complete production unit schedule and workload")
    status = commands.add_parser("status", help="report resumable progress, timing, and within-unit ETA")
    status.add_argument("--output-directory", type=Path)
    diagnostic = commands.add_parser("diagnose-prefix", help="diagnose two independent ensembles at a common prefix")
    diagnostic.add_argument("--case", choices=("radiation_only", "full_except_acc"), required=True)
    diagnostic.add_argument("--method", choices=("sccm", "dnn", "pdr"), required=True)
    diagnostic.add_argument("--prefix", type=int, required=True)
    diagnostic.add_argument("--output-directory", type=Path)
    production = commands.add_parser(
        "production-gate", help="fail unless every production field is resolved and approved"
    )
    production.set_defaults(production_gate=True)
    run = commands.add_parser("run-unit", help="advance one resumable case/method/ensemble unit")
    run.add_argument("--case", choices=("radiation_only", "full_except_acc"), required=True)
    run.add_argument("--method", choices=("sccm", "dnn", "pdr"), required=True)
    run.add_argument("--ensemble", type=int, required=True)
    run.add_argument("--output-directory", type=Path)
    run.add_argument("--dnn-device", default="cpu")
    run.add_argument(
        "--dnn-backend",
        choices=(
            "torch-baseline",
            "torch-baseline-cpu",
            "torch-baseline-cuda",
            "torch-persistent-cuda",
            "torch-compile-cuda",
        ),
    )
    run.add_argument("--pdr-workers", type=int, default=1)
    run.add_argument("--pdr-device", choices=("cpu", "cuda"))
    run.add_argument(
        "--pdr-backend",
        choices=(
            "numpy-cpu",
            "torch-cuda",
            "cupy-fused-cuda",
            "cupy-zero-skip-cuda",
            "cupy-cached-sparse-cuda",
            "cupy-cached-sparse-float32-cuda",
        ),
    )
    run.add_argument(
        "--pdr-prune-exact-zeros", action=argparse.BooleanOptionalAction, default=None
    )
    run.add_argument("--sccm-workers", type=int, default=32)
    run.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print in-block MCMC diagnostics every N transitions (default: 100)",
    )
    run.add_argument("--max-blocks", type=int, help="optional bounded timing or scheduler advance")
    return result


def main() -> int:
    arguments = parser().parse_args()
    read_only_without_evaluators = arguments.command in {
        "validate", "plan", "status", "diagnose-prefix"
    }
    config = load_config(
        arguments.config,
        validate_registered_artifacts=not read_only_without_evaluators,
    )
    if arguments.command == "validate":
        print(f"Controlled MCMC {config.study_role} configuration is structurally valid")
        print("parameter_box_fingerprint:", config.parameter_box.fingerprint)
        print("production_blockers:")
        for blocker in config.unresolved_fields():
            print(" -", blocker)
        try:
            load_config(arguments.config, validate_registered_artifacts=True)
        except ValueError as error:
            print(" - external_artifacts:", error)
        return 0
    if arguments.command == "plan":
        print(json.dumps({
            "workload": workload_summary(config),
            "units": [
                {"likelihood_case": unit.likelihood_case, "ensemble_id": unit.ensemble_id,
                 "method": unit.method, "relative_directory": str(unit.relative_directory)}
                for unit in planned_units(config)
            ],
        }, indent=2, sort_keys=True))
        return 0
    if arguments.command == "status":
        print(json.dumps(production_status(config, output_root=arguments.output_directory), indent=2, sort_keys=True))
        return 0
    if arguments.command == "diagnose-prefix":
        paths = run_prefix_diagnostics(
            config,
            likelihood_case=arguments.case,
            method=arguments.method,
            prefix=arguments.prefix,
            output_root=arguments.output_directory,
        )
        print(json.dumps({"data": str(paths[0]), "manifest": str(paths[1])}, indent=2))
        return 0
    if arguments.command == "production-gate":
        config.require_production_ready()
        print("Controlled MCMC configuration is production-ready")
        return 0
    if arguments.command == "run-unit":
        def _terminate_at_block_attempt(signum, frame):
            del signum, frame
            raise KeyboardInterrupt("termination signal received; preserving partial-block timing")

        signal.signal(signal.SIGTERM, _terminate_at_block_attempt)
        if arguments.max_blocks is not None and arguments.max_blocks < 1:
            raise ValueError("--max-blocks must be positive")
        if arguments.progress_every < 1:
            raise ValueError("--progress-every must be positive")
        result = run_unit(
            config,
            ProductionUnit(arguments.case, arguments.ensemble, arguments.method),
            output_root=arguments.output_directory,
            dnn_device=arguments.dnn_device,
            dnn_backend=arguments.dnn_backend,
            dnn_validation_audit=None,
            pdr_workers=arguments.pdr_workers,
            pdr_device=arguments.pdr_device,
            pdr_backend=arguments.pdr_backend,
            pdr_prune_exact_zeros=arguments.pdr_prune_exact_zeros,
            pdr_validation_audit=None,
            sccm_workers=arguments.sccm_workers,
            prebuilt_sccm=False,
            maximum_blocks=arguments.max_blocks,
            production_authorized=True,
            progress_every_transitions=arguments.progress_every,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
