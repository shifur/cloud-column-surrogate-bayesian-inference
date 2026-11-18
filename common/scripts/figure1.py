#!/usr/bin/env python3
"""Validate, plan, and plot the controlled Figure 1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.figure1.config import load_config, work_plan
from cloud_column_reproduction.figure1.metrics import (
    read_metrics,
    summarize_metrics,
    validate_metrics,
    write_summary,
)
from cloud_column_reproduction.provenance import (
    sha256_file,
    write_manifest,
)


DEFAULT_CONFIG = REPOSITORY / "configs" / "figure1.toml"


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate", help="validate the frozen experiment specification")
    subcommands.add_parser("plan", help="print the production workload implied by the config")
    training_parser = subcommands.add_parser(
        "generate-training", help="generate one deterministic nested SCCM training pool"
    )
    training_parser.add_argument("--trial", type=int, required=True)
    training_parser.add_argument(
        "--samples",
        type=int,
        help="number of samples; defaults to the configured maximum of 5000",
    )
    training_parser.add_argument("--workers", type=int, default=1)
    training_parser.add_argument(
        "--output-directory", type=Path, default=REPOSITORY / "data" / "figure1" / "training"
    )
    evaluation_parser = subcommands.add_parser(
        "generate-evaluation",
        help="generate the configured primary evaluation set",
    )
    evaluation_parser.add_argument("--workers", type=int, default=1)
    evaluation_parser.add_argument(
        "--grid-only",
        action="store_true",
        help="save evaluation inputs without launching SCCM calls",
    )
    evaluation_parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY / "data" / "figure1" / "production" / "evaluation",
    )
    validation_parser = subcommands.add_parser(
        "generate-validation", help="generate the fixed DNN model-selection set"
    )
    validation_parser.add_argument("--workers", type=int, default=1)
    validation_parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="save validation inputs without launching SCCM calls",
    )
    validation_parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY / "data" / "figure1" / "validation",
    )
    plot_parser = subcommands.add_parser("plot", help="plot validated raw paired-trial metrics")
    plot_parser.add_argument("--metrics", type=Path, required=True)
    plot_parser.add_argument("--output", type=Path)
    suite_parser = subcommands.add_parser(
        "generate-training-suite", help="generate or resume every maximum-size training pool"
    )
    suite_parser.add_argument("--workers", type=int, default=1)
    suite_parser.add_argument(
        "--output-directory", type=Path, default=REPOSITORY / "data" / "figure1" / "training"
    )
    return command_parser


def main() -> int:
    arguments = parser().parse_args()
    config = load_config(arguments.config)

    if arguments.command == "validate":
        print(f"valid: {config.path}")
        print(f"status: {config.raw['status']}")
        return 0

    if arguments.command == "plan":
        print(json.dumps(work_plan(config), indent=2, sort_keys=True))
        return 0

    if arguments.command == "generate-training":
        from cloud_column_reproduction.figure1.data import generate_training_pool

        sample_count = arguments.samples or int(config.experiment["train_size_max"])
        data_path, manifest_path = generate_training_pool(
            config,
            REPOSITORY,
            arguments.output_directory,
            trial=arguments.trial,
            sample_count=sample_count,
            workers=arguments.workers,
            prebuilt=False,
        )
        print(data_path)
        print(manifest_path)
        return 0

    if arguments.command == "generate-evaluation":
        from cloud_column_reproduction.figure1.evaluation import (
            generate_primary_evaluation_data,
        )

        data_path, manifest_path = generate_primary_evaluation_data(
            config,
            REPOSITORY,
            arguments.output_directory,
            workers=arguments.workers,
            prebuilt=False,
            inputs_only=arguments.grid_only,
        )
        print(data_path)
        print(manifest_path)
        return 0

    if arguments.command == "generate-validation":
        from cloud_column_reproduction.figure1.evaluation import generate_validation_data

        data_path, manifest_path = generate_validation_data(
            config,
            REPOSITORY,
            arguments.output_directory,
            workers=arguments.workers,
            prebuilt=False,
            inputs_only=arguments.inputs_only,
        )
        print(data_path)
        print(manifest_path)
        return 0

    if arguments.command == "generate-training-suite":
        from cloud_column_reproduction.figure1.data import generate_training_pool
        from cloud_column_reproduction.invariants import audit_manifests

        maximum = int(config.experiment["train_size_max"])
        for trial in range(int(config.experiment["trials"])):
            destination = arguments.output_directory / f"trial_{trial:03d}_n{maximum:05d}.npz"
            manifest = destination.with_name(f"{destination.stem}.manifest.json")
            if destination.is_file() and manifest.is_file():
                audit_manifests(
                    [manifest], config.parameter_box, config.parameter_box_path
                )
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if payload.get("data_sha256") != sha256_file(destination):
                    raise ValueError(f"Existing training pool checksum mismatch: {destination}")
                print(f"existing: {destination}")
                continue
            generate_training_pool(
                config,
                REPOSITORY,
                arguments.output_directory,
                trial=trial,
                sample_count=maximum,
                workers=arguments.workers,
                prebuilt=False,
            )
        return 0

    rows = validate_metrics(read_metrics(arguments.metrics), config)
    summary = summarize_metrics(rows, config)
    from cloud_column_reproduction.figure1.plot import plot_learning_curves

    configured_output = REPOSITORY / config.plot["output"]
    output = arguments.output or configured_output
    output = output.resolve()
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    write_summary(summary_path, summary, config)
    plot_learning_curves(summary, config, output)
    write_manifest(
        manifest_path,
        {
            "artifact": "manuscript-figure-1",
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(REPOSITORY)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"base_seed": int(config.experiment["base_seed"])},
            "parameter_names": list(config.parameters["names"]),
            "parameter_lower": list(config.parameters["lower"]),
            "parameter_upper": list(config.parameters["upper"]),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(REPOSITORY)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "raw_metrics": str(arguments.metrics.resolve()),
            "raw_metrics_sha256": sha256_file(arguments.metrics),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "figure": str(output),
            "figure_sha256": sha256_file(output),
            "standard_deviation_ddof": int(config.plot["standard_deviation_ddof"]),
        },
        repository=REPOSITORY,
    )
    print(output)
    print(summary_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
