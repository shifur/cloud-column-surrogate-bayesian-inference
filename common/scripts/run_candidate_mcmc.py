#!/usr/bin/env python3
"""Plan, report, or run the 20 paired candidate-posterior calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.mcmc.production import production_status
from cloud_column_reproduction.provenance import sha256_file


def _configs(directory: Path, trials: list[int] | None) -> list[tuple[int, Path]]:
    selected = list(range(20)) if trials is None else sorted(set(trials))
    if any(trial < 0 or trial >= 20 for trial in selected):
        raise ValueError("Candidate trial must be in [0, 19]")
    records = []
    for trial in selected:
        path = directory / f"mcmc_candidate_trial_{trial:03d}.toml"
        config = load_config(path, validate_registered_artifacts=True)
        config.require_production_ready()
        if config.execution_cases() != (
            "full_except_acc",
        ) or config.execution_methods() != ("dnn", "pdr"):
            raise ValueError(f"Candidate execution scope changed: {path}")
        if int(config.raw["study"]["candidate_trial"]) != trial:
            raise ValueError(f"Candidate trial identity mismatch: {path}")
        records.append((trial, path))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "status", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config-directory", type=Path, default=ROOT / "configs")
        command.add_argument("--trial", type=int, action="append")
        if name == "run":
            command.add_argument("--progress-every", type=int, default=1000)
            command.add_argument("--maximum-block-rounds", type=int)
    arguments = parser.parse_args()
    records = _configs(arguments.config_directory.resolve(), arguments.trial)
    if arguments.command == "plan":
        print(
            json.dumps(
                {
                    "study": "paired-m5000-candidate-posterior-variability",
                    "trial_count": len(records),
                    "model_count": 2 * len(records),
                    "chain_unit_count": 4 * len(records),
                    "case": "full_except_acc",
                    "records": [
                        {
                            "trial": trial,
                            "config": str(path),
                            "config_sha256": sha256_file(path),
                        }
                        for trial, path in records
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "status":
        print(
            json.dumps(
                {
                    "records": [
                        {"trial": trial, **production_status(load_config(path))}
                        for trial, path in records
                    ]
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for trial, path in records:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/run_mcmc_surrogates.py"),
            "--config",
            str(path),
            "--case",
            "full_except_acc",
            "--target-burn-in",
            "30000",
            "--target-production",
            "300000",
            "--progress-every",
            str(arguments.progress_every),
        ]
        if arguments.maximum_block_rounds is not None:
            command.extend(
                ["--maximum-block-rounds", str(arguments.maximum_block_rounds)]
            )
        print(json.dumps({"candidate_trial": trial, "command": command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
