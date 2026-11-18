#!/usr/bin/env python3
"""Evaluate a deterministic SCCM shard under an OpenMPI/HPC-X launch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.figure1.config import load_config
from cloud_column_reproduction.sccm import runner_from_repository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    inputs = np.load(arguments.inputs)["inputs"]
    indices = np.arange(rank, inputs.shape[0], size, dtype=np.int64)
    outputs = np.full((indices.size, 36), np.nan, dtype=np.float64)
    status = np.ones(indices.size, dtype=np.int8)
    messages: list[str] = []
    runner = runner_from_repository(
        REPOSITORY,
        prebuilt=False,
        parameter_box=load_config(arguments.config).parameter_box,
    )
    for local_index, input_index in enumerate(indices):
        try:
            outputs[local_index] = runner.run(inputs[input_index])
            status[local_index] = 0
            messages.append("")
        except Exception as error:
            messages.append(f"{type(error).__name__}: {error}")

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output_directory / f"rank_{rank:04d}.npz",
        indices=indices,
        outputs=outputs,
        status=status,
        messages=np.asarray(messages, dtype=str),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
