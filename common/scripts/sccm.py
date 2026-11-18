#!/usr/bin/env python3
"""SCCM regression and single-run commands for the source build."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.sccm import runner_from_repository
from cloud_column_reproduction.figure1.config import load_config


TRUE_PARAMETERS = np.asarray(
    [200.0, 0.3, 400.0, 0.4, 0.5, 0.5, 0.5, 0.2, 0.4, 1.0e-3, 6.0e-4]
)


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("verify", help="compare a true-parameter run with the fixture")
    run_parser = subcommands.add_parser("run", help="run one 11-parameter forward evaluation")
    run_parser.add_argument("parameters", nargs=11, type=float)
    return command_parser


def main() -> int:
    arguments = parser().parse_args()
    config = load_config(REPOSITORY / "configs" / "figure1.toml")
    runner = runner_from_repository(
        REPOSITORY,
        prebuilt=False,
        parameter_box=config.parameter_box,
    )
    if arguments.command == "run":
        output = runner.run(arguments.parameters)
        print(" ".join(f"{value:.10f}" for value in output))
        return 0

    expected = np.asarray(
        np.loadtxt(REPOSITORY / "tests" / "fixtures" / "sccm_true_output.txt"),
        dtype=float,
    ).reshape(-1)
    actual = runner.run(TRUE_PARAMETERS)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-10)
    print("SCCM regression verified")
    print("diagnostics@120min:", " ".join(f"{value:.10f}" for value in actual[18:24]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
