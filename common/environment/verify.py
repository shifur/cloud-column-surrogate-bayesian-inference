"""Verify the curated cloud-column runtime without emitting local paths."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
REQUIRED_DISTRIBUTIONS = (
    "cloud-column-study-optimized",
    "cupy-cuda12x",
    "emcee",
    "arviz",
    "h5netcdf",
    "h5py",
    "matplotlib",
    "numpy",
    "qmcpy",
    "scikit-learn",
    "scipy",
    "Tasmanian",
    "torch",
    "xarray",
)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Required distribution is not installed: {name}") from error


def _first_line(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(f"Required executable is unavailable: {command[0]}")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    arguments = parser.parse_args()

    import cupy
    import numpy
    import torch

    packages = {name: _version(name) for name in REQUIRED_DISTRIBUTIONS}
    try:
        cupy_device_count = int(cupy.cuda.runtime.getDeviceCount())
        cupy_probe_error = None
    except Exception as error:  # Login nodes may not expose the CUDA driver.
        cupy_device_count = 0
        cupy_probe_error = type(error).__name__

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "packages": packages,
        "gfortran": _first_line(["gfortran", "--version"]),
        "make": _first_line(["make", "--version"]),
        "sccm_executable": (
            REPOSITORY / "vendor" / "sccm" / "cloud_column_model.x"
        ).is_file(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cupy_device_count": cupy_device_count,
    }
    if cupy_probe_error is not None:
        report["cupy_probe_error"] = cupy_probe_error

    if not report["sccm_executable"]:
        raise RuntimeError("The SCCM executable has not been built")
    if arguments.require_gpu:
        if not report["torch_cuda_available"]:
            raise RuntimeError("PyTorch cannot access a CUDA device")
        if report["cupy_device_count"] < 1:
            raise RuntimeError("CuPy cannot access a CUDA device")
        left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        torch_value = float((left @ left.T).sum().cpu())
        cupy_left = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
        cupy_value = float(cupy.asnumpy((cupy_left @ cupy_left.T).sum()))
        if not numpy.isfinite(torch_value) or not numpy.isfinite(cupy_value):
            raise RuntimeError("A CUDA matrix-product smoke test was non-finite")
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_matmul_sum": torch_value,
            "cupy_matmul_sum": cupy_value,
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
