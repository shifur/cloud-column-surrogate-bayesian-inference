#!/usr/bin/env python3
"""Read-only readiness audit for the production Figure 1 RQMC rerun."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.figure1.config import load_config
from cloud_column_reproduction.figure1.dnn import DNN_ALGORITHM_REVISION
from cloud_column_reproduction.figure1.pdr import PDR_ALGORITHM_REVISION
from cloud_column_reproduction.provenance import sha256_file


DEFAULT_CONFIG = REPOSITORY / "configs" / "figure1.toml"
DEFAULT_TRAINING = REPOSITORY / "data" / "figure1" / "production" / "training"
DEFAULT_VALIDATION = (
    REPOSITORY
    / "data"
    / "figure1"
    / "production"
    / "validation"
    / "uniform_mc_n5000.npz"
)
DEFAULT_EVALUATION = (
    REPOSITORY
    / "data"
    / "figure1"
    / "production"
    / "evaluation"
    / "scrambled_sobol_r10_n16384.npz"
)
DEFAULT_OUTPUT = REPOSITORY / "outputs" / "figure1" / "production_rqmc_q23_v2"


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def pass_(self, message: str) -> None:
        print(f"PASS: {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL: {message}")

    def warn(self, message: str) -> None:
        print(f"WARN: {message}")

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.pass_(message)
        else:
            self.fail(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-directory", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--design-cache-directory", type=Path,
        help="validate every optional q=23 trial cache before production fitting",
    )
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="skip CUDA checks (not suitable before a GPU production run)",
    )
    return parser


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.stem}.manifest.json")


def _load_registered_npz(
    path: Path,
    audit: Audit,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = path.resolve()
    manifest_path = _sidecar(path)
    if not path.is_file() or not manifest_path.is_file():
        audit.fail(f"registered artifact or sidecar is missing: {path}")
        return {}, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit.require(manifest.get("status") == "complete", f"complete: {path.name}")
        audit.require(
            manifest.get("data_sha256") == sha256_file(path),
            f"checksum: {path.name}",
        )
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        audit.fail(f"cannot read {path}: {error}")
        return {}, {}
    config_name = manifest.get("config")
    config_hash = manifest.get("config_sha256")
    if isinstance(config_name, str) and isinstance(config_hash, str):
        recorded_path = REPOSITORY / config_name
        if not recorded_path.is_file() or sha256_file(recorded_path) != config_hash:
            audit.warn(
                f"{path.name}: producing config path no longer has its recorded "
                "checksum; the immutable data checksum and embedded contract are "
                "validated, and the recorded historical hash is preserved"
            )
    return arrays, manifest


def _finite(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) and bool(np.all(np.isfinite(array)))


def _row_keys(array: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"row-set input has shape {values.shape}, expected a matrix")
    dtype = np.dtype((np.void, values.dtype.itemsize * values.shape[1]))
    return values.view(dtype).reshape(-1)


def _check_config(config: Any, audit: Audit) -> None:
    expected_sizes = tuple(range(100, 5001, 100))
    audit.require(int(config.experiment["trials"]) == 20, "20 paired trials")
    audit.require(config.train_sizes == expected_sizes, "50 nested sizes from 100 to 5000")
    audit.require(config.outputs == ("PCP", "ACC", "LWP", "IWP", "OLR", "OSR"), "six canonical outputs")
    audit.require(config.is_rqmc_primary, "RQMC is the primary evaluation contract")
    audit.require(int(config.cs["hyperbolic_cross_order"]) == 23, "CS hyperbolic-cross order q=23")
    audit.require(int(config.cs["expected_basis_size"]) == 7569, "CS basis contains 7,569 terms")
    audit.require(
        config.cs.get("index_set_selection_policy")
        == "predeclared-smallest-hc-basis-strictly-above-max-m-not-rqmc-tuned",
        "CS basis is predeclared as the smallest HC space with N greater than max(m)",
    )
    audit.require(int(config.cs["target_pd_iterations"]) == 5000, "CS target is 5,000 iterations")
    audit.require(config.cs["spectral_norm_method"] == "power-iteration", "CS uses power iteration for the spectral norm")
    audit.require(int(config.cs["spectral_norm_power_iterations"]) == 50, "spectral-norm estimate uses 50 iterations")
    audit.require(float(config.cs["spectral_norm_safety_factor"]) == 1.01, "spectral-norm safety factor is 1.01")
    audit.require(int(config.dnn["epochs"]) == 50000, "DNN always trains for 50,000 epochs")
    audit.require(config.dnn["batch_mode"] == "full", "DNN is full batch")
    audit.require(config.dnn["validation_patience"] == "infinite", "DNN validation patience is infinite")
    audit.require(int(config.dnn["validation_interval_epochs"]) == 100, "DNN validation is sampled every 100 epochs")
    audit.require(int(config.plot["standard_deviation_ddof"]) == 1, "trial standard deviations use ddof=1")


def _check_data(arguments: argparse.Namespace, config: Any, audit: Audit) -> dict[str, str]:
    box_hash = config.parameter_box.fingerprint
    training_rows: list[np.ndarray] = []
    training_hashes: dict[str, str] = {}
    training_seeds: list[int] = []
    for trial in range(20):
        path = arguments.training_directory.resolve() / f"trial_{trial:03d}_n05000.npz"
        arrays, manifest = _load_registered_npz(path, audit)
        if not arrays:
            continue
        expected = (
            arrays.get("parameters", np.empty((0, 0))).shape == (5000, 11)
            and arrays.get("diagnostics_120min", np.empty((0, 0))).shape == (5000, 6)
            and _finite(arrays["parameters"])
            and _finite(arrays["diagnostics_120min"])
        )
        audit.require(expected, f"training shapes and finite values: trial {trial:03d}")
        audit.require(manifest.get("artifact") == "figure1-training-pool", f"training artifact type: trial {trial:03d}")
        audit.require(int(manifest.get("trial", -1)) == trial, f"training trial identity: {trial:03d}")
        audit.require(manifest.get("parameter_box_sha256") == box_hash, f"training parameter box: trial {trial:03d}")
        audit.require(tuple(manifest.get("diagnostic_names", ())) == config.outputs, f"training output order: trial {trial:03d}")
        training_rows.append(np.asarray(arrays["parameters"], dtype=np.float64))
        training_hashes[str(trial)] = sha256_file(path)
        training_seeds.append(int(manifest.get("seed", -1)))
    audit.require(len(training_rows) == 20, "all 20 training pools are registered")
    audit.require(len(set(training_seeds)) == 20, "all 20 training-pool seeds are distinct")

    validation, validation_manifest = _load_registered_npz(arguments.validation, audit)
    evaluation, evaluation_manifest = _load_registered_npz(arguments.evaluation, audit)
    if validation:
        validation_ok = (
            validation.get("physical_parameters", np.empty((0, 0))).shape == (5000, 11)
            and validation.get("diagnostics_120min", np.empty((0, 0))).shape == (5000, 6)
            and _finite(validation["physical_parameters"])
            and _finite(validation["diagnostics_120min"])
        )
        audit.require(validation_ok, "validation shapes and finite values")
        audit.require(validation_manifest.get("artifact") == "figure1-validation-data", "validation artifact type")
        audit.require(validation_manifest.get("parameter_box_sha256") == box_hash, "validation parameter box")
        audit.require(validation_manifest.get("used_for_model_selection") is True, "validation is used only for model selection")
    if evaluation:
        evaluation_ok = (
            evaluation.get("physical_parameters", np.empty((0, 0, 0))).shape == (10, 16384, 11)
            and evaluation.get("diagnostics_120min", np.empty((0, 0, 0))).shape == (10, 16384, 6)
            and evaluation.get("equal_weights", np.empty((0, 0))).shape == (10, 16384)
            and evaluation.get("replication_seeds", np.empty(0)).shape == (10,)
            and _finite(evaluation["physical_parameters"])
            and _finite(evaluation["diagnostics_120min"])
            and _finite(evaluation["equal_weights"])
        )
        audit.require(evaluation_ok, "RQMC shapes and finite values")
        audit.require(evaluation_manifest.get("artifact") == "figure1-primary-rqmc-evaluation-data", "RQMC artifact type")
        audit.require(evaluation_manifest.get("parameter_box_sha256") == box_hash, "RQMC parameter box")
        audit.require(evaluation_manifest.get("metric") == config.evaluation["metric"], "RQMC metric identity")
        audit.require(evaluation_manifest.get("used_for_model_selection") is False, "RQMC test data are excluded from model selection")
        expected_weights = np.full((10, 16384), 1.0 / 16384.0)
        audit.require(np.array_equal(evaluation["equal_weights"], expected_weights), "RQMC weights are exactly equal and positive")
        seeds = np.asarray(evaluation["replication_seeds"], dtype=np.uint64)
        audit.require(len(np.unique(seeds)) == 10, "all 10 RQMC scrambles use distinct seeds")

    if len(training_rows) == 20 and validation and evaluation:
        all_training = np.concatenate(training_rows, axis=0)
        validation_x = np.asarray(validation["physical_parameters"], dtype=np.float64)
        evaluation_x = np.asarray(evaluation["physical_parameters"], dtype=np.float64).reshape(-1, 11)
        train_keys = _row_keys(all_training)
        validation_keys = _row_keys(validation_x)
        evaluation_keys = _row_keys(evaluation_x)
        audit.require(len(np.unique(train_keys)) == len(train_keys), "no exact duplicates among 100,000 training rows")
        audit.require(len(np.unique(validation_keys)) == len(validation_keys), "no exact duplicates within validation data")
        audit.require(len(np.unique(evaluation_keys)) == len(evaluation_keys), "no exact duplicates within RQMC data")
        audit.require(np.intersect1d(train_keys, validation_keys).size == 0, "training and validation rows are disjoint")
        audit.require(np.intersect1d(train_keys, evaluation_keys).size == 0, "training and RQMC rows are disjoint")
        audit.require(np.intersect1d(validation_keys, evaluation_keys).size == 0, "validation and RQMC rows are disjoint")

    return {
        "validation": sha256_file(arguments.validation) if arguments.validation.is_file() else "",
        "evaluation": sha256_file(arguments.evaluation) if arguments.evaluation.is_file() else "",
        **{f"training_{key}": value for key, value in training_hashes.items()},
    }


def _check_design_caches(arguments: argparse.Namespace, config: Any, audit: Audit) -> None:
    if arguments.design_cache_directory is None:
        audit.warn("q=23 design cache was not requested for this readiness pass")
        return
    from cloud_column_reproduction.figure1.design_cache import load_design_cache

    directory = arguments.design_cache_directory.resolve()
    maximum = int(config.experiment["train_size_max"])
    validated = 0
    for trial in range(int(config.experiment["trials"])):
        training = (
            arguments.training_directory.resolve()
            / f"trial_{trial:03d}_n{maximum:05d}.npz"
        )
        try:
            design, indices, weights, norms, manifest = load_design_cache(
                config, REPOSITORY, training, directory, trial=trial
            )
            if design.shape != (maximum, int(config.cs["expected_basis_size"])):
                raise ValueError("wrong memory-mapped design shape")
            if len(indices) != len(weights) or len(indices) != design.shape[1]:
                raise ValueError("basis metadata do not match cached design")
            if tuple(sorted(norms)) != config.train_sizes:
                raise ValueError("cached spectral-norm prefixes are incomplete")
            if manifest.get("artifact_role") != "regenerable-performance-cache-not-scientific-result":
                raise ValueError("cache has the wrong evidentiary role")
            validated += 1
        except Exception as error:
            audit.fail(f"q=23 design cache trial {trial:03d}: {error}")
    audit.require(
        validated == int(config.experiment["trials"]),
        f"all {validated} q=23 design caches are checksum-valid",
    )


def _check_gpu(audit: Audit) -> None:
    try:
        import torch

        audit.require(torch.cuda.is_available(), "PyTorch can access CUDA")
        if torch.cuda.is_available():
            left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            value = float((left @ left).sum().cpu())
            audit.require(np.isfinite(value), f"PyTorch CUDA matmul succeeds on {torch.cuda.get_device_name(0)}")
            print(f"INFO: torch={torch.__version__}, CUDA runtime={torch.version.cuda}")
    except Exception as error:  # readiness must report both import and runtime failures
        audit.fail(f"PyTorch CUDA check failed: {error}")
    try:
        import cupy as cp

        audit.require(cp.cuda.runtime.getDeviceCount() > 0, "CuPy can access CUDA")
        if cp.cuda.runtime.getDeviceCount() > 0:
            left = cp.arange(16, dtype=cp.float32).reshape(4, 4)
            value = float(cp.asnumpy((left @ left).sum()))
            audit.require(np.isfinite(value), "CuPy CUDA matmul succeeds")
            print(f"INFO: cupy={cp.__version__}")
    except Exception as error:
        audit.fail(f"CuPy CUDA check failed: {error}")


def _check_destination(
    output: Path,
    config_hash: str,
    data_hashes: dict[str, str],
    minimum_free_gib: float,
    audit: Audit,
) -> None:
    probe = output.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gib = shutil.disk_usage(probe).free / 2**30
    audit.require(free_gib >= minimum_free_gib, f"free disk {free_gib:.1f} GiB >= {minimum_free_gib:.1f} GiB")
    if not output.exists():
        audit.pass_(f"production destination is new: {output}")
        return
    manifests = sorted(output.glob("trial_*/n*/fit.manifest.json"))
    stale: list[Path] = []
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            trial = int(path.parents[1].name.removeprefix("trial_"))
            if (
                payload.get("config_sha256") != config_hash
                or payload.get("training_data_sha256") != data_hashes.get(f"training_{trial}")
                or payload.get("validation_data_sha256") != data_hashes.get("validation")
                or payload.get("evaluation_data_sha256") != data_hashes.get("evaluation")
                or payload.get("pdr_hyperparameters", {}).get("algorithm_revision") != PDR_ALGORITHM_REVISION
                or payload.get("dnn_diagnostics", {}).get("algorithm_revision") != DNN_ALGORITHM_REVISION
            ):
                stale.append(path)
        except (OSError, ValueError, json.JSONDecodeError):
            stale.append(path)
    audit.require(not stale, f"no stale completed fits in destination ({len(manifests)} current manifests)")
    expected_directories = {
        output / f"trial_{trial:03d}" / f"n{size:05d}"
        for trial in range(20)
        for size in range(100, 5001, 100)
    }
    unexpected = [
        path for path in output.glob("trial_*/n*") if path.is_dir() and path not in expected_directories
    ]
    audit.require(not unexpected, "destination contains no out-of-design fit directories")


def main() -> int:
    arguments = _parser().parse_args()
    audit = Audit()
    try:
        config = load_config(arguments.config)
    except Exception as error:
        print(f"FAIL: production config is invalid: {error}")
        print("OVERALL: NOT READY (1 failure)")
        return 2
    print(f"INFO: config={config.path}")
    print(f"INFO: config_sha256={sha256_file(config.path)}")
    print(f"INFO: conda_environment={os.environ.get('CONDA_DEFAULT_ENV')}")
    audit.require(os.environ.get("CONDA_DEFAULT_ENV") == "cloud-column-study-derecho", "cloud-column-study-derecho is active")
    _check_config(config, audit)
    data_hashes = _check_data(arguments, config, audit)
    _check_design_caches(arguments, config, audit)
    if not arguments.skip_gpu:
        _check_gpu(audit)
    _check_destination(
        arguments.output_directory.resolve(),
        sha256_file(config.path),
        data_hashes,
        arguments.minimum_free_gib,
        audit,
    )
    if audit.failures:
        print(f"OVERALL: NOT READY ({len(audit.failures)} failure(s))")
        return 2
    print("OVERALL: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
