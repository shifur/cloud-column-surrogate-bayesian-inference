"""End-to-end fitting and evaluation for one paired Figure 1 dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from cloud_column_reproduction.invariants import audit_manifests
from cloud_column_reproduction.provenance import sha256_file, write_manifest

from .config import Figure1Config, pdr_spectral_norm_options
from .data import derived_seed
from .dnn import (
    DNN_DEFAULT_GPU_TRAINING_BACKEND,
    DNN_TRAINING_BACKEND_EAGER,
    DNN_TRAINING_BACKEND_OPTIMIZED,
    diagnostics_dict as dnn_diagnostics_dict,
    final_checkpoint_payload,
    fit_dnn,
    predict_dnn,
    save_final_dnn,
    state_checkpoint_payload,
    trainable_parameter_count,
)
from .l2 import (
    RQMCRatioEstimate,
    replicated_rqmc_relative_l2,
    signed_quadrature_relative_l2,
    signed_quadrature_terms,
)
from .pdr import (
    diagnostics_dict as pdr_diagnostics_dict,
    fit_pdr,
    hyperparameters_dict,
    predict_pdr,
)
from .pdr_gpu import build_cuda_pdr_predictor


METRIC_COLUMNS = ("trial", "train_size", "output", "method", "relative_l2")


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.stem}.manifest.json")


def _load_registered_data(
    data_path: str | Path,
    config: Figure1Config,
    cache: dict[Path, tuple[dict[str, np.ndarray], dict[str, Any], Path]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    path = Path(data_path).resolve()
    if cache is not None and path in cache:
        return cache[path]
    manifest_path = _sidecar(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Registered data or manifest is missing for {path}")
    audit_manifests(
        [manifest_path],
        config.parameter_box,
        config.parameter_box_path,
        # These are immutable input data produced before the fit configuration
        # was frozen.  Preserve their recorded generation-config hashes, while
        # validating their data hashes and embedded semantic contracts below.
        verify_recorded_config_file=False,
        allow_legacy_schema_v1=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_sha256") != sha256_file(path):
        raise ValueError(f"Data checksum does not match its manifest: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    loaded = (arrays, manifest, manifest_path)
    if cache is not None:
        cache[path] = loaded
    return loaded


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_validation_history(path: Path, history: Any, outputs: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = ("epoch", "validation_mse") + tuple(
        f"{name}_validation_mse" for name in outputs
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        for epoch, loss, per_output in zip(
            history.epochs, history.mse, history.per_output_mse, strict=True
        ):
            writer.writerow((epoch, f"{loss:.17g}", *(f"{value:.17g}" for value in per_output)))
    temporary.replace(path)


def _write_pdr_convergence(path: Path, history: Any, outputs: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = (
        "restart",
        "realized_iterations",
        "output",
        "primal_objective",
        "normalized_objective",
        "dual_objective",
        "relative_primal_dual_gap",
        "residual_dual_objective",
        "relative_residual_dual_gap",
        "relative_training_residual",
        "relative_coefficient_change",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        histories = zip(
            history.restarts,
            history.realized_iterations,
            history.primal_objectives,
            history.normalized_objectives,
            history.dual_objectives,
            history.relative_primal_dual_gaps,
            history.residual_dual_objectives,
            history.relative_residual_dual_gaps,
            history.relative_training_residuals,
            history.relative_coefficient_changes,
            strict=True,
        )
        for restart_values in histories:
            restart, iterations, *metric_rows = restart_values
            for output_index, output in enumerate(outputs):
                writer.writerow(
                    (
                        restart,
                        iterations,
                        output,
                        *(f"{row[output_index]:.17g}" for row in metric_rows),
                    )
                )
    temporary.replace(path)


def _write_rqmc_convergence(
    path: Path,
    estimates: dict[tuple[str, int], RQMCRatioEstimate],
    outputs: tuple[str, ...],
) -> None:
    """Persist prefix estimates and every replicate integral used to form them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = (
        "method",
        "prefix_log2",
        "points_per_replication",
        "output",
        "relative_l2",
        "standard_error",
        "confidence_lower",
        "confidence_upper",
        "replication",
        "squared_error_mean",
        "squared_truth_mean",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        for (method, prefix_log2), estimate in sorted(estimates.items()):
            for output_index, output in enumerate(outputs):
                for replication in range(
                    estimate.squared_error_by_replication.shape[0]
                ):
                    writer.writerow(
                        (
                            method,
                            prefix_log2,
                            2**prefix_log2,
                            output,
                            f"{estimate.relative_l2[output_index]:.17g}",
                            f"{estimate.standard_error[output_index]:.17g}",
                            f"{estimate.confidence_lower[output_index]:.17g}",
                            f"{estimate.confidence_upper[output_index]:.17g}",
                            replication,
                            f"{estimate.squared_error_by_replication[replication, output_index]:.17g}",
                            f"{estimate.squared_truth_by_replication[replication, output_index]:.17g}",
                        )
                    )
    temporary.replace(path)


def _require_disjoint_rows(
    left_name: str,
    left: np.ndarray,
    right_name: str,
    right: np.ndarray,
) -> None:
    left_values = np.ascontiguousarray(left, dtype=np.float64)
    right_values = np.ascontiguousarray(right, dtype=np.float64)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("Dataset inputs must be matrices")
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("Dataset input dimensions do not match")
    row_dtype = np.dtype((np.void, left_values.dtype.itemsize * left_values.shape[1]))
    left_rows = left_values.view(row_dtype).reshape(-1)
    right_rows = right_values.view(row_dtype).reshape(-1)
    if np.intersect1d(left_rows, right_rows).size:
        raise ValueError(f"{left_name} and {right_name} inputs overlap")


def fit_surrogate_pair(
    config: Figure1Config,
    repository: str | Path,
    training_path: str | Path,
    validation_path: str | Path | None,
    evaluation_path: str | Path,
    output_directory: str | Path,
    *,
    trial: int,
    sample_count: int,
    use_gpu: bool,
    registered_data_cache: (
        dict[Path, tuple[dict[str, np.ndarray], dict[str, Any], Path]] | None
    ) = None,
    pdr_design_cache: tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[float, float]], dict[str, Any]
    ] | None = None,
    cs_only_timing_benchmark: bool = False,
) -> tuple[Path, Path]:
    pipeline_started = time.perf_counter()
    if config.cs.get("offline_cache", {}).get("enabled") and pdr_design_cache is None:
        raise ValueError("Authoritative q=23 fit requires its checksum-bound offline CS cache")
    repository = Path(repository).resolve()
    output_directory = Path(output_directory).resolve()
    data_load_started = time.perf_counter()
    training, training_manifest, training_manifest_path = _load_registered_data(
        training_path, config, registered_data_cache
    )
    if cs_only_timing_benchmark:
        validation = validation_manifest = validation_manifest_path = None
    else:
        if validation_path is None:
            raise ValueError("Paired Figure 1 fitting requires validation data")
        validation, validation_manifest, validation_manifest_path = _load_registered_data(
            validation_path, config, registered_data_cache
        )
    evaluation, evaluation_manifest, evaluation_manifest_path = _load_registered_data(
        evaluation_path, config, registered_data_cache
    )
    data_load_wall_seconds = time.perf_counter() - data_load_started
    preprocessing_started = time.perf_counter()
    if not cs_only_timing_benchmark:
        assert validation_manifest is not None
        if not bool(validation_manifest.get("sccm_evaluated")):
            raise ValueError("The Figure 1 validation set has inputs but no SCCM outputs")
        if not bool(validation_manifest.get("used_for_model_selection")):
            raise ValueError("The DNN validation set must be marked for model selection")
        if int(validation_manifest.get("point_count", -1)) != int(
            config.validation["expected_points"]
        ):
            raise ValueError("Validation-set size does not match the controlled configuration")
    if not bool(evaluation_manifest.get("sccm_evaluated")):
        raise ValueError("The Figure 1 test set has inputs but no SCCM outputs")
    if bool(evaluation_manifest.get("used_for_model_selection")):
        raise ValueError("The primary test set cannot be used for model selection")
    evaluation_metric = str(config.evaluation["metric"])
    if evaluation_manifest.get("metric") != evaluation_metric:
        raise ValueError("The primary Figure 1 test artifact uses the wrong metric")
    if config.is_rqmc_primary:
        if evaluation_manifest.get("artifact") != (
            "figure1-primary-rqmc-evaluation-data"
        ):
            raise ValueError("The primary Figure 1 artifact is not registered RQMC truth")
        if evaluation_manifest.get("weight_policy") != "equal-positive":
            raise ValueError("The RQMC primary artifact must use equal positive weights")
        if evaluation_manifest.get("uncertainty_method") != config.evaluation.get(
            "uncertainty_method"
        ):
            raise ValueError("RQMC uncertainty contract does not match the config")
    elif config.is_tasmanian_evaluation:
        if evaluation_manifest.get("corrected_weight_policy") != "signed-quadrature":
            raise ValueError("The sparse-grid artifact must preserve signed weights")
    else:
        raise ValueError("The fitting pipeline does not support this evaluation kind")
    if int(evaluation_manifest.get("point_count", -1)) != int(
        config.evaluation["expected_points"]
    ):
        raise ValueError("Evaluation size does not match the configuration")
    if sample_count > len(training["parameters"]):
        raise ValueError("Requested training prefix exceeds its registered pool")
    if int(training_manifest["trial"]) != trial:
        raise ValueError("Training-pool trial does not match the requested fit")

    training_pool_physical = np.asarray(training["parameters"], dtype=np.float64)
    physical = training_pool_physical[:sample_count]
    targets = np.asarray(training["diagnostics_120min"][:sample_count], dtype=np.float64)
    canonical = config.parameter_box.to_canonical(physical)
    if not cs_only_timing_benchmark:
        assert validation is not None
        validation_physical = np.asarray(
            validation["physical_parameters"], dtype=np.float64
        )
        validation_canonical = config.parameter_box.to_canonical(validation_physical)
        validation_truth = np.asarray(
            validation["diagnostics_120min"], dtype=np.float64
        )
    evaluation_physical = np.asarray(evaluation["physical_parameters"], dtype=np.float64)
    if evaluation_physical.ndim not in (2, 3) or evaluation_physical.shape[-1] != 11:
        raise ValueError("Evaluation physical parameters have the wrong shape")
    evaluation_shape = evaluation_physical.shape[:-1]
    evaluation_physical_flat = evaluation_physical.reshape(-1, 11)
    evaluation_canonical = config.parameter_box.to_canonical(evaluation_physical_flat)
    truth = np.asarray(evaluation["diagnostics_120min"], dtype=np.float64)
    if truth.shape != (*evaluation_shape, len(config.outputs)):
        raise ValueError("Evaluation truth has the wrong shape")
    if not cs_only_timing_benchmark:
        _require_disjoint_rows(
            "training", training_pool_physical, "validation", validation_physical
        )
    _require_disjoint_rows(
        "training", training_pool_physical, "evaluation", evaluation_physical_flat
    )
    if not cs_only_timing_benchmark:
        _require_disjoint_rows(
            "validation", validation_physical, "evaluation", evaluation_physical_flat
        )
    quadrature_weights: np.ndarray | None = None
    if config.is_tasmanian_evaluation:
        quadrature_weights = np.asarray(
            evaluation["quadrature_weights"], dtype=np.float64
        )
        if quadrature_weights.shape != (truth.shape[0],):
            raise ValueError("Sparse-grid quadrature weights do not match the truth array")
    else:
        replications = int(config.evaluation["replications"])
        points_per_replication = int(config.evaluation["points_per_replication"])
        if evaluation_shape != (replications, points_per_replication):
            raise ValueError("RQMC truth does not match its replication design")
        equal_weights = np.asarray(evaluation["equal_weights"], dtype=np.float64)
        if equal_weights.shape != evaluation_shape or np.any(equal_weights <= 0.0):
            raise ValueError("RQMC weights must be positive and match the design")
        expected_weight = 1.0 / points_per_replication
        if not np.array_equal(
            equal_weights, np.full(evaluation_shape, expected_weight)
        ):
            raise ValueError("RQMC point weights must be exactly equal within replications")

    dnn_training_backend = config.dnn.get(
        "training_backend",
        DNN_DEFAULT_GPU_TRAINING_BACKEND
        if use_gpu
        else DNN_TRAINING_BACKEND_EAGER,
    )
    destination = output_directory / f"trial_{trial:03d}" / f"n{sample_count:05d}"
    destination.mkdir(parents=True, exist_ok=True)
    pdr_checkpoint = destination / "pdr_configurable_norm_index_v6.progress.npz"
    dnn_checkpoint = destination / (
        "dnn_compiled_fused_v1.progress.pt"
        if dnn_training_backend == DNN_TRAINING_BACKEND_OPTIMIZED
        else "dnn_complete_history_v3.progress.pt"
    )
    pdr_model = destination / "pdr_model.npz"
    pdr_convergence = destination / "pdr_convergence.csv"
    dnn_model = destination / "dnn_model.pt"
    dnn_final_epoch_model = destination / "dnn_final_epoch_model.pt"
    dnn_validation_history = destination / "dnn_validation_history.csv"
    dnn_training_history = destination / "dnn_training_history.npz"
    rqmc_convergence = destination / "rqmc_convergence.csv"
    metrics_path = destination / "metrics.csv"
    manifest_path = destination / "fit.manifest.json"

    seed_index = trial * 100_000 + sample_count
    dnn_seed = derived_seed(
        int(config.experiment["base_seed"]), "figure1-dnn-fit", seed_index
    )
    preprocessing_wall_seconds = time.perf_counter() - preprocessing_started
    cached_design = None
    cached_indices = None
    cached_weights = None
    cached_spectral_norm = None
    design_cache_manifest: dict[str, Any] | None = None
    if pdr_design_cache is not None:
        (
            cached_design_all, cached_indices, cached_weights, cached_norms,
            design_cache_manifest,
        ) = pdr_design_cache
        cached_design = cached_design_all[:sample_count]
        cached_spectral_norm = cached_norms[sample_count]
    (
        coefficients,
        indices,
        intrinsic_weights,
        pdr_hyperparameters,
        pdr_diagnostics,
        pdr_history,
    ) = fit_pdr(
        canonical,
        targets,
        order=int(config.cs["hyperbolic_cross_order"]),
        target_iterations=int(config.cs["target_pd_iterations"]),
        additive_tolerance=float(config.cs["restart_additive_tolerance"]),
        contraction=float(config.cs["restart_contraction"]),
        **pdr_spectral_norm_options(config),
        use_gpu=use_gpu,
        dtype=np.dtype(config.cs["compute_dtype"]),
        checkpoint_path=pdr_checkpoint,
        precomputed_unscaled_design=cached_design,
        precomputed_indices=cached_indices,
        precomputed_intrinsic_weights=cached_weights,
        precomputed_spectral_norm=cached_spectral_norm,
    )
    pdr_convergence_started = time.perf_counter()
    _write_pdr_convergence(pdr_convergence, pdr_history, config.outputs)
    pdr_convergence_write_wall_seconds = time.perf_counter() - pdr_convergence_started
    pdr_prediction_started = time.perf_counter()
    if use_gpu and config.is_rqmc_primary:
        pdr_predictor = build_cuda_pdr_predictor(
            coefficients,
            indices,
            backend="cupy-cached-sparse-cuda",
            prune_exact_zeros=True,
        )
        try:
            chunk = int(config.evaluation["points_per_replication"])
            pdr_prediction_flat = np.concatenate(
                [
                    pdr_predictor(evaluation_canonical[start : start + chunk])
                    for start in range(0, len(evaluation_canonical), chunk)
                ],
                axis=0,
            )
            pdr_prediction_backend = dict(pdr_predictor.metadata)
        finally:
            pdr_predictor.close()
        audit_count = min(128, len(evaluation_canonical))
        pdr_cpu_audit = predict_pdr(
            evaluation_canonical[:audit_count], coefficients, indices,
            prune_exact_zeros=False,
        )
        pdr_gpu_audit = pdr_prediction_flat[:audit_count]
        pdr_prediction_maximum_absolute_audit_difference = float(
            np.max(np.abs(pdr_cpu_audit - pdr_gpu_audit), initial=0.0)
        )
        if not np.allclose(pdr_cpu_audit, pdr_gpu_audit, rtol=1.0e-10, atol=1.0e-9):
            raise RuntimeError("Cached-sparse CUDA CS prediction failed dense CPU audit")
    else:
        pdr_prediction_flat = predict_pdr(
            evaluation_canonical, coefficients, indices,
            prune_exact_zeros=True,
        )
        pdr_prediction_backend = {
            "backend": "numpy-cpu-chunked",
            "prune_exact_zeros": "true",
        }
        pdr_prediction_maximum_absolute_audit_difference = 0.0
        audit_count = 0
    pdr_prediction_wall_seconds = time.perf_counter() - pdr_prediction_started
    pdr_prediction = pdr_prediction_flat.reshape(truth.shape)
    rqmc_estimates: dict[tuple[str, int], RQMCRatioEstimate] = {}
    signed_terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pdr_metric_started = time.perf_counter()
    if config.is_rqmc_primary:
        for prefix_log2 in config.evaluation["prefix_log2"]:
            prefix_log2 = int(prefix_log2)
            prefix_count = 2**prefix_log2
            rqmc_estimates[("CS", prefix_log2)] = replicated_rqmc_relative_l2(
                truth[:, :prefix_count],
                pdr_prediction[:, :prefix_count],
                confidence_level=float(config.evaluation["confidence_level"]),
            )
        pdr_errors = rqmc_estimates[
            ("CS", int(config.evaluation["log2_points_per_replication"]))
        ].relative_l2
    else:
        assert quadrature_weights is not None
        pdr_numerator, pdr_denominator = signed_quadrature_terms(
            truth, pdr_prediction, quadrature_weights
        )
        signed_terms["CS"] = (pdr_numerator, pdr_denominator)
        pdr_errors = signed_quadrature_relative_l2(
            truth, pdr_prediction, quadrature_weights
        )
    pdr_metric_wall_seconds = time.perf_counter() - pdr_metric_started
    pdr_model_write_started = time.perf_counter()
    _atomic_npz(
        pdr_model,
        coefficients=coefficients,
        indices=indices,
        intrinsic_weights=intrinsic_weights,
        parameter_lower=np.asarray(config.parameter_box.lower),
        parameter_upper=np.asarray(config.parameter_box.upper),
        parameter_names=np.asarray(config.parameter_box.names),
        output_names=np.asarray(config.outputs),
        sample_count=np.asarray(sample_count),
        trial=np.asarray(trial),
        parameter_box_sha256=np.asarray(config.parameter_box.fingerprint),
        algorithm_revision=np.asarray(pdr_hyperparameters.algorithm_revision),
        iteration_budget_policy=np.asarray(pdr_hyperparameters.iteration_budget_policy),
        restart_output=np.asarray(pdr_hyperparameters.restart_output),
        nominal_iteration_budget=np.asarray(pdr_hyperparameters.target_iterations),
        inner_iterations=np.asarray(pdr_hyperparameters.inner_iterations),
        planned_restarts=np.asarray(pdr_hyperparameters.planned_restarts),
        planned_iterations=np.asarray(pdr_hyperparameters.planned_iterations),
        additive_tolerance=np.asarray(pdr_hyperparameters.additive_tolerance),
        contraction=np.asarray(pdr_hyperparameters.contraction),
        spectral_norm=np.asarray(pdr_hyperparameters.spectral_norm),
        spectral_norm_raw_estimate=np.asarray(
            pdr_hyperparameters.spectral_norm_raw_estimate
        ),
        spectral_norm_method=np.asarray(pdr_hyperparameters.spectral_norm_method),
        spectral_norm_safety_factor=np.asarray(
            pdr_hyperparameters.spectral_norm_safety_factor
        ),
        spectral_norm_seed=np.asarray(pdr_hyperparameters.spectral_norm_seed),
        spectral_norm_power_iterations=np.asarray(
            pdr_hyperparameters.spectral_norm_power_iterations
        ),
        spectral_norm_randomized_components=np.asarray(
            pdr_hyperparameters.spectral_norm_randomized_components
        ),
        spectral_norm_randomized_oversamples=np.asarray(
            pdr_hyperparameters.spectral_norm_randomized_oversamples
        ),
        spectral_norm_randomized_power_iterations=np.asarray(
            pdr_hyperparameters.spectral_norm_randomized_power_iterations
        ),
        spectral_norm_randomized_power_normalizer=np.asarray(
            pdr_hyperparameters.spectral_norm_randomized_power_normalizer
        ),
    )
    pdr_model_write_wall_seconds = time.perf_counter() - pdr_model_write_started

    if cs_only_timing_benchmark:
        if not config.is_rqmc_primary:
            raise ValueError("The CS-only timing benchmark requires the primary RQMC metric")
        if design_cache_manifest is None or cached_spectral_norm is None:
            raise ValueError("The CS-only timing benchmark requires the offline q=23 cache")
        _write_rqmc_convergence(rqmc_convergence, rqmc_estimates, config.outputs)
        rows = [
            {
                "trial": trial, "train_size": sample_count, "output": output,
                "method": "CS", "relative_l2": f"{float(error):.17g}",
            }
            for output, error in zip(config.outputs, pdr_errors, strict=True)
        ]
        metrics_write_started = time.perf_counter()
        _write_metrics(metrics_path, rows)
        metrics_write_wall_seconds = time.perf_counter() - metrics_write_started
        wall_seconds = time.perf_counter() - pipeline_started
        full_prefix = int(config.evaluation["log2_points_per_replication"])
        full_estimate = rqmc_estimates[("CS", full_prefix)]
        write_manifest(
            manifest_path,
            {
                "artifact": "figure1-cs-only-execution-timing-benchmark",
                "artifact_role": "execution-timing-benchmark-not-manuscript-result",
                "experiment_id": config.experiment["id"],
                "config": str(config.path.relative_to(repository)),
                "config_sha256": sha256_file(config.path),
                "parameter_names": list(config.parameter_box.names),
                "parameter_lower": list(config.parameter_box.lower),
                "parameter_upper": list(config.parameter_box.upper),
                "parameter_box_sha256": config.parameter_box.fingerprint,
                "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
                "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
                "random_seeds": {},
                "trial": trial, "sample_count": sample_count, "workers": 1,
                "wall_seconds": wall_seconds,
                "stage_timings": {
                    "registered_data_load": data_load_wall_seconds,
                    "canonicalization_and_invariant_checks": preprocessing_wall_seconds,
                    "cs_index_and_design": pdr_diagnostics.index_and_design_wall_seconds,
                    "cs_spectral_norm": pdr_diagnostics.spectral_norm_wall_seconds,
                    "cs_backend_transfer_setup": pdr_diagnostics.backend_transfer_setup_wall_seconds,
                    "cs_solver": pdr_diagnostics.wall_seconds,
                    "cs_solver_compute": max(0.0, pdr_diagnostics.wall_seconds - pdr_diagnostics.checkpoint_wall_seconds),
                    "cs_checkpoint_during_solve": pdr_diagnostics.checkpoint_wall_seconds,
                    "cs_total_fit": pdr_diagnostics.total_fit_wall_seconds,
                    "cs_convergence_write": pdr_convergence_write_wall_seconds,
                    "cs_model_write": pdr_model_write_wall_seconds,
                    "cs_rqmc_prediction": pdr_prediction_wall_seconds,
                    "cs_rqmc_metric": pdr_metric_wall_seconds,
                    "metrics_write": metrics_write_wall_seconds,
                },
                "cs_design_cache": {
                    "artifact_role": design_cache_manifest["artifact_role"],
                    "data": design_cache_manifest["data"],
                    "data_sha256": design_cache_manifest["data_sha256"],
                    "basis_fingerprint_sha256": design_cache_manifest["basis_fingerprint_sha256"],
                    "prefix_rows": sample_count,
                    "spectral_norm": cached_spectral_norm[0],
                    "spectral_norm_raw_estimate": cached_spectral_norm[1],
                },
                "cs_prediction_backend": pdr_prediction_backend,
                "cs_prediction_dense_cpu_audit_points": audit_count,
                "cs_prediction_maximum_absolute_dense_cpu_audit_difference": pdr_prediction_maximum_absolute_audit_difference,
                "gpu_requested": use_gpu,
                "training_data": str(Path(training_path).absolute()),
                "training_data_sha256": sha256_file(training_path),
                "evaluation_used_for_model_selection": False,
                "evaluation_data": str(Path(evaluation_path).absolute()),
                "evaluation_data_sha256": sha256_file(evaluation_path),
                "evaluation_metric": evaluation_metric,
                "evaluation_weight_policy": "equal-positive",
                "rqmc_replications": int(config.evaluation["replications"]),
                "rqmc_points_per_replication": int(config.evaluation["points_per_replication"]),
                "rqmc_convergence": rqmc_convergence.name,
                "rqmc_convergence_sha256": sha256_file(rqmc_convergence),
                "rqmc_full_relative_l2": {
                    output: float(full_estimate.relative_l2[index])
                    for index, output in enumerate(config.outputs)
                },
                "pdr_model": pdr_model.name,
                "pdr_model_sha256": sha256_file(pdr_model),
                "pdr_hyperparameters": hyperparameters_dict(pdr_hyperparameters),
                "pdr_diagnostics": pdr_diagnostics_dict(pdr_diagnostics),
                "pdr_convergence": pdr_convergence.name,
                "pdr_convergence_sha256": sha256_file(pdr_convergence),
                "metrics": metrics_path.name,
                "metrics_sha256": sha256_file(metrics_path),
            },
            repository=repository,
        )
        return metrics_path, manifest_path

    (
        model,
        output_minimum,
        output_span,
        final_epoch_state,
        dnn_diagnostics,
        training_history,
        validation_history,
    ) = fit_dnn(
        canonical,
        targets,
        validation_canonical,
        validation_truth,
        hidden_layers=int(config.dnn["hidden_layers"]),
        width=int(config.dnn["width"]),
        epochs=int(config.dnn["epochs"]),
        learning_rate_initial=float(config.dnn["learning_rate_initial"]),
        learning_rate_final=float(config.dnn["learning_rate_final"]),
        seed=dnn_seed,
        checkpoint_interval=int(config.dnn["checkpoint_interval_epochs"]),
        progress_interval=int(
            config.dnn.get("progress_interval_epochs", config.dnn["checkpoint_interval_epochs"])
        ),
        validation_interval=int(config.dnn["validation_interval_epochs"]),
        validation_min_delta=float(config.dnn["validation_min_delta"]),
        checkpoint_path=dnn_checkpoint,
        use_gpu=use_gpu,
        training_backend=dnn_training_backend,
    )
    actual_parameters = trainable_parameter_count(model)
    expected_parameters = int(config.dnn["expected_trainable_parameters"])
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"DNN has {actual_parameters} trainable parameters, expected {expected_parameters}"
        )
    dnn_prediction_started = time.perf_counter()
    dnn_prediction_flat = predict_dnn(
        model, evaluation_canonical, output_minimum, output_span
    )
    dnn_prediction_wall_seconds = time.perf_counter() - dnn_prediction_started
    dnn_prediction = dnn_prediction_flat.reshape(truth.shape)
    dnn_metric_started = time.perf_counter()
    if config.is_rqmc_primary:
        for prefix_log2 in config.evaluation["prefix_log2"]:
            prefix_log2 = int(prefix_log2)
            prefix_count = 2**prefix_log2
            rqmc_estimates[("DNN", prefix_log2)] = replicated_rqmc_relative_l2(
                truth[:, :prefix_count],
                dnn_prediction[:, :prefix_count],
                confidence_level=float(config.evaluation["confidence_level"]),
            )
        dnn_errors = rqmc_estimates[
            ("DNN", int(config.evaluation["log2_points_per_replication"]))
        ].relative_l2
        _write_rqmc_convergence(
            rqmc_convergence, rqmc_estimates, config.outputs
        )
    else:
        assert quadrature_weights is not None
        dnn_numerator, dnn_denominator = signed_quadrature_terms(
            truth, dnn_prediction, quadrature_weights
        )
        signed_terms["DNN"] = (dnn_numerator, dnn_denominator)
        dnn_errors = signed_quadrature_relative_l2(
            truth, dnn_prediction, quadrature_weights
        )
    dnn_metric_wall_seconds = time.perf_counter() - dnn_metric_started
    dnn_artifact_write_started = time.perf_counter()
    common_dnn_metadata = {
        "parameter_lower": np.asarray(config.parameter_box.lower),
        "parameter_upper": np.asarray(config.parameter_box.upper),
        "parameter_names": config.parameter_box.names,
        "output_names": config.outputs,
        "sample_count": sample_count,
        "trial": trial,
        "hidden_layers": int(config.dnn["hidden_layers"]),
        "width": int(config.dnn["width"]),
        "epochs": int(config.dnn["epochs"]),
        "optimizer_steps": dnn_diagnostics.optimizer_steps,
        "learning_rate_initial": float(config.dnn["learning_rate_initial"]),
        "learning_rate_final": float(config.dnn["learning_rate_final"]),
        "algorithm_revision": dnn_diagnostics.algorithm_revision,
        "training_backend": dnn_training_backend,
        "batch_mode": dnn_diagnostics.batch_mode,
        "validation_patience": dnn_diagnostics.validation_patience,
        "validation_selection_metric": dnn_diagnostics.validation_selection_metric,
        "validation_interval_epochs": dnn_diagnostics.validation_interval_epochs,
        "stopped_early": dnn_diagnostics.stopped_early,
        "best_epoch": dnn_diagnostics.best_epoch,
        "best_validation_mse": dnn_diagnostics.best_validation_mse,
        "final_validation_mse": dnn_diagnostics.final_validation_mse,
        "parameter_box_sha256": config.parameter_box.fingerprint,
        "validation_data_sha256": sha256_file(validation_path),
    }
    save_final_dnn(
        dnn_model,
        final_checkpoint_payload(
            model,
            output_minimum,
            output_span,
            {
                **common_dnn_metadata,
                "model_role": "best-validation-restored",
            },
        ),
    )
    save_final_dnn(
        dnn_final_epoch_model,
        state_checkpoint_payload(
            final_epoch_state,
            output_minimum,
            output_span,
            {
                **common_dnn_metadata,
                "model_role": "final-epoch-audit",
            },
        ),
    )
    _write_validation_history(
        dnn_validation_history, validation_history, config.outputs
    )
    _atomic_npz(
        dnn_training_history,
        epoch=np.asarray(training_history.epochs, dtype=np.int64),
        optimizer_mse=np.asarray(training_history.optimizer_mse, dtype=np.float64),
    )

    dnn_artifact_write_wall_seconds = time.perf_counter() - dnn_artifact_write_started
    rows: list[dict[str, Any]] = []
    for method, errors in (("CS", pdr_errors), ("DNN", dnn_errors)):
        for output, error in zip(config.outputs, errors, strict=True):
            rows.append(
                {
                    "trial": trial,
                    "train_size": sample_count,
                    "output": output,
                    "method": method,
                    "relative_l2": f"{float(error):.17g}",
                }
            )
    metrics_write_started = time.perf_counter()
    _write_metrics(metrics_path, rows)
    metrics_write_wall_seconds = time.perf_counter() - metrics_write_started
    wall_seconds = time.perf_counter() - pipeline_started
    if config.is_rqmc_primary:
        full_prefix = int(config.evaluation["log2_points_per_replication"])
        evaluation_details: dict[str, Any] = {
            "evaluation_metric": evaluation_metric,
            "evaluation_weight_policy": "equal-positive",
            "rqmc_replications": int(config.evaluation["replications"]),
            "rqmc_points_per_replication": int(
                config.evaluation["points_per_replication"]
            ),
            "rqmc_prefix_log2": [
                int(value) for value in config.evaluation["prefix_log2"]
            ],
            "rqmc_uncertainty_method": config.evaluation["uncertainty_method"],
            "rqmc_confidence_level": float(config.evaluation["confidence_level"]),
            "rqmc_convergence": rqmc_convergence.name,
            "rqmc_convergence_sha256": sha256_file(rqmc_convergence),
            "rqmc_full_estimates": {
                method: {
                    output: {
                        "relative_l2": float(estimate.relative_l2[index]),
                        "standard_error": float(estimate.standard_error[index]),
                        "confidence_lower": float(estimate.confidence_lower[index]),
                        "confidence_upper": float(estimate.confidence_upper[index]),
                        "squared_error_means_by_replication": [
                            float(value)
                            for value in estimate.squared_error_by_replication[:, index]
                        ],
                        "squared_truth_means_by_replication": [
                            float(value)
                            for value in estimate.squared_truth_by_replication[:, index]
                        ],
                    }
                    for index, output in enumerate(config.outputs)
                }
                for method, estimate in (
                    ("CS", rqmc_estimates[("CS", full_prefix)]),
                    ("DNN", rqmc_estimates[("DNN", full_prefix)]),
                )
            },
        }
    else:
        assert quadrature_weights is not None
        evaluation_details = {
            "evaluation_metric": evaluation_metric,
            "quadrature_weight_policy": "signed-tasmanian-unchanged",
            "quadrature_weight_sum": float(np.sum(quadrature_weights)),
            "quadrature_absolute_weight_sum": float(
                np.sum(np.abs(quadrature_weights))
            ),
            "quadrature_negative_weight_count": int(
                np.count_nonzero(quadrature_weights < 0.0)
            ),
            "signed_quadrature_terms": {
                method: {
                    output: {
                        "squared_error_integral": float(numerator),
                        "squared_truth_integral": float(denominator),
                    }
                    for output, numerator, denominator in zip(
                        config.outputs, numerators, denominators, strict=True
                    )
                }
                for method, (numerators, denominators) in signed_terms.items()
            },
        }
    write_manifest(
        manifest_path,
        {
            "artifact": "figure1-surrogate-fit-pair",
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(repository)),
            "config_sha256": sha256_file(config.path),
            "parameter_names": list(config.parameter_box.names),
            "parameter_lower": list(config.parameter_box.lower),
            "parameter_upper": list(config.parameter_box.upper),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(repository)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "random_seeds": {"dnn": dnn_seed},
            "trial": trial,
            "sample_count": sample_count,
            "workers": 1,
            "wall_seconds": wall_seconds,
            "stage_timings": {
                "registered_data_load": data_load_wall_seconds,
                "canonicalization_and_invariant_checks": preprocessing_wall_seconds,
                "cs_index_and_design": pdr_diagnostics.index_and_design_wall_seconds,
                "cs_spectral_norm": pdr_diagnostics.spectral_norm_wall_seconds,
                "cs_backend_transfer_setup": pdr_diagnostics.backend_transfer_setup_wall_seconds,
                "cs_solver": pdr_diagnostics.wall_seconds,
                "cs_solver_compute": max(
                    0.0, pdr_diagnostics.wall_seconds - pdr_diagnostics.checkpoint_wall_seconds
                ),
                "cs_checkpoint_during_solve": pdr_diagnostics.checkpoint_wall_seconds,
                "cs_total_fit": pdr_diagnostics.total_fit_wall_seconds,
                "cs_convergence_write": pdr_convergence_write_wall_seconds,
                "cs_model_write": pdr_model_write_wall_seconds,
                "cs_rqmc_prediction": pdr_prediction_wall_seconds,
                "cs_rqmc_metric": pdr_metric_wall_seconds,
                "dnn_optimizer_compute": dnn_diagnostics.optimizer_compute_wall_seconds,
                "dnn_validation": dnn_diagnostics.validation_wall_seconds,
                "dnn_checkpoint_during_training": dnn_diagnostics.checkpoint_wall_seconds,
                "dnn_progress_reporting": dnn_diagnostics.progress_wall_seconds,
                "dnn_total_training": dnn_diagnostics.wall_seconds,
                "dnn_final_artifact_write": dnn_artifact_write_wall_seconds,
                "dnn_rqmc_prediction": dnn_prediction_wall_seconds,
                "dnn_rqmc_metric": dnn_metric_wall_seconds,
                "metrics_write": metrics_write_wall_seconds,
            },
            "dnn_checkpoint_policy": {
                "checkpoint_interval_epochs": int(config.dnn["checkpoint_interval_epochs"]),
                "progress_interval_epochs": int(
                    config.dnn.get("progress_interval_epochs", config.dnn["checkpoint_interval_epochs"])
                ),
                "incomplete_fit_resume_policy": "discard-and-rerun-complete-pair",
            },
            "cs_checkpoint_policy": {
                "restart_boundary": True,
                "write_when_crossing_iterations": 1000,
                "write_at_completion": True,
                "incomplete_fit_resume_policy": "discard-and-rerun-complete-pair",
            },
            "cs_design_cache": (
                None
                if design_cache_manifest is None
                else {
                    "artifact_role": design_cache_manifest["artifact_role"],
                    "data": design_cache_manifest["data"],
                    "data_sha256": design_cache_manifest["data_sha256"],
                    "basis_fingerprint_sha256": design_cache_manifest["basis_fingerprint_sha256"],
                    "prefix_rows": sample_count,
                    "spectral_norm": cached_spectral_norm[0],
                    "spectral_norm_raw_estimate": cached_spectral_norm[1],
                }
            ),
            "cs_prediction_backend": pdr_prediction_backend,
            "cs_prediction_dense_cpu_audit_points": audit_count,
            "cs_prediction_maximum_absolute_dense_cpu_audit_difference": (
                pdr_prediction_maximum_absolute_audit_difference
            ),
            "gpu_requested": use_gpu,
            "training_data": str(Path(training_path).absolute()),
            "training_data_sha256": sha256_file(training_path),
            "training_manifest": str(training_manifest_path),
            "training_manifest_sha256": sha256_file(training_manifest_path),
            "validation_data": str(Path(validation_path).absolute()),
            "validation_data_sha256": sha256_file(validation_path),
            "validation_manifest": str(validation_manifest_path),
            "validation_manifest_sha256": sha256_file(validation_manifest_path),
            "validation_used_for_model_selection": True,
            "evaluation_used_for_model_selection": False,
            "evaluation_data": str(Path(evaluation_path).absolute()),
            "evaluation_data_sha256": sha256_file(evaluation_path),
            "evaluation_manifest": str(evaluation_manifest_path),
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
            **evaluation_details,
            "pdr_model": pdr_model.name,
            "pdr_model_sha256": sha256_file(pdr_model),
            "pdr_hyperparameters": hyperparameters_dict(pdr_hyperparameters),
            "pdr_diagnostics": pdr_diagnostics_dict(pdr_diagnostics),
            "pdr_convergence": pdr_convergence.name,
            "pdr_convergence_sha256": sha256_file(pdr_convergence),
            "dnn_model": dnn_model.name,
            "dnn_model_sha256": sha256_file(dnn_model),
            "dnn_model_role": "best-validation-restored",
            "dnn_final_epoch_model": dnn_final_epoch_model.name,
            "dnn_final_epoch_model_sha256": sha256_file(dnn_final_epoch_model),
            "dnn_validation_history": dnn_validation_history.name,
            "dnn_validation_history_sha256": sha256_file(dnn_validation_history),
            "dnn_training_history": dnn_training_history.name,
            "dnn_training_history_sha256": sha256_file(dnn_training_history),
            "dnn_trainable_parameters": actual_parameters,
            "dnn_training_backend": common_dnn_metadata["training_backend"],
            "dnn_diagnostics": dnn_diagnostics_dict(dnn_diagnostics),
            "metrics": metrics_path.name,
            "metrics_sha256": sha256_file(metrics_path),
        },
        repository=repository,
    )
    return metrics_path, manifest_path
