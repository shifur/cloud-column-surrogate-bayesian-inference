"""Construct exact production evaluator and posterior callback adapters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cloud_column_reproduction.figure1.dnn import build_network, predict_dnn
from cloud_column_reproduction.figure1.pdr import predict_pdr
from cloud_column_reproduction.provenance import sha256_file
from cloud_column_reproduction.sccm import runner_from_repository

from .config import ControlledMCMCConfig, OUTPUT_ORDER, OUTPUT_UNITS
from .evaluators import DNNEvaluator, EvaluatorMetadata, PDREvaluator, SCCMEvaluator
from .likelihood import GaussianLikelihood, evaluate_posterior
from .problem import InverseProblem, LikelihoodCase


SURROGATE_BLOB_DTYPE = np.dtype(
    [
        ("log_prior", "f8"),
        ("log_likelihood", "f8"),
        ("status", "i1"),
    ]
)

PRODUCTION_BLOB_DTYPE = np.dtype(
    [
        ("log_prior", "f8"),
        ("log_likelihood", "f8"),
        ("status", "i1"),
        ("prediction", "f8", (6,)),
    ]
)


@dataclass
class RuntimeEvaluator:
    evaluator: Any
    likelihood: GaussianLikelihood
    vectorize: bool
    metadata: EvaluatorMetadata
    binding_hashes: dict[str, str]
    include_prediction_blobs: bool = True
    executor: ThreadPoolExecutor | None = None
    resources: tuple[Any, ...] = ()

    def close(self) -> None:
        if self.resources and hasattr(self.resources[0], "call_on_worker"):
            broker, *gpu_resources = self.resources
            # Synchronize/close CUDA resources before their owning worker exits.
            def close_gpu_resources():
                for resource in gpu_resources:
                    resource.close()
            broker.call_on_worker(close_gpu_resources)
            broker.close()
        else:
            for resource in self.resources:
                close = getattr(resource, "close", None)
                if close is not None:
                    close()
        self.resources = ()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    @property
    def blobs_dtype(self) -> np.dtype:
        return PRODUCTION_BLOB_DTYPE if self.include_prediction_blobs else SURROGATE_BLOB_DTYPE

    def log_probability(self, values: np.ndarray) -> Any:
        posterior = evaluate_posterior(values, self.evaluator, self.likelihood)
        rows = [
            (
                float(log_posterior),
                float(log_prior),
                float(log_likelihood),
                int(status),
                *(() if not self.include_prediction_blobs else (
                    np.asarray(prediction, dtype=np.float64),
                )),
            )
            for log_posterior, log_prior, log_likelihood, status, prediction in zip(
                posterior.log_posterior,
                posterior.log_prior,
                posterior.log_likelihood,
                posterior.status,
                posterior.predictions,
                strict=True,
            )
        ]
        return rows if np.asarray(values).ndim == 2 else rows[0]


def load_registered_observation(config: ControlledMCMCConfig) -> np.ndarray:
    repository = config.path.parent.parent
    path = repository / config.raw["observations"]["artifact"]
    if sha256_file(path) != config.raw["observations"]["artifact_sha256"]:
        raise ValueError("Registered MCMC observation hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype=np.float64)
        np.testing.assert_array_equal(archive["truth"], config.truth)
        np.testing.assert_array_equal(archive["parameter_names"], np.asarray(config.parameter_box.names))
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Registered MCMC observation must contain six finite values")
    return values


def build_inverse_problem(
    config: ControlledMCMCConfig, case_name: str, observation: np.ndarray
) -> InverseProblem:
    if case_name not in config.masks:
        raise ValueError(f"Unknown likelihood case: {case_name}")
    return InverseProblem(
        parameter_box=config.parameter_box,
        parameter_names=config.parameter_box.names,
        output_names=OUTPUT_ORDER,
        output_units=OUTPUT_UNITS,
        truth=tuple(config.truth),
        observation=tuple(np.asarray(observation, dtype=np.float64)),
        sigma=tuple(config.sigma),
        observation_time_minutes=120,
        likelihood_case=LikelihoodCase(
            case_name, tuple(int(value) for value in config.masks[case_name])
        ),
    )


def _common_metadata(
    config: ControlledMCMCConfig,
    *,
    method: str,
    artifact_sha256: str,
    device: str,
    dtype: str,
    batch_size: int | None,
    workers: int | None,
    evaluation_mode: str,
) -> EvaluatorMetadata:
    return EvaluatorMetadata(
        method=method,
        parameter_names=config.parameter_box.names,
        output_names=OUTPUT_ORDER,
        parameter_box_fingerprint=config.parameter_box.fingerprint,
        artifact_sha256=artifact_sha256,
        evaluation_mode=evaluation_mode,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        workers=workers,
    )


def _paired_microbatch(
    predict: Any,
    resources: tuple[Any, ...],
    bindings: dict[str, str],
    *,
    requests: int,
    window_seconds: float,
    cuda_enabled: bool,
) -> tuple[Any, tuple[Any, ...], dict[str, str]]:
    if requests == 1:
        return predict, resources, bindings
    if requests != 2 or not cuda_enabled:
        raise ValueError("Paired microbatching requires exactly two CUDA request streams")
    from .gpu_batching import PairedMicrobatchPredictor

    broker = PairedMicrobatchPredictor(
        predict, window_seconds=window_seconds
    )
    static = broker.metadata
    bindings = {
        **bindings,
        "inference_microbatch_policy": static["microbatch_policy"],
        "inference_microbatch_maximum_requests":
            static["microbatch_maximum_requests"],
        "inference_microbatch_window_seconds":
            static["microbatch_window_seconds"],
    }
    return broker, (broker, *resources), bindings

def build_runtime_evaluator(
    config: ControlledMCMCConfig,
    *,
    case_name: str,
    method: str,
    dnn_device: str = "cpu",
    dnn_backend: str = "torch-baseline",
    dnn_validation_audit: str | Path | None = None,
    gpu_microbatch_requests: int = 1,
    gpu_microbatch_window_seconds: float = 0.002,
    pdr_workers: int = 1,
    pdr_device: str = "cpu",
    pdr_backend: str = "numpy-cpu",
    pdr_prune_exact_zeros: bool = False,
    pdr_validation_audit: str | Path | None = None,
    sccm_workers: int = 32,
    prebuilt_sccm: bool = False,
    include_prediction_blobs: bool = True,
) -> RuntimeEvaluator:
    """Load one checksum-bound forward method under the common likelihood."""
    if method == "dnn" and gpu_microbatch_requests == 2:
        # Compiled CUDA graphs use thread-local ownership. Construction, probes,
        # prewarming, and all later inference must execute on this broker worker.
        from .gpu_batching import PairedMicrobatchPredictor

        holder: dict[str, Any] = {}
        broker = PairedMicrobatchPredictor(
            lambda values: holder["predictor"](values),
            window_seconds=gpu_microbatch_window_seconds,
        )

        def construct() -> RuntimeEvaluator:
            runtime = build_runtime_evaluator(
                config, case_name=case_name, method=method,
                dnn_device=dnn_device, dnn_backend=dnn_backend,
                dnn_validation_audit=dnn_validation_audit,
                gpu_microbatch_requests=1,
                include_prediction_blobs=include_prediction_blobs,
            )
            holder["predictor"] = runtime.evaluator._predict_canonical
            return runtime

        try:
            runtime = broker.call_on_worker(construct)
        except BaseException:
            broker.close()
            raise
        runtime.evaluator._predict_canonical = broker
        runtime.resources = (broker, *runtime.resources)
        runtime.binding_hashes.update({
            "inference_microbatch_policy": broker.metadata["microbatch_policy"],
            "inference_microbatch_maximum_requests": "2",
            "inference_microbatch_window_seconds": f"{gpu_microbatch_window_seconds:.12g}",
            "inference_thread_policy": "construction-and-inference-on-broker-worker",
        })
        return runtime
    repository = config.path.parent.parent
    observation = load_registered_observation(config)
    likelihood = GaussianLikelihood(build_inverse_problem(config, case_name, observation))
    if method == "dnn":
        import torch

        section = config.raw["surrogates"]["dnn"]
        path = repository / section["model"]
        if sha256_file(path) != section["model_sha256"]:
            raise ValueError("Frozen DNN artifact hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model_state", "output_minimum", "output_span", "parameter_box_sha256",
            "parameter_names", "output_names", "hidden_layers", "width",
        }
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("Frozen DNN artifact is missing required scientific metadata")
        if str(payload["parameter_box_sha256"]) != config.parameter_box.fingerprint:
            raise ValueError("Frozen DNN parameter-box fingerprint mismatch")
        if tuple(payload["parameter_names"]) != config.parameter_box.names:
            raise ValueError("Frozen DNN parameter order mismatch")
        if tuple(payload["output_names"]) != OUTPUT_ORDER:
            raise ValueError("Frozen DNN output order mismatch")
        if int(payload["hidden_layers"]) != 8 or int(payload["width"]) != 50:
            raise ValueError("Frozen DNN architecture mismatch")
        model = build_network(11, 6, 8, 50)
        model.load_state_dict(payload["model_state"])
        device = torch.device(dnn_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA DNN evaluation but CUDA is unavailable")
        resource = None
        if device.type == "cpu":
            if dnn_backend not in {"torch-baseline", "torch-baseline-cpu"}:
                raise ValueError("CPU DNN evaluation requires the baseline backend")
            model.to(device); model.eval()
            predict = lambda canonical: predict_dnn(
                model,
                canonical,
                payload["output_minimum"],
                payload["output_span"],
                batch_size=len(canonical),
            )
            backend_bindings = {"inference_backend": "torch-baseline-cpu"}
        elif dnn_backend == "torch-baseline-cuda":
            model.to(device); model.eval()
            predict = lambda canonical: predict_dnn(
                model,
                canonical,
                payload["output_minimum"],
                payload["output_span"],
                batch_size=len(canonical),
            )
            backend_bindings = {"inference_backend": dnn_backend}
        elif dnn_backend in {"torch-persistent-cuda", "torch-compile-cuda"}:
            from cloud_column_reproduction.figure1.dnn_gpu import (
                build_cuda_dnn_predictor,
            )

            probe = np.linspace(
                -0.8, 0.8, 2 * config.parameter_box.dimension, dtype=np.float64
            ).reshape(2, config.parameter_box.dimension)
            model.eval()
            reference = predict_dnn(
                model,
                probe,
                payload["output_minimum"],
                payload["output_span"],
                batch_size=len(probe),
            )
            resource = build_cuda_dnn_predictor(
                model,
                payload["output_minimum"],
                payload["output_span"],
                backend=dnn_backend,
            )
            predict = resource
            candidate = resource(probe)
            if not np.all(np.isfinite(candidate)):
                resource.close()
                raise ValueError("CUDA DNN construction probe produced nonfinite values")
            np.testing.assert_allclose(
                candidate,
                reference,
                rtol=2.0e-5,
                atol=1.0e-6,
                err_msg="CUDA DNN backend disagrees with CPU evaluation",
            )
            if dnn_validation_audit is None:
                audit_path = None
            else:
                audit_path = Path(dnn_validation_audit)
                if not audit_path.is_absolute(): audit_path = repository / audit_path
                if not audit_path.is_file(): resource.close(); raise FileNotFoundError(audit_path)
            backend_bindings = {
                f"inference_{key}": str(value)
                for key, value in resource.metadata.items()
            }
            backend_bindings["cpu_gpu_construction_check"] = "passed"
            if audit_path is not None:
                backend_bindings.update(validation_audit_path=str(audit_path.relative_to(repository)), validation_audit_sha256=sha256_file(audit_path))
        else:
            raise ValueError(f"Unsupported CUDA DNN backend: {dnn_backend}")
        resources = () if resource is None else (resource,)
        predict, resources, backend_bindings = _paired_microbatch(
            predict,
            resources,
            backend_bindings,
            requests=gpu_microbatch_requests,
            window_seconds=gpu_microbatch_window_seconds,
            cuda_enabled=device.type == "cuda",
        )
        metadata = _common_metadata(
            config, method="dnn", artifact_sha256=section["model_sha256"],
            device=str(device), dtype="float32-model/float64-output", batch_size=32,
            workers=None, evaluation_mode="vectorized",
        )
        evaluator = DNNEvaluator(config.parameter_box, metadata, predict)
        return RuntimeEvaluator(
            evaluator, likelihood, True, metadata,
            {"model_sha256": section["model_sha256"], **backend_bindings},
            include_prediction_blobs=include_prediction_blobs,
            resources=resources,
        )
    if method == "pdr":
        from cloud_column_reproduction.figure1.polynomials import (
            hyperbolic_cross_indices,
            polynomial_basis_fingerprint,
        )

        section = config.raw["surrogates"]["pdr"]
        path = repository / section["model"]
        if sha256_file(path) != section["model_sha256"]:
            raise ValueError("Frozen PDR artifact hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
            indices = np.asarray(archive["indices"], dtype=np.int64)
            box_fingerprint = str(archive["parameter_box_sha256"])
            parameter_names = tuple(str(value) for value in archive["parameter_names"])
            output_names = tuple(str(value) for value in archive["output_names"])
            fingerprint = (
                str(archive["basis_fingerprint_sha256"])
                if "basis_fingerprint_sha256" in archive.files
                else polynomial_basis_fingerprint(11, "hyperbolic-cross", int(section["q"]))
            )
        basis_size = int(section["basis_size"])
        if coefficients.shape != (basis_size, 6) or indices.shape != (basis_size, 11):
            raise ValueError("Frozen PDR artifact has the wrong configured shape")
        if box_fingerprint != config.parameter_box.fingerprint:
            raise ValueError("Frozen PDR parameter-box fingerprint mismatch")
        if parameter_names != config.parameter_box.names or output_names != OUTPUT_ORDER:
            raise ValueError("Frozen PDR parameter or output order mismatch")
        expected_indices = hyperbolic_cross_indices(11, int(section["q"]))
        if not np.array_equal(indices, expected_indices):
            raise ValueError("Frozen PDR ordered indices do not match configured q")
        if fingerprint != section["basis_fingerprint_sha256"]:
            raise ValueError("Frozen PDR basis fingerprint mismatch")
        if pdr_device == "cpu":
            if pdr_backend != "numpy-cpu":
                raise ValueError("CPU PDR evaluation requires pdr_backend='numpy-cpu'")
            predict = lambda canonical: predict_pdr(
                canonical, coefficients, indices, chunk_size=max(1, len(canonical)),
                workers=pdr_workers, prune_exact_zeros=pdr_prune_exact_zeros,
            )
            resource = None
            device = "cpu"
            backend_bindings = {
                "inference_backend": "numpy-cpu",
                "inference_dtype": "float64",
                "prune_exact_zeros": str(bool(pdr_prune_exact_zeros)).lower(),
                "active_basis_size": str(
                    int(np.sum(np.any(coefficients != 0.0, axis=1)))
                    if pdr_prune_exact_zeros else basis_size
                ),
            }
        elif pdr_device == "cuda":
            from cloud_column_reproduction.figure1.pdr_gpu import (
                build_cuda_pdr_predictor,
            )

            if pdr_backend not in {
                "torch-cuda",
                "cupy-fused-cuda",
                "cupy-zero-skip-cuda",
                "cupy-cached-sparse-cuda",
                "cupy-cached-sparse-float32-cuda",
            }:
                raise ValueError("CUDA CS evaluation requires an explicit CUDA backend")
            probe = np.linspace(
                -0.8, 0.8, 2 * config.parameter_box.dimension, dtype=np.float64
            ).reshape(2, config.parameter_box.dimension)
            reference = predict_pdr(
                probe,
                coefficients,
                indices,
                chunk_size=len(probe),
                workers=1,
                prune_exact_zeros=pdr_prune_exact_zeros,
            )
            resource = build_cuda_pdr_predictor(
                coefficients,
                indices,
                backend=pdr_backend,
                prune_exact_zeros=pdr_prune_exact_zeros,
            )
            predict = resource
            device = f"cuda:0 ({resource.device_name})"
            candidate = resource(probe)
            if not np.all(np.isfinite(candidate)):
                resource.close()
                raise ValueError("CUDA CS construction probe produced nonfinite values")
            np.testing.assert_allclose(
                candidate,
                reference,
                rtol=2.0e-5 if "float32" in pdr_backend else 5.0e-8,
                atol=1.0e-6 if "float32" in pdr_backend else 1.0e-9,
                err_msg="CUDA CS backend disagrees with CPU evaluation",
            )
            precision_bindings = {}
            if "float32" in pdr_backend:
                # Test the actual frozen expansion, independently of MCMC timing.
                generator = np.random.default_rng(164032)
                audit_points = np.concatenate((
                    generator.uniform(-1.0, 1.0, (256, config.parameter_box.dimension)),
                    generator.choice([-1.0, 1.0], (64, config.parameter_box.dimension)),
                ))
                audit_reference = predict_pdr(
                    audit_points, coefficients, indices, chunk_size=64, workers=1,
                    prune_exact_zeros=pdr_prune_exact_zeros,
                )
                audit_candidate = resource(audit_points)
                noise_error = float(np.max(np.abs(audit_candidate - audit_reference) / config.sigma))
                if not np.isfinite(noise_error) or noise_error >= 1.0e-3:
                    resource.close()
                    raise ValueError("CS float32 validation exceeds 0.001 observation standard deviations")
                log_error = float(np.max(np.abs(
                    likelihood.log_likelihood(audit_candidate)
                    - likelihood.log_likelihood(audit_reference))))
                precision_bindings = {
                    "float32_cpu_reference_points": str(len(audit_points)),
                    "float32_max_prediction_delta_over_sigma": repr(noise_error),
                    "float32_max_absolute_log_likelihood_delta": repr(log_error),
                }
            if pdr_validation_audit is None:
                configured_size = config.raw.get("runtime", {}).get("pdr", {}).get("active_basis_size")
                configured_hash = config.raw.get("runtime", {}).get("pdr", {}).get("active_support_sha256")
                if configured_size != resource.active_basis_size or configured_hash != resource.active_support_sha256:
                    resource.close(); raise ValueError("CUDA CS active support differs from the materialized contract")
                audit_path = None
                audit = {
                    "status": "passed",
                    "selected_backends": {"cs": pdr_backend},
                    "models": {"cs": {"sha256": section["model_sha256"]}},
                    "results": {"cs": {pdr_backend: {
                        "validation_passed": True,
                        "metadata": {
                            "active_basis_size": str(resource.active_basis_size),
                            "active_support_sha256": resource.active_support_sha256,
                            "kernel_source_sha256": resource.kernel_source_sha256,
                        },
                    }}},
                }
            else:
                audit_path = Path(pdr_validation_audit)
            if audit_path is not None:
                if not audit_path.is_absolute():
                    audit_path = repository / audit_path
                if not audit_path.is_file():
                    resource.close()
                    raise FileNotFoundError(f"Missing CUDA CS validation audit: {audit_path}")
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit_backend = audit.get("results", {}).get("cs", {}).get(pdr_backend, {})
                if audit.get("status") != "passed" or audit_backend.get("validation_passed") is not True:
                    resource.close(); raise ValueError("CUDA CS validation audit did not pass")
            if audit_path is not None:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if "selected_backends" in audit:
                audit_backend = audit.get("results", {}).get("cs", {}).get(
                    pdr_backend, {}
                )
                observed = {
                    "status": audit.get("status"),
                    "selected_backend": audit.get("selected_backends", {}).get("cs"),
                    "model_sha256": audit.get("models", {}).get("cs", {}).get("sha256"),
                    "validation_passed": audit_backend.get("validation_passed"),
                    "active_basis_size": int(
                        audit_backend.get("metadata", {}).get(
                            "active_basis_size", -1
                        )
                    ),
                }
                expected = {
                    "status": "passed",
                    "selected_backend": pdr_backend,
                    "model_sha256": section["model_sha256"],
                    "validation_passed": True,
                    "active_basis_size": resource.active_basis_size,
                }
                audit_metadata = audit_backend.get("metadata", {})
            else:
                observed = {
                    "status": audit.get("status"),
                    "selected_backend": audit.get("recommended_backend"),
                    "model_sha256": audit.get("model_sha256"),
                    "validation_passed": audit.get("status") == "passed",
                    "active_basis_size": audit.get("active_basis_size"),
                }
                expected = {
                    "status": "passed",
                    "selected_backend": pdr_backend,
                    "model_sha256": section["model_sha256"],
                    "validation_passed": True,
                    "active_basis_size": resource.active_basis_size,
                }
                audit_metadata = audit.get("backend_metadata", {}).get(
                    pdr_backend, {}
                )
            mismatches = {
                key: (observed.get(key), value)
                for key, value in expected.items()
                if observed.get(key) != value
            }
            if audit_metadata.get("active_support_sha256") != resource.active_support_sha256:
                mismatches["active_support_sha256"] = (
                    audit_metadata.get("active_support_sha256"),
                    resource.active_support_sha256,
                )
            if audit_metadata.get("kernel_source_sha256") != resource.kernel_source_sha256:
                mismatches["kernel_source_sha256"] = (
                    audit_metadata.get("kernel_source_sha256"),
                    resource.kernel_source_sha256,
                )
            if mismatches:
                resource.close()
                raise ValueError(f"CUDA CS validation audit mismatch: {mismatches}")
            backend_bindings = {
                f"inference_{key}": str(value)
                for key, value in resource.metadata.items()
            }
            backend_bindings["cpu_gpu_construction_check"] = "passed"
            if audit_path is not None:
                backend_bindings.update(validation_audit_path=str(audit_path.relative_to(repository)), validation_audit_sha256=sha256_file(audit_path))
        else:
            raise ValueError("pdr_device must be 'cpu' or 'cuda'")
        resources = () if resource is None else (resource,)
        backend_bindings.update(precision_bindings if pdr_device != "cpu" else {})
        predict, resources, backend_bindings = _paired_microbatch(
            predict,
            resources,
            backend_bindings,
            requests=gpu_microbatch_requests,
            window_seconds=gpu_microbatch_window_seconds,
            cuda_enabled=pdr_device == "cuda",
        )
        metadata = _common_metadata(
            config, method="pdr", artifact_sha256=section["model_sha256"],
            device=device, dtype=("float32-model/float64-output" if "float32" in pdr_backend else "float64"), batch_size=32,
            workers=pdr_workers if pdr_device == "cpu" else None,
            evaluation_mode="vectorized",
        )
        evaluator = PDREvaluator(config.parameter_box, metadata, predict)
        return RuntimeEvaluator(
            evaluator, likelihood, True, metadata,
            {
                "model_sha256": section["model_sha256"],
                "q": str(section["q"]),
                "basis_size": str(section["basis_size"]),
                "basis_fingerprint_sha256": section["basis_fingerprint_sha256"],
                **backend_bindings,
            },
            include_prediction_blobs=include_prediction_blobs,
            resources=resources,
        )
    if method == "sccm":
        runner = runner_from_repository(
            repository, prebuilt=prebuilt_sccm, parameter_box=config.parameter_box
        )
        executor = ThreadPoolExecutor(max_workers=sccm_workers)
        metadata = _common_metadata(
            config, method="sccm", artifact_sha256=sha256_file(runner.executable),
            device="cpu-subprocess", dtype="float64", batch_size=32,
            workers=sccm_workers, evaluation_mode="persistent-thread-launched-scalar-subprocesses",
        )
        evaluator = SCCMEvaluator(
            config.parameter_box, metadata, runner.diagnostics_at_120_minutes, executor
        )
        hashes = {
            "executable_sha256": sha256_file(runner.executable),
            "namelist_sha256": sha256_file(runner.namelist),
            "rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "radiation_table_sha256": sha256_file(runner.radiation_table),
        }
        identity = hashlib.sha256(
            "\n".join(f"{key}={hashes[key]}" for key in sorted(hashes)).encode("ascii")
        ).hexdigest()
        return RuntimeEvaluator(
            evaluator, likelihood, True, metadata,
            {**hashes, "identity_sha256": identity},
            include_prediction_blobs=include_prediction_blobs,
            executor=executor,
        )
    raise ValueError(f"Unknown controlled MCMC method: {method}")
