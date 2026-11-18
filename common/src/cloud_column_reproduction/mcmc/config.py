"""Load and validate the controlled MCMC scientific specification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

from cloud_column_reproduction.domain import ParameterBox
from cloud_column_reproduction.provenance import sha256_file, validate_manifest


PARAMETER_ORDER = (
    "as", "bs", "ag", "bg", "N0r", "N0s", "N0g", "rhos", "rhog", "qc0", "qi0"
)
OUTPUT_ORDER = ("PCP", "ACC", "LWP", "IWP", "OLR", "OSR")
OUTPUT_UNITS = ("mm h^-1", "mm", "kg m^-2", "kg m^-2", "W m^-2", "W m^-2")
APPROVED_MASKS = {
    "radiation_only": (0, 0, 0, 0, 1, 1),
    "full_except_acc": (1, 0, 1, 1, 1, 1),
}
APPROVED_METHODS = ("sccm", "dnn", "pdr")
UNRESOLVED_PREFIXES = ("UNRESOLVED", "SET_")


@dataclass(frozen=True)
class ControlledMCMCConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def study_role(self) -> str:
        return str(self.raw.get("study", {}).get("role", ""))

    @property
    def parameter_box_path(self) -> Path:
        return (self.path.parent / self.raw["parameters"]["source"]).resolve()

    @property
    def parameter_box(self) -> ParameterBox:
        with self.parameter_box_path.open("rb") as stream:
            box = tomllib.load(stream)
        return ParameterBox.from_sequences(box["names"], box["lower"], box["upper"])

    @property
    def truth(self) -> np.ndarray:
        return np.asarray(self.raw["synthetic_truth"]["values"], dtype=np.float64)

    @property
    def sigma(self) -> np.ndarray:
        return np.asarray(self.raw["observations"]["error"]["sigma"], dtype=np.float64)

    @property
    def masks(self) -> dict[str, np.ndarray]:
        return {
            case["id"]: np.asarray(case["mask"], dtype=np.int8)
            for case in self.raw["likelihood_cases"]
        }

    @property
    def initialization_scales(self) -> np.ndarray:
        return np.asarray(self.raw["sampler"]["initialization"]["scales"], dtype=np.float64)

    def execution_cases(self) -> tuple[str, ...]:
        scope = self.raw.get("execution_scope", {})
        return tuple(scope.get("likelihood_cases", self.masks.keys()))

    def execution_methods(self) -> tuple[str, ...]:
        scope = self.raw.get("execution_scope", {})
        return tuple(scope.get("methods", APPROVED_METHODS))

    def ensemble_count(self, method: str) -> int:
        return int(self.raw["sampler"]["ensemble_counts"][method])

    def initialization_seeds(self, method: str) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["sampler"]["seeds"][method]["initialization"])

    def proposal_seeds(self, method: str) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["sampler"]["seeds"][method]["proposal"])

    def unresolved_fields(self) -> tuple[str, ...]:
        found: list[str] = []

        def visit(value: Any, path: str) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")
            elif isinstance(value, str) and value.startswith(UNRESOLVED_PREFIXES):
                found.append(path)

        visit(self.raw, "")
        return tuple(found)

    def require_production_ready(self) -> None:
        unresolved = self.unresolved_fields()
        if unresolved:
            raise ValueError(
                "Controlled MCMC configuration is blocked by unresolved fields: "
                + ", ".join(unresolved)
            )
        if not str(self.raw["status"]).startswith("production-approved"):
            raise ValueError("Controlled MCMC configuration is not marked production-approved")


def load_config(
    path: str | Path, *, validate_registered_artifacts: bool = True
) -> ControlledMCMCConfig:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("schema_version") != 1:
        raise ValueError("Controlled MCMC config must use schema_version = 1")
    if "extends" in raw:
        base_name = str(raw["extends"])
        if Path(base_name).name != base_name:
            raise ValueError("Controlled MCMC extends must name a config in the same directory")
        base_path = path.parent / base_name
        if sha256_file(base_path) != raw.get("extends_sha256"):
            raise ValueError("Controlled MCMC base-config checksum mismatch")
        with base_path.open("rb") as stream:
            base = tomllib.load(stream)
        if "extends" in base:
            raise ValueError("Nested controlled MCMC config inheritance is forbidden")
        raw = _deep_merge(base, raw)
    config = ControlledMCMCConfig(path=path, raw=raw)
    _validate(config)
    if validate_registered_artifacts:
        _validate_registered_bindings(config)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a small, checksum-bound config override."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_registered_bindings(config: ControlledMCMCConfig) -> None:
    repository = config.path.parent.parent
    raw = config.raw
    observation = repository / raw["observations"]["artifact"]
    expected_observation = raw["observations"]["artifact_sha256"]
    if not expected_observation.startswith(UNRESOLVED_PREFIXES):
        if not observation.is_file() or sha256_file(observation) != expected_observation:
            raise ValueError("Registered synthetic-observation artifact hash mismatch")
        observation_manifest = repository / raw["observations"]["manifest"]
        if not observation_manifest.is_file():
            raise ValueError("Registered synthetic-observation manifest is missing")
        manifest = json.loads(observation_manifest.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if (
            manifest.get("artifact") != "mcmc-synthetic-observation"
            or manifest.get("data") != raw["observations"]["artifact"]
            or manifest.get("data_sha256") != expected_observation
            or manifest.get("parameter_box_sha256") != config.parameter_box.fingerprint
        ):
            raise ValueError("Registered synthetic-observation manifest mismatch")
    for method in ("dnn", "pdr"):
        section = raw["surrogates"][method]
        if section["model_sha256"].startswith(UNRESOLVED_PREFIXES):
            continue
        model = repository / section["model"]
        expected = section["model_sha256"]
        if not model.is_file() or sha256_file(model) != expected:
            raise ValueError(f"Registered frozen {method.upper()} artifact hash mismatch")
    pdr = raw["surrogates"]["pdr"]
    if not pdr["model_sha256"].startswith(UNRESOLVED_PREFIXES):
        if (pdr.get("q"), pdr.get("basis_size"), pdr.get("basis_fingerprint_sha256")) != (
            23, 7569, "a22d9594d73d3b779dc08d87dc34470ef8d2a5fcb2c60e786acc8429c3eed76f"
        ):
            raise ValueError("Registered frozen q=23 PDR basis identity mismatch")
        from cloud_column_reproduction.figure1.polynomials import (
            hyperbolic_cross_indices,
            polynomial_basis_fingerprint,
        )
        expected_indices = hyperbolic_cross_indices(11, int(pdr["q"]))
        expected_fingerprint = polynomial_basis_fingerprint(
            11, "hyperbolic-cross", int(pdr["q"])
        )
        if len(expected_indices) != int(pdr["basis_size"]):
            raise ValueError("Configured PDR basis cardinality is inconsistent")
        if expected_fingerprint != pdr["basis_fingerprint_sha256"]:
            raise ValueError("Configured PDR basis fingerprint is inconsistent")
        with np.load(repository / pdr["model"], allow_pickle=False) as archive:
            indices = np.asarray(archive["indices"], dtype=np.int64)
            coefficients = np.asarray(archive["coefficients"])
            embedded = str(archive["basis_fingerprint_sha256"]) if (
                "basis_fingerprint_sha256" in archive.files
            ) else None
        if not np.array_equal(indices, expected_indices):
            raise ValueError("Frozen PDR ordered indices do not match the configured basis")
        if coefficients.shape != (int(pdr["basis_size"]), 6):
            raise ValueError("Frozen PDR coefficient matrix has the wrong shape")
        if embedded is not None and embedded != expected_fingerprint:
            raise ValueError("Frozen PDR embedded basis fingerprint mismatch")
    pdr_runtime = raw.get("runtime", {}).get("pdr")
    if pdr_runtime is not None:
        required_runtime = {
            "device": "cuda",
            "dtype": ("float32-model/float64-output" if pdr_runtime.get("backend") == "cupy-cached-sparse-float32-cuda" else "float64"),
            "prune_exact_zeros": True,
            "cpu_fallback_allowed": False,
        }
        if pdr_runtime.get("backend") not in {
            "cupy-fused-cuda", "cupy-cached-sparse-cuda", "cupy-cached-sparse-float32-cuda"
        }:
            raise ValueError("Controlled CUDA CS backend is not validation-approved")
        mismatches = {
            key: (pdr_runtime.get(key), expected)
            for key, expected in required_runtime.items()
            if pdr_runtime.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Controlled CUDA CS runtime mismatch: {mismatches}")
        if not isinstance(pdr_runtime.get("active_basis_size"), int):
            raise ValueError("Materialized CS runtime must record its active basis size")
        if len(str(pdr_runtime.get("active_support_sha256", ""))) != 64:
            raise ValueError("Materialized CS runtime must record its support checksum")
    dnn_runtime = raw.get("runtime", {}).get("dnn")
    if dnn_runtime is not None:
        expected_dnn = {
            "device": "cuda",
            "backend": "torch-compile-cuda",
            "dtype": "float32-model/float64-output",
            "cpu_fallback_allowed": False,
        }
        mismatches = {
            key: (dnn_runtime.get(key), expected)
            for key, expected in expected_dnn.items()
            if dnn_runtime.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Controlled CUDA DNN runtime mismatch: {mismatches}")
    execution = raw.get("runtime", {}).get("execution")
    if execution is not None:
        expected_execution = {
            "dnn_ensemble_concurrency": (2 if pdr_runtime and pdr_runtime.get("backend") == "cupy-cached-sparse-float32-cuda" else 1),
            "pdr_ensemble_concurrency": 2,
            "dnn_runtime_policy":
                ("one-persistent-compiled-runtime-paired-ensemble-microbatches"
                 if pdr_runtime and pdr_runtime.get("backend") == "cupy-cached-sparse-float32-cuda"
                 else "one-persistent-compiled-runtime-sequential-ensembles"),
            "pdr_runtime_policy":
                "one-persistent-cached-sparse-runtime-paired-ensemble-microbatches",
            "microbatch_window_us": 2000.0,
        }
        mismatches = {
            key: (execution.get(key), expected)
            for key, expected in expected_execution.items()
            if execution.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Controlled surrogate GPU execution policy mismatch: {mismatches}"
            )

def _validate(config: ControlledMCMCConfig) -> None:
    raw = config.raw
    if config.study_role != "production-study":
        raise ValueError("This repository accepts only the production MCMC study role")
    execution_cases = config.execution_cases()
    execution_methods = config.execution_methods()
    if not execution_cases or len(set(execution_cases)) != len(execution_cases):
        raise ValueError("Execution scope requires distinct likelihood cases")
    if not execution_methods or len(set(execution_methods)) != len(execution_methods):
        raise ValueError("Execution scope requires distinct methods")
    if not set(execution_cases).issubset(APPROVED_MASKS):
        raise ValueError("Execution scope contains an unapproved likelihood case")
    if not set(execution_methods).issubset(APPROVED_METHODS):
        raise ValueError("Execution scope contains an unapproved method")
    if tuple(raw["parameters"]["expected_order"]) != PARAMETER_ORDER:
        raise ValueError("MCMC expected parameter order is not authoritative")
    if config.parameter_box.names != PARAMETER_ORDER:
        raise ValueError("Parameter-box order does not match the controlled MCMC order")
    if tuple(raw["observations"]["names"]) != OUTPUT_ORDER:
        raise ValueError("MCMC output order must be PCP, ACC, LWP, IWP, OLR, OSR")
    if tuple(raw["observations"]["units"]) != OUTPUT_UNITS:
        raise ValueError("MCMC observation units do not match the approved units")
    if int(raw["observations"]["time_minutes"]) != 120:
        raise ValueError("Controlled observations must be evaluated at exactly 120 minutes")
    if raw["observations"]["error"]["entries_are"] != "standard-deviations":
        raise ValueError("Observation uncertainty entries must be standard deviations")
    if raw["observations"]["error"]["covariance_construction"] != "diag(sigma**2)":
        raise ValueError("Controlled covariance must be constructed as diag(sigma**2)")
    if config.truth.shape != (11,) or not config.parameter_box.contains(config.truth):
        raise ValueError("Synthetic truth must be an in-box 11-vector")
    if config.sigma.shape != (6,) or np.any(~np.isfinite(config.sigma)) or np.any(config.sigma <= 0):
        raise ValueError("Observation sigma must contain six positive finite values")
    if config.masks.keys() != APPROVED_MASKS.keys():
        raise ValueError("Exactly radiation_only and full_except_acc are required")
    for name, approved in APPROVED_MASKS.items():
        mask = tuple(int(x) for x in config.masks[name])
        if mask != approved:
            raise ValueError(f"Likelihood mask for {name} is not approved")
        if mask[1] != 0:
            raise ValueError("ACC must be excluded from both controlled cases")
    cases = raw["likelihood_cases"]
    if tuple(cases[0]["active_outputs"]) != ("OLR", "OSR"):
        raise ValueError("radiation_only active outputs are not approved")
    if tuple(cases[1]["active_outputs"]) != ("PCP", "LWP", "IWP", "OLR", "OSR"):
        raise ValueError("full_except_acc active outputs are not approved")
    sampler = raw["sampler"]
    coordinate_system = str(sampler.get("coordinate_system", ""))
    if coordinate_system != "box-logit":
        raise ValueError("The production sampler must use box-logit coordinates")
    expected = {
        "implementation": "emcee.EnsembleSampler",
        "walkers_per_ensemble": 32,
        "thinning": 1,
        "thinning_policy": "none",
    }
    for key, value in expected.items():
        if sampler[key] != value:
            raise ValueError(f"Controlled sampler field {key} must equal {value!r}")
    if sampler["move"] != "emcee.moves.WeightedMixture":
        raise ValueError("The production sampler must use the DE/snooker mixture")
    from .resumable import build_emcee_move

    build_emcee_move(sampler["move"], sampler.get("move_parameters", {}))
    if raw.get("study", {}).get("optimized_gpu_30k_300k") is not True:
        raise ValueError("The production study must declare the optimized 30k/300k contract")
    if sampler["burn_in_steps"] != 30000 or sampler["production_steps"] != 300000:
        raise ValueError("The production study requires 30k burn-in and 300k retained")
    if not sampler["discard_burn_in"] or not sampler["reset_before_production"]:
        raise ValueError("Burn-in must be discarded by resetting before production")
    expected_counts = {"sccm": 2, "dnn": 2, "pdr": 2}
    if sampler["ensemble_counts"] != expected_counts:
        raise ValueError(f"Controlled ensemble counts must equal {expected_counts}")
    if config.initialization_scales.shape != (11,) or np.any(config.initialization_scales <= 0):
        raise ValueError("Initialization scales must be eleven positive values")
    reference_initialization: tuple[int, ...] | None = None
    reference_proposals: tuple[int, ...] | None = None
    for method, count in expected_counts.items():
        initialization = config.initialization_seeds(method)
        proposal = config.proposal_seeds(method)
        hashes = raw["sampler"]["seeds"][method]["initial_position_sha256"]
        if len(initialization) != count or len(proposal) != count or len(hashes) != count:
            raise ValueError(f"{method} seed/hash schedule must contain {count} ensemble entries")
        if any(len(str(value)) != 64 for value in hashes):
            raise ValueError(f"{method} initial-position hashes must be SHA-256 values")
        if len(set(initialization)) != len(initialization) or len(set(proposal)) != len(proposal):
            raise ValueError(f"{method} independent ensembles require distinct seeds")
        if set(initialization).intersection(proposal):
            raise ValueError(f"{method} initialization and proposal seeds must differ")
        reference_initialization = initialization if reference_initialization is None else reference_initialization
        reference_proposals = proposal if reference_proposals is None else reference_proposals
        if initialization != reference_initialization or proposal != reference_proposals:
            raise ValueError(
                "SCCM, DNN, and PDR must use the same paired ensemble seed schedule"
            )
    initialization = sampler["initialization"]
    if initialization["law"] != "latin-hypercube-prior":
        raise ValueError("Production ensembles require dispersed Latin-hypercube starts")
    checkpointing = sampler["checkpointing"]
    expected_checkpointing = {
        "backend": "atomic-npz-blocks",
        "checkpoint_schema_version": 3,
        "block_transitions": 1000,
        "partial_block_policy": "discard-uncommitted-block-and-resume-last-atomic-state",
        "preserve_rng_state": True,
        "preserve_timing": True,
    }
    for key, expected_value in expected_checkpointing.items():
        if checkpointing[key] != expected_value:
            raise ValueError(f"Controlled checkpoint field {key} changed")
    expected_prefixes = [
        10000, 20000, 50000, 100000, 150000, 200000, 250000, 300000
    ]
    if checkpointing["diagnostic_prefixes"] != expected_prefixes:
        raise ValueError("Controlled diagnostic prefixes changed")
    diagnostics = raw["diagnostics"]
    if diagnostics["formal_methods"] != ["sccm", "dnn", "pdr"]:
        raise ValueError("Formal diagnostics must apply to SCCM, DNN, and PDR")
    if diagnostics["reference_method_exempt"] != "none":
        raise ValueError("No forward method may be exempt from the independent-run gate")
    required_reference_checks = {
        "rank_normalized_rhat", "folded_rhat", "bulk_ess", "tail_ess",
        "acceptance_fraction", "traces", "autocorrelation", "stability",
        "boundary_occupancy", "mode_occupancy", "artifact_completeness",
    }
    if set(diagnostics["reference_method_required_checks"]) != required_reference_checks:
        raise ValueError("SCCM reference-chain quality checks are incomplete")
    if float(diagnostics["rhat_threshold_strictly_less_than"]) != 1.01:
        raise ValueError("Approved rank-normalized/folded R-hat threshold is 1.01")
    if "chain axis contains the two independently-initialized ensembles" not in diagnostics["array_construction"]:
        raise ValueError("R-hat/ESS must use the approved independent-ensemble construction")
    if "never flattened or promoted to chains" not in diagnostics["array_construction"]:
        raise ValueError("Interacting walkers must never populate the chain dimension")
    approved_diagnostics = {
        "marginal_diagnostic_policy": "compute-rank-and-folded-rhat-and-bulk-tail-ess-for-each-fixed-walker-trajectory-across-two-independent-ensembles",
        "walker_aggregation": "maximum-rhat-and-minimum-ess-across-32-matched-walker-trajectories",
        "bulk_ess_estimator": "minimum-arviz-rank-normalized-bulk-ess-across-matched-walker-trajectories",
        "tail_ess_estimator": "minimum-across-walkers-of-arviz-quantile-ess-at-probabilities-0.05-and-0.95",
        "bulk_ess_threshold": 400, "tail_probability_levels": [0.05, 0.95],
        "tail_ess_threshold": 400, "report_cross_walker_dependence": True,
        "cross_walker_dependence_is_pass_fail": False,
        "mode_occupancy_method": "pooled-quantile-decile-bin-occupancy-by-independent-ensemble",
        "maximum_mode_occupancy_difference": 0.10,
    }
    for key, expected in approved_diagnostics.items():
        if diagnostics[key] != expected: raise ValueError(f"Approved diagnostic field {key} changed")
    cache = raw["prediction_cache"]
    if cache["policy"] != "method-aware-retained-production-only" or cache["burn_in_predictions"] != "never":
        raise ValueError("Prediction-cache policy changed")
    if cache["retained_state_alignment"] != "transition-walker-output" or cache["dtype"] != "float64":
        raise ValueError("SCCM prediction-cache representation changed")
    expected_cache_shape = [int(sampler["production_steps"]), 32, 6]
    if not cache["sccm"]["store"] or cache["sccm"]["shape_per_chain"] != expected_cache_shape:
        raise ValueError("SCCM must cache all retained float64 predictions")
    expected_samples_per_case = (
        int(sampler["production_steps"])
        * int(sampler["walkers_per_ensemble"])
        * expected_counts["sccm"]
    )
    if cache["sccm"]["sample_count_per_likelihood_case"] != expected_samples_per_case:
        raise ValueError("SCCM prediction-cache count is inconsistent with the sampler")
    if cache["dnn"]["store"] or cache["pdr"]["store"]:
        raise ValueError("DNN/PDR production predictions must be regenerated, not cached")
    artifacts = raw["artifacts"]
    if artifacts["chain_format"] != "netcdf4-hdf5" or artifacts["chain_backend"] != "h5netcdf":
        raise ValueError("Controlled chains must use the approved h5netcdf backend")
    scoped_counts = {
        method: (expected_counts[method] if method in execution_methods else 0)
        for method in APPROVED_METHODS
    }
    scoped_chain_count = len(execution_cases) * sum(scoped_counts.values())
    expected_artifacts = {
        "expected_likelihood_case_count": len(execution_cases),
        "expected_method_count": len(execution_methods),
        "expected_ensemble_counts_by_method": scoped_counts,
        "expected_chain_file_count": scoped_chain_count,
    }
    for key, value in expected_artifacts.items():
        if artifacts[key] != value:
            raise ValueError(f"Controlled artifact field {key} must equal {value}")
    derived_chain_count = len(execution_cases) * sum(scoped_counts.values())
    if artifacts["expected_chain_file_count"] != derived_chain_count:
        raise ValueError("Expected chain-file count is inconsistent with method ensemble counts")
    workload = raw["workload"]
    steps_per_ensemble = int(sampler["burn_in_steps"]) + int(sampler["production_steps"])
    sccm_run_count = (
        len(execution_cases) * expected_counts["sccm"]
        if "sccm" in execution_methods else 0
    )
    transition_proposals = (
        sccm_run_count * int(sampler["walkers_per_ensemble"]) * steps_per_ensemble
    )
    initial_evaluations = sccm_run_count * int(sampler["walkers_per_ensemble"])
    expected_workload = {
        "likelihood_case_count": len(execution_cases),
        "independent_ensemble_counts_by_method": scoped_counts,
        "steps_per_ensemble_including_burn_in": steps_per_ensemble,
        "sccm_run_count": sccm_run_count,
        "sccm_transition_proposals_total": transition_proposals,
        "sccm_initial_state_evaluations_total": initial_evaluations,
        "sccm_posterior_evaluations_upper_bound": transition_proposals + initial_evaluations,
    }
    for key, value in expected_workload.items():
        if workload[key] != value:
            raise ValueError(f"Controlled workload field {key} must equal {value}")
    derived_evaluations = (
        len(execution_cases)
        * scoped_counts["sccm"]
        * sampler["walkers_per_ensemble"]
        * workload["steps_per_ensemble_including_burn_in"]
    )
    if workload["sccm_transition_proposals_total"] != derived_evaluations:
        raise ValueError("SCCM workload is inconsistent with cases, ensembles, walkers, and steps")
    if workload["sccm_initial_state_evaluations_total"] != sccm_run_count * 32:
        raise ValueError("SCCM initial-state workload is inconsistent")
    derived_seconds = workload["sccm_posterior_evaluations_upper_bound"] / workload["sccm_throughput_evaluations_per_second"]
    if not math.isclose(workload["sccm_estimated_wall_seconds"], derived_seconds, rel_tol=1e-12):
        raise ValueError("SCCM runtime estimate is inconsistent with workload and throughput")
