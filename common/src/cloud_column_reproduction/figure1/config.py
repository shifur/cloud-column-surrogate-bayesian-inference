"""Load and validate the controlled Figure 1 experiment specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from cloud_column_reproduction.domain import ParameterBox

from .polynomials import hyperbolic_cross_size
from .dnn import DNN_TRAINING_BACKEND_EAGER, DNN_TRAINING_BACKEND_OPTIMIZED


SIGNED_QUADRATURE_STATUS_PREFIX = "controlled-signed-quadrature"
RQMC_PRIMARY_STATUS_PREFIX = "controlled-rqmc-primary"


@dataclass(frozen=True)
class Figure1Config:
    path: Path
    raw: dict[str, Any]

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]

    @property
    def parameters(self) -> dict[str, Any]:
        with self.parameter_box_path.open("rb") as stream:
            raw = tomllib.load(stream)
        if raw.get("schema_version") != 1:
            raise ValueError("Parameter-box config must use schema_version = 1")
        return raw

    @property
    def parameter_box_path(self) -> Path:
        source = self.raw["parameters"].get("source")
        if not source:
            raise ValueError("Figure 1 must reference a shared parameter-box config")
        return (self.path.parent / source).resolve()

    @property
    def parameter_box(self) -> ParameterBox:
        return ParameterBox.from_sequences(
            self.parameters["names"], self.parameters["lower"], self.parameters["upper"]
        )

    @property
    def validation(self) -> dict[str, Any]:
        return self.raw["validation"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def sparse_grid_reference(self) -> dict[str, Any]:
        # The signed-quadrature contract makes the sparse grid the primary
        # evaluation section.  Retain this compatibility alias so historical
        # versioned configs and their manifests remain readable.
        return self.raw.get("sparse_grid_reference", self.raw["evaluation"])

    @property
    def monte_carlo_diagnostic(self) -> dict[str, Any]:
        # Historical configs called the Monte Carlo dataset ``evaluation``.
        # The corrected contract preserves it only as an optional diagnostic.
        return self.raw.get("monte_carlo_diagnostic", self.raw["evaluation"])

    @property
    def evaluation_kind(self) -> str:
        return str(self.evaluation.get("kind", ""))

    @property
    def is_rqmc_primary(self) -> bool:
        return self.evaluation_kind == "replicated-scrambled-sobol-rqmc"

    @property
    def is_tasmanian_evaluation(self) -> bool:
        return self.evaluation_kind == "tasmanian-global-sparse-grid"

    @property
    def cs(self) -> dict[str, Any]:
        return self.raw["cs"]

    @property
    def dnn(self) -> dict[str, Any]:
        return self.raw["dnn"]

    @property
    def plot(self) -> dict[str, Any]:
        return self.raw["plot"]

    @property
    def train_sizes(self) -> tuple[int, ...]:
        e = self.experiment
        return tuple(
            range(
                int(e["train_size_min"]),
                int(e["train_size_max"]) + 1,
                int(e["train_size_step"]),
            )
        )

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(self.experiment["outputs"])

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.plot["methods"])

    @property
    def uses_signed_quadrature_contract(self) -> bool:
        return str(self.raw.get("status", "")).startswith(
            SIGNED_QUADRATURE_STATUS_PREFIX
        )

    @property
    def uses_rqmc_primary_contract(self) -> bool:
        return str(self.raw.get("status", "")).startswith(
            RQMC_PRIMARY_STATUS_PREFIX
        )


def dnn_parameter_count(input_dim: int, output_dim: int, hidden_layers: int, width: int) -> int:
    """Return the parameter count for exactly ``hidden_layers`` dense hidden layers."""
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be positive")
    return (
        input_dim * width
        + width
        + (hidden_layers - 1) * (width * width + width)
        + width * output_dim
        + output_dim
    )


def pdr_spectral_norm_options(config: Figure1Config) -> dict[str, Any]:
    """Return explicit estimator settings, including legacy exact-SVD defaults."""
    cs = config.cs
    method = str(cs["spectral_norm_method"])
    return {
        "spectral_norm_method": method,
        "spectral_norm_safety_factor": float(
            cs.get("spectral_norm_safety_factor", 1.0 if method == "exact-svd" else 1.01)
        ),
        "spectral_norm_seed": int(cs.get("spectral_norm_seed", 8675309)),
        "spectral_norm_power_iterations": int(
            cs.get("spectral_norm_power_iterations", 50)
        ),
        "spectral_norm_randomized_oversamples": int(
            cs.get("spectral_norm_randomized_oversamples", 10)
        ),
        "spectral_norm_randomized_power_iterations": int(
            cs.get("spectral_norm_randomized_power_iterations", 5)
        ),
    }


def load_config(path: str | Path) -> Figure1Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    config = Figure1Config(path=config_path, raw=raw)
    validate_config(config)
    return config


def validate_config(config: Figure1Config) -> None:
    if config.raw.get("schema_version") != 1:
        raise ValueError("Figure 1 config must use schema_version = 1")

    parameter_box = config.parameter_box
    parameter_names = parameter_box.names
    if parameter_box.dimension != 11:
        raise ValueError("Figure 1 requires 11 named parameter bounds")
    if config.parameters["sampling_measure"] != "independent-uniform":
        raise ValueError("The controlled Figure 1 run requires an independent uniform box")

    invariants = config.raw["pipeline_invariants"]
    if not all(bool(value) for value in invariants.values()):
        raise ValueError("Every declared pipeline invariant must be enabled")

    if config.outputs != ("PCP", "ACC", "LWP", "IWP", "OLR", "OSR"):
        raise ValueError("Figure 1 must contain the six SCCM diagnostics in canonical order")
    if config.methods != ("CS", "DNN"):
        raise ValueError("The main Figure 1 comparison must contain exactly CS and DNN")
    is_preflight = config.experiment.get("role") == "production-path-preflight"
    minimum_trials = 1 if config.raw.get("status") == "smoke-test" or is_preflight else 2
    if int(config.experiment["trials"]) < minimum_trials:
        raise ValueError(f"At least {minimum_trials} trial(s) are required")
    if not config.train_sizes or config.train_sizes[-1] != int(config.experiment["train_size_max"]):
        raise ValueError("Training-size range does not include its declared maximum")

    if config.validation.get("kind") != "independent-uniform-monte-carlo":
        raise ValueError("DNN validation must use an independent uniform Monte Carlo set")
    if int(config.validation["dimension"]) != len(parameter_names):
        raise ValueError("Validation-set dimension does not match the parameter box")
    if int(config.validation["expected_points"]) < 1:
        raise ValueError("Validation set must contain at least one point")
    if not bool(config.validation["used_for_model_selection"]):
        raise ValueError("The declared validation set must be used for model selection")
    if config.validation.get("seed_stage") == config.evaluation.get("seed_stage"):
        raise ValueError("Validation and evaluation sets must use distinct seed stages")

    evaluation = config.evaluation
    if int(evaluation["dimension"]) != len(parameter_names):
        raise ValueError("Evaluation dimension does not match the parameter box")
    if bool(evaluation["used_for_model_selection"]):
        raise ValueError("The evaluation set must not be used for model selection")

    if config.is_rqmc_primary:
        if not config.uses_rqmc_primary_contract:
            raise ValueError("The RQMC evaluation must use the controlled primary contract")
        required = {
            "generator_package": "QMCPy",
            "generator_version": "2.3",
            "generator_class": "DigitalNetB2",
            "randomization": "NUS",
            "order": "RADICAL INVERSE",
            "metric": "replicated-rqmc-relative-l2",
            "weight_policy": "equal-positive",
            "metric_status": "primary-figure1-error-metric",
            "uncertainty_method": (
                "paired-delete-one-replicate-jackknife-student-t"
            ),
        }
        for key, expected in required.items():
            if evaluation.get(key) != expected:
                raise ValueError(f"RQMC {key} must be {expected!r}")
        replications = int(evaluation["replications"])
        log2_points = int(evaluation["log2_points_per_replication"])
        points = int(evaluation["points_per_replication"])
        if replications < 2:
            raise ValueError("RQMC uncertainty requires at least two scrambles")
        if points != 2**log2_points:
            raise ValueError("RQMC points per scramble must be an exact power of two")
        if int(evaluation["expected_points"]) != replications * points:
            raise ValueError("RQMC total point count is inconsistent")
        prefixes = tuple(int(value) for value in evaluation["prefix_log2"])
        if not prefixes or tuple(sorted(set(prefixes))) != prefixes:
            raise ValueError("RQMC prefix powers must be unique and increasing")
        if prefixes[-1] != log2_points or prefixes[0] < 1:
            raise ValueError("RQMC prefixes must end at the full sample size")
        if int(evaluation.get("bits", 0)) < log2_points:
            raise ValueError("RQMC digital-net precision is too small")
        if float(evaluation.get("confidence_level", 0.0)) != 0.95:
            raise ValueError("The controlled RQMC interval must use 95% confidence")
    elif config.is_tasmanian_evaluation:
        sparse_grid = config.sparse_grid_reference
        if config.uses_signed_quadrature_contract and (
            config.evaluation.get("kind") != "tasmanian-global-sparse-grid"
        ):
            raise ValueError("The primary Figure 1 evaluation must be the Tasmanian grid")
        if config.evaluation.get("metric") != "signed-quadrature-relative-l2":
            raise ValueError("Tasmanian evaluation must use signed quadrature")
        if config.evaluation.get("weight_policy") != "signed-tasmanian-unchanged":
            raise ValueError("Tasmanian weights must be used unchanged")
        allowed_status = {
            "primary-figure1-error-metric",
            "secondary-sparse-grid-adequacy-check",
        }
        if config.evaluation.get("metric_status") not in allowed_status:
            raise ValueError("Tasmanian metric status must identify its role")
        if config.evaluation.get("metric_status") == "secondary-sparse-grid-adequacy-check":
            if config.evaluation.get("rule") != "gauss-patterson":
                raise ValueError("The secondary sparse-grid target must use Gauss-Patterson")
            if config.evaluation.get("depth_type") != "level":
                raise ValueError("Gauss-Patterson must use raw level hierarchy, not qptotal")
        if int(config.evaluation["level"]) < 1:
            raise ValueError("The signed sparse-grid level must be positive")
        reuse_data = config.evaluation.get("nested_reuse_data")
        if reuse_data is not None and not str(reuse_data).endswith(".npz"):
            raise ValueError("Nested sparse-grid reuse must reference an NPZ artifact")
        if reuse_data is not None:
            reused = int(config.evaluation["nested_reuse_expected_points"])
            new = int(config.evaluation["expected_new_points"])
            if reused < 1 or new < 1:
                raise ValueError("Nested reuse point counts must be positive")
            if reused + new != int(config.evaluation["expected_points"]):
                raise ValueError("Nested reused and new points must sum to the target grid")
        if config.uses_signed_quadrature_contract:
            monte_carlo = config.monte_carlo_diagnostic
            if monte_carlo.get("kind") != "independent-uniform-monte-carlo":
                raise ValueError("The secondary diagnostic must be uniform Monte Carlo")
            if monte_carlo.get("metric") != "equal-weight-relative-l2":
                raise ValueError("The Monte Carlo diagnostic must use equal-weight L2")
            if monte_carlo.get("weight_policy") != "equal-positive":
                raise ValueError("The Monte Carlo diagnostic must use equal positive weights")
            if monte_carlo.get("metric_status") != "secondary-robustness-diagnostic":
                raise ValueError("Monte Carlo must be explicitly marked as secondary")
    else:
        # Compatibility validation for the superseded equal-weight contract.
        if bool(config.evaluation["used_for_model_selection"]):
            raise ValueError("The reported Figure 1 evaluation set must remain untouched")
        if int(config.evaluation["dimension"]) != len(parameter_names):
            raise ValueError("Evaluation-set dimension does not match the parameter box")
        if config.evaluation["metric"] != "equal-weight-relative-l2":
            raise ValueError("The historical controlled metric must use equal test weights")
        if config.evaluation["weight_policy"] != "equal-positive":
            raise ValueError("The historical controlled test set must use equal weights")

    order = int(config.cs["hyperbolic_cross_order"])
    actual_basis_size = hyperbolic_cross_size(len(parameter_names), order)
    expected_basis_size = int(config.cs["expected_basis_size"])
    if actual_basis_size != expected_basis_size:
        raise ValueError(
            f"Hyperbolic-cross size is {actual_basis_size}, expected {expected_basis_size}"
        )
    if config.is_rqmc_primary and not is_preflight:
        expected_policy = (
            "predeclared-smallest-hc-basis-strictly-above-max-m-not-rqmc-tuned"
        )
        if config.cs.get("index_set_selection_policy") != expected_policy:
            raise ValueError("RQMC config does not record the production q=23 policy")
        if order != 23 or expected_basis_size != 7569:
            raise ValueError("Current production RQMC requires q=23 with N=7,569")
        cache = config.cs.get("offline_cache", {})
        expected_cache = {
            "enabled": True,
            "artifact_role": "regenerable-performance-cache-not-scientific-result",
            "design_dtype": "float64",
            "maximum_rows_per_trial": 5000,
            "reuse_nested_row_prefixes": True,
            "cache_configured_spectral_norms": True,
            "parallelization": "independent-trials",
        }
        if any(cache.get(key) != value for key, value in expected_cache.items()):
            raise ValueError("Current q=23 RQMC config lacks the offline CS cache contract")
        if int(cache.get("default_processes", 0)) < 1:
            raise ValueError("Offline CS cache process count must be positive")
    if not bool(config.cs["record_realized_iterations"]):
        raise ValueError("PDR realized iteration counts must be recorded")
    if int(config.cs["target_pd_iterations"]) < 1:
        raise ValueError("PDR iteration budget must be positive")
    if config.cs.get("iteration_budget_policy") != "ceil-full-restarts":
        raise ValueError("PDR must realize its budget using complete restarts")
    if config.cs.get("restart_output") != "ergodic-average":
        raise ValueError("PDR restarts must use the MEMS-book ergodic output")
    if float(config.cs["restart_additive_tolerance"]) <= 0.0:
        raise ValueError("PDR restart additive tolerance must be positive")
    if config.cs["compute_dtype"] != "float32":
        raise ValueError("The controlled PDR GPU run is pinned to float32")
    norm_options = pdr_spectral_norm_options(config)
    if norm_options["spectral_norm_method"] not in {
        "exact-svd",
        "power-iteration",
        "randomized-svd",
    }:
        raise ValueError("The controlled PDR spectral-norm method is unsupported")
    if norm_options["spectral_norm_safety_factor"] < 1.0:
        raise ValueError("PDR spectral-norm safety factor must be at least one")
    if norm_options["spectral_norm_power_iterations"] != 50:
        raise ValueError("The default PDR power estimate must use 50 iterations")
    if int(config.cs.get("spectral_norm_randomized_components", 1)) != 1:
        raise ValueError("Only the leading randomized singular value is required")
    if norm_options["spectral_norm_randomized_oversamples"] < 10:
        raise ValueError("Randomized SVD must use at least ten oversamples")
    if norm_options["spectral_norm_randomized_power_iterations"] < 5:
        raise ValueError("Randomized SVD must use at least five power iterations")
    if config.cs.get("spectral_norm_randomized_power_normalizer", "auto") != "auto":
        raise ValueError("Randomized SVD power normalization must be automatic")

    hidden_layers = int(config.dnn["hidden_layers"])
    width = int(config.dnn["width"])
    actual_parameters = dnn_parameter_count(
        len(parameter_names), len(config.outputs), hidden_layers, width
    )
    expected_parameters = int(config.dnn["expected_trainable_parameters"])
    if actual_parameters != expected_parameters:
        raise ValueError(
            f"DNN has {actual_parameters} parameters, expected {expected_parameters}"
        )
    if int(config.dnn["epochs"]) < 1:
        raise ValueError("DNN epoch budget must be positive")
    if config.dnn.get("batch_mode") != "full" or "batch_size" in config.dnn:
        raise ValueError("Controlled DNN fits must use one full batch per epoch")
    if not bool(config.dnn["deterministic_algorithms"]):
        raise ValueError("Controlled DNN runs require deterministic algorithms")
    checkpoint_interval = int(config.dnn["checkpoint_interval_epochs"])
    if checkpoint_interval < 1 or checkpoint_interval > int(config.dnn["epochs"]):
        raise ValueError("DNN checkpoint interval must be between one and the epoch budget")
    progress_interval = int(config.dnn.get("progress_interval_epochs", checkpoint_interval))
    if progress_interval < 1 or progress_interval > int(config.dnn["epochs"]):
        raise ValueError("DNN progress interval must be between one and the epoch budget")
    validation_interval = int(config.dnn["validation_interval_epochs"])
    if validation_interval < 1 or validation_interval > int(config.dnn["epochs"]):
        raise ValueError("DNN validation interval must be between one and the epoch budget")
    if config.dnn.get("validation_patience") != "infinite":
        raise ValueError("Controlled DNN fits must use infinite validation patience")
    if config.dnn.get("validation_selection_metric") != (
        "mean-mse-in-training-output-scale"
    ):
        raise ValueError("Controlled DNN validation must use the declared scaled MSE")
    if float(config.dnn["validation_min_delta"]) != 0.0:
        raise ValueError("Controlled DNN validation min_delta must be exactly zero")
    training_backend = config.dnn.get("training_backend")
    if training_backend is not None and training_backend not in (
        DNN_TRAINING_BACKEND_EAGER,
        DNN_TRAINING_BACKEND_OPTIMIZED,
    ):
        raise ValueError(f"Unsupported DNN training backend: {training_backend}")


def work_plan(config: Figure1Config) -> dict[str, int | bool]:
    trials = int(config.experiment["trials"])
    sizes = config.train_sizes
    nested = bool(config.experiment["nested_training_subsets"])
    if nested:
        training_forward_runs = trials * max(sizes)
    else:
        training_forward_runs = trials * sum(sizes)
    corrected_contract = config.uses_signed_quadrature_contract
    evaluation_grid_points = int(config.evaluation["expected_points"])
    nested_reused_evaluations = int(
        config.evaluation.get("nested_reuse_expected_points", 0)
    )
    primary_evaluation_runs = int(
        config.evaluation.get("expected_new_points", evaluation_grid_points)
    )
    optional_monte_carlo_runs = (
        int(config.monte_carlo_diagnostic["expected_points"])
        if corrected_contract
        else 0
    )
    plan = {
        "trials": trials,
        "training_sizes": len(sizes),
        "nested_training_subsets": nested,
        "training_forward_runs": training_forward_runs,
        "validation_forward_runs": int(config.validation["expected_points"]),
        "evaluation_forward_runs": primary_evaluation_runs,
        "total_forward_runs": (
            training_forward_runs
            + int(config.validation["expected_points"])
            + primary_evaluation_runs
        ),
        "cs_scalar_fits": trials * len(sizes) * len(config.outputs),
        "dnn_fits": trials * len(sizes) * int(config.dnn["initializations_per_dataset"]),
    }
    if corrected_contract:
        plan["optional_monte_carlo_diagnostic_forward_runs"] = optional_monte_carlo_runs
    if config.is_rqmc_primary:
        plan["rqmc_replications"] = int(config.evaluation["replications"])
        plan["rqmc_points_per_replication"] = int(
            config.evaluation["points_per_replication"]
        )
    if nested_reused_evaluations:
        plan["evaluation_grid_points"] = evaluation_grid_points
        plan["nested_reused_evaluations"] = nested_reused_evaluations
    return plan
