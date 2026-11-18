"""Deterministic, checkpointed Figure 1 neural-network surrogate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import time
from typing import Any

import numpy as np


DNN_ALGORITHM_REVISION = "full-batch-complete-history-v3"
DNN_OPTIMIZED_ALGORITHM_REVISION = "full-batch-compiled-fused-gpu-selection-v1"
DNN_BATCH_MODE = "full"
DNN_VALIDATION_PATIENCE = "infinite"
DNN_VALIDATION_SELECTION_METRIC = "mean-mse-in-training-output-scale"
DNN_TRAINING_BACKEND_EAGER = "torch-eager-v1"
DNN_TRAINING_BACKEND_OPTIMIZED = "torch-compile-fused-cuda-v1"
DNN_DEFAULT_GPU_TRAINING_BACKEND = DNN_TRAINING_BACKEND_OPTIMIZED


@dataclass(frozen=True)
class DNNDiagnostics:
    algorithm_revision: str
    batch_mode: str
    epochs: int
    optimizer_steps: int
    final_training_mse: float
    validation_patience: str
    validation_selection_metric: str
    validation_interval_epochs: int
    validation_checks: int
    best_epoch: int
    best_validation_mse: float
    best_validation_per_output_mse: tuple[float, ...]
    final_validation_mse: float
    final_validation_per_output_mse: tuple[float, ...]
    restored_best_weights: bool
    stopped_early: bool
    checkpoint_epochs: tuple[int, ...]
    wall_seconds: float
    optimizer_compute_wall_seconds: float
    validation_wall_seconds: float
    checkpoint_wall_seconds: float
    progress_wall_seconds: float
    device: str
    deterministic_algorithms: bool


@dataclass(frozen=True)
class DNNValidationHistory:
    epochs: tuple[int, ...]
    mse: tuple[float, ...]
    per_output_mse: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DNNTrainingHistory:
    epochs: tuple[int, ...]
    optimizer_mse: tuple[float, ...]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for DNN fitting") from error
    return torch


def build_network(input_dim: int, output_dim: int, hidden_layers: int, width: int):
    """Build exactly ``hidden_layers`` dense Tanh layers."""
    torch = _torch()
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be positive")
    modules: list[Any] = [torch.nn.Linear(input_dim, width), torch.nn.Tanh()]
    for _ in range(hidden_layers - 1):
        modules.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
    modules.append(torch.nn.Linear(width, output_dim))
    return torch.nn.Sequential(*modules)


def trainable_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _initialize_weights(module: Any) -> None:
    torch = _torch()
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _device_state_dict(model: Any) -> dict[str, Any]:
    """Clone a model state without moving it off its current device."""
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _copy_model_state_(destination: dict[str, Any], model: Any) -> None:
    """Update a preallocated best-state buffer on the model's current device."""
    torch = _torch()
    source = model.state_dict()
    if destination.keys() != source.keys():
        raise RuntimeError("DNN best-state buffer does not match the model")
    with torch.no_grad():
        for key, value in source.items():
            destination[key].copy_(value.detach())


def _state_dict_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _mean_squared_errors(model: Any, inputs: Any, targets: Any) -> tuple[float, tuple[float, ...]]:
    torch = _torch()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        squared = (model(inputs) - targets) ** 2
        per_output = squared.mean(dim=0)
        total = squared.mean()
    if was_training:
        model.train()
    total_value = float(total.detach().cpu())
    per_output_values = tuple(
        float(value) for value in per_output.detach().cpu()
    )
    if not np.isfinite(total_value) or not np.all(np.isfinite(per_output_values)):
        raise RuntimeError("DNN loss became non-finite")
    return total_value, per_output_values


def _mean_squared_errors_optimized(
    model: Any, inputs: Any, targets: Any
) -> tuple[float, tuple[float, ...]]:
    """Evaluate all validation statistics with one device-to-host transfer."""
    torch = _torch()
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        squared = (model(inputs) - targets) ** 2
        statistics = torch.cat((squared.mean().reshape(1), squared.mean(dim=0)))
    if was_training:
        model.train()
    values = statistics.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("DNN loss became non-finite")
    return float(values[0]), tuple(float(value) for value in values[1:])


def fit_dnn(
    canonical_inputs: np.ndarray,
    targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    *,
    hidden_layers: int,
    width: int,
    epochs: int,
    learning_rate_initial: float,
    learning_rate_final: float,
    seed: int,
    checkpoint_interval: int,
    progress_interval: int | None = None,
    validation_interval: int,
    validation_min_delta: float,
    checkpoint_path: str | Path,
    use_gpu: bool,
    training_backend: str | None = None,
) -> tuple[
    Any,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    DNNDiagnostics,
    DNNTrainingHistory,
    DNNValidationHistory,
]:
    torch = _torch()
    inputs = np.asarray(canonical_inputs, dtype=np.float32)
    responses = np.asarray(targets, dtype=np.float32)
    validation_x = np.asarray(validation_inputs, dtype=np.float32)
    validation_y = np.asarray(validation_targets, dtype=np.float32)
    if inputs.ndim != 2 or responses.ndim != 2 or len(inputs) != len(responses):
        raise ValueError("DNN inputs and targets must be aligned matrices")
    if (
        validation_x.ndim != 2
        or validation_y.ndim != 2
        or len(validation_x) != len(validation_y)
        or validation_x.shape[1] != inputs.shape[1]
        or validation_y.shape[1] != responses.shape[1]
    ):
        raise ValueError("DNN validation inputs and targets must be aligned matrices")
    if checkpoint_interval < 1 or checkpoint_interval > epochs:
        raise ValueError("checkpoint_interval must be between one and epochs")
    progress_interval = checkpoint_interval if progress_interval is None else progress_interval
    if progress_interval < 1 or progress_interval > epochs:
        raise ValueError("progress_interval must be between one and epochs")
    if validation_interval < 1 or validation_interval > epochs:
        raise ValueError("validation_interval must be between one and epochs")
    if validation_min_delta != 0.0:
        raise ValueError("Controlled validation selection requires min_delta=0")
    if training_backend is None:
        training_backend = (
            DNN_DEFAULT_GPU_TRAINING_BACKEND
            if use_gpu
            else DNN_TRAINING_BACKEND_EAGER
        )
    if training_backend not in (
        DNN_TRAINING_BACKEND_EAGER,
        DNN_TRAINING_BACKEND_OPTIMIZED,
    ):
        raise ValueError(f"Unsupported DNN training backend: {training_backend}")
    optimized_backend = training_backend == DNN_TRAINING_BACKEND_OPTIMIZED
    if optimized_backend and not use_gpu:
        raise ValueError("The optimized DNN training backend requires CUDA")
    algorithm_revision = (
        DNN_OPTIMIZED_ALGORITHM_REVISION
        if optimized_backend
        else DNN_ALGORITHM_REVISION
    )
    if not (
        np.all(np.isfinite(inputs))
        and np.all(np.isfinite(responses))
        and np.all(np.isfinite(validation_x))
        and np.all(np.isfinite(validation_y))
    ):
        raise ValueError("DNN training and validation arrays must be finite")
    minimum = responses.min(axis=0)
    maximum = responses.max(axis=0)
    span = np.where(maximum > minimum, maximum - minimum, 1.0).astype(np.float32)
    scaled = (2.0 * (responses - minimum) / span - 1.0).astype(np.float32)
    validation_scaled = (
        2.0 * (validation_y - minimum) / span - 1.0
    ).astype(np.float32)
    inputs_sha256 = _array_sha256(inputs)
    targets_sha256 = _array_sha256(responses)
    validation_inputs_sha256 = _array_sha256(validation_x)
    validation_targets_sha256 = _array_sha256(validation_y)

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("DNN GPU execution was requested but CUDA is unavailable")
    device = torch.device("cuda:0" if use_gpu else "cpu")
    deterministic_seed = int(seed % (2**63 - 1))
    torch.manual_seed(deterministic_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(deterministic_seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = build_network(inputs.shape[1], responses.shape[1], hidden_layers, width)
    model.apply(_initialize_weights)
    model.to(device)
    if optimized_backend:
        if not hasattr(model, "compile"):
            raise RuntimeError("The optimized DNN backend requires torch.nn.Module.compile")
        model.compile(mode="reduce-overhead", dynamic=True)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate_initial, fused=True
        )
        validation_function = _mean_squared_errors_optimized
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_initial)
        validation_function = _mean_squared_errors
    gamma = (learning_rate_final / learning_rate_initial) ** (1.0 / max(epochs, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_function = torch.nn.MSELoss()
    input_tensor = torch.from_numpy(inputs).to(device)
    target_tensor = torch.from_numpy(scaled).to(device)
    validation_input_tensor = torch.from_numpy(validation_x).to(device)
    validation_target_tensor = torch.from_numpy(validation_scaled).to(device)
    training_loss_tensor = torch.empty(epochs, dtype=torch.float32, device=device)
    checkpoint = Path(checkpoint_path)
    completed_epoch = 0
    optimizer_steps = 0
    checkpoint_epochs: list[int] = []
    loaded_final_loss = float("nan")
    validation_epochs: list[int] = []
    validation_losses: list[float] = []
    validation_per_output_losses: list[tuple[float, ...]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_per_output: tuple[float, ...] = ()
    best_model_state: dict[str, Any] | None = None
    validation_wall_seconds = 0.0
    checkpoint_wall_seconds = 0.0
    progress_wall_seconds = 0.0

    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        for key, expected_value in {
            "algorithm_revision": algorithm_revision,
            "batch_mode": DNN_BATCH_MODE,
            "inputs_sha256": inputs_sha256,
            "targets_sha256": targets_sha256,
            "validation_inputs_sha256": validation_inputs_sha256,
            "validation_targets_sha256": validation_targets_sha256,
            "validation_patience": DNN_VALIDATION_PATIENCE,
            "validation_selection_metric": DNN_VALIDATION_SELECTION_METRIC,
        }.items():
            if saved.get(key) != expected_value:
                raise ValueError(
                    f"DNN checkpoint field {key!r} does not match the full-batch run"
                )
        if optimized_backend and saved.get("training_backend") != training_backend:
            raise ValueError("DNN checkpoint training backend does not match the run")
        expected = {
            "hidden_layers": hidden_layers,
            "width": width,
            "epochs": epochs,
            "seed": deterministic_seed,
            "sample_count": len(inputs),
            "validation_sample_count": len(validation_x),
            "validation_interval": validation_interval,
            "checkpoint_interval": checkpoint_interval,
            "progress_interval": progress_interval,
        }
        for key, value in expected.items():
            if int(saved[key]) != int(value):
                raise ValueError(f"DNN checkpoint field {key!r} does not match the run")
        for key, value in {
            "learning_rate_initial": learning_rate_initial,
            "learning_rate_final": learning_rate_final,
            "validation_min_delta": validation_min_delta,
        }.items():
            if float(saved[key]) != float(value):
                raise ValueError(f"DNN checkpoint field {key!r} does not match the run")
        if not np.array_equal(np.asarray(saved["output_minimum"]), minimum):
            raise ValueError("DNN checkpoint output scaling does not match the data")
        if not np.array_equal(np.asarray(saved["output_span"]), span):
            raise ValueError("DNN checkpoint output span does not match the data")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        completed_epoch = int(saved["completed_epoch"])
        optimizer_steps = int(saved["optimizer_steps"])
        if optimizer_steps != completed_epoch:
            raise ValueError(
                "Full-batch DNN checkpoint must contain exactly one optimizer step per epoch"
            )
        checkpoint_epochs = [int(value) for value in saved["checkpoint_epochs"]]
        loaded_final_loss = float(saved["final_training_mse"])
        saved_training_losses = np.asarray(
            saved["training_optimizer_mse"], dtype=np.float32
        )
        if saved_training_losses.shape != (completed_epoch,):
            raise ValueError("DNN checkpoint training-loss history is inconsistent")
        if not np.all(np.isfinite(saved_training_losses)):
            raise ValueError("DNN checkpoint contains non-finite training losses")
        training_loss_tensor[:completed_epoch] = torch.from_numpy(
            saved_training_losses
        ).to(device)
        validation_epochs = [int(value) for value in saved["validation_epochs"]]
        validation_losses = [float(value) for value in saved["validation_losses"]]
        validation_per_output_losses = [
            tuple(float(value) for value in row)
            for row in saved["validation_per_output_losses"]
        ]
        if not (
            len(validation_epochs)
            == len(validation_losses)
            == len(validation_per_output_losses)
        ):
            raise ValueError("DNN checkpoint validation history is inconsistent")
        if not validation_epochs or validation_epochs[0] != 0:
            raise ValueError("DNN checkpoint must include the epoch-zero validation loss")
        if any(
            right <= left for left, right in zip(validation_epochs, validation_epochs[1:])
        ):
            raise ValueError("DNN checkpoint validation epochs must be strictly increasing")
        if validation_epochs[-1] > completed_epoch:
            raise ValueError("DNN checkpoint validation history exceeds its completed epoch")
        best_epoch = int(saved["best_epoch"])
        best_validation_loss = float(saved["best_validation_mse"])
        best_validation_per_output = tuple(
            float(value) for value in saved["best_validation_per_output_mse"]
        )
        if optimized_backend:
            best_model_state = {
                key: value.detach().to(device).clone()
                for key, value in saved["best_model_state"].items()
            }
        else:
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in saved["best_model_state"].items()
            }
    else:
        validation_started = time.perf_counter()
        initial_validation, initial_per_output = validation_function(
            model, validation_input_tensor, validation_target_tensor
        )
        validation_wall_seconds += time.perf_counter() - validation_started
        validation_epochs.append(0)
        validation_losses.append(initial_validation)
        validation_per_output_losses.append(initial_per_output)
        best_validation_loss = initial_validation
        best_validation_per_output = initial_per_output
        best_model_state = (
            _device_state_dict(model) if optimized_backend else _cpu_state_dict(model)
        )

    if best_model_state is None:
        raise RuntimeError("DNN best-weight state was not initialized")

    if use_gpu:
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    final_loss = loaded_final_loss
    for epoch_index in range(completed_epoch, epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(input_tensor)
        loss = loss_function(prediction, target_tensor)
        loss.backward()
        optimizer.step()
        training_loss_tensor[epoch_index] = loss.detach()
        optimizer_steps += 1
        scheduler.step()
        completed_epoch = epoch_index + 1
        if completed_epoch % validation_interval == 0 or completed_epoch == epochs:
            validation_started = time.perf_counter()
            validation_loss, validation_per_output = validation_function(
                model, validation_input_tensor, validation_target_tensor
            )
            validation_wall_seconds += time.perf_counter() - validation_started
            validation_epochs.append(completed_epoch)
            validation_losses.append(validation_loss)
            validation_per_output_losses.append(validation_per_output)
            if validation_loss < best_validation_loss - validation_min_delta:
                best_epoch = completed_epoch
                best_validation_loss = validation_loss
                best_validation_per_output = validation_per_output
                if optimized_backend:
                    _copy_model_state_(best_model_state, model)
                else:
                    best_model_state = _cpu_state_dict(model)
        if completed_epoch % checkpoint_interval == 0 or completed_epoch == epochs:
            checkpoint_started = time.perf_counter()
            final_loss, _ = validation_function(model, input_tensor, target_tensor)
            checkpoint_epochs.append(completed_epoch)
            state = {
                "algorithm_revision": algorithm_revision,
                "batch_mode": DNN_BATCH_MODE,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "completed_epoch": completed_epoch,
                "optimizer_steps": optimizer_steps,
                "checkpoint_epochs": checkpoint_epochs,
                "hidden_layers": hidden_layers,
                "width": width,
                "epochs": epochs,
                "seed": deterministic_seed,
                "sample_count": len(inputs),
                "validation_sample_count": len(validation_x),
                "learning_rate_initial": learning_rate_initial,
                "learning_rate_final": learning_rate_final,
                "validation_interval": validation_interval,
                "checkpoint_interval": checkpoint_interval,
                "progress_interval": progress_interval,
                "validation_min_delta": validation_min_delta,
                "validation_patience": DNN_VALIDATION_PATIENCE,
                "validation_selection_metric": DNN_VALIDATION_SELECTION_METRIC,
                "inputs_sha256": inputs_sha256,
                "targets_sha256": targets_sha256,
                "validation_inputs_sha256": validation_inputs_sha256,
                "validation_targets_sha256": validation_targets_sha256,
                "output_minimum": minimum,
                "output_span": span,
                "final_training_mse": final_loss,
                "training_optimizer_mse": (
                    training_loss_tensor[:completed_epoch].detach().cpu().numpy()
                ),
                "validation_epochs": validation_epochs,
                "validation_losses": validation_losses,
                "validation_per_output_losses": validation_per_output_losses,
                "best_epoch": best_epoch,
                "best_validation_mse": best_validation_loss,
                "best_validation_per_output_mse": best_validation_per_output,
                "best_model_state": (
                    _state_dict_to_cpu(best_model_state)
                    if optimized_backend
                    else best_model_state
                ),
            }
            if optimized_backend:
                state["training_backend"] = training_backend
            _atomic_torch_save(checkpoint, state)
            checkpoint_wall_seconds += time.perf_counter() - checkpoint_started
        if completed_epoch % progress_interval == 0 or completed_epoch == epochs:
            progress_started = time.perf_counter()
            reported_training_mse = float(loss.detach().cpu())
            progress_wall_seconds += time.perf_counter() - progress_started
            print(
                f"DNN epochs: {completed_epoch}/{epochs} "
                f"optimizer_mse={reported_training_mse:.6e} "
                f"val_mse={validation_losses[-1]:.6e} "
                f"best_epoch={best_epoch} steps={optimizer_steps}",
                flush=True,
            )

    if use_gpu:
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    optimizer_compute_wall_seconds = max(
        0.0, wall_seconds - validation_wall_seconds - checkpoint_wall_seconds - progress_wall_seconds
    )
    if completed_epoch != epochs or optimizer_steps != epochs:
        raise RuntimeError("DNN validation selection may only finalize after all epochs")
    final_epoch_state = _cpu_state_dict(model)
    training_loss_values = (
        training_loss_tensor[:completed_epoch]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    final_validation_loss = validation_losses[-1]
    final_validation_per_output = validation_per_output_losses[-1]
    if validation_epochs[-1] != epochs:
        raise RuntimeError("Final DNN epoch was not included in validation tracking")
    model.load_state_dict(best_model_state)
    diagnostics = DNNDiagnostics(
        algorithm_revision=algorithm_revision,
        batch_mode=DNN_BATCH_MODE,
        epochs=completed_epoch,
        optimizer_steps=optimizer_steps,
        final_training_mse=final_loss,
        validation_patience=DNN_VALIDATION_PATIENCE,
        validation_selection_metric=DNN_VALIDATION_SELECTION_METRIC,
        validation_interval_epochs=validation_interval,
        validation_checks=len(validation_epochs),
        best_epoch=best_epoch,
        best_validation_mse=best_validation_loss,
        best_validation_per_output_mse=best_validation_per_output,
        final_validation_mse=final_validation_loss,
        final_validation_per_output_mse=final_validation_per_output,
        restored_best_weights=True,
        stopped_early=False,
        checkpoint_epochs=tuple(checkpoint_epochs),
        wall_seconds=wall_seconds,
        optimizer_compute_wall_seconds=optimizer_compute_wall_seconds,
        validation_wall_seconds=validation_wall_seconds,
        checkpoint_wall_seconds=checkpoint_wall_seconds,
        progress_wall_seconds=progress_wall_seconds,
        device=str(device),
        deterministic_algorithms=True,
    )
    history = DNNValidationHistory(
        epochs=tuple(validation_epochs),
        mse=tuple(validation_losses),
        per_output_mse=tuple(validation_per_output_losses),
    )
    training_history = DNNTrainingHistory(
        epochs=tuple(range(1, completed_epoch + 1)),
        optimizer_mse=tuple(float(value) for value in training_loss_values),
    )
    model.eval()
    return (
        model,
        minimum.astype(np.float64),
        span.astype(np.float64),
        final_epoch_state,
        diagnostics,
        training_history,
        history,
    )


def predict_dnn(
    model: Any,
    canonical_inputs: np.ndarray,
    output_minimum: np.ndarray,
    output_span: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    torch = _torch()
    device = next(model.parameters()).device
    inputs = np.asarray(canonical_inputs, dtype=np.float32)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    scaled = np.concatenate(predictions, axis=0)
    return (scaled + 1.0) * 0.5 * output_span + output_minimum


def final_checkpoint_payload(
    model: Any,
    output_minimum: np.ndarray,
    output_span: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "output_minimum": np.asarray(output_minimum),
        "output_span": np.asarray(output_span),
        **metadata,
    }


def state_checkpoint_payload(
    model_state: dict[str, Any],
    output_minimum: np.ndarray,
    output_span: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": {
            key: value.detach().cpu().clone() for key, value in model_state.items()
        },
        "output_minimum": np.asarray(output_minimum),
        "output_span": np.asarray(output_span),
        **metadata,
    }


def save_final_dnn(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    _atomic_torch_save(destination, payload)
    return destination


def diagnostics_dict(value: DNNDiagnostics) -> dict[str, Any]:
    return asdict(value)
