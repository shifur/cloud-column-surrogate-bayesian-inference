from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_legendre

import cloud_column_reproduction.figure1.dnn as dnn_module
from cloud_column_reproduction.figure1.dnn import (
    DNN_OPTIMIZED_ALGORITHM_REVISION,
    DNN_TRAINING_BACKEND_OPTIMIZED,
    build_network,
    fit_dnn,
    predict_dnn,
    trainable_parameter_count,
)
from cloud_column_reproduction.figure1.pdr import (
    _progress_report_due,
    exact_spectral_norm,
    fit_pdr,
    predict_pdr,
    spectral_norm_estimate,
    spectral_norm_estimate,
    table51_hyperparameters,
)
from cloud_column_reproduction.figure1.polynomials import (
    intrinsic_legendre_weights,
    legendre_design_matrix,
    polynomial_indices,
)


def test_vectorized_legendre_design_matches_definition() -> None:
    points = np.asarray([[-0.4, 0.2], [0.3, -0.7]])
    indices = np.asarray([[0, 0], [1, 0], [0, 2], [2, 1]])
    actual = legendre_design_matrix(points, indices)
    expected = np.empty_like(actual)
    for row, point in enumerate(points):
        for column, degrees in enumerate(indices):
            expected[row, column] = np.prod(
                [
                    np.sqrt(2 * degree + 1) * eval_legendre(degree, coordinate)
                    for degree, coordinate in zip(degrees, point, strict=True)
                ]
            )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


def test_pdr_exact_zero_pruning_and_threading_preserve_predictions() -> None:
    generator = np.random.default_rng(903)
    inputs = generator.uniform(-1.0, 1.0, size=(19, 3))
    indices = polynomial_indices(3, "hyperbolic-cross", 7)
    coefficients = generator.normal(size=(len(indices), 6))
    coefficients[[1, 4, 9, 12]] = 0.0

    dense = predict_pdr(
        inputs,
        coefficients,
        indices,
        chunk_size=3,
        workers=1,
        prune_exact_zeros=False,
    )
    optimized = predict_pdr(
        inputs,
        coefficients,
        indices,
        chunk_size=3,
        workers=4,
        prune_exact_zeros=True,
    )

    np.testing.assert_allclose(optimized, dense, rtol=1.0e-14, atol=1.0e-14)


def test_pdr_prediction_rejects_invalid_parallel_settings() -> None:
    inputs = np.zeros((2, 1))
    coefficients = np.ones((1, 1))
    indices = np.zeros((1, 1), dtype=np.int64)

    with pytest.raises(ValueError, match="chunk size and worker count"):
        predict_pdr(inputs, coefficients, indices, chunk_size=0)
    with pytest.raises(ValueError, match="chunk size and worker count"):
        predict_pdr(inputs, coefficients, indices, workers=0)


def test_table51_restart_length_and_budget_follow_mems_book() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    contraction = float(np.exp(-1.0))
    values = table51_hyperparameters(
        matrix,
        sample_count=2,
        target_iterations=100,
        contraction=contraction,
        additive_tolerance=1.0e-8,
        spectral_norm_method="exact-svd",
        spectral_norm_safety_factor=1.0,
    )
    expected_inner = int(np.ceil(2.0 * values.spectral_norm / contraction))
    expected_restarts = int(np.ceil(100 / expected_inner))
    assert values.inner_iterations == expected_inner
    assert values.planned_restarts == expected_restarts
    assert values.planned_iterations == expected_restarts * expected_inner
    assert values.planned_iterations >= values.target_iterations
    assert values.planned_iterations - values.target_iterations < values.inner_iterations
    assert values.iteration_budget_policy == "ceil-full-restarts"
    assert values.restart_output == "ergodic-average"
    assert values.spectral_norm_method == "exact-svd"
    assert values.spectral_norm_raw_estimate == 2.0
    assert values.spectral_norm_safety_factor == 1.0
    assert values.spectral_norm == 2.0


def test_spectral_norm_estimators_are_reproducible_and_safely_inflated() -> None:
    generator = np.random.default_rng(218)
    matrix = generator.normal(size=(80, 40))
    exact = exact_spectral_norm(matrix)
    common = dict(safety_factor=1.01, seed=8675309)
    power, power_raw = spectral_norm_estimate(
        matrix, method="power-iteration", power_iterations=50, **common
    )
    repeated, repeated_raw = spectral_norm_estimate(
        matrix, method="power-iteration", power_iterations=50, **common
    )
    randomized, randomized_raw = spectral_norm_estimate(
        matrix,
        method="randomized-svd",
        randomized_oversamples=10,
        randomized_power_iterations=5,
        **common,
    )
    assert power == repeated
    assert power_raw == repeated_raw
    assert abs(power_raw / exact - 1.0) < 1.0e-2
    assert abs(randomized_raw / exact - 1.0) < 1.0e-2
    assert power >= exact
    assert randomized >= exact


def test_pdr_progress_reports_once_per_1000_iteration_boundary_and_at_end() -> None:
    assert not _progress_report_due(0, 11, 5005)
    assert _progress_report_due(999, 1010, 5005)
    assert not _progress_report_due(1010, 1021, 5005)
    assert _progress_report_due(1998, 2009, 5005)
    assert _progress_report_due(4994, 5005, 5005)


def test_pdr_batching_is_decoupled_and_uses_full_restarts(tmp_path: Path) -> None:
    generator = np.random.default_rng(14)
    inputs = generator.uniform(-1.0, 1.0, size=(18, 3))
    targets = generator.normal(size=(18, 2))
    common = dict(
        order=3,
        target_iterations=60,
        additive_tolerance=1.0e-8,
        contraction=float(np.exp(-1.0)),
        spectral_norm_method="exact-svd",
        spectral_norm_safety_factor=1.0,
        use_gpu=False,
        dtype=np.float64,
    )
    batch, indices, _, hyperparameters, diagnostics, history = fit_pdr(
        inputs, targets, checkpoint_path=tmp_path / "batch.npz", **common
    )
    columns = []
    for column in range(targets.shape[1]):
        fitted, single_indices, _, _, single_diagnostics, single_history = fit_pdr(
            inputs,
            targets[:, [column]],
            checkpoint_path=tmp_path / f"single-{column}.npz",
            **common,
        )
        assert np.array_equal(single_indices, indices)
        assert single_diagnostics.realized_iterations == hyperparameters.planned_iterations
        np.testing.assert_allclose(
            np.asarray(single_history.relative_primal_dual_gaps)[:, 0],
            np.asarray(history.relative_primal_dual_gaps)[:, column],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        columns.append(fitted[:, 0])
    np.testing.assert_allclose(batch, np.column_stack(columns), rtol=1.0e-12, atol=1.0e-12)
    assert diagnostics.realized_iterations == hyperparameters.planned_iterations
    assert diagnostics.realized_iterations >= 60
    assert hyperparameters.planned_restarts > 1
    assert set(diagnostics.restart_lengths) == {hyperparameters.inner_iterations}
    assert diagnostics.completed_restarts == hyperparameters.planned_restarts
    assert len(history.restarts) == hyperparameters.planned_restarts
    assert np.all(np.isfinite(history.primal_objectives))
    assert np.all(np.asarray(history.relative_primal_dual_gaps) >= 0.0)
    assert np.all(np.asarray(history.relative_residual_dual_gaps) >= 0.0)
    assert np.all(np.isfinite(history.residual_dual_objectives))
    assert np.all(np.asarray(history.relative_training_residuals) >= 0.0)
    assert np.all(np.asarray(history.relative_coefficient_changes) >= 0.0)
    assert diagnostics.relative_primal_dual_gaps == history.relative_primal_dual_gaps[-1]
    prediction = predict_pdr(inputs, batch, indices)
    assert prediction.shape == targets.shape

    resumed, _, _, _, resumed_diagnostics, resumed_history = fit_pdr(
        inputs, targets, checkpoint_path=tmp_path / "batch.npz", **common
    )
    np.testing.assert_array_equal(resumed, batch)
    assert resumed_diagnostics.realized_iterations == diagnostics.realized_iterations
    assert resumed_history == history

    with pytest.raises(ValueError, match="right_hand_side_sha256"):
        fit_pdr(
            inputs,
            targets + 1.0,
            checkpoint_path=tmp_path / "batch.npz",
            **common,
        )


def test_dnn_has_exactly_eight_hidden_layers_and_expected_parameters() -> None:
    model = build_network(11, 6, hidden_layers=8, width=50)
    linear_layers = [module for module in model if module.__class__.__name__ == "Linear"]
    assert len(linear_layers) == 9
    assert trainable_parameter_count(model) == 18_756


def test_dnn_uses_infinite_patience_and_restores_best_weights(tmp_path: Path) -> None:
    generator = np.random.default_rng(91)
    inputs = generator.uniform(-1.0, 1.0, size=(7, 3))
    targets = generator.normal(size=(7, 2))
    validation_inputs = generator.uniform(-1.0, 1.0, size=(9, 3))
    validation_targets = generator.normal(size=(9, 2))
    model, minimum, span, _, diagnostics, training_history, history = fit_dnn(
        inputs,
        targets,
        validation_inputs,
        validation_targets,
        hidden_layers=2,
        width=5,
        epochs=3,
        learning_rate_initial=1.0e-3,
        learning_rate_final=5.0e-7,
        seed=17,
        checkpoint_interval=2,
        validation_interval=1,
        validation_min_delta=0.0,
        checkpoint_path=tmp_path / "dnn-full-batch.pt",
        use_gpu=False,
    )
    assert diagnostics.batch_mode == "full"
    assert diagnostics.algorithm_revision == "full-batch-complete-history-v3"
    assert diagnostics.epochs == 3
    assert diagnostics.optimizer_steps == 3
    assert diagnostics.validation_patience == "infinite"
    assert diagnostics.stopped_early is False
    assert diagnostics.validation_checks == 4
    assert history.epochs == (0, 1, 2, 3)
    assert training_history.epochs == (1, 2, 3)
    assert len(training_history.optimizer_mse) == 3
    assert np.all(np.isfinite(training_history.optimizer_mse))
    best_index = int(np.argmin(history.mse))
    assert diagnostics.best_epoch == history.epochs[best_index]
    assert diagnostics.best_validation_mse == history.mse[best_index]
    restored_prediction = predict_dnn(model, validation_inputs, minimum, span)
    restored_scaled = 2.0 * (restored_prediction - minimum) / span - 1.0
    validation_scaled = 2.0 * (validation_targets - minimum) / span - 1.0
    restored_mse = float(np.mean((restored_scaled - validation_scaled) ** 2))
    assert restored_mse == pytest.approx(diagnostics.best_validation_mse, rel=2.0e-6)
    assert diagnostics.progress_wall_seconds >= 0.0

    with pytest.raises(ValueError, match="validation_targets_sha256"):
        fit_dnn(
            inputs,
            targets,
            validation_inputs,
            validation_targets + 1.0,
            hidden_layers=2,
            width=5,
            epochs=3,
            learning_rate_initial=1.0e-3,
            learning_rate_final=5.0e-7,
            seed=17,
            checkpoint_interval=2,
            validation_interval=1,
            validation_min_delta=0.0,
            checkpoint_path=tmp_path / "dnn-full-batch.pt",
            use_gpu=False,
        )


def test_dnn_final_only_checkpoint_is_independent_of_progress(tmp_path: Path) -> None:
    generator = np.random.default_rng(119)
    inputs = generator.normal(size=(8, 3))
    targets = generator.normal(size=(8, 2))
    validation_inputs = generator.normal(size=(7, 3))
    validation_targets = generator.normal(size=(7, 2))
    result = fit_dnn(
        inputs, targets, validation_inputs, validation_targets,
        hidden_layers=2, width=5, epochs=4,
        learning_rate_initial=1.0e-3, learning_rate_final=5.0e-7, seed=33,
        checkpoint_interval=4, progress_interval=2, validation_interval=2,
        validation_min_delta=0.0, checkpoint_path=tmp_path / "final-only.pt",
        use_gpu=False,
    )
    assert result[4].checkpoint_epochs == (4,)
    assert result[4].optimizer_steps == 4
    assert result[5].epochs == (1, 2, 3, 4)


def test_optimized_dnn_backend_requires_cuda(tmp_path: Path) -> None:
    generator = np.random.default_rng(441)
    inputs = generator.normal(size=(8, 3))
    targets = generator.normal(size=(8, 2))
    with pytest.raises(ValueError, match="requires CUDA"):
        fit_dnn(
            inputs, targets, inputs, targets,
            hidden_layers=2, width=5, epochs=2,
            learning_rate_initial=1.0e-3, learning_rate_final=5.0e-7,
            seed=7, checkpoint_interval=2, validation_interval=1,
            validation_min_delta=0.0, checkpoint_path=tmp_path / "optimized.pt",
            use_gpu=False, training_backend=DNN_TRAINING_BACKEND_OPTIMIZED,
        )


@pytest.mark.skipif(not dnn_module._torch().cuda.is_available(), reason="CUDA required")
def test_optimized_dnn_backend_completes_full_history_and_restores_best(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(442)
    inputs = generator.normal(size=(16, 3))
    targets = generator.normal(size=(16, 2))
    validation_inputs = generator.normal(size=(13, 3))
    validation_targets = generator.normal(size=(13, 2))
    common = dict(
        hidden_layers=2, width=5, epochs=4,
        learning_rate_initial=1.0e-3, learning_rate_final=5.0e-7,
        seed=9, checkpoint_interval=4, validation_interval=1,
        validation_min_delta=0.0, checkpoint_path=tmp_path / "optimized.pt",
        use_gpu=True,
    )
    result = fit_dnn(
        inputs, targets, validation_inputs, validation_targets, **common
    )
    diagnostics = result[4]
    assert diagnostics.algorithm_revision == DNN_OPTIMIZED_ALGORITHM_REVISION
    assert diagnostics.epochs == diagnostics.optimizer_steps == 4
    assert diagnostics.stopped_early is False
    assert diagnostics.best_validation_mse == min(result[6].mse)
    saved = dnn_module._torch().load(
        tmp_path / "optimized.pt", map_location="cpu", weights_only=False
    )
    assert saved["training_backend"] == DNN_TRAINING_BACKEND_OPTIMIZED
    resumed = fit_dnn(
        inputs, targets, validation_inputs, validation_targets, **common
    )
    assert resumed[4].optimizer_steps == 4
    assert resumed[5] == result[5]
    assert resumed[6] == result[6]
    for expected, actual in zip(
        result[0].state_dict().values(),
        resumed[0].state_dict().values(),
        strict=True,
    ):
        np.testing.assert_array_equal(
            expected.detach().cpu().numpy(), actual.detach().cpu().numpy()
        )


def test_dnn_resume_preserves_validation_history_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = np.random.default_rng(812)
    inputs = generator.normal(size=(10, 3))
    targets = generator.normal(size=(10, 2))
    validation_inputs = generator.normal(size=(8, 3))
    validation_targets = generator.normal(size=(8, 2))
    common = dict(
        hidden_layers=2,
        width=5,
        epochs=6,
        learning_rate_initial=1.0e-3,
        learning_rate_final=5.0e-7,
        seed=113,
        checkpoint_interval=2,
        validation_interval=1,
        validation_min_delta=0.0,
        use_gpu=False,
    )
    baseline = fit_dnn(
        inputs,
        targets,
        validation_inputs,
        validation_targets,
        checkpoint_path=tmp_path / "baseline.pt",
        **common,
    )

    original_save = dnn_module._atomic_torch_save
    interrupted = False

    def save_then_interrupt(path, payload):
        nonlocal interrupted
        original_save(path, payload)
        if payload["completed_epoch"] == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("planned interruption")

    monkeypatch.setattr(dnn_module, "_atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="planned interruption"):
        fit_dnn(
            inputs,
            targets,
            validation_inputs,
            validation_targets,
            checkpoint_path=tmp_path / "resumed.pt",
            **common,
        )
    monkeypatch.setattr(dnn_module, "_atomic_torch_save", original_save)
    resumed = fit_dnn(
        inputs,
        targets,
        validation_inputs,
        validation_targets,
        checkpoint_path=tmp_path / "resumed.pt",
        **common,
    )

    for expected, actual in zip(
        baseline[0].state_dict().values(), resumed[0].state_dict().values(), strict=True
    ):
        np.testing.assert_array_equal(expected.detach().numpy(), actual.detach().numpy())
    for key in baseline[3]:
        np.testing.assert_array_equal(baseline[3][key].numpy(), resumed[3][key].numpy())
    assert baseline[4].optimizer_steps == resumed[4].optimizer_steps == 6
    assert baseline[4].best_epoch == resumed[4].best_epoch
    assert baseline[5] == resumed[5]
    assert baseline[6] == resumed[6]


@pytest.mark.parametrize(
    ("spectral_method", "safety_factor"),
    (("exact-svd", 1.0), ("power-iteration", 1.01)),
)
def test_precomputed_float64_design_is_exactly_solver_equivalent(
    tmp_path: Path, spectral_method: str, safety_factor: float
) -> None:
    generator = np.random.default_rng(812)
    inputs = generator.uniform(-1.0, 1.0, size=(24, 3))
    targets = generator.normal(size=(24, 2))
    indices = polynomial_indices(3, "hyperbolic-cross", 5)
    design = legendre_design_matrix(inputs, indices, dtype=np.float64)
    weights = intrinsic_legendre_weights(indices)
    norm = spectral_norm_estimate(
        design / np.sqrt(float(len(inputs))), method=spectral_method,
        safety_factor=safety_factor
    )
    common = dict(
        order=5, target_iterations=40, additive_tolerance=1.0e-8,
        contraction=float(np.exp(-1.0)), use_gpu=False, dtype=np.float64,
        spectral_norm_method=spectral_method, spectral_norm_safety_factor=safety_factor,
    )
    direct = fit_pdr(inputs, targets, checkpoint_path=tmp_path / "direct.npz", **common)
    cached = fit_pdr(
        inputs, targets, checkpoint_path=tmp_path / "cached.npz",
        precomputed_unscaled_design=design, precomputed_indices=indices,
        precomputed_intrinsic_weights=weights, precomputed_spectral_norm=norm, **common,
    )
    np.testing.assert_array_equal(cached[0], direct[0])
    np.testing.assert_array_equal(cached[1], direct[1])
    np.testing.assert_array_equal(cached[2], direct[2])
    assert cached[3] == direct[3]
    assert direct[4].design_source == "computed-in-fit"
    assert cached[4].design_source == "checksum-validated-float64-prefix-cache"
    assert direct[4].spectral_norm_source == "computed-in-fit"
    assert cached[4].spectral_norm_source == "checksum-validated-prefix-cache"
