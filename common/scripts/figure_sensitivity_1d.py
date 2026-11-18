#!/usr/bin/env python3
"""
Dedicated 1-D CRM1D sensitivity experiment.

For each selected physical parameter:
    1. Hold the other 10 CRM1D parameters fixed at the synthetic truth.
    2. Generate 150 uniformly distributed training points across the full
       prior range of that one parameter.
    3. Evaluate the true SCCM/CRM1D model.
    4. Train a fresh one-input DNN surrogate.
    5. Train a fresh one-input CS/PDR Legendre surrogate.
    6. Evaluate SCCM, DNN, and CS on a dense 1-D grid.
    7. Produce the historical-style sensitivity figure.

Default figure:
    columns : as, ag, rhog
    rows    : PCP, LWP, IWP, OLR

IMPORTANT
---------
This is a separate one-dimensional approximation experiment.
It is NOT an evaluation of the frozen 11-D DNN/CS checkpoints used in MCMC.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Repository setup
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


from cloud_column_reproduction.figure1.dnn import (
    fit_dnn,
    predict_dnn,
)

from cloud_column_reproduction.figure1.pdr import (
    fit_pdr,
    predict_pdr,
)

from cloud_column_reproduction.mcmc.config import (
    OUTPUT_ORDER,
    load_config,
)

from cloud_column_reproduction.sccm import runner_from_repository


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_CONFIG = ROOT / "configs" / "mcmc.toml"

DEFAULT_OUTPUT = (
    ROOT
    / "figures"
    / "sensitivity_comparison_1D_CRM_CS_DNN.png"
)

DEFAULT_PARAMETERS = (
    "as",
    "ag",
    "rhog",
)

DEFAULT_OUTPUTS = (
    "PCP",
    "LWP",
    "IWP",
    "OLR",
)


PARAMETER_LABELS = {
    "as": r"$a_s$",
    "bs": r"$b_s$",
    "ag": r"$a_g$",
    "bg": r"$b_g$",
    "N0r": r"$N_{0r}$",
    "N0s": r"$N_{0s}$",
    "N0g": r"$N_{0g}$",
    "rhos": r"$\rho_s$",
    "rhog": r"$\rho_g$",
    "qc0": r"$q_{c0}$",
    "qi0": r"$q_{i0}$",
}


# =============================================================================
# Utilities
# =============================================================================

def canonical_1d(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Map physical parameter values to [-1, 1].
    """

    values = np.asarray(values, dtype=np.float64)

    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")

    canonical = (
        2.0 * (values - lower) / (upper - lower)
        - 1.0
    )

    return canonical.reshape(-1, 1)


def physical_slice(
    truth: np.ndarray,
    parameter_index: int,
    parameter_values: np.ndarray,
) -> np.ndarray:
    """
    Construct 11-D CRM1D inputs while varying only one parameter.
    """

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    parameter_values = np.asarray(
        parameter_values,
        dtype=np.float64,
    )

    values = np.repeat(
        truth[None, :],
        len(parameter_values),
        axis=0,
    )

    values[:, parameter_index] = parameter_values

    return values


def evaluate_sccm(
    runner,
    physical_inputs: np.ndarray,
    workers: int,
) -> np.ndarray:
    """
    Evaluate SCCM in parallel.
    """

    if workers < 1:
        raise ValueError("workers must be positive")

    print(
        f"Running SCCM for {len(physical_inputs)} points "
        f"with {workers} workers...",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:

        results = list(
            pool.map(
                runner.diagnostics_at_120_minutes,
                physical_inputs,
            )
        )

    values = np.asarray(
        results,
        dtype=np.float64,
    )

    expected_shape = (
        len(physical_inputs),
        len(OUTPUT_ORDER),
    )

    if values.shape != expected_shape:
        raise RuntimeError(
            f"SCCM output shape {values.shape}; "
            f"expected {expected_shape}"
        )

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "SCCM produced non-finite values"
        )

    return values


def stage_seed(
    base_seed: int,
    parameter_index: int,
    stage: int,
) -> int:
    """
    Deterministic independent seeds.
    """

    return int(
        base_seed
        + 10000 * parameter_index
        + stage
    )


# =============================================================================
# Train one parameter
# =============================================================================

def train_parameter(
    *,
    config,
    runner,
    parameter_name: str,
    output_root: Path,
    training_points: int,
    validation_points: int,
    plot_points: int,
    workers: int,
    base_seed: int,
    dnn_use_gpu: bool,
    dnn_epochs: int,
    dnn_hidden_layers: int,
    dnn_width: int,
    pdr_degree: int,
    pdr_iterations: int,
):
    """
    Train dedicated one-dimensional DNN and CS surrogates for one parameter.
    """

    parameter_names = tuple(
        config.parameter_box.names
    )

    if parameter_name not in parameter_names:
        raise ValueError(
            f"Unknown parameter {parameter_name}"
        )

    parameter_index = parameter_names.index(
        parameter_name
    )

    lower = float(
        config.parameter_box.lower[
            parameter_index
        ]
    )

    upper = float(
        config.parameter_box.upper[
            parameter_index
        ]
    )

    truth_value = float(
        config.truth[
            parameter_index
        ]
    )

    print()
    print("=" * 75)
    print(
        f"Parameter: {parameter_name}"
    )
    print(
        f"Range    : {lower} -> {upper}"
    )
    print(
        f"Truth    : {truth_value}"
    )
    print("=" * 75)
    print()

    parameter_root = (
        output_root
        / parameter_name
    )

    parameter_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================================
    # Generate one-dimensional training data
    # =========================================================================

    training_rng = np.random.default_rng(
        stage_seed(
            base_seed,
            parameter_index,
            1,
        )
    )

    validation_rng = np.random.default_rng(
        stage_seed(
            base_seed,
            parameter_index,
            2,
        )
    )


    train_x = training_rng.uniform(
        lower,
        upper,
        size=training_points,
    )

    validation_x = validation_rng.uniform(
        lower,
        upper,
        size=validation_points,
    )

    grid_x = np.linspace(
        lower,
        upper,
        plot_points,
        dtype=np.float64,
    )


    # =========================================================================
    # Construct full 11-D SCCM inputs
    # =========================================================================

    train_physical = physical_slice(
        config.truth,
        parameter_index,
        train_x,
    )

    validation_physical = physical_slice(
        config.truth,
        parameter_index,
        validation_x,
    )

    grid_physical = physical_slice(
        config.truth,
        parameter_index,
        grid_x,
    )


    # =========================================================================
    # Run SCCM
    # =========================================================================

    print(
        f"[{parameter_name}] SCCM training data"
    )

    train_y = evaluate_sccm(
        runner,
        train_physical,
        workers,
    )


    print(
        f"[{parameter_name}] SCCM validation data"
    )

    validation_y = evaluate_sccm(
        runner,
        validation_physical,
        workers,
    )


    print(
        f"[{parameter_name}] SCCM dense plotting grid"
    )

    grid_truth = evaluate_sccm(
        runner,
        grid_physical,
        workers,
    )


    # =========================================================================
    # Convert varied parameter to canonical [-1, 1]
    # =========================================================================

    train_canonical = canonical_1d(
        train_x,
        lower,
        upper,
    )

    validation_canonical = canonical_1d(
        validation_x,
        lower,
        upper,
    )

    grid_canonical = canonical_1d(
        grid_x,
        lower,
        upper,
    )


    # =========================================================================
    # DNN
    # =========================================================================

    print()
    print(
        f"[{parameter_name}] Training 1-D DNN..."
    )
    print()


    dnn_checkpoint = (
        parameter_root
        / "dnn_training.checkpoint.pt"
    )


    (
        dnn_model,
        dnn_minimum,
        dnn_span,
        _,
        dnn_diagnostics,
        _,
        _,
    ) = fit_dnn(

        train_canonical,
        train_y,

        validation_canonical,
        validation_y,

        hidden_layers=dnn_hidden_layers,

        width=dnn_width,

        epochs=dnn_epochs,

        learning_rate_initial=1.0e-3,

        learning_rate_final=5.0e-7,

        seed=stage_seed(
            base_seed,
            parameter_index,
            3,
        ),

        checkpoint_interval=dnn_epochs,

        progress_interval=max(
            1,
            min(
                5000,
                dnn_epochs,
            ),
        ),

        validation_interval=max(
            1,
            min(
                100,
                dnn_epochs,
            ),
        ),

        validation_min_delta=0.0,

        checkpoint_path=dnn_checkpoint,

        use_gpu=dnn_use_gpu,
    )


    dnn_prediction = predict_dnn(

        dnn_model,

        grid_canonical,

        dnn_minimum,

        dnn_span,
    )


    # =========================================================================
    # CS / PDR
    # =========================================================================

    print()
    print(
        f"[{parameter_name}] Training 1-D CS/PDR..."
    )
    print()


    pdr_checkpoint = (
        parameter_root
        / "pdr_training.checkpoint.npz"
    )


    (
        pdr_coefficients,
        pdr_indices,
        pdr_weights,
        pdr_hyperparameters,
        pdr_diagnostics,
        _,
    ) = fit_pdr(

        train_canonical,

        train_y,

        order=pdr_degree,

        index_set="hyperbolic-cross",

        target_iterations=pdr_iterations,

        additive_tolerance=1.0e-8,

        contraction=float(
            np.exp(-1.0)
        ),

        use_gpu=False,

        dtype=np.float64,

        checkpoint_path=pdr_checkpoint,

        spectral_norm_method="exact-svd",

        spectral_norm_safety_factor=1.01,

        spectral_norm_seed=stage_seed(
            base_seed,
            parameter_index,
            4,
        ),
    )


    cs_prediction = predict_pdr(

        grid_canonical,

        pdr_coefficients,

        pdr_indices,

        chunk_size=max(
            1,
            len(grid_canonical),
        ),

        prune_exact_zeros=True,
    )


    # =========================================================================
    # Validation
    # =========================================================================

    expected_shape = (
        plot_points,
        len(OUTPUT_ORDER),
    )


    arrays = {
        "SCCM": grid_truth,
        "DNN": dnn_prediction,
        "CS": cs_prediction,
    }


    for name, array in arrays.items():

        array = np.asarray(array)

        if array.shape != expected_shape:

            raise RuntimeError(
                f"{name} shape for "
                f"{parameter_name}: "
                f"{array.shape}; "
                f"expected {expected_shape}"
            )

        if not np.all(
            np.isfinite(array)
        ):

            raise RuntimeError(
                f"{name} contains non-finite "
                f"values for {parameter_name}"
            )


    # =========================================================================
    # Save numerical data
    # =========================================================================

    data_path = (
        parameter_root
        / "sensitivity_1d_data.npz"
    )


    np.savez_compressed(

        data_path,

        parameter_name=np.asarray(
            parameter_name
        ),

        parameter_index=np.asarray(
            parameter_index
        ),

        lower=np.asarray(
            lower
        ),

        upper=np.asarray(
            upper
        ),

        truth=np.asarray(
            truth_value
        ),

        train_x=train_x,

        train_y=train_y,

        validation_x=validation_x,

        validation_y=validation_y,

        grid_x=grid_x,

        sccm=grid_truth,

        dnn=dnn_prediction,

        cs=cs_prediction,

        pdr_coefficients=pdr_coefficients,

        pdr_indices=pdr_indices,

        pdr_weights=pdr_weights,
    )


    print()
    print(
        f"[{parameter_name}] finished"
    )

    print(
        f"DNN best epoch: "
        f"{dnn_diagnostics.best_epoch}"
    )

    print(
        f"DNN best validation MSE: "
        f"{dnn_diagnostics.best_validation_mse:.6e}"
    )

    print(
        f"CS basis size: "
        f"{len(pdr_indices)}"
    )

    print(
        f"CS realized iterations: "
        f"{pdr_diagnostics.realized_iterations}"
    )


    return {

        "parameter":
            parameter_name,

        "parameter_index":
            parameter_index,

        "lower":
            lower,

        "upper":
            upper,

        "truth":
            truth_value,

        "grid_x":
            grid_x,

        "SCCM":
            grid_truth,

        "DNN":
            dnn_prediction,

        "CS":
            cs_prediction,

        "data_path":
            data_path,

        "dnn_checkpoint":
            dnn_checkpoint,

        "pdr_checkpoint":
            pdr_checkpoint,

        "dnn_best_epoch":
            int(
                dnn_diagnostics.best_epoch
            ),

        "dnn_best_validation_mse":
            float(
                dnn_diagnostics.best_validation_mse
            ),

        "pdr_basis_size":
            int(
                len(pdr_indices)
            ),

        "pdr_realized_iterations":
            int(
                pdr_diagnostics.realized_iterations
            ),

        "pdr_target_iterations":
            int(
                pdr_hyperparameters.target_iterations
            ),
    }


# =============================================================================
# Plot
# =============================================================================

def make_plot(
    results,
    output_names,
    output_path,
    dpi,
):
    """
    Historical-style 4 x 3 sensitivity plot.
    """

    output_indices = [

        OUTPUT_ORDER.index(
            output_name
        )

        for output_name
        in output_names
    ]


    n_rows = len(
        output_names
    )

    n_columns = len(
        results
    )


    plt.rcParams.update(
        {
            "font.family":
                "sans-serif",

            "font.size":
                10,

            "axes.labelsize":
                10,

            "xtick.labelsize":
                9,

            "ytick.labelsize":
                9,

            "legend.fontsize":
                9,
        }
    )


    figure, axes = plt.subplots(

        n_rows,

        n_columns,

        figsize=(
            10.5,
            10.2,
        ),

        squeeze=False,
    )


    styles = {

        "SCCM": {
            "color":
                "black",

            "linestyle":
                "-",

            "linewidth":
                1.5,
        },

        "DNN": {
            "color":
                "red",

            "linestyle":
                "--",

            "linewidth":
                1.35,
        },

        "CS": {
            "color":
                "#1f77b4",

            "linestyle":
                "-.",

            "linewidth":
                1.35,
        },
    }


    legend_labels = {

        "SCCM":
            "Cloud model",

        "DNN":
            "DNN",

        "CS":
            "CS",
    }


    for row, (
        output_name,
        output_index,
    ) in enumerate(

        zip(
            output_names,
            output_indices,
            strict=True,
        )
    ):

        for column, result in enumerate(
            results
        ):

            axis = axes[
                row,
                column,
            ]

            x = result[
                "grid_x"
            ]


            # Cloud / DNN / CS
            for method in (
                "SCCM",
                "DNN",
                "CS",
            ):

                axis.plot(

                    x,

                    result[
                        method
                    ][
                        :,
                        output_index,
                    ],

                    label=legend_labels[
                        method
                    ],

                    **styles[
                        method
                    ],
                )


            # truth parameter value
            axis.axvline(

                result[
                    "truth"
                ],

                color="black",

                linestyle=":",

                linewidth=1.0,

                alpha=0.8,
            )


            axis.set_xlabel(

                PARAMETER_LABELS.get(

                    result[
                        "parameter"
                    ],

                    result[
                        "parameter"
                    ],
                )
            )


            axis.set_ylabel(
                output_name
            )


            axis.margins(
                x=0.0
            )


            if (
                row == 0
                and column == 0
            ):

                axis.legend(
                    loc="best",
                    frameon=True,
                )


    figure.tight_layout(

        w_pad=2.0,

        h_pad=1.5,
    )


    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,
    )


    # PNG
    figure.savefig(

        output_path,

        dpi=dpi,

        bbox_inches="tight",
    )


    # PDF
    pdf_path = (
        output_path
        .with_suffix(
            ".pdf"
        )
    )


    figure.savefig(

        pdf_path,

        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    print()
    print(
        f"PNG saved: {output_path}"
    )

    print(
        f"PDF saved: {pdf_path}"
    )


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    results,
    output_names,
    path,
):
    """
    Save plotted numerical values.
    """

    rows = []


    for result in results:

        for output_name in output_names:

            output_index = (
                OUTPUT_ORDER.index(
                    output_name
                )
            )


            for point_index, x_value in enumerate(

                result[
                    "grid_x"
                ]
            ):

                rows.append(
                    {
                        "parameter":
                            result[
                                "parameter"
                            ],

                        "parameter_value":
                            float(
                                x_value
                            ),

                        "truth_parameter_value":
                            float(
                                result[
                                    "truth"
                                ]
                            ),

                        "output":
                            output_name,

                        "SCCM":
                            float(
                                result[
                                    "SCCM"
                                ][
                                    point_index,
                                    output_index,
                                ]
                            ),

                        "DNN":
                            float(
                                result[
                                    "DNN"
                                ][
                                    point_index,
                                    output_index,
                                ]
                            ),

                        "CS":
                            float(
                                result[
                                    "CS"
                                ][
                                    point_index,
                                    output_index,
                                ]
                            ),
                    }
                )


    path.parent.mkdir(

        parents=True,

        exist_ok=True,
    )


    with path.open(

        "w",

        newline="",

        encoding="utf-8",

    ) as stream:

        writer = csv.DictWriter(

            stream,

            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print(
        f"CSV saved: {path}"
    )


# =============================================================================
# Argument parser
# =============================================================================

def parser():
    result = argparse.ArgumentParser(
        description=__doc__
    )


    result.add_argument(

        "--config",

        type=Path,

        default=DEFAULT_CONFIG,
    )


    result.add_argument(

        "--parameters",

        nargs="+",

        default=list(
            DEFAULT_PARAMETERS
        ),
    )


    result.add_argument(

        "--outputs",

        nargs="+",

        default=list(
            DEFAULT_OUTPUTS
        ),
    )


    result.add_argument(

        "--training-points",

        type=int,

        default=150,
    )


    result.add_argument(

        "--validation-points",

        type=int,

        default=64,
    )


    result.add_argument(

        "--plot-points",

        type=int,

        default=201,
    )


    result.add_argument(

        "--workers",

        type=int,

        default=32,
    )


    result.add_argument(

        "--base-seed",

        type=int,

        default=20261023,
    )


    result.add_argument(

        "--dnn-epochs",

        type=int,

        default=50000,
    )


    result.add_argument(

        "--dnn-hidden-layers",

        type=int,

        default=8,
    )


    result.add_argument(

        "--dnn-width",

        type=int,

        default=50,
    )


    result.add_argument(

        "--pdr-degree",

        type=int,

        default=50,
    )


    result.add_argument(

        "--pdr-iterations",

        type=int,

        default=5000,
    )


    result.add_argument(

        "--device",

        choices=(
            "auto",
            "cpu",
            "cuda",
        ),

        default="auto",
    )


    result.add_argument(

        "--output",

        type=Path,

        default=DEFAULT_OUTPUT,
    )


    result.add_argument(

        "--dpi",

        type=int,

        default=300,
    )


    return result


# =============================================================================
# Main
# =============================================================================

def main():
    args = parser().parse_args()


    # =========================================================================
    # Basic checks
    # =========================================================================

    positive_arguments = {

        "training_points":
            args.training_points,

        "validation_points":
            args.validation_points,

        "plot_points":
            args.plot_points,

        "workers":
            args.workers,

        "dnn_epochs":
            args.dnn_epochs,

        "dnn_hidden_layers":
            args.dnn_hidden_layers,

        "dnn_width":
            args.dnn_width,

        "pdr_degree":
            args.pdr_degree,

        "pdr_iterations":
            args.pdr_iterations,

        "dpi":
            args.dpi,
    }


    for name, value in (
        positive_arguments.items()
    ):

        if value < 1:

            raise ValueError(
                f"{name} must be positive"
            )


    if args.plot_points < 3:

        raise ValueError(
            "plot_points must be >= 3"
        )


    # =========================================================================
    # Load controlled configuration
    # =========================================================================

    config = load_config(

        args.config,

        validate_registered_artifacts=False,
    )


    parameter_names = tuple(
        config.parameter_box.names
    )


    for parameter_name in args.parameters:

        if parameter_name not in parameter_names:

            raise ValueError(
                f"Unknown parameter "
                f"{parameter_name}. "
                f"Available: "
                f"{parameter_names}"
            )


    for output_name in args.outputs:

        if output_name not in OUTPUT_ORDER:

            raise ValueError(
                f"Unknown output "
                f"{output_name}. "
                f"Available: "
                f"{OUTPUT_ORDER}"
            )


    # =========================================================================
    # GPU
    # =========================================================================

    import torch


    if args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but "
                "PyTorch cannot see GPU"
            )

        dnn_use_gpu = True


    elif args.device == "cpu":

        dnn_use_gpu = False


    else:

        dnn_use_gpu = (
            torch.cuda.is_available()
        )


    print()
    print("=" * 75)
    print("CRM1D dedicated 1-D sensitivity experiment")
    print("=" * 75)

    print(
        f"Parameters       : "
        f"{args.parameters}"
    )

    print(
        f"Outputs          : "
        f"{args.outputs}"
    )

    print(
        f"Training points  : "
        f"{args.training_points}"
    )

    print(
        f"Validation points: "
        f"{args.validation_points}"
    )

    print(
        f"Plot points      : "
        f"{args.plot_points}"
    )

    print(
        f"DNN              : "
        f"{args.dnn_hidden_layers} x "
        f"{args.dnn_width} tanh"
    )

    print(
        f"DNN epochs       : "
        f"{args.dnn_epochs}"
    )

    print(
        f"DNN device       : "
        f"{'cuda' if dnn_use_gpu else 'cpu'}"
    )

    print(
        f"CS degree        : "
        f"{args.pdr_degree}"
    )

    print(
        f"Expected 1-D "
        f"CS basis size    : "
        f"{args.pdr_degree + 1}"
    )

    print(
        f"PDR iterations   : "
        f"{args.pdr_iterations}"
    )

    print("=" * 75)
    print()


    # =========================================================================
    # Output locations
    # =========================================================================

    output_path = (
        args.output
        .resolve()
    )


    artifact_root = (

        ROOT

        / "outputs"

        / "sensitivity_1d_v1"

        / (
            f"m{args.training_points}"
            f"_p{args.pdr_degree}"
        )
    )


    artifact_root.mkdir(

        parents=True,

        exist_ok=True,
    )


    # =========================================================================
    # SCCM runner
    # =========================================================================

    runner = runner_from_repository(

        ROOT,

        parameter_box=(
            config.parameter_box
        ),
    )


    # =========================================================================
    # Train each parameter independently
    # =========================================================================

    results = []


    for parameter_name in (
        args.parameters
    ):

        result = train_parameter(

            config=config,

            runner=runner,

            parameter_name=parameter_name,

            output_root=artifact_root,

            training_points=(
                args.training_points
            ),

            validation_points=(
                args.validation_points
            ),

            plot_points=(
                args.plot_points
            ),

            workers=(
                args.workers
            ),

            base_seed=(
                args.base_seed
            ),

            dnn_use_gpu=(
                dnn_use_gpu
            ),

            dnn_epochs=(
                args.dnn_epochs
            ),

            dnn_hidden_layers=(
                args.dnn_hidden_layers
            ),

            dnn_width=(
                args.dnn_width
            ),

            pdr_degree=(
                args.pdr_degree
            ),

            pdr_iterations=(
                args.pdr_iterations
            ),
        )


        results.append(
            result
        )


    # =========================================================================
    # Plot
    # =========================================================================

    output_names = tuple(
        args.outputs
    )


    make_plot(

        results,

        output_names,

        output_path,

        args.dpi,
    )


    # =========================================================================
    # CSV
    # =========================================================================

    csv_path = (

        artifact_root

        / "sensitivity_comparison_1d.csv"
    )


    write_csv(

        results,

        output_names,

        csv_path,
    )


    # =========================================================================
    # Simple experiment metadata
    # =========================================================================

    metadata = {

        "experiment":
            "dedicated-1d-sensitivity",

        "warning":
            (
                "Fresh one-input DNN and "
                "CS/PDR models are trained "
                "for each varied parameter. "
                "These are not the frozen "
                "11-D MCMC surrogates."
            ),

        "parameters":
            list(
                args.parameters
            ),

        "outputs":
            list(
                args.outputs
            ),

        "training_points_per_parameter":
            args.training_points,

        "validation_points_per_parameter":
            args.validation_points,

        "plot_points_per_parameter":
            args.plot_points,

        "other_parameters":
            "held at synthetic truth",

        "dnn": {

            "input_dimension":
                1,

            "output_dimension":
                6,

            "hidden_layers":
                args.dnn_hidden_layers,

            "width":
                args.dnn_width,

            "activation":
                "tanh",

            "epochs":
                args.dnn_epochs,

            "device":
                (
                    "cuda"
                    if dnn_use_gpu
                    else "cpu"
                ),
        },

        "cs": {

            "input_dimension":
                1,

            "output_dimension":
                6,

            "basis":
                "orthonormal Legendre",

            "index_set":
                "hyperbolic-cross",

            "degree":
                args.pdr_degree,

            "expected_basis_size":
                args.pdr_degree + 1,

            "solver":
                "restarted primal-dual",

            "target_iterations":
                args.pdr_iterations,
        },

        "figure_png":
            str(
                output_path
            ),

        "figure_pdf":
            str(
                output_path.with_suffix(
                    ".pdf"
                )
            ),

        "csv":
            str(
                csv_path
            ),
    }


    metadata_path = (

        artifact_root

        / "experiment_metadata.json"
    )


    metadata_path.write_text(

        json.dumps(

            metadata,

            indent=2,

            sort_keys=True,
        )
        + "\n",

        encoding="utf-8",
    )


    # =========================================================================
    # Final status
    # =========================================================================

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        f"PNG: "
        f"{output_path}"
    )

    print(
        f"PDF: "
        f"{output_path.with_suffix('.pdf')}"
    )

    print(
        f"CSV: "
        f"{csv_path}"
    )

    print(
        f"Metadata: "
        f"{metadata_path}"
    )

    print(
        f"Artifacts: "
        f"{artifact_root}"
    )

    print("=" * 75)
    print()


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
