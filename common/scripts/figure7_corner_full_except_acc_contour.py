#!/usr/bin/env python3
"""
Alternative Figure 7 visualization.

IMPORTANT
---------
This script DOES NOT modify the maintained Figure 7 implementation.

It uses exactly the same:
- production full_except_acc chains
- SCCM / DNN / CS methods
- two independent ensembles per method
- transition subsampling policy
- Figure 7 random seeds
- as, bs, ag, bg parameter subset
- 1-D diagonal histograms
- truth markers
- colors
- manuscript typography
- 60 x 60 two-dimensional histogram resolution
- 50%, 80%, 95% posterior-mass contour concept

The only visualization experiment is:

    smooth the 60 x 60 two-dimensional density field slightly
    before drawing the contour lines.

This is intended to test whether cleaner contour visualization is
preferable to the current jagged / point-like contour appearance.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tomllib

import h5netcdf
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


# =====================================================================
# Repository
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.provenance import (
    sha256_file,
    validate_manifest,
)


# =====================================================================
# Manuscript style
# =====================================================================

LABELS = {
    "sccm": "SCCM",
    "dnn": "DNN",
    "pdr": "CS",
}

COLORS = {
    "sccm": "black",
    "dnn": "#d62728",
    "pdr": "#1f77b4",
}

PARAMETER_LATEX = {
    "as": r"$a_s$",
    "bs": r"$b_s$",
    "ag": r"$a_g$",
    "bg": r"$b_g$",
    "N0r": r"$N_{0r}$",
    "N0s": r"$N_{0s}$",
    "N0g": r"$N_{0g}$",
    "rhos": r"$\rho_s$",
    "rhog": r"$\rho_g$",
    "qc0": r"$q_{c,0}$",
    "qi0": r"$q_{i,0}$",
}


def manuscript_style() -> None:
    """
    Exact maintained manuscript typography.
    """

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
        }
    )


# =====================================================================
# Load actual manuscript + MCMC configurations
# =====================================================================

MANUSCRIPT_CONFIG = ROOT / "configs" / "manuscript.toml"


def load_settings():
    if not MANUSCRIPT_CONFIG.is_file():
        raise FileNotFoundError(
            f"Missing manuscript config: {MANUSCRIPT_CONFIG}"
        )

    with MANUSCRIPT_CONFIG.open("rb") as stream:
        settings = tomllib.load(stream)

    if settings.get("schema_version") != 1:
        raise ValueError(
            "Manuscript config must use schema_version = 1"
        )

    mcmc_path = (
        ROOT / settings["mcmc_config"]
    ).resolve()

    config = load_config(
        mcmc_path,
        validate_registered_artifacts=True,
    )

    return settings, config


# =====================================================================
# Production chain validation
#
# Same validation contract used by finalize_manuscript.py
# =====================================================================

def chain_paths(
    config,
    root: Path,
    case: str,
    method: str,
) -> tuple[Path, Path]:

    paths = []

    for ensemble in range(2):

        directory = (
            root
            / case
            / method
            / f"ensemble_{ensemble:03d}"
        )

        chain = directory / "chain.nc"

        manifest_path = (
            directory
            / "chain.manifest.json"
        )

        if (
            not chain.is_file()
            or not manifest_path.is_file()
        ):
            raise FileNotFoundError(
                f"Incomplete finalized chain: "
                f"{directory}"
            )

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        validate_manifest(
            manifest
        )

        actual_chain_sha = sha256_file(
            chain
        )

        if (
            manifest["chain_sha256"]
            != actual_chain_sha
        ):
            raise ValueError(
                f"Chain checksum mismatch: "
                f"{chain}"
            )

        expected = {
            "config_sha256":
                sha256_file(config.path),

            "method":
                method,

            "likelihood_case":
                case,

            "ensemble_id":
                ensemble,

            "parameter_box_sha256":
                config.parameter_box.fingerprint,
        }

        for key, value in expected.items():

            if manifest.get(key) != value:

                raise ValueError(
                    "Chain manifest binding "
                    f"mismatch for {key}: "
                    f"{manifest_path}"
                )

        paths.append(
            chain
        )

    return tuple(paths)


# =====================================================================
# Exact Figure 7 chain subsampling
#
# This duplicates the maintained _sample_chains() behavior:
#
# method seed:
#   SCCM = 74001
#   DNN  = 74101
#   CS   = 74201
#
# ensemble offset:
#   ensemble 0 -> +0
#   ensemble 1 -> +1009
#
# 512 transitions selected / ensemble.
# All 32 walkers retained for each selected transition.
# =====================================================================

def sample_chains(
    paths: tuple[Path, Path],
    transitions: int,
    seed: int,
) -> np.ndarray:

    arrays = []

    for ensemble, path in enumerate(
        paths
    ):

        with h5netcdf.File(
            path,
            "r",
        ) as dataset:

            variable = (
                dataset.variables["position"]
            )

            available = (
                variable.shape[0]
            )

            if transitions > available:

                raise ValueError(
                    f"Requested {transitions} "
                    f"transitions from "
                    f"{available}: {path}"
                )

            # EXACT same ensemble seed policy
            rng = np.random.default_rng(
                seed
                + 1009 * ensemble
            )

            indices = np.sort(
                rng.choice(
                    available,
                    transitions,
                    replace=False,
                )
            )

            rows = np.empty(
                (
                    transitions,
                    variable.shape[1],
                    variable.shape[2],
                ),
                dtype=np.float64,
            )

            for row, index in enumerate(
                indices
            ):
                rows[row] = (
                    variable[
                        int(index),
                        :,
                        :,
                    ]
                )

            arrays.append(
                rows.reshape(
                    -1,
                    rows.shape[-1],
                )
            )

    values = np.concatenate(
        arrays,
        axis=0,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Posterior sample contains "
            "nonfinite values"
        )

    return values


# =====================================================================
# Cleaner alternative contour renderer
# =====================================================================

def smooth_density_contours(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    color: str,
    smoothing_sigma: float = 1.0,
):
    """
    Draw 50%, 80%, and 95% posterior-mass contours.

    Same histogram resolution as Figure 7, but apply a small Gaussian
    smoothing to the histogram density before contour extraction.

    No scatter points are plotted.
    """

    density, x_edges, y_edges = (
        np.histogram2d(
            x,
            y,
            bins=bins,
            density=True,
        )
    )

    # -------------------------------------------------------------
    # Visualization-only smoothing.
    # The underlying posterior samples are unchanged.
    # -------------------------------------------------------------

    smooth = gaussian_filter(
        density,
        sigma=smoothing_sigma,
        mode="nearest",
    )

    if not np.any(
        smooth > 0
    ):
        return

    # -------------------------------------------------------------
    # Since np.histogram2d with integer bins produces equal-width
    # cells in each direction, sorting density values is equivalent
    # to ranking cell probability mass within this panel.
    # -------------------------------------------------------------

    positive = np.sort(
        smooth[smooth > 0]
    )[::-1]

    if len(positive) < 4:
        return

    cumulative = (
        np.cumsum(positive)
        / np.sum(positive)
    )

    levels = []

    for mass in (
        0.50,
        0.80,
        0.95,
    ):

        location = int(
            np.searchsorted(
                cumulative,
                mass,
            )
        )

        location = min(
            location,
            len(positive) - 1,
        )

        levels.append(
            positive[location]
        )

    # Matplotlib requires ascending
    # contour levels.
    levels = np.unique(
        levels
    )

    if len(levels) == 0:
        return

    x_centers = 0.5 * (
        x_edges[:-1]
        + x_edges[1:]
    )

    y_centers = 0.5 * (
        y_edges[:-1]
        + y_edges[1:]
    )

    axis.contour(
        x_centers,
        y_centers,
        smooth.T,
        levels=levels,
        colors=[color],
        linewidths=1.15,
        alpha=0.95,
    )


# =====================================================================
# Main
# =====================================================================

def main(settings_config=None):

    settings, config = (
        load_settings() if settings_config is None else settings_config
    )

    # -------------------------------------------------------------
    # Use EXACT Figure 7 manuscript policy.
    # -------------------------------------------------------------

    policy = settings["corner"]

    case = settings[
        "primary_likelihood_case"
    ]

    if case != "full_except_acc":
        raise RuntimeError(
            "Expected primary manuscript "
            "case full_except_acc, "
            f"found {case}"
        )

    chain_root = (
        ROOT
        / settings["chain_root"]
    ).resolve()

    parameter_names = list(
        config.parameter_box.names
    )

    requested_parameters = list(
        policy["parameter_names"]
    )

    selected = [
        parameter_names.index(name)
        for name
        in requested_parameters
    ]

    if requested_parameters != [
        "as",
        "bs",
        "ag",
        "bg",
    ]:
        raise RuntimeError(
            "Unexpected Figure 7 "
            "parameter selection: "
            f"{requested_parameters}"
        )

    transitions = int(
        policy[
            "transitions_per_ensemble"
        ]
    )

    seed = int(
        policy["seed"]
    )

    bins_1d = int(
        policy["bins_1d"]
    )

    bins_2d = int(
        policy["bins_2d"]
    )


    # -------------------------------------------------------------
    # Validate production chains.
    # -------------------------------------------------------------

    paths = {
        method:
            chain_paths(
                config,
                chain_root,
                case,
                method,
            )

        for method
        in (
            "sccm",
            "dnn",
            "pdr",
        )
    }


    # -------------------------------------------------------------
    # EXACT maintained Figure 7 sampling policy.
    #
    # SCCM -> seed + 0
    # DNN  -> seed + 100
    # PDR  -> seed + 200
    # -------------------------------------------------------------

    samples = {

        method:
            sample_chains(
                method_paths,
                transitions,
                seed
                + 100 * method_index,
            )[:, selected]

        for method_index,
            (
                method,
                method_paths,
            )

        in enumerate(
            paths.items()
        )
    }


    # -------------------------------------------------------------
    # QA
    # -------------------------------------------------------------

    expected_rows = (
        transitions
        * 32
        * 2
    )

    for method in (
        "sccm",
        "dnn",
        "pdr",
    ):

        expected_shape = (
            expected_rows,
            4,
        )

        if (
            samples[method].shape
            != expected_shape
        ):
            raise RuntimeError(
                f"{method} sample shape "
                f"{samples[method].shape}; "
                f"expected "
                f"{expected_shape}"
            )


    labels = [
        PARAMETER_LATEX[
            parameter_names[index]
        ]
        for index in selected
    ]

    truth = (
        config.truth[selected]
    )


    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    manuscript_style()

    figure, axes = plt.subplots(
        4,
        4,
        figsize=(9.2, 9.2),
    )


    for row in range(4):

        for column in range(4):

            axis = axes[
                row,
                column,
            ]


            # =====================================================
            # Upper triangle blank
            # =====================================================

            if column > row:

                axis.axis(
                    "off"
                )

                continue


            # =====================================================
            # Diagonal:
            # keep EXACT Figure 7 1-D histogram presentation
            # =====================================================

            if row == column:

                for method in (
                    "sccm",
                    "dnn",
                    "pdr",
                ):

                    axis.hist(
                        samples[
                            method
                        ][
                            :,
                            column,
                        ],

                        bins=bins_1d,

                        density=True,

                        histtype="step",

                        linewidth=1.3,

                        color=COLORS[
                            method
                        ],

                        label=LABELS[
                            method
                        ],
                    )


                axis.axvline(
                    truth[column],
                    color="0.35",
                    linestyle="--",
                    linewidth=1.0,
                )


            # =====================================================
            # Lower triangle:
            # CLEAN CONTOUR-ONLY alternative
            # =====================================================

            else:

                for method in (
                    "sccm",
                    "dnn",
                    "pdr",
                ):

                    smooth_density_contours(
                        axis,

                        samples[
                            method
                        ][
                            :,
                            column,
                        ],

                        samples[
                            method
                        ][
                            :,
                            row,
                        ],

                        bins=bins_2d,

                        color=COLORS[
                            method
                        ],

                        smoothing_sigma=1.0,
                    )


                # synthetic truth
                axis.plot(
                    truth[column],
                    truth[row],
                    marker="+",
                    color="0.2",
                    markersize=7,
                )


            # =====================================================
            # EXACT Figure 7 labeling convention
            # =====================================================

            if row == 3:

                axis.set_xlabel(
                    labels[column]
                )

            else:

                axis.set_xticklabels(
                    []
                )


            if (
                column == 0
                and row > 0
            ):

                axis.set_ylabel(
                    labels[row]
                )

            elif column > 0:

                axis.set_yticklabels(
                    []
                )


            axis.grid(
                alpha=0.12
            )


    axes[0, 0].legend(
        frameon=False,
        fontsize=8,
    )


    figure.suptitle(
        "Posterior marginals: "
        "full except acc",
        y=0.995,
    )


    figure.tight_layout()


    # -------------------------------------------------------------
    # NEW outputs only.
    # Original Figure 7 is untouched.
    # -------------------------------------------------------------

    figure_dir = (
        ROOT
        / settings[
            "figure_directory"
        ]
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    pdf = (
        figure_dir
        / (
            "figure7_corner_"
            "full_except_acc_"
            "smooth_contour.pdf"
        )
    )


    png = (
        figure_dir
        / (
            "figure7_corner_"
            "full_except_acc_"
            "smooth_contour.png"
        )
    )


    figure.savefig(
        pdf,
        bbox_inches="tight",
    )


    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "FIGURE 7 CLEAN-CONTOUR "
        "VISUALIZATION COMPLETE"
    )
    print("=" * 76)

    print(
        f"Likelihood case          : "
        f"{case}"
    )

    print(
        f"Parameters               : "
        f"{requested_parameters}"
    )

    print(
        f"Transitions / ensemble   : "
        f"{transitions}"
    )

    print(
        "Walkers / transition    : 32"
    )

    print(
        "Independent ensembles   : 2"
    )

    print(
        f"States / method          : "
        f"{expected_rows}"
    )

    print(
        f"Base Figure-7 seed       : "
        f"{seed}"
    )

    print(
        "Method seeds            : "
        "SCCM=74001, "
        "DNN=74101, "
        "CS=74201"
    )

    print(
        "Ensemble seed offset    : "
        "+1009"
    )

    print(
        f"1-D bins                 : "
        f"{bins_1d}"
    )

    print(
        f"2-D bins                 : "
        f"{bins_2d}"
    )

    print(
        "Contour masses          : "
        "50%, 80%, 95%"
    )

    print(
        "Contour smoothing       : "
        "Gaussian sigma=1.0 bin"
    )

    print(
        "Scatter points          : NONE"
    )

    print()
    print(
        f"PDF: {pdf}"
    )

    print(
        f"PNG: {png}"
    )

    print()
    print(
        "Original "
        "figure7_corner_full_except_acc.pdf "
        "was NOT modified."
    )

    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
