#!/usr/bin/env python3
"""
Degree-resolved CS coefficient interpretability figure for PCP and OLR.

This script is visualization-only:
- no CRM rerun
- no CS/PDR refit
- no new polynomial basis
- loads the checksum-registered frozen production PDR model
- uses the production q=23, N=7569 hyperbolic-cross basis
- focuses on a_s, b_s, a_g, b_g
- plots coefficient-energy fractions c_nu^2 / V
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
from matplotlib import colormaps
import numpy as np


# ---------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_column_reproduction.mcmc.config import OUTPUT_ORDER, load_config
from cloud_column_reproduction.provenance import sha256_file


# ---------------------------------------------------------------------
# Same manuscript typography used by finalize_manuscript.py
# ---------------------------------------------------------------------
def manuscript_style():
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


SUBSET_NAMES = ("as", "bs", "ag", "bg")

PARAMETER_LATEX = {
    "as": r"$a_s$",
    "bs": r"$b_s$",
    "ag": r"$a_g$",
    "bg": r"$b_g$",
}

TARGET_OUTPUTS = ("PCP", "OLR")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mcmc.toml",
    )

    parser.add_argument("--output-dir", type=Path)

    parser.add_argument(
        "--max-degree",
        type=int,
        default=10,
        help="Largest polynomial degree displayed.",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=1.0e-8,
        help="Shared LogNorm display floor.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Load the exact frozen production PDR artifact
# ---------------------------------------------------------------------
def load_frozen_pdr(config_path):
    config = load_config(
        config_path,
        validate_registered_artifacts=True,
    )

    config.require_production_ready()

    section = config.raw["surrogates"]["pdr"]

    model = config.path.parent.parent / section["model"]

    actual_sha = sha256_file(model)
    expected_sha = section["model_sha256"]

    if actual_sha != expected_sha:
        raise ValueError(
            "Frozen PDR model checksum mismatch:\n"
            f"expected: {expected_sha}\n"
            f"actual:   {actual_sha}"
        )

    with np.load(model, allow_pickle=False) as archive:
        coefficients = np.asarray(
            archive["coefficients"],
            dtype=np.float64,
        )
        indices = np.asarray(
            archive["indices"],
            dtype=np.int64,
        )

    if coefficients.shape != (7569, 6):
        raise ValueError(
            f"Expected coefficient shape (7569, 6), got {coefficients.shape}"
        )

    if indices.shape != (7569, 11):
        raise ValueError(
            f"Expected index shape (7569, 11), got {indices.shape}"
        )

    return config, model, coefficients, indices, actual_sha


# ---------------------------------------------------------------------
# Production Figure-2 coefficient-energy definition
# ---------------------------------------------------------------------
def coefficient_energy_fraction(
    coefficients,
    indices,
    output_index,
):
    """
    E_nu = c_nu^2 / V

    where

        V = sum_{nu != 0} c_nu^2

    exactly matching the nonconstant-energy normalization
    used by the production Figure 2.
    """

    square = coefficients[:, output_index] ** 2

    nonconstant = np.any(indices > 0, axis=1)

    variance = float(np.sum(square[nonconstant]))

    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Invalid nonconstant coefficient energy.")

    return square / variance, variance


def make_lookup(indices):
    return {
        tuple(row.tolist()): n
        for n, row in enumerate(indices)
    }


# ---------------------------------------------------------------------
# Pure 1-D coefficient-energy strip
# ---------------------------------------------------------------------
def solo_strip(
    energy,
    lookup,
    dim,
    dimension,
    max_degree,
):
    values = np.full(max_degree, np.nan)

    for degree in range(1, max_degree + 1):
        key = [0] * dimension
        key[dim] = degree

        n = lookup.get(tuple(key))

        if n is not None:
            values[degree - 1] = energy[n]

    return values


# ---------------------------------------------------------------------
# Pure pairwise coefficient-energy tile
# ---------------------------------------------------------------------
def pair_tile(
    energy,
    lookup,
    dim_i,
    dim_j,
    dimension,
    max_degree,
):
    """
    Only pure two-variable terms are included:

        nu_i > 0
        nu_j > 0

    and all other dimensions equal zero.
    """

    values = np.full(
        (max_degree, max_degree),
        np.nan,
    )

    for degree_i in range(1, max_degree + 1):
        for degree_j in range(1, max_degree + 1):

            key = [0] * dimension
            key[dim_i] = degree_i
            key[dim_j] = degree_j

            n = lookup.get(tuple(key))

            if n is not None:
                values[degree_i - 1, degree_j - 1] = energy[n]

    return values


def display_floor(values, vmin):
    """
    Keep NaN = structurally absent basis terms.

    Basis-present coefficients below vmin are shown at the
    colormap floor.
    """

    result = values.copy()

    finite = np.isfinite(result)

    result[finite] = np.maximum(
        result[finite],
        vmin,
    )

    return result


# ---------------------------------------------------------------------
# Assemble one output
# ---------------------------------------------------------------------
def build_payload(
    output_name,
    coefficients,
    indices,
    lookup,
    subset_dims,
    max_degree,
):
    output_index = OUTPUT_ORDER.index(output_name)

    energy, variance = coefficient_energy_fraction(
        coefficients,
        indices,
        output_index,
    )

    dimension = indices.shape[1]

    solo = {}
    pair = {}
    rows = []

    # Diagonal terms
    for name, dim in zip(
        SUBSET_NAMES,
        subset_dims,
    ):
        values = solo_strip(
            energy,
            lookup,
            dim,
            dimension,
            max_degree,
        )

        solo[name] = values

        for degree, value in enumerate(
            values,
            start=1,
        ):
            rows.append(
                {
                    "output": output_name,
                    "kind": "solo",
                    "parameter_i": name,
                    "parameter_j": "",
                    "degree_i": degree,
                    "degree_j": "",
                    "basis_present": np.isfinite(value),
                    "coefficient_energy_fraction": (
                        float(value)
                        if np.isfinite(value)
                        else ""
                    ),
                }
            )

    # Pure pair terms
    for i in range(len(SUBSET_NAMES)):
        for j in range(i + 1, len(SUBSET_NAMES)):

            name_i = SUBSET_NAMES[i]
            name_j = SUBSET_NAMES[j]

            values = pair_tile(
                energy,
                lookup,
                subset_dims[i],
                subset_dims[j],
                dimension,
                max_degree,
            )

            pair[(name_i, name_j)] = values

            for di in range(1, max_degree + 1):
                for dj in range(1, max_degree + 1):

                    value = values[
                        di - 1,
                        dj - 1,
                    ]

                    rows.append(
                        {
                            "output": output_name,
                            "kind": "pure_pair",
                            "parameter_i": name_i,
                            "parameter_j": name_j,
                            "degree_i": di,
                            "degree_j": dj,
                            "basis_present": np.isfinite(value),
                            "coefficient_energy_fraction": (
                                float(value)
                                if np.isfinite(value)
                                else ""
                            ),
                        }
                    )

    return {
        "output": output_name,
        "variance": variance,
        "solo": solo,
        "pair": pair,
        "rows": rows,
    }


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------
def degree_ticks(max_degree):
    candidates = [1, 2, 4, 6, 8, 10]

    ticks = [
        x
        for x in candidates
        if x <= max_degree
    ]

    if max_degree not in ticks:
        ticks.append(max_degree)

    return sorted(set(ticks))


def draw_strip(
    ax,
    values,
    cmap,
    norm,
    vmin,
):
    n = len(values)

    shown = display_floor(
        values,
        vmin,
    )

    xedges = np.arange(
        0.5,
        n + 1.5,
        1.0,
    )

    yedges = np.array(
        [0.0, 1.0]
    )

    ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_invalid(shown)[None, :],
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidth=0.0,
    )

    ax.set_xlim(
        0.5,
        n + 0.5,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_yticks([])

    ticks = degree_ticks(n)

    ax.set_xticks(ticks)

    ax.tick_params(
        axis="x",
        labelsize=7,
        length=2,
    )


def draw_tile(
    ax,
    values,
    cmap,
    norm,
    vmin,
):
    n = values.shape[0]

    shown = display_floor(
        values,
        vmin,
    )

    edges = np.arange(
        0.5,
        n + 1.5,
        1.0,
    )

    ax.pcolormesh(
        edges,
        edges,
        np.ma.masked_invalid(shown),
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidth=0.0,
    )

    ax.set_xlim(
        0.5,
        n + 0.5,
    )

    ax.set_ylim(
        0.5,
        n + 0.5,
    )

    ax.set_aspect("equal")

    ticks = degree_ticks(n)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.tick_params(
        axis="both",
        labelsize=7,
        length=2,
    )


# ---------------------------------------------------------------------
# Draw a 4 x 4 upper-triangular corner
# ---------------------------------------------------------------------
def draw_corner(
    figure,
    grid,
    payload,
    cmap,
    norm,
    vmin,
    panel_label,
):
    axes = np.empty(
        (4, 4),
        dtype=object,
    )

    for row in range(4):
        for col in range(4):

            ax = figure.add_subplot(
                grid[row, col]
            )

            axes[row, col] = ax

            # Lower triangle unused
            if col < row:
                ax.axis("off")
                continue

            row_name = SUBSET_NAMES[row]
            col_name = SUBSET_NAMES[col]

            # Diagonal: pure one-parameter degree structure
            if row == col:

                draw_strip(
                    ax,
                    payload["solo"][row_name],
                    cmap,
                    norm,
                    vmin,
                )

                for spine in ax.spines.values():
                    spine.set_color("0.25")
                    spine.set_linewidth(1.0)

            # Upper triangle: pure pair structure
            else:

                draw_tile(
                    ax,
                    payload["pair"][
                        (row_name, col_name)
                    ],
                    cmap,
                    norm,
                    vmin,
                )

            # Parameter labels across top
            if row == 0:
                ax.set_title(
                    PARAMETER_LATEX[col_name],
                    fontsize=9.5,
                    pad=4,
                )

            # Parameter labels down diagonal
            if row == col:
                ax.set_ylabel(
                    PARAMETER_LATEX[row_name],
                    fontsize=9.5,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=6,
                )

            # Remove all degree tick labels and tick marks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

    axes[0, 0].text(
        0.0,
        1.45,
        f"({panel_label}) {payload['output']} coefficient-energy structure",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    if args.max_degree < 1:
        raise ValueError(
            "--max-degree must be >= 1"
        )

    if args.vmin <= 0:
        raise ValueError(
            "--vmin must be > 0"
        )

    manuscript_style()

    (
        config,
        model,
        coefficients,
        indices,
        model_sha,
    ) = load_frozen_pdr(
        args.config
    )

    parameter_names = tuple(
        config.parameter_box.names
    )

    expected_parameter_order = (
        "as",
        "bs",
        "ag",
        "bg",
        "N0r",
        "N0s",
        "N0g",
        "rhos",
        "rhog",
        "qc0",
        "qi0",
    )

    if parameter_names != expected_parameter_order:
        raise ValueError(
            "Unexpected production parameter order:\n"
            f"{parameter_names}"
        )

    subset_dims = tuple(
        parameter_names.index(name)
        for name in SUBSET_NAMES
    )

    lookup = make_lookup(indices)

    payloads = {}

    for output in TARGET_OUTPUTS:

        payloads[output] = build_payload(
            output,
            coefficients,
            indices,
            lookup,
            subset_dims,
            args.max_degree,
        )

    # --------------------------------------------------------------
    # Shared color normalization across PCP and OLR
    # --------------------------------------------------------------
    visible_positive = []

    for payload in payloads.values():

        for values in payload["solo"].values():

            finite = values[
                np.isfinite(values)
                & (values > 0)
            ]

            visible_positive.extend(
                finite.tolist()
            )

        for values in payload["pair"].values():

            finite = values[
                np.isfinite(values)
                & (values > 0)
            ]

            visible_positive.extend(
                finite.tolist()
            )

    if not visible_positive:
        raise ValueError(
            "No positive coefficient-energy values found."
        )

    vmax = float(
        np.max(visible_positive)
    )

    if vmax <= args.vmin:
        raise ValueError(
            f"vmax={vmax} <= vmin={args.vmin}"
        )

    cmap = colormaps["inferno"].copy()

    # Structurally absent HC terms
    cmap.set_bad("0.85")

    norm = LogNorm(
        vmin=args.vmin,
        vmax=vmax,
    )

    # --------------------------------------------------------------
    # Combined PCP + OLR manuscript figure
    # --------------------------------------------------------------
    figure = plt.figure(
        figsize=(10.2, 6.3)
    )

    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=[
            1.0,
            1.0,
            0.045,
        ],
        wspace=0.22,
    )

    grid_pcp = outer[
        0, 0
    ].subgridspec(
        4,
        4,
        wspace=0.06,
        hspace=0.10,
    )

    grid_olr = outer[
        0, 1
    ].subgridspec(
        4,
        4,
        wspace=0.06,
        hspace=0.10,
    )

    colorbar_axis = figure.add_subplot(
        outer[0, 2]
    )

    draw_corner(
        figure,
        grid_pcp,
        payloads["PCP"],
        cmap,
        norm,
        args.vmin,
        "a",
    )

    draw_corner(
        figure,
        grid_olr,
        payloads["OLR"],
        cmap,
        norm,
        args.vmin,
        "b",
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    colorbar = figure.colorbar(
        sm,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        r"Normalized coefficient energy",
        fontsize=9.5,
    )

    colorbar.ax.tick_params(
        labelsize=8
    )

    # --------------------------------------------------------------
    # Output files
    # --------------------------------------------------------------
    figure_dir = args.output_dir or ROOT / "figures"

    table_dir = args.output_dir or (
        ROOT
        / "outputs"
        / "manuscript"
        / "tables"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        figure_dir
        / "figure2_cs_degree_resolved_4d.pdf"
    )

    png_path = (
        figure_dir
        / "figure2_cs_degree_resolved_4d.png"
    )

    csv_path = (
        table_dir
        / "figure2_cs_degree_resolved_4d.csv"
    )

    # Preserve the published crop for each format after removing the footer.
    figure.savefig(
        pdf_path,
        bbox_inches=Bbox.from_extents(
            0.359802083333, -0.030239583333, 9.826223958333, 6.294891957364,
        ),
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches=Bbox.from_extents(
            0.360375, -0.028833333333, 9.819552188789, 6.293329457364,
        ),
    )

    plt.close(figure)

    # --------------------------------------------------------------
    # Auditable table of displayed values
    # --------------------------------------------------------------
    rows = []

    for output in TARGET_OUTPUTS:

        for row in payloads[
            output
        ]["rows"]:

            rows.append(
                {
                    **row,
                    "display_floor": args.vmin,
                    "max_degree_shown": args.max_degree,
                    "nonconstant_output_energy_V": (
                        payloads[
                            output
                        ]["variance"]
                    ),
                    "frozen_model": str(
                        model.relative_to(config.path.parent.parent)
                    ),
                    "frozen_model_sha256": model_sha,
                    "basis_q": int(
                        config.raw[
                            "surrogates"
                        ]["pdr"]["q"]
                    ),
                    "basis_size": int(
                        config.raw[
                            "surrogates"
                        ]["pdr"]["basis_size"]
                    ),
                }
            )

    with csv_path.open(
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
        writer.writerows(rows)

    print()
    print(
        "Production Figure 2 interpretability visualization complete."
    )

    print(
        f"Frozen model: {model}"
    )

    print(
        f"SHA-256: {model_sha}"
    )

    print(
        "Basis:",
        f"q={config.raw['surrogates']['pdr']['q']},",
        f"N={config.raw['surrogates']['pdr']['basis_size']}",
    )

    print(
        "Parameters:",
        ", ".join(SUBSET_NAMES),
    )

    print(
        "Outputs:",
        ", ".join(TARGET_OUTPUTS),
    )

    print(
        f"Shared LogNorm: "
        f"{args.vmin:.3e} to {vmax:.3e}"
    )

    print()
    print(
        f"PDF: {pdf_path}"
    )

    print(
        f"PNG: {png_path}"
    )

    print(
        f"CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()
