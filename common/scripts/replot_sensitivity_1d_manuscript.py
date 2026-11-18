#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# Canonical production ordering
# ================================================================

OUTPUT_ORDER = (
    "PCP",
    "ACC",
    "LWP",
    "IWP",
    "OLR",
    "OSR",
)

PARAMETERS = (
    "as",
    "ag",
    "rhog",
)

PARAMETER_LABELS = {
    "as": r"$a_s$",
    "ag": r"$a_g$",
    "rhog": r"$\rho_g$",
}


# ================================================================
# EXACT manuscript typography already used by the production
# sensitivity finalizer.
# ================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
    }
)


# ================================================================
# EXACT model styling from production sensitivity finalizer
# ================================================================

COLORS = {
    "SCCM": "black",
    "DNN": "#d62728",
    "CS": "#1f77b4",
}

LINESTYLES = {
    "SCCM": "-",
    "DNN": "--",
    "CS": ":",
}

LINEWIDTH = 1.25


def load_parameter(parameter, data_root):

    path = (
        data_root
        / parameter
        / "sensitivity_1d_data.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing sensitivity data: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        required = (
            "grid_x",
            "sccm",
            "dnn",
            "cs",
            "truth",
        )

        missing = [
            name
            for name in required
            if name not in archive
        ]

        if missing:
            raise RuntimeError(
                f"{path} missing arrays: {missing}"
            )

        result = {
            "grid_x": np.asarray(
                archive["grid_x"],
                dtype=np.float64,
            ),
            "SCCM": np.asarray(
                archive["sccm"],
                dtype=np.float64,
            ),
            "DNN": np.asarray(
                archive["dnn"],
                dtype=np.float64,
            ),
            "CS": np.asarray(
                archive["cs"],
                dtype=np.float64,
            ),
            "truth": float(
                np.asarray(
                    archive["truth"]
                ).reshape(-1)[0]
            ),
        }

    expected_shape = (
        result["grid_x"].shape[0],
        6,
    )

    for method in (
        "SCCM",
        "DNN",
        "CS",
    ):
        if result[method].shape != expected_shape:
            raise RuntimeError(
                f"{parameter} {method}: "
                f"{result[method].shape} != "
                f"{expected_shape}"
            )

        if not np.all(
            np.isfinite(result[method])
        ):
            raise RuntimeError(
                f"{parameter} {method} "
                "contains non-finite values"
            )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    png_path = options.output_dir / "sensitivity_comparison_1D_CRM_CS_DNN_manuscript.png"
    pdf_path = options.output_dir / "sensitivity_comparison_1D_CRM_CS_DNN_manuscript.pdf"


    results = {
        parameter:
            load_parameter(parameter, options.data_dir)
        for parameter in PARAMETERS
    }

    # ------------------------------------------------------------
    # Historical requested orientation:
    #
    # rows    = six CRM1D outputs
    # columns = as, ag, rhog
    #
    # Typography/style remains the maintained manuscript style.
    # ------------------------------------------------------------

    figure, axes = plt.subplots(
        6,
        3,
        figsize=(10.5, 12.2),
        squeeze=False,
    )

    panel_index = 0

    for row, output_name in enumerate(
        OUTPUT_ORDER
    ):

        output_index = row

        for column, parameter in enumerate(
            PARAMETERS
        ):

            axis = axes[row, column]

            result = results[parameter]

            x = result["grid_x"]

            # ----------------------------------------------------
            # Keep manuscript method ordering:
            # SCCM, DNN, CS
            # ----------------------------------------------------

            for method in (
                "SCCM",
                "DNN",
                "CS",
            ):

                axis.plot(
                    x,
                    result[method][:, output_index],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=LINEWIDTH,
                    label=method,
                )

            # Synthetic-truth parameter value
            axis.axvline(
                result["truth"],
                color="black",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
            )

            # ----------------------------------------------------
            # Panel title convention used throughout manuscript:
            # (a), (b), ...
            # ----------------------------------------------------

            panel_letter = chr(
                ord("a") + panel_index
            )

            axis.set_title(
                f"({panel_letter}) {output_name}"
            )

            panel_index += 1

            # Parameter label
            axis.set_xlabel(
                PARAMETER_LABELS[parameter]
            )

            # Output label on first column only.
            # Keeps the 6x3 grid readable without repeating labels.
            if column == 0:
                axis.set_ylabel(
                    output_name
                )

            axis.grid(
                alpha=0.22
            )

            axis.margins(
                x=0.0
            )

    # ------------------------------------------------------------
    # Same no-frame legend convention used by production
    # sensitivity finalizer.
    # ------------------------------------------------------------

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.002),
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.985),
        w_pad=1.5,
        h_pad=1.2,
    )

    options.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print()
    print("=" * 72)
    print("MANUSCRIPT 1-D SENSITIVITY FIGURE COMPLETE")
    print("=" * 72)
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print()
    print("Typography:")
    print("  DejaVu Serif")
    print("  mathtext.fontset = dejavuserif")
    print("  font.size = 8.5")
    print()
    print("Model styling:")
    print("  SCCM = black solid")
    print("  DNN  = red dashed")
    print("  CS   = blue dotted")
    print()
    print("Rows:")
    print("  PCP ACC LWP IWP OLR OSR")
    print()
    print("Columns:")
    print("  a_s a_g rho_g")
    print("=" * 72)


if __name__ == "__main__":
    main()
