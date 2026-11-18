#!/usr/bin/env python3
"""Plot registered common-prefix convergence diagnostics for production MCMC."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("sccm", "dnn", "pdr")
LABELS = {"sccm": "SCCM", "dnn": "DNN", "pdr": "CS"}
COLORS = {"sccm": "black", "dnn": "red", "pdr": "blue"}
MARKERS = {"sccm": "o", "dnn": "s", "pdr": "^"}
PREFIX_PATTERN = re.compile(r"prefix_(\d+)\.npz$")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def _load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for path in sorted((root / method).glob("prefix_*.npz")):
            match = PREFIX_PATTERN.search(path.name)
            if match is None or not path.with_suffix(".manifest.json").is_file():
                continue
            with np.load(path, allow_pickle=False) as arrays:
                names = np.asarray(arrays["parameter_names"]).astype(str)
                rhat = np.asarray(arrays["rhat_final"], dtype=np.float64)
                bulk = np.asarray(arrays["bulk_ess"], dtype=np.float64)
                tail = np.asarray(arrays["tail_ess"], dtype=np.float64)
                occupancy = np.asarray(
                    arrays["maximum_quantile_bin_occupancy_difference"],
                    dtype=np.float64,
                )
            passes = (rhat < 1.01) & (bulk >= 400.0) & (tail >= 400.0)
            rows.append(
                {
                    "method": method,
                    "prefix": int(match.group(1)),
                    "max_rhat": float(np.max(rhat)),
                    "max_rhat_parameter": names[int(np.argmax(rhat))],
                    "min_bulk_ess": float(np.min(bulk)),
                    "min_bulk_ess_parameter": names[int(np.argmin(bulk))],
                    "min_tail_ess": float(np.min(tail)),
                    "min_tail_ess_parameter": names[int(np.argmin(tail))],
                    "max_occupancy_difference": float(np.max(occupancy)),
                    "parameters_passing_all": int(np.count_nonzero(passes)),
                    "parameter_count": int(names.size),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No registered prefix diagnostics found beneath {root}")
    return rows


def _write_summary(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    panels = (
        (axes[0, 0], "max_rhat", r"Worst-case $\hat R$", 1.01, "< 1.01"),
        (axes[0, 1], "min_bulk_ess", "Minimum bulk ESS", 400.0, r"$\geq 400$"),
        (axes[1, 0], "min_tail_ess", "Minimum tail ESS", 400.0, r"$\geq 400$"),
        (
            axes[1, 1],
            "parameters_passing_all",
            "Parameters passing all criteria",
            11.0,
            "11 of 11",
        ),
    )
    for method in METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["prefix"]),
        )
        x = np.asarray([int(row["prefix"]) / 1000 for row in selected])
        for axis, key, _, _, _ in panels:
            y = np.asarray([float(row[key]) for row in selected])
            axis.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                markersize=6,
                label=LABELS[method],
            )

    for axis, _, title, threshold, threshold_label in panels:
        axis.axhline(threshold, color="0.45", linestyle="--", linewidth=1.1)
        axis.annotate(
            threshold_label,
            xy=(0.99, threshold),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="0.35",
            fontsize=8.5,
        )
        axis.set_title(title)
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.set_axisbelow(True)

    axes[0, 0].set_ylim(bottom=1.0)
    axes[0, 1].set_ylim(bottom=0.0)
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].set_ylim(-0.25, 11.75)
    axes[1, 1].set_yticks(range(0, 12, 2))
    axes[1, 0].set_xlabel("Retained transitions per walker (thousands)")
    axes[1, 1].set_xlabel("Retained transitions per walker (thousands)")
    axes[0, 0].set_ylabel(r"$\hat R$")
    axes[0, 1].set_ylabel("Effective samples")
    axes[1, 0].set_ylabel("Effective samples")
    axes[1, 1].set_ylabel("Parameter count")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper right")
    figure.suptitle(
        "MCMC convergence progress: two independent 32-walker ensembles",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.012,
        "Worst-case aggregation across the 11 parameters; strict targets shown by dashed lines.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.955))
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = _arguments()
    root = arguments.diagnostics_root.resolve()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(root)
    _write_summary(rows, output.with_suffix(".csv"))
    _plot(rows, output)
    print(output)
    print(output.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
