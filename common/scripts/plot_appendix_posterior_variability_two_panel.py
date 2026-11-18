#!/usr/bin/env python3
"""Plot v4 Appendix A1 from the accepted compact table on Derecho."""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEM = "appendixB_relative_l2_vs_energy_distance_two_panel_manuscript_style"
SOURCE_HASHES = {
    ".csv": "cf0154743ab3ede386d8365979bafe4165158bd367c9c166508b4698cae845a9",
    ".manifest.json": "4bb33326b9e1f484b5971444a76c172a511ef7b9e63c48d30317cbe8f29e6078",
    ".pdf": "fab5ff5b54e06471b0c5b58d3e85628891b3e7f9a79eb08b698c125a2db738d6",
}
EXTENDED_CS = [2, 9, 14, 18, 19]
EXPECTED_MINIMA = {"DNN": (0, 10), "CS": (1, 0)}
X = "forward_geometric_mean_relative_l2"
Y = "posterior_energy_distance"
SE = "posterior_energy_distance_standard_error"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_points(content):
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if len(rows) != 40 or any(row["method"] not in EXPECTED_MINIMA for row in rows):
        raise ValueError("Expected exactly 40 DNN/CS rows")
    groups = {}
    for method, expected in EXPECTED_MINIMA.items():
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["trial"]),
        )
        trials = [int(row["trial"]) for row in selected]
        if trials != list(range(20)):
            raise ValueError(f"{method}: expected exactly one row per trial 000–019")
        x, y, se, lower, upper = np.array(
            [[float(row[key]) for row in selected]
             for key in (X, Y, SE, "ci95_lower", "ci95_upper")]
        )
        if not np.isfinite([x, y, se, lower, upper]).all() or np.any(se < 0):
            raise ValueError(f"{method}: invalid finite values or standard errors")
        minima = int(np.argmin(x)), int(np.argmin(y))
        if minima != expected or np.count_nonzero(x == x.min()) != 1 or np.count_nonzero(y == y.min()) != 1:
            raise ValueError(f"{method}: unexpected or tied minima {minima}")
        groups[method] = (x, y, se, minima)
    return groups


def make_figure(groups):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True,
                               layout="constrained")
    fits = {}
    for panel, (axis, (method, (x, y, se, minima))) in enumerate(
        zip(axes, groups.items(), strict=True)
    ):
        slope, intercept = np.polynomial.polynomial.polyfit(x, y, 1)[::-1]
        domain = np.array([x.min(), x.max()])
        fit, = axis.plot(domain, slope * domain + intercept, "--", color="tab:red",
                         label="Least-squares fit", zorder=2)
        points = axis.errorbar(x, y, yerr=se, fmt="o", ms=4, color="tab:blue",
                               elinewidth=0.9, capsize=2, label=f"{method} trials", zorder=3)
        highlights = []
        for trial, marker, color, label in (
            (minima[0], "*", "tab:orange", "Minimum forward error"),
            (minima[1], "D", "tab:green", "Minimum energy distance"),
        ):
            highlights.append(axis.scatter(x[trial], y[trial], marker=marker,
                                           s=85 if marker == "*" else 35, color=color,
                                           edgecolor="white", linewidth=0.5,
                                           label=label, zorder=4))
        axis.set_title(f"({chr(97 + panel)}) MCMC-{method}")
        axis.set_xlabel(r"Geometric mean RQMC relative $L^2$ error")
        axis.set_xscale("linear")
        axis.set_yscale("linear")
        axis.grid(alpha=0.23)
        axis.set_axisbelow(True)
        axis.legend(handles=[points, *highlights, fit], loc="upper right" if panel == 0 else "upper left",
                    fontsize=8, frameon=False)
        np.testing.assert_array_equal(points.lines[0].get_xdata(), x)
        np.testing.assert_array_equal(points.lines[0].get_ydata(), y)
        np.testing.assert_array_equal(points.lines[2][0].get_segments(),
                                      np.stack((np.column_stack((x, y-se)),
                                                np.column_stack((x, y+se))), axis=1))
        for highlight, trial in zip(highlights, minima, strict=True):
            np.testing.assert_array_equal(highlight.get_offsets(), [[x[trial], y[trial]]])
        fits[method] = {"slope": float(slope), "intercept": float(intercept)}
    axes[0].set_ylabel("Prior-normalized joint posterior\nenergy distance to SCCM")
    return figure, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-binding", type=Path, required=True)
    parser.add_argument("--reference-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    sources = {".csv": options.csv, ".manifest.json": options.manifest,
               ".pdf": options.reference_pdf}
    content = options.csv.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_HASHES[".csv"]:
        raise ValueError("Accepted Appendix A1 CSV hash mismatch")
    binding = json.loads(options.manifest_binding.read_text())
    if binding["original_sha256"] != SOURCE_HASHES[".manifest.json"]:
        raise ValueError("Accepted Appendix A1 manifest linkage mismatch")
    expected = {**SOURCE_HASHES, ".manifest.json": binding["distribution_sha256"]}
    for suffix in (".manifest.json", ".pdf"):
        if sha256(sources[suffix]) != expected[suffix]:
            raise ValueError(f"Accepted Appendix A1 provenance hash mismatch: {suffix}")
    groups = load_points(content)
    figure, fits = make_figure(groups)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=False)
    pdf, png, table = (output / f"{STEM}.{ext}" for ext in ("pdf", "png", "csv"))
    figure.savefig(pdf)
    figure.savefig(png, dpi=300)
    plt.close(figure)
    table.write_bytes(content)
    generator = Path(__file__).resolve()
    manifest = {
        "artifact": "Appendix A1 v4 two-panel presentation-only reconstruction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_data_policy": "Only the accepted saved CSV supplies plot data. The accepted manifest and PDF are hashed for provenance.",
        "source_sha256": {sources[ext].name: digest
                          for ext, digest in expected.items()},
        "generator": generator.name,
        "generator_sha256": sha256(generator),
        "output_sha256": {path.name: sha256(path) for path in (pdf, png, table)},
        "row_counts": {method: len(values[0]) for method, values in groups.items()},
        "extended_cs_trials": [f"{trial:03d}" for trial in EXTENDED_CS],
        "extended_cs_retained_transitions": 450000,
        "minima": {method: {"forward_error": f"{values[3][0]:03d}",
                            "energy_distance": f"{values[3][1]:03d}"}
                   for method, values in groups.items()},
        "least_squares_fits": fits,
        "standard_error_column": SE,
        "plotting_environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "executed": {"mcmc": False, "sccm": False, "surrogate_training": False,
                     "energy_distance_recomputation": False},
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Appendix A1 for manuscript v4\n\n"
        "This is a presentation-only reconstruction from the accepted candidate-extension "
        "CSV listed in the provenance record. The plotted CSV is a byte-for-byte copy. "
        "The five extended CS trials retain their accepted 450,000-transition estimates.\n\n"
        "Each panel shows 20 trials with stored ±1 standard error bars and a separate "
        "ordinary least-squares fit. The panels use shared linear axes and follow the "
        "v4 caption.\n\n"
        "Run `python common/scripts/plot_appendix_posterior_variability_two_panel.py` "
        "with the Derecho study environment and a fresh output directory. The script "
        "reads no chains and runs no MCMC, SCCM, surrogate training, or energy-distance "
        "recomputation. The accepted source manifest and PDF are checked by hash.\n"
    )


if __name__ == "__main__":
    main()
