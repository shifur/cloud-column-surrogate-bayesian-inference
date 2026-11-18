"""Render Figure 4 from the audited Derecho timings without running any fits."""

import argparse
import csv
import hashlib
import io
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

TIMINGS_SHA256 = "dd48057c67fc694c0390ade482b07d5320bb7215b55cbf9940c1a8df6ee011ff"
SIZES = range(100, 5001, 100)
TRIALS = set(range(20))
FIELDS = ("dnn_total_training", "cs_total_fit")
STEM = "figure4_training_cost_derecho_final"


def load_statistics(timings):
    raw = timings.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TIMINGS_SHA256:
        raise ValueError("Timing data differ from the audited production input")

    by_size = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        size, trial = int(row["train_size"]), int(row["trial"])
        if trial in by_size[size]:
            raise ValueError(f"Duplicate timing for m={size}, trial={trial}")
        values = tuple(float(row[field]) for field in FIELDS)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"Invalid timing for m={size}, trial={trial}")
        by_size[size][trial] = values

    if set(by_size) != set(SIZES) or any(set(rows) != TRIALS for rows in by_size.values()):
        raise ValueError("Expected 50 training sizes with 20 paired trials per size")

    summaries = []
    for size in SIZES:
        dnn, cs = zip(*by_size[size].values())
        summaries.append((size, len(dnn), mean(dnn), stdev(dnn), mean(cs), stdev(cs)))
    return summaries


def build_figure(summaries):
    sizes, _, dnn_mean, dnn_sd, cs_mean, cs_sd = zip(*summaries)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), sharex=True, sharey=True,
                             layout="none")
    panels = (
        ("(a) DNN Training Time", dnn_mean, dnn_sd, "#245b8b"),
        ("(b) CS Fitting Time", cs_mean, cs_sd, "#a04d23"),
    )
    for ax, (title, means, errors, color) in zip(axes, panels):
        if any(center - error <= 0 for center, error in zip(means, errors)):
            raise ValueError("The full standard deviation bar must fit on a log axis")
        ax.errorbar(sizes, means, yerr=errors, fmt="o", linestyle="none",
                    color=color, markersize=2.0, elinewidth=0.8, capsize=1.6,
                    capthick=0.8, alpha=0.9)
        ax.set_title(title + r" ($d=11$)", pad=9)
        ax.set_xlabel(r"Training samples $m$")
        ax.set_ylabel("Time (seconds, log scale)")
        ax.set_yscale("log")
        ax.set_xlim(-20, 5120)
        ax.set_ylim(1, 1000)
        ax.set_xticks([100, 1000, 2000, 3000, 4000, 5000])
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="minor", length=0)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", which="both", color="0.88", linewidth=0.55)
        ax.set_axisbelow(True)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    summaries = load_statistics(options.timings)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=False)
    with (output / f"{STEM}_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("train_size", "trial_count", "dnn_mean_seconds", "dnn_sample_sd_seconds",
                         "cs_mean_seconds", "cs_sample_sd_seconds"))
        writer.writerows(summaries)

    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.titlesize": 10, "pdf.fonttype": 42}):
        fig = build_figure(summaries)
        # Keep the original panel bounds for each output renderer.
        for extension, left, right, bottom, top, options in (
            ("pdf", 0.067830479336, 0.985202319999, 0.139854453263, 0.911815520282,
             {"metadata": {"CreationDate": None}}),
            ("png", 0.067863296070, 0.985252202577, 0.142152733686, 0.911639153439,
             {"dpi": 300}),
        ):
            panel_width = right - left - 0.5
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                                wspace=(0.5 - panel_width) / panel_width)
            fig.savefig(output / f"{STEM}.{extension}", **options)
        plt.close(fig)


if __name__ == "__main__":
    main()
