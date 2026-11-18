"""Generate manuscript Figure 1 exclusively from validated paired-trial metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Figure1Config


METHOD_STYLE = {
    "CS": {"color": "#1f77b4", "label": "CS"},
    "DNN": {"color": "#2ca02c", "label": "DNN"},
}


def plot_learning_curves(
    summary: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    config: Figure1Config,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(float(config.plot["width_inches"]), float(config.plot["height_inches"])),
    )
    sizes = np.asarray(config.train_sizes)
    panel_labels = "abcdef"

    for index, (axis, output) in enumerate(zip(axes.ravel(), config.outputs, strict=True)):
        for method in config.methods:
            means, standard_deviations = summary[(output, method)]
            style = METHOD_STYLE[method]
            axis.plot(sizes, means, linewidth=1.8, color=style["color"], label=style["label"])
            lower = np.maximum(means - standard_deviations, np.finfo(float).tiny)
            upper = means + standard_deviations
            axis.fill_between(sizes, lower, upper, color=style["color"], alpha=0.20)
        axis.set_title(f"({panel_labels[index]}) {output}", fontweight="bold")
        axis.set_xlabel("Training Sample Size")
        axis.set_ylabel(r"Relative $L^2$ Error")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=int(config.plot["dpi"]), bbox_inches="tight")
    plt.close(figure)
    return destination

