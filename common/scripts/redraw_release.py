"""Reproduce a v4 figure from archived evidence in a fresh output directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from cloud_column_reproduction.mcmc.config import load_config


def registered_config(formal, project):
    """Recreate the small registered-input layout without copying the chains."""
    inputs = {
        "configs": formal / "configs",
        "data/mcmc/production": formal / "synthetic_observation",
        "outputs/figure1/production_v1": formal / "selected_models",
    }
    for destination, source in inputs.items():
        shutil.copytree(source, project / destination)
    return load_config(project / "configs/mcmc.toml", validate_registered_artifacts=True)


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("1", "2", "3", "4", "5", "6", "7", "A1", "B1"))
    parser.add_argument("--archive", type=Path, required=True, help="Root of the companion scientific-data record")
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    archive = options.archive.resolve()
    output = options.output_dir.resolve()
    for protected in ("data", "figures", "formal_mcmc", "configs", "provenance", "software"):
        if output.is_relative_to(archive / protected):
            parser.error("Use a fresh directory outside the archived inputs and canonical figures")
    if output.exists():
        parser.error("Output directory must be new")
    policy = tomllib.loads((archive / "configs/plot-release.toml").read_text())
    formal = archive / policy["formal"]
    figure = options.figure
    if figure == "1":
        output.mkdir(parents=True)
        run("figure1.py", "--config", HERE.parent / "configs/figure1.toml", "plot",
            "--metrics", archive / policy["figure1_metrics"], "--output", output / "figure1.pdf")
    elif figure == "3":
        run("replot_sensitivity_1d_manuscript.py", "--data-dir", archive / policy["figure3_data"],
            "--output-dir", output)
    elif figure == "4":
        run("generate_figure4.py", "--timings", archive / policy["figure4_timings"], "--output-dir", output)
    elif figure == "5":
        run("replot_figure5_with_sccm.py", "--archive", archive, "--output-dir", output)
    elif figure == "A1":
        run("plot_appendix_posterior_variability_two_panel.py",
            "--csv", archive / policy["appendix_csv"], "--manifest", archive / policy["appendix_manifest"],
            "--manifest-binding", archive / "provenance/appendix-a1/distribution_binding.json",
            "--reference-pdf", archive / policy["appendix_reference"], "--output-dir", output)
    elif figure == "B1":
        output.mkdir(parents=True)
        run("plot_mcmc_convergence.py", "--diagnostics-root", formal / "diagnostics/full_except_acc",
            "--output", output / "appendix_convergence_full_except_acc.pdf")
    else:
        output.mkdir(parents=True)
        with tempfile.TemporaryDirectory(dir=output, prefix=".registered-") as work:
            config = registered_config(formal, Path(work))
            if figure == "2":
                run("figure2_interpretability_4d.py", "--config", config.path, "--output-dir", output)
                return
            settings = tomllib.loads((formal / "configs/manuscript.toml").read_text())
            settings.update(chain_root=str(formal), figure_directory=str(output), table_directory=str(output))
            if figure == "6":
                import finalize_manuscript
                # Provenance is relative to a common directory for archive and output.
                finalize_manuscript.ROOT = Path(os.path.commonpath([archive, output]))
                settings["environment_specification"] = str(
                    (archive / "environment/plot-environment.yml").relative_to(finalize_manuscript.ROOT)
                )
                finalize_manuscript.figure6(settings, config)
            else:
                import figure7_corner_full_except_acc_contour
                figure7_corner_full_except_acc_contour.main((settings, config))


if __name__ == "__main__":
    main()
