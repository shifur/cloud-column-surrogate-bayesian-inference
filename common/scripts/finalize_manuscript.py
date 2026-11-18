#!/usr/bin/env python3
"""Finalize production manuscript figures and tables from registered artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import tomllib

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_column_reproduction.mcmc.production import _relative_artifact_path
from cloud_column_reproduction.mcmc.config import OUTPUT_ORDER, load_config
from cloud_column_reproduction.mcmc.energy_distance import (
    normalize_prior,
    pairwise_ed_matrix,
    replicated_subsample_ed,
)
from cloud_column_reproduction.mcmc.production import run_prefix_diagnostics
from cloud_column_reproduction.provenance import (
    sha256_file,
    validate_manifest,
    write_manifest,
)

LABELS = {"sccm": "SCCM", "dnn": "DNN", "pdr": "CS"}
COLORS = {"sccm": "black", "dnn": "#d62728", "pdr": "#1f77b4"}
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


def _settings(path: Path) -> tuple[dict, object]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("schema_version") != 1:
        raise ValueError("Manuscript finalization config must use schema_version=1")
    mcmc_path = (ROOT / raw["mcmc_config"]).resolve()
    return raw, load_config(mcmc_path, validate_registered_artifacts=True)


def _style() -> None:
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


def _chain_paths(config, root: Path, case: str, method: str) -> tuple[Path, Path]:
    paths = []
    for ensemble in range(2):
        directory = root / case / method / f"ensemble_{ensemble:03d}"
        chain = directory / "chain.nc"
        manifest_path = directory / "chain.manifest.json"
        if not chain.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete finalized chain: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest["chain_sha256"] != sha256_file(chain):
            raise ValueError(f"Chain checksum mismatch: {chain}")
        expected = {
            "config_sha256": sha256_file(config.path),
            "method": method,
            "likelihood_case": case,
            "ensemble_id": ensemble,
            "parameter_box_sha256": config.parameter_box.fingerprint,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Chain manifest binding mismatch for {key}: {manifest_path}"
                )
        paths.append(chain)
    return tuple(paths)


def _sample_chains(paths: tuple[Path, Path], transitions: int, seed: int) -> np.ndarray:
    import h5netcdf

    arrays = []
    for ensemble, path in enumerate(paths):
        with h5netcdf.File(path, "r") as dataset:
            variable = dataset.variables["position"]
            available = variable.shape[0]
            if transitions > available:
                raise ValueError(
                    f"Requested {transitions} transitions from {available}: {path}"
                )
            rng = np.random.default_rng(seed + 1009 * ensemble)
            indices = np.sort(rng.choice(available, transitions, replace=False))
            rows = np.empty(
                (transitions, variable.shape[1], variable.shape[2]), dtype=np.float64
            )
            for row, index in enumerate(indices):
                rows[row] = variable[int(index), :, :]
            arrays.append(rows.reshape(-1, rows.shape[-1]))
    values = np.concatenate(arrays, axis=0)
    if not np.all(np.isfinite(values)):
        raise ValueError("Posterior sample contains nonfinite values")
    return values


def _manifest(
    artifact: str, outputs: list[Path], config, source_paths: list[Path], extra: dict,
    *, environment_specification: str = "environment/derecho.yml",
) -> None:
    manifest_path = outputs[0].with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        {
            "artifact": artifact,
            "artifact_role": "production-manuscript-finalizer",
            "config": _relative_artifact_path(config.path, ROOT),
            "config_sha256": sha256_file(config.path),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "outputs": [_relative_artifact_path(path, ROOT) for path in outputs],
            "output_sha256": {
                _relative_artifact_path(path, ROOT): sha256_file(path) for path in outputs
            },
            "sources": [_relative_artifact_path(path, ROOT) for path in source_paths],
            "source_sha256": {
                _relative_artifact_path(path, ROOT): sha256_file(path) for path in source_paths
            },
            **extra,
        },
        repository=ROOT,
        environment_specification=environment_specification,
    )


def figure2(settings: dict, config) -> list[Path]:
    model = ROOT / config.raw["surrogates"]["pdr"]["model"]
    if sha256_file(model) != config.raw["surrogates"]["pdr"]["model_sha256"]:
        raise ValueError("Selected CS model checksum mismatch")
    with np.load(model, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        indices = np.asarray(archive["indices"], dtype=np.int64)
    if coefficients.shape[0] != len(indices) and coefficients.shape[1] == len(indices):
        coefficients = coefficients.T
    if coefficients.shape != (len(indices), 6) or indices.shape[1] != 11:
        raise ValueError("Selected CS coefficient artifact has an unexpected shape")
    figure_dir = ROOT / settings["figure_directory"]
    table_dir = ROOT / settings["table_directory"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        OUTPUT_ORDER.index(name)
        for name in settings["coefficient_structure"]["outputs"]
    ]
    rows = []
    _style()
    figure, axes = plt.subplots(len(selected), 2, figsize=(10.2, 6.3), squeeze=False)
    for panel, output_index in enumerate(selected):
        square = coefficients[:, output_index] ** 2
        nonconstant = np.any(indices > 0, axis=1)
        variance = float(np.sum(square[nonconstant]))
        if variance <= 0:
            raise ValueError("CS output has zero nonconstant coefficient energy")
        main = np.zeros(11)
        total = np.zeros(11)
        pair = np.zeros((11, 11))
        support_count = np.count_nonzero(indices, axis=1)
        for j in range(11):
            total[j] = np.sum(square[indices[:, j] > 0]) / variance
            main[j] = (
                np.sum(square[(indices[:, j] > 0) & (support_count == 1)]) / variance
            )
            for k in range(j + 1, 11):
                mask = (indices[:, j] > 0) & (indices[:, k] > 0) & (support_count == 2)
                pair[j, k] = pair[k, j] = np.sum(square[mask]) / variance
            rows.append(
                {
                    "output": OUTPUT_ORDER[output_index],
                    "parameter": config.parameter_box.names[j],
                    "main_effect": main[j],
                    "total_effect": total[j],
                }
            )
        x = np.arange(11)
        axes[panel, 0].bar(x - 0.19, main, width=0.38, label="Main", color="#4c78a8")
        axes[panel, 0].bar(x + 0.19, total, width=0.38, label="Total", color="#f58518")
        axes[panel, 0].set_xticks(
            x,
            [PARAMETER_LATEX[name] for name in config.parameter_box.names],
            rotation=45,
            ha="right",
        )
        axes[panel, 0].set_ylabel("Fraction of coefficient variance")
        axes[panel, 0].set_title(
            f"({chr(97 + 2*panel)}) {OUTPUT_ORDER[output_index]} main and total effects"
        )
        axes[panel, 0].grid(axis="y", alpha=0.25)
        axes[panel, 0].legend(frameon=False)
        image = axes[panel, 1].imshow(pair, origin="lower", cmap="magma", vmin=0)
        axes[panel, 1].set_xticks(
            x,
            [PARAMETER_LATEX[name] for name in config.parameter_box.names],
            rotation=45,
            ha="right",
        )
        axes[panel, 1].set_yticks(
            x, [PARAMETER_LATEX[name] for name in config.parameter_box.names]
        )
        axes[panel, 1].set_title(
            f"({chr(98 + 2*panel)}) {OUTPUT_ORDER[output_index]} pure pair effects"
        )
        figure.colorbar(image, ax=axes[panel, 1], fraction=0.046, pad=0.04)
    figure.tight_layout()
    pdf = figure_dir / "figure2_cs_coefficient_structure.pdf"
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    table = table_dir / "figure2_cs_effects.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _manifest(
        "manuscript-figure-2",
        [pdf, table],
        config,
        [model],
        {
            "coefficient_statistic": "orthonormal-coefficient squared-energy fractions",
            "outputs_displayed": settings["coefficient_structure"]["outputs"],
        },
    )
    return [pdf, table]


def figure5(settings: dict, config) -> list[Path]:
    """Plot observed paired-workload elapsed clocks, never averaged ensemble clocks."""
    from cloud_column_reproduction.mcmc.matched_timing import timing_summary

    root = ROOT / settings["chain_root"]
    case = settings["primary_likelihood_case"]
    methods = ("dnn", "pdr")
    rows = [timing_summary(root, case, method) for method in methods]
    for row in rows:
        if row["config_sha256"] != sha256_file(config.path):
            raise ValueError("Figure 5 timing config checksum mismatch")
        if row["milliseconds_per_paired_transition"] is None:
            raise ValueError("Figure 5 requires balanced ensembles and complete matched elapsed clocks")
        if row["ensemble_transitions"] != [330000, 330000]:
            raise ValueError("Final manuscript Figure 5 requires completed 30k + 300k sampling")
    figure_dir = ROOT / settings["figure_directory"]
    table_dir = ROOT / settings["table_directory"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    table = table_dir / "figure5_mcmc_timing.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _style()
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    axis.bar(np.arange(2), [r["milliseconds_per_paired_transition"] for r in rows],
             color=[COLORS[m] for m in methods])
    axis.set_xticks(np.arange(2), [LABELS[m] for m in methods])
    axis.set_ylabel("Elapsed ms per paired transition (log scale)")
    axis.set_title("Matched concurrent MCMC transition time")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    pdf = figure_dir / "figure5_mcmc_timing.pdf"
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    sources = [Path(row["ledger"]) for row in rows] + [
        root / case / method / f"ensemble_{ensemble:03d}" / "checkpoint.json"
        for method in methods for ensemble in range(2)
    ]
    _manifest("manuscript-figure-5", [pdf, table], config, sources, {
        "likelihood_case": case, "timing_scope": rows[0]["scope"],
        "paired_transition_definition": rows[0]["paired_transition_definition"],
        "sccm": "omitted: historical per-ensemble productive clocks do not measure paired elapsed time",
        "error_bars": "none: the two concurrent ensembles share a single elapsed clock",
    })
    return [pdf, table]


def figure6(settings: dict, config) -> list[Path]:
    root = ROOT / settings["chain_root"]
    case = settings["primary_likelihood_case"]
    policy = settings["energy_distance"]
    paths = {m: _chain_paths(config, root, case, m) for m in ("sccm", "dnn", "pdr")}
    pools = {
        m: normalize_prior(
            _sample_chains(
                p, int(policy["source_transitions_per_ensemble"]), 81000 + 100 * i
            ),
            config.parameter_box,
        )
        for i, (m, p) in enumerate(paths.items())
    }
    kwargs = {
        "sample_size": int(policy["subsample_size"]),
        "seeds": tuple(int(x) for x in policy["seeds"]),
        "chunk_rows": int(policy["chunk_rows"]),
    }
    matrices = {}
    ses = {}
    for method in ("dnn", "pdr"):
        matrices[method], ses[method] = pairwise_ed_matrix(
            pools[method], pools["sccm"], **kwargs
        )
    reference_a = normalize_prior(
        _sample_chains(
            (paths["sccm"][0], paths["sccm"][0]),
            int(policy["source_transitions_per_ensemble"]),
            85001,
        )[: int(policy["source_transitions_per_ensemble"]) * 32],
        config.parameter_box,
    )
    reference_b = normalize_prior(
        _sample_chains(
            (paths["sccm"][1], paths["sccm"][1]),
            int(policy["source_transitions_per_ensemble"]),
            86001,
        )[: int(policy["source_transitions_per_ensemble"]) * 32],
        config.parameter_box,
    )
    floor, floor_se = pairwise_ed_matrix(reference_a, reference_b, **kwargs)
    full = {
        m: replicated_subsample_ed(pools[m], pools["sccm"], **kwargs)
        for m in ("dnn", "pdr")
    }
    full["sccm_floor"] = replicated_subsample_ed(reference_a, reference_b, **kwargs)
    figure_dir = ROOT / settings["figure_directory"]
    table_dir = ROOT / settings["table_directory"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    vmax = max(
        float(np.max(matrices["dnn"])),
        float(np.max(matrices["pdr"])),
        float(np.max(floor)),
    )
    _style()
    figure = plt.figure(figsize=(13.2, 4.3))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.06],
        wspace=0.26,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_axis = figure.add_subplot(grid[0, 3])

    names = [PARAMETER_LATEX[n] for n in config.parameter_box.names]
    for axis, key, title in zip(
        axes,
        ("dnn", "pdr", "floor"),
        ("(a) DNN vs SCCM", "(b) CS vs SCCM", "(c) SCCM ensemble floor"),
        strict=True,
    ):
        matrix = floor if key == "floor" else matrices[key]
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )
        axis.set_xticks(
            range(11),
            names,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(range(11), names)
        axis.tick_params(axis="x", pad=7)
        axis.tick_params(axis="y", pad=4)
        axis.set_title(title)

    figure.colorbar(
        image,
        cax=colorbar_axis,
        label="Prior-normalized energy distance",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.965,
        bottom=0.245,
        top=0.90,
    )
    pdf = figure_dir / "figure6_posterior_energy_distance.pdf"
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    table = table_dir / "figure6_joint_energy_distance.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "comparison",
            "mean",
            "standard_error",
            "ci95_lower",
            "ci95_upper",
            "subsample_size",
            "replications",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, result in full.items():
            writer.writerow(
                {
                    "comparison": key,
                    "mean": result.mean,
                    "standard_error": result.standard_error,
                    "ci95_lower": result.ci95[0],
                    "ci95_upper": result.ci95[1],
                    "subsample_size": result.sample_size_x,
                    "replications": len(result.seeds),
                }
            )
    sources = [path for group in paths.values() for path in group]
    _manifest(
        "manuscript-figure-6",
        [pdf, table],
        config,
        sources,
        {
            "likelihood_case": case,
            "coordinate_system": "prior-normalized",
            "reference_floor": "independent SCCM ensembles",
            "subsample_size": policy["subsample_size"],
            "replication_seeds": policy["seeds"],
        },
        environment_specification=settings.get("environment_specification", "environment/derecho.yml"),
    )
    return [pdf, table]


def _density_contours(axis, x, y, bins, color):
    counts, xe, ye = np.histogram2d(x, y, bins=bins, density=True)
    positive = np.sort(counts[counts > 0])[::-1]
    if len(positive) < 4:
        return
    cumulative = np.cumsum(positive) / np.sum(positive)
    levels = []
    for mass in (0.5, 0.8, 0.95):
        levels.append(
            positive[min(np.searchsorted(cumulative, mass), len(positive) - 1)]
        )
    levels = np.unique(levels)
    centers_x = 0.5 * (xe[:-1] + xe[1:])
    centers_y = 0.5 * (ye[:-1] + ye[1:])
    axis.contour(
        centers_x,
        centers_y,
        counts.T,
        levels=levels,
        colors=[color],
        linewidths=0.9,
        alpha=0.9,
    )


def figure7(settings: dict, config) -> list[Path]:
    root = ROOT / settings["chain_root"]
    policy = settings["corner"]
    figure_dir = ROOT / settings["figure_directory"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    parameter_names = list(config.parameter_box.names)
    selected = [parameter_names.index(name) for name in policy["parameter_names"]]
    outputs = []
    for case in (
        settings["primary_likelihood_case"],
        settings["secondary_likelihood_case"],
    ):
        paths = {m: _chain_paths(config, root, case, m) for m in ("sccm", "dnn", "pdr")}
        samples = {
            m: _sample_chains(
                p,
                int(policy["transitions_per_ensemble"]),
                int(policy["seed"]) + 100 * i,
            )[:, selected]
            for i, (m, p) in enumerate(paths.items())
        }
        _style()
        figure, axes = plt.subplots(4, 4, figsize=(9.2, 9.2))
        labels = [PARAMETER_LATEX[parameter_names[j]] for j in selected]
        truth = config.truth[selected]
        for row in range(4):
            for column in range(4):
                axis = axes[row, column]
                if column > row:
                    axis.axis("off")
                    continue
                if row == column:
                    for method in ("sccm", "dnn", "pdr"):
                        axis.hist(
                            samples[method][:, column],
                            bins=int(policy["bins_1d"]),
                            density=True,
                            histtype="step",
                            linewidth=1.3,
                            color=COLORS[method],
                            label=LABELS[method],
                        )
                    axis.axvline(
                        truth[column], color="0.35", linestyle="--", linewidth=1
                    )
                else:
                    for method in ("sccm", "dnn", "pdr"):
                        _density_contours(
                            axis,
                            samples[method][:, column],
                            samples[method][:, row],
                            int(policy["bins_2d"]),
                            COLORS[method],
                        )
                    axis.plot(
                        truth[column], truth[row], marker="+", color="0.2", markersize=7
                    )
                if row == 3:
                    axis.set_xlabel(labels[column])
                else:
                    axis.set_xticklabels([])
                if column == 0 and row > 0:
                    axis.set_ylabel(labels[row])
                elif column > 0:
                    axis.set_yticklabels([])
                axis.grid(alpha=0.12)
        axes[0, 0].legend(frameon=False, fontsize=8)
        figure.suptitle(f"Posterior marginals: {case.replace('_',' ')}", y=0.995)
        figure.tight_layout()
        pdf = figure_dir / f"figure7_corner_{case}.pdf"
        figure.savefig(pdf, bbox_inches="tight")
        plt.close(figure)
        outputs.append(pdf)
        _manifest(
            "manuscript-figure-7",
            [pdf],
            config,
            [path for group in paths.values() for path in group],
            {
                "likelihood_case": case,
                "parameters": policy["parameter_names"],
                "sample_source": "fixed random transition subsample from both independent ensembles",
            },
        )
    return outputs


def convergence(settings: dict, config) -> list[Path]:
    root = ROOT / settings["chain_root"]
    outputs = []
    for case in config.execution_cases():
        for method in config.execution_methods():
            for prefix in config.raw["sampler"]["checkpointing"]["diagnostic_prefixes"]:
                target = (
                    root
                    / "diagnostics"
                    / case
                    / method
                    / f"prefix_{int(prefix):09d}.npz"
                )
                if target.is_file() and target.with_suffix(".manifest.json").is_file():
                    continue
                run_prefix_diagnostics(
                    config,
                    likelihood_case=case,
                    method=method,
                    prefix=int(prefix),
                    output_root=root,
                )
        output = (
            ROOT / settings["figure_directory"] / f"appendix_convergence_{case}.pdf"
        )
        subprocess = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/plot_mcmc_convergence.py"),
            "--diagnostics-root",
            str(root / "diagnostics" / case),
            "--output",
            str(output),
        ]
        import subprocess as process

        process.run(subprocess, cwd=ROOT, check=True)
        outputs.extend([output, output.with_suffix(".csv")])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/manuscript.toml")
    parser.add_argument(
        "target",
        choices=("figure2", "figure5", "figure6", "figure7", "convergence", "all"),
    )
    arguments = parser.parse_args()
    settings, config = _settings(arguments.config.resolve())
    targets = (
        ("figure2", "figure5", "figure6", "figure7", "convergence")
        if arguments.target == "all"
        else (arguments.target,)
    )
    result = {}
    for target in targets:
        result[target] = [str(path) for path in globals()[target](settings, config)]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
