#!/usr/bin/env python3
"""Run, resume, aggregate, and plot paired Figure 1 surrogate fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cloud_column_reproduction.figure1.config import load_config
from cloud_column_reproduction.figure1.fitting import fit_surrogate_pair
from cloud_column_reproduction.figure1.dnn import (
    DNN_ALGORITHM_REVISION,
    DNN_DEFAULT_GPU_TRAINING_BACKEND,
    DNN_OPTIMIZED_ALGORITHM_REVISION,
    DNN_TRAINING_BACKEND_EAGER,
    DNN_TRAINING_BACKEND_OPTIMIZED,
)
from cloud_column_reproduction.figure1.pdr import PDR_ALGORITHM_REVISION
from cloud_column_reproduction.figure1.metrics import (
    read_metrics,
    summarize_metrics,
    validate_metrics,
    write_summary,
)
from cloud_column_reproduction.figure1.plot import plot_learning_curves
from cloud_column_reproduction.provenance import sha256_file, write_manifest


DEFAULT_CONFIG = REPOSITORY / "configs" / "figure1.toml"


def _fit_directory(output_directory: Path, trial: int, sample_count: int) -> Path:
    """Return a validated fit directory below the requested output root."""
    if trial < 0:
        raise ValueError("Figure 1 trial indices must be nonnegative")
    if sample_count < 1:
        raise ValueError("Figure 1 training sizes must be positive")
    root = output_directory.resolve()
    destination = (
        root / f"trial_{trial:03d}" / f"n{sample_count:05d}"
    ).resolve()
    if destination.parent.parent != root:
        raise ValueError(
            f"Refusing to use a fit directory outside the output root: {destination}"
        )
    if destination.parent.name != f"trial_{trial:03d}":
        raise ValueError(f"Unexpected Figure 1 trial directory: {destination.parent}")
    if destination.name != f"n{sample_count:05d}":
        raise ValueError(f"Unexpected Figure 1 fit directory: {destination}")
    return destination


def _discard_incomplete_fit(
    directory: Path,
    output_directory: Path,
    trial: int,
    sample_count: int,
) -> None:
    """Remove one exact incomplete fit so method timings restart from zero."""
    expected = _fit_directory(output_directory, trial, sample_count)
    if directory.resolve() != expected:
        raise ValueError(
            f"Refusing to discard unexpected fit directory {directory}; "
            f"expected {expected}"
        )
    if not directory.exists():
        return
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Refusing to discard a non-directory fit path: {directory}")
    print(f"discarding incomplete fit directory: {directory}", flush=True)
    shutil.rmtree(directory)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit one trial/training-size pair")
    fit.add_argument("--trial", type=int, required=True)
    fit.add_argument("--samples", type=int, required=True)
    fit.add_argument("--training", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--evaluation", type=Path, required=True)
    fit.add_argument("--output-directory", type=Path, required=True)
    fit.add_argument(
        "--design-cache-directory", type=Path,
        help="checksum-bound q=23 float64 design cache prepared by figure1_design_cache.py",
    )
    fit.add_argument("--cpu", action="store_true")

    suite = commands.add_parser("suite", help="resume every fit declared by the config")
    suite.add_argument("--training-directory", type=Path, required=True)
    suite.add_argument("--validation", type=Path, required=True)
    suite.add_argument("--evaluation", type=Path, required=True)
    suite.add_argument("--output-directory", type=Path, required=True)
    suite.add_argument(
        "--design-cache-directory", type=Path,
        help="checksum-bound q=23 float64 design cache prepared by figure1_design_cache.py",
    )
    suite.add_argument("--cpu", action="store_true")
    suite.add_argument(
        "--trial",
        type=int,
        help="run only one configured trial; default runs all trials",
    )

    cs_suite = commands.add_parser(
        "cs-timing-suite",
        help="resume the 1,000-fit optimized q=23 CS-only timing benchmark",
    )
    cs_suite.add_argument("--training-directory", type=Path, required=True)
    cs_suite.add_argument("--evaluation", type=Path, required=True)
    cs_suite.add_argument("--output-directory", type=Path, required=True)
    cs_suite.add_argument("--design-cache-directory", type=Path, required=True)
    cs_suite.add_argument("--cpu", action="store_true")

    aggregate = commands.add_parser("aggregate", help="validate all metrics and plot")
    aggregate.add_argument("--fit-directory", type=Path, required=True)
    aggregate.add_argument("--metrics-output", type=Path, required=True)
    aggregate.add_argument("--figure-output", type=Path)
    aggregate.add_argument("--timings-output", type=Path)
    aggregate.add_argument("--cache-timings-output", type=Path)
    return root


def _complete_fit(
    directory: Path,
    config_hash: str,
    training_hash: str,
    validation_hash: str,
    evaluation_hash: str,
    dnn_algorithm_revision: str = DNN_ALGORITHM_REVISION,
) -> bool:
    manifest_path = directory / "fit.manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("config_sha256") != config_hash:
            return False
        if payload.get("training_data_sha256") != training_hash:
            return False
        if payload.get("validation_data_sha256") != validation_hash:
            return False
        if payload.get("evaluation_data_sha256") != evaluation_hash:
            return False
        if (
            payload.get("pdr_hyperparameters", {}).get("algorithm_revision")
            != PDR_ALGORITHM_REVISION
        ):
            return False
        if (
            payload.get("dnn_diagnostics", {}).get("algorithm_revision")
            != dnn_algorithm_revision
        ):
            return False
        for name_field, hash_field in (
            ("pdr_model", "pdr_model_sha256"),
            ("pdr_convergence", "pdr_convergence_sha256"),
            ("dnn_model", "dnn_model_sha256"),
            ("dnn_final_epoch_model", "dnn_final_epoch_model_sha256"),
            ("dnn_validation_history", "dnn_validation_history_sha256"),
            ("dnn_training_history", "dnn_training_history_sha256"),
            ("metrics", "metrics_sha256"),
        ):
            artifact = directory / payload[name_field]
            if not artifact.is_file() or sha256_file(artifact) != payload[hash_field]:
                return False
        if payload.get("evaluation_metric") == "replicated-rqmc-relative-l2":
            artifact = directory / payload["rqmc_convergence"]
            if (
                not artifact.is_file()
                or sha256_file(artifact) != payload["rqmc_convergence_sha256"]
            ):
                return False
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _configured_dnn_algorithm_revision(config, *, use_gpu: bool) -> str:
    backend = config.dnn.get(
        "training_backend",
        DNN_DEFAULT_GPU_TRAINING_BACKEND if use_gpu else DNN_TRAINING_BACKEND_EAGER,
    )
    if backend == DNN_TRAINING_BACKEND_EAGER:
        return DNN_ALGORITHM_REVISION
    if backend == DNN_TRAINING_BACKEND_OPTIMIZED:
        return DNN_OPTIMIZED_ALGORITHM_REVISION
    raise ValueError(f"Unsupported DNN training backend: {backend}")


def _complete_cs_timing_fit(
    directory: Path, config_hash: str, training_hash: str,
    evaluation_hash: str, design_cache_hash: str,
) -> bool:
    manifest_path = directory / "fit.manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("artifact") != "figure1-cs-only-execution-timing-benchmark":
            return False
        if payload.get("artifact_role") != "execution-timing-benchmark-not-manuscript-result":
            return False
        if any((
            payload.get("config_sha256") != config_hash,
            payload.get("training_data_sha256") != training_hash,
            payload.get("evaluation_data_sha256") != evaluation_hash,
            payload.get("cs_design_cache", {}).get("data_sha256") != design_cache_hash,
            payload.get("pdr_hyperparameters", {}).get("algorithm_revision") != PDR_ALGORITHM_REVISION,
        )):
            return False
        for name_field, hash_field in (
            ("pdr_model", "pdr_model_sha256"),
            ("pdr_convergence", "pdr_convergence_sha256"),
            ("rqmc_convergence", "rqmc_convergence_sha256"),
            ("metrics", "metrics_sha256"),
        ):
            artifact = directory / payload[name_field]
            if not artifact.is_file() or sha256_file(artifact) != payload[hash_field]:
                return False
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _run_fit(arguments: argparse.Namespace, config) -> None:
    if config.cs.get("offline_cache", {}).get("enabled") and arguments.design_cache_directory is None:
        raise ValueError("Authoritative q=23 fit requires --design-cache-directory")
    training_hash = sha256_file(arguments.training)
    validation_hash = sha256_file(arguments.validation)
    evaluation_hash = sha256_file(arguments.evaluation)
    directory = _fit_directory(
        arguments.output_directory, arguments.trial, arguments.samples
    )
    if _complete_fit(
        directory,
        sha256_file(config.path),
        training_hash,
        validation_hash,
        evaluation_hash,
        _configured_dnn_algorithm_revision(config, use_gpu=not arguments.cpu),
    ):
        print(f"existing complete fit: {directory}")
        print(directory / "metrics.csv")
        print(directory / "fit.manifest.json")
        return
    _discard_incomplete_fit(
        directory, arguments.output_directory, arguments.trial, arguments.samples
    )
    pdr_design_cache = None
    if arguments.design_cache_directory is not None:
        from cloud_column_reproduction.figure1.design_cache import load_design_cache
        pdr_design_cache = load_design_cache(
            config, REPOSITORY, arguments.training, arguments.design_cache_directory,
            trial=arguments.trial,
        )
    metrics, manifest = fit_surrogate_pair(
        config,
        REPOSITORY,
        arguments.training,
        arguments.validation,
        arguments.evaluation,
        arguments.output_directory,
        trial=arguments.trial,
        sample_count=arguments.samples,
        use_gpu=not arguments.cpu,
        pdr_design_cache=pdr_design_cache,
    )
    print(metrics)
    print(manifest)


def _run_suite(arguments: argparse.Namespace, config) -> None:
    cs_only = arguments.command == "cs-timing-suite"
    if config.cs.get("offline_cache", {}).get("enabled") and arguments.design_cache_directory is None:
        raise ValueError("Authoritative q=23 suite requires --design-cache-directory")
    training_directory = arguments.training_directory.resolve()
    output_directory = arguments.output_directory.resolve()
    config_hash = sha256_file(config.path)
    validation_hash = (
        sha256_file(arguments.validation) if not cs_only else None
    )
    if cs_only and int(config.cs["hyperbolic_cross_order"]) != 23:
        raise ValueError("The optimized CS timing benchmark is defined for q=23")
    evaluation_hash = sha256_file(arguments.evaluation)
    configured_trials = int(config.experiment["trials"])
    requested_trial = getattr(arguments, "trial", None)
    if requested_trial is None:
        trial_indices = tuple(range(configured_trials))
    else:
        if requested_trial < 0 or requested_trial >= configured_trials:
            raise ValueError(
                f"Figure 1 trial must be in 0..{configured_trials - 1}: "
                f"{requested_trial}"
            )
        trial_indices = (requested_trial,)
    total = len(trial_indices) * len(config.train_sizes)
    completed = 0
    registered_data_cache = {}
    design_caches = {}
    maximum = int(config.experiment["train_size_max"])
    for sample_count in config.train_sizes:
        for trial in trial_indices:
            training = training_directory / f"trial_{trial:03d}_n{maximum:05d}.npz"
            training_hash = sha256_file(training)
            if arguments.design_cache_directory is not None and trial not in design_caches:
                from cloud_column_reproduction.figure1.design_cache import load_design_cache
                design_caches[trial] = load_design_cache(
                    config, REPOSITORY, training, arguments.design_cache_directory,
                    trial=trial,
                )
            fit_directory = _fit_directory(output_directory, trial, sample_count)
            if cs_only:
                complete = _complete_cs_timing_fit(
                    fit_directory, config_hash, training_hash, evaluation_hash,
                    design_caches[trial][4]["data_sha256"],
                )
            else:
                assert validation_hash is not None
                complete = _complete_fit(
                    fit_directory, config_hash, training_hash, validation_hash,
                    evaluation_hash,
                    _configured_dnn_algorithm_revision(
                        config, use_gpu=not arguments.cpu
                    ),
                )
            if complete:
                completed += 1
                print(f"[{completed}/{total}] existing trial={trial} n={sample_count}")
                continue
            _discard_incomplete_fit(
                fit_directory, output_directory, trial, sample_count
            )
            print(f"[{completed + 1}/{total}] fitting trial={trial} n={sample_count}")
            fit_surrogate_pair(
                config,
                REPOSITORY,
                training,
                getattr(arguments, "validation", None),
                arguments.evaluation,
                output_directory,
                trial=trial,
                sample_count=sample_count,
                use_gpu=not arguments.cpu,
                registered_data_cache=registered_data_cache,
                pdr_design_cache=design_caches.get(trial),
                cs_only_timing_benchmark=cs_only,
            )
            completed += 1

    if cs_only:
        timing_rows = []
        for manifest_path in sorted(output_directory.glob("trial_*/n*/fit.manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("artifact") != "figure1-cs-only-execution-timing-benchmark":
                continue
            backend = payload["cs_prediction_backend"]
            timing_rows.append({
                "trial": int(payload["trial"]),
                "train_size": int(payload["sample_count"]),
                "wall_seconds": float(payload["wall_seconds"]),
                "cs_dense_basis_size": int(backend["dense_basis_size"]),
                "cs_active_basis_size": int(backend["active_basis_size"]),
                "cs_coefficient_nonzeros": int(backend["coefficient_nonzeros"]),
                **{name: float(value) for name, value in payload["stage_timings"].items()},
            })
        if len(timing_rows) != total:
            raise RuntimeError(f"Expected {total} complete CS timing records, found {len(timing_rows)}")
        timing_path = output_directory / "timings_all.csv"
        _write_timing_table(timing_path, timing_rows)
        non_timing_columns = {
            "trial", "train_size", "cs_dense_basis_size",
            "cs_active_basis_size", "cs_coefficient_nonzeros",
        }
        stage_names = sorted({
            name for row in timing_rows for name in row if name not in non_timing_columns
        })
        cache_set_manifest_path = arguments.design_cache_directory.resolve() / "design_cache_set.manifest.json"
        if not cache_set_manifest_path.is_file():
            raise FileNotFoundError(f"CS cache-set manifest is missing: {cache_set_manifest_path}")
        cache_set_manifest = json.loads(cache_set_manifest_path.read_text(encoding="utf-8"))
        if cache_set_manifest.get("artifact") != "figure1-cs-design-cache-set":
            raise ValueError(f"Unexpected CS cache-set artifact: {cache_set_manifest_path}")
        summary = {
            "artifact": "figure1-cs-only-execution-timing-summary",
            "artifact_role": "execution-timing-benchmark-not-manuscript-result",
            "fit_count": len(timing_rows),
            "trials": int(config.experiment["trials"]),
            "train_sizes": len(config.train_sizes),
            "hyperbolic_cross_order": int(config.cs["hyperbolic_cross_order"]),
            "offline_cache_parallel_wall_seconds": float(cache_set_manifest["wall_seconds"]),
            "offline_cache_set_manifest": str(cache_set_manifest_path),
            "offline_cache_set_manifest_sha256": sha256_file(cache_set_manifest_path),
            "totals_seconds": {name: sum(float(row.get(name, 0.0)) for row in timing_rows) for name in stage_names},
            "means_seconds": {name: sum(float(row.get(name, 0.0)) for row in timing_rows) / len(timing_rows) for name in stage_names},
            "timings": timing_path.name,
            "timings_sha256": sha256_file(timing_path),
        }
        summary_path = output_directory / "timing_summary.json"
        temporary = summary_path.with_name(f".{summary_path.name}.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
        summary_manifest_path = output_directory / "timing_summary.manifest.json"
        write_manifest(
            summary_manifest_path,
            {
                "artifact": "figure1-cs-only-execution-timing-summary",
                "artifact_role": "execution-timing-benchmark-not-manuscript-result",
                "experiment_id": config.experiment["id"],
                "config": str(config.path.relative_to(REPOSITORY)),
                "config_sha256": config_hash,
                "parameter_names": list(config.parameter_box.names),
                "parameter_lower": list(config.parameter_box.lower),
                "parameter_upper": list(config.parameter_box.upper),
                "parameter_box_sha256": config.parameter_box.fingerprint,
                "parameter_box_config": str(config.parameter_box_path.relative_to(REPOSITORY)),
                "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
                "random_seeds": {"base_seed": int(config.experiment["base_seed"])},
                "workers": 1,
                "wall_seconds": sum(float(row["wall_seconds"]) for row in timing_rows),
                "wall_seconds_semantics": "sum-of-1000-complete-cs-fit-wall-times",
                "fit_count": len(timing_rows),
                "trials": int(config.experiment["trials"]),
                "train_sizes": [int(value) for value in config.train_sizes],
                "hyperbolic_cross_order": int(config.cs["hyperbolic_cross_order"]),
                "timings": str(timing_path),
                "timings_sha256": sha256_file(timing_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "offline_cache_set_manifest": str(cache_set_manifest_path),
                "offline_cache_set_manifest_sha256": sha256_file(cache_set_manifest_path),
            },
            repository=REPOSITORY,
        )
        print(timing_path)
        print(summary_path)
        print(summary_manifest_path)


def _write_timing_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run_aggregate(arguments: argparse.Namespace, config) -> None:
    fit_directory = arguments.fit_directory.resolve()
    config_hash = sha256_file(config.path)
    metric_paths = sorted(fit_directory.glob("trial_*/n*/metrics.csv"))
    rows = []
    timing_rows = []
    cache_timing_rows_by_hash = {}
    for path in metric_paths:
        manifest_path = path.parent / "fit.manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Fit manifest is missing beside {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_hash:
            raise ValueError(
                f"Fit manifest uses a stale or different configuration: {manifest_path}"
            )
        validation_path = Path(manifest["validation_data"])
        if (
            not validation_path.is_file()
            or manifest.get("validation_data_sha256") != sha256_file(validation_path)
        ):
            raise ValueError(f"Validation-data checksum mismatch: {validation_path}")
        if manifest.get("dnn_model_role") != "best-validation-restored":
            raise ValueError(f"Fit did not report the selected DNN model: {manifest_path}")
        if manifest.get("evaluation_metric") != config.evaluation["metric"]:
            raise ValueError(f"Fit used the wrong primary metric: {manifest_path}")
        dnn_training_backend = manifest.get("dnn_training_backend")
        if dnn_training_backend == DNN_TRAINING_BACKEND_EAGER:
            expected_dnn_revision = DNN_ALGORITHM_REVISION
        elif dnn_training_backend == DNN_TRAINING_BACKEND_OPTIMIZED:
            expected_dnn_revision = DNN_OPTIMIZED_ALGORITHM_REVISION
        else:
            raise ValueError(
                f"Fit uses an unsupported DNN training backend: {manifest_path}"
            )
        if (
            manifest.get("pdr_hyperparameters", {}).get("algorithm_revision")
            != PDR_ALGORITHM_REVISION
            or manifest.get("dnn_diagnostics", {}).get("algorithm_revision")
            != expected_dnn_revision
        ):
            raise ValueError(f"Fit uses a stale algorithm revision: {manifest_path}")
        for name_field, hash_field in (
            ("pdr_convergence", "pdr_convergence_sha256"),
            ("dnn_validation_history", "dnn_validation_history_sha256"),
            ("dnn_training_history", "dnn_training_history_sha256"),
        ):
            artifact = path.parent / manifest[name_field]
            if not artifact.is_file() or sha256_file(artifact) != manifest[hash_field]:
                raise ValueError(f"Diagnostic checksum mismatch: {artifact}")
        if config.is_rqmc_primary:
            artifact = path.parent / manifest["rqmc_convergence"]
            if (
                not artifact.is_file()
                or sha256_file(artifact) != manifest["rqmc_convergence_sha256"]
            ):
                raise ValueError(f"RQMC diagnostic checksum mismatch: {artifact}")
        if manifest.get("metrics_sha256") != sha256_file(path):
            raise ValueError(f"Metric checksum mismatch: {path}")
        stages = manifest.get("stage_timings", {})
        if not stages:
            raise ValueError(f"Fit manifest lacks stage timings: {manifest_path}")
        cache_record = manifest.get("cs_design_cache")
        timing_rows.append({
            "trial": int(manifest["trial"]),
            "train_size": int(manifest["sample_count"]),
            "cs_design_cache_used": cache_record is not None,
            "cs_design_source": manifest.get("pdr_diagnostics", {}).get("design_source", ""),
            **{name: float(value) for name, value in stages.items()},
        })
        if cache_record is not None and cache_record["data_sha256"] not in cache_timing_rows_by_hash:
            cache_data = REPOSITORY / cache_record["data"]
            cache_manifest_path = cache_data.with_suffix(".manifest.json")
            if not cache_manifest_path.is_file():
                raise FileNotFoundError(f"CS design-cache manifest is missing: {cache_manifest_path}")
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            if (
                cache_manifest.get("data_sha256") != cache_record["data_sha256"]
                or sha256_file(cache_data) != cache_record["data_sha256"]
            ):
                raise ValueError(f"CS design-cache identity mismatch: {cache_manifest_path}")
            cache_timing_rows_by_hash[cache_record["data_sha256"]] = {
                "trial": int(cache_manifest["trial"]),
                "sample_count": int(cache_manifest["sample_count"]),
                "order": int(cache_manifest["order"]),
                "basis_size": int(cache_manifest["basis_size"]),
                "cache_bytes": cache_data.stat().st_size,
                "offline_total_wall_seconds": float(cache_manifest["wall_seconds"]),
                **{
                    f"offline_{name}": float(value)
                    for name, value in cache_manifest.get("stage_timings", {}).items()
                },
            }
        rows.extend(read_metrics(path))
    rows = validate_metrics(rows, config)

    metrics_output = arguments.metrics_output.resolve()
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = metrics_output.with_name(f".{metrics_output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trial", "train_size", "output", "method", "relative_l2"))
        for row in rows:
            writer.writerow(
                (row.trial, row.train_size, row.output, row.method, f"{row.relative_l2:.17g}")
            )
    temporary.replace(metrics_output)

    timings_output = (
        arguments.timings_output.resolve()
        if arguments.timings_output
        else metrics_output.with_name("timings_all.csv")
    )
    cache_timings_output = (
        arguments.cache_timings_output.resolve()
        if arguments.cache_timings_output
        else metrics_output.with_name("cs_design_cache_timings.csv")
    )
    _write_timing_table(timings_output, timing_rows)
    _write_timing_table(
        cache_timings_output, list(cache_timing_rows_by_hash.values())
    )

    summary = summarize_metrics(rows, config)
    figure = (
        arguments.figure_output.resolve()
        if arguments.figure_output
        else (REPOSITORY / config.plot["output"]).resolve()
    )
    summary_path = figure.with_name(f"{figure.stem}_summary.csv")
    manifest_path = figure.with_name(f"{figure.stem}_manifest.json")
    write_summary(summary_path, summary, config)
    plot_learning_curves(summary, config, figure)
    write_manifest(
        manifest_path,
        {
            "artifact": "manuscript-figure-1",
            "experiment_id": config.experiment["id"],
            "config": str(config.path.relative_to(REPOSITORY)),
            "config_sha256": sha256_file(config.path),
            "random_seeds": {"base_seed": int(config.experiment["base_seed"])},
            "parameter_names": list(config.parameter_box.names),
            "parameter_lower": list(config.parameter_box.lower),
            "parameter_upper": list(config.parameter_box.upper),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": str(config.parameter_box_path.relative_to(REPOSITORY)),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "raw_metrics": str(metrics_output),
            "raw_metrics_sha256": sha256_file(metrics_output),
            "fit_stage_timings": str(timings_output),
            "fit_stage_timings_sha256": sha256_file(timings_output),
            "cs_offline_cache_timings": str(cache_timings_output),
            "cs_offline_cache_timings_sha256": sha256_file(cache_timings_output),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "figure": str(figure),
            "figure_sha256": sha256_file(figure),
            "standard_deviation_ddof": int(config.plot["standard_deviation_ddof"]),
        },
        repository=REPOSITORY,
    )
    print(metrics_output)
    print(timings_output)
    print(cache_timings_output)
    print(figure)
    print(manifest_path)


def main() -> int:
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    if arguments.command == "fit":
        _run_fit(arguments, config)
    elif arguments.command in ("suite", "cs-timing-suite"):
        _run_suite(arguments, config)
    else:
        _run_aggregate(arguments, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
