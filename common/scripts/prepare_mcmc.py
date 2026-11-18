#!/usr/bin/env python3
"""Select Figure 1 candidates and materialize the production MCMC contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_column_reproduction.figure1.pdr_gpu import active_polynomial_support
from cloud_column_reproduction.mcmc.config import OUTPUT_ORDER, OUTPUT_UNITS, load_config
from cloud_column_reproduction.provenance import sha256_file, write_manifest
from cloud_column_reproduction.sccm import runner_from_repository

ACTIVE_OUTPUTS = {"PCP", "LWP", "IWP", "OLR", "OSR"}


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _relative(path: Path) -> str:
    # Preserve logical repository paths when data/, outputs/, or figures/
    # are symlinks to Scratch.
    try:
        return str(path.absolute().relative_to(ROOT))
    except ValueError:
        pass

    absolute = path.absolute()
    for logical_name in ("data", "outputs", "figures"):
        physical_root = (ROOT / logical_name).resolve()
        try:
            suffix = absolute.relative_to(physical_root)
        except ValueError:
            continue
        return str(Path(logical_name) / suffix)

    raise ValueError(
        f"{absolute} is neither inside the repository nor one of its "
        "Scratch-backed artifact directories"
    )


def _score(path: Path, method: str) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = {method, "CS_HC23" if method == "CS" else method}
    values = [
        float(row["relative_l2"])
        for row in rows
        if row["method"] in labels and row["output"] in ACTIVE_OUTPUTS
    ]
    if len(values) != 5 or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{path} does not contain five finite positive {method} errors")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def select_models(figure1_root: Path, template: Path, output: Path) -> dict[str, object]:
    candidates: dict[str, list[tuple[float, int, Path, dict[str, object]]]] = {
        "DNN": [], "CS": []
    }
    for trial in range(20):
        directory = figure1_root / f"trial_{trial:03d}" / "n05000"
        manifest_path = directory / "fit.manifest.json"
        metrics_path = directory / "metrics.csv"
        if not manifest_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete Figure 1 m=5000 fit: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["trial"]) != trial or int(manifest["sample_count"]) != 5000:
            raise ValueError(f"Figure 1 manifest identity mismatch: {manifest_path}")
        for method in candidates:
            candidates[method].append((_score(metrics_path, method), trial, directory, manifest))
    selected = {method: min(records, key=lambda record: (record[0], record[1])) for method, records in candidates.items()}
    dnn_score, dnn_trial, dnn_dir, dnn_manifest = selected["DNN"]
    cs_score, cs_trial, cs_dir, cs_manifest = selected["CS"]
    dnn_model = dnn_dir / str(dnn_manifest["dnn_model"])
    cs_model = cs_dir / str(cs_manifest["pdr_model"])
    with np.load(cs_model, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        indices = np.asarray(archive["indices"], dtype=np.int64)
    active_coefficients, active_indices, active_mask = active_polynomial_support(
        coefficients, indices, prune_exact_zeros=True
    )
    digest = hashlib.sha256()
    for array in (active_mask, active_indices, active_coefficients):
        values = np.ascontiguousarray(array)
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    substitutions = {
        "SET_DNN_MODEL_PATH": _relative(dnn_model),
        "SET_DNN_MODEL_SHA256": sha256_file(dnn_model),
        "SET_DNN_FIT_MANIFEST_PATH": _relative(dnn_dir / "fit.manifest.json"),
        "SET_DNN_TRAINING_DATA_PATH": _relative(Path(dnn_manifest["training_data"])),
        "SET_DNN_TRAINING_DATA_SHA256": str(dnn_manifest["training_data_sha256"]),
        "SET_CS_MODEL_PATH": _relative(cs_model),
        "SET_CS_MODEL_SHA256": sha256_file(cs_model),
        "SET_CS_FIT_MANIFEST_PATH": _relative(cs_dir / "fit.manifest.json"),
        "SET_CS_TRAINING_DATA_PATH": _relative(Path(cs_manifest["training_data"])),
        "SET_CS_TRAINING_DATA_SHA256": str(cs_manifest["training_data_sha256"]),
        '"SET_CS_ACTIVE_BASIS_SIZE"': str(len(active_indices)),
        "SET_CS_ACTIVE_SUPPORT_SHA256": digest.hexdigest(),
    }
    rendered = template.read_text(encoding="utf-8")
    for old, new in substitutions.items():
        rendered = rendered.replace(old, new)
    if "SET_" in rendered.replace("SET_OBSERVATION_SHA256", ""):
        raise ValueError("Unresolved non-observation field remains in MCMC template")
    _atomic(output, rendered.encode("utf-8"))
    record = {
        "status": "selected-awaiting-observation",
        "selection_rule": "separate minimum five-output geometric-mean corrected RQMC relative L2 error",
        "sample_count": 5000,
        "trials_considered": 20,
        "active_outputs": sorted(ACTIVE_OUTPUTS),
        "dnn": {"trial": dnn_trial, "score": dnn_score, "model": _relative(dnn_model), "sha256": sha256_file(dnn_model), "training_data": _relative(Path(dnn_manifest["training_data"])), "training_data_sha256": str(dnn_manifest["training_data_sha256"])},
        "cs": {"trial": cs_trial, "score": cs_score, "model": _relative(cs_model), "sha256": sha256_file(cs_model), "training_data": _relative(Path(cs_manifest["training_data"])), "training_data_sha256": str(cs_manifest["training_data_sha256"]), "q": 23, "basis_size": 7569, "active_basis_size": len(active_indices), "active_support_sha256": digest.hexdigest()},
        "config": _relative(output),
        "config_sha256": sha256_file(output),
    }
    selection_path = output.with_suffix(".selection.json")
    _atomic(selection_path, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    return record


def materialize_observation(pending: Path, output: Path) -> dict[str, object]:
    config = load_config(pending, validate_registered_artifacts=False)
    runner = runner_from_repository(ROOT, parameter_box=config.parameter_box)
    values = runner.diagnostics_at_120_minutes(config.truth)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, values=values, names=np.asarray(OUTPUT_ORDER), units=np.asarray(OUTPUT_UNITS), truth=config.truth, time_minutes=np.asarray(120), parameter_names=np.asarray(config.parameter_box.names), parameter_lower=np.asarray(config.parameter_box.lower), parameter_upper=np.asarray(config.parameter_box.upper), parameter_box_fingerprint=np.asarray(config.parameter_box.fingerprint))
    artifact = ROOT / config.raw["observations"]["artifact"]
    _atomic(artifact, buffer.getvalue())
    manifest = ROOT / config.raw["observations"]["manifest"]
    write_manifest(
        manifest,
        {
            "artifact": "mcmc-synthetic-observation",
            "data": _relative(artifact),
            "data_sha256": sha256_file(artifact),
            "time_minutes": 120,
            "diagnostic_names": list(OUTPUT_ORDER),
            "diagnostic_units": list(OUTPUT_UNITS),
            "truth": [float(value) for value in config.truth],
            "parameter_names": list(config.parameter_box.names),
            "parameter_box_sha256": config.parameter_box.fingerprint,
            "parameter_box_config": _relative(config.parameter_box_path),
            "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
            "sccm_executable": _relative(runner.executable),
            "sccm_executable_sha256": sha256_file(runner.executable),
            "sccm_namelist_sha256": sha256_file(runner.namelist),
            "sccm_rendered_namelist_sha256": runner.rendered_namelist_sha256(),
            "sccm_radiation_table_sha256": sha256_file(runner.radiation_table),
        },
        repository=ROOT,
    )
    rendered = pending.read_text(encoding="utf-8").replace("SET_OBSERVATION_SHA256", sha256_file(artifact)).replace("production-template-unmaterialized", "production-approved-v1")
    if "SET_" in rendered:
        raise ValueError("MCMC contract still contains unresolved fields")
    _atomic(output, rendered.encode("utf-8"))
    load_config(output, validate_registered_artifacts=True)
    return {"status": "production-ready", "config": _relative(output), "config_sha256": sha256_file(output), "observation": _relative(artifact), "observation_sha256": sha256_file(artifact), "observation_manifest": _relative(manifest), "observation_manifest_sha256": sha256_file(manifest)}


def upgrade_runtime(source: Path, output: Path, chain_root: str) -> dict:
    """Preserve the scientific contract and artifacts; replace only execution policy."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if source.parent != output.parent:
        raise ValueError("Keep the upgraded config beside its source to preserve relative paths")
    old = load_config(source, validate_registered_artifacts=False)
    old.require_production_ready()
    if "extends" in old.raw:
        raise ValueError("Upgrade the fully materialized base config, not an inherited override")
    repository = source.parent.parent
    new_root = (repository / chain_root).resolve()
    if new_root == (repository / old.raw["artifacts"]["root"]).resolve() or new_root.exists():
        raise ValueError("The matched timing run requires a fresh chain root")
    rendered = source.read_text()
    template = (ROOT / "configs/mcmc.matched-fp32.template.toml").read_text()
    # Replace entire runtime tables, dropping obsolete float64 audit references.
    for name in ("runtime.dnn", "runtime.pdr", "runtime.execution"):
        pattern = rf"(?ms)^\[{re.escape(name)}\]\n.*?(?=^\[|\Z)"
        replacement = re.search(pattern, template).group()
        if name == "runtime.pdr":
            from cloud_column_reproduction.figure1.pdr_gpu import active_polynomial_support
            with np.load(repository / old.raw["surrogates"]["pdr"]["model"], allow_pickle=False) as archive:
                coefficients, indices, mask = active_polynomial_support(
                    archive["coefficients"], archive["indices"], prune_exact_zeros=True)
            digest = hashlib.sha256()
            for array in (mask, indices, coefficients):
                values = np.ascontiguousarray(array)
                digest.update(values.dtype.str.encode("ascii"))
                digest.update(str(values.shape).encode("ascii"))
                digest.update(values.tobytes(order="C"))
            replacement = replacement.replace('"SET_CS_ACTIVE_BASIS_SIZE"', str(len(indices)))
            replacement = replacement.replace("SET_CS_ACTIVE_SUPPORT_SHA256", digest.hexdigest())
        if re.search(pattern, rendered):
            rendered = re.sub(pattern, lambda match: replacement, rendered)
        else:
            rendered += "\n" + replacement
    pattern = r'(?ms)(^\[artifacts\]\n.*?^root\s*=\s*)"[^"]*"'
    rendered, count = re.subn(pattern, lambda m: m[1] + json.dumps(chain_root), rendered, count=1)
    if count != 1:
        raise ValueError("Expected exactly one artifact root")
    rendered = re.sub(r'(?m)^experiment_id\s*=.*$',
                      'experiment_id = "cloud_column_mcmc_matched_fp32_v2"', rendered)
    rendered = re.sub(r'(?m)^status\s*=.*$', 'status = "production-approved-matched-fp32-v2"', rendered)
    _atomic(output, rendered.encode())
    try:
        updated = load_config(output, validate_registered_artifacts=True)
        updated.require_production_ready()
    except BaseException:
        output.unlink()
        raise
    return {"status": "production-ready-matched-fp32-v2", "config": str(output),
            "config_sha256": sha256_file(output), "source_config_sha256": sha256_file(source),
            "chain_root": str(new_root), "frozen_models_and_observation": "unchanged"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select-models")
    select.add_argument("--figure1-output", type=Path, required=True)
    select.add_argument("--template", type=Path, default=ROOT / "configs/mcmc.template.toml")
    select.add_argument("--output", type=Path, default=ROOT / "configs/mcmc.pending.toml")
    observe = commands.add_parser("materialize-observation")
    observe.add_argument("--pending", type=Path, default=ROOT / "configs/mcmc.pending.toml")
    observe.add_argument("--output", type=Path, default=ROOT / "configs/mcmc.toml")
    upgrade = commands.add_parser("upgrade-runtime", help="Reuse frozen artifacts with matched float32 timing")
    upgrade.add_argument("--config", type=Path, default=ROOT / "configs/mcmc.toml")
    upgrade.add_argument("--output", type=Path, default=ROOT / "configs/mcmc.matched-fp32.toml")
    upgrade.add_argument("--chain-root", default="outputs/mcmc/matched_fp32_v2/chains")
    args = parser.parse_args()
    if args.command == "select-models":
        result = select_models(args.figure1_output.absolute(), args.template.resolve(), args.output.resolve())
    elif args.command == "upgrade-runtime":
        result = upgrade_runtime(args.config.resolve(), args.output.resolve(), args.chain_root)
    else:
        result = materialize_observation(args.pending.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
