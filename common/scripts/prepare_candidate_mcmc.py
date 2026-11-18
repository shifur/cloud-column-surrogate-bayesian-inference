#!/usr/bin/env python3
"""Bind all 20 paired Figure 1 m=5000 fits to production MCMC configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_column_reproduction.figure1.pdr_gpu import active_polynomial_support
from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.provenance import sha256_file


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _relative(path: Path) -> str:
    # Preserve logical repository paths when data/, outputs/, or figures/
    # are symlinks to Scratch.
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        pass

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


def _support(model: Path) -> tuple[int, str]:
    with np.load(model, allow_pickle=False) as archive:
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
    return len(active_indices), digest.hexdigest()


def _replace_section(text: str, name: str, replacement: str) -> str:
    pattern = rf"(?ms)^\[{re.escape(name)}\]\n.*?(?=^\[|\Z)"
    rendered, count = re.subn(pattern, replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace [{name}] in the selected-model config")
    return rendered


def _render_trial(
    base: Path, figure1: Path, trial: int, output: Path
) -> dict[str, object]:
    directory = figure1 / f"trial_{trial:03d}" / "n05000"
    manifest_path = directory / "fit.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing paired m=5000 fit: {manifest_path}")
    fit = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(fit["trial"]) != trial or int(fit["sample_count"]) != 5000:
        raise ValueError(f"Figure 1 fit identity mismatch: {manifest_path}")
    dnn_model = directory / str(fit["dnn_model"])
    cs_model = directory / str(fit["pdr_model"])
    training = Path(fit["training_data"]).resolve()
    for artifact in (dnn_model, cs_model, training):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    active_size, active_hash = _support(cs_model)

    rendered = base.read_text(encoding="utf-8")
    rendered = re.sub(
        r'^status = ".*"$',
        'status = "production-approved-candidate-variability-v1"',
        rendered,
        count=1,
        flags=re.M,
    )
    rendered = re.sub(
        r'^experiment_id = ".*"$',
        f'experiment_id = "cloud_column_candidate_trial_{trial:03d}"',
        rendered,
        count=1,
        flags=re.M,
    )
    rendered = re.sub(
        r'^description = ".*"$',
        f'description = "Paired DNN and CS posterior variability for Figure 1 trial {trial:03d}"',
        rendered,
        count=1,
        flags=re.M,
    )
    rendered = _replace_section(
        rendered,
        "study",
        f"""[study]
role = "production-study"
final_manuscript_evidence = true
candidate_policy = "all paired Figure 1 m=5000 fits; no post-hoc selection"
candidate_trial = {trial}
candidate_score_outputs = ["PCP", "LWP", "IWP", "OLR", "OSR"]
optimized_gpu_30k_300k = true

[execution_scope]
likelihood_cases = ["full_except_acc"]
methods = ["dnn", "pdr"]""",
    )
    rendered = _replace_section(
        rendered,
        "surrogates.dnn",
        f'''[surrogates.dnn]
model = "{_relative(dnn_model)}"
model_sha256 = "{sha256_file(dnn_model)}"
fit_manifest = "{_relative(manifest_path)}"
training_data = "{_relative(training)}"
training_data_sha256 = "{fit['training_data_sha256']}"''',
    )
    rendered = _replace_section(
        rendered,
        "surrogates.pdr",
        f'''[surrogates.pdr]
model = "{_relative(cs_model)}"
model_sha256 = "{sha256_file(cs_model)}"
fit_manifest = "{_relative(manifest_path)}"
training_data = "{_relative(training)}"
training_data_sha256 = "{fit['training_data_sha256']}"
q = 23
basis_size = 7569
basis_fingerprint_sha256 = "a22d9594d73d3b779dc08d87dc34470ef8d2a5fcb2c60e786acc8429c3eed76f"''',
    )
    artifacts = re.search(
        r"(?ms)^\[artifacts\]\n.*?(?=^\[artifacts\.storage_estimate\])", rendered
    )
    if artifacts is None:
        raise ValueError("Selected-model config has no [artifacts] section")
    artifact_text = artifacts.group(0)
    changes = {
        r'^root = ".*"$': f'root = "outputs/mcmc/candidate_variability_v1/trial_{trial:03d}/chains"',
        r"^expected_likelihood_case_count = .*?$": "expected_likelihood_case_count = 1",
        r"^expected_method_count = .*?$": "expected_method_count = 2",
        r"^expected_ensemble_counts_by_method = .*?$": "expected_ensemble_counts_by_method = { sccm = 0, dnn = 2, pdr = 2 }",
        r"^expected_chain_file_count = .*?$": "expected_chain_file_count = 4",
    }
    for pattern, value in changes.items():
        artifact_text, count = re.subn(
            pattern, value, artifact_text, count=1, flags=re.M
        )
        if count != 1:
            raise ValueError(f"Missing artifact contract field: {pattern}")
    rendered = (
        rendered[: artifacts.start()] + artifact_text + rendered[artifacts.end() :]
    )
    rendered = _replace_section(
        rendered,
        "workload",
        '''[workload]
likelihood_case_count = 1
independent_ensemble_counts_by_method = { sccm = 0, dnn = 2, pdr = 2 }
steps_per_ensemble_including_burn_in = 330000
sccm_run_count = 0
sccm_transition_proposals_total = 0
sccm_initial_state_evaluations_total = 0
sccm_posterior_evaluations_upper_bound = 0
sccm_forward_evaluations_note = "not part of the candidate-variability execution scope"
sccm_throughput_evaluations_per_second = 1.0
sccm_estimated_wall_seconds = 0.0
sccm_estimated_wall_hours = 0.0
sccm_estimated_wall_days = 0.0
estimate_scope = "DNN and CS only"''',
    )
    rendered = re.sub(
        r"^active_basis_size = .*$",
        f"active_basis_size = {active_size}",
        rendered,
        count=1,
        flags=re.M,
    )
    rendered = re.sub(
        r'^active_support_sha256 = ".*"$',
        f'active_support_sha256 = "{active_hash}"',
        rendered,
        count=1,
        flags=re.M,
    )
    _atomic(output, rendered)
    load_config(output, validate_registered_artifacts=True)
    return {
        "trial": trial,
        "config": _relative(output),
        "config_sha256": sha256_file(output),
        "dnn_model": _relative(dnn_model),
        "dnn_model_sha256": sha256_file(dnn_model),
        "cs_model": _relative(cs_model),
        "cs_model_sha256": sha256_file(cs_model),
        "cs_active_basis_size": active_size,
        "cs_active_support_sha256": active_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure1-output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/mcmc.toml")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "configs")
    arguments = parser.parse_args()
    base = arguments.base_config.resolve()
    load_config(base, validate_registered_artifacts=True).require_production_ready()
    records = [
        _render_trial(
            base,
            arguments.figure1_output.resolve(),
            trial,
            arguments.output_directory.resolve()
            / f"mcmc_candidate_trial_{trial:03d}.toml",
        )
        for trial in range(20)
    ]
    registry = arguments.output_directory.resolve() / "mcmc_candidate_registry.json"
    _atomic(
        registry,
        json.dumps(
            {
                "study": "paired-m5000-candidate-posterior-variability",
                "likelihood_case": "full_except_acc",
                "methods": ["dnn", "pdr"],
                "model_count": 40,
                "trial_count": 20,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        json.dumps(
            {"registry": _relative(registry), "trial_count": 20, "model_count": 40},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
