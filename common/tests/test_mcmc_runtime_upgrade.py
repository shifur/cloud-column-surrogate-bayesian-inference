import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from cloud_column_reproduction.figure1.polynomials import hyperbolic_cross_indices
from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_upgrade_reuses_artifacts_changes_execution_and_requires_fresh_root(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_mcmc", ROOT / "scripts/prepare_mcmc.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copyfile(ROOT / "configs/parameter_box.toml", configs / "parameter_box.toml")
    model = tmp_path / "cs.npz"
    indices = hyperbolic_cross_indices(11, 23)
    coefficients = np.zeros((len(indices), 6))
    coefficients[0] = 1
    np.savez(model, coefficients=coefficients, indices=indices)
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"opaque artifact for config-binding test")
    rendered = (ROOT / "configs/mcmc.matched-fp32.template.toml").read_text()
    replacements = {
        "SET_DNN_MODEL_PATH": str(fixture), "SET_DNN_MODEL_SHA256": sha256_file(fixture),
        "SET_DNN_FIT_MANIFEST_PATH": str(fixture), "SET_DNN_TRAINING_DATA_PATH": str(fixture),
        "SET_DNN_TRAINING_DATA_SHA256": sha256_file(fixture),
        "SET_CS_MODEL_PATH": str(model), "SET_CS_MODEL_SHA256": sha256_file(model),
        "SET_CS_FIT_MANIFEST_PATH": str(fixture), "SET_CS_TRAINING_DATA_PATH": str(fixture),
        "SET_CS_TRAINING_DATA_SHA256": sha256_file(fixture),
        '"SET_CS_ACTIVE_BASIS_SIZE"': "1", "SET_CS_ACTIVE_SUPPORT_SHA256": "a" * 64,
        "SET_OBSERVATION_SHA256": sha256_file(fixture),
        'artifact = "data/mcmc/production/synthetic_observation_t120.npz"': f'artifact = "{fixture}"',
        'manifest = "data/mcmc/production/synthetic_observation_t120.manifest.json"': f'manifest = "{tmp_path / "observation.json"}"',
        "production-template-unmaterialized": "production-approved-v1",
        "outputs/mcmc/matched_fp32_v2/chains": "outputs/old/chains",
        "cupy-cached-sparse-float32-cuda": "cupy-cached-sparse-cuda",
        "dnn_ensemble_concurrency = 2": "dnn_ensemble_concurrency = 1",
        "one-persistent-compiled-runtime-paired-ensemble-microbatches": "one-persistent-compiled-runtime-sequential-ensembles",
    }
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    rendered = rendered.replace('backend = "cupy-cached-sparse-cuda"\ndtype = "float32-model/float64-output"',
                                'backend = "cupy-cached-sparse-cuda"\ndtype = "float64"')
    source = configs / "old.toml"
    source.write_text(rendered)
    original = load_config(source, validate_registered_artifacts=False)
    manifest = {
        "manifest_schema_version": 2, "run_id": "test", "created_utc": "test",
        "status": "complete", "command": {"argv": ["test"], "cwd": str(tmp_path)},
        "git": {}, "scientific_source_snapshot": {"sha256": "a" * 64},
        "environment": dict.fromkeys(("specification", "specification_sha256", "python",
            "conda_environment", "platform", "packages", "runtime_controls", "tools")),
        "artifact": "mcmc-synthetic-observation", "data": str(fixture),
        "data_sha256": sha256_file(fixture), "parameter_box_sha256": original.parameter_box.fingerprint,
    }
    (tmp_path / "observation.json").write_text(json.dumps(manifest))
    destination = configs / "new.toml"
    old_bytes = source.read_bytes()
    script.upgrade_runtime(source, destination, "outputs/new/chains")
    upgraded = load_config(destination)
    assert source.read_bytes() == old_bytes
    for field in ("surrogates", "observations", "sampler", "likelihood_cases", "parameters"):
        assert upgraded.raw[field] == original.raw[field]
    assert upgraded.raw["runtime"]["dnn"]["dtype"] == upgraded.raw["runtime"]["pdr"]["dtype"]
    assert upgraded.raw["runtime"]["execution"]["dnn_ensemble_concurrency"] == 2
    assert upgraded.raw["runtime"]["execution"]["pdr_ensemble_concurrency"] == 2
    with pytest.raises(FileExistsError):
        script.upgrade_runtime(source, destination, "outputs/another/chains")
    (tmp_path / "outputs/existing").mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh"):
        script.upgrade_runtime(source, configs / "another.toml", "outputs/existing")
