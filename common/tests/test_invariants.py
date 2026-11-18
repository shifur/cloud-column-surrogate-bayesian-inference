import json
from pathlib import Path

import pytest

from cloud_column_reproduction.figure1.config import load_config
from cloud_column_reproduction.invariants import audit_manifests
from cloud_column_reproduction.provenance import sha256_file, write_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_audit_rejects_parameter_box_drift(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "figure1.toml")
    valid = {
        "artifact": "figure1-primary-evaluation-inputs",
        "config": "configs/figure1.toml",
        "config_sha256": sha256_file(ROOT / "configs" / "figure1.toml"),
        "parameter_box_sha256": config.parameter_box.fingerprint,
        "parameter_box_config": "configs/parameter_box.toml",
        "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
        "metric": "equal-weight-relative-l2",
        "random_seeds": {"primary_evaluation": 1},
        "workers": 1,
        "wall_seconds": 0.0,
    }
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, valid, repository=ROOT)
    assert audit_manifests([manifest], config.parameter_box, config.parameter_box_path) == [
        manifest.resolve()
    ]

    valid["parameter_box_sha256"] = "wrong"
    existing = json.loads(manifest.read_text(encoding="utf-8"))
    existing["parameter_box_sha256"] = "wrong"
    manifest.write_text(json.dumps(existing), encoding="utf-8")
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        audit_manifests([manifest], config.parameter_box, config.parameter_box_path)


def test_manifest_audit_can_consume_data_after_producing_config_evolves(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs" / "figure1.toml")
    payload = {
        "artifact": "figure1-primary-evaluation-inputs",
        "config": "configs/figure1.toml",
        "config_sha256": "0" * 64,
        "parameter_box_sha256": config.parameter_box.fingerprint,
        "parameter_box_config": "configs/parameter_box.toml",
        "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
        "metric": "equal-weight-relative-l2",
        "random_seeds": {"primary_evaluation": 1},
        "workers": 1,
        "wall_seconds": 0.0,
    }
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, payload, repository=ROOT)
    with pytest.raises(ValueError, match="experiment config checksum mismatch"):
        audit_manifests([manifest], config.parameter_box, config.parameter_box_path)
    assert audit_manifests(
        [manifest],
        config.parameter_box,
        config.parameter_box_path,
        verify_recorded_config_file=False,
    ) == [manifest.resolve()]



def test_manifest_audit_allows_only_explicit_legacy_schema_v1_inputs(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs" / "figure1.toml")
    payload = {
        "artifact": "figure1-training-pool",
        "config": "configs/figure1.data-generation-legacy.toml",
        "config_sha256": "1" * 64,
        "parameter_box_sha256": config.parameter_box.fingerprint,
        "parameter_box_config": "configs/parameter_box.toml",
        "parameter_box_config_sha256": sha256_file(config.parameter_box_path),
        "random_seeds": {"training_pool": 1},
        "workers": 16,
        "wall_seconds": 1.0,
    }
    manifest = tmp_path / "legacy.manifest.json"
    write_manifest(manifest, payload, repository=ROOT)
    legacy = json.loads(manifest.read_text(encoding="utf-8"))
    legacy["manifest_schema_version"] = 1
    legacy["environment"] = {
        "path": "environment/snapshots/legacy.json",
        "sha256": "2" * 64,
        "python": {"version": "3.11"},
        "packages": {"numpy": "2.4.3"},
        "runtime_controls": {},
    }
    manifest.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported run-manifest schema version"):
        audit_manifests(
            [manifest], config.parameter_box, config.parameter_box_path,
            verify_recorded_config_file=False,
        )
    assert audit_manifests(
        [manifest], config.parameter_box, config.parameter_box_path,
        verify_recorded_config_file=False,
        allow_legacy_schema_v1=True,
    ) == [manifest.resolve()]

    legacy["parameter_box_sha256"] = "wrong"
    manifest.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        audit_manifests(
            [manifest], config.parameter_box, config.parameter_box_path,
            verify_recorded_config_file=False,
            allow_legacy_schema_v1=True,
        )
