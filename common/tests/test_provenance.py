import json
from pathlib import Path

from cloud_column_reproduction.provenance import (
    scientific_source_snapshot,
    validate_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_source_snapshot_is_deterministic() -> None:
    first = scientific_source_snapshot(ROOT)
    second = scientific_source_snapshot(ROOT)
    assert first == second
    assert first["file_count"] > 0
    assert len(first["sha256"]) == 64


def test_manifest_has_common_metadata(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "derecho.yml").write_text(
        "name: cloud-column-study-derecho\n", encoding="utf-8"
    )
    destination = tmp_path / "artifact.manifest.json"
    write_manifest(destination, {"artifact": "test-artifact"}, repository=tmp_path)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    validate_manifest(payload)
    assert payload["command"]["argv"]
    assert "numpy" in payload["environment"]["packages"]
    specification = tmp_path / payload["environment"]["specification"]
    assert specification.is_file()
    assert "python_distributions" not in payload["environment"]
    assert "prefix" not in payload["environment"]
    assert payload["scientific_source_snapshot"]["file_count"] > 0
