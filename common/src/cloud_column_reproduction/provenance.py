"""Dependency-free, validated provenance records for generated artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable
from uuid import uuid4


MANIFEST_SCHEMA_VERSION = 2
ENVIRONMENT_SPECIFICATION = Path("environment") / "derecho.yml"
SCIENTIFIC_SOURCE_PATHS = (
    "configs",
    "pyproject.toml",
    "scripts",
    "src",
    "vendor/sccm",
)
PACKAGE_DISTRIBUTIONS = (
    "cloud-column-study-optimized",
    "cupy-cuda12x",
    "emcee",
    "arviz",
    "h5netcdf",
    "h5py",
    "matplotlib",
    "numpy",
    "qmcpy",
    "scikit-learn",
    "scipy",
    "Tasmanian",
    "torch",
    "xarray",
)
SOURCE_EXCLUDED_SUFFIXES = {
    ".a",
    ".mod",
    ".o",
    ".pyc",
}
SOURCE_EXCLUDED_NAMES = {
    ".DS_Store",
    "cloud_column_model.x",  # captured separately as the executed binary
}
SOURCE_EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__MACOSX",
    "__pycache__",
    "snapshots",  # content-addressed environment records are provenance outputs
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def local_git_revision(repository: str | Path) -> str | None:
    return _run_text(["git", "rev-parse", "HEAD"], Path(repository).resolve())


def git_state(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    revision = local_git_revision(root)
    branch = _run_text(["git", "branch", "--show-current"], root)
    status = _run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    entries = [] if not status else status.splitlines()
    status_digest = hashlib.sha256((status or "").encode("utf-8")).hexdigest()
    return {
        "revision": revision,
        "branch": branch,
        "has_commits": revision is not None,
        "dirty": bool(entries),
        "status_entry_count": len(entries),
        "status_porcelain_sha256": status_digest,
    }


def _source_files(repository: Path, roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for relative in roots:
        candidate = repository / relative
        if candidate.is_file():
            candidates = [candidate]
        elif candidate.is_dir():
            candidates = [path for path in candidate.rglob("*") if path.is_file()]
        else:
            continue
        for path in candidates:
            rel = path.relative_to(repository)
            if any(part in SOURCE_EXCLUDED_PARTS for part in rel.parts):
                continue
            if path.name in SOURCE_EXCLUDED_NAMES or path.suffix in SOURCE_EXCLUDED_SUFFIXES:
                continue
            files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(repository).as_posix())


def scientific_source_snapshot(
    repository: str | Path,
    roots: Iterable[str] = SCIENTIFIC_SOURCE_PATHS,
) -> dict[str, Any]:
    """Fingerprint run-defining source files even when Git has no usable revision."""
    repository = Path(repository).resolve()
    normalized_roots = tuple(roots)
    digest = hashlib.sha256()
    files = _source_files(repository, normalized_roots)
    for path in files:
        relative = path.relative_to(repository).as_posix()
        file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(relative-path\\0file-sha256\\n)",
        "roots": list(normalized_roots),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _first_line(command: list[str]) -> str | None:
    output = _run_text(command)
    return None if output is None else output.splitlines()[0]


def _last_line(command: list[str]) -> str | None:
    output = _run_text(command)
    return None if output is None else output.splitlines()[-1]


def environment_record(
    repository: str | Path,
    specification_path: str | Path = ENVIRONMENT_SPECIFICATION,
) -> dict[str, Any]:
    """Return compact, non-sensitive runtime metadata for a completed run."""
    repository = Path(repository).resolve()
    specification_path = Path(specification_path)
    specification = repository / specification_path
    if not specification.is_file():
        raise FileNotFoundError(
            f"Required environment specification is missing: {specification}"
        )
    runtime_controls = {
        name: os.environ.get(name)
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "PYTHONHASHSEED",
        )
    }
    return {
        "specification": specification_path.as_posix(),
        "specification_sha256": sha256_file(specification),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "runtime_controls": runtime_controls,
        "tools": {
            "gfortran": _first_line(["gfortran", "--version"]),
            "make": _first_line(["make", "--version"]),
            "cuda_compiler": _last_line(["nvcc", "--version"]),
            "gpu": _run_text(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "manifest_schema_version",
        "run_id",
        "created_utc",
        "status",
        "command",
        "git",
        "scientific_source_snapshot",
        "environment",
        "artifact",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported run-manifest schema version")
    if manifest["status"] != "complete":
        raise ValueError("Only complete artifacts may receive a final manifest")
    command = manifest["command"]
    if not isinstance(command, dict) or not command.get("argv") or not command.get("cwd"):
        raise ValueError("Manifest command must contain nonempty argv and cwd")
    snapshot = manifest["scientific_source_snapshot"]
    if not isinstance(snapshot, dict) or not snapshot.get("sha256"):
        raise ValueError("Manifest scientific-source snapshot is incomplete")
    if not isinstance(manifest["environment"], dict):
        raise ValueError("Manifest environment metadata must be an object")
    required_environment = {
        "specification",
        "specification_sha256",
        "python",
        "conda_environment",
        "platform",
        "packages",
        "runtime_controls",
        "tools",
    }
    missing_environment = sorted(required_environment.difference(manifest["environment"]))
    if missing_environment:
        raise ValueError(
            "Manifest environment metadata is incomplete: "
            + ", ".join(missing_environment)
        )
    artifact = str(manifest["artifact"])
    if artifact.startswith("figure1-") or artifact == "manuscript-figure-1":
        required_experiment = {
            "config",
            "config_sha256",
            "parameter_box_sha256",
            "parameter_box_config",
            "parameter_box_config_sha256",
            "random_seeds",
        }
        missing_experiment = sorted(required_experiment.difference(manifest))
        if missing_experiment:
            raise ValueError(
                "Figure 1 manifest is missing required fields: "
                + ", ".join(missing_experiment)
            )
    if artifact.startswith("figure1-"):
        required_run = {"workers", "wall_seconds"}
        missing_run = sorted(required_run.difference(manifest))
        if missing_run:
            raise ValueError(
                "Figure 1 run manifest is missing required fields: "
                + ", ".join(missing_run)
            )
    if manifest.get("sccm_executable") is not None:
        required_sccm = {
            "sccm_executable_sha256",
            "sccm_namelist_sha256",
            "sccm_rendered_namelist_sha256",
            "sccm_radiation_table_sha256",
        }
        missing_sccm = sorted(
            field for field in required_sccm if not manifest.get(field)
        )
        if missing_sccm:
            raise ValueError(
                "SCCM manifest is missing required fields: " + ", ".join(missing_sccm)
            )
    if (
        manifest.get("artifact") == "mcmc-surrogate-data"
        and manifest.get("artifact_role") == "production"
    ):
        required_authoritative = {
            "rng_algorithm",
            "sampling_policy",
            "sccm_success_count",
            "sccm_failure_count",
            "non_finite_count",
            "approved_runtime",
        }
        missing_authoritative = sorted(required_authoritative.difference(manifest))
        if missing_authoritative:
            raise ValueError(
                "Authoritative surrogate-data manifest is missing required fields: "
                + ", ".join(missing_authoritative)
            )
        approved_runtime = manifest["approved_runtime"]
        required_runtime = {
            "status",
            "environment_name",
            "interpreter_prefix",
            "interpreter_executable",
            "specification_sha256",
            "environment_fingerprint_sha256",
        }
        if not isinstance(approved_runtime, dict):
            raise ValueError("Authoritative approved_runtime metadata must be an object")
        missing_runtime = sorted(required_runtime.difference(approved_runtime))
        if missing_runtime:
            raise ValueError(
                "Authoritative approved_runtime metadata is incomplete: "
                + ", ".join(missing_runtime)
            )


def write_manifest(
    path: str | Path,
    payload: dict[str, Any],
    *,
    repository: str | Path,
    environment_specification: str | Path = ENVIRONMENT_SPECIFICATION,
) -> Path:
    """Atomically write a validated, self-identifying final artifact manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository = Path(repository).resolve()
    environment = environment_record(repository, environment_specification)
    envelope = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "command": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "git": git_state(repository),
        "scientific_source_snapshot": scientific_source_snapshot(repository),
        "environment": environment,
        **payload,
    }
    validate_manifest(envelope)
    serialized = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination
