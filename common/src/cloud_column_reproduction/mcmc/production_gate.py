"""Fail-closed storage and restart checks for controlled MCMC units."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from cloud_column_reproduction.provenance import sha256_file
from cloud_column_reproduction.mcmc.resumable import CHECKPOINT_SCHEMA_VERSION


def estimated_production_storage_bytes(
    *, transitions: int = 400_000, walkers: int = 32,
    likelihood_cases: int = 2, ensembles_per_method: int = 2,
) -> dict[str, int]:
    """Conservative uncompressed estimate for the matched twelve-chain design."""
    chain_count = likelihood_cases * 3 * ensembles_per_method
    core_per_chain = 1_459_200_000
    core_all = chain_count * core_per_chain
    sccm_cache_per_chain = transitions * walkers * 6 * 8
    sccm_chain_count = likelihood_cases * ensembles_per_method
    sccm_cache_all = sccm_chain_count * sccm_cache_per_chain
    total = core_all + sccm_cache_all
    return {"chain_count": chain_count, "core_per_chain": core_per_chain,
            "core_all_chains": core_all, "sccm_chain_count": sccm_chain_count,
            "sccm_cache_per_chain": sccm_cache_per_chain,
            "sccm_cache_all_chains": sccm_cache_all,
            "total_uncompressed": total,
            "required_with_25_percent_headroom": int(total * 1.25)}


def storage_preflight(output_root: str | Path, *, available_bytes: int | None = None) -> dict[str, Any]:
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    estimate = estimated_production_storage_bytes()
    available = shutil.disk_usage(root).free if available_bytes is None else int(available_bytes)
    required = estimate["required_with_25_percent_headroom"]
    if available < required:
        raise RuntimeError(f"Insufficient MCMC storage: require {required} bytes with headroom, found {available}")
    return {**estimate, "available_bytes": available, "status": "passed", "headroom_fraction": 0.25}


def restart_integrity_preflight(
    chain_path: str | Path, manifest_path: str | Path, *, expected_bindings: dict[str, str]
) -> dict[str, str]:
    """Allow new work or checksum-valid completed skip; never resume a partial chain."""
    chain = Path(chain_path).resolve(); manifest = Path(manifest_path).resolve()
    if not chain.exists() and not manifest.exists(): return {"status": "ready-new-unit"}
    if chain.exists() != manifest.exists():
        raise RuntimeError("Incomplete MCMC unit found; restart from zero after explicit disposition")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("status") != "complete" or record.get("chain_sha256") != sha256_file(chain):
        raise RuntimeError("Completed MCMC unit failed checksum integrity; do not resume")
    for key, expected in expected_bindings.items():
        if record.get(key) != expected:
            raise RuntimeError(f"Completed MCMC unit binding mismatch: {key}")
    return {"status": "skip-checksum-valid-complete-unit", "chain_sha256": record["chain_sha256"]}


def resumable_unit_preflight(
    unit_directory: str | Path,
    *,
    expected_bindings: dict[str, str],
) -> dict[str, Any]:
    """Classify a production unit without deleting recoverable checkpoints."""
    root = Path(unit_directory).resolve()
    final_chain = root / "chain.nc"
    final_manifest = root / "chain.manifest.json"
    if final_chain.exists() or final_manifest.exists():
        result = restart_integrity_preflight(
            final_chain, final_manifest, expected_bindings=expected_bindings
        )
        return {**result, "unit_directory": str(root)}
    checkpoint = root / "checkpoint.json"
    if not checkpoint.exists():
        return {"status": "ready-new-unit", "unit_directory": str(root)}
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    if record.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("MCMC checkpoint schema is not supported")
    if record.get("status") not in ("running", "sampling-complete"):
        raise RuntimeError("MCMC checkpoint has an invalid status")
    for key, expected in expected_bindings.items():
        if record.get("bindings", {}).get(key) != expected:
            raise RuntimeError(f"MCMC checkpoint binding mismatch: {key}")
    burn = 0
    production = 0
    for block in record.get("blocks", []):
        phase = block.get("phase")
        if phase not in ("burn-in", "production"):
            raise RuntimeError("MCMC checkpoint contains an unknown block phase")
        expected_start = burn if phase == "burn-in" else production
        if int(block.get("start", -1)) != expected_start:
            raise RuntimeError("MCMC checkpoint block indices are not contiguous")
        path = (root / block["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("MCMC checkpoint block escapes its unit directory") from error
        if not path.is_file() or sha256_file(path) != block.get("sha256"):
            raise RuntimeError(f"MCMC checkpoint block failed checksum: {path}")
        if phase == "burn-in":
            burn = int(block["stop"])
        else:
            production = int(block["stop"])
    if burn != int(record.get("completed_burn_in", 0)) or production != int(
        record.get("completed_production", 0)
    ):
        raise RuntimeError("MCMC checkpoint counts disagree with committed blocks")
    return {
        "status": "resume-checksum-valid-atomic-block",
        "unit_directory": str(root),
        "completed_burn_in": burn,
        "completed_production": production,
    }
