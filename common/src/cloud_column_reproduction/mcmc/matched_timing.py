"""Elapsed clocks for a method's complete concurrent two-ensemble workload."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

from cloud_column_reproduction.provenance import sha256_file


POLICY = "matched-fp32-two-concurrent-ensembles-v2"


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def snapshot(root: Path, case: str, method: str) -> list[dict]:
    result = []
    for ensemble in (0, 1):
        directory = root / case / method / f"ensemble_{ensemble:03d}"
        path = directory / "checkpoint.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        blocks = record.get("blocks", [])
        result.append({
            "burn_in": int(record.get("completed_burn_in", 0)),
            "production": int(record.get("completed_production", 0)),
            "blocks": [{"path": b["path"], "sha256": b["sha256"]} for b in blocks],
        })
    return result


class MatchedTiming:
    """Keep completed elapsed intervals across resumes; expose hard-loss gaps."""

    def __init__(self, root: Path, case: str, method: str, *, config_sha256: str,
                 runtime: dict, advance_blocks: int) -> None:
        self.root, self.case, self.method = Path(root), case, method
        self.path = self.root / case / "matched_timing" / f"{method}.json"
        self.contract = {"policy": POLICY, "config_sha256": config_sha256,
                         "advance_blocks": advance_blocks, "runtime": runtime}
        if self.path.is_file():
            self.record = json.loads(self.path.read_text())
            if self.record["contract"] != self.contract:
                raise ValueError("Matched timing contract changed; use a fresh output root")
            if self.record.get("inflight"):
                self.record["unknown_elapsed_intervals"].append(self.record.pop("inflight"))
                _write(self.path, self.record)
        else:
            if any(s["blocks"] for s in snapshot(self.root, case, method)):
                raise ValueError("Existing chains have no matched elapsed clocks; use a fresh root")
            self.record = {"schema_version": 1, "contract": self.contract,
                           "case": case, "method": method, "host": platform.node(),
                           "intervals": [], "unknown_elapsed_intervals": []}
            _write(self.path, self.record)

    @contextmanager
    def measure(self):
        before = snapshot(self.root, self.case, self.method)
        marker = {"started_utc": datetime.now(timezone.utc).isoformat(), "before": before}
        self.record["inflight"] = marker
        _write(self.path, self.record)
        status = "failed-or-interrupted"
        started = time.perf_counter()
        try:
            yield
            status = "completed"
        finally:
            elapsed = time.perf_counter() - started
            # Inventory and ledger serialization are outside the elapsed clock.
            after = snapshot(self.root, self.case, self.method)
            self.record["intervals"].append({**marker, "after": after,
                                              "elapsed_seconds": elapsed, "status": status})
            self.record.pop("inflight", None)
            _write(self.path, self.record)


def timing_summary(root: Path, case: str, method: str) -> dict:
    path = Path(root) / case / "matched_timing" / f"{method}.json"
    record = json.loads(path.read_text())
    if record["contract"]["policy"] != POLICY:
        raise ValueError("Unmatched timing policy")
    current = snapshot(Path(root), case, method)
    # Bind the reported timing to every committed block, including resumed work.
    previous = [{"burn_in": 0, "production": 0, "blocks": []} for _ in (0, 1)]
    gaps = bool(record.get("inflight") or record["unknown_elapsed_intervals"])
    for interval in record["intervals"]:
        if interval["before"] != previous:
            gaps = True
        previous = interval["after"]
    gaps |= previous != current
    for ensemble, state in enumerate(current):
        directory = Path(root) / case / method / f"ensemble_{ensemble:03d}"
        for block in state["blocks"]:
            if sha256_file(directory / block["path"]) != block["sha256"]:
                raise ValueError("Matched timing references a corrupt chain block")
    counts = [s["burn_in"] + s["production"] for s in current]
    checkpoints = []
    for ensemble in (0, 1):
        checkpoint = Path(root) / case / method / f"ensemble_{ensemble:03d}" / "checkpoint.json"
        checkpoints.append(json.loads(checkpoint.read_text()) if checkpoint.is_file() else {})
    productive = [float(c.get("timing", {}).get("committed_sampler_wall_seconds", 0.0))
                  for c in checkpoints]
    elapsed = sum(i["elapsed_seconds"] for i in record["intervals"])
    balanced = counts[0] == counts[1] and counts[0] > 0
    valid = balanced and not gaps
    return {
        "method": method, "case": case, "policy": POLICY,
        "config_sha256": record["contract"]["config_sha256"],
        "elapsed_seconds": elapsed,
        "ensemble_transitions": counts,
        "burn_in_transitions": [s["burn_in"] for s in current],
        "retained_transitions": [s["production"] for s in current],
        "productive_sampler_seconds_by_ensemble": productive,
        "mean_productive_ms_per_ensemble_transition": (
            sum(1000 * seconds / count for seconds, count in zip(productive, counts)) / 2
            if all(counts) else None),
        "productive_clock_note": "ensemble clocks overlap; their mean does not measure the paired workload elapsed time",
        "complete_elapsed_coverage": not gaps,
        "milliseconds_per_paired_transition": 1000 * elapsed / counts[0] if valid else None,
        "paired_transition_definition": "both 32-walker ensembles advance once (64 walker proposals)",
        "scope": "active advance elapsed time including sampler, session setup, checkpoint I/O and retries; excludes runtime construction/prewarm, idle gaps, diagnostics and finalization",
        "failed_interval_seconds": sum(i["elapsed_seconds"] for i in record["intervals"] if i["status"] != "completed"),
        "ledger": str(path),
    }


@contextmanager
def exclusive_case(root: Path, case: str):
    """Prevent two driver processes from writing the same case's clocks/chains."""
    import fcntl

    directory = Path(root) / case
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".matched-driver.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
