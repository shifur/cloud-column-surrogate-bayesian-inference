"""Atomic block checkpoints and exact emcee continuation.

Immutable NPZ blocks are the live authoritative sampling record.  A compact
JSON index points only to checksum-valid committed blocks.  An interrupted
block is never incorporated and is rerun from the preceding saved Markov and
random-number state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from cloud_column_reproduction.provenance import sha256_file


CHECKPOINT_SCHEMA_VERSION = 3


def build_emcee_move(name: str, parameters: dict[str, Any] | None = None) -> Any:
    """Construct one explicitly supported, checksum-bound emcee move."""
    try:
        import emcee
    except ImportError as error:
        raise RuntimeError("emcee==3.1.6 is required for controlled MCMC") from error
    values = dict(parameters or {})
    if name == "emcee.moves.WalkMove" and not values:
        return emcee.moves.WalkMove()
    if name == "emcee.moves.StretchMove" and set(values) == {"a"}:
        scale = float(values["a"])
        if not np.isfinite(scale) or scale <= 1.0:
            raise ValueError("StretchMove scale a must be finite and greater than one")
        return emcee.moves.StretchMove(a=scale)
    if name == "emcee.moves.DEMove" and set(values).issubset({"sigma", "gamma0"}):
        sigma = float(values.get("sigma", 1.0e-5))
        gamma0 = values.get("gamma0")
        gamma0 = None if gamma0 is None else float(gamma0)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("DEMove sigma must be finite and positive")
        if gamma0 is not None and (not np.isfinite(gamma0) or gamma0 <= 0.0):
            raise ValueError("DEMove gamma0 must be finite and positive or null")
        return emcee.moves.DEMove(sigma=sigma, gamma0=gamma0)
    if name == "emcee.moves.DESnookerMove" and set(values).issubset({"gammas"}):
        gammas = float(values.get("gammas", 1.7))
        if not np.isfinite(gammas) or gammas <= 0.0:
            raise ValueError("DESnookerMove gammas must be finite and positive")
        return emcee.moves.DESnookerMove(gammas=gammas)
    if name == "emcee.moves.WeightedMixture" and set(values) == {"components"}:
        components = values["components"]
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("WeightedMixture requires at least two components")
        result = []
        weight_sum = 0.0
        for component in components:
            if not isinstance(component, dict) or set(component) != {
                "name", "parameters", "weight"
            }:
                raise ValueError("Each move component requires name, parameters, and weight")
            weight = float(component["weight"])
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("Move mixture weights must be finite and positive")
            result.append(
                (
                    build_emcee_move(component["name"], component["parameters"]),
                    weight,
                )
            )
            weight_sum += weight
        if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("Move mixture weights must sum to one")
        return result
    raise ValueError(f"Unsupported emcee move specification: {name}, {values}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _random_state_arrays(state: tuple[Any, ...]) -> dict[str, np.ndarray]:
    if len(state) != 5 or state[0] != "MT19937":
        raise ValueError("Controlled emcee continuation requires NumPy MT19937 state")
    return {
        "rng_algorithm": np.asarray(str(state[0])),
        "rng_keys": np.asarray(state[1], dtype=np.uint32),
        "rng_position": np.asarray(int(state[2]), dtype=np.int64),
        "rng_has_gauss": np.asarray(int(state[3]), dtype=np.int8),
        "rng_cached_gaussian": np.asarray(float(state[4]), dtype=np.float64),
    }


def _random_state_from_archive(archive: Any) -> tuple[Any, ...]:
    return (
        str(archive["rng_algorithm"]),
        np.asarray(archive["rng_keys"], dtype=np.uint32),
        int(archive["rng_position"]),
        int(archive["rng_has_gauss"]),
        float(archive["rng_cached_gaussian"]),
    )


@dataclass(frozen=True)
class ResumeState:
    coords: np.ndarray
    log_prob: np.ndarray
    blobs: np.ndarray
    random_state: tuple[Any, ...]


class TimedLogProbability:
    """Measure callback time and proposal evaluations without changing values."""

    def __init__(self, function: Callable[[np.ndarray], Any]) -> None:
        self.function = function
        self.calls = 0
        self.evaluations = 0
        self.wall_seconds = 0.0
        self.successful_evaluations = 0
        self.evaluator_failures = 0
        self.prior_rejections = 0

    def __call__(self, values: np.ndarray) -> Any:
        array = np.asarray(values)
        count = 1 if array.ndim == 1 else len(array)
        started = time.perf_counter()
        try:
            result = self.function(values)
            rows = result if array.ndim == 2 else [result]
            for row in rows:
                status = int(row[3])
                if status == 0:
                    self.successful_evaluations += 1
                elif status == 1:
                    self.evaluator_failures += 1
                elif status == 2:
                    self.prior_rejections += 1
                else:
                    raise RuntimeError(f"Unknown posterior evaluation status: {status}")
            return result
        finally:
            self.wall_seconds += time.perf_counter() - started
            self.calls += 1
            self.evaluations += count

    def snapshot(self) -> tuple[int, int, float, int, int, int]:
        return (
            self.calls,
            self.evaluations,
            self.wall_seconds,
            self.successful_evaluations,
            self.evaluator_failures,
            self.prior_rejections,
        )


class AtomicBlockStore:
    """Checksum-indexed chain blocks with timing and interruption accounting."""

    def __init__(self, root: str | Path, *, bindings: dict[str, str]) -> None:
        self.root = Path(root).resolve()
        self.blocks_directory = self.root / "blocks"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.inflight_path = self.root / "inflight.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blocks_directory.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path.exists():
            self.record = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if self.record.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise RuntimeError("Unsupported MCMC checkpoint schema")
            if self.record.get("bindings") != bindings:
                raise RuntimeError("MCMC checkpoint bindings do not match the requested unit")
            self._verify_blocks()
            self._quarantine_orphan_blocks()
        else:
            self.record = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "status": "running",
                "bindings": dict(bindings),
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
                "completed_burn_in": 0,
                "completed_production": 0,
                "blocks": [],
                "sessions": [],
                "interrupted_attempts": [],
                "orphaned_artifacts": [],
                "timing": {
                    "committed_sampler_wall_seconds": 0.0,
                    "committed_log_probability_wall_seconds": 0.0,
                    "checkpoint_io_wall_seconds": 0.0,
                    "session_setup_wall_seconds": 0.0,
                    "committed_log_probability_calls": 0,
                    "committed_proposal_evaluations": 0,
                    "committed_successful_evaluations": 0,
                    "committed_evaluator_failures": 0,
                    "committed_prior_rejections": 0,
                    "interrupted_wall_seconds_upper_bound": 0.0,
                    "observed_interrupted_sampler_wall_seconds": 0.0,
                    "observed_interrupted_log_probability_wall_seconds": 0.0,
                    "observed_interrupted_log_probability_calls": 0,
                    "observed_interrupted_proposal_evaluations": 0,
                    "observed_interrupted_successful_evaluations": 0,
                    "observed_interrupted_evaluator_failures": 0,
                    "observed_interrupted_prior_rejections": 0,
                },
            }
            _atomic_json(self.checkpoint_path, self.record)
            self._quarantine_orphan_blocks()
        self.session_id: str | None = None

    def _verify_blocks(self) -> None:
        burn = 0
        production = 0
        for block in self.record.get("blocks", []):
            path = self.root / block["path"]
            if not path.is_file() or sha256_file(path) != block["sha256"]:
                raise RuntimeError(f"MCMC committed block failed checksum: {path}")
            expected_start = burn if block["phase"] == "burn-in" else production
            if int(block["start"]) != expected_start:
                raise RuntimeError("MCMC block index is not contiguous")
            if block["phase"] == "burn-in":
                burn = int(block["stop"])
            elif block["phase"] == "production":
                production = int(block["stop"])
            else:
                raise RuntimeError("Unknown MCMC block phase")
        if burn != int(self.record["completed_burn_in"]) or production != int(
            self.record["completed_production"]
        ):
            raise RuntimeError("MCMC checkpoint counts disagree with committed blocks")

    def _quarantine_orphan_blocks(self) -> None:
        """Preserve files not referenced by the atomic checkpoint index.

        Such files can remain if a node dies after an NPZ rename but before the
        JSON index commits. They are not part of the chain and must not prevent
        the same deterministic block from being written again.
        """
        referenced = {
            str((self.root / block["path"]).resolve())
            for block in self.record.get("blocks", [])
        }
        candidates = [path for path in self.blocks_directory.iterdir() if path.is_file()]
        orphaned = [path for path in candidates if str(path.resolve()) not in referenced]
        if not orphaned:
            return
        directory = self.root / "orphaned"
        directory.mkdir(parents=True, exist_ok=True)
        for path in orphaned:
            destination = directory / f"{path.name}.{uuid4().hex}.orphaned"
            os.replace(path, destination)
            self.record.setdefault("orphaned_artifacts", []).append(
                {
                    "original_path": str(path.relative_to(self.root)),
                    "quarantined_path": str(destination.relative_to(self.root)),
                    "sha256": sha256_file(destination),
                    "detected_utc": _utc_now(),
                    "reason": "file-not-referenced-by-atomic-checkpoint-index",
                }
            )
        self._write_index()

    def start_session(
        self,
        *,
        setup_wall_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = _utc_now()
        if self.inflight_path.exists():
            abandoned = json.loads(self.inflight_path.read_text(encoding="utf-8"))
            committed = any(
                block["phase"] == abandoned["phase"]
                and int(block["start"]) == int(abandoned["start"])
                and int(block["stop"]) == int(abandoned["stop"])
                for block in self.record.get("blocks", [])
            )
            if committed:
                upper = 0.0
                abandoned.update(
                    status="stale-inflight-marker-after-committed-block",
                    detected_utc=now,
                    wall_seconds_upper_bound=0.0,
                )
            else:
                started = datetime.fromisoformat(abandoned["started_utc"])
                detected = datetime.fromisoformat(now)
                upper = max(0.0, (detected - started).total_seconds())
                abandoned.update(
                    status="interrupted-uncommitted-block",
                    detected_utc=now,
                    wall_seconds_upper_bound=upper,
                )
            self.record["interrupted_attempts"].append(abandoned)
            self.record["timing"]["interrupted_wall_seconds_upper_bound"] += upper
            self.inflight_path.unlink()
        for session in self.record["sessions"]:
            if session["status"] == "active":
                session["status"] = "interrupted"
                session["detected_utc"] = now
        self.session_id = uuid4().hex
        self.record["sessions"].append(
            {
                "session_id": self.session_id,
                "status": "active",
                "started_utc": now,
                "setup_wall_seconds": float(setup_wall_seconds),
                "metadata": dict(metadata or {}),
            }
        )
        self.record["timing"]["session_setup_wall_seconds"] += float(setup_wall_seconds)
        self._write_index()
        return self.session_id

    def begin_block(self, phase: str, start: int, stop: int) -> None:
        if self.session_id is None:
            raise RuntimeError("Start a timing session before an MCMC block")
        _atomic_json(
            self.inflight_path,
            {
                "session_id": self.session_id,
                "phase": phase,
                "start": int(start),
                "stop": int(stop),
                "started_utc": _utc_now(),
            },
        )

    def commit_block(
        self,
        *,
        phase: str,
        start: int,
        stop: int,
        arrays: dict[str, np.ndarray],
        sampler_wall_seconds: float,
        log_probability_wall_seconds: float,
        log_probability_calls: int,
        proposal_evaluations: int,
        successful_evaluations: int,
        evaluator_failures: int,
        prior_rejections: int,
    ) -> Path:
        if successful_evaluations + evaluator_failures + prior_rejections != proposal_evaluations:
            raise RuntimeError(
                "MCMC evaluator-status counts do not equal proposal evaluations"
            )
        expected = int(self.record[f"completed_{'burn_in' if phase == 'burn-in' else 'production'}"])
        if start != expected or stop <= start:
            raise RuntimeError("MCMC block does not continue the committed chain")
        name = f"{phase.replace('-', '_')}_{start:09d}_{stop:09d}.npz"
        path = self.blocks_directory / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite committed MCMC block: {path}")
        io_started = time.perf_counter()
        _atomic_npz(path, arrays)
        digest = sha256_file(path)
        checkpoint_io = time.perf_counter() - io_started
        block = {
            "phase": phase,
            "start": int(start),
            "stop": int(stop),
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
            "session_id": self.session_id,
            "sampler_wall_seconds": float(sampler_wall_seconds),
            "log_probability_wall_seconds": float(log_probability_wall_seconds),
            "log_probability_calls": int(log_probability_calls),
            "proposal_evaluations": int(proposal_evaluations),
            "successful_evaluations": int(successful_evaluations),
            "evaluator_failures": int(evaluator_failures),
            "prior_rejections": int(prior_rejections),
            "checkpoint_io_wall_seconds": float(checkpoint_io),
            "committed_utc": _utc_now(),
        }
        self.record["blocks"].append(block)
        self.record[f"completed_{'burn_in' if phase == 'burn-in' else 'production'}"] = int(stop)
        timing = self.record["timing"]
        timing["committed_sampler_wall_seconds"] += float(sampler_wall_seconds)
        timing["committed_log_probability_wall_seconds"] += float(log_probability_wall_seconds)
        timing["committed_log_probability_calls"] += int(log_probability_calls)
        timing["committed_proposal_evaluations"] += int(proposal_evaluations)
        timing["committed_successful_evaluations"] += int(successful_evaluations)
        timing["committed_evaluator_failures"] += int(evaluator_failures)
        timing["committed_prior_rejections"] += int(prior_rejections)
        timing["checkpoint_io_wall_seconds"] += float(checkpoint_io)
        self._write_index()
        self.inflight_path.unlink(missing_ok=True)
        return path

    def abort_block(
        self,
        *,
        sampler_wall_seconds: float,
        log_probability_wall_seconds: float,
        log_probability_calls: int,
        proposal_evaluations: int,
        successful_evaluations: int,
        evaluator_failures: int,
        prior_rejections: int,
        reason: str,
    ) -> None:
        """Record measured work from a caught, uncommitted block.

        The Markov state is deliberately not advanced.  On resume the entire
        block is repeated from the preceding committed state and RNG state.
        This timing is therefore reported as interruption overhead, never as
        productive sampling throughput.
        """
        if not self.inflight_path.exists():
            return
        attempt = json.loads(self.inflight_path.read_text(encoding="utf-8"))
        attempt.update(
            status="caught-interruption-uncommitted-block",
            detected_utc=_utc_now(),
            reason=str(reason),
            observed_sampler_wall_seconds=float(sampler_wall_seconds),
            observed_log_probability_wall_seconds=float(log_probability_wall_seconds),
            observed_log_probability_calls=int(log_probability_calls),
            observed_proposal_evaluations=int(proposal_evaluations),
            observed_successful_evaluations=int(successful_evaluations),
            observed_evaluator_failures=int(evaluator_failures),
            observed_prior_rejections=int(prior_rejections),
        )
        self.record["interrupted_attempts"].append(attempt)
        timing = self.record["timing"]
        timing["observed_interrupted_sampler_wall_seconds"] += float(sampler_wall_seconds)
        timing["observed_interrupted_log_probability_wall_seconds"] += float(
            log_probability_wall_seconds
        )
        timing["observed_interrupted_log_probability_calls"] += int(
            log_probability_calls
        )
        timing["observed_interrupted_proposal_evaluations"] += int(
            proposal_evaluations
        )
        timing["observed_interrupted_successful_evaluations"] += int(
            successful_evaluations
        )
        timing["observed_interrupted_evaluator_failures"] += int(
            evaluator_failures
        )
        timing["observed_interrupted_prior_rejections"] += int(prior_rejections)
        self.inflight_path.unlink()
        self._write_index()

    def finish_session(self, *, status: str = "complete") -> None:
        if self.session_id is None:
            return
        for session in self.record["sessions"]:
            if session["session_id"] == self.session_id:
                session["status"] = status
                session["finished_utc"] = _utc_now()
                break
        self.session_id = None
        self._write_index()

    def mark_sampling_complete(self) -> None:
        self.record["status"] = "sampling-complete"
        self._write_index()

    def _write_index(self) -> None:
        self.record["updated_utc"] = _utc_now()
        _atomic_json(self.checkpoint_path, self.record)

    def last_state(self) -> ResumeState | None:
        if not self.record["blocks"]:
            return None
        block = self.root / self.record["blocks"][-1]["path"]
        with np.load(block, allow_pickle=False) as archive:
            return ResumeState(
                coords=np.asarray(archive["state_coords"], dtype=np.float64),
                log_prob=np.asarray(archive["state_log_prob"], dtype=np.float64),
                blobs=np.asarray(archive["state_blobs"]),
                random_state=_random_state_from_archive(archive),
            )

    def block_paths(self, phase: str) -> tuple[Path, ...]:
        return tuple(
            self.root / block["path"]
            for block in self.record["blocks"]
            if block["phase"] == phase
        )

    def timing_summary(self) -> dict[str, Any]:
        timing = dict(self.record["timing"])
        created = datetime.fromisoformat(self.record["created_utc"])
        updated = datetime.fromisoformat(self.record["updated_utc"])
        timing["calendar_elapsed_seconds"] = max(0.0, (updated - created).total_seconds())
        timing["completed_burn_in"] = int(self.record["completed_burn_in"])
        timing["completed_production"] = int(self.record["completed_production"])
        timing["timing_scope"] = (
            "productive committed blocks are exact; caught interruptions retain "
            "measured partial-block overhead; hard-loss intervals are upper bounds; "
            "all uncommitted work is excluded from productive throughput"
        )
        return timing


def _block_arrays(
    sampler: Any,
    *,
    starting_coords: np.ndarray,
    phase: str,
    store_predictions: bool,
) -> dict[str, np.ndarray]:
    positions = np.asarray(sampler.get_chain(), dtype=np.float64)
    log_posterior = np.asarray(sampler.get_log_prob(), dtype=np.float64)
    blobs = np.asarray(sampler.get_blobs())
    if blobs.dtype.names is None:
        raise RuntimeError("Controlled MCMC blobs must use a structured dtype")
    accepted = np.zeros(positions.shape[:2], dtype=np.int8)
    accepted[0] = np.any(positions[0] != starting_coords, axis=1)
    accepted[1:] = np.any(positions[1:] != positions[:-1], axis=2)
    state = sampler.get_last_sample()
    arrays: dict[str, np.ndarray] = {
        "position": positions,
        "log_posterior": log_posterior,
        "log_prior": np.asarray(blobs["log_prior"], dtype=np.float64),
        "log_likelihood": np.asarray(blobs["log_likelihood"], dtype=np.float64),
        "evaluation_status": np.asarray(blobs["status"], dtype=np.int8),
        "accepted": accepted,
        "state_coords": np.asarray(state.coords, dtype=np.float64),
        "state_log_prob": np.asarray(state.log_prob, dtype=np.float64),
        "state_blobs": np.asarray(state.blobs),
        "phase": np.asarray(phase),
    }
    if store_predictions and phase == "production":
        if "prediction" not in blobs.dtype.names:
            raise RuntimeError("SCCM production blocks require prediction blobs")
        arrays["sccm_prediction_cache"] = np.asarray(blobs["prediction"], dtype=np.float64)
    arrays.update(_random_state_arrays(sampler.random_state))
    return arrays


def run_resumable_emcee(
    store: AtomicBlockStore,
    log_probability: TimedLogProbability,
    initial_positions: np.ndarray,
    *,
    burn_in_steps: int,
    production_steps: int,
    block_transitions: int,
    proposal_seed: int,
    vectorize: bool,
    blobs_dtype: np.dtype,
    store_predictions: bool,
    maximum_blocks: int | None = None,
    progress_every_transitions: int | None = None,
    move_name: str = "emcee.moves.WalkMove",
    move_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one ensemble to its requested target, resuming exactly."""
    try:
        import emcee
    except ImportError as error:
        raise RuntimeError("emcee==3.1.6 is required for controlled MCMC") from error
    if block_transitions < 1 or burn_in_steps < 0 or production_steps < 1:
        raise ValueError("Invalid resumable MCMC lengths")
    if progress_every_transitions is not None and progress_every_transitions < 1:
        raise ValueError("progress_every_transitions must be positive or None")
    initial = np.asarray(initial_positions, dtype=np.float64)
    if initial.shape[0] != 32 or initial.ndim != 2:
        raise ValueError("Controlled MCMC requires initial shape (32, parameter)")
    completed_blocks = 0
    state_record = store.last_state()
    state: Any
    if state_record is None:
        state = initial
        current_coords = initial
    else:
        state = emcee.State(
            state_record.coords,
            log_prob=state_record.log_prob,
            blobs=state_record.blobs,
            random_state=state_record.random_state,
        )
        current_coords = state_record.coords

    for phase, target in (("burn-in", burn_in_steps), ("production", production_steps)):
        completed_key = "completed_burn_in" if phase == "burn-in" else "completed_production"
        while int(store.record[completed_key]) < target:
            if maximum_blocks is not None and completed_blocks >= maximum_blocks:
                return {"status": "paused-at-block-boundary", **store.timing_summary()}
            start = int(store.record[completed_key])
            stop = min(target, start + block_transitions)
            steps = stop - start
            sampler = emcee.EnsembleSampler(
                initial.shape[0],
                initial.shape[1],
                log_probability,
                moves=build_emcee_move(move_name, move_parameters),
                vectorize=vectorize,
                blobs_dtype=blobs_dtype,
            )
            if state_record is None and not store.record["blocks"]:
                sampler.random_state = np.random.RandomState(proposal_seed).get_state()
            elif hasattr(state, "random_state") and state.random_state is not None:
                sampler.random_state = state.random_state
            (
                before_calls,
                before_evaluations,
                before_log_wall,
                before_successes,
                before_failures,
                before_rejections,
            ) = log_probability.snapshot()
            store.begin_block(phase, start, stop)
            started = time.perf_counter()
            if progress_every_transitions is not None:
                print(
                    f"MCMC block started: phase={phase} transitions={start}->{stop} "
                    f"report_every={progress_every_transitions}",
                    flush=True,
                )
            try:
                for completed, state in enumerate(
                    sampler.sample(state, iterations=steps, progress=False), start=1
                ):
                    if progress_every_transitions is None or (
                        completed % progress_every_transitions != 0
                        and completed != steps
                    ):
                        continue
                    elapsed = time.perf_counter() - started
                    (
                        current_calls,
                        current_evaluations,
                        current_log_wall,
                        current_successes,
                        current_failures,
                        current_rejections,
                    ) = log_probability.snapshot()
                    rate = completed / elapsed if elapsed > 0.0 else float("nan")
                    eta = (steps - completed) / rate if rate > 0.0 else float("nan")
                    acceptance = float(np.mean(sampler.acceptance_fraction))
                    print(
                        f"MCMC {phase}: {completed}/{steps} transitions in block "
                        f"(global={start + completed}/{target}) | "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                        f"rate={rate:.3f} transitions/s | "
                        f"block_acceptance={acceptance:.3f} | "
                        f"proposals={current_evaluations - before_evaluations} "
                        f"forward_success={current_successes - before_successes} "
                        f"prior_rejected={current_rejections - before_rejections} "
                        f"failures={current_failures - before_failures} | "
                        f"logp_wall={current_log_wall - before_log_wall:.1f}s "
                        f"logp_batches={current_calls - before_calls}",
                        flush=True,
                    )
            except BaseException as error:
                sampler_wall = time.perf_counter() - started
                (
                    after_calls,
                    after_evaluations,
                    after_log_wall,
                    after_successes,
                    after_failures,
                    after_rejections,
                ) = log_probability.snapshot()
                store.abort_block(
                    sampler_wall_seconds=sampler_wall,
                    log_probability_wall_seconds=after_log_wall - before_log_wall,
                    log_probability_calls=after_calls - before_calls,
                    proposal_evaluations=after_evaluations - before_evaluations,
                    successful_evaluations=after_successes - before_successes,
                    evaluator_failures=after_failures - before_failures,
                    prior_rejections=after_rejections - before_rejections,
                    reason=type(error).__name__,
                )
                raise
            sampler_wall = time.perf_counter() - started
            (
                after_calls,
                after_evaluations,
                after_log_wall,
                after_successes,
                after_failures,
                after_rejections,
            ) = log_probability.snapshot()
            block_failures = after_failures - before_failures
            if block_failures:
                store.abort_block(
                    sampler_wall_seconds=sampler_wall,
                    log_probability_wall_seconds=after_log_wall - before_log_wall,
                    log_probability_calls=after_calls - before_calls,
                    proposal_evaluations=after_evaluations - before_evaluations,
                    successful_evaluations=after_successes - before_successes,
                    evaluator_failures=block_failures,
                    prior_rejections=after_rejections - before_rejections,
                    reason="evaluator-failure",
                )
                raise RuntimeError(
                    "Refusing to commit an MCMC block containing "
                    f"{block_failures} evaluator failures; resume retries the block"
                )
            arrays = _block_arrays(
                sampler,
                starting_coords=current_coords,
                phase=phase,
                store_predictions=store_predictions,
            )
            committed_path = store.commit_block(
                phase=phase,
                start=start,
                stop=stop,
                arrays=arrays,
                sampler_wall_seconds=sampler_wall,
                log_probability_wall_seconds=after_log_wall - before_log_wall,
                log_probability_calls=after_calls - before_calls,
                proposal_evaluations=after_evaluations - before_evaluations,
                successful_evaluations=after_successes - before_successes,
                evaluator_failures=block_failures,
                prior_rejections=after_rejections - before_rejections,
            )
            if progress_every_transitions is not None:
                print(
                    f"MCMC block committed: phase={phase} transitions={start}->{stop} "
                    f"sampler_wall={sampler_wall:.1f}s "
                    f"checkpoint={committed_path}",
                    flush=True,
                )
            current_coords = np.asarray(state.coords, dtype=np.float64)
            state_record = store.last_state()
            completed_blocks += 1
    store.mark_sampling_complete()
    return {"status": "sampling-complete", **store.timing_summary()}
