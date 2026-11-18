import json
from pathlib import Path

import numpy as np
import pytest

from cloud_column_reproduction.mcmc.resumable import (
    AtomicBlockStore,
    TimedLogProbability,
    build_emcee_move,
    run_resumable_emcee,
)


BLOB_DTYPE = np.dtype(
    [("log_prior", "f8"), ("log_likelihood", "f8"), ("status", "i1")]
)


def _bindings() -> dict[str, str]:
    return {"config_sha256": "a" * 64, "unit": "test"}


def _log_probability(value: np.ndarray):
    logp = -0.5 * float(np.dot(value, value))
    return logp, 0.0, logp, 0


def _positions() -> np.ndarray:
    return np.random.default_rng(9).normal(scale=0.1, size=(32, 2))


def _production_arrays(store: AtomicBlockStore) -> dict[str, np.ndarray]:
    blocks = []
    for path in store.block_paths("production"):
        with np.load(path, allow_pickle=False) as archive:
            blocks.append(np.asarray(archive["position"]))
    return {"position": np.concatenate(blocks, axis=0)}


def test_supported_de_and_weighted_mixture_moves() -> None:
    pytest.importorskip("emcee")
    assert build_emcee_move("emcee.moves.DEMove", {}) is not None
    assert build_emcee_move("emcee.moves.DESnookerMove", {}) is not None
    mixture = build_emcee_move(
        "emcee.moves.WeightedMixture",
        {
            "components": [
                {"name": "emcee.moves.StretchMove", "parameters": {"a": 1.5}, "weight": 0.8},
                {"name": "emcee.moves.DEMove", "parameters": {}, "weight": 0.2},
            ]
        },
    )
    assert len(mixture) == 2
    with pytest.raises(ValueError, match="sum to one"):
        build_emcee_move(
            "emcee.moves.WeightedMixture",
            {
                "components": [
                    {"name": "emcee.moves.DEMove", "parameters": {}, "weight": 0.8},
                    {"name": "emcee.moves.DESnookerMove", "parameters": {}, "weight": 0.3},
                ]
            },
        )


def test_caught_interruption_timing_is_preserved_separately(tmp_path: Path) -> None:
    store = AtomicBlockStore(tmp_path / "unit", bindings=_bindings())
    store.start_session(setup_wall_seconds=1.25)
    store.begin_block("production", 0, 1000)
    store.abort_block(
        sampler_wall_seconds=3.5,
        log_probability_wall_seconds=3.0,
        log_probability_calls=7,
        proposal_evaluations=224,
        successful_evaluations=200,
        evaluator_failures=3,
        prior_rejections=21,
        reason="KeyboardInterrupt",
    )
    store.finish_session(status="failed-or-interrupted-after-catch")
    record = json.loads(store.checkpoint_path.read_text(encoding="utf-8"))
    assert record["completed_production"] == 0
    assert record["timing"]["observed_interrupted_sampler_wall_seconds"] == 3.5
    assert record["timing"]["observed_interrupted_log_probability_wall_seconds"] == 3.0
    assert record["timing"]["committed_sampler_wall_seconds"] == 0.0
    assert record["timing"]["observed_interrupted_successful_evaluations"] == 200
    assert record["timing"]["observed_interrupted_evaluator_failures"] == 3
    assert record["timing"]["observed_interrupted_prior_rejections"] == 21
    assert not store.inflight_path.exists()


def test_hard_loss_is_retained_as_an_upper_bound(tmp_path: Path) -> None:
    root = tmp_path / "unit"
    first = AtomicBlockStore(root, bindings=_bindings())
    first.start_session()
    first.begin_block("production", 0, 1000)
    resumed = AtomicBlockStore(root, bindings=_bindings())
    resumed.start_session()
    assert len(resumed.record["interrupted_attempts"]) == 1
    assert resumed.record["interrupted_attempts"][0]["status"] == (
        "interrupted-uncommitted-block"
    )
    assert resumed.record["timing"]["interrupted_wall_seconds_upper_bound"] >= 0.0


def test_unindexed_block_file_is_quarantined_and_does_not_block_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unit"
    store = AtomicBlockStore(root, bindings=_bindings())
    orphan = store.blocks_directory / "production_000000000_000001000.npz"
    orphan.write_bytes(b"partial-or-complete-but-unindexed")
    resumed = AtomicBlockStore(root, bindings=_bindings())
    assert not orphan.exists()
    assert len(resumed.record["orphaned_artifacts"]) == 1
    quarantined = root / resumed.record["orphaned_artifacts"][0]["quarantined_path"]
    assert quarantined.read_bytes() == b"partial-or-complete-but-unindexed"


def test_stale_inflight_marker_after_index_commit_adds_no_lost_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unit"
    store = AtomicBlockStore(root, bindings=_bindings())
    store.start_session()
    store.begin_block("production", 0, 4)
    store.commit_block(
        phase="production",
        start=0,
        stop=4,
        arrays={"marker": np.arange(4)},
        sampler_wall_seconds=1.0,
        log_probability_wall_seconds=0.8,
        log_probability_calls=4,
        proposal_evaluations=128,
        successful_evaluations=100,
        evaluator_failures=0,
        prior_rejections=28,
    )
    store.inflight_path.write_text(
        json.dumps(
            {
                "session_id": store.session_id,
                "phase": "production",
                "start": 0,
                "stop": 4,
                "started_utc": store.record["created_utc"],
            }
        )
    )
    resumed = AtomicBlockStore(root, bindings=_bindings())
    resumed.start_session()
    attempt = resumed.record["interrupted_attempts"][-1]
    assert attempt["status"] == "stale-inflight-marker-after-committed-block"
    assert resumed.record["timing"]["interrupted_wall_seconds_upper_bound"] == 0.0


def test_pause_and_resume_is_bitwise_identical_to_uninterrupted_blocks(
    tmp_path: Path,
) -> None:
    pytest.importorskip("emcee")
    initial = _positions()

    continuous = AtomicBlockStore(tmp_path / "continuous", bindings=_bindings())
    continuous.start_session()
    run_resumable_emcee(
        continuous,
        TimedLogProbability(_log_probability),
        initial,
        burn_in_steps=4,
        production_steps=12,
        block_transitions=4,
        proposal_seed=871,
        vectorize=False,
        blobs_dtype=BLOB_DTYPE,
        store_predictions=False,
    )
    continuous.finish_session()

    resumed = AtomicBlockStore(tmp_path / "resumed", bindings=_bindings())
    resumed.start_session()
    result = run_resumable_emcee(
        resumed,
        TimedLogProbability(_log_probability),
        initial,
        burn_in_steps=4,
        production_steps=12,
        block_transitions=4,
        proposal_seed=871,
        vectorize=False,
        blobs_dtype=BLOB_DTYPE,
        store_predictions=False,
        maximum_blocks=2,
    )
    resumed.finish_session()
    assert result["status"] == "paused-at-block-boundary"

    resumed = AtomicBlockStore(tmp_path / "resumed", bindings=_bindings())
    resumed.start_session()
    run_resumable_emcee(
        resumed,
        TimedLogProbability(_log_probability),
        initial,
        burn_in_steps=4,
        production_steps=12,
        block_transitions=4,
        proposal_seed=871,
        vectorize=False,
        blobs_dtype=BLOB_DTYPE,
        store_predictions=False,
    )
    resumed.finish_session()

    np.testing.assert_array_equal(
        _production_arrays(continuous)["position"],
        _production_arrays(resumed)["position"],
    )
    assert resumed.record["timing"]["committed_sampler_wall_seconds"] > 0.0
    assert len(resumed.record["sessions"]) == 2


def test_progress_reports_transition_metrics_without_changing_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("emcee")
    store = AtomicBlockStore(tmp_path / "progress", bindings=_bindings())
    store.start_session()
    run_resumable_emcee(
        store,
        TimedLogProbability(_log_probability),
        _positions(),
        burn_in_steps=4,
        production_steps=4,
        block_transitions=4,
        proposal_seed=871,
        vectorize=False,
        blobs_dtype=BLOB_DTYPE,
        store_predictions=False,
        maximum_blocks=1,
        progress_every_transitions=2,
    )
    store.finish_session()
    output = capsys.readouterr().out
    assert "MCMC block started: phase=burn-in transitions=0->4" in output
    assert "MCMC burn-in: 2/4 transitions in block" in output
    assert "block_acceptance=" in output
    assert "forward_success=" in output
    assert "prior_rejected=" in output
    assert "MCMC block committed:" in output
    assert store.record["completed_burn_in"] == 4


def test_progress_interval_must_be_positive(tmp_path: Path) -> None:
    pytest.importorskip("emcee")
    store = AtomicBlockStore(tmp_path / "bad-progress", bindings=_bindings())
    with pytest.raises(ValueError, match="progress_every_transitions"):
        run_resumable_emcee(
            store,
            TimedLogProbability(_log_probability),
            _positions(),
            burn_in_steps=4,
            production_steps=4,
            block_transitions=4,
            proposal_seed=871,
            vectorize=False,
            blobs_dtype=BLOB_DTYPE,
            store_predictions=False,
            progress_every_transitions=0,
        )


def test_stretch_move_pause_resume_is_bitwise_identical(tmp_path: Path) -> None:
    pytest.importorskip("emcee")
    initial = _positions()
    common = dict(
        burn_in_steps=4,
        production_steps=12,
        block_transitions=4,
        proposal_seed=912,
        vectorize=False,
        blobs_dtype=BLOB_DTYPE,
        store_predictions=False,
        move_name="emcee.moves.StretchMove",
        move_parameters={"a": 1.5},
    )
    continuous = AtomicBlockStore(tmp_path / "stretch-continuous", bindings=_bindings())
    continuous.start_session()
    run_resumable_emcee(
        continuous, TimedLogProbability(_log_probability), initial, **common
    )
    continuous.finish_session()

    resumed = AtomicBlockStore(tmp_path / "stretch-resumed", bindings=_bindings())
    resumed.start_session()
    run_resumable_emcee(
        resumed,
        TimedLogProbability(_log_probability),
        initial,
        maximum_blocks=2,
        **common,
    )
    resumed.finish_session()
    resumed = AtomicBlockStore(tmp_path / "stretch-resumed", bindings=_bindings())
    resumed.start_session()
    run_resumable_emcee(
        resumed, TimedLogProbability(_log_probability), initial, **common
    )
    resumed.finish_session()
    np.testing.assert_array_equal(
        _production_arrays(continuous)["position"],
        _production_arrays(resumed)["position"],
    )
