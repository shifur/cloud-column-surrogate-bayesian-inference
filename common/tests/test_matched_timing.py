import json

import pytest

from cloud_column_reproduction.mcmc.matched_timing import MatchedTiming, timing_summary
from cloud_column_reproduction.provenance import sha256_file


def _commit(root, ensemble, count=1000):
    directory = root / "full_except_acc" / "dnn" / f"ensemble_{ensemble:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    block = directory / "block.npz"
    block.write_bytes(b"test atomic block")
    (directory / "checkpoint.json").write_text(json.dumps({
        "completed_burn_in": count, "completed_production": 0,
        "blocks": [{"path": block.name, "sha256": sha256_file(block)}],
    }))


def _clock(root, **kwargs):
    return MatchedTiming(root, "full_except_acc", "dnn", config_sha256="a" * 64,
                         runtime={"dtype": "float32"}, advance_blocks=50, **kwargs)


def test_elapsed_clock_uses_paired_not_sum_of_ensemble_transitions(tmp_path):
    clock = _clock(tmp_path)
    with clock.measure():
        _commit(tmp_path, 0)
        _commit(tmp_path, 1)
    record = json.loads(clock.path.read_text())
    record["intervals"][0]["elapsed_seconds"] = 2.8
    clock.path.write_text(json.dumps(record))
    result = timing_summary(tmp_path, "full_except_acc", "dnn")
    assert result["milliseconds_per_paired_transition"] == pytest.approx(2.8)
    resumed = _clock(tmp_path)
    assert resumed.record["intervals"] == record["intervals"]
    block = tmp_path / "full_except_acc/dnn/ensemble_000/block.npz"
    block.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        timing_summary(tmp_path, "full_except_acc", "dnn")


def test_hard_loss_preserves_old_clocks_and_refuses_complete_elapsed_claim(tmp_path):
    clock = _clock(tmp_path)
    clock.record["inflight"] = {"started_utc": "test", "before": []}
    clock.path.write_text(json.dumps(clock.record))
    resumed = _clock(tmp_path)
    with resumed.measure():
        _commit(tmp_path, 0)
        _commit(tmp_path, 1)
    result = timing_summary(tmp_path, "full_except_acc", "dnn")
    assert not result["complete_elapsed_coverage"]
    assert result["milliseconds_per_paired_transition"] is None


def test_existing_unclocked_chains_and_changed_contract_are_rejected(tmp_path):
    clock = _clock(tmp_path)
    with pytest.raises(ValueError, match="contract changed"):
        MatchedTiming(tmp_path, "full_except_acc", "dnn", config_sha256="b" * 64,
                      runtime={}, advance_blocks=50)
    clock.path.unlink()
    _commit(tmp_path, 0)
    with pytest.raises(ValueError, match="no matched elapsed"):
        _clock(tmp_path)


def test_caught_partial_interval_is_measured_without_inventing_transitions(tmp_path):
    clock = _clock(tmp_path)
    with pytest.raises(RuntimeError):
        with clock.measure():
            _commit(tmp_path, 0)
            raise RuntimeError("interrupted")
    result = timing_summary(tmp_path, "full_except_acc", "dnn")
    assert result["complete_elapsed_coverage"]
    assert result["failed_interval_seconds"] > 0
    assert result["milliseconds_per_paired_transition"] is None
