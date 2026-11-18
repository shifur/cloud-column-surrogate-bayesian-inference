"""Saved timing conversion preserves distinct workload and clock definitions."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("figure5_with_sccm", Path(__file__).resolve().parents[1] / "scripts/replot_figure5_with_sccm.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_sccm_uses_only_saved_sccm_productive_records():
    seconds = [330000.0, 660000.0]
    rows = [dict(method="sccm", ensemble=e, transitions=330000,
                 productive_seconds=t, milliseconds_per_transition=t / 330)
            for e, t in enumerate(seconds)]
    rows.append(dict(method="dnn", productive_seconds="obsolete; do not read"))
    checkpoints = [dict(completed_burn_in=30000, completed_production=300000,
                        timing={"committed_sampler_wall_seconds": t}) for t in seconds]
    result = plot.sccm_statistics(rows, checkpoints)
    assert result["recorded_seconds"] == 495000
    assert result["milliseconds_per_transition"] == 1500
    assert result["recorded_workload_hours"] == 137.5
    assert result["ensembles_per_recorded_workload"] == 1
    assert result["sd_seconds"] == pytest.approx(np.std(seconds, ddof=1))
    checkpoints[0]["timing"]["committed_sampler_wall_seconds"] = 1
    with pytest.raises(AssertionError):
        plot.sccm_statistics(rows, checkpoints)


def _matched():
    row = dict(method="dnn", case="full_except_acc", ensemble_transitions=[330000]*2,
               burn_in_transitions=[30000]*2, retained_transitions=[300000]*2,
               complete_elapsed_coverage=True, config_sha256="frozen",
               policy="matched-fp32-two-concurrent-ensembles-v2",
               elapsed_seconds=660, milliseconds_per_paired_transition=2)
    ledger = dict(contract={"policy": row["policy"], "config_sha256": "frozen"},
                  unknown_elapsed_intervals=[], intervals=[{"elapsed_seconds": 300}, {"elapsed_seconds": 360}])
    return row, ledger


def test_matched_uses_elapsed_without_factor_of_two_or_fake_sd():
    row, ledger = _matched()
    result = plot.matched_statistics("dnn", row, ledger)
    assert result["recorded_seconds"] == 660
    assert result["milliseconds_per_transition"] == 2
    assert result["recorded_workload_hours"] == pytest.approx(660/3600)
    assert result["ensembles_per_recorded_workload"] == 2
    assert result["sd_seconds"] is None
    row["milliseconds_per_paired_transition"] = 1
    with pytest.raises(AssertionError):
        plot.matched_statistics("dnn", row, ledger)


def test_matched_rejects_gapped_or_incomplete_saved_timing():
    row, ledger = _matched()
    ledger["unknown_elapsed_intervals"].append({"started_utc": "lost"})
    with pytest.raises(ValueError, match="complete elapsed coverage"):
        plot.matched_statistics("dnn", row, ledger)
    row, ledger = _matched()
    row["retained_transitions"] = [2000, 2000]
    with pytest.raises(ValueError, match="complete elapsed coverage"):
        plot.matched_statistics("dnn", row, ledger)
