"""Figure 5 consumes paired elapsed ledgers and leaves scientific plots alone."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cloud_column_reproduction.mcmc.matched_timing import MatchedTiming
from cloud_column_reproduction.provenance import sha256_file


def _figure5(root):
    source = Path(__file__).resolve().parents[1] / "scripts/finalize_manuscript.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "figure5")
    manifests = []
    namespace = dict(ROOT=root, Path=Path, sha256_file=sha256_file, csv=csv,
                     np=np, plt=plt, COLORS={"dnn": "blue", "pdr": "orange"},
                     LABELS={"dnn": "DNN", "pdr": "CS"}, _style=lambda: None,
                     _manifest=lambda *args: manifests.append(args))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["figure5"], manifests


def _ledgers(root, config, transitions=330000):
    for method, seconds in (("dnn", 660), ("pdr", 330)):
        clock = MatchedTiming(root / "chains", "full_except_acc", method,
                             config_sha256=sha256_file(config.path), runtime={}, advance_blocks=50)
        with clock.measure():
            for ensemble in (0, 1):
                directory = root / "chains/full_except_acc" / method / f"ensemble_{ensemble:03d}"
                directory.mkdir(parents=True)
                (directory / "checkpoint.json").write_text(json.dumps({
                    "completed_burn_in": 30000, "completed_production": transitions - 30000,
                    "blocks": [], "timing": {"committed_sampler_wall_seconds": 999},
                }))
        record = json.loads(clock.path.read_text())
        record["intervals"][0]["elapsed_seconds"] = seconds
        clock.path.write_text(json.dumps(record))


def test_figure5_uses_paired_elapsed_not_productive_clocks(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("frozen fixture")
    config = type("Config", (), {"path": path})()
    _ledgers(tmp_path, config)
    figure5, manifests = _figure5(tmp_path)
    outputs = figure5({"chain_root": "chains", "primary_likelihood_case": "full_except_acc",
                       "figure_directory": "new_figures", "table_directory": "new_tables"}, config)
    assert outputs[0].is_file()
    with outputs[1].open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(r["milliseconds_per_paired_transition"]) for r in rows] == [2, 1]
    assert len(manifests) == 1
    assert manifests[0][-1]["error_bars"].startswith("none:")
    path.write_text("changed config")
    with pytest.raises(ValueError, match="checksum mismatch"):
        figure5({"chain_root": "chains", "primary_likelihood_case": "full_except_acc"}, config)


def test_figure5_rejects_smoke_results(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("fixture")
    config = type("Config", (), {"path": path})()
    _ledgers(tmp_path, config, transitions=32000)
    figure5, _ = _figure5(tmp_path)
    with pytest.raises(ValueError, match="completed 30k"):
        figure5({"chain_root": "chains", "primary_likelihood_case": "full_except_acc"}, config)
