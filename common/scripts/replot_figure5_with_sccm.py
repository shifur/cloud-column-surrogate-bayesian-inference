#!/usr/bin/env python3
"""Additional Figure 5 presentation from saved production/matched timing only.

Run from common: python scripts/replot_figure5_with_sccm.py
SCCM's single-ensemble productive clock is never converted to a paired elapsed
clock. No sampler, evaluator, model, or posterior-finalization code is invoked.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_manuscript import COLORS, _style
from cloud_column_reproduction.provenance import sha256_file

TOTAL = 330000
NOTE = ("SCCM: saved production productive time for one ensemble (mean ± SD of two).\n"
        "DNN/CS: saved matched FP32 elapsed time for a concurrent pair; timing scopes differ.")
CAVEAT = ("SCCM uses committed productive sampler time per single 32-walker ensemble, "
          "excluding setup, checkpoint I/O and interrupted work. DNN and CS use complete "
          "active elapsed time for two concurrent 32-walker ensembles, including session "
          "checks, checkpoint I/O and measured retries; construction/prewarm and idle "
          "gaps are excluded. These are different clocks and workloads. SCCM is a "
          "historical reference, not a matched paired elapsed benchmark. No SCCM speedup "
          "ratio is inferred, and no factor-of-two conversion is applied.")


def sccm_statistics(rows: list[dict], checkpoints: list[dict]) -> dict:
    selected = sorted((r for r in rows if r["method"] == "sccm"), key=lambda r: int(r["ensemble"]))
    if [int(r["ensemble"]) for r in selected] != [0, 1] or len(checkpoints) != 2:
        raise ValueError("Expected two saved SCCM ensemble timing records")
    seconds = []
    for row, checkpoint in zip(selected, checkpoints, strict=True):
        if int(row["transitions"]) != TOTAL or (
            checkpoint["completed_burn_in"], checkpoint["completed_production"]
        ) != (30000, 300000):
            raise ValueError("SCCM requires the saved complete 30k + 300k workload")
        elapsed = float(row["productive_seconds"])
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid SCCM productive time")
        np.testing.assert_equal(elapsed, checkpoint["timing"]["committed_sampler_wall_seconds"])
        np.testing.assert_allclose(float(row["milliseconds_per_transition"]), 1000 * elapsed / TOTAL, rtol=1e-14)
        seconds.append(elapsed)
    return dict(method="sccm", label="SCCM", source_workflow="historical-production",
                clock="committed productive sampler", ensembles_per_recorded_workload=1,
                walkers_per_ensemble=32, transitions_per_ensemble=TOTAL,
                timing_records=2, aggregation="mean of two ensemble clocks; sample SD ddof=1",
                recorded_seconds=float(np.mean(seconds)), sd_seconds=float(np.std(seconds, ddof=1)),
                milliseconds_per_transition=float(np.mean(seconds)) * 1000 / TOTAL,
                sd_milliseconds_per_transition=float(np.std(seconds, ddof=1)) * 1000 / TOTAL,
                recorded_workload_hours=float(np.mean(seconds)) / 3600,
                sd_recorded_workload_hours=float(np.std(seconds, ddof=1)) / 3600)


def matched_statistics(method: str, row: dict, ledger: dict) -> dict:
    if row["method"] != method or row["case"] != "full_except_acc" or (
        row["ensemble_transitions"] != [TOTAL, TOTAL]
        or row["burn_in_transitions"] != [30000, 30000]
        or row["retained_transitions"] != [300000, 300000]
        or not row["complete_elapsed_coverage"]
        or ledger.get("inflight") or ledger["unknown_elapsed_intervals"]
    ):
        raise ValueError("Matched timing requires completed pairs and complete elapsed coverage")
    if row["policy"] != "matched-fp32-two-concurrent-ensembles-v2" or row["policy"] != ledger["contract"]["policy"]:
        raise ValueError("Unexpected matched precision/workload policy")
    if row["config_sha256"] != ledger["contract"]["config_sha256"]:
        raise ValueError("Matched timing config binding mismatch")
    elapsed = float(row["elapsed_seconds"])
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid matched elapsed time")
    np.testing.assert_allclose(elapsed, sum(i["elapsed_seconds"] for i in ledger["intervals"]), rtol=1e-14)
    np.testing.assert_allclose(row["milliseconds_per_paired_transition"], elapsed * 1000 / TOTAL, rtol=1e-14)
    return dict(method=method, label="DNN" if method == "dnn" else "CS",
                source_workflow="matched-fp32-v2", clock="concurrent pair active elapsed",
                ensembles_per_recorded_workload=2, walkers_per_ensemble=32,
                transitions_per_ensemble=TOTAL, timing_records=1,
                aggregation="one complete matched timing run; no timing-replicate SD",
                recorded_seconds=elapsed, sd_seconds=None,
                milliseconds_per_transition=elapsed * 1000 / TOTAL,
                sd_milliseconds_per_transition=None, recorded_workload_hours=elapsed / 3600,
                sd_recorded_workload_hours=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    archive = options.archive.resolve()
    output = options.output_dir.resolve()
    stem = "figure5_mcmc_timing_with_sccm"
    outputs = {kind: output / f"{stem}.{ext}" for kind, ext in (
        ("pdf", "pdf"), ("png", "png"), ("csv", "csv"), ("json", "json"),
        ("manifest", "manifest.json"),
    )}
    timing_data = archive / "data/figure5"
    historical_root = timing_data / "historical_sccm"
    historical = historical_root / "figure5_mcmc_timing.csv"
    summary_path = timing_data / "matched_timing/summary.json"
    sccm_paths = [historical_root / f"ensemble_{e:03d}/checkpoint.json" for e in (0, 1)]
    ledgers = {m: timing_data / f"matched_timing/{m}.json" for m in ("dnn", "pdr")}
    config_path = archive / "configs/mcmc.matched-fp32.toml"
    reference = json.loads((archive / "provenance/figure5/figure5_provenance_validation.json").read_text())
    bindings = {
        historical: "outputs/manuscript/tables/figure5_mcmc_timing.csv",
        config_path: "configs/mcmc.matched-fp32.toml",
    }
    for i, path in enumerate(sccm_paths):
        bindings[path] = f"outputs/mcmc/production_v1/chains/full_except_acc/sccm/ensemble_{i:03d}/checkpoint.json"
    timing_root = "outputs/mcmc/matched_fp32_v2/chains/full_except_acc/matched_timing"
    for path in [summary_path, *ledgers.values()]:
        bindings[path] = f"{timing_root}/{path.name}"
    before = {str(p.relative_to(archive)): sha256_file(p) for p in bindings}
    derivatives = json.loads((archive / "provenance/figure5/distribution_bindings.json").read_text())
    for path, key in bindings.items():
        expected = reference["numerical_input_sha256"][key]
        derivative = derivatives.get(key)
        if derivative is not None:
            if derivative["original_sha256"] != expected:
                raise ValueError(f"Figure 5 provenance linkage mismatch: {key}")
            expected = derivative["distribution_sha256"]
        if before[str(path.relative_to(archive))] != expected:
            raise ValueError(f"Archived Figure 5 input hash mismatch: {key}")
    output.mkdir(parents=True, exist_ok=False)
    with historical.open(newline="") as stream:
        historical_rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text())
    rows = [sccm_statistics(historical_rows, [json.loads(p.read_text()) for p in sccm_paths])]
    for method in ("dnn", "pdr"):
        assert summary[method]["config_sha256"] == sha256_file(config_path)
        rows.append(matched_statistics(method, summary[method], json.loads(ledgers[method].read_text())))
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.35))
    for axis, value, error, ylabel, title in (
        (axes[0], "milliseconds_per_transition", "sd_milliseconds_per_transition", "Wall-clock time (ms, log scale)", "(a) MCMC transition time"),
        (axes[1], "recorded_workload_hours", "sd_recorded_workload_hours", "Wall-clock time (h, log scale)", "(b) MCMC workload time"),
    ):
        heights = [r[value] for r in rows]
        bars = axis.bar(np.arange(3), heights, color=[COLORS[r["method"]] for r in rows])
        axis.errorbar([0], [heights[0]], yerr=[rows[0][error]], fmt="none", ecolor="black", capsize=4)
        np.testing.assert_array_equal([bar.get_height() for bar in bars], heights)
        axis.set_xticks(np.arange(3), ["SCCM", "DNN", "CS"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        for i, height in enumerate(heights):
            top = height + (rows[0][error] if i == 0 else 0)
            annotation = f"{height:.3g}"
            if value == "milliseconds_per_transition" and i == 0:
                annotation = f"{float(annotation):.0f}"
            axis.annotate(annotation, (i, top), xytext=(0, 6),
                          textcoords="offset points", ha="center", va="bottom", fontsize=10)
        axis.set_ylim(top=max(heights[0] + rows[0][error], *heights[1:]) * 3)
    figure.tight_layout()
    for key in ("pdf", "png"):
        figure.savefig(outputs[key], bbox_inches="tight", dpi=300)
    plt.close(figure)
    if outputs["csv"].exists():
        with outputs["csv"].open(newline="") as stream:
            assert list(csv.DictReader(stream)) == [
                {k: str(v) if v is not None else "" for k, v in row.items()} for row in rows
            ], "Saved timing CSV differs; refusing to change data"
    else:
        with outputs["csv"].open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {"case": "full_except_acc", "burn_in": 30000, "retained": 300000,
               "statistics": rows, "caption_note": NOTE, "comparability_caveat": CAVEAT,
               "normalization": "1000 * recorded_seconds / 330000; hours = recorded_seconds / 3600",
               "second_panel": "measured recorded-workload hours; not a common resource-hours-per-ensemble metric",
               "error_bars": "SCCM only: sample SD of two single-ensemble productive clocks; DNN/CS: none (one paired timing run each)",
               "cs_over_dnn_matched_ratio": rows[2]["recorded_seconds"] / rows[1]["recorded_seconds"]}
    if outputs["json"].exists():
        assert json.loads(outputs["json"].read_text()) == payload, "Saved timing JSON differs"
    else:
        outputs["json"].write_text(json.dumps(payload, indent=2) + "\n")
    after = {str(p.relative_to(archive)): sha256_file(p) for p in bindings}
    assert after == before, "A saved source changed during plotting"
    manifest = {"artifact": "additional-figure5-timing-with-sccm", "artifact_role": "saved-timing-presentation-only",
                "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": "replot_figure5_with_sccm.py --archive ARCHIVE --output-dir OUTPUT",
                "source_sha256": before, "outputs": {k: p.name for k, p in outputs.items()},
                "output_sha256": {p.name: sha256_file(p) for k, p in outputs.items() if k != "manifest"},
                "saved_sources_unchanged": True, "experiments_executed": [],
                "style": "historical Figure 5: serif typography, black/red/blue bars, two log-scale panels; MCMC labels and value annotations; workload caveat in metadata only",
                **payload}
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: str(p) for k, p in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
