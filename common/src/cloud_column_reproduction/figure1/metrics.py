"""Strict tidy-table schema for Figure 1 paired-trial metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Figure1Config


REQUIRED_COLUMNS = ("trial", "train_size", "output", "method", "relative_l2")


@dataclass(frozen=True)
class MetricRow:
    trial: int
    train_size: int
    output: str
    method: str
    relative_l2: float


def read_metrics(path: str | Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(REQUIRED_COLUMNS):
            raise ValueError(
                f"Metric columns must be exactly {REQUIRED_COLUMNS}; got {reader.fieldnames}"
            )
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = MetricRow(
                    trial=int(raw["trial"]),
                    train_size=int(raw["train_size"]),
                    output=raw["output"],
                    method=raw["method"],
                    relative_l2=float(raw["relative_l2"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid metric row at line {line_number}: {error}") from error
            rows.append(row)
    return rows


def validate_metrics(rows: Iterable[MetricRow], config: Figure1Config) -> list[MetricRow]:
    materialized = list(rows)
    trials = range(int(config.experiment["trials"]))
    expected = {
        (trial, size, output, method)
        for trial in trials
        for size in config.train_sizes
        for output in config.outputs
        for method in config.methods
    }
    observed: dict[tuple[int, int, str, str], MetricRow] = {}
    for row in materialized:
        key = (row.trial, row.train_size, row.output, row.method)
        if key in observed:
            raise ValueError(f"Duplicate metric row: {key}")
        if not np.isfinite(row.relative_l2) or row.relative_l2 <= 0.0:
            raise ValueError(f"relative_l2 must be positive and finite for {key}")
        observed[key] = row

    missing = expected - observed.keys()
    unexpected = observed.keys() - expected
    if missing or unexpected:
        examples = []
        if missing:
            examples.append(f"missing={len(missing)} (example {sorted(missing)[0]})")
        if unexpected:
            examples.append(f"unexpected={len(unexpected)} (example {sorted(unexpected)[0]})")
        raise ValueError("Metric table does not match the config: " + "; ".join(examples))
    return materialized


def summarize_metrics(
    rows: Iterable[MetricRow], config: Figure1Config
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[list[float]]] = {
        (output, method): [[] for _ in config.train_sizes]
        for output in config.outputs
        for method in config.methods
    }
    size_index = {size: index for index, size in enumerate(config.train_sizes)}
    for row in rows:
        grouped[(row.output, row.method)][size_index[row.train_size]].append(row.relative_l2)

    ddof = int(config.plot["standard_deviation_ddof"])
    summary: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, values_by_size in grouped.items():
        values = np.asarray(values_by_size, dtype=float)
        summary[key] = (values.mean(axis=1), values.std(axis=1, ddof=ddof))
    return summary


def write_summary(
    path: str | Path,
    summary: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    config: Figure1Config,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("train_size", "output", "method", "mean_relative_l2", "std_relative_l2"))
        for output in config.outputs:
            for method in config.methods:
                means, standard_deviations = summary[(output, method)]
                for size, mean, standard_deviation in zip(
                    config.train_sizes, means, standard_deviations, strict=True
                ):
                    writer.writerow((size, output, method, f"{mean:.17g}", f"{standard_deviation:.17g}"))

