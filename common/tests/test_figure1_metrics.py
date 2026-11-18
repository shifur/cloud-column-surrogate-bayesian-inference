from pathlib import Path

import numpy as np

from cloud_column_reproduction.figure1.config import load_config
from cloud_column_reproduction.figure1.metrics import (
    MetricRow,
    summarize_metrics,
    validate_metrics,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def complete_metric_rows():
    config = load_config(ROOT / "configs" / "figure1.toml")
    rows = [
        MetricRow(
            trial=trial,
            train_size=size,
            output=output,
            method=method,
            relative_l2=(1.0 + trial / 100.0) / np.sqrt(size),
        )
        for trial in range(config.experiment["trials"])
        for size in config.train_sizes
        for output in config.outputs
        for method in config.methods
    ]
    return config, rows


def test_complete_metric_table_is_summarized_and_written(tmp_path: Path) -> None:
    config, rows = complete_metric_rows()
    validated = validate_metrics(rows, config)
    assert len(validated) == 12000

    summary = summarize_metrics(validated, config)
    assert set(summary) == {
        (output, method) for output in config.outputs for method in config.methods
    }
    means, standard_deviations = summary[("PCP", "CS")]
    assert means.shape == (50,)
    assert standard_deviations.shape == (50,)
    assert np.all(standard_deviations > 0.0)

    output = tmp_path / "summary.csv"
    write_summary(output, summary, config)
    assert output.read_text(encoding="utf-8").count("\n") == 601


def test_duplicate_metric_row_is_rejected() -> None:
    config, rows = complete_metric_rows()
    duplicate = [*rows, rows[0]]
    try:
        validate_metrics(duplicate, config)
    except ValueError as error:
        assert "Duplicate metric row" in str(error)
    else:
        raise AssertionError("duplicate metric was accepted")

