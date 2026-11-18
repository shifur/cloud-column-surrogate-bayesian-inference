from pathlib import Path
import re

from cloud_column_reproduction.mcmc.config import load_config
from cloud_column_reproduction.mcmc.production import planned_units, workload_summary

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_scope_contains_exactly_four_surrogate_chain_units(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "parameter_box.toml").write_bytes(
        (ROOT / "configs/parameter_box.toml").read_bytes()
    )
    text = (ROOT / "configs/mcmc.template.toml").read_text(encoding="utf-8")
    text = text.replace(
        "[parameters]",
        '[execution_scope]\nlikelihood_cases = ["full_except_acc"]\nmethods = ["dnn", "pdr"]\n\n[parameters]',
        1,
    )
    replacements = {
        "expected_likelihood_case_count = 2": "expected_likelihood_case_count = 1",
        "expected_method_count = 3": "expected_method_count = 2",
        "expected_ensemble_counts_by_method = { sccm = 2, dnn = 2, pdr = 2 }": "expected_ensemble_counts_by_method = { sccm = 0, dnn = 2, pdr = 2 }",
        "expected_chain_file_count = 12": "expected_chain_file_count = 4",
        "likelihood_case_count = 2": "likelihood_case_count = 1",
        "independent_ensemble_counts_by_method = { sccm = 2, dnn = 2, pdr = 2 }": "independent_ensemble_counts_by_method = { sccm = 0, dnn = 2, pdr = 2 }",
        "sccm_run_count = 4": "sccm_run_count = 0",
        "sccm_transition_proposals_total = 42240000": "sccm_transition_proposals_total = 0",
        "sccm_initial_state_evaluations_total = 128": "sccm_initial_state_evaluations_total = 0",
        "sccm_posterior_evaluations_upper_bound = 42240128": "sccm_posterior_evaluations_upper_bound = 0",
        "sccm_estimated_wall_seconds = 42240128.0": "sccm_estimated_wall_seconds = 0.0",
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new, 1)
    path = configs / "candidate.toml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path, validate_registered_artifacts=False)
    units = planned_units(config)
    assert config.execution_cases() == ("full_except_acc",)
    assert config.execution_methods() == ("dnn", "pdr")
    assert [(unit.method, unit.ensemble_id) for unit in units] == [
        ("dnn", 0),
        ("pdr", 0),
        ("dnn", 1),
        ("pdr", 1),
    ]
    summary = workload_summary(config)
    assert summary["unit_count"] == 4
    assert summary["methods"] == ["dnn", "pdr"]
