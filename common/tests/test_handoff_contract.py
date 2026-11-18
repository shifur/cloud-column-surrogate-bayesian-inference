from pathlib import Path

from cloud_column_reproduction.mcmc.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_mcmc_template_records_modern_production_contract() -> None:
    config = load_config(
        ROOT / "configs/mcmc.template.toml", validate_registered_artifacts=False
    )
    sampler = config.raw["sampler"]
    assert config.study_role == "production-study"
    assert sampler["coordinate_system"] == "box-logit"
    assert sampler["burn_in_steps"] == 30_000
    assert sampler["production_steps"] == 300_000
    components = sampler["move_parameters"]["components"]
    assert [(entry["name"], entry["weight"]) for entry in components] == [
        ("emcee.moves.DEMove", 0.8),
        ("emcee.moves.DESnookerMove", 0.2),
    ]
    assert config.masks["radiation_only"].tolist() == [0, 0, 0, 0, 1, 1]
    assert config.masks["full_except_acc"].tolist() == [1, 0, 1, 1, 1, 1]
    surrogates = config.raw["surrogates"]
    assert "training_data" not in surrogates
    for method in ("dnn", "pdr"):
        assert surrogates[method]["training_data"].startswith("SET_")
        assert len(surrogates[method]["training_data_sha256"]) > 0


def test_scratch_symlink_paths_remain_logical_for_mcmc_binding() -> None:
    prepare_source = (ROOT / "scripts/prepare_mcmc.py").read_text(encoding="utf-8")
    fitting_source = (
        ROOT / "src/cloud_column_reproduction/figure1/fitting.py"
    ).read_text(encoding="utf-8")

    # MCMC selection must not follow outputs/ into Scratch before making
    # repository-relative provenance paths.
    assert "path.absolute().relative_to(ROOT)" in prepare_source
    assert "figure1_output.absolute()" in prepare_source

    # Figure 1 provenance must preserve logical data/ paths rather than
    # following the data -> Scratch symlink.
    assert 'str(Path(training_path).resolve())' not in fitting_source
    assert 'str(Path(validation_path).resolve())' not in fitting_source
    assert 'str(Path(evaluation_path).resolve())' not in fitting_source

    assert 'str(Path(training_path).absolute())' in fitting_source
    assert 'str(Path(validation_path).absolute())' in fitting_source
    assert 'str(Path(evaluation_path).absolute())' in fitting_source
