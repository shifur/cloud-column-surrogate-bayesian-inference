from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sccm_is_external() -> None:
    assert not (ROOT / "vendor/sccm").exists()
    assert not [p for p in ROOT.rglob("*") if p.suffix in {".o", ".a", ".mod", ".x"}]


def test_external_sccm_inputs_if_supplied() -> None:
    import os
    import pytest

    directory = os.environ.get("SCCM_INSTALL_DIR")
    if not directory:
        pytest.skip("External SCCM installation not supplied")
    for filename in ("cloud_column_model.x", "namelist_3h_t30-180.f90", "trp.dat"):
        assert (Path(directory) / filename).is_file()


def test_no_generated_study_artifacts_are_packaged() -> None:
    for name in ("data", "outputs", "figures"):
        assert not (ROOT / name).exists()


def test_only_declared_production_entry_points_are_packaged() -> None:
    expected = {
        "figure1.py",
        "figure1_design_cache.py",
        "figure1_fit.py",
        "figure2_interpretability_4d.py",
        "figure7_corner_full_except_acc_contour.py",
        "figure_sensitivity_1d.py",
        "finalize_manuscript.py",
        "generate_figure4.py",
        "mcmc.py",
        "plot_appendix_posterior_variability_two_panel.py",
        "plot_mcmc_convergence.py",
        "prepare_candidate_mcmc.py",
        "prepare_mcmc.py",
        "redraw_release.py",
        "replot_figure5_with_sccm.py",
        "replot_sensitivity_1d_manuscript.py",
        "run_candidate_mcmc.py",
        "run_mcmc_surrogates.py",
        "sccm.py",
        "sccm_mpi_worker.py",
        "validate_figure1.py",
    }
    names = {path.name for path in (ROOT / "scripts").iterdir() if path.is_file()}
    assert names == expected


def test_gpu_batch_environment_supports_deterministic_cublas() -> None:
    gpu_template = (ROOT / "environment/pbs/study_gpu.pbs").read_text(
        encoding="utf-8"
    )
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in gpu_template

def test_clean_entry_points() -> None:
    for script in (ROOT / "scripts").glob("*.py"):
        assert "/" + "glade/" not in script.read_text()
        assert ".bak" not in script.name


def test_companion_archive_bindings_if_supplied() -> None:
    import os
    import pytest

    archive_root = os.environ.get("CLOUD_COLUMN_DATA_ARCHIVE")
    if not archive_root:
        pytest.skip("Companion scientific-data archive not supplied")
    archive = Path(archive_root)
    assert len(list((archive / "figures/Updated Figures").glob("*.pdf"))) == 9
    formal = archive / "formal_mcmc"
    chains = list(formal.glob("*/*/ensemble_*/chain.nc"))
    assert len(chains) == 12
    assert all(p.with_suffix(".manifest.json").is_file() for p in chains)
    assert len(list((formal / "diagnostics").glob("*/*/*.npz"))) == 48
    for path in ("configs/plot-release.toml", "environment/plot-environment.yml",
                 "data/figure1/metrics.csv", "data/figure4/timings.csv",
                 "data/figure5/matched_timing/summary.json",
                 "data/appendix-a1/appendix_candidate_posterior_variability.csv"):
        assert (archive / path).is_file(), path
