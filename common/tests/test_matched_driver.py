import importlib.util
from pathlib import Path
import threading

from cloud_column_reproduction.mcmc.config import load_config


def test_both_methods_request_the_same_broker_and_advance_ensembles_concurrently(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("matched_driver", root / "scripts/run_mcmc_surrogates.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    config = load_config(root / "configs/mcmc.template.toml", validate_registered_artifacts=False)
    calls = []
    monkeypatch.setattr(driver, "build_runtime_evaluator", lambda *args, **kwargs: calls.append(kwargs))
    for method in ("dnn", "pdr"):
        driver._shared_runtime(config, "full_except_acc", method, 0.002)
    for call in calls:
        assert call["gpu_microbatch_requests"] == 2
        assert call["gpu_microbatch_window_seconds"] == 0.002
        assert call["include_prediction_blobs"] is False
    for method in ("dnn", "pdr"):
        barrier = threading.Barrier(2, timeout=5)
        observed = []
        runtime = type("Runtime", (), {"resources": ()})()
        def run(config, unit, **kwargs):
            observed.append(unit.ensemble_id)
            assert kwargs["runtime_override"] is runtime
            assert kwargs["defer_finalization"] is True
            barrier.wait()
            return {"ensemble": unit.ensemble_id}
        monkeypatch.setattr(driver, "run_unit", run)
        driver._advance_pair(config, tmp_path, "full_except_acc", method, runtime,
                             progress_every=1000, maximum_blocks={0: 2, 1: 2})
        assert sorted(observed) == [0, 1]
