from pathlib import Path

from cloud_column_reproduction.figure1.config import load_config, work_plan

ROOT = Path(__file__).resolve().parents[1]


def test_production_contract_is_q23_rqmc() -> None:
    config = load_config(ROOT / "configs/figure1.toml")
    assert config.is_rqmc_primary
    assert config.train_sizes == tuple(range(100, 5001, 100))
    assert config.experiment["trials"] == 20
    assert config.cs["hyperbolic_cross_order"] == 23
    assert config.cs["expected_basis_size"] == 7569
    assert config.dnn["batch_mode"] == "full"
    assert config.dnn["epochs"] == 50000
    assert config.dnn["validation_patience"] == "infinite"


def test_production_work_plan() -> None:
    plan = work_plan(load_config(ROOT / "configs/figure1.toml"))
    assert plan["trials"] == 20
    assert plan["training_sizes"] == 50
    assert plan["dnn_fits"] == 1000
    assert plan["cs_scalar_fits"] == 6000
    assert plan["rqmc_replications"] == 10
    assert plan["rqmc_points_per_replication"] == 16384
