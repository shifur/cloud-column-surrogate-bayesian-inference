from pathlib import Path

import numpy as np

from cloud_column_reproduction.mcmc.chain import (
    read_chain_netcdf,
    write_chain_netcdf_from_blocks,
)
from cloud_column_reproduction.mcmc.transforms import (
    log_abs_det_jacobian,
    logit_to_physical,
)


def test_logit_blocks_finalize_as_physical_chain_and_physical_density(tmp_path: Path) -> None:
    generator = np.random.default_rng(92)
    lower = np.array([-2.0, 1.0])
    upper = np.array([4.0, 9.0])
    z = generator.normal(size=(8, 32, 2))
    physical = logit_to_physical(z, lower, upper)
    log_likelihood = generator.normal(size=(8, 32))
    jacobian = log_abs_det_jacobian(z, lower, upper)
    block = tmp_path / "production.npz"
    np.savez_compressed(
        block,
        position=z,
        log_prior=jacobian,
        log_likelihood=log_likelihood,
        log_posterior=log_likelihood + jacobian,
        evaluation_status=np.zeros((8, 32), dtype=np.int8),
        accepted=np.ones((8, 32), dtype=np.int8),
    )
    destination = tmp_path / "chain.nc"
    write_chain_netcdf_from_blocks(
        destination,
        (block,),
        parameter_names=("a", "b"),
        initial_positions=physical[0],
        post_burn_positions=physical[1],
        metadata={"test": "box-logit-finalization"},
        sampler_coordinate_system="box-logit",
        parameter_lower=lower,
        parameter_upper=upper,
    )
    result = read_chain_netcdf(destination)
    np.testing.assert_allclose(result["position"], physical)
    np.testing.assert_array_equal(result["log_prior"], np.zeros((8, 32)))
    np.testing.assert_array_equal(result["log_likelihood"], log_likelihood)
    np.testing.assert_array_equal(result["log_posterior"], log_likelihood)


def test_logit_transform_preserves_compact_surrogate_blobs() -> None:
    from cloud_column_reproduction.mcmc.transforms import transformed_log_probability

    lower = np.asarray([0.0, -2.0])
    upper = np.asarray([1.0, 3.0])

    def compact(values):
        rows = [(1.5, 0.0, 1.5, 0) for _ in np.atleast_2d(values)]
        return rows if np.asarray(values).ndim == 2 else rows[0]

    transformed = transformed_log_probability(compact, lower, upper)
    values = np.asarray([[0.0, 0.0], [1.0, -1.0]])
    result = transformed(values)
    assert len(result) == 2
    assert all(len(row) == 4 for row in result)
    expected = log_abs_det_jacobian(values, lower, upper)
    np.testing.assert_allclose([row[0] for row in result], 1.5 + expected)
    np.testing.assert_allclose([row[1] for row in result], expected)
