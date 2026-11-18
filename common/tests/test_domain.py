from pathlib import Path

import numpy as np
import pytest

from cloud_column_reproduction.figure1.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_box_round_trip_and_boundary_enforcement() -> None:
    config = load_config(ROOT / "configs" / "figure1.toml")
    box = config.parameter_box
    physical = np.asarray([box.lower, box.upper, (np.asarray(box.lower) + box.upper) / 2.0])
    canonical = box.to_canonical(physical)
    np.testing.assert_allclose(canonical[0], -1.0)
    np.testing.assert_allclose(canonical[1], 1.0)
    np.testing.assert_allclose(canonical[2], 0.0, atol=4.0 * np.finfo(float).eps)
    np.testing.assert_allclose(box.to_physical(canonical), physical)

    invalid = physical[1].copy()
    invalid[0] += 1.0
    with pytest.raises(ValueError, match="outside the authoritative parameter box"):
        box.to_canonical(invalid)
    with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
        box.to_physical(np.full(box.dimension, 1.01))


def test_figure1_uses_the_shared_parameter_box_file() -> None:
    config = load_config(ROOT / "configs" / "figure1.toml")
    assert config.parameter_box_path == (ROOT / "configs" / "parameter_box.toml").resolve()
    assert len(config.parameter_box.fingerprint) == 64
