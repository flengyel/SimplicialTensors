"""Invalid inputs must not count as successful horn checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _experiment(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index", [-1, 10])
def test_dimension_bound_does_not_hide_invalid_horn(index):
    module = _experiment("horn_map_reduce")
    with pytest.raises(ValueError, match="horn_j"):
        module.compute_missing_indices_dask((10, 10), index)


@pytest.mark.parametrize("shape", [(0, 2), (-1, 3), (2.5, 3)])
def test_horn_coordinates_reject_invalid_shapes(shape):
    module = _experiment("horn_map_reduce")
    with pytest.raises(ValueError, match="shape"):
        module.compute_missing_indices_dask(shape, 0)


def test_horn_coordinate_examples():
    dask = pytest.importorskip("dask")
    module = _experiment("horn_map_reduce")
    with dask.config.set(scheduler="synchronous"):
        assert module.compute_missing_indices_dask((2, 2), 1) == {
            (0, 0), (0, 1), (1, 0)
        }
        assert module.compute_missing_indices_dask((4, 4, 4), 0) == {
            (1, 2, 3), (1, 3, 2), (2, 1, 3),
            (2, 3, 1), (3, 1, 2), (3, 2, 1)
        }


@pytest.mark.parametrize("degeneracy,horn", [(-1, 0), (2, 0), (0, -1), (0, 3)])
def test_degenerate_recovery_does_not_skip_invalid_indices(degeneracy, horn):
    module = _experiment("verify_degenerate_preference")
    with pytest.raises((IndexError, ValueError)):
        module.verify_degenerate_preference(np.array([[0, 1], [0, 0]]), degeneracy, horn)
