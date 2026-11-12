"""Regression checks for bounded, import-safe experiment entry points."""

import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["examples_range_tensors", "shape_generator", "simplicial_permutation_group", "n_hypergroupoid_conjecture", "Gemini_hallicination"])
def test_import_does_not_run_experiments(name):
    result = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(EXPERIMENTS / (name + '.py'))!r})"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    assert result.stdout == ""
    assert result.stderr == ""


def test_bounded_shape_enumeration_covers_requested_degrees():
    module = _load("shape_generator")
    shapes = list(module.generate_shapes_by_dimension())
    expected = []
    for degree in range(2, 5):
        for axes in range(degree, 5):
            candidates = [
                offsets for offsets in itertools.product(range(3), repeat=axes)
                if min(offsets) == 0 and sum(offsets) <= 2
            ]
            for offsets in sorted(candidates, key=lambda value: (sum(value), value)):
                expected.append(tuple(degree + 1 + offset for offset in offsets))
    assert shapes == expected
    assert len(shapes) == 70
    assert {min(shape) - 1 for shape in shapes} == {2, 3, 4}
    assert list(module.generate_shapes_by_dimension(35)) == shapes[:35]
    assert min(shapes[34]) - 1 == 3
    assert list(module.generate_shapes_by_dimension(0)) == []
    assert len(list(module.generate_shapes_by_dimension(100))) == 70


@pytest.mark.parametrize("kwargs", [
    {"max_total_shapes": -1}, {"max_total_shapes": 1.5}, {"min_degree": 0},
    {"min_degree": 3, "max_degree": 2}, {"max_axes": 3}, {"max_offset_sum": -1},
])
def test_invalid_shape_bounds(kwargs):
    with pytest.raises(ValueError):
        list(_load("shape_generator").generate_shapes_by_dimension(**kwargs))


def test_cycle_deletion_matches_successor_permutation():
    module = _load("simplicial_permutation_group")
    for perm in itertools.permutations(range(4)):
        for removed in range(4):
            survivors = [i for i in range(4) if i != removed]
            expected = np.zeros((3, 3), dtype=int)
            for row, original in enumerate(survivors):
                successor = perm[original]
                if successor == removed:
                    successor = perm[removed]
                expected[row, survivors.index(successor)] = 1
            actual = module.face_matrix(module.perm_matrix(perm), removed)
            np.testing.assert_array_equal(actual, expected)
    assert module.check_permutation_faces(3) == 18


def test_range_examples_report_each_return_value(monkeypatch, capsys):
    module = _load("examples_range_tensors")
    monkeypatch.setattr(module, "print_range_tensor", lambda shape: None)
    monkeypatch.setattr(
        module, "reconstruct_range_tensor_from_horn",
        lambda shape, **kwargs: shape == (3, 3),
    )
    assert module.run_range_examples(((3, 3), (4, 4))) == [True, False]
    output = capsys.readouterr().out
    assert "shape (3, 3): True" in output
    assert "shape (4, 4): False" in output


def test_forced_degeneracy_prediction_uses_final_shape(monkeypatch):
    module = _load("n_hypergroupoid_conjecture")
    predicted_shapes = []

    def predict(shape, **kwargs):
        predicted_shapes.append(shape)
        return min(shape) - 1 > len(shape)

    monkeypatch.setattr(module, "n_hypergroupoid_conjecture", predict)
    assert module.run_n_hypergroupoid_conjecture(
        1, 4, force_degeneracy=True, single_shape=(3, 3), verbose=False,
    )
    assert predicted_shapes == [(4, 4)]


def test_boundary_rejection_stops_at_attempt_limit(monkeypatch):
    module = _load("n_hypergroupoid_conjecture")
    draws = []

    def draw(shape, **kwargs):
        draws.append(shape)
        return np.zeros(shape, dtype=int)

    monkeypatch.setattr(module, "random_tensor", draw)
    with pytest.raises(RuntimeError, match="3 attempts"):
        module.generate_tensor((3, 3), False, True, max_attempts=3)
    assert draws == [(3, 3)] * 3


def test_descent_rejection_stops_at_attempt_limit(monkeypatch):
    module = _load("n_hypergroupoid_conjecture")
    monkeypatch.setattr(module, "random_tensor", lambda shape, **kwargs: np.zeros(shape, dtype=int))
    with pytest.raises(RuntimeError, match="2 attempts"):
        module.verify_unique_down_to_order(shape=(3, 3), max_attempts=2)


def test_horn_experiment_seed_reproduces_samples():
    module = _load("n_hypergroupoid_conjecture")
    first_rng = np.random.default_rng(71)
    second_rng = np.random.default_rng(71)
    first = module.generate_tensor((3, 4), False, False, rng=first_rng)
    second = module.generate_tensor((3, 4), False, False, rng=second_rng)
    np.testing.assert_array_equal(first, second)
    next_sample = module.generate_tensor((3, 4), False, False, rng=first_rng)
    assert not np.array_equal(first, next_sample)


def test_impossible_boundary_filter_rejected_before_sampling(monkeypatch):
    module = _load("n_hypergroupoid_conjecture")

    def unexpected_draw(*args, **kwargs):
        pytest.fail("an impossible filter must be rejected before sampling")

    monkeypatch.setattr(module, "random_tensor", unexpected_draw)
    with pytest.raises(ValueError, match="simplicial degree 3 exceeds 2 axes"):
        module.generate_tensor((4, 4), False, True)
    with pytest.raises(ValueError, match="simplicial degree 7 exceeds 3 axes"):
        module.verify_unique_down_to_order(shape=(8, 10, 12))


def test_tensor_product_examples_execute_both_cases(capsys):
    results = _load("Gemini_hallicination").main()
    assert [result["shape"] for result in results] == [(3, 2, 2), (6, 7, 6, 7)]
    assert [result["degree"] for result in results] == [1, 5]
    assert [result["unique_fillers"] for result in results] == [False, True]
    assert all(result["compatible"] and result["included_faces_equal"] for result in results)
    assert results[1]["selected_filler_equal"]
    assert "no inner horns in degree one" in capsys.readouterr().out
