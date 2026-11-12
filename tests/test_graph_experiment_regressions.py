"""Regressions for relabeling formulas and graph experiment entry points."""

import importlib.util
from itertools import permutations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_induced_relabeling_on_every_four_vertex_permutation():
    experiment = load_experiment("matrix_boundary_homotopy_identity")
    matrix = np.arange(16).reshape(4, 4)
    for sigma in permutations(range(4)):
        inverse = [sigma.index(i) for i in range(4)]
        relabeled = np.array([[matrix[inverse[r], inverse[c]] for c in range(4)]
                              for r in range(4)])
        expected_boundary = np.zeros((3, 3), dtype=int)
        for deleted in range(4):
            survivors = [j for j in range(4) if j != deleted]
            expected_face = relabeled[np.ix_(survivors, survivors)]
            original_survivors = [j for j in range(4) if j != inverse[deleted]]
            original_face = matrix[np.ix_(original_survivors, original_survivors)]
            Q = experiment.induced_subperm_matrix(sigma, deleted)
            np.testing.assert_array_equal(Q @ original_face @ Q.T, expected_face)
            expected_boundary += (-1) ** deleted * expected_face
        np.testing.assert_array_equal(experiment.conjugate_by_sigma(matrix, sigma), relabeled)
        np.testing.assert_array_equal(
            experiment.boundary_by_faces_via_sigma(matrix, sigma), expected_boundary)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("mod2", [False, True])
def test_relabeling_random_trials(directed, mod2):
    experiment = load_experiment("matrix_boundary_homotopy_identity")
    assert experiment.run_many_trials(trials=10, n=6, directed=directed, mod2=mod2, seed=0)


def test_relabeling_main_reports_failure(monkeypatch):
    experiment = load_experiment("matrix_boundary_homotopy_identity")
    monkeypatch.setattr(experiment, "run_many_trials", lambda **kwargs: False)
    assert experiment.main() == 1


def test_mask_boundary_preserves_entry_position_and_coefficients():
    experiment = load_experiment("random_zero_one_mask_cocycle_test")
    x = sp.Symbol("x")
    source = np.zeros((3, 3), dtype=object)
    source[0, 2] = x
    signed = experiment.boundary_difference_check(source, np.zeros_like(source), modulus=None)
    mod2 = experiment.boundary_difference_check(source, np.zeros_like(source), modulus=2)
    np.testing.assert_array_equal(signed["boundary_difference"], [[0, -x], [0, 0]])
    np.testing.assert_array_equal(mod2["boundary_difference"], [[0, x], [0, 0]])
    assert signed["ok"] and mod2["ok"]
    assert signed["nonzero_entries"] == 1


def test_petersen_identity_relabeling_compares_the_masked_source(monkeypatch):
    experiment = load_experiment("random_zero_one_mask_cocycle_test")
    monkeypatch.setattr(experiment, "random_permutation", lambda n, rng: list(range(n)))
    result = experiment.test_petersen_permutation_boundaries(num_trials=1)[0]
    assert result["ok"] and result["nonzero_entries"] == 0
    masked = experiment.SymbolicTensor((10, 10)).tensor * experiment.kneser_petersen_adjacency()
    expected = experiment._reduce_tensor(experiment.bdry(masked), 2)
    np.testing.assert_array_equal(result["boundary_a"], expected)


def test_mask_sampler_is_reproducible_without_global_rng_mutation():
    experiment = load_experiment("random_zero_one_mask_cocycle_test")
    np.random.seed(17)
    expected = np.random.random(3)
    np.random.seed(17)
    first = experiment.random_zero_one_matrix(5, seed=4)
    second = experiment.random_zero_one_matrix(5, seed=4)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_mask_main_propagates_failed_checks(monkeypatch):
    experiment = load_experiment("random_zero_one_mask_cocycle_test")
    monkeypatch.setattr(experiment, "test_petersen_permutation_boundaries",
                        lambda **kwargs: [{"ok": False}])
    monkeypatch.setattr(experiment, "test_random_zero_one_boundaries",
                        lambda **kwargs: [{"ok": True}])
    assert experiment.main(["--permutations", "1", "--masks", "1"]) == 1


def test_permutation_cli_help_reaches_argument_parser():
    result = subprocess.run([sys.executable, str(ROOT / "experiments/permutation_statistics.py"),
                             "--help"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--max-powers" in result.stdout
    assert "--tolerance" in result.stdout


def test_permutation_cli_forwards_analysis_parameters(monkeypatch, tmp_path):
    experiment = load_experiment("permutation_statistics")
    calls = []
    monkeypatch.setattr(experiment, "test_permutation_matrices",
                        lambda sizes, **kwargs: calls.append((sizes, kwargs)) or [])
    monkeypatch.setattr(experiment, "generate_visualizations", lambda *args: None)
    experiment.main(["--sizes", "3", "5", "--seed", "8", "--max-powers", "7",
                     "--tolerance", "0.001", "--output-dir", str(tmp_path)])
    assert calls == [([3, 5], {"verbose": False, "seed": 8, "max_powers": 7, "tolerance": 0.001})]


def test_permutation_comparisons_use_the_absolute_tolerance():
    experiment = load_experiment("permutation_statistics")
    assert not experiment.is_idempotent(np.array([[1 + 1e-6]]), tolerance=1e-8)
    assert experiment.is_idempotent(np.array([[1 + 1e-6]]), tolerance=1e-5)
    cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    assert experiment.check_cycles(cycle, max_powers=2) == (False, 0, 0)
    assert experiment.check_cycles(cycle, max_powers=3) == (True, 0, 3)
