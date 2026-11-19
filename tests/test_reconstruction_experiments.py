"""Regression checks for statistical tensor reconstruction experiments."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return lambda name: importlib.import_module(f'experiments.{name}')


@pytest.mark.parametrize('name', ['consolidated_experiment', 'consolidated_experiment_with_n01'])
def test_paired_permutation_test_has_exact_small_sample_pvalue(experiment, name):
    module = experiment(name)
    statistic, pvalue = module.permutation_test([1, 2, 3], [0, 0, 0], n_resamples=8)
    assert statistic == 2
    assert pvalue == 0.25  # Two extreme sign assignments among the eight possible assignments.
    assert module.permutation_test([1, 2, 3], [1, 2, 3], n_resamples=8) == (0, 1)


def test_reconstruction_comparison_is_seeded(experiment):
    module = experiment('consolidated_experiment')
    result = module.run_experiment_with_initialization((3, 3), 'n01', 6, seed=8, n_resamples=64)
    assert result == module.run_experiment_with_initialization((3, 3), 'n01', 6,
                                                              seed=8, n_resamples=64)
    assert 0 < result['Permutation Test']['p-value'] <= 1
    assert result != module.run_experiment_with_initialization((3, 3), 'n01', 6,
                                                              seed=9, n_resamples=64)


def test_loss_experiment_draws_distinct_reproducible_comparisons(experiment, monkeypatch):
    module = experiment('loss_experiment')
    comparisons = []
    original = module.compute_loss

    def record(T, S):
        comparisons.append(S.copy())
        return original(T, S)

    monkeypatch.setattr(module, 'compute_loss', record)
    first = module.run_experiment((3, 3), 6, seed=8)
    random_arrays = comparisons[::2]
    assert len({array.tobytes() for array in random_arrays}) == 6
    comparisons.clear()
    assert first == module.run_experiment((3, 3), 6, seed=8)
    for a, b in zip(random_arrays, comparisons[::2]):
        np.testing.assert_array_equal(a, b)


def test_loss_import_does_not_create_files_or_reconfigure_logging(experiment, monkeypatch, tmp_path):
    import logging

    monkeypatch.chdir(tmp_path)
    handlers = logging.root.handlers[:]
    module = experiment('loss_experiment')
    importlib.reload(module)
    assert logging.root.handlers == handlers
    assert list(tmp_path.iterdir()) == []
