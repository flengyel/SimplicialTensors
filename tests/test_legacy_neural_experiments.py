"""Regression checks for the repaired statistical and PyTorch entry points."""

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


@pytest.mark.parametrize('name', ['bdry_network_experiment1', 'boundary_augmented_network'])
def test_boundary_training_runs_through_backward(experiment, name):
    torch = pytest.importorskip('torch')
    module = experiment(name)
    torch.manual_seed(8)
    net = module.BoundaryAugmentedNet(4, [6, 6, 4, 4], 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    args = (net, torch.nn.MSELoss(), optimizer)
    if name == 'boundary_augmented_network':
        history = module.train_network(*args, None, 2, augment_type='boundary', batch_size=3, seed=8)
    else:
        history = module.train_network(*args, 2, batch_size=3, seed=8)
    assert np.all(np.isfinite(history))
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_boundary_and_random_updates_have_matched_relative_norms(experiment):
    torch = pytest.importorskip('torch')
    module = experiment('boundary_augmented_network')
    torch.manual_seed(8)
    boundary = module.BoundaryAugmentedNet(4, [6, 6, 4, 4], 2, boundary_scale=0.02)
    random = module.RandomTensorAugmentedNet(4, [6, 6, 4, 4], 2,
                                             random_tensor_scale=0.02, seed=8)
    random.load_state_dict(boundary.state_dict())
    before = {name: p.detach().clone() for name, p in boundary.named_parameters()}
    boundary.integrate_boundary()
    random.integrate_random_tensor()
    for net in (boundary, random):
        for name, parameter in net.named_parameters():
            if name in ('fc3.weight', 'fc4.weight', 'fc5.weight'):
                relative_norm = torch.linalg.vector_norm(parameter - before[name]) / torch.linalg.vector_norm(before[name])
                assert float(relative_norm.detach()) == pytest.approx(0.02, abs=1e-7)
            else:
                assert torch.equal(parameter, before[name])


def test_random_augmentation_advances_its_seeded_generator(experiment):
    torch = pytest.importorskip('torch')
    module = experiment('boundary_augmented_network')
    torch.manual_seed(8)
    net = module.RandomTensorAugmentedNet(4, [6, 6, 4, 4], 2,
                                         random_tensor_scale=0.02, seed=8)
    initial = {name: p.detach().clone() for name, p in net.named_parameters()}
    net.integrate_random_tensor()
    first = net.fc3.weight.detach().clone() - initial['fc3.weight']
    net.load_state_dict(initial)
    net.integrate_random_tensor()
    second = net.fc3.weight.detach() - initial['fc3.weight']
    assert not torch.allclose(first, second)


def test_loss_import_does_not_create_files_or_reconfigure_logging(experiment, monkeypatch, tmp_path):
    import logging

    monkeypatch.chdir(tmp_path)
    handlers = logging.root.handlers[:]
    module = experiment('loss_experiment')
    importlib.reload(module)
    assert logging.root.handlers == handlers
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('name,cls_name,augment_type', [
    ('bdry_network_experiment1', 'BoundaryAugmentedNet', 'boundary'),
    ('boundary_augmented_network', 'BoundaryAugmentedNet', 'boundary'),
    ('boundary_augmented_network', 'RandomTensorAugmentedNet', 'random'),
])
def test_scalar_output_has_no_boundary_intervention(experiment, name, cls_name, augment_type):
    torch = pytest.importorskip('torch')
    module = experiment(name)
    torch.manual_seed(8)
    net = getattr(module, cls_name)(4, [6, 6, 4, 4], 1)
    output_weight = net.fc5.weight.detach().clone()
    if augment_type == 'boundary':
        net.integrate_boundary()
    else:
        net.integrate_random_tensor()
    assert torch.equal(net.fc5.weight, output_weight)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    args = (net, torch.nn.MSELoss(), optimizer)
    if name == 'boundary_augmented_network':
        history = module.train_network(*args, None, 2, augment_type=augment_type, batch_size=3, seed=8)
    else:
        history = module.train_network(*args, 2, batch_size=3, seed=8)
    assert np.all(np.isfinite(history))
