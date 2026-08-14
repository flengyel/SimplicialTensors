"""Tests for typed neural-architecture boundary operations."""

from __future__ import annotations

import numpy as np

from simplicial_tensors.architecture_ops import (
    equinormalize_relu_network,
    hidden_balance_energy,
    hidden_balance_gradients,
    hidden_balance_residuals,
    path_diamond_energy,
    path_product_tensor,
)


def _network(seed: int = 0):
    rng = np.random.default_rng(seed)
    weights = [rng.normal(size=(5, 7)), rng.normal(size=(4, 5)), rng.normal(size=(3, 4))]
    biases = [rng.normal(size=5), rng.normal(size=4), rng.normal(size=3)]
    return weights, biases


def _relu_forward(x, weights, biases):
    value = x
    for weight, bias in zip(weights[:-1], biases[:-1]):
        value = np.maximum(value @ weight.T + bias, 0.0)
    return value @ weights[-1].T + biases[-1]


def test_hidden_balance_gradient_matches_finite_difference() -> None:
    weights, biases = _network(10)
    weight_gradients, bias_gradients = hidden_balance_gradients(weights, biases)
    assert bias_gradients is not None
    epsilon = 1e-6

    for layer, index in ((0, (2, 3)), (1, (1, 2)), (2, (0, 1))):
        plus = [weight.copy() for weight in weights]
        minus = [weight.copy() for weight in weights]
        plus[layer][index] += epsilon
        minus[layer][index] -= epsilon
        numeric = (
            hidden_balance_energy(plus, biases) - hidden_balance_energy(minus, biases)
        ) / (2 * epsilon)
        assert np.isclose(numeric, weight_gradients[layer][index], rtol=2e-6, atol=2e-6)

    plus_biases = [bias.copy() for bias in biases]
    minus_biases = [bias.copy() for bias in biases]
    plus_biases[1][2] += epsilon
    minus_biases[1][2] -= epsilon
    numeric = (
        hidden_balance_energy(weights, plus_biases)
        - hidden_balance_energy(weights, minus_biases)
    ) / (2 * epsilon)
    assert np.isclose(numeric, bias_gradients[1][2], rtol=2e-6, atol=2e-6)


def test_balance_is_hidden_permutation_equivariant() -> None:
    weights, biases = _network(20)
    rng = np.random.default_rng(21)
    first = rng.permutation(5)
    second = rng.permutation(4)
    permuted_weights = [
        weights[0][first],
        weights[1][second][:, first],
        weights[2][:, second],
    ]
    permuted_biases = [biases[0][first], biases[1][second], biases[2].copy()]

    original_residuals = hidden_balance_residuals(weights, biases)
    permuted_residuals = hidden_balance_residuals(permuted_weights, permuted_biases)
    assert np.allclose(permuted_residuals[0], original_residuals[0][first])
    assert np.allclose(permuted_residuals[1], original_residuals[1][second])
    assert np.isclose(
        hidden_balance_energy(permuted_weights, permuted_biases),
        hidden_balance_energy(weights, biases),
    )


def test_equinormalization_preserves_relu_function_and_reduces_balance() -> None:
    weights, biases = _network(30)
    x = np.random.default_rng(31).normal(size=(20, 7))
    prediction = _relu_forward(x, weights, biases)
    initial_energy = hidden_balance_energy(weights, biases)
    balanced_weights, balanced_biases = equinormalize_relu_network(weights, biases)
    assert balanced_biases is not None

    assert np.allclose(
        _relu_forward(x, balanced_weights, balanced_biases),
        prediction,
        rtol=1e-11,
        atol=1e-11,
    )
    assert hidden_balance_energy(balanced_weights, balanced_biases) < 1e-14 * initial_energy


def test_path_diamond_diagnostic_respects_permutation_and_positive_gauge() -> None:
    rng = np.random.default_rng(40)
    incoming = rng.normal(size=(6, 8))
    outgoing = rng.normal(size=(4, 6))
    permutation = rng.permutation(6)
    scale = np.exp(rng.normal(size=6))

    permuted_incoming = incoming[permutation]
    permuted_outgoing = outgoing[:, permutation]
    scaled_incoming = scale[:, None] * incoming
    scaled_outgoing = outgoing / scale[None, :]

    products = path_product_tensor(incoming, outgoing)
    assert np.allclose(
        path_product_tensor(permuted_incoming, permuted_outgoing),
        products[:, permutation, :],
    )
    assert np.allclose(path_product_tensor(scaled_incoming, scaled_outgoing), products)
    assert np.isclose(
        path_diamond_energy(permuted_incoming, permuted_outgoing),
        path_diamond_energy(incoming, outgoing),
    )
    assert np.isclose(
        path_diamond_energy(scaled_incoming, scaled_outgoing),
        path_diamond_energy(incoming, outgoing),
    )
