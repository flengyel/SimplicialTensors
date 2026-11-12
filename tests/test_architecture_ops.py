"""Tests for typed neural-architecture boundary operations."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int64, np.uint8, np.uint64, np.float32, np.float64]
)
def test_hidden_balance_gradients_promote_real_inputs(dtype) -> None:
    weights = [np.array([[1]], dtype=dtype), np.array([[2]], dtype=dtype)]
    original = [weight.copy() for weight in weights]

    gradients, bias_gradients = hidden_balance_gradients(weights)

    assert bias_gradients is None
    assert hidden_balance_energy(weights) == 4.5
    np.testing.assert_array_equal(hidden_balance_residuals(weights)[0], [-3.0])
    for actual, expected in zip(gradients, ([-6.0], [12.0])):
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual.ravel(), expected)
    for actual, expected in zip(weights, original):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_hidden_balance_promotes_before_squaring_weights_and_biases(dtype) -> None:
    weights = [np.array([[2**32]], dtype=dtype), np.array([[2**31]], dtype=dtype)]
    biases = [np.array([2**31], dtype=dtype), np.array([7], dtype=dtype)]

    gradients, bias_gradients = hidden_balance_gradients(weights, biases)

    np.testing.assert_array_equal(
        hidden_balance_residuals(weights, biases)[0], [float(2**64)]
    )
    assert hidden_balance_energy(weights, biases) == float(2**127)
    np.testing.assert_array_equal(gradients[0], [[float(2**97)]])
    np.testing.assert_array_equal(gradients[1], [[-float(2**96)]])
    assert bias_gradients is not None
    np.testing.assert_array_equal(bias_gradients[0], [float(2**96)])
    np.testing.assert_array_equal(bias_gradients[1], [0.0])


def test_complex_bias_gradient_preserves_imaginary_direction() -> None:
    weights = [np.array([[1.0]]), np.array([[2.0]])]
    biases = [np.array([1 + 2j]), np.array([0.0])]
    gradients, bias_gradients = hidden_balance_gradients(weights, biases)
    assert bias_gradients is not None
    assert all(np.iscomplexobj(value) for value in gradients + bias_gradients)
    np.testing.assert_allclose(gradients[0], [[4.0]], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gradients[1], [[-8.0]], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(bias_gradients[0], [4 + 8j], rtol=1e-14, atol=1e-14)
    epsilon = 1e-6
    plus = [bias.copy() for bias in biases]
    minus = [bias.copy() for bias in biases]
    plus[0][0] += epsilon * 1j
    minus[0][0] -= epsilon * 1j
    numeric = (
        hidden_balance_energy(weights, plus) - hidden_balance_energy(weights, minus)
    ) / (2 * epsilon)
    assert np.isclose(numeric, bias_gradients[0][0].imag, rtol=1e-8, atol=1e-8)


def test_integer_equinormalization_preserves_relu_function() -> None:
    weights = [np.array([[1, 2], [3, 4]]), np.array([[2, 1]])]
    biases = [np.array([1, -2]), np.array([3])]
    inputs = np.array([[2.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
    prediction = _relu_forward(inputs, weights, biases)

    balanced_weights, balanced_biases = equinormalize_relu_network(weights, biases)

    assert balanced_biases is not None
    assert all(value.dtype == np.float64 for value in balanced_weights + balanced_biases)
    np.testing.assert_allclose(
        _relu_forward(inputs, balanced_weights, balanced_biases),
        prediction, rtol=1e-12, atol=1e-12,
    )
    assert hidden_balance_energy(
        balanced_weights, balanced_biases
    ) < hidden_balance_energy(weights, biases)


def test_equinormalization_keeps_complex_bias_dtype() -> None:
    weights = [np.array([[1.0]]), np.array([[2.0]])]
    biases = [np.array([1 + 2j]), np.array([0.0])]
    balanced_weights, balanced_biases = equinormalize_relu_network(
        weights, biases, max_sweeps=0
    )
    assert balanced_biases is not None
    assert all(np.iscomplexobj(value) for value in balanced_weights + balanced_biases)
    for actual, expected in zip(balanced_biases, biases):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_path_products_promote_before_multiplication(dtype) -> None:
    incoming = np.array([[2**32], [2**31]], dtype=dtype)
    outgoing = np.array([[2**32, 2**31]], dtype=dtype)

    products = path_product_tensor(incoming, outgoing)

    assert products.dtype == np.float64
    np.testing.assert_array_equal(products, [[[float(2**64)], [float(2**62)]]])
    assert path_diamond_energy(incoming, outgoing) == float(9 * 2**122)
