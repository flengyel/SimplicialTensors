"""Typed architecture-boundary operations for feed-forward networks.

Unlike the diagonal tensor boundary, the incidence boundary of a network's
directed architecture does not identify the row and column labels of a
weight matrix.  For squared edge weights, its restriction to hidden nodes is
the familiar incoming-minus-outgoing synaptic balance residual.

This module is deliberately NumPy-only.  It provides diagnostics and exact
function-preserving positive rescalings without adding a neural-network
framework dependency to the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_network(
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    arrays = [np.asarray(weight) for weight in weights]
    if len(arrays) < 2:
        raise ValueError("at least two weight matrices are required")
    if any(weight.ndim != 2 for weight in arrays):
        raise ValueError("every weight must be a matrix")
    for left, right in zip(arrays, arrays[1:]):
        if left.shape[0] != right.shape[1]:
            raise ValueError(
                "adjacent weights have incompatible hidden widths: "
                f"{left.shape} followed by {right.shape}"
            )

    if biases is None:
        return arrays, None
    bias_arrays = [np.asarray(bias) for bias in biases]
    if len(bias_arrays) != len(arrays):
        raise ValueError("there must be one bias vector per weight matrix")
    for weight, bias in zip(arrays, bias_arrays):
        if bias.shape != (weight.shape[0],):
            raise ValueError(
                f"bias shape {bias.shape} is incompatible with weight shape {weight.shape}"
            )
    return arrays, bias_arrays


def hidden_balance_residuals(
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray] | None = None
) -> list[np.ndarray]:
    """Return architecture-boundary residuals at every hidden layer.

    For hidden neuron ``v``, the residual is

    ``sum_{e enters v} weight_e**2 - sum_{e leaves v} weight_e**2``.

    A hidden bias is treated as an edge entering from a fixed constant node.
    Output biases do not enter a hidden-node residual.
    """

    matrices, vectors = _validate_network(weights, biases)
    residuals: list[np.ndarray] = []
    for layer in range(len(matrices) - 1):
        incoming = np.sum(np.abs(matrices[layer]) ** 2, axis=1)
        if vectors is not None:
            incoming = incoming + np.abs(vectors[layer]) ** 2
        outgoing = np.sum(np.abs(matrices[layer + 1]) ** 2, axis=0)
        residuals.append(incoming - outgoing)
    return residuals


def hidden_balance_energy(
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray] | None = None
) -> float:
    """Return ``0.5`` times the squared hidden architecture boundary norm."""

    return 0.5 * sum(
        float(np.vdot(residual, residual).real)
        for residual in hidden_balance_residuals(weights, biases)
    )


def hidden_balance_gradients(
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Return exact gradients of :func:`hidden_balance_energy`."""

    matrices, vectors = _validate_network(weights, biases)
    residuals = hidden_balance_residuals(matrices, vectors)
    weight_gradients = [np.zeros_like(weight) for weight in matrices]
    bias_gradients = None if vectors is None else [np.zeros_like(bias) for bias in vectors]

    for layer, residual in enumerate(residuals):
        weight_gradients[layer] += 2.0 * residual[:, None] * matrices[layer]
        weight_gradients[layer + 1] -= 2.0 * matrices[layer + 1] * residual[None, :]
        if bias_gradients is not None and vectors is not None:
            bias_gradients[layer] += 2.0 * residual * vectors[layer]
    return weight_gradients, bias_gradients


def equinormalize_relu_network(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray] | None = None,
    *,
    max_sweeps: int = 25,
    tolerance: float = 1e-10,
    eps: float = 1e-15,
    max_scale: float = 1e6,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Balance hidden nodes by positive function-preserving ReLU rescaling.

    Scaling all incoming parameters of a ReLU neuron by ``c > 0`` and its
    outgoing weights by ``1/c`` leaves the represented function unchanged.
    The locally optimal scale is the fourth root of the ratio of outgoing to
    incoming squared norm.  Repeated coordinate sweeps converge to the
    architecture-balanced representative under ordinary nonzero conditions.
    """

    if max_sweeps < 0:
        raise ValueError("max_sweeps must be non-negative")
    if tolerance < 0 or eps <= 0 or max_scale < 1:
        raise ValueError("invalid tolerance, eps, or max_scale")
    matrices, vectors = _validate_network(weights, biases)
    dtype = np.result_type(*(matrix.dtype for matrix in matrices), np.float64)
    balanced_weights: list[np.ndarray] = [
        matrix.astype(dtype, copy=True) for matrix in matrices
    ]
    balanced_biases: list[np.ndarray] | None = (
        None
        if vectors is None
        else [vector.astype(dtype, copy=True) for vector in vectors]
    )

    for _ in range(max_sweeps):
        for layer in range(len(balanced_weights) - 1):
            incoming = np.sum(np.abs(balanced_weights[layer]) ** 2, axis=1)
            if balanced_biases is not None:
                incoming = incoming + np.abs(balanced_biases[layer]) ** 2
            outgoing = np.sum(np.abs(balanced_weights[layer + 1]) ** 2, axis=0)
            scale = np.power((outgoing + eps) / (incoming + eps), 0.25)
            scale = np.clip(scale, 1.0 / max_scale, max_scale)
            balanced_weights[layer] *= scale[:, None]
            if balanced_biases is not None:
                balanced_biases[layer] *= scale
            balanced_weights[layer + 1] /= scale[None, :]

        residual_norm = np.sqrt(
            sum(
                float(np.vdot(residual, residual).real)
                for residual in hidden_balance_residuals(
                    balanced_weights, balanced_biases
                )
            )
        )
        edge_energy = np.sqrt(
            sum(float(np.vdot(weight, weight).real) for weight in balanced_weights)
        )
        if residual_norm <= tolerance * max(edge_energy**2, eps):
            break

    return balanced_weights, balanced_biases


def path_product_tensor(incoming: np.ndarray, outgoing: np.ndarray) -> np.ndarray:
    """Return all two-edge path products through one hidden layer.

    ``incoming`` has shape ``(hidden, input)`` and ``outgoing`` has shape
    ``(output, hidden)``.  The result has shape ``(output, hidden, input)``.
    It is invariant under positive hidden-neuron gauge rescaling and is merely
    relabeled under hidden permutations.
    """

    incoming = np.asarray(incoming)
    outgoing = np.asarray(outgoing)
    if incoming.ndim != 2 or outgoing.ndim != 2:
        raise ValueError("incoming and outgoing weights must be matrices")
    if incoming.shape[0] != outgoing.shape[1]:
        raise ValueError("incoming and outgoing hidden widths must agree")
    return np.einsum("oh,hi->ohi", outgoing, incoming)


def path_diamond_energy(incoming: np.ndarray, outgoing: np.ndarray) -> float:
    """Return a permutation- and positive-gauge-invariant path discrepancy.

    This is the squared variation, across hidden vertices, of two-edge path
    products.  Up to a constant it equals the sum of squared differences over
    every pair of parallel two-edge paths (the boundaries of path diamonds).
    """

    products = path_product_tensor(incoming, outgoing)
    centered = products - np.mean(products, axis=1, keepdims=True)
    return 0.5 * float(np.vdot(centered, centered).real)
