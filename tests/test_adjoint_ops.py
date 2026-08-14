"""Tests for adjoint boundary operators and their exact spectral filters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from simplicial_tensors.adjoint_ops import (
    bdry_adjoint,
    boundary_homeostasis_feedback,
    boundary_homeostasis_gradient,
    boundary_pseudoinverse,
    boundary_range_projection,
    boundary_sobolev_filter,
    boundary_spectral_values,
    exact_cycle_projection,
    face_adjoint,
    lower_hodge_laplacian,
    project_to_boundary,
)
from simplicial_tensors.tensor_ops import bdry, face


SHAPES = ((3, 3), (4, 5), (3, 3, 3), (4, 3, 3, 3))


def _boundary_matrix(shape: tuple[int, ...]) -> np.ndarray:
    """Construct a small dense boundary matrix for independent checks."""

    input_size = int(np.prod(shape))
    output_shape = tuple(size - 1 for size in shape)
    output_size = int(np.prod(output_shape))
    matrix = np.zeros((output_size, input_size))
    for column in range(input_size):
        basis = np.zeros(input_size)
        basis[column] = 1.0
        matrix[:, column] = bdry(basis.reshape(shape)).reshape(-1)
    return matrix


@pytest.mark.parametrize("shape", SHAPES)
def test_face_adjoint_identity(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(100 + sum(shape))
    w = rng.normal(size=shape)
    y = rng.normal(size=tuple(size - 1 for size in shape))
    for i in range(min(shape)):
        left = np.vdot(face(w, i), y)
        right = np.vdot(w, face_adjoint(y, shape, i))
        assert np.allclose(left, right, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_boundary_adjoint_identity(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(200 + sum(shape))
    w = rng.normal(size=shape)
    y = rng.normal(size=tuple(size - 1 for size in shape))
    assert np.allclose(
        np.vdot(bdry(w), y),
        np.vdot(w, bdry_adjoint(y, shape)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_homeostasis_gradient_reduces_energy(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(300 + sum(shape))
    w = rng.normal(size=shape)
    gradient = boundary_homeostasis_gradient(w)
    if np.linalg.norm(gradient) <= 1e-12:
        pytest.skip("sample is already a cycle")

    energy = 0.5 * np.linalg.norm(bdry(w)) ** 2
    step = 1.0
    for _ in range(30):
        candidate_energy = 0.5 * np.linalg.norm(bdry(w - step * gradient)) ** 2
        if candidate_energy < energy:
            break
        step *= 0.5
    assert candidate_energy < energy


def test_homeostasis_feedback_is_norm_matched() -> None:
    w = np.random.default_rng(400).normal(size=(4, 5))
    feedback = boundary_homeostasis_feedback(w, alpha=0.025)
    assert feedback.shape == w.shape
    assert np.isclose(np.linalg.norm(feedback), 0.025 * np.linalg.norm(w))


@pytest.mark.parametrize(
    "shape",
    ((2, 3), (3, 5), (4, 4), (3, 3, 3), (2, 3, 4, 5)),
)
def test_exact_nonzero_lower_hodge_spectrum(shape: tuple[int, ...]) -> None:
    boundary = _boundary_matrix(shape)
    eigenvalues = np.linalg.eigvalsh(boundary.T @ boundary)
    observed = tuple(
        sorted(set(np.rint(eigenvalues[eigenvalues > 1e-9]).astype(int).tolist()))
    )
    assert observed == boundary_spectral_values(shape)[1:]
    assert np.max(np.abs(eigenvalues - np.rint(eigenvalues))) < 1e-10


def _pattern_count(shape: tuple[int, ...], r: int) -> int:
    degree = min(shape) - 1
    excess = [size - degree - 1 for size in shape]
    return sum(
        (-1) ** j
        * math.comb(r, j)
        * math.prod(offset + r - j for offset in excess)
        for j in range(r + 1)
    )


@pytest.mark.parametrize("shape", ((3, 5), (4, 4), (3, 4, 5), (2, 3, 4, 5)))
def test_exact_lower_hodge_multiplicities(shape: tuple[int, ...]) -> None:
    degree = min(shape) - 1
    boundary = _boundary_matrix(shape)
    eigenvalues = np.linalg.eigvalsh(boundary.T @ boundary)
    for r in range(1, min(len(shape), degree) + 1):
        alternating_binomial_sum = sum(
            math.comb(degree - 2 * ell - 1, r - 1)
            for ell in range((degree - r) // 2 + 1)
        )
        expected = _pattern_count(shape, r) * alternating_binomial_sum
        observed = int(np.count_nonzero(np.isclose(eigenvalues, r + 1, atol=1e-9)))
        assert observed == expected


@pytest.mark.parametrize("shape", ((3, 5), (3, 4, 5), (2, 3, 4, 5)))
def test_full_hodge_laplacian_is_entrywise_diagonal(shape: tuple[int, ...]) -> None:
    degree = min(shape) - 1
    lower_boundary = _boundary_matrix(shape)
    upper_boundary = _boundary_matrix(tuple(size + 1 for size in shape))
    full_hodge = lower_boundary.T @ lower_boundary + upper_boundary @ upper_boundary.T
    expected = np.array(
        [
            1 + len({index for index in multi_index if index <= degree})
            for multi_index in np.ndindex(shape)
        ],
        dtype=float,
    )
    assert np.allclose(full_hodge, np.diag(expected), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("shape", ((3, 5), (4, 4), (3, 4, 5), (2, 3, 4, 5)))
def test_exact_cycle_projection(shape: tuple[int, ...]) -> None:
    w = np.random.default_rng(500 + sum(shape)).normal(size=shape)
    projected = exact_cycle_projection(w)
    projected_twice = exact_cycle_projection(projected)
    removed = w - projected

    assert np.linalg.norm(bdry(projected)) <= 2e-12 * np.linalg.norm(w)
    assert np.allclose(projected_twice, projected, rtol=1e-12, atol=1e-12)
    assert abs(np.vdot(removed, projected)) <= 2e-11 * np.linalg.norm(w) ** 2
    assert np.allclose(boundary_range_projection(w), removed, rtol=1e-12, atol=1e-12)


def test_cycle_projection_matches_dense_svd() -> None:
    shape = (3, 4)
    boundary = _boundary_matrix(shape)
    dense_projector = np.eye(boundary.shape[1]) - np.linalg.pinv(boundary) @ boundary
    w = np.random.default_rng(600).normal(size=shape)
    expected = (dense_projector @ w.reshape(-1)).reshape(shape)
    assert np.allclose(exact_cycle_projection(w), expected, rtol=1e-11, atol=1e-11)


def test_boundary_pseudoinverse_matches_dense_pseudoinverse() -> None:
    shape = (4, 5)
    boundary = _boundary_matrix(shape)
    y = np.random.default_rng(700).normal(size=(3, 4))
    expected = (np.linalg.pinv(boundary) @ y.reshape(-1)).reshape(shape)
    actual = boundary_pseudoinverse(y, shape)
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_project_to_realizable_boundary_is_exact_and_minimum_distance() -> None:
    rng = np.random.default_rng(800)
    shape = (4, 5)
    w = rng.normal(size=shape)
    witness = rng.normal(size=shape)
    target = bdry(witness)
    projected = project_to_boundary(w, target)
    displacement = w - projected

    assert np.allclose(bdry(projected), target, rtol=1e-11, atol=1e-11)
    assert np.allclose(
        project_to_boundary(w), exact_cycle_projection(w), rtol=1e-11, atol=1e-11
    )
    # A minimum-norm correction is orthogonal to every cycle direction.
    cycle = exact_cycle_projection(rng.normal(size=shape))
    assert abs(np.vdot(displacement, cycle)) <= 2e-11 * np.linalg.norm(displacement)


@pytest.mark.parametrize("normalized", (False, True))
def test_sobolev_filter_matches_dense_linear_solve(normalized: bool) -> None:
    shape = (3, 4, 5)
    boundary = _boundary_matrix(shape)
    laplacian = boundary.T @ boundary
    mu = 0.7
    if normalized:
        laplacian = laplacian / boundary_spectral_values(shape)[-1]
    w = np.random.default_rng(900).normal(size=shape)
    expected = np.linalg.solve(
        np.eye(laplacian.shape[0]) + mu * laplacian, w.reshape(-1)
    ).reshape(shape)
    actual = boundary_sobolev_filter(w, mu=mu, normalized=normalized)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_return_feedback_is_not_nilpotent() -> None:
    w = np.random.default_rng(1000).normal(size=(4, 5))
    first = lower_hodge_laplacian(w)
    second = lower_hodge_laplacian(first)
    assert np.linalg.norm(second) > 1e-6
    assert np.allclose(bdry(bdry(w)), 0.0)


@pytest.mark.parametrize(
    ("call", "error"),
    (
        (lambda: face_adjoint(np.zeros((2, 2)), (3, 4), 0), ValueError),
        (lambda: face_adjoint(np.zeros((2, 3)), (3, 4), 3), IndexError),
        (lambda: bdry_adjoint(np.zeros((2, 2)), (3, 4)), ValueError),
        (lambda: bdry_adjoint(np.zeros((0,)), (0,)), ValueError),
        (lambda: boundary_sobolev_filter(np.ones((3, 3)), mu=-1), ValueError),
    ),
)
def test_validation(call, error: type[Exception]) -> None:
    with pytest.raises(error):
        call()
