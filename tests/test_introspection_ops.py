"""Tests for exact boundary analysis--controller--synthesis operations."""

from __future__ import annotations

import numpy as np
import pytest

from simplicial_tensors.adjoint_ops import exact_cycle_projection, lower_hodge_laplacian
from simplicial_tensors.introspection_ops import (
    boundary_analyze,
    boundary_controller_feedback,
    boundary_projector_feedback,
    boundary_synthesize,
    boundary_target_update,
    project_boundary_signal,
)
from simplicial_tensors.tensor_ops import bdry


@pytest.mark.parametrize("shape", ((4, 5), (3, 4, 5), (3, 3, 4, 5)))
def test_analysis_synthesis_is_lossless(shape: tuple[int, ...]) -> None:
    w = np.random.default_rng(1100 + sum(shape)).normal(size=shape)
    analysis = boundary_analyze(w)
    reconstructed = boundary_synthesize(
        analysis.cycle, analysis.syndrome, original_shape=shape
    )

    assert np.allclose(reconstructed, w, rtol=1e-11, atol=1e-11)
    assert np.linalg.norm(bdry(analysis.cycle)) <= 2e-12 * np.linalg.norm(w)
    assert np.allclose(bdry(analysis.syndrome), 0.0, rtol=0.0, atol=1e-12)


def test_target_update_attains_target_and_preserves_cycle() -> None:
    rng = np.random.default_rng(1200)
    shape = (4, 5)
    w = rng.normal(size=shape)
    target = bdry(rng.normal(size=shape))
    updated = boundary_target_update(w, target)

    assert np.allclose(bdry(updated), target, rtol=1e-11, atol=1e-11)
    assert np.allclose(
        exact_cycle_projection(updated),
        exact_cycle_projection(w),
        rtol=1e-11,
        atol=1e-11,
    )


def test_nonrealizable_target_is_projected_before_synthesis() -> None:
    rng = np.random.default_rng(1300)
    shape = (4, 5)
    w = rng.normal(size=shape)
    proposed = rng.normal(size=(3, 4))
    expected = project_boundary_signal(proposed, shape)
    updated = boundary_target_update(w, proposed)

    assert np.allclose(bdry(updated), expected, rtol=1e-11, atol=1e-11)
    assert np.allclose(bdry(expected), 0.0, rtol=0.0, atol=1e-12)


def test_fixed_target_update_is_idempotent() -> None:
    rng = np.random.default_rng(1400)
    shape = (4, 5)
    w = rng.normal(size=shape)
    target = rng.normal(size=(3, 4))
    once = boundary_target_update(w, target)
    twice = boundary_target_update(once, target)

    assert np.allclose(twice, once, rtol=1e-11, atol=1e-11)


def test_identity_controller_reconstructs_original_tensor() -> None:
    w = np.random.default_rng(1500).normal(size=(4, 5))
    updated = boundary_controller_feedback(w, lambda syndrome: syndrome)
    assert np.allclose(updated, w, rtol=1e-11, atol=1e-11)


def test_controller_reaches_returned_realizable_target() -> None:
    rng = np.random.default_rng(1600)
    shape = (4, 5)
    w = rng.normal(size=shape)
    proposed = rng.normal(size=(3, 4))
    updated = boundary_controller_feedback(w, lambda _: proposed)

    assert np.allclose(
        bdry(updated),
        project_boundary_signal(proposed, shape),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("gates", ((0, 0), (1, 0), (0, 1), (1, 1)))
def test_projector_feedback_is_idempotent(gates: tuple[int, int]) -> None:
    w = np.random.default_rng(1700 + sum(gates)).normal(size=(4, 5))
    once = boundary_projector_feedback(w, gates)
    twice = boundary_projector_feedback(once, gates)

    assert np.allclose(twice, once, rtol=1e-10, atol=1e-10)
    assert np.allclose(
        exact_cycle_projection(once),
        exact_cycle_projection(w),
        rtol=1e-10,
        atol=1e-10,
    )


def test_projector_feedback_gate_meanings() -> None:
    w = np.random.default_rng(1800).normal(size=(4, 5))
    cycle = exact_cycle_projection(w)
    only_two = boundary_projector_feedback(w, (1, 0))

    assert np.allclose(boundary_projector_feedback(w, (0, 0)), cycle, atol=1e-11)
    assert np.allclose(boundary_projector_feedback(w, (1, 1)), w, atol=1e-11)
    assert np.allclose(
        lower_hodge_laplacian(only_two - cycle),
        2.0 * (only_two - cycle),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("call", "match"),
    (
        (
            lambda: boundary_synthesize(
                np.zeros((4, 5)), np.zeros((3, 4)), original_shape=(5, 4)
            ),
            "cycle has shape",
        ),
        (
            lambda: boundary_synthesize(
                np.arange(20).reshape(4, 5), np.zeros((3, 4))
            ),
            "cycle must lie in ker",
        ),
        (lambda: boundary_projector_feedback(np.ones((4, 5)), (1,)), "expected 2"),
        (
            lambda: boundary_projector_feedback(np.ones((4, 5)), (1, 0.5)),
            "must be 0 or 1",
        ),
        (
            lambda: boundary_controller_feedback(
                np.ones((4, 5)), lambda _: np.ones((2, 2))
            ),
            "boundary tensor has shape",
        ),
    ),
)
def test_validation(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
