"""Tests for exact boundary analysis--controller--synthesis operations."""

from __future__ import annotations

import numpy as np
import pytest

from simplicial_tensors.adjoint_ops import (
    bdry_adjoint,
    exact_cycle_projection,
    lower_hodge_laplacian,
)
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


@pytest.mark.parametrize(
    "dtype",
    (np.bool_, np.uint8, np.int8, np.float16, np.float32, np.float64, np.complex64, np.complex128),
)
def test_analysis_promotes_before_observing_boundary(dtype) -> None:
    w = np.array([1, 0], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        w += np.array([2j, -1j], dtype=dtype)
    numeric = w.astype(np.result_type(w.dtype, np.float64))
    analysis = boundary_analyze(w)
    np.testing.assert_array_equal(analysis.syndrome, bdry(numeric))
    np.testing.assert_allclose(boundary_synthesize(*analysis), numeric, atol=1e-13)
    np.testing.assert_allclose(boundary_target_update(w, analysis.syndrome), numeric, atol=1e-13)
    np.testing.assert_allclose(boundary_controller_feedback(w, lambda signal: signal), numeric, atol=1e-13)
    assert analysis.cycle.dtype == numeric.dtype
    assert analysis.syndrome.dtype == numeric.dtype


@pytest.mark.parametrize("shape", ((4, 5), (4, 4, 4)))
@pytest.mark.parametrize("scale", (1e-200, 1.0, 1e200))
@pytest.mark.parametrize("complex_input", (False, True))
def test_pure_visible_analysis_and_feedback_at_all_scales(
    shape: tuple[int, ...], scale: float, complex_input: bool
) -> None:
    rng = np.random.default_rng(20261112)
    boundary_shape = tuple(size - 1 for size in shape)
    y = rng.normal(size=boundary_shape)
    if complex_input:
        y = y + 1j * rng.normal(size=boundary_shape)
    w = scale * bdry_adjoint(y, shape)
    analysis = boundary_analyze(w)
    for actual in (
        boundary_synthesize(*analysis),
        boundary_target_update(w, analysis.syndrome),
        boundary_controller_feedback(w, lambda signal: signal),
    ):
        np.testing.assert_allclose(actual / scale, w / scale, rtol=2e-12, atol=2e-12)
    # A vanishing new visible channel must not make the analyzed cycle fail
    # its own relative residual test.
    zero_target = np.zeros(boundary_shape, dtype=w.dtype)
    for actual in (
        boundary_target_update(w, zero_target),
        boundary_controller_feedback(w, lambda _: zero_target),
        boundary_projector_feedback(w, (0,) * min(len(shape), min(shape) - 1)),
    ):
        np.testing.assert_allclose(actual / scale, 0.0, atol=2e-12)
        np.testing.assert_allclose(bdry(actual) / scale, 0.0, atol=2e-12)


def test_synthesis_uses_both_channel_scales_for_roundoff() -> None:
    y = np.random.default_rng(20261112).normal(size=(3, 4))
    w = bdry_adjoint(y, (4, 5))
    # The first projection leaves cancellation noise that is small relative
    # to w but need not be a cycle relative to its own tiny norm.
    cycle = exact_cycle_projection(w)
    assert np.linalg.norm(cycle) < 1e-14 * np.linalg.norm(w)
    reconstructed = boundary_synthesize(cycle, bdry(w))
    np.testing.assert_allclose(reconstructed, w, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("scale", (1e-200, 1e-20, 1.0, 1e200))
@pytest.mark.parametrize("complex_input", (False, True))
def test_synthesis_rejects_noncycles_without_absolute_tolerance(
    scale: float, complex_input: bool
) -> None:
    cycle = scale * np.array([1.0, 0.0])
    if complex_input:
        cycle = 1j * cycle
    with pytest.raises(ValueError, match="cycle must lie in ker"):
        boundary_synthesize(cycle, np.zeros(1))
    # A visible channel does not excuse a substantive invalid cycle.
    with pytest.raises(ValueError, match="cycle must lie in ker"):
        boundary_synthesize(cycle, np.array([scale]))


@pytest.mark.parametrize("value", (np.nan, np.inf))
def test_synthesis_rejects_nonfinite_channels(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        boundary_synthesize(np.array([value, value]), np.zeros(1))
    with pytest.raises(ValueError, match="must be finite"):
        boundary_synthesize(np.zeros(2), np.array([value]))



@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.complex64))
def test_low_precision_cycle_inputs_do_not_weaken_validation(dtype) -> None:
    with pytest.raises(ValueError, match="cycle must lie in ker"):
        boundary_synthesize(np.array([1.0, 0.9], dtype=dtype), np.zeros(1))
