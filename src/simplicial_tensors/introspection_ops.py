"""Exact boundary analysis--controller--synthesis operations.

This module treats ``bdry(w)`` as an observation, not as a quantity to
penalize.  The cycle projector and Moore--Penrose decoder give the orthogonal
decomposition

``w = cycle + boundary_pseudoinverse(bdry(w), w.shape)``.

A controller may replace the observed boundary by another realizable boundary
without changing the cycle component.  The second boundary of every observed
signal is zero by the chain identity ``bdry @ bdry = 0``; this terminates the
graded observation, while the typed pseudoinverse returns a controller target
to the original tensor space.  These operations add no scalar penalty to a
training objective.  A controller can nevertheless impose a hard constraint:
choosing the zero target, for example, projects the tensor onto ``ker(bdry)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from .adjoint_ops import (
    boundary_pseudoinverse,
    boundary_spectral_values,
    exact_cycle_projection,
    lower_hodge_laplacian,
)
from .tensor_ops import bdry

Shape = tuple[int, ...]
BoundaryController = Callable[[np.ndarray], np.ndarray]


class BoundaryAnalysis(NamedTuple):
    """The boundary-invisible and boundary-visible coordinates of a tensor."""

    cycle: np.ndarray
    syndrome: np.ndarray


def boundary_analyze(w: np.ndarray) -> BoundaryAnalysis:
    """Split ``w`` into its cycle coordinate and boundary observation.

    The returned pair is lossless: passing it to :func:`boundary_synthesize`
    reconstructs ``w`` up to floating-point roundoff.
    """

    source = np.asarray(w)
    cycle = exact_cycle_projection(source)
    return BoundaryAnalysis(cycle=cycle, syndrome=bdry(source))


def project_boundary_signal(
    syndrome: np.ndarray,
    original_shape: Sequence[int],
) -> np.ndarray:
    """Project a proposed signal orthogonally onto ``range(bdry)``.

    A controller can emit an arbitrary tensor of the boundary shape.  This
    projection makes it a valid target before synthesis.
    """

    shape = tuple(original_shape)
    decoded = boundary_pseudoinverse(np.asarray(syndrome), shape)
    return bdry(decoded)


def boundary_synthesize(
    cycle: np.ndarray,
    syndrome: np.ndarray,
    original_shape: Sequence[int] | None = None,
) -> np.ndarray:
    """Synthesize a tensor from a cycle coordinate and boundary signal.

    ``cycle`` is required to lie numerically in ``ker(bdry)``.  Non-realizable
    syndrome components are discarded by the Moore--Penrose decoder.  When
    ``original_shape`` is supplied, it must equal ``cycle.shape``; the
    argument makes the return type explicit at call sites where the shape is
    carried separately.
    """

    cycle_array = np.asarray(cycle)
    shape: Shape = cycle_array.shape
    if original_shape is not None and tuple(original_shape) != shape:
        raise ValueError(
            f"cycle has shape {shape}; received original_shape "
            f"{tuple(original_shape)}"
        )
    cycle_boundary = bdry(cycle_array)
    cycle_norm = float(np.linalg.norm(cycle_array))
    residual_norm = float(np.linalg.norm(cycle_boundary))
    if not np.isfinite(cycle_norm) or not np.isfinite(residual_norm):
        raise ValueError("cycle and its boundary must contain only finite values")

    if np.issubdtype(cycle_array.dtype, np.complexfloating):
        real_dtype = np.empty((), dtype=cycle_array.dtype).real.dtype
    elif np.issubdtype(cycle_array.dtype, np.floating):
        real_dtype = cycle_array.dtype
    else:
        real_dtype = np.dtype(np.float64)
    precision = np.finfo(real_dtype)
    scale = max(cycle_norm, precision.tiny)
    tolerance = 128.0 * precision.eps * np.sqrt(max(1, cycle_array.size)) * scale
    if residual_norm > tolerance:
        raise ValueError(
            "cycle must lie in ker(bdry): boundary residual norm "
            f"{residual_norm:.6g} exceeds tolerance {tolerance:.6g}"
        )
    decoded = boundary_pseudoinverse(np.asarray(syndrome), shape)
    return cycle_array + decoded


def boundary_target_update(w: np.ndarray, target_boundary: np.ndarray) -> np.ndarray:
    """Replace ``bdry(w)`` by a fixed target while preserving its cycle part.

    If the target is not realizable, its orthogonal projection onto
    ``range(bdry)`` is attained.  For a fixed target, this map is idempotent.
    The update adds no scalar penalty; a zero target instead imposes the hard
    constraint ``bdry(w) = 0`` by exact projection.
    """

    analysis = boundary_analyze(w)
    target = project_boundary_signal(target_boundary, analysis.cycle.shape)
    return boundary_synthesize(analysis.cycle, target)


def boundary_controller_feedback(
    w: np.ndarray,
    controller: BoundaryController,
) -> np.ndarray:
    """Analyze ``w``, run a boundary controller, and synthesize its target.

    The callable receives a copy of ``bdry(w)`` and must return a tensor of
    the same boundary shape.  Task state can be supplied through a closure.
    The proposed target is projected onto ``range(bdry)`` before synthesis.
    """

    analysis = boundary_analyze(w)
    proposed = np.asarray(controller(analysis.syndrome.copy()))
    target = project_boundary_signal(proposed, analysis.cycle.shape)
    return boundary_synthesize(analysis.cycle, target)


def _validate_binary_gates(gates: Sequence[bool | int], count: int) -> tuple[bool, ...]:
    try:
        values = tuple(float(gate) for gate in gates)
    except (TypeError, ValueError) as exc:
        raise ValueError("gates must be a one-dimensional binary sequence") from exc
    if len(values) != count:
        raise ValueError(f"expected {count} spectral gates, received {len(values)}")
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("every spectral gate must be 0 or 1")
    return tuple(bool(value) for value in values)


def _lower_hodge_spectral_component(
    w: np.ndarray,
    eigenvalue: int,
    nodes: Sequence[int],
) -> np.ndarray:
    """Apply the exact Lagrange projector for one lower-Hodge eigenvalue."""

    dtype = np.result_type(w.dtype, np.float64)
    component: np.ndarray = w.astype(dtype, copy=True)
    for other in nodes:
        if other == eigenvalue:
            continue
        component = (
            lower_hodge_laplacian(component) - other * component
        ) / (eigenvalue - other)
    return component


def boundary_projector_feedback(
    w: np.ndarray,
    gates: Sequence[bool | int],
) -> np.ndarray:
    """Apply a fixed idempotent gate to the boundary-visible spectral bands.

    Gates correspond, in order, to the nonzero lower-Hodge eigenvalues from
    :func:`boundary_spectral_values`.  The zero-eigenvalue cycle component is
    always retained.  Binary gates make the complete feedback map an
    orthogonal projector, hence idempotent up to floating-point roundoff.

    For a matrix of simplicial degree at least two, for example, ``gates`` has
    two entries corresponding to eigenvalues 2 and 3.
    """

    analysis = boundary_analyze(w)
    nodes = boundary_spectral_values(analysis.cycle.shape)
    checked_gates = _validate_binary_gates(gates, len(nodes) - 1)
    visible = boundary_pseudoinverse(analysis.syndrome, analysis.cycle.shape)
    selected = np.zeros_like(visible, dtype=np.result_type(visible.dtype, np.float64))
    for keep, eigenvalue in zip(checked_gates, nodes[1:], strict=True):
        if keep:
            selected += _lower_hodge_spectral_component(visible, eigenvalue, nodes)

    # ``selected`` lies in range(bdry_adjoint), so this is the minimum-norm
    # synthesis of its boundary and leaves the analyzed cycle untouched.
    return boundary_synthesize(analysis.cycle, bdry(selected))
