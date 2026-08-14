"""Adjoints and finite spectral filters for the diagonal tensor boundary.

The functions in this module use the entrywise Euclidean/Frobenius inner
product.  The diagonal face ``d_i`` deletes index ``i`` along every tensor
axis, so its adjoint inserts a tensor in the complementary positions and
fills all deleted hyperplanes with zero.

For a tensor of order ``k`` and simplicial degree ``p = min(shape) - 1``,
the non-zero eigenvalues of the lower Hodge operator
``L = bdry_adjoint @ bdry`` are the integers
``2, ..., min(k, p) + 1``.  This finite spectrum permits exact, matrix-free
polynomial evaluation of the cycle projector, Moore--Penrose boundary
decoder, and Sobolev resolvent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .tensor_ops import bdry

Shape = tuple[int, ...]


def _validate_original_shape(original_shape: Sequence[int]) -> Shape:
    """Return a validated integer shape."""

    try:
        shape = tuple(int(size) for size in original_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("original_shape must be a non-empty sequence of integers") from exc
    if not shape:
        raise ValueError("original_shape must not be empty")
    if any(size < 1 for size in shape):
        raise ValueError("every dimension of original_shape must be at least 1")
    return shape


def _validate_boundary_shape(y: np.ndarray, original_shape: Sequence[int]) -> Shape:
    shape = _validate_original_shape(original_shape)
    expected = tuple(size - 1 for size in shape)
    if y.shape != expected:
        raise ValueError(
            f"boundary tensor has shape {y.shape}; expected {expected} "
            f"for original shape {shape}"
        )
    return shape


def face_adjoint(y: np.ndarray, original_shape: tuple[int, ...], i: int) -> np.ndarray:
    """Return the Frobenius adjoint of the diagonal face map ``d_i``.

    ``d_i`` deletes index ``i`` along every axis.  This adjoint places ``y``
    at the entries whose indices all avoid ``i`` and sets the complementary
    hyperplanes to zero.
    """

    y = np.asarray(y)
    shape = _validate_boundary_shape(y, original_shape)
    if i < 0 or i >= min(shape):
        raise IndexError(f"face index {i} out of bounds for tensor shape {shape}")

    result = np.zeros(shape, dtype=y.dtype)
    keep = tuple(np.delete(np.arange(size), i) for size in shape)
    result[np.ix_(*keep)] = y
    return result


def bdry_adjoint(y: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Return the Frobenius adjoint of the diagonal boundary map."""

    y = np.asarray(y)
    shape = _validate_boundary_shape(y, original_shape)
    result = np.zeros(shape, dtype=y.dtype)
    for i in range(min(shape)):
        keep = tuple(np.delete(np.arange(size), i) for size in shape)
        result[np.ix_(*keep)] += y if i % 2 == 0 else -y
    return result


def lower_hodge_laplacian(w: np.ndarray) -> np.ndarray:
    """Return ``bdry_adjoint(bdry(w))`` with the same shape as ``w``."""

    w = np.asarray(w)
    shape = _validate_original_shape(w.shape)
    return bdry_adjoint(bdry(w), shape)


def boundary_homeostasis_gradient(
    w: np.ndarray,
    target_boundary: np.ndarray | None = None,
) -> np.ndarray:
    """Return the gradient of ``0.5 * ||bdry(w) - target_boundary||_F**2``."""

    w = np.asarray(w)
    shape = _validate_original_shape(w.shape)
    residual = bdry(w)
    if target_boundary is not None:
        target = np.asarray(target_boundary)
        _validate_boundary_shape(target, shape)
        residual = residual - target
    return bdry_adjoint(residual, shape)


def boundary_homeostasis_feedback(
    w: np.ndarray,
    target_boundary: np.ndarray | None = None,
    alpha: float = 1e-3,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return norm-matched negative boundary feedback.

    When the gradient is nonzero, the returned perturbation has Frobenius
    norm ``alpha * ||w||_F``.  A numerically zero gradient returns zeros.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    w = np.asarray(w)
    gradient = boundary_homeostasis_gradient(w, target_boundary)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= eps:
        return np.zeros_like(gradient)
    scale = alpha * float(np.linalg.norm(w)) / max(gradient_norm, eps)
    return -scale * gradient


def _spectral_nodes(shape: Sequence[int]) -> tuple[int, ...]:
    """Return the distinct eigenvalues of ``bdry_adjoint @ bdry``."""

    checked = _validate_original_shape(shape)
    degree = min(checked) - 1
    rank_parameter = min(len(checked), degree)
    return (0, *range(2, rank_parameter + 2))


def boundary_spectral_values(shape: Sequence[int]) -> tuple[int, ...]:
    """Return the exact distinct spectrum ``(0, 2, ..., r + 1)``.

    Here ``r = min(order, min(shape) - 1)``.  Some listed nonzero values can
    have zero multiplicity only in the trivial degree-zero case, for which
    the tuple is simply ``(0,)``.
    """

    return _spectral_nodes(shape)


def _newton_coefficients(nodes: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Return Newton divided-difference coefficients."""

    x = np.asarray(nodes, dtype=float)
    coefficients = np.asarray(values, dtype=float).copy()
    if x.ndim != 1 or coefficients.shape != x.shape:
        raise ValueError("nodes and values must be one-dimensional arrays of equal length")
    for order in range(1, len(x)):
        denominator = x[order:] - x[:-order]
        coefficients[order:] = (
            coefficients[order:] - coefficients[order - 1 : -1]
        ) / denominator
    return coefficients


def _apply_spectral_function(
    w: np.ndarray,
    values: Sequence[float],
    operator: Callable[[np.ndarray], np.ndarray] = lower_hodge_laplacian,
) -> np.ndarray:
    """Apply an interpolating polynomial in the lower Hodge operator."""

    w = np.asarray(w)
    nodes = _spectral_nodes(w.shape)
    if len(values) != len(nodes):
        raise ValueError(f"expected {len(nodes)} spectral values, received {len(values)}")
    coefficients = _newton_coefficients(nodes, values)
    dtype = np.result_type(w.dtype, np.float64)
    source: np.ndarray = w.astype(dtype, copy=False)
    result = coefficients[-1] * source
    for position in range(len(coefficients) - 2, -1, -1):
        result = operator(result) - nodes[position] * result + coefficients[position] * source
    return result


def exact_cycle_projection(w: np.ndarray) -> np.ndarray:
    """Orthogonally project ``w`` onto ``ker(bdry)`` in finitely many passes.

    The implementation evaluates

    ``prod_lambda (I - L / lambda) w``

    over ``lambda = 2, ..., min(order, degree) + 1`` and
    ``L = bdry_adjoint @ bdry``.  It requires no dense matrix or SVD.
    """

    w = np.asarray(w)
    _validate_original_shape(w.shape)
    dtype = np.result_type(w.dtype, np.float64)
    result: np.ndarray = w.astype(dtype, copy=True)
    for eigenvalue in _spectral_nodes(w.shape)[1:]:
        result = result - lower_hodge_laplacian(result) / eigenvalue
    return result


def boundary_range_projection(w: np.ndarray) -> np.ndarray:
    """Orthogonally project ``w`` onto ``range(bdry_adjoint)``."""

    w = np.asarray(w)
    return w - exact_cycle_projection(w)


def boundary_pseudoinverse(y: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Apply the Moore--Penrose pseudoinverse of ``bdry`` to ``y``.

    For an arbitrary ``y`` this returns the minimum-norm preimage of the
    orthogonal projection of ``y`` onto ``range(bdry)``.  For a realizable
    boundary it is the exact minimum-Frobenius-norm decoder.
    """

    y = np.asarray(y)
    shape = _validate_boundary_shape(y, original_shape)
    adjoint_signal = bdry_adjoint(y, shape)
    nodes = _spectral_nodes(shape)
    inverse_values = [0.0, *(1.0 / value for value in nodes[1:])]
    return _apply_spectral_function(adjoint_signal, inverse_values)


def project_to_boundary(w: np.ndarray, target_boundary: np.ndarray | None = None) -> np.ndarray:
    """Return the closest tensor with the requested realizable boundary.

    If ``target_boundary`` is not in ``range(bdry)``, its orthogonal
    projection onto that range is used.  With no target this is identical to
    :func:`exact_cycle_projection` up to roundoff.
    """

    w = np.asarray(w)
    shape = _validate_original_shape(w.shape)
    current = bdry(w)
    if target_boundary is None:
        target = np.zeros_like(current)
    else:
        target = np.asarray(target_boundary)
        _validate_boundary_shape(target, shape)
    correction = boundary_pseudoinverse(current - target, shape)
    return w - correction


def boundary_sobolev_filter(w: np.ndarray, mu: float = 1.0, normalized: bool = True) -> np.ndarray:
    """Apply the exact resolvent ``(I + mu * L_hat)^-1`` to ``w``.

    ``L_hat`` is ``L / lambda_max`` when ``normalized`` is true and ``L``
    otherwise.  The resolvent is evaluated as a degree-``min(order, degree)``
    polynomial using the exact finite spectrum, so it needs no iterative
    linear solver.
    """

    if mu < 0:
        raise ValueError("mu must be non-negative")
    w = np.asarray(w)
    nodes = _spectral_nodes(w.shape)
    if len(nodes) == 1 or mu == 0:
        return np.result_type(w.dtype, np.float64).type(1) * w
    scale = float(nodes[-1]) if normalized else 1.0
    values = [1.0 / (1.0 + mu * node / scale) for node in nodes]
    return _apply_spectral_function(w, values)
