"""Marked presentation kernels and exact sparse tensor realizations over fields.

The source is the normalized chain complex of the standard simplex, with
its ordered injective vertex labels. Matrices act on coordinate columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, islice
from math import comb, prod

import sympy as sp

from .chain_complex import ChainComplex, ExactField, _nonnegative_integer
from .chain_maps import ChainMap, chain_embedding
from .rank_shifted import SparseTensor
from .tensor_chain_complex import (
    _columns,
    _fraction,
    _input_tensor,
    _normalized_images,
    generated_chain_complex,
)

__all__ = ["KernelSequence", "kernel_sequence", "realize_kernel", "simplex_chain_complex"]


def simplex_chain_complex(n, *, modulus=None):
    """Return N(K[Delta^n]) in ordered normalized injective-simplex bases.

    All degrees zero through n are retained. The differential on vertex
    subsets is alternating deletion; the degree-zero differential is zero.
    """
    n = _nonnegative_integer(n, "Simplex dimension")
    field = ExactField(modulus)
    labels = tuple(tuple(combinations(range(n + 1), q + 1)) for q in range(n + 1))
    dimensions = tuple(len(basis) for basis in labels)
    differentials = [sp.zeros(0, dimensions[0])]
    for q in range(1, n + 1):
        rows = {vertices: row for row, vertices in enumerate(labels[q - 1])}
        boundary = sp.zeros(dimensions[q - 1], dimensions[q])
        for column, vertices in enumerate(labels[q]):
            for index in range(q + 1):
                boundary[rows[vertices[:index] + vertices[index + 1:]], column] = (-1) ** index
        differentials.append(boundary)
    return ChainComplex(dimensions, tuple(differentials), modulus=field.modulus,
                        ambient_bases=labels)


def _quotient(inclusion):
    """Choose quotient coordinates and representatives for a chain inclusion."""
    source, field = inclusion.target, inclusion.target.field
    lifts, projections = [], []
    for q, dimension in enumerate(source.dimensions):
        kernel_basis = inclusion.matrix(q)
        full = field.column_basis(kernel_basis.row_join(sp.eye(dimension)))
        count = kernel_basis.cols
        lifts.append(field.matrix(full[:, count:]))
        inverse = field.coordinates(full, sp.eye(dimension))
        projections.append(field.matrix(inverse[count:, :]))
    dimensions = tuple(lift.cols for lift in lifts)
    differentials = [sp.zeros(0, dimensions[0])]
    for q in range(1, len(dimensions)):
        differentials.append(field.matmul(projections[q - 1],
                                         field.matmul(source.differential(q), lifts[q])))
    quotient = ChainComplex(dimensions, tuple(differentials), modulus=field.modulus,
                            lifts=tuple(lifts), ambient_bases=source.ambient_bases)
    return quotient, ChainMap(source, quotient, tuple(projections))


@dataclass(frozen=True)
class KernelSequence:
    """The marked exact presentation of one tensor-generated simplicial module.

    ``g_matrices[q]`` realizes every ordered normalized source simplex in
    sparse tensor-coordinate rows ``tensor_bases[q]``. ``kernel.lifts[q]``
    gives a basis of its kernel in the source coordinates. ``projection``
    maps source to quotient; ``comparison`` identifies that quotient with
    ``image``. ``realization`` is their composite from source to image.
    """

    tensor: SparseTensor
    source: ChainComplex
    kernel: ChainComplex
    quotient: ChainComplex
    image: ChainComplex
    g_matrices: tuple[sp.ImmutableMatrix, ...]
    tensor_bases: tuple[tuple[tuple[int, ...], ...], ...]
    inclusion: ChainMap
    projection: ChainMap
    realization: ChainMap
    comparison: ChainMap


def kernel_sequence(tensor, *, modulus=None):
    """Compute all marked kernels L_q=ker(N(f_T)_q), zero <= q <= n.

    Dense arrays have degree min(shape)-1. SparseTensor input retains its
    explicit degree and shifts. Coefficients lie in Q or a prime field;
    finite-field rational denominators must be invertible. No dense ambient
    tensor is allocated. Degrees above tensor order are retained in source
    and kernel, even when their realization matrices have no rows.
    """
    field = ExactField(modulus)
    tensor = _input_tensor(tensor, field)
    source = simplex_chain_complex(tensor.degree, modulus=field.modulus)
    image = generated_chain_complex(tensor, modulus=field.modulus)
    labels, matrices, kernels, realization = [], [], [], []
    for q, dimension in enumerate(source.dimensions):
        if q < len(image.dimensions):
            coordinates = image.ambient_bases[q]
            columns = [value for _, value in _normalized_images(tensor, q)]
            matrix = _columns(columns, coordinates, field)
            image_coordinates = field.coordinates(image.lifts[q], matrix)
        else:
            coordinates = ()
            matrix = sp.ImmutableMatrix.zeros(0, dimension)
            image_coordinates = sp.ImmutableMatrix.zeros(0, dimension)
        labels.append(coordinates)
        matrices.append(matrix)
        kernels.append(field.nullspace(matrix))
        realization.append(image_coordinates)
    dimensions = tuple(basis.cols for basis in kernels)
    differentials = [sp.zeros(0, dimensions[0])]
    for q in range(1, len(dimensions)):
        differentials.append(field.coordinates(kernels[q - 1],
                                              field.matmul(source.differential(q), kernels[q])))
    kernel = ChainComplex(dimensions, tuple(differentials), modulus=field.modulus,
                          lifts=tuple(kernels), ambient_bases=source.ambient_bases)
    inclusion = ChainMap(kernel, source, tuple(kernels))
    quotient, projection = _quotient(inclusion)
    realized = ChainMap(source, image, tuple(realization))
    comparison = ChainMap(quotient, image, tuple(
        field.matmul(realized.matrix(q), quotient.lifts[q]) for q in range(len(source.dimensions))
    ))
    if not comparison.is_isomorphism():
        raise ArithmeticError("The presentation quotient did not identify with the generated image.")
    return KernelSequence(tensor, source, kernel, quotient, image, tuple(matrices),
                          tuple(labels), inclusion, projection, realized, comparison)


def _covering_coordinates(degree, shifts):
    """Enumerate coordinates covering 0,...,degree, pruning impossible prefixes."""
    lengths = tuple(degree + shift + 1 for shift in shifts)

    def extend(prefix, missing):
        axis = len(prefix)
        remaining = len(lengths) - axis
        if len(missing) > remaining:
            return
        if not remaining:
            yield prefix
            return
        candidates = sorted(missing) if remaining == len(missing) else range(lengths[axis])
        for value in candidates:
            if value >= lengths[axis]:
                continue
            yield from extend(prefix + (value,), missing - {value})

    yield from extend((), frozenset(range(degree + 1)))


def _cycle_capacity(degree, shifts):
    return sum((-1) ** omitted * comb(degree + 1, omitted)
               * prod(degree + shift + 1 - omitted for shift in shifts)
               for omitted in range(degree + 2))


def _small_tensor_target(quotient, shifts):
    """Choose only the coordinate disks and tail classes needed for an embedding."""
    selected = []
    for q in range(len(quotient.dimensions)):
        demand = quotient.cycles(q).cols
        count = min(demand, _cycle_capacity(q, shifts))
        selected.append(tuple(islice(_covering_coordinates(q, shifts), count)))
    tails_needed = quotient.dimension(0) - len(selected[0])
    tail_count = min(tails_needed, prod(shifts))
    tails = []
    for ordinal in range(tail_count):
        values = []
        for length in reversed(shifts):
            ordinal, remainder = divmod(ordinal, length)
            values.append(remainder + 1)
        tails.append(tuple(reversed(values)))
    tails = tuple(tails)
    labels = []
    for q in range(len(selected) + 1):
        cycles = selected[q] if q < len(selected) else ()
        sections = (tuple(tuple(value + 1 for value in index) for index in selected[q - 1])
                    if q else tails)
        labels.append(cycles + sections)
    dimensions = tuple(len(basis) for basis in labels)
    differentials = [sp.zeros(0, dimensions[0])]
    for q in range(1, len(dimensions)):
        boundary = sp.zeros(dimensions[q - 1], dimensions[q])
        offset = len(selected[q]) if q < len(selected) else 0
        for index in range(len(selected[q - 1])):
            boundary[index, offset + index] = 1
        differentials.append(boundary)
    return ChainComplex(dimensions, tuple(differentials), modulus=quotient.modulus,
                        ambient_bases=tuple(labels))


def _reduce_tensor(tensor, field):
    return SparseTensor(tensor.degree, tensor.shifts, {
        index: _fraction(field.scalar(value)) for index, value in tensor.entries.items()
    })


def realize_kernel(inclusion, shifts):
    """Realize an embedded subcomplex of N(K[Delta^n]) by a sparse tensor.

    ``inclusion`` must be injective and its target must have the standard
    simplex dimensions, differentials, and ordered injective vertex labels.
    ``shifts`` specifies the nonnegative trailing lengths on each tensor
    axis. A CapacityError reports a failed cycle or boundary inequality.
    The construction selects only as many coordinate disks and degree-zero
    tail classes as the quotient needs; it never enumerates the full tensor
    space. The returned tensor has precisely the supplied embedded kernel.
    """
    if not isinstance(inclusion, ChainMap):
        raise TypeError("Supply a ChainMap inclusion into a standard simplex complex.")
    if not inclusion.is_injective():
        raise ValueError("The supplied kernel inclusion must be injective.")
    source = inclusion.target
    if not source.dimensions or not source.dimension(0):
        raise ValueError("The inclusion target must be a standard simplex complex.")
    n = source.dimension(0) - 1
    standard = simplex_chain_complex(n, modulus=source.modulus)
    if (source.dimensions != standard.dimensions
            or source.differentials != standard.differentials
            or source.ambient_bases != standard.ambient_bases):
        raise ValueError("The inclusion target must use the standard ordered simplex bases.")
    shifts = tuple(_nonnegative_integer(shift, "Axis shift") for shift in shifts)
    shifts = SparseTensor.zero(n, shifts).shifts
    field = source.field
    quotient, projection = _quotient(inclusion)
    target = _small_tensor_target(quotient, shifts)
    realized = chain_embedding(quotient, target).compose(projection)
    raw = {}
    for q, subsets in enumerate(source.ambient_bases):
        for column, vertices in enumerate(subsets):
            normalized = SparseTensor(q, shifts, {
                index: _fraction(realized.matrix(q)[row, column])
                for row, index in enumerate(target.ambient_bases[q])
                if realized.matrix(q)[row, column]
            })
            filler = SparseTensor.zero(q, shifts)
            for index in range(q, 0, -1):
                face = raw[vertices[:index] + vertices[index + 1:]]
                correction = (face - filler.face(index)).degeneracy(index - 1)
                filler = _reduce_tensor(filler + correction, field)
            raw[vertices] = _reduce_tensor(filler + normalized, field)
    return raw[tuple(range(n + 1))]
