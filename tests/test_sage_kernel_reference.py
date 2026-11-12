"""Independent native Sage checks of universal kernel presentations and lifts."""

from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pytest

sage = pytest.importorskip("sage.all")

from simplicial_tensors.chain_maps import chain_isomorphism, lift_quotient_isomorphism
from simplicial_tensors.kernel_sequences import kernel_sequence, realize_kernel
from simplicial_tensors.rank_shifted import SparseTensor


def _scalar(field, value):
    value = Fraction(value)
    return field(value.numerator) / field(value.denominator)


def _native(field, matrix):
    return sage.matrix(field, matrix.rows, matrix.cols,
                       [_scalar(field, value) for value in matrix])


def _projected_simplex(vertices):
    # Independent normalization in the free simplicial source: d_i removes a
    # domain entry, and s_(i-1) repeats the preceding domain entry.
    column = {vertices: 1}
    for i in range(len(vertices) - 1, 0, -1):
        updated = dict(column)
        for alpha, coefficient in column.items():
            face = alpha[:i] + alpha[i + 1:]
            degenerate = face[:i] + (face[i - 1],) + face[i:]
            updated[degenerate] = updated.get(degenerate, 0) - coefficient
        column = {alpha: coefficient for alpha, coefficient in updated.items() if coefficient}
    return column


def _reference(field, tensor, degree):
    vertices = tuple(combinations(range(tensor.degree + 1), degree + 1))
    rows = tuple(product(*(range(degree + shift + 1) for shift in tensor.shifts)))
    projected = tuple(_projected_simplex(face) for face in vertices)

    def coefficient(row, column):
        position = rows[row]
        value = field.zero()
        for alpha, scalar in projected[column].items():
            original = tuple(alpha[index] if index <= degree
                             else tensor.degree + index - degree for index in position)
            value += field(scalar) * _scalar(field, tensor.entries.get(original, 0))
        return value

    return rows, sage.matrix(field, len(rows), len(vertices), coefficient)


def _source_boundary(field, n, degree):
    columns = tuple(combinations(range(n + 1), degree + 1))
    rows = tuple(combinations(range(n + 1), degree)) if degree else ()
    indices = {face: row for row, face in enumerate(rows)}
    result = sage.matrix(field, len(rows), len(columns))
    if degree:
        for column, face in enumerate(columns):
            for i in range(len(face)):
                result[indices[face[:i] + face[i + 1:]], column] += (-1)**i
    return result


def _cases():
    rng = np.random.default_rng(20261112)
    result = [
        SparseTensor(0, (1, 2), {(0, 0): Fraction(2**60 + 3, 5), (1, 2): -2}),
        SparseTensor(1, (0,), {(0,): 2, (1,): -3}),
        SparseTensor(2, (0, 0), {(0, 1): 1, (1, 1): 1, (1, 2): 1,
                                  (2, 0): 1, (2, 1): 1, (2, 2): 4}),
        SparseTensor(3, (0, 0), {(0, 1): 1, (2, 3): -2, (3, 3): 3}),
    ]
    for degree, shifts in [(2, (1, 2, 0)), (2, (2, 1))]:
        positions = tuple(product(*(range(degree + shift + 1) for shift in shifts)))
        entries = {positions[int(index)]: int(rng.integers(-3, 4))
                   for index in rng.choice(len(positions), size=7, replace=False)}
        result.append(SparseTensor(degree, shifts, entries))
    return tuple(result)


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("tensor", _cases())
def test_full_kernel_matches_independent_native_normalized_pullbacks(tensor, modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    model = kernel_sequence(tensor, modulus=modulus)
    for degree in range(tensor.degree + 1):
        rows, reference = _reference(field, tensor, degree)
        row_index = {position: row for row, position in enumerate(rows)}
        raw = model.g_matrices[degree]
        actual = sage.matrix(field, len(rows), raw.cols)
        for source_row, position in enumerate(model.tensor_bases[degree]):
            actual[row_index[position], :] = sage.matrix(
                field, 1, raw.cols,
                [_scalar(field, value) for value in raw[source_row, :]])
        assert actual == reference
        expected_kernel = reference.right_kernel().basis_matrix().transpose()
        inclusion = _native(field, model.kernel.lifts[degree])
        assert inclusion.rank() == inclusion.ncols() == expected_kernel.ncols()
        assert inclusion.augment(expected_kernel).rank() == inclusion.ncols()
        assert reference * inclusion == 0
        boundary = _source_boundary(field, tensor.degree, degree)
        assert _native(field, model.source.differential(degree)) == boundary
        if degree:
            previous = _native(field, model.kernel.lifts[degree - 1])
            assert boundary * inclusion == previous * _native(
                field, model.kernel.differential(degree))
        projection = _native(field, model.projection.matrix(degree))
        comparison = _native(field, model.comparison.matrix(degree))
        realization = _native(field, model.realization.matrix(degree))
        assert projection * inclusion == 0
        assert projection.rank() == model.quotient.dimension(degree)
        assert comparison.rank() == model.quotient.dimension(degree)
        assert comparison * projection == realization


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("tensor", [
    _cases()[2],
    SparseTensor.zero(1, (0, 0)),
    SparseTensor(2, (0, 0, 0), {(0, 1, 2): 1}),
    SparseTensor(1, (1, 1), {(0, 1): 1, (1, 2): -1, (2, 2): 2}),
])
def test_realized_tensor_has_requested_kernel_in_native_sage(tensor, modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    original = kernel_sequence(tensor, modulus=modulus)
    realized = realize_kernel(original.inclusion, tensor.shifts)
    assert realized.degree == tensor.degree
    assert realized.shifts == tensor.shifts
    for degree in range(tensor.degree + 1):
        _, reference = _reference(field, realized, degree)
        inclusion = _native(field, original.kernel.lifts[degree])
        assert reference * inclusion == 0
        assert reference.rank() + inclusion.ncols() == reference.ncols()


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_specified_quotient_isomorphism_lift_commutes_in_native_sage(modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    tensor = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 2]], dtype=object)
    left, right = (kernel_sequence(values, modulus=modulus) for values in (tensor, tensor + 1))
    phi = chain_isomorphism(left.quotient, right.quotient)
    lifted = lift_quotient_isomorphism(left.projection, right.projection, phi)
    for degree, dimension in enumerate(left.source.dimensions):
        matrix = _native(field, lifted.matrix(degree))
        assert matrix.rank() == dimension
        assert _native(field, right.projection.matrix(degree)) * matrix == (
            _native(field, phi.matrix(degree)) * _native(field, left.projection.matrix(degree)))
        image = matrix * _native(field, left.kernel.lifts[degree])
        target = _native(field, right.kernel.lifts[degree])
        assert image.augment(target).rank() == target.ncols()


def test_native_reference_projects_injective_columns_independently():
    # The reference deliberately includes degenerate source maps before
    # projecting. This guards its indexing convention without calling the API.
    for n in range(4):
        for degree in range(n + 1):
            all_maps = set(combinations_with_replacement(range(n + 1), degree + 1))
            for vertices in combinations(range(n + 1), degree + 1):
                projected = _projected_simplex(vertices)
                assert set(projected) <= all_maps
                assert projected[vertices] == 1
