"""Exact marked-kernel and inverse-realization regressions."""

from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np
import pytest
import sympy as sp

from simplicial_tensors.chain_complex import ChainComplex
from simplicial_tensors.chain_maps import CapacityError, ChainMap
from simplicial_tensors.kernel_sequences import (
    kernel_sequence,
    realize_kernel,
    simplex_chain_complex,
)
from simplicial_tensors.rank_shifted import SparseTensor


def _same_span(field, left, right):
    return (left.rows == right.rows and field.rank(left) == field.rank(right)
            == field.rank(left.row_join(right)))


def _check_sequence(sequence):
    field = sequence.source.field
    assert sequence.inclusion.is_injective()
    assert sequence.comparison.is_isomorphism()
    assert sequence.comparison.compose(sequence.projection).matrices == sequence.realization.matrices
    for q, dimension in enumerate(sequence.source.dimensions):
        inclusion = sequence.inclusion.matrix(q)
        assert sequence.g_matrices[q].cols == dimension
        assert not any(field.matmul(sequence.g_matrices[q], inclusion))
        assert field.rank(sequence.g_matrices[q]) + inclusion.cols == dimension
        assert not any(field.matmul(sequence.projection.matrix(q), inclusion))
        assert field.rank(sequence.projection.matrix(q)) == sequence.quotient.dimension(q)
        assert field.matmul(sequence.projection.matrix(q), sequence.quotient.lifts[q]) == sp.eye(
            sequence.quotient.dimension(q))
        if q < len(sequence.image.dimensions):
            assert field.matmul(sequence.image.lifts[q], sequence.realization.matrix(q)) == sequence.g_matrices[q]
        else:
            assert sequence.g_matrices[q].rows == 0


def _oracle_normalized_pullback(array, vertices):
    """Independent dense deletion/duplication reference, used only in tests."""
    value = np.array(array, dtype=object)
    for deleted in reversed(range(min(value.shape))):
        if deleted not in vertices:
            for axis in range(value.ndim):
                value = np.delete(value, deleted, axis=axis)
    for index in range(len(vertices) - 1, 0, -1):
        face = value
        for axis in range(value.ndim):
            face = np.delete(face, index, axis=axis)
        degenerate = face
        for axis in range(value.ndim):
            indices = list(range(face.shape[axis]))
            indices.insert(index - 1, index - 1)
            degenerate = np.take(degenerate, indices, axis=axis)
        value = value - degenerate
    return value


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("n", range(4))
def test_simplex_source_all_degrees_and_labels(n, modulus):
    source = simplex_chain_complex(n, modulus=modulus)
    assert source.dimensions == tuple(comb(n + 1, q + 1) for q in range(n + 1))
    assert source.ambient_bases == tuple(tuple(combinations(range(n + 1), q + 1))
                                         for q in range(n + 1))
    assert source.homology().homology_dimensions == (1,) + (0,) * n
    for q in range(1, n + 1):
        for column, vertices in enumerate(source.ambient_bases[q]):
            expected = {vertices[:i] + vertices[i + 1:]: (-1) ** i for i in range(q + 1)}
            assert list(source.differential(q)[:, column]) == [
                source.field.scalar(expected.get(row, 0)) for row in source.ambient_bases[q - 1]
            ]


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_kernel_matrices_match_independent_dense_pullbacks(modulus):
    array = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 2]], dtype=object)
    sequence = kernel_sequence(array, modulus=modulus)
    _check_sequence(sequence)
    assert sequence.kernel.dimensions == (2, 0, 0)
    for q, vertices in enumerate(sequence.source.ambient_bases):
        images = [_oracle_normalized_pullback(array, vertex) for vertex in vertices]
        expected = sp.Matrix(len(sequence.tensor_bases[q]), len(vertices), [
            image[position] for position in sequence.tensor_bases[q] for image in images
        ])
        assert sequence.g_matrices[q] == sequence.source.field.matrix(expected)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_marking_distinguishes_same_embedded_image(modulus):
    array = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 2]], dtype=object)
    original = kernel_sequence(array, modulus=modulus)
    translated = kernel_sequence(array + 1, modulus=modulus)
    assert original.image == translated.image
    assert original.kernel.dimensions == translated.kernel.dimensions
    assert not _same_span(original.source.field, original.inclusion.matrix(0),
                          translated.inclusion.matrix(0))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_zero_and_degree_above_tensor_order_are_retained(modulus):
    for tensor in [SparseTensor.zero(3, (0, 0)),
                   SparseTensor(3, (0, 0), {(0, 1): 1, (2, 3): 2})]:
        sequence = kernel_sequence(tensor, modulus=modulus)
        _check_sequence(sequence)
        assert sequence.source.dimensions == (4, 6, 4, 1)
        assert sequence.g_matrices[3].shape == (0, 1)
        assert sequence.kernel.dimension(3) == 1
    zero = kernel_sequence(SparseTensor.zero(3, (0, 0)), modulus=modulus)
    assert zero.kernel.dimensions == zero.source.dimensions
    assert all(value == sp.eye(value.rows) for value in zero.inclusion.matrices)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_exact_coefficients_dense_sparse_and_explicit_grading(modulus):
    huge = Fraction(2**60 + 1, 5)
    array = np.array([[huge, 2], [3, 7]], dtype=object)
    dense = kernel_sequence(array, modulus=modulus)
    sparse = kernel_sequence(SparseTensor(1, (0, 0), {
        index: array[index] for index in np.ndindex(array.shape)
    }), modulus=modulus)
    assert dense.g_matrices == sparse.g_matrices
    assert dense.inclusion.matrices == sparse.inclusion.matrices
    assert dense.g_matrices[0][0, 0] == dense.source.field.scalar(huge)
    shifted = kernel_sequence(SparseTensor(1, (4,), {(0,): 1, (5,): 2}), modulus=modulus)
    assert shifted.source.dimensions == (2, 1)
    assert shifted.tensor.shape == (6,)
    with pytest.raises(ValueError, match="denominator"):
        kernel_sequence(np.array([[Fraction(1, 2)]]), modulus=2)


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("n", range(4))
def test_inverse_realizes_coordinate_spheres(n, modulus):
    tensor = SparseTensor(n, (0,) * (n + 1), {tuple(range(n + 1)): 1})
    expected = kernel_sequence(tensor, modulus=modulus)
    realized = realize_kernel(expected.inclusion, tensor.shifts)
    actual = kernel_sequence(realized, modulus=modulus)
    assert realized.degree == n and realized.shifts == tensor.shifts
    for q in range(n + 1):
        assert _same_span(expected.source.field, expected.inclusion.matrix(q), actual.inclusion.matrix(q))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_inverse_realizes_nonclosed_degree_zero_kernel_and_tail_class(modulus):
    source = simplex_chain_complex(2, modulus=modulus)
    field = source.field
    plane = field.nullspace([[0, 1, 2]])
    kernel = ChainComplex((2, 0, 0), (sp.zeros(0, 2), sp.zeros(2, 0), sp.zeros(0, 0)), modulus)
    inclusion = ChainMap(kernel, source, (plane, sp.zeros(3, 0), sp.zeros(1, 0)))
    tensor = realize_kernel(inclusion, (0, 0))
    actual = kernel_sequence(tensor, modulus=modulus)
    assert _same_span(field, actual.inclusion.matrix(0), plane)
    assert actual.kernel.dimensions == (2, 0, 0)
    # C(Delta1) embeds in a shifted vector module using its nonboundary H0 tail.
    interval = simplex_chain_complex(1, modulus=modulus)
    zero = ChainComplex((0, 0), (sp.zeros(0, 0), sp.zeros(0, 0)), modulus)
    inclusion = ChainMap(zero, interval, (sp.zeros(2, 0), sp.zeros(1, 0)))
    tensor = realize_kernel(inclusion, (1,))
    assert tensor.degree == 1 and tensor.shape == (3,)
    assert kernel_sequence(tensor, modulus=modulus).kernel.dimensions == (0, 0)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_inverse_capacity_obstruction_is_not_just_linear_containment(modulus):
    source = simplex_chain_complex(2, modulus=modulus)
    plane = source.field.nullspace([[1, 1, 1]])
    kernel = ChainComplex((2, 0, 0), (sp.zeros(0, 2), sp.zeros(2, 0), sp.zeros(0, 0)), modulus)
    inclusion = ChainMap(kernel, source, (plane, sp.zeros(3, 0), sp.zeros(1, 0)))
    with pytest.raises(CapacityError) as error:
        realize_kernel(inclusion, (0, 0))
    assert (error.value.degree, error.value.kind, error.value.required, error.value.available) == (1, "cycle", 3, 2)


def test_inverse_avoids_large_ambient_coordinate_allocation():
    tensor = SparseTensor(2, (10**8,) * 3, {(0, 1, 2): 1})
    sequence = kernel_sequence(tensor)
    result = realize_kernel(sequence.inclusion, tensor.shifts)
    assert result.shape == (100000003,) * 3
    assert result.nnz == 1
    actual = kernel_sequence(result)
    assert actual.inclusion.matrices == sequence.inclusion.matrices


def test_inverse_rejects_nonstandard_target_or_noninclusion():
    standard = simplex_chain_complex(1)
    nonstandard = ChainComplex(standard.dimensions, standard.differentials)
    zero = ChainComplex((0, 0), (sp.zeros(0, 0), sp.zeros(0, 0)))
    with pytest.raises(ValueError, match="ordered simplex"):
        realize_kernel(ChainMap(zero, nonstandard, (sp.zeros(2, 0), sp.zeros(1, 0))), (0, 0))
    with pytest.raises(ValueError, match="injective"):
        realize_kernel(ChainMap(standard, standard, (sp.zeros(2, 2), sp.zeros(1, 1))), (0, 0))
    with pytest.raises(TypeError, match="ChainMap"):
        realize_kernel(standard, (0, 0))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_native_sage_marked_kernels_and_reconstructed_faces(modulus):
    sage = pytest.importorskip("sage.all")
    coefficient_field = sage.QQ if modulus is None else sage.GF(modulus)
    array = np.array([[Fraction(2**60 + 1, 5), 1, 0], [0, 1, 1], [1, 1, 2]], dtype=object)
    sequence = kernel_sequence(array, modulus=modulus)
    reconstructed = realize_kernel(sequence.inclusion, (0, 0))
    again = kernel_sequence(reconstructed, modulus=modulus)
    for q, vertices in enumerate(sequence.source.ambient_bases):
        images = [_oracle_normalized_pullback(array, vertex) for vertex in vertices]
        positions = tuple(product(*(range(length) for length in images[0].shape)))
        native = sage.matrix(coefficient_field, len(positions), len(vertices), [
            coefficient_field(sage.QQ(str(images[j][position])))
            for position in positions for j in range(len(vertices))
        ])
        assert native.rank() == sequence.source.dimension(q) - sequence.kernel.dimension(q)
        independent_kernel = native.right_kernel().basis_matrix().transpose()
        candidate = sage.matrix(coefficient_field, sequence.source.dimension(q), sequence.kernel.dimension(q),
                                [coefficient_field(str(value)) for value in sequence.inclusion.matrix(q)])
        assert candidate.column_space() == independent_kernel.column_space()
        assert _same_span(sequence.source.field, again.inclusion.matrix(q), sequence.inclusion.matrix(q))
    # The reconstructed raw simplex has all prescribed normalized pullbacks.
    _check_sequence(again)


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (2, 3, 3)])
def test_inverse_recovers_general_marked_kernels(shape, modulus):
    array = np.array([(index * index + 3 * index + 1) % 7 - 3
                      for index in range(int(np.prod(shape)))], dtype=object).reshape(shape)
    sequence = kernel_sequence(array, modulus=modulus)
    result = realize_kernel(sequence.inclusion, sequence.tensor.shifts)
    actual = kernel_sequence(result, modulus=modulus)
    for q in range(len(sequence.source.dimensions)):
        assert _same_span(sequence.source.field, sequence.inclusion.matrix(q), actual.inclusion.matrix(q))


def test_inverse_selects_nonzero_tail_prefix_without_materializing_axis():
    interval = simplex_chain_complex(1)
    zero = ChainComplex((0, 0), (sp.zeros(0, 0), sp.zeros(0, 0)))
    inclusion = ChainMap(zero, interval, (sp.zeros(2, 0), sp.zeros(1, 0)))
    result = realize_kernel(inclusion, (10**8,))
    assert result.shape == (100000002,)
    assert result.nnz <= 3
    assert kernel_sequence(result).kernel.dimensions == (0, 0)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_native_sage_preparsed_integer_and_exact_rational_inputs(modulus):
    sage = pytest.importorskip("sage.all")
    from simplicial_tensors.chain_complex import ExactField

    native_modulus = None if modulus is None else sage.Integer(modulus)
    field = ExactField(native_modulus)
    assert simplex_chain_complex(sage.Integer(2), modulus=native_modulus).dimensions == (3, 3, 1)
    rational = sage.QQ(2**60 + 1) / 5
    assert field.scalar(rational) == ExactField(modulus).scalar(Fraction(2**60 + 1, 5))
    assert field.scalar(sage.Integer(2)) == field.scalar(2)
    array = np.array([[rational, sage.QQ(1)], [sage.QQ(0), sage.QQ(2)]], dtype=object)
    sequence = kernel_sequence(array, modulus=native_modulus)
    ordinary = kernel_sequence(np.array([[Fraction(2**60 + 1, 5), 1], [0, 2]], dtype=object),
                               modulus=modulus)
    assert sequence.g_matrices == ordinary.g_matrices
    result = realize_kernel(sequence.inclusion, (sage.Integer(0), sage.Integer(0)))
    again = kernel_sequence(result, modulus=native_modulus)
    for q in range(len(sequence.source.dimensions)):
        assert _same_span(field, sequence.inclusion.matrix(q), again.inclusion.matrix(q))
    with pytest.raises(TypeError, match="integers or rationals"):
        field.scalar(sage.RR(1) / 5)
    with pytest.raises(TypeError, match="integer"):
        simplex_chain_complex(sage.RR(2))
