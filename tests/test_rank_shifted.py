"""Exact checks for shifted simplicial tensors and their graph models."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import sympy as sp

from simplicial_tensors.rank_shifted import (
    SparseTensor,
    circle_generator,
    graph_embedding,
    petersen_edges,
)
from simplicial_tensors.tensor_ops import degen as dense_degeneracy
from simplicial_tensors.tensor_ops import face as dense_face


def zero(degree, shifts):
    return SparseTensor(degree, shifts, {})


def sample(degree, shifts):
    """Use every coordinate, including fixed-tail coordinates."""
    shape = tuple(degree + shift + 1 for shift in shifts)
    return SparseTensor(
        degree,
        shifts,
        {
            index: Fraction((position * 7) % 11 - 5, (position % 3) + 1)
            for position, index in enumerate(product(*(range(n) for n in shape)))
        },
    )


def dense(tensor):
    result = np.zeros(tensor.shape, dtype=object)
    for index, value in tensor.entries.items():
        result[index] = value
    return result


def rank(tensors):
    if not tensors:
        return 0
    coordinates = sorted(set().union(*(tensor.entries for tensor in tensors)))
    if not coordinates:
        return 0
    return sp.Matrix(
        [
            [sp.Rational(tensor.coefficient(index)) for tensor in tensors]
            for index in coordinates
        ]
    ).rank()


def assert_simplicial_identities(tensor):
    degree = tensor.degree
    if degree >= 2:
        for i in range(degree):
            for j in range(i + 1, degree + 1):
                assert tensor.face(j).face(i) == tensor.face(i).face(j - 1)

    for i in range(degree + 1):
        for j in range(i, degree + 1):
            assert tensor.degeneracy(j).degeneracy(i) == tensor.degeneracy(
                i
            ).degeneracy(j + 1)

    for j in range(degree + 1):
        for i in range(degree + 2):
            left = tensor.degeneracy(j).face(i)
            if i < j:
                right = tensor.face(i).degeneracy(j - 1)
            elif i in (j, j + 1):
                right = tensor
            else:
                right = tensor.face(i - 1).degeneracy(j)
            assert left == right


@pytest.mark.parametrize("degree", range(4))
@pytest.mark.parametrize("shifts", [(0, 2), (1, 0, 1)])
def test_simplicial_identities(degree, shifts):
    assert_simplicial_identities(sample(degree, shifts))


def test_simplicial_identities_on_every_basis_tensor_in_a_small_shifted_module():
    # Basis checks rule out cancellation obscuring an incorrect coordinate map.
    for index in product(range(3), range(4)):
        assert_simplicial_identities(SparseTensor.unit(2, (0, 1), index))


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("shifts", [(1, 2), (0, 1, 2)])
def test_sparse_operators_match_existing_dense_operators(degree, shifts):
    tensor = sample(degree, shifts)
    array = dense(tensor)
    for i in range(degree + 1):
        np.testing.assert_array_equal(dense(tensor.degeneracy(i)), dense_degeneracy(array, i))
    if degree:
        boundary = np.zeros(tuple(n - 1 for n in tensor.shape), dtype=object)
        for i in range(degree + 1):
            expected = dense_face(array, i)
            np.testing.assert_array_equal(dense(tensor.face(i)), expected)
            boundary += (-1) ** i * expected
        # Shifted degree, rather than the smallest array dimension, fixes this sum.
        np.testing.assert_array_equal(dense(tensor.boundary()), boundary)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_normalization_is_a_projection_and_kills_degeneracies(degree):
    shifts = (0, 1, 2)
    tensor = sample(degree, shifts)
    normalized = tensor.normalize()
    assert normalized.normalize() == normalized
    for i in range(1, degree + 1):
        assert normalized.face(i) == zero(degree - 1, shifts)
    lower = sample(degree - 1, shifts)
    for i in range(degree):
        assert lower.degeneracy(i).normalize() == zero(degree, shifts)
    assert normalized.boundary() == normalized.face(0)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_boundary_squared_is_zero(degree):
    tensor = sample(degree, (1, 0, 2))
    assert tensor.boundary().boundary() == zero(degree - 2, tensor.shifts)


def test_exact_coefficients_and_immutable_sparse_storage():
    entries = {(0, 1): Fraction(2, 3), (1, 0): 0}
    tensor = SparseTensor(1, (0, 1), entries)
    entries[(0, 1)] = 99
    assert tensor.shape == (2, 3)
    assert tensor.nnz == 1
    assert tensor.coefficient((0, 1)) == Fraction(2, 3)
    assert tensor.coefficient((1, 0)) == 0
    assert (Fraction(3, 2) * tensor).coefficient((0, 1)) == 1
    assert tensor * 3 == 3 * tensor
    assert tensor + tensor == tensor * 2
    assert tensor - tensor == zero(1, (0, 1))
    with pytest.raises(TypeError):
        tensor.entries[(0, 1)] = 1
    with pytest.raises(AttributeError):
        tensor.degree = 2

    assert SparseTensor.unit(1, (0, 1), (0, 1), Fraction(2, 3)) == tensor
    assert SparseTensor.zero(1, (0, 1)) == tensor * 0
    assert not SparseTensor.zero(1, (0, 1))
    assert bool(tensor)
    assert sum([tensor, tensor]) == 2 * tensor
    assert tensor / 2 == Fraction(1, 2) * tensor
    assert hash(tensor) == hash(SparseTensor(1, (0, 1), {(0, 1): Fraction(2, 3)}))


@pytest.mark.parametrize("index", [-1, 2, 4])
def test_fixed_tail_indices_are_not_simplicial_operator_indices(index):
    tensor = sample(1, (3, 4))
    with pytest.raises((IndexError, ValueError)):
        tensor.face(index)
    with pytest.raises((IndexError, ValueError)):
        tensor.degeneracy(index)


def test_invalid_degrees_shapes_coefficients_and_operations():
    with pytest.raises((ValueError, TypeError)):
        SparseTensor(-1, (0, 0), {})
    with pytest.raises((ValueError, TypeError)):
        SparseTensor(1, (-1, 0), {})
    with pytest.raises((ValueError, TypeError)):
        SparseTensor(1, (), {})
    with pytest.raises((ValueError, IndexError)):
        SparseTensor(1, (0, 0), {(2, 0): 1})
    with pytest.raises((ValueError, IndexError)):
        SparseTensor(1, (0, 0), {(0,): 1})
    with pytest.raises((TypeError, ValueError)):
        SparseTensor(1, (0, 0), {(0, 0): 0.5})
    tensor = sample(1, (1, 1))
    with pytest.raises((TypeError, ValueError)):
        tensor * 0.5
    with pytest.raises(ValueError):
        tensor + sample(2, tensor.shifts)
    with pytest.raises(ValueError):
        tensor + sample(1, (2, 1))
    with pytest.raises(ValueError):
        tensor.outer(sample(2, (0,)))
    with pytest.raises(ValueError):
        tensor.pad_shifts((0, 1))
    with pytest.raises(ValueError):
        tensor.pad_shifts((2,))
    degree_zero = sample(0, (2, 2))
    with pytest.raises((ValueError, IndexError)):
        degree_zero.face(0)
    with pytest.raises((ValueError, IndexError)):
        degree_zero.boundary()


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_outer_product_and_padding_commute_with_simplicial_maps(degree):
    left = sample(degree, (0, 1))
    right = sample(degree, (2,))
    joined = left.outer(right)
    assert joined.shifts == (0, 1, 2)
    assert joined.nnz == left.nnz * right.nnz
    padded = left.pad_shifts((2, 3))
    assert padded.nnz == left.nnz
    for index, value in left.entries.items():
        assert padded.coefficient(index) == value
    for i in range(degree + 1):
        assert joined.degeneracy(i) == left.degeneracy(i).outer(right.degeneracy(i))
        assert padded.degeneracy(i) == left.degeneracy(i).pad_shifts((2, 3))
        if degree:
            assert joined.face(i) == left.face(i).outer(right.face(i))
            assert padded.face(i) == left.face(i).pad_shifts((2, 3))


def test_petersen_graph_embedding_has_exact_incidence_complex():
    edges = petersen_edges()
    assert len(edges) == 15
    assert len({frozenset(edge) for edge in edges}) == 15
    assert all(sum(vertex in edge for edge in edges) == 3 for vertex in range(1, 11))
    graph = graph_embedding(10, edges)
    assert rank(graph.vertices) == 10
    assert rank(graph.normalized_edges) == 15
    for (source, target), tensor, normalized in zip(
        edges, graph.edge_tensors, graph.normalized_edges
    ):
        assert tensor.face(1) == graph.vertices[source - 1]
        assert tensor.face(0) == graph.vertices[target - 1]
        assert normalized == tensor.normalize()
        assert normalized.boundary() == graph.vertices[target - 1] - graph.vertices[source - 1]
    boundary_rank = rank([tensor.boundary() for tensor in graph.normalized_edges])
    assert boundary_rank == 9
    assert len(graph.vertices) - boundary_rank == 1
    assert len(graph.normalized_edges) - boundary_rank == 6


def test_graph_embedding_retains_loops_parallel_edges_and_isolated_vertices():
    graph = graph_embedding(4, [(1, 2), (1, 2), (2, 2)])
    assert rank(graph.vertices) == 4
    assert rank(graph.normalized_edges) == 3
    assert graph.edge_tensors[0] != graph.edge_tensors[1]
    boundaries = [edge.boundary() for edge in graph.normalized_edges]
    assert rank(boundaries) == 1
    assert boundaries[2] == zero(0, graph.vertices[0].shifts)
    # Three components (two isolated vertices) and two independent cycles.
    assert rank(graph.vertices) - rank(boundaries) == 3
    assert rank(graph.normalized_edges) - rank(boundaries) == 2


def test_graph_embedding_empty_graphs_and_invalid_endpoints():
    empty = graph_embedding(0, [])
    assert empty.vertices == empty.edge_tensors == empty.normalized_edges == ()
    isolated = graph_embedding(3, [])
    assert rank(isolated.vertices) == 3
    assert isolated.normalized_edges == ()
    for edges in [[(0, 1)], [(1, 3)], [(1,)], [(1, 1, 1)]]:
        with pytest.raises(ValueError):
            graph_embedding(2, edges)
    with pytest.raises(ValueError):
        graph_embedding(2, [(1, 2)], shift=1)
    with pytest.raises(ValueError):
        graph_embedding(1, [(1, 1), (1, 1)], shift=1)
    padded = graph_embedding(2, [(1, 2)], shift=4)
    smaller = graph_embedding(2, [(1, 2)])
    assert padded.edge_tensors[0] == smaller.edge_tensors[0].pad_shifts((4, 4, 4))


def circle_basis(degree):
    """Generate all surjective pullbacks of the single nondegenerate edge."""
    if not degree:
        return []
    basis = [circle_generator()]
    for current_degree in range(1, degree):
        distinct = {}
        for tensor in basis:
            for i in range(current_degree + 1):
                degenerate = tensor.degeneracy(i)
                key = tuple(sorted(degenerate.entries.items()))
                distinct[key] = degenerate
        basis = list(distinct.values())
    assert len(basis) == degree
    assert rank(basis) == degree
    return basis


def test_circle_tensor_square_has_reduced_two_sphere_homology():
    normalized_dimensions = []
    boundary_ranks = []
    for degree in range(4):
        basis = circle_basis(degree)
        generators = [left.outer(right).normalize() for left in basis for right in basis]
        normalized_dimensions.append(rank(generators))
        boundary_ranks.append(rank([tensor.boundary() for tensor in generators]) if degree else 0)
    assert normalized_dimensions == [0, 1, 2, 0]
    assert boundary_ranks == [0, 0, 1, 0]
    assert normalized_dimensions[1] - boundary_ranks[1] - boundary_ranks[2] == 0
    assert normalized_dimensions[2] - boundary_ranks[2] - boundary_ranks[3] == 1

    circle = circle_generator()
    shuffle_cycle = circle.degeneracy(1).outer(circle.degeneracy(0)) - circle.degeneracy(
        0
    ).outer(circle.degeneracy(1))
    assert shuffle_cycle.nnz > 0
    assert shuffle_cycle.normalize() == shuffle_cycle
    for i in range(3):
        assert shuffle_cycle.face(i) == zero(1, shuffle_cycle.shifts)
