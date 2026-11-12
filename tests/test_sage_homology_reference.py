"""Independent exact Sage ranks for normalized absolute and relative complexes."""

from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np
import pytest

sage = pytest.importorskip("sage.all")

from simplicial_tensors.graph_ops import graph_chain_homology
from simplicial_tensors.homology import generated_homology, relative_horn_homology
from simplicial_tensors.rank_shifted import graph_embedding, petersen_edges
from simplicial_tensors.tensor_ops import bdry, degen, face


def _rank(columns, field):
    """Rank after exact conversion, with no floating-point intermediate."""
    if not columns:
        return 0
    rows = []
    for column in columns:
        row = []
        for value in column.flat:
            rational = Fraction(value)
            row.append(field(rational.numerator) / field(rational.denominator))
        rows.append(row)
    return int(sage.matrix(field, rows).rank())


def _pullbacks(tensor, degree):
    """Generate chains using every monotone map, including nonsurjections."""
    active = min(tensor.shape)
    tails = [tuple(range(active, length)) for length in tensor.shape]
    return [
        tensor[np.ix_(*(vertices + tail for tail in tails))]
        for vertices in combinations_with_replacement(range(active), degree + 1)
    ]


def _quotient_reference(tensor, field, missing=None):
    """Compute C/(D + horn) directly, without the Moore projection."""
    active = min(tensor.shape)
    top = min(active - 1, tensor.ndim)
    included = [] if missing is None else [
        face(tensor, index) for index in range(active) if index != missing
    ]
    dimensions, ranks = [], [0]
    previous_relations, previous_rank = [], 0
    for degree in range(top + 2):
        chains = _pullbacks(tensor, degree)
        degenerates = [] if degree == 0 else [
            degen(column, index)
            for column in _pullbacks(tensor, degree - 1)
            for index in range(degree)
        ]
        horn = [column for facet in included for column in _pullbacks(facet, degree)]
        relations = degenerates + horn
        relation_rank = _rank(relations, field)
        dimensions.append(_rank(chains, field) - relation_rank)
        if degree:
            ranks.append(
                _rank([bdry(column) for column in chains] + previous_relations, field)
                - previous_rank
            )
        previous_relations, previous_rank = relations, relation_rank
    betti = tuple(dimensions[n] - ranks[n] - ranks[n + 1] for n in range(top + 1))
    assert dimensions[-1] == 0
    assert ranks[-1] == 0
    return tuple(dimensions[:-1]), tuple(ranks[:-1]), betti


def _exact_cases():
    # Native Sage coefficients are explicitly converted to the public input types.
    native = [sage.QQ(2**53 + 1), sage.QQ(1) / 5, sage.QQ(-7) / 5]
    large, fifth, negative = [Fraction(int(q.numerator()), int(q.denominator())) for q in native]
    return [
        np.array([[1, 2, 3, 4], [5, 2, 1, 0], [0, 1, 1, 2]], dtype=object),
        np.arange(8, dtype=object).reshape(2, 2, 2),
        np.array([1, 2, 4], dtype=object),
        np.array([[fifth, negative]], dtype=object),
        np.array([[large, large + 1], [large + 1, large + 2]], dtype=object),
        np.array([[0, fifth], [0, 0]], dtype=object),
        np.zeros((3, 3), dtype=object),
        np.array([
            [large, fifth, negative, 2],
            [Fraction(large, 5), 1, 0, Fraction(1, 7)],
            [3, negative, 4, fifth],
        ], dtype=object),
    ]


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("tensor", _exact_cases(), ids=[
    "rectangular", "three-axes", "vector", "degree-zero", "large-exact",
    "rational-circle", "zero", "large-rational-rectangular",
])
def test_generated_and_relative_homology_against_sage_quotient_ranks(tensor, modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    result = generated_homology(tensor, modulus=modulus)
    expected = _quotient_reference(tensor, field)
    assert (result.chain_dimensions, result.boundary_ranks, result.homology_dimensions) == expected
    assert result.boundary_ranks[0] == 0
    if min(tensor.shape) > 1:
        for missing in range(min(tensor.shape)):
            result = relative_horn_homology(tensor, missing, modulus=modulus)
            expected = _quotient_reference(tensor, field, missing)
            assert (result.chain_dimensions, result.boundary_ranks, result.homology_dimensions) == expected
            assert result.boundary_ranks[0] == 0


def _sparse_rank(tensors, field):
    coordinates = sorted({index for tensor in tensors for index in tensor.entries})
    columns = [np.array([tensor.coefficient(index) for index in coordinates], dtype=object)
               for tensor in tensors]
    return _rank(columns, field) if coordinates else 0


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_petersen_shifted_tensor_complex_against_sage_incidence(modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    graph = graph_embedding(10, petersen_edges())
    incidence = sage.matrix(field, 10, 15)
    for column, (source, target) in enumerate(graph.edges):
        incidence[source - 1, column] -= 1
        incidence[target - 1, column] += 1
    assert incidence.rank() == 9
    normalized = list(graph.normalized_edges)
    assert _sparse_rank(list(graph.vertices), field) == 10
    assert _sparse_rank(normalized, field) == 15
    assert _sparse_rank([tensor.face(0) for tensor in normalized], field) == 9
    result = graph_chain_homology(graph, modulus=modulus)
    assert result.chain_dimensions == (10, 15)
    assert result.boundary_ranks == (0, 9)
    assert result.homology_dimensions == (1, 6)


def _ambient_basis(degree):
    shape = (degree + 1, degree + 1)
    return [np.eye((degree + 1)**2, dtype=object)[index].reshape(shape)
            for index in range((degree + 1)**2)]


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_circle_submodule_and_full_shape_module_have_different_homology(modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    circle = np.array([[0, 1], [0, 0]], dtype=object)
    result = generated_homology(circle, modulus=modulus)
    assert _quotient_reference(circle, field) == ((0, 1), (0, 0), (0, 1))
    assert result.homology_dimensions == (0, 1)
    dimensions, ranks = [], [0]
    previous_degenerates, previous_rank = [], 0
    for degree in range(4):
        chains = _ambient_basis(degree)
        degenerates = [] if degree == 0 else [
            degen(column, index)
            for column in _ambient_basis(degree - 1)
            for index in range(degree)
        ]
        degenerate_rank = _rank(degenerates, field)
        dimensions.append(len(chains) - degenerate_rank)
        if degree:
            ranks.append(_rank([bdry(column) for column in chains] + previous_degenerates, field)
                         - previous_rank)
        previous_degenerates, previous_rank = degenerates, degenerate_rank
    assert dimensions == [1, 3, 2, 0]
    assert ranks == [0, 1, 2, 0]
    assert tuple(dimensions[n] - ranks[n] - ranks[n + 1] for n in range(3)) == (0, 0, 0)
