"""Native graph coordinates from independent raw tensor boundaries."""

from dataclasses import replace
from fractions import Fraction

import pytest

sage = pytest.importorskip("sage.all")

from simplicial_tensors.graph_ops import graph_chain_homology, graph_incidence_matrix
from simplicial_tensors.rank_shifted import GraphEmbedding, SparseTensor, graph_embedding


def _rebased_triangle(reverse_support=False):
    base = graph_embedding(3, [(1, 2), (2, 3), (3, 1)])
    change = ((1, 1, 1), (1, 2, 3), (1, 3, 6))  # determinant one
    vertices = []
    for column in range(3):
        value = sum((base.vertices[row] * Fraction(change[row][column], 7)
                     for row in range(3)), SparseTensor.zero(0, (3, 3, 3)))
        if reverse_support:
            value = SparseTensor(0, value.shifts, dict(reversed(tuple(value.entries.items()))))
        vertices.append(value)
    edges = []
    for j, (source, target) in enumerate(base.edges):
        difference = vertices[target - 1] - vertices[source - 1]
        section = SparseTensor(1, difference.shifts,
                               {tuple(x + 1 for x in index): value
                                for index, value in difference.entries.items()})
        edges.append(vertices[source - 1].degeneracy(0) + section
                     + SparseTensor.unit(1, (3, 3, 3), (0, 1, j + 2)))
    return GraphEmbedding(3, base.edges, 3, tuple(vertices), tuple(edges))


def _raw_boundary(tensor):
    # Independent degree-one alternating deletion, not package normalization.
    result = {}
    for removed in (0, 1):
        for index, value in tensor.entries.items():
            if removed in index:
                continue
            target = tuple(x - (x > removed) for x in index)
            result[target] = result.get(target, Fraction(0)) + (-1)**removed * value
    return {index: value for index, value in result.items() if value}


def _reference(model, modulus):
    field = sage.QQ if modulus is None else sage.GF(modulus)
    boundaries = tuple(_raw_boundary(edge) for edge in model.edge_tensors)
    rows = sorted({index for vertex in model.vertices for index in vertex.entries}
                  | {index for boundary in boundaries for index in boundary})

    def scalar(value):
        value = Fraction(value)
        return field(value.numerator) / field(value.denominator)

    vertices = sage.matrix(field, len(rows), len(model.vertices),
                           lambda i, j: scalar(model.vertices[j].entries.get(rows[i], 0)))
    images = sage.matrix(field, len(rows), len(boundaries),
                         lambda i, j: scalar(boundaries[j].get(rows[i], 0)))
    assert vertices.rank() == len(model.vertices)
    return vertices.solve_right(images)


@pytest.mark.parametrize("modulus", [None, 2, 3, 5, 11])
@pytest.mark.parametrize("reverse_support", [False, True])
def test_rebased_rational_graph_coordinates_match_native_sage(modulus, reverse_support):
    model = _rebased_triangle(reverse_support)
    expected = _reference(model, modulus)
    actual = graph_incidence_matrix(model, modulus=modulus)
    field = expected.base_ring()
    native = sage.matrix(field, actual.rows, actual.cols,
                         [field(sage.QQ(str(value))) for value in actual])
    assert native == expected
    assert graph_chain_homology(model, modulus=modulus).homology_dimensions == (1, 1)


@pytest.mark.parametrize("modulus,coefficient", [(3, Fraction(1, 2)),
                                                (3, Fraction(3, 2)),
                                                (2, Fraction(1, 3))])
def test_modified_rational_edge_rank_matches_native_sage(modulus, coefficient):
    model = graph_embedding(2, [(1, 2)])
    model = replace(model, edge_tensors=(coefficient * model.edge_tensors[0],))
    expected = _reference(model, modulus)
    rank = int(expected.rank())
    result = graph_chain_homology(model, modulus=modulus)
    assert result.boundary_ranks == (0, rank)
    assert result.homology_dimensions == (2 - rank, 1 - rank)


def test_invalid_marker_denominator_is_checked_before_graph_fast_path():
    model = graph_embedding(2, [(1, 2)])
    invisible = SparseTensor.unit(1, model.vertices[0].shifts, (0, 1, 2), Fraction(1, 2))
    # Both faces vanish, so incidence/component checks alone cannot see this.
    modified = replace(model, edge_tensors=(model.edge_tensors[0] + invisible,))
    with pytest.raises(ValueError, match="denominator"):
        graph_chain_homology(modified, modulus=2)
