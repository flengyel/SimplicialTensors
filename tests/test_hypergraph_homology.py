"""Embedded homology against independently assembled supremum complexes."""

from fractions import Fraction
from itertools import combinations, permutations

import pytest
import sympy as sp

from simplicial_tensors.hypergraph_homology import (
    associated_simplicial_complex, embedded_chain_complex, embedded_homology,
)


def _rank(columns, modulus):
    """Independent Gaussian elimination; columns use ordinary rational entries."""
    if not columns:
        return 0
    def coefficient(value):
        rational = sp.Rational(value)
        return (Fraction(int(rational.p), int(rational.q)) if modulus is None else
                int(rational.p) * pow(int(rational.q), -1, modulus) % modulus)
    rows = [[coefficient(column[row]) for column in columns] for row in range(len(columns[0]))]
    pivot_row = 0
    for column in range(len(columns)):
        pivot = next((row for row in range(pivot_row, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        value = rows[pivot_row][column]
        inverse = 1 / value if modulus is None else pow(value, -1, modulus)
        rows[pivot_row] = [entry * inverse for entry in rows[pivot_row]]
        if modulus is not None:
            rows[pivot_row] = [entry % modulus for entry in rows[pivot_row]]
        for row in range(len(rows)):
            if row != pivot_row:
                multiple = rows[row][column]
                rows[row] = [entry - multiple * upper for entry, upper in zip(rows[row], rows[pivot_row])]
                if modulus is not None:
                    rows[row] = [entry % modulus for entry in rows[row]]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def _supremum_reference(hyperedges, modulus):
    """Use D_n + boundary(D_(n+1)), rather than infimum nullspaces."""
    hyperedges = set(map(tuple, hyperedges))
    top = max(map(len, hyperedges), default=0)
    closure = [sorted({face for edge in hyperedges for face in combinations(edge, size)})
               for size in range(1, top + 1)]
    graded = [[edge for edge in sorted(hyperedges) if len(edge) == size]
              for size in range(1, top + 1)]
    spans = []
    for degree, basis in enumerate(closure):
        columns = [[int(face == edge) for face in basis] for edge in graded[degree]]
        for edge in graded[degree + 1] if degree + 1 < top else []:
            columns.append([sum((-1)**i for i in range(len(edge))
                                if edge[:i] + edge[i + 1:] == face) for face in basis])
        spans.append(columns)
    dimensions = [_rank(columns, modulus) for columns in spans]
    ranks = [0] if top else []
    for degree in range(1, top):
        images = []
        for column in spans[degree]:
            image = []
            for face in closure[degree - 1]:
                image.append(sum(value * (-1)**i for edge, value in zip(closure[degree], column)
                                 for i in range(len(edge)) if edge[:i] + edge[i + 1:] == face))
            images.append(image)
        ranks.append(_rank(images, modulus))
    incoming = ranks[1:] + [0] if ranks else []
    return tuple(dimension - outgoing - following
                 for dimension, outgoing, following in zip(dimensions, ranks, incoming))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_all_127_hypergraphs_on_three_labels_against_supremum_reference(modulus):
    possible = [edge for size in range(1, 4) for edge in combinations(range(3), size)]
    for mask in range(1, 128):
        hyperedges = [edge for index, edge in enumerate(possible) if mask & (1 << index)]
        expected = _supremum_reference(hyperedges, modulus)
        summary = embedded_homology(hyperedges, modulus=modulus)
        model = embedded_chain_complex(hyperedges, modulus=modulus)
        assert summary.homology_dimensions == expected, (mask, modulus)
        assert model.homology() == summary
        for degree, betti in enumerate(expected):
            representatives = model.homology_representatives(degree)
            assert representatives.cols == betti
            assert not any(model.field.matmul(model.differential(degree), representatives))
            assert model.field.rank(model.boundaries(degree).row_join(representatives)) == (
                model.boundaries(degree).cols + betti
            )


CANCELLATION = [(1,), (2,), (3,), (4,), (1, 2), (2, 3), (3, 4), (1, 4), (1, 2, 3), (1, 3, 4)]


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_joint_missing_face_constraints_keep_a_combination_of_hyperedges(modulus):
    model = embedded_chain_complex(CANCELLATION, modulus=modulus)
    assert model.homology().chain_dimensions == (4, 4, 1)
    assert model.homology().boundary_ranks == (0, 3, 1)
    assert model.homology().homology_dimensions == (1, 0, 0)
    assert model.ambient_bases[2] == ((1, 2, 3), (1, 3, 4))
    lift = model.lifts[2]
    assert model.field.rank(lift) == 1
    assert model.field.rank(lift.row_join(sp.ones(2, 1))) == 1
    assert all(value != 0 for value in lift)
    assert model.field.rank(model.differentials[2]) == 1


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_projecting_away_missing_faces_would_not_give_a_chain_complex(modulus):
    # Keeping just face12 of boundary123 gives boundary12 = vertex2-vertex1.
    projected_d1 = sp.Matrix([[-1], [1], [0]])
    projected_d2 = sp.Matrix([[1]])
    assert any(projected_d1 * projected_d2)
    hyperedges = [(1,), (2,), (3,), (1, 2), (1, 2, 3)]
    model = embedded_chain_complex(hyperedges, modulus=modulus)
    assert model.dimensions == (3, 1, 0)
    assert model.homology().homology_dimensions == (2, 0, 0)
    assert not any(model.field.matmul(model.differentials[1], model.differentials[2]))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_singletons_are_explicit_and_empty_input_is_well_typed(modulus):
    assert embedded_homology([], modulus=modulus).homology_dimensions == ()
    assert embedded_chain_complex([], modulus=modulus).homology().chain_dimensions == ()
    assert embedded_homology([(1, 2, 3)], modulus=modulus).homology_dimensions == (0, 0, 0)
    edges = [(1,), (2,), (3,), (1, 2, 3)]
    assert embedded_homology(edges, modulus=modulus).homology_dimensions == (3, 0, 0)
    closure = associated_simplicial_complex(edges)
    assert embedded_homology(closure, modulus=modulus).homology_dimensions == (1, 0, 0)
    assert embedded_homology([(1, 1), (1,)], modulus=modulus).homology_dimensions == (1,)
    with pytest.raises(ValueError, match="nonempty"):
        embedded_homology([()])
    with pytest.raises(TypeError, match="total order"):
        embedded_homology([(1, "a")])
    with pytest.raises(TypeError, match="hashable"):
        embedded_homology([([1],)])


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_relabeling_preserves_the_embedded_homology(modulus):
    expected = embedded_homology(CANCELLATION, modulus=modulus)
    for labels in permutations(range(4)):
        relabeled = [tuple(labels[vertex - 1] for vertex in edge) for edge in CANCELLATION]
        assert embedded_homology(relabeled, modulus=modulus) == expected
    names = {1: "d", 2: "b", 3: "a", 4: "c"}
    assert embedded_homology([[names[v] for v in edge] for edge in CANCELLATION], modulus=modulus) == expected


@pytest.mark.parametrize("modulus,betti", [(None, (1, 0, 0)), (2, (1, 1, 1)), (3, (1, 0, 0))])
def test_six_vertex_facet_fixture_has_field_dependent_ranks(modulus, betti):
    facets = [(1, 2, 3), (1, 2, 4), (1, 3, 5), (1, 4, 6), (1, 5, 6),
              (2, 3, 6), (2, 4, 5), (2, 5, 6), (3, 4, 5), (3, 4, 6)]
    hyperedges = associated_simplicial_complex(facets)
    assert _supremum_reference(hyperedges, modulus) == betti
    summary = embedded_homology(hyperedges, modulus=modulus)
    assert summary.homology_dimensions == betti
    assert embedded_chain_complex(hyperedges, modulus=modulus).homology() == summary
    # Without singleton/pair hyperedges, a characteristic-two 2-cycle remains.
    expected = (0, 0, int(modulus == 2))
    assert embedded_homology(facets, modulus=modulus).homology_dimensions == expected


def test_one_large_hyperedge_does_not_require_its_entire_downward_closure():
    # There are 2**32-1 faces in the closure but only 32 immediate facets.
    hyperedges = [tuple(range(32))]
    zero = (0,) * 32
    summary = embedded_homology(hyperedges)
    assert summary.chain_dimensions == zero
    assert summary.boundary_ranks == zero
    assert summary.homology_dimensions == zero
    assert embedded_chain_complex(hyperedges).homology() == summary


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_uniform_hypergraph_fast_summary_preserves_sparse_boundaries(monkeypatch, modulus):
    from simplicial_tensors.chain_complex import ExactField
    calls = []
    original = ExactField.scalar
    def record(field, value):
        calls.append(value)
        return original(field, value)
    monkeypatch.setattr(ExactField, "scalar", record)
    edges = list(combinations(range(10), 3))
    summary = embedded_homology(edges, modulus=modulus)
    assert summary.homology_dimensions == (0, 0, 84)
    # Full and missing-face ranks each inspect the three nonzero facets per edge.
    assert len(calls) <= 2 * 3 * len(edges)
