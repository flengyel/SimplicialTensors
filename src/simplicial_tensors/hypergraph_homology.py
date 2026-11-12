"""Embedded hypergraph homology over Q and prime fields.

A hyperedge is a nonempty finite set of hashable, mutually orderable vertex
labels. Repeated hyperedges and vertices within an edge are ignored. The
vertex set is the union of the supplied edges; singleton hyperedges are
never inserted automatically. To include an isolated vertex, supply its
singleton hyperedge. The empty hypergraph has the empty chain complex.
"""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from .chain_complex import ChainComplex, ExactField
from .homology import GeneratedHomology

__all__ = ["associated_simplicial_complex", "embedded_homology", "embedded_chain_complex"]


def _canonical_edges(hyperedges):
    edges = set()
    vertices = set()
    for edge in hyperedges:
        if isinstance(edge, (str, bytes)):
            raise TypeError("Each hyperedge must be an iterable of vertex labels.")
        try:
            edge = frozenset(edge)
        except TypeError as exc:
            raise TypeError("Hyperedges must contain hashable vertex labels.") from exc
        if not edge:
            raise ValueError("Hyperedges must be nonempty.")
        edges.add(edge)
        vertices.update(edge)
    try:
        order = {vertex: index for index, vertex in enumerate(sorted(vertices))}
    except TypeError as exc:
        raise TypeError("Vertex labels must admit a common total order.") from exc
    canonical = [tuple(sorted(edge, key=order.__getitem__)) for edge in edges]
    return tuple(sorted(canonical, key=lambda edge: (len(edge), tuple(order[v] for v in edge))))


def associated_simplicial_complex(hyperedges):
    """Return all nonempty subsets of the hyperedges, in dimension/vertex order."""
    edges = _canonical_edges(hyperedges)
    return _canonical_edges({face for edge in edges for size in range(1, len(edge) + 1)
                             for face in combinations(edge, size)})


def _graded(edges):
    return tuple(tuple(edge for edge in edges if len(edge) == size)
                 for size in range(1, max((len(edge) for edge in edges), default=0) + 1))


def _boundary_columns(edges, target_basis):
    """Alternating vertex-deletion boundary in an explicitly supplied basis."""
    target_rows = {face: row for row, face in enumerate(target_basis)}
    result = sp.MutableSparseMatrix(len(target_basis), len(edges), {})
    for column, edge in enumerate(edges):
        for omitted in range(len(edge)):
            facet = edge[:omitted] + edge[omitted + 1:]
            if facet in target_rows:
                result[target_rows[facet], column] += (-1)**omitted
    return sp.ImmutableSparseMatrix(result)


def _data(hyperedges, modulus):
    field = ExactField(modulus)
    edges = _canonical_edges(hyperedges)
    graded = _graded(edges)
    boundaries, missing_rows, included_rows = [], [], []
    for degree, basis in enumerate(graded):
        # Only immediate facets enter the boundary constraints. Actual lower
        # hyperedges retain zero rows needed by the lift coordinate maps.
        target = (_canonical_edges([
            *graded[degree - 1],
            *(edge[:i] + edge[i + 1:] for edge in basis for i in range(len(edge))),
        ]) if degree else ())
        boundary = _boundary_columns(basis, target)
        previous = set(graded[degree - 1]) if degree else set()
        included_rows.append(tuple(row for row, face in enumerate(target) if face in previous))
        missing_rows.append(tuple(row for row, face in enumerate(target) if face not in previous))
        boundaries.append(boundary)
    return field, graded, boundaries, missing_rows, included_rows


def embedded_homology(hyperedges, *, modulus: int | None = None) -> GeneratedHomology:
    """Return dimensions and ranks of the infimum complex, without bases.

    Inside the simplicial chains of the downward closure, let D_n be the
    span of the supplied (n+1)-vertex hyperedges. The infimum complex is
    Inf_n = D_n intersect boundary_n^(-1)(D_(n-1)), with boundary zero in
    degree zero. Joint missing-face constraints allow cancellations between
    different hyperedges. All ranks are taken in the specified field.
    """
    field, graded, boundaries, missing_rows, _ = _data(hyperedges, modulus)
    dimensions, ranks = [], []
    for basis, boundary, missing in zip(graded, boundaries, missing_rows):
        missing_rank = field.rank(boundary.extract(missing, range(boundary.cols)))
        dimensions.append(len(basis) - missing_rank)
        ranks.append(field.rank(boundary) - missing_rank)
    incoming = ranks[1:] + [0] if ranks else []
    betti = tuple(dimension - outgoing - following
                  for dimension, outgoing, following in zip(dimensions, ranks, incoming))
    return GeneratedHomology(tuple(dimensions), tuple(ranks), betti)


def embedded_chain_complex(hyperedges, *, modulus: int | None = None) -> ChainComplex:
    """Return an exact based infimum complex with lifts to hyperedge coordinates.

    ``ambient_bases[n]`` lists the actual hyperedges, not every face in their
    downward closure. ``lifts[n]`` expresses each chain basis vector as a
    linear combination of those hyperedges. Differentials and cycle or
    homology representatives are in the resulting chain coordinates.
    """
    field, graded, boundaries, missing_rows, included_rows = _data(hyperedges, modulus)
    lifts, differentials, dimensions = [], [], []
    for degree, boundary in enumerate(boundaries):
        constraints = boundary.extract(missing_rows[degree], range(boundary.cols))
        lift = field.nullspace(constraints)
        lifts.append(lift)
        dimensions.append(lift.cols)
        if degree:
            allowed = boundary.extract(included_rows[degree], range(boundary.cols))
            image = field.matmul(allowed, lift)
            differentials.append(field.coordinates(lifts[degree - 1], image))
        else:
            differentials.append(sp.ImmutableMatrix.zeros(0, lift.cols))
    return ChainComplex(tuple(dimensions), tuple(differentials), modulus=field.modulus,
                        lifts=tuple(lifts), ambient_bases=graded)
