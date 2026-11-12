"""Exact invariants and homology calculations for ordered simple graphs."""

from collections import Counter
from collections.abc import Iterable
from functools import reduce
from itertools import combinations, permutations
from math import gcd

import numpy as np
import sympy as sp

from .chain_complex import ExactField
from .homology import GeneratedHomology
from .rank_shifted import GraphEmbedding, SparseTensor
from .tensor_ops import bdry


def _graph_tensors(embedding: GraphEmbedding, field: ExactField):
    """Validate tensor types and coefficients without allocating ambient arrays."""
    if not isinstance(embedding, GraphEmbedding):
        raise TypeError("Supply a GraphEmbedding tensor model.")
    vertices, edges = tuple(embedding.vertices), tuple(embedding.edge_tensors)
    shifts = None
    for degree, tensors in ((0, vertices), (1, edges)):
        for tensor in tensors:
            if not isinstance(tensor, SparseTensor) or tensor.degree != degree:
                raise ValueError(f"Graph tensors in degree {degree} must be SparseTensor objects of that degree.")
            if shifts is None:
                shifts = tensor.shifts
            elif tensor.shifts != shifts:
                raise ValueError("Graph tensors must have the same axis shifts.")
            for coefficient in tensor.entries.values():
                field.scalar(coefficient)
    return vertices, edges


def graph_incidence_matrix(
    embedding: GraphEmbedding, *, modulus: int | None = None,
) -> sp.Matrix:
    """Return normalized edge boundaries in the supplied vertex basis.

    Solve V * X = B over Q, or over F_p for a prime modulus. Vertex
    tensors must be independent in that field, and every normalized edge
    boundary must lie in their span. Rational denominators must be invertible
    in the selected field. Finite-field entries are canonical integer residues.
    """
    field = ExactField(modulus)
    vertices, edges = _graph_tensors(embedding, field)
    boundaries = tuple(edge.normalize().face(0) for edge in edges)
    coordinates = sorted({index for tensor in vertices + boundaries for index in tensor.entries})
    vertex_basis = field.matrix(sp.Matrix(
        len(coordinates), len(vertices),
        lambda row, column: vertices[column].entries.get(coordinates[row], 0),
    ))
    boundary_columns = field.matrix(sp.Matrix(
        len(coordinates), len(boundaries),
        lambda row, column: boundaries[column].entries.get(coordinates[row], 0),
    ))
    if field.rank(vertex_basis) != len(vertices):
        raise ValueError("Vertex tensors must be independent in the coefficient field.")
    try:
        incidence = field.coordinates(vertex_basis, boundary_columns)
    except ValueError as exc:
        raise ValueError("Normalized edge boundaries must lie in the vertex span.") from exc
    return sp.Matrix(incidence)


def _graph_components_if_incidence_matches(embedding: GraphEmbedding) -> int | None:
    """Count components when sparse tensor boundaries match the graph data."""
    if any(not isinstance(values, (tuple, list)) for values in
           (embedding.vertices, embedding.edges, embedding.edge_tensors)):
        return None
    count = embedding.vertex_count
    if (not isinstance(count, (int, np.integer))
            or isinstance(count, (bool, np.bool_)) or count < 0):
        return None
    count = int(count)
    if len(embedding.vertices) != count or len(embedding.edge_tensors) != len(embedding.edges):
        return None
    coordinates = set()
    for vertex in embedding.vertices:
        if not isinstance(vertex, SparseTensor) or vertex.degree != 0 or len(vertex.entries) != 1:
            return None
        coordinate, coefficient = next(iter(vertex.entries.items()))
        if coefficient != 1 or coordinate in coordinates:
            return None
        coordinates.add(coordinate)
    parents = list(range(count))

    def find(vertex):
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    for edge, tensor in zip(embedding.edges, embedding.edge_tensors):
        if (not isinstance(edge, (tuple, list)) or len(edge) != 2
                or not isinstance(tensor, SparseTensor) or tensor.degree != 1):
            return None
        source, target = edge
        if any(not isinstance(vertex, (int, np.integer))
               or isinstance(vertex, (bool, np.bool_)) or not 1 <= vertex <= count
               for vertex in edge):
            return None
        source, target = int(source) - 1, int(target) - 1
        try:
            expected = embedding.vertices[target] - embedding.vertices[source]
            if tensor.normalize().face(0) != expected:
                return None
        except (TypeError, ValueError):
            return None
        parents[find(source)] = find(target)
    return len({find(vertex) for vertex in range(count)})


def graph_chain_homology(
    embedding: GraphEmbedding, *, modulus: int | None = None,
) -> GeneratedHomology:
    """Return dimensions and ranks of the normalized tensor-incidence complex.

    Coefficients lie in Q or a prime field. Canonical graph_embedding
    models give ordinary graph homology; modified tensor models retain
    one degree-one basis element per supplied edge tensor.
    """
    field = ExactField(modulus)
    _graph_tensors(embedding, field)
    components = _graph_components_if_incidence_matches(embedding)
    if components is not None:
        vertices, edges = len(embedding.vertices), len(embedding.edge_tensors)
        rank = vertices - components
    else:
        incidence = graph_incidence_matrix(embedding, modulus=field.modulus)
        vertices, edges = incidence.shape
        rank = field.rank(incidence)
    return GeneratedHomology(
        (vertices, edges), (0, rank), (vertices - rank, edges - rank),
    )


def _adjacency(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a nonempty square adjacency matrix.")
    if (
        np.any((matrix != 0) & (matrix != 1))
        or np.any(matrix.diagonal())
        or not np.array_equal(matrix, matrix.T)
    ):
        raise ValueError("Expected the adjacency matrix of a simple undirected graph.")
    return matrix


def adjacency_h1_relations(matrix: np.ndarray) -> int:
    """Give the generator of the relation ideal of integral H_1.

    An edgeless graph generates the zero module. Otherwise the normalized
    chains in degree one are generated by E_01+E_10, and H_1 is Z/(g),
    where this function returns the nonnegative integer g. All other
    normalized homology groups vanish.
    """
    entries = _adjacency(matrix).astype(int)
    coefficients = (
        int(entries[j, k] - entries[i, k] + entries[i, j])
        for i, j, k in combinations(range(len(entries)), 3)
    )
    return reduce(gcd, coefficients, 0)


def _binary_basis(rows: Iterable[int]) -> tuple[int, ...]:
    pivots: dict[int, int] = {}
    for row in rows:
        row = int(row)
        if row < 0:
            raise ValueError("A packed binary row must be nonnegative.")
        while row:
            leading = row.bit_length() - 1
            if leading not in pivots:
                pivots[leading] = row
                break
            row ^= pivots[leading]
    return tuple(pivots.values())


def rank_binary_rows(rows: Iterable[int]) -> int:
    """Rank over F_2 of row vectors encoded as nonnegative bit masks."""
    return len(_binary_basis(rows))


def deletion_homology_mod2(matrix: np.ndarray) -> GeneratedHomology:
    """Homology of the deletion-only chains of one ordered graph over F_2.

    In degree q the span consists of the induced matrices on q+1 vertices.
    Use their strictly upper triangles as coordinates. The differential
    is the sum of ordered principal deletions and is zero in degree zero.
    """
    adjacency = _adjacency(matrix).astype(bool).tolist()
    dimensions = [0]
    ranks = [0]
    for size in range(2, len(adjacency) + 1):
        pairs = list(combinations(range(size), 2))
        generators = (
            sum(1 << j for j, (a, b) in enumerate(pairs) if adjacency[vertices[a]][vertices[b]])
            for vertices in combinations(range(len(adjacency)), size)
        )
        basis = _binary_basis(generators)
        dimensions.append(len(basis))
        upper = np.triu_indices(size, 1)
        image_upper = np.triu_indices(size - 1, 1)
        images = []
        for row in basis:
            representative = np.zeros((size, size), dtype=bool)
            representative[upper] = [bool(row & (1 << j)) for j in range(len(pairs))]
            representative |= representative.T
            coordinates = bdry(representative)[image_upper]
            images.append(sum(1 << j for j in np.flatnonzero(coordinates).tolist()))
        ranks.append(rank_binary_rows(images))
    homology = tuple(
        dimension - incoming - outgoing
        for dimension, incoming, outgoing in zip(dimensions, ranks, ranks[1:] + [0])
    )
    return GeneratedHomology(tuple(dimensions), tuple(ranks), homology)


def boundary_rank_distribution(matrix: np.ndarray) -> dict[int, int]:
    """Count mod-two boundary matrix ranks over all n! vertex orderings.

    Here a permutation tuple lists the old vertices in the new order. Use
    this full enumeration for small graphs.
    """
    adjacency = _adjacency(matrix).astype(bool)
    counts: Counter[int] = Counter()
    for order in permutations(range(len(adjacency))):
        boundary = bdry(adjacency[np.ix_(order, order)])
        rows = (sum(1 << j for j in np.flatnonzero(row).tolist()) for row in boundary)
        counts[rank_binary_rows(rows)] += 1
    return dict(sorted(counts.items()))
