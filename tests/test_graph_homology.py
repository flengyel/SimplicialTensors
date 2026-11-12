"""Graph differentials are read from tensor faces, over Q and prime fields."""

import numpy as np
import pytest
import sympy as sp

from simplicial_tensors.graph_ops import (
    adjacency_h1_relations,
    boundary_rank_distribution,
    deletion_homology_mod2,
    graph_chain_homology,
    graph_incidence_matrix,
)
from simplicial_tensors.homology import generated_homology
from simplicial_tensors.rank_shifted import graph_embedding, petersen_edges


@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("vertices,edges,betti", [
    (3, [(1, 2), (2, 3)], (1, 0)),
    (3, [(1, 2), (2, 3), (3, 1)], (1, 1)),
    (4, [(1, 2), (3, 4)], (2, 0)),
    (4, [(1, 1), (1, 2), (2, 1), (3, 3)], (3, 3)),
    (3, [], (3, 0)),
    (0, [], (0, 0)),
])
def test_known_graphs(vertices, edges, betti, modulus):
    graph = graph_embedding(vertices, edges)
    expected = sp.zeros(vertices, len(edges))
    for column, (source, target) in enumerate(edges):
        expected[source - 1, column] -= 1
        expected[target - 1, column] += 1
    assert graph_incidence_matrix(graph) == expected
    result = graph_chain_homology(graph, modulus=modulus)
    assert result.chain_dimensions == (vertices, len(edges))
    assert result.homology_dimensions == betti
    assert result.boundary_ranks == (0, vertices - betti[0])
    for tensor in graph.edge_tensors:
        # Higher graph generators are all degeneracies and normalize to zero.
        assert not tensor.degeneracy(0).normalize()
        assert not tensor.degeneracy(1).normalize()


def test_petersen_tensor_incidence_complex():
    graph = graph_embedding(10, petersen_edges())
    result = graph_chain_homology(graph)
    assert result.chain_dimensions == (10, 15)
    assert result.boundary_ranks == (0, 9)
    assert result.homology_dimensions == (1, 6)


def test_adjacency_tensor_and_graph_chain_complex_have_different_homology():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert adjacency_h1_relations(adjacency) == 2
    assert generated_homology(adjacency).homology_dimensions == (0, 0, 0)
    assert generated_homology(adjacency, modulus=2).homology_dimensions == (0, 1, 0)
    assert deletion_homology_mod2(adjacency).homology_dimensions == (0, 1, 1)
    assert graph_chain_homology(graph_embedding(3, [(1, 2), (2, 3)])).homology_dimensions == (1, 0)
    path4 = np.array([[int(abs(i - j) == 1) for j in range(4)] for i in range(4)])
    assert boundary_rank_distribution(path4) == {0: 4, 2: 20}
    order = (1, 3, 0, 2)
    assert boundary_rank_distribution(path4[np.ix_(order, order)]) == {0: 4, 2: 20}
