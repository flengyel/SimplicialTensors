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



@pytest.mark.parametrize("modulus", [None, 2, 3])
@pytest.mark.parametrize("vertices,edges,betti", [
    (0, [], (0, 0)),
    (4, [], (4, 0)),
    (4, [(1, 2), (1, 2), (3, 3)], (3, 2)),
])
def test_valid_graph_homology_avoids_dense_incidence(monkeypatch, vertices, edges, betti, modulus):
    import simplicial_tensors.graph_ops as graph_ops

    def unexpected_dense_matrix(_embedding):
        pytest.fail("Validated graph homology should not allocate dense incidence.")

    monkeypatch.setattr(graph_ops, "graph_incidence_matrix", unexpected_dense_matrix)
    result = graph_ops.graph_chain_homology(graph_embedding(vertices, edges), modulus=modulus)
    assert result.homology_dimensions == betti


@pytest.mark.parametrize("modulus,expected_rank", [(None, 1), (2, 0), (3, 1)])
def test_modified_graph_tensors_keep_incidence_semantics(monkeypatch, modulus, expected_rank):
    from dataclasses import replace
    import simplicial_tensors.graph_ops as graph_ops

    graph = graph_embedding(2, [(1, 2)])
    modified = replace(graph, edge_tensors=(2 * graph.edge_tensors[0],))
    calls = []
    original = graph_ops.graph_incidence_matrix

    def record_matrix(embedding):
        calls.append(embedding)
        return original(embedding)

    monkeypatch.setattr(graph_ops, "graph_incidence_matrix", record_matrix)
    result = graph_ops.graph_chain_homology(modified, modulus=modulus)
    assert result.boundary_ranks == (0, expected_rank)
    assert result.homology_dimensions == (2 - expected_rank, 1 - expected_rank)
    assert calls == [modified]


def test_manual_graph_count_mismatch_keeps_tensor_matrix_dimensions():
    from dataclasses import replace

    graph = replace(graph_embedding(2, [(1, 2)]), vertex_count=7)
    result = graph_chain_homology(graph)
    assert result.chain_dimensions == (2, 1)
    assert result.homology_dimensions == (1, 0)



def test_unusable_manual_edge_metadata_falls_back_to_tensor_incidence():
    from dataclasses import replace

    graph = replace(graph_embedding(2, [(1, 2)]), edges=(None,))
    result = graph_chain_homology(graph)
    assert result.chain_dimensions == (2, 1)
    assert result.homology_dimensions == (1, 0)
