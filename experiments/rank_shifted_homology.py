#!/usr/bin/env python3
"""Exact rational checks of graph and reduced circle-smash representations.

Run from an installed checkout, or with PYTHONPATH=src. Only coordinates
occurring in the supplied tensors enter the linear algebra; the full ambient
tensor spaces are never allocated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sympy import Matrix, Rational

from simplicial_tensors.rank_shifted import (
    SparseTensor,
    circle_generator,
    graph_embedding,
    petersen_edges,
)


def exact_rank(tensors: list[SparseTensor]) -> int:
    """Rank over Q after restriction to the union of nonzero coordinates."""
    coordinates = sorted({index for tensor in tensors for index in tensor.entries})
    if not coordinates or not tensors:
        return 0
    matrix = Matrix([
        [Rational(t.coefficient(index).numerator, t.coefficient(index).denominator)
         for t in tensors]
        for index in coordinates
    ])
    return int(matrix.rank())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def petersen_check() -> dict:
    graph = graph_embedding(10, petersen_edges())
    for (u, v), tensor in zip(graph.edges, graph.edge_tensors):
        require(tensor.face(1) == graph.vertices[u - 1], "Incorrect source face")
        require(tensor.face(0) == graph.vertices[v - 1], "Incorrect target face")
    normalized = list(graph.normalized_edges)
    vertex_rank = exact_rank(list(graph.vertices))
    edge_rank = exact_rank(normalized)
    boundary_rank = exact_rank([t.face(0) for t in normalized])
    require((vertex_rank, edge_rank, boundary_rank) == (10, 15, 9),
            "Unexpected Petersen chain ranks")
    # Every degree-two generator is a degeneracy of a vertex or an edge.
    degree_two = [t.degeneracy(i) for t in graph.edge_tensors for i in (0, 1)]
    require(all(t.normalize().nnz == 0 for t in degree_two),
            "A graph degeneracy survived normalization")
    return {
        "vertices": 10,
        "edges": 15,
        "tensor_order": 3,
        "shift": graph.shift,
        "degree_one_axis_length": graph.shift + 2,
        "ambient_degree_one_coordinates": (graph.shift + 2) ** 3,
        "stored_edge_entries_total": sum(t.nnz for t in graph.edge_tensors),
        "normalized_dimensions": [vertex_rank, edge_rank, 0],
        "boundary_rank_1": boundary_rank,
        "betti_numbers": [vertex_rank - boundary_rank, edge_rank - boundary_rank],
        "higher_homology": "zero: all higher graph simplices are degenerate",
    }


def circle_basis(degree: int) -> list[SparseTensor]:
    """Pullbacks of E01 along the degree-many surjections [degree] -> [1]."""
    return [
        SparseTensor(degree, (0, 0), {
            (i, j): 1
            for i in range(cut + 1)
            for j in range(cut + 1, degree + 1)
        })
        for cut in range(degree)
    ]


def circle_smash_check() -> dict:
    normalized_dimensions = []
    boundary_ranks = []
    for degree in range(4):
        basis = circle_basis(degree)
        normalized = [a.outer(b).normalize() for a in basis for b in basis]
        normalized_dimensions.append(exact_rank(normalized))
        boundary_ranks.append(
            exact_rank([t.face(0) for t in normalized]) if degree else 0
        )
        for tensor in normalized:
            require(all(tensor.face(i).nnz == 0 for i in range(1, degree + 1)),
                    "Normalization has a nonzero positive-index face")
    require(normalized_dimensions == [0, 1, 2, 0], "Unexpected smash dimensions")
    require(boundary_ranks == [0, 0, 1, 0], "Unexpected smash boundary ranks")
    a = circle_generator()
    shuffle = a.degeneracy(1).outer(a.degeneracy(0)) - a.degeneracy(0).outer(a.degeneracy(1))
    require(shuffle.nnz > 0, "Shuffle cycle vanished")
    require(all(shuffle.face(i).nnz == 0 for i in range(3)),
            "Shuffle tensor is not a normalized cycle")
    betti = [normalized_dimensions[n] - boundary_ranks[n] - boundary_ranks[n + 1]
             for n in range(3)]
    return {
        "construction": "reduced circle module tensor reduced circle module",
        "tensor_order": 4,
        "shift": 0,
        "normalized_dimensions_degrees_0_to_3": normalized_dimensions,
        "boundary_ranks_degrees_0_to_3": boundary_ranks,
        "reduced_betti_numbers_degrees_0_to_2": betti,
        "shuffle_cycle_nonzero_entries": shuffle.nnz,
        "shuffle_cycle_all_faces_zero": True,
        "higher_homology": "zero: every degree at least three is degenerate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the JSON result to this file")
    args = parser.parse_args()
    report = {
        "coefficient_field": "Q",
        "linear_algebra": "exact rational rank on occurring coordinates",
        "petersen": petersen_check(),
        "circle_smash_circle": circle_smash_check(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
