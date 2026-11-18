"""Count full-rank interleaved boundaries for three fixed graphs over F_2.

Run with the installed simplicial_tensors package. No arguments are needed.
The printed counts are Hamiltonian vertex orders, including both directions.
Use --small-examples for the path in Figure 1 and the two four-vertex graphs.
"""

import argparse
from collections import Counter
from functools import cache
from itertools import combinations, permutations

import numpy as np

from simplicial_tensors.graph_ops import rank_binary_rows
from simplicial_tensors.rank_shifted import petersen_edges
from simplicial_tensors.tensor_ops import bdry


def examples():
    """Yield each adjacency matrix with a Hamiltonian order for the face check."""
    petersen = np.zeros((10, 10), dtype=bool)
    for u, v in petersen_edges():  # The package uses labels 1 through 10.
        petersen[u - 1, v - 1] = petersen[v - 1, u - 1] = True
    yield "Petersen", petersen, (0, 7, 3, 4, 9, 1, 5, 8, 2, 6)

    rook = np.zeros((16, 16), dtype=bool)
    shrikhande = np.zeros_like(rook)
    steps = {(0, 1), (0, 3), (1, 0), (3, 0), (1, 1), (3, 3)}
    for u, v in combinations(range(16), 2):
        a, b = divmod(u, 4)
        c, d = divmod(v, 4)
        rook[u, v] = rook[v, u] = a == c or b == d
        shrikhande[u, v] = shrikhande[v, u] = ((c - a) % 4, (d - b) % 4) in steps
    snake = (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12)
    for name, matrix in (("4 by 4 rook", rook), ("Shrikhande", shrikhande)):
        # The diagonal of A^2 gives the degrees and its other entries count
        # common neighbors. These two graphs have parameters (16, 6, 2, 2).
        adjacency = matrix.astype(int)
        np.testing.assert_array_equal(adjacency @ adjacency, 4 * np.eye(16, dtype=int) + 2)
        yield name, matrix, snake


def hamiltonian_orders(matrix):
    """Count paths that use every vertex once by their endpoint and unused set."""
    n = len(matrix)
    neighbors = [np.flatnonzero(row).tolist() for row in matrix]

    @cache
    def extensions(vertex, unused):
        if unused == 0:
            return 1
        return sum(
            extensions(w, unused ^ (1 << w))
            for w in neighbors[vertex] if unused & (1 << w)
        )

    full = (1 << n) - 1
    count = sum(extensions(v, full ^ (1 << v)) for v in range(n))
    extensions.cache_clear()
    return count


def binary_rank(matrix):
    return rank_binary_rows(
        sum(1 << j for j in np.flatnonzero(row).tolist()) for row in matrix
    )


def interleaved_boundary(ordered):
    """Insert isolated vertices and take the mod-two adjacency boundary."""
    n = len(ordered)
    padded = np.zeros((2 * n - 1, 2 * n - 1), dtype=bool)
    padded[::2, ::2] = ordered
    return bdry(padded)


def check_boundary(matrix, order):
    """Compare the actual faces with the two forward-adjacency blocks."""
    ordered = matrix[np.ix_(order, order)]
    n = len(order)
    boundary = interleaved_boundary(ordered)
    forward = np.triu(ordered, 1)
    expected = np.zeros_like(boundary)
    expected[::2, 1::2] = forward[:-1, 1:]
    expected[1::2, ::2] = forward[:-1, 1:].T
    np.testing.assert_array_equal(boundary, expected)
    rank = binary_rank(boundary)
    assert rank == 2 * binary_rank(forward)
    assert (rank == 2 * n - 2) == ordered.diagonal(1).all()
    return rank


def small_examples():
    """Compute the examples on three and four vertices in the manuscript."""
    path = np.eye(3, k=1, dtype=bool) | np.eye(3, k=-1, dtype=bool)
    boundary = interleaved_boundary(path)
    edges = [(i, j) for i, j in combinations(range(len(boundary)), 2) if boundary[i, j]]
    rank = binary_rank(boundary)
    assert edges == [(0, 1), (2, 3)] and rank == 4
    print(f"P3 after insertion: boundary edges {edges}, rank {rank}")

    complete = ~np.eye(4, dtype=bool)
    for name, matrix in (("K4", complete), ("Edgeless", np.zeros_like(complete))):
        ranks = Counter()
        paths = 0
        for order in permutations(range(4)):
            ordered = matrix[np.ix_(order, order)]
            ranks[binary_rank(bdry(ordered))] += 1
            if ordered.diagonal(1).all():
                paths += 1
        assert ranks == {0: 24}
        assert paths == hamiltonian_orders(matrix)
        print(f"{name} without insertion: boundary ranks {dict(ranks)}, oriented paths {paths}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-examples", action="store_true",
                        help="Compute the three- and four-vertex examples.")
    if parser.parse_args(argv).small_examples:
        small_examples()
        return
    print("Graph | Full boundary rank | Orders attaining full rank")
    for name, matrix, path in examples():
        vertices = list(range(len(matrix)))
        check_boundary(matrix, vertices)
        check_boundary(matrix, vertices[::2] + vertices[1::2])
        rank = check_boundary(matrix, path)
        assert rank == 2 * len(matrix) - 2
        print(f"{name} | {rank} | {hamiltonian_orders(matrix):,}")


if __name__ == "__main__":
    main()
