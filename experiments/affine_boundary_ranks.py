"""Compute the mod-2 adjacency boundary ranks for every affine order on (Z/4)^2.

From the repository root, run PYTHONPATH=src python experiments/affine_boundary_ranks.py.
Each affine map counts once, including those giving the same adjacency matrix.
"""

from collections import Counter
from itertools import product

import numpy as np

from simplicial_tensors.tensor_ops import bdry

if __package__:
    from .interleaved_graph_paths import binary_rank, examples
else:
    from interleaved_graph_paths import binary_rank, examples


def affine_orders():
    """Apply each invertible affine map to the lexicographic vertex order."""
    vertices = tuple(product(range(4), repeat=2))
    for a, b, c, d in product(range(4), repeat=4):
        if (a * d - b * c) % 2 == 0:
            continue
        for s, t in vertices:
            yield tuple(
                4 * ((a * x + b * y + s) % 4) + ((c * x + d * y + t) % 4)
                for x, y in vertices
            )


def main():
    graphs = {name: adjacency for name, adjacency, _ in examples()}
    names = ("4 by 4 rook", "Shrikhande")
    histograms = {name: Counter() for name in names}

    for order in affine_orders():
        indices = np.ix_(order, order)
        for name in names:
            boundary = bdry(graphs[name][indices])
            histograms[name][binary_rank(boundary)] += 1

    print("Boundary rank | " + " | ".join(histograms))
    for rank in sorted(set().union(*histograms.values())):
        counts = (str(histogram[rank]) for histogram in histograms.values())
        print(f"{rank} | " + " | ".join(counts))
    totals = (str(sum(histogram.values())) for histogram in histograms.values())
    print("Total | " + " | ".join(totals))


if __name__ == "__main__":
    main()
