"""Enumerate the kernel family of all labelled 3-by-3 matrices over F_2."""

from collections import Counter
from itertools import product
import json

import numpy as np

from simplicial_tensors.kernel_sequences import kernel_sequence


def three_by_three_kernels() -> dict:
    """Count maximal-rank matrices for each plane in the binary projective plane."""
    counts = Counter()
    for entries in product(range(2), repeat=9):
        tensor = np.array(entries).reshape(3, 3)
        sequence = kernel_sequence(tensor, modulus=2)
        ranks = sequence.quotient.dimensions
        # A nonzero row of g_0 is the unique binary normal to its kernel plane.
        normal = tuple(int(value) for value in sequence.g_matrices[0])
        counts[ranks, normal] += 1
        if all(entries):
            all_ones = {
                "degree_zero_kernel_normal": normal,
                "kernel_dimensions": sequence.kernel.dimensions,
                "betti_numbers": sequence.quotient.homology().homology_dimensions,
            }

    maxima = tuple(max(ranks[q] for ranks, _ in counts) for q in range(3))
    fibers = [
        {"normal": normal, "matrices": counts[maxima, normal]}
        for normal in product(range(2), repeat=3) if any(normal)
    ]
    total = sum(counts.values())
    maximal = sum(fiber["matrices"] for fiber in fibers)
    return {
        "coefficient_field": "GF(2)",
        "total_matrices": total,
        "maximal_ranks": maxima,
        "matrices_on_maximal_rank_locus": maximal,
        "matrices_outside_maximal_rank_locus": total - maximal,
        "fibers_on_maximal_rank_locus": fibers,
        "all_ones_matrix": all_ones,
    }


if __name__ == "__main__":
    print(json.dumps(three_by_three_kernels(), indent=2))
