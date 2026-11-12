"""Enumerate Petersen adjacency deletion-boundary ranks over GF(2)."""

from collections import Counter

import numpy as np
from sage.all import GF, factorial, graphs, libgap, matrix

from simplicial_tensors.tensor_ops import bdry


def boundary_mod2(T):
    """Apply the shared boundary over the Boolean model of GF(2)."""
    entries = np.fromiter((bool(x) for x in T.list()), dtype=bool).reshape(T.nrows(), T.ncols())
    boundary = bdry(entries)
    return matrix(GF(2), boundary.astype(int).tolist())


def permute_adjacency(T, perm):
    """Reorder old vertices by ``T[perm[i], perm[j]]``."""
    return T.matrix_from_rows_and_columns(perm, perm)


def gap_perm_to_list(g, size):
    """Convert a GAP right-coset representative to old vertices in new order."""
    return [int(i) - 1 for i in libgap.ListPerm(g**-1, size)]


def right_coset_rank_distribution(graph):
    """Return group order, coset count, distinct matrices, and two histograms."""
    P = graph.relabel({v: i for i, v in enumerate(sorted(graph.vertices()))},
                      inplace=False)
    N = P.order()
    T0 = matrix(GF(2), P.adjacency_matrix())
    Aut = P.automorphism_group()
    Sgap = libgap.SymmetricGroup(N)
    Agap = libgap(Aut)
    reps_gap = [c.Representative() for c in Sgap.RightCosets(Agap)]

    # GAP right cosets are Aut*g. With old vertices in new order, use g^-1:
    # an automorphism of T0 then acts before the new vertex labels are assigned.
    counts = Counter()
    seen = set()
    for g in reps_gap:
        Tp = permute_adjacency(T0, gap_perm_to_list(g, N))
        key = bytes(bool(x) for x in Tp.list())
        if key in seen:
            raise AssertionError("Right transversal produced duplicate adjacency matrices")
        seen.add(key)
        counts[int(boundary_mod2(Tp).rank())] += 1

    expected = factorial(N) // Aut.order()
    if len(reps_gap) != expected or len(seen) != expected:
        raise AssertionError("Right transversal does not cover every labeled adjacency matrix")
    weighted = {r: c * int(Aut.order()) for r, c in sorted(counts.items())}
    return int(Aut.order()), len(reps_gap), len(seen), dict(sorted(counts.items())), weighted


def main():
    P = graphs.PetersenGraph()
    order, num_cosets, num_distinct, counts, weighted = right_coset_rank_distribution(P)
    assert order == 120
    assert num_cosets == num_distinct == 30240
    assert counts == {4: 376, 6: 8496, 8: 21368}
    assert weighted == {4: 45120, 6: 1019520, 8: 2564160}
    assert sum(weighted.values()) == factorial(P.order())
    print("N =", P.order())
    print("|Aut(P)| =", order)
    print("Number of right cosets =", num_cosets)
    print("Distinct relabeled adjacency matrices =", num_distinct)
    print("Unweighted counts:", counts)
    print("Weighted counts:", weighted)
    print("Check totals:", sum(weighted.values()), "vs", factorial(P.order()))


if __name__ == "__main__":
    main()
