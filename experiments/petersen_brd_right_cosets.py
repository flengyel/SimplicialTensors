#!/usr/bin/env python3
"""
petersen_brd_right_cosets.py

Exact computation of the mod-2 boundary-rank distribution (BRD) of the
Petersen graph using the vertical boundary in the rank-shifted bisimplicial
DSTM / total-decalage model.

In bidegree (-1, N-1) over F_2, the vertical simplicial boundary is

    ∂_v(T) = Σ_{k=0}^{N-1} d_k^v(T),

where d_k^v deletes row/column k and reindexes the remaining vertices in
increasing order. Over F_2 this is the XOR-sum of principal deletions.

For the unrestricted boundary-rank distribution of a graph G on N vertices,
we want the multiset of ranks

    rank_F2( ∂_v( A(G)^π ) )

as π ranges over S_N. Since A^{πσ} = A^π for every automorphism σ of A,
this function is constant on RIGHT cosets of S_N / Aut(A). Hence it is enough
(and exact) to enumerate a right transversal of S_N / Aut(A).

This script computes that distribution for the Petersen graph.
It is self-contained and uses:
  - sympy  (permutation groups, right coset transversal)
  - only Python integer bit-operations for GF(2) linear algebra.

Important convention:
  A permutation p is represented by its image list

      p[i] = new label of old vertex i.

  With this convention, using RIGHT cosets is the correct quotient for the
  action A -> A^p = P_p A P_p^T.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import factorial
from typing import Dict, Iterable, List, Sequence, Tuple, TypeAlias

from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

RowMasks: TypeAlias = List[int]
Vertex: TypeAlias = Tuple[int, int]


# -----------------------------------------------------------------------------
# Petersen graph in the Kneser-model labeling KG(5,2)
# -----------------------------------------------------------------------------


def petersen_vertices() -> List[Vertex]:
    """Vertices labeled by the 2-subsets of {0,1,2,3,4} in lexicographic order."""
    return list(combinations(range(5), 2))


VERTICES: List[Vertex] = petersen_vertices()
VERTEX_TO_INDEX: Dict[Vertex, int] = {v: i for i, v in enumerate(VERTICES)}
N: int = len(VERTICES)  # N = 10


def petersen_edges() -> List[Tuple[int, int]]:
    """Edge list of the Petersen graph in the above labeling.

    Two 2-subsets are adjacent iff they are disjoint.
    """
    edges: List[Tuple[int, int]] = []
    for i, a in enumerate(VERTICES):
        sa = set(a)
        for j in range(i + 1, N):
            if sa.isdisjoint(VERTICES[j]):
                edges.append((i, j))
    return edges


EDGES: List[Tuple[int, int]] = petersen_edges()  # 15 edges


def edge_list_to_rowmasks(n: int, edges: Iterable[Tuple[int, int]]) -> RowMasks:
    """Adjacency matrix as symmetric row bitmasks."""
    rows = [0] * n
    for i, j in edges:
        if not (0 <= i < j < n):
            raise ValueError(f"invalid edge {(i, j)} for n={n}")
        rows[i] |= 1 << j
        rows[j] |= 1 << i
    return rows


PETERSEN_ROWS: RowMasks = edge_list_to_rowmasks(N, EDGES)


# -----------------------------------------------------------------------------
# Vertical faces / vertical boundary on adjacency matrices over F_2
# -----------------------------------------------------------------------------


def vertical_face_rows(rows: Sequence[int], k: int) -> RowMasks:
    """Return d_k^v(rows): delete row/column k and reindex.

    Input:
        rows = adjacency matrix on N vertices, encoded as row bitmasks.
        k    = vertex to delete, 0 <= k < N.

    Output:
        adjacency matrix on N-1 vertices, encoded as row bitmasks.

    This is the bidegree (-1, N-1) vertical face of the bisimplicial DSTM.
    """
    n = len(rows)
    if not (0 <= k < n):
        raise IndexError(f"k={k} out of range for n={n}")

    out: RowMasks = []
    low_mask = (1 << k) - 1
    for i in range(n):
        if i == k:
            continue
        r = rows[i]
        low = r & low_mask
        high = (r >> (k + 1)) << k
        out.append(low | high)
    return out


def vertical_boundary_mod2_rows(rows: Sequence[int]) -> RowMasks:
    """Return ∂_v(rows) = Σ_k d_k^v(rows) over F_2.

    Since the coefficient field is F_2, this is the XOR-sum of all principal
    deletions.
    """
    n = len(rows)
    if n <= 1:
        return []

    out = [0] * (n - 1)
    for k in range(n):
        face = vertical_face_rows(rows, k)
        for i in range(n - 1):
            out[i] ^= face[i]

    mask = (1 << (n - 1)) - 1
    return [r & mask for r in out]


# -----------------------------------------------------------------------------
# GF(2) rank (rows stored as Python ints)
# -----------------------------------------------------------------------------


def gf2_rank_rowmasks(rows: Sequence[int], ncols: int) -> int:
    """Rank over F_2 of a matrix whose rows are encoded as Python ints."""
    if ncols < 0:
        raise ValueError("ncols must be nonnegative")
    if ncols == 0:
        return 0

    mask = (1 << ncols) - 1
    A = [r & mask for r in rows if (r & mask) != 0]
    rank = 0
    pivot_row = 0
    m = len(A)

    for col in range(ncols - 1, -1, -1):
        pivot = None
        for i in range(pivot_row, m):
            if (A[i] >> col) & 1:
                pivot = i
                break
        if pivot is None:
            continue

        A[pivot_row], A[pivot] = A[pivot], A[pivot_row]
        piv = A[pivot_row]

        for i in range(m):
            if i != pivot_row and ((A[i] >> col) & 1):
                A[i] ^= piv

        rank += 1
        pivot_row += 1
        if pivot_row >= m:
            break

    return rank


# -----------------------------------------------------------------------------
# Permutations and the Petersen automorphism group
# -----------------------------------------------------------------------------


def permutation_image_list(g: Permutation, size: int) -> List[int]:
    """Return the image list [g(0),...,g(size-1)] using array_form.

    This avoids SymPy's callable-permutation interface, which Pylance
    types poorly.
    """
    af = list(g.array_form)
    if len(af) < size:
        af.extend(range(len(af), size))
    return af[:size]


def induced_perm_on_vertices(g: Permutation) -> Permutation:
    """Induce a permutation of the 10 vertices from a permutation g of {0,...,4}.

    The vertex set is the set of 2-subsets of {0,...,4}. Every permutation of
    the underlying 5-set induces an automorphism of the Petersen graph.
    """
    g_img = permutation_image_list(g, 5)
    image = [0] * N
    for idx, (a, b) in enumerate(VERTICES):
        aa = g_img[a]
        bb = g_img[b]
        q = (aa, bb) if aa < bb else (bb, aa)
        image[idx] = VERTEX_TO_INDEX[q]
    return Permutation(image)


def petersen_automorphism_group() -> PermutationGroup:
    """Return Aut(Petersen) as the induced S_5-action on 2-subsets.

    Using generators (0 1) and (0 1 2 3 4) of S_5 is enough.
    """
    transposition = Permutation([1, 0, 2, 3, 4])
    five_cycle = Permutation([1, 2, 3, 4, 0])
    gen1 = induced_perm_on_vertices(transposition)
    gen2 = induced_perm_on_vertices(five_cycle)
    return PermutationGroup([gen1, gen2])


AUT_P: PermutationGroup = petersen_automorphism_group()
_AUT_P_ORDER = AUT_P.order()
if not isinstance(_AUT_P_ORDER, int):
    raise TypeError(f"Expected integer group order, got {type(_AUT_P_ORDER)!r}")
AUT_P_ORDER: int = _AUT_P_ORDER


# -----------------------------------------------------------------------------
# Adjacency relabeling
# -----------------------------------------------------------------------------


def permute_rows_by_image_perm(rows: Sequence[int], perm: Sequence[int]) -> RowMasks:
    """Apply the relabeling A -> A^perm = P_perm A P_perm^T.

    Convention:
        perm[i] = new label of old vertex i.

    Output is again encoded as row bitmasks on the new labels 0,...,N-1.
    """
    n = len(rows)
    if len(perm) != n:
        raise ValueError("perm length must match matrix size")

    out = [0] * n
    for i in range(n):
        ri = rows[i]
        for j in range(i + 1, n):
            if (ri >> j) & 1:
                a = perm[i]
                b = perm[j]
                if a > b:
                    a, b = b, a
                out[a] |= 1 << b
                out[b] |= 1 << a
    return out


# -----------------------------------------------------------------------------
# Exact BRD via right cosets S_10 / Aut(P)
# -----------------------------------------------------------------------------


def right_coset_transversal_as_permutations(
    group: PermutationGroup, subgroup: PermutationGroup
) -> List[Permutation]:
    """Return a right transversal, narrowed to actual Permutation objects.

    SymPy's typing for coset_transversal is loose; this function performs an
    explicit runtime check and gives Pylance a concrete return type.
    """
    raw = list(group.coset_transversal(subgroup))
    out: List[Permutation] = []
    for item in raw:
        if not isinstance(item, Permutation):
            raise TypeError(
                f"Expected Permutation in coset transversal, got {type(item)!r}"
            )
        out.append(item)
    return out


def brd_histogram_over_right_cosets() -> Tuple[Dict[int, int], Dict[int, int]]:
    """Compute the Petersen BRD exactly.

    Returns:
        (counts_cosets, counts_all_labelings)

    where
        counts_cosets[r]        = number of right-coset representatives τ with
                                  rank(∂_v(A(P)^τ)) = r,
        counts_all_labelings[r] = |Aut(P)| * counts_cosets[r].
    """
    s10 = SymmetricGroup(N)
    transversal = right_coset_transversal_as_permutations(s10, AUT_P)

    counts_cosets: Counter[int] = Counter()
    for rep in transversal:
        perm = permutation_image_list(rep, N)
        relabeled = permute_rows_by_image_perm(PETERSEN_ROWS, perm)
        boundary = vertical_boundary_mod2_rows(relabeled)
        r = gf2_rank_rowmasks(boundary, N - 1)
        counts_cosets[r] += 1

    counts_all: Dict[int, int] = {
        r: AUT_P_ORDER * c for r, c in sorted(counts_cosets.items())
    }
    return dict(sorted(counts_cosets.items())), counts_all


# -----------------------------------------------------------------------------
# Optional helpers / sanity checks
# -----------------------------------------------------------------------------


def verify_basic_facts() -> None:
    """Sanity checks for the Petersen computation."""
    if AUT_P_ORDER != 120:
        raise AssertionError(f"expected |Aut(P)| = 120, got {AUT_P_ORDER}")

    s10 = SymmetricGroup(N)
    transversal = right_coset_transversal_as_permutations(s10, AUT_P)
    expected = factorial(N) // AUT_P_ORDER
    if len(transversal) != expected:
        raise AssertionError(f"expected {expected} right cosets, got {len(transversal)}")


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------


def main() -> None:
    verify_basic_facts()

    counts_cosets, counts_all = brd_histogram_over_right_cosets()

    print("Petersen graph BRD using the mod-2 vertical boundary partial_v")
    print("-------------------------------------------------------")
    print(f"Number of vertices              : {N}")
    print(f"Number of edges                 : {len(EDGES)}")
    print(f"|Aut(P)|                        : {AUT_P_ORDER}")
    print(f"|S_10 / Aut(P)|                 : {factorial(N) // AUT_P_ORDER}")
    print()
    print("Unweighted counts over a right-coset transversal:")
    for r, c in counts_cosets.items():
        print(f"  rank {r}: {c}")
    print()
    print("Weighted counts over all 10! labelings:")
    for r, c in counts_all.items():
        print(f"  rank {r}: {c}")
    print()
    print(f"Check total labelings           : {sum(counts_all.values())}")


if __name__ == "__main__":
    main()
