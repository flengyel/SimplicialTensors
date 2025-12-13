#!/usr/bin/env python3
"""
Full enumeration of the mod-2 "adjacency boundary" rank spectrum
for the Graph Atlas cospectral pair G115 and G117 (both on 6 vertices).

Boundary definition (mod 2):
  ∂T := sum_{k=0}^{N-1} d_k(T), where
  (d_k T)[u,v] = T[δ_k(u), δ_k(v)] and δ_k: [N-2] -> [N-1] skips k.

This is equivalent to summing all principal (N-1)x(N-1) submatrices
obtained by deleting row/col k, but implemented via the simplicial coface map.

Requires: numpy, networkx
Run:
  python g115_g117_full_enumeration.py
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Iterable, Tuple, Dict

import numpy as np
import networkx as nx


def delta(k: int, idx: int) -> int:
    """Cosimplicial coface δ_k: [N-2] -> [N-1] (skip index k)."""
    return idx if idx < k else idx + 1


def face_map_adj(T: np.ndarray, k: int) -> np.ndarray:
    """
    d_k(T) for an adjacency matrix T (N x N), returning an (N-1) x (N-1) matrix:
      (d_k T)[u,v] = T[δ_k(u), δ_k(v)].
    """
    N = int(T.shape[0])
    dim = N - 1
    out = np.zeros((dim, dim), dtype=np.uint8)
    for u in range(dim):
        ru = delta(k, u)
        for v in range(dim):
            rv = delta(k, v)
            out[u, v] = T[ru, rv]
    return out


def boundary_mod2(T: np.ndarray) -> np.ndarray:
    """∂T = sum_k d_k(T) (mod 2). Uses XOR for addition in GF(2)."""
    N = int(T.shape[0])
    dim = N - 1
    B = np.zeros((dim, dim), dtype=np.uint8)
    for k in range(N):
        B ^= face_map_adj(T, k)
    return B


def rank_gf2(M: np.ndarray) -> int:
    """Gaussian elimination over GF(2)."""
    A = M.copy().astype(np.uint8)
    rows, cols = A.shape
    rank = 0
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break

        if A[pivot_row, col] == 0:
            swap = None
            for r in range(pivot_row + 1, rows):
                if A[r, col] == 1:
                    swap = r
                    break
            if swap is None:
                continue
            A[[pivot_row, swap]] = A[[swap, pivot_row]]

        # eliminate below
        for r in range(pivot_row + 1, rows):
            if A[r, col] == 1:
                A[r, :] ^= A[pivot_row, :]

        rank += 1
        pivot_row += 1

    return rank


def permute_matrix(T: np.ndarray, perm: Tuple[int, ...]) -> np.ndarray:
    perm = np.array(perm, dtype=int)
    return T[perm][:, perm]


def rank_distribution_full_enum(T: np.ndarray) -> Counter:
    """
    Enumerate all N! labelings of the adjacency matrix T and count ranks of ∂(PTP^T).
    Returns Counter{rank: count}.
    """
    N = int(T.shape[0])
    counts: Counter = Counter()
    for perm in itertools.permutations(range(N)):
        Tp = permute_matrix(T, perm)
        B = boundary_mod2(Tp)
        counts[rank_gf2(B)] += 1
    return counts


def main() -> None:
    atlas = nx.graph_atlas_g()
    G115 = atlas[115]
    G117 = atlas[117]

    # Adjacency matrices over GF(2) (uint8 0/1)
    T115 = nx.to_numpy_array(G115, dtype=np.uint8)
    T117 = nx.to_numpy_array(G117, dtype=np.uint8)

    # sanity: cospectral over R
    eig115 = np.linalg.eigvalsh(T115.astype(float))
    eig117 = np.linalg.eigvalsh(T117.astype(float))
    print("G115:", G115.name, "n=", G115.number_of_nodes(), "m=", G115.number_of_edges())
    print("G117:", G117.name, "n=", G117.number_of_nodes(), "m=", G117.number_of_edges())
    print("Cospectral (rounded)?", np.allclose(eig115, eig117))

    print("\nEnumerating all labelings (S_6 has 720 permutations)...")
    c115 = rank_distribution_full_enum(T115)
    c117 = rank_distribution_full_enum(T117)

    print("\nRank distribution for ∂(PTP^T) over GF(2):")
    print("G115:", dict(sorted(c115.items())))
    print("G117:", dict(sorted(c117.items())))
    print("\nTotals:", sum(c115.values()), sum(c117.values()))


if __name__ == "__main__":
    main()
