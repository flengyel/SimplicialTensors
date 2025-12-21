#!/usr/bin/env python3
"""
atlas_cospectral_scan.py

Scan the NetworkX Graph Atlas for adjacency-cospectral, non-isomorphic pairs
and compute the mod-2 DSTM-inspired adjacency-boundary rank distributions.

Boundary over GF(2):
  ∂T := sum_{k=0}^{N-1} d_k(T)  (mod 2),
  where (d_k T)[u,v] = T[δ_k(u), δ_k(v)], δ_k skips k.
Equivalent to summing principal submatrices (delete row/col k) mod 2.

For N <= exact_max_n (default 8), compute the EXACT distribution over all N! labelings.
For larger N, compute a sampled distribution over random labelings.

Requires: numpy, networkx
Usage examples:
  python atlas_cospectral_scan.py --n 6 --max_pairs 10
  python atlas_cospectral_scan.py --n 10 --max_pairs 5 --samples 20000
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx


def gf2_rank(M: np.ndarray) -> int:
    A = (M.copy() & 1).astype(np.uint8)
    rows, cols = A.shape
    rank = 0
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        pivot = None
        for r in range(pivot_row, rows):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
        for r in range(rows):
            if r != pivot_row and A[r, col]:
                A[r] ^= A[pivot_row]
        rank += 1
        pivot_row += 1
    return rank


def boundary_rank_mod2(T: np.ndarray) -> int:
    """
    Return rank_GF2( ∂T ), where ∂T = sum_k d_k(T) mod 2 and d_k deletes k.
    """
    N = T.shape[0]
    dim = N - 1
    B = np.zeros((dim, dim), dtype=np.uint8)
    for k in range(N):
        mask = np.ones(N, dtype=bool)
        mask[k] = False
        face = T[mask][:, mask]
        B ^= face
    return gf2_rank(B)


def exact_rank_distribution(A: np.ndarray) -> Counter:
    N = A.shape[0]
    counts = Counter()
    for p in itertools.permutations(range(N)):
        Ap = A[np.ix_(p, p)]
        counts[boundary_rank_mod2(Ap)] += 1
    return counts


def sampled_rank_distribution(A: np.ndarray, samples: int, seed: int) -> Counter:
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    counts = Counter()
    for _ in range(samples):
        p = rng.permutation(N)
        Ap = A[np.ix_(p, p)]
        counts[boundary_rank_mod2(Ap)] += 1
    return counts


def spec_signature(G: nx.Graph, decimals: int = 6) -> Tuple[float, ...]:
    A = nx.to_numpy_array(G, dtype=float)
    evals = np.linalg.eigvalsh(A)
    return tuple(sorted(np.round(evals, decimals)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of vertices N to scan in atlas")
    ap.add_argument("--max_pairs", type=int, default=10, help="max cospectral non-isomorphic pairs to report")
    ap.add_argument("--exact_max_n", type=int, default=8, help="compute exact distributions only for N<=this")
    ap.add_argument("--samples", type=int, default=20000, help="samples for N>exact_max_n")
    ap.add_argument("--seed", type=int, default=1, help="rng seed for sampling")
    args = ap.parse_args()

    atlas = nx.graph_atlas_g()
    graphs = [(idx, g) for idx, g in enumerate(atlas) if len(g) == args.n]

    buckets: Dict[Tuple[float, ...], List[Tuple[int, nx.Graph]]] = defaultdict(list)
    for idx, g in graphs:
        buckets[spec_signature(g)].append((idx, g))

    found = 0
    print(f"Scanning atlas graphs with N={args.n}. Buckets={len(buckets)}")

    for sig, items in buckets.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idxA, GA = items[i]
                idxB, GB = items[j]
                if nx.is_isomorphic(GA, GB):
                    continue

                A = nx.to_numpy_array(GA, dtype=np.uint8)
                B = nx.to_numpy_array(GB, dtype=np.uint8)

                if args.n <= args.exact_max_n:
                    distA = exact_rank_distribution(A)
                    distB = exact_rank_distribution(B)
                    mode = "exact"
                else:
                    distA = sampled_rank_distribution(A, args.samples, args.seed)
                    distB = sampled_rank_distribution(B, args.samples, args.seed + 1)
                    mode = f"sampled({args.samples})"

                print("\n--- Cospectral non-isomorphic pair ---")
                print(f"Atlas IDs: G{idxA} vs G{idxB}   (mode: {mode})")
                print(f"Edges: {GA.number_of_edges()} vs {GB.number_of_edges()}")
                print("Rank dist A:", dict(sorted(distA.items())))
                print("Rank dist B:", dict(sorted(distB.items())))
                if distA != distB:
                    print("[SPLIT] distributions differ")
                else:
                    print("[NO SPLIT] distributions match (under this mode)")
                found += 1
                if found >= args.max_pairs:
                    return


if __name__ == "__main__":
    main()
