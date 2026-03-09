# `petersen_brd_right_cosets.py` documentation

## Location
- Script: `experiments/petersen_brd_right_cosets.py`

## Purpose
This script computes the exact mod-2 boundary-rank distribution (BRD) for the Petersen graph under the vertical simplicial boundary used in the rank-shifted bisimplicial DSTM / total-decalage model.

For each relabeling of vertices, it computes the rank over `F_2` of the vertical boundary matrix and tabulates how often each rank occurs.

## Mathematical Model
- Graph: Petersen graph with `N = 10` vertices and `15` edges.
- Vertex model: vertices are the 2-subsets of `{0,1,2,3,4}` (Kneser labeling).
- Edge rule: two vertices are adjacent iff the two 2-subsets are disjoint.

Adjacency matrices are encoded as integer row bitmasks.

### Vertical Faces and Boundary
For an `N x N` adjacency matrix `T`, the vertical face `d_k^v(T)` is principal deletion of row/column `k` with reindexing.

The mod-2 vertical boundary is:

`partial_v(T) = sum_{k=0}^{N-1} d_k^v(T)` over `F_2`.

In code, this is implemented as XOR of all principal deletions.

### Boundary Rank
The script computes `rank_{F_2}(partial_v(T))` using bit-level Gaussian elimination on row bitmasks.

## Group-Theoretic Reduction
Let `A` be the Petersen adjacency matrix and let relabelings act by:

`A^p = P_p A P_p^T`, with convention `p[i] = new label of old vertex i`.

The script builds `Aut(P)` from the induced `S_5` action on 2-subsets and obtains `|Aut(P)| = 120`.

Because `A^{p sigma} = A^p` for `sigma in Aut(P)`, the boundary-rank value is constant on right cosets of `S_10 / Aut(P)`.

So it is exact to enumerate a right transversal only:
- `|S_10 / Aut(P)| = 10! / 120 = 30240` representatives.

SymPy is used for:
- permutation groups,
- right coset transversal generation,
- induced permutation handling.

## Algorithm
1. Build Petersen adjacency row bitmasks from the Kneser model.
2. Build `Aut(P)` as induced permutations of 2-subsets.
3. Enumerate a right transversal of `S_10 / Aut(P)`.
4. For each representative:
   - relabel adjacency matrix,
   - compute `partial_v`,
   - compute `rank_{F_2}`.
5. Accumulate:
   - unweighted counts over coset representatives,
   - weighted counts over all `10!` labelings by multiplying each bucket by `|Aut(P)|`.

## Built-In Checks
Before BRD computation, `verify_basic_facts()` asserts:
- `|Aut(P)| = 120`,
- transversal size equals `10! / 120`.

## Output
The script prints:
- graph size summary,
- automorphism-group size,
- right-coset count,
- BRD over right-coset representatives,
- BRD over all `10!` labelings,
- total-labelings consistency check.

Current output:
- Unweighted (right-coset transversal):
  - rank `4`: `376`
  - rank `6`: `8496`
  - rank `8`: `21368`
- Weighted (all labelings):
  - rank `4`: `45120`
  - rank `6`: `1019520`
  - rank `8`: `2564160`
- Total check: `3628800` (equals `10!`).

## Logging
This script does not configure file logging.
All reporting is printed to standard output.

## Run
From repository root:

```bash
.venv/Scripts/python.exe experiments/petersen_brd_right_cosets.py
```

