# Petersen boundary ranks from automorphism cosets

[The Python experiment](../experiments/petersen_brd_right_cosets.py)
computes the full principal-deletion boundary-rank distribution for the
Petersen graph over $\mathbb F_2$. It uses $KG(5,2)$, whose vertices
are the two-subsets of a five-element set and whose edges join disjoint
subsets.

Each boundary is the XOR sum of ten principal deletions, represented as
nine Python integer row masks. Binary elimination computes its rank.
The program uses SymPy for permutation groups and no NumPy or NetworkX
operations.

## Run

```sh
python -m pip install -e .
python experiments/petersen_brd_right_cosets.py
```

There are no command-line options. The script checks the automorphism
group order and transversal size and prints both unweighted counts
and counts weighted by the group order.

| Boundary rank | Distinct labeled adjacency matrices | All permutations |
|---:|---:|---:|
| 4 | 376 | 45,120 |
| 6 | 8,496 | 1,019,520 |
| 8 | 21,368 | 2,564,160 |
| Total | 30,240 | 3,628,800 |

## Permutation convention

`perm[i]` is the new label of old vertex `i`. The relabeling moves
every edge `(i, j)` to `(perm[i], perm[j])`. SymPy's transversal
representatives are used directly with this convention. They produce
30,240 distinct labeled adjacency matrices, so weighting by the
automorphism-group order 120 recovers every element of $S_{10}$.
Inverting those representatives would change the action convention and
produce duplicate matrices in this implementation.

The ranks belong to the $9\times9$ deletion boundaries of the
relabeled adjacency tensors. They differ from the rank of the graph's
incidence differential. No relabeled adjacency tensor is a cycle,
since the displayed ranks are positive. This computation does not
assert nonzero homology.

[The Sage version](petersen_coset_enumeration.md) uses a separate
reordering convention. Compare its output with this table when
validating that implementation.
