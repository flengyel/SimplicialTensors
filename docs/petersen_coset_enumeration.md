# Petersen coset enumeration in SageMath

[The SageMath script](../experiments/notebooks/petersen_coset_enumeration.sage)
constructs the Petersen graph over $\mathbb F_2$, obtains its
automorphism group, and requests a right-coset transversal from GAP.
It computes the rank of the XOR sum of principal deletions for each
representative and multiplies the counts by the group order.

## Run

Use SageMath with its `libgap` interface.

```sh
sage experiments/notebooks/petersen_coset_enumeration.sage
```

The script has no command-line options and executes at top level.
Writing $P$ for the Petersen graph, the expected group sizes are
$$
|\operatorname{Aut}(P)|=120,\qquad
[S_{10}:\operatorname{Aut}(P)]=30{,}240.
$$
It prints these sizes, the two histograms, and the weighted total
compared with $10!$.

`permute_adjacency` interprets a permutation list as old vertices in
their new order and builds `T[perm[i], perm[j]]`. Validate this
convention against the GAP coset action when checking a run. A total
weight of $10!$ alone does not verify that every distinct adjacency
matrix was represented once.

The independently checked
[Python computation](petersen_brd_right_cosets.md) gives transversal
counts `{4: 376, 6: 8496, 8: 21368}` and weighted counts
`{4: 45120, 6: 1019520, 8: 2564160}`. Compare the Sage output with
those values and verify 30,240 distinct relabeled matrices. The
documentation audit did not execute this script because SageMath
was unavailable.

These ranks describe adjacency deletion boundaries. Homology requires
a separately specified chain complex.
