# Petersen coset enumeration in SageMath

[The SageMath script](../../experiments/notebooks/petersen_coset_enumeration.sage)
constructs the Petersen graph and its adjacency matrix over $\mathbb F_2$, obtains its
automorphism group, and requests a right-coset transversal from GAP.
For each representative, it applies the shared tensor boundary to the
Boolean adjacency array and computes the resulting matrix rank over
Sage $\mathbb F_2$. It multiplies the counts by the group order.

## Run

Use SageMath with its `libgap` interface and install the package in that
environment, or add the repository source directory to `PYTHONPATH`.

```sh
PYTHONPATH=src sage experiments/notebooks/petersen_coset_enumeration.sage
```

The script has no command-line options. Its `main()` function computes
the distribution and checks the group order, distinct-matrix coverage,
both histograms, and the weighted total. The companion notebook runs
this same entry point.

Writing $P$ for the Petersen graph, the group sizes are
$$
|\operatorname{Aut}(P)|=120,\qquad
[S_{10}:\operatorname{Aut}(P)]=30{,}240.
$$

| Boundary rank | Distinct labeled adjacency matrices | All permutations |
|---:|---:|---:|
| 4 | 376 | 45,120 |
| 6 | 8,496 | 1,019,520 |
| 8 | 21,368 | 2,564,160 |
| Total | 30,240 | 3,628,800 |

## Permutation convention

GAP supplies right cosets $\operatorname{Aut}(P)g$.
`permute_adjacency` interprets its list as old vertices in their new
order: the entry at $(i,j)$ is `T[perm[i], perm[j]]`.
The list for a representative $g$ is therefore the image list of
$g^{-1}$, shifted from GAP labels $1,\ldots,10$ to labels $0,\ldots,9$.
It is obtained with `libgap.ListPerm(g**-1, 10)`.
With this action, the 30,240 representatives give 30,240 distinct
adjacency matrices, checked directly before accepting the histogram.

The [Python computation](petersen_brd_right_cosets.md) uses image lists
as new labels of old vertices, so its representatives are used directly.
Both conventions give the displayed distribution.

These ranks describe adjacency deletion boundaries. The ordinary graph
incidence differential is a different map; for the Petersen graph it
has rank 9 and gives Betti numbers $(1,6)$.
