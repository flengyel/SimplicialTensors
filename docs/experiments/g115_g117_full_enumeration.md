# Exact boundary-rank distributions for G115 and G117

[The experiment](../../experiments/g115_g117_full_enumeration.py) selects
NetworkX Graph Atlas entries 115 and 117. Both have six vertices and
seven edges. For every vertex permutation $p$, it forms $A^p_{ij}=A_{p(i),p(j)}$,
takes the XOR sum of all six principal deletions, and computes the
resulting $5\times5$ matrix's rank over $\mathbb F_2$.

## Run

The program requires NumPy and NetworkX and has no command-line options.

```sh
python -m pip install -e . networkx
python experiments/g115_g117_full_enumeration.py
```

| Boundary rank | G115 | G117 |
|---:|---:|---:|
| 0 | 12 | 0 |
| 2 | 244 | 224 |
| 4 | 464 | 496 |
| Total | 720 | 720 |

Every permutation is counted, including those giving the same adjacency
matrix. Changing the starting vertex order permutes the full set of
labelings, so the differing histograms distinguish these two graph
isomorphism classes.

The printed adjacency-spectrum comparison uses `numpy.linalg.eigvalsh`
and `numpy.allclose`. It is a floating-point check. The boundary ranks
and counts are exact, since the face entries and elimination are binary.

The program implements principal deletion and binary rank locally.
It does not call `tensor_ops`. The output counts ranks of individual boundary matrices.
