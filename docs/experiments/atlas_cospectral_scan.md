# Boundary-rank scan of the Graph Atlas

[The script](../../experiments/atlas_cospectral_scan.py) groups NetworkX Graph
Atlas graphs by adjacency eigenvalues rounded to six decimal places.
It then compares pairs that NetworkX identifies as nonisomorphic.
The spectral grouping produces candidates for cospectrality.

For an ordered $N\times N$ adjacency matrix $A$, it forms
$$
B(A)=\sum_{i=0}^{N-1}d_i(A)\quad\text{over }\mathbb F_2,
$$
where $d_i$ deletes row and column $i$. Binary Gaussian elimination
computes the matrix rank of $B(A)$, which is counted across vertex
relabelings. This is a statistic of boundary matrices. Computing homology
requires a specified chain complex and its differentials.

## Run

Install NetworkX in addition to the package dependencies.

```sh
python -m pip install -e . networkx
python experiments/atlas_cospectral_scan.py --n 6 --max_pairs 1
```

The first pair in the six-vertex scan is G63 and G64. Exact enumeration
gives counts `{0: 64, 2: 336, 4: 320}` and `{2: 432, 4: 288}`.
Each total is $6!=720$.

| Option | Meaning | Default |
|---|---|---:|
| `--n` | Required vertex count | — |
| `--max_pairs` | Maximum number of pairs to report | 10 |
| `--exact_max_n` | Enumerate all permutations through this order | 8 |
| `--samples` | Permutations sampled above the exact threshold | 20,000 |
| `--seed` | NumPy sampling seed | 1 |

The Graph Atlas contains graphs on at most seven vertices, so `--n 10`
yields an empty scan. Sampling can be exercised on an available order by
lowering `--exact_max_n`. Members of a sampled pair use seeds `seed`
and `seed + 1`.

Exact counts include all $N!$ permutations, including repeated
adjacency matrices arising from automorphisms. Sampled counts estimate
that distribution. Results are printed to standard output.
