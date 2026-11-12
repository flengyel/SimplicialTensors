# Exhaustive graph boundary checks

[The script](../experiments/cospectral_proof.py) contains two computations
over $\mathbb F_2$. It first finds a pair of nonisomorphic six-vertex
Graph Atlas graphs whose adjacency eigenvalues agree after rounding to
five decimal places. It counts boundary-matrix ranks over every labeling
of each graph.

The second computation checks up to all $10!$ labelings of the
Petersen graph and stops if any labeling has zero boundary. The boundary
is the XOR sum of principal submatrices obtained by deleting one row
and the corresponding column. Rank uses binary Gaussian elimination.

## Run

The script requires NumPy and NetworkX and has no command-line options.

```sh
python -m pip install -e . networkx
python experiments/cospectral_proof.py
```

Output includes the six-vertex pair, both histograms, Petersen progress,
elapsed time, and whether rank zero occurred. The program saves no result
file. The Petersen loop processes up to 3,628,800 permutations.

Run only the six-vertex experiment as follows.

```sh
python - <<'PYTHON'
import runpy

experiment = runpy.run_path("experiments/cospectral_proof.py")
experiment["run_cospectral_experiment"]()
PYTHON
```

Spectral grouping uses floating-point arithmetic. Boundary ranks and
permutation counts use exact binary arithmetic. For a specified pair
with stable Atlas identifiers, use
[the G115/G117 enumeration](g115_g117_full_enumeration.md).

A nonzero boundary means that the adjacency matrix fails the cycle
condition and cannot represent a homology class in that degree.
The Petersen test establishes absence of a rank-zero deletion boundary
when its full loop completes. It does not compute incidence homology.
