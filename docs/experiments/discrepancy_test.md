# Matrix multiplication and the deletion boundary

[The script](../../experiments/discrepancy_test.py) computes
$$
D(A)=\partial(A^2)-(\partial A)^2
$$
for square matrices. It compares this matrix with each principal-deletion face of $A$ using cosine similarity, a Frobenius norm ratio, and a least-squares scalar fit.

The default run examines random square matrices of sizes 3, 9, 11, and 23. It also examines seeded permutation matrices, tests powers of their faces for nilpotence, idempotence and finite cycles, and summarizes 100 random matrices of size 5.

```sh
python -m pip install -e .
python -m pip install matplotlib
MPLBACKEND=Agg python experiments/discrepancy_test.py
```

`--output` selects the summary figure path. Results are printed to the terminal, and the statistical summary writes `experiments/results/discrepancy_test/runs/figures/discrepancy_analysis.png`.

The tests use floating-point tolerances and bounded searches over powers. They report observations for the sampled matrices. Cosine similarity concerns flattened matrix entries, and a scalar-fit residual concerns multiples of an individual face. Neither test decides whether $D(A)$ lies in the span of all faces or whether matrices are similar by conjugation.
