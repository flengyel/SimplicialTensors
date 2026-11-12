# Permutation formula for matrix boundaries

[The experiment](../../experiments/matrix_boundary_homotopy_identity.py)
verifies the deletion-boundary formula under simultaneous relabeling of
matrix rows and columns.

For a permutation $\sigma$ of $\{0,\ldots,n-1\}$, define the permutation
matrix $P$ by $P_{\sigma(j),j}=1$. The relabeled matrix is $PAP^T$.
Let $Q_i$ be the submatrix of $P$ obtained by deleting row $i$ and column
$\sigma^{-1}(i)$. Its rows use the surviving new labels in increasing order,
and its columns use the surviving old labels in increasing order. Then

$$
d_i(PAP^T)=Q_i\,d_{\sigma^{-1}(i)}(A)\,Q_i^T,
$$

so the signed boundary satisfies

$$
\partial(PAP^T)=\sum_{i=0}^{n-1}(-1)^i
Q_i\,d_{\sigma^{-1}(i)}(A)\,Q_i^T.
$$

Each summand relabels one principal face. The matrix $Q_i$ depends on the
deleted vertex.

```sh
python experiments/matrix_boundary_homotopy_identity.py
```

The default run checks 100 directed zero-diagonal matrices of size 7 over
the integers and 100 after reduction modulo 2. Trial $t$ uses seed
`123 + t`. A failed comparison prints the input, permutation, and both
sides and exits with status 1. `run_many_trials` returns a Boolean result
and accepts `trials`, `n`, `directed`, `mod2`, and `seed`.

The script uses NumPy and the package face and boundary operations. It
writes no files.
