# Count degenerate coordinate tensors

[Source](../../experiments/count_standard_basis_tensors.py) enumerates arrays with one coordinate equal to one and calls `tensor_ops.is_degen` on each one. For every configured shape it reports degenerate and nondegenerate counts and the nondegenerate indices.

```sh
python experiments/count_standard_basis_tensors.py
```

Run from the repository root. The program logs to the console and a
timestamped `standard_basis_tensors_<timestamp>.log` file under
`experiments/results/count_standard_basis_tensors/runs/logs/`.
`--log-dir` selects another directory; the script creates it when needed. Its configured cases include matrices
through size 8 and three-axis arrays. Each coordinate test computes a
Moore projection with exact coefficients.

Importing the script creates no files and does not configure logging.
For a smaller calculation, import `count_basis_tensors_for_shape` from `experiments.count_standard_basis_tensors` and call it on `(3, 3)` and `(4, 4)`. These report respectively 2 degenerate and 7 nondegenerate coordinate tensors, then 16 degenerate and no nondegenerate tensors.

These counts concern membership in the sum of degeneracy images. They do not count all nondegenerate linear combinations or give homology dimensions. See [the degeneracy API](../api/is_degen.md).
