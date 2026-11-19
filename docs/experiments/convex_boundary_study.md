# Convex matrix regression study

[The experiment](../../experiments/convex_boundary_study.py) compares four quadratic penalties on underdetermined matrix regression. They are the diagonal DSTM boundary, ridge, one fixed Haar-orthogonal conjugate of the DSTM penalty, and a four-neighbour grid Laplacian. Positive eigenvalues are normalized to mean one. Every method sees the same train, validation, and test data within a replicate.

The truth families are a vector in the DSTM kernel, an isotropic vector, and a coordinate permutation of a kernel-aligned vector. Validation MSE chooses the regularization parameter. The selected training fit is evaluated on test data.

```sh
python -m pip install -e .
python experiments/convex_boundary_study.py --help
python experiments/convex_boundary_study.py \
  --matrix-size 3 --n-train 6 --n-val 8 --n-test 12 \
  --seeds 1 --lambda-count 3 --output-prefix /tmp/convex-boundary-smoke
```

Defaults are matrix size 8, sample counts 48/128/1024, noise standard deviation 0.5, signal norm 2, 12 seeds, and a 17-point logarithmic regularization grid from `1e-5` to `1e3`. `--seed-start` selects the first replicate. `--base-seed` and `--control-seed` control data generation and the orthogonal control.

The script prints aggregate MSE and writes `PREFIX.json` and `PREFIX.csv`.
The default prefix is
`experiments/results/convex_boundary_study/runs/convex_boundary_study_results`;
`--output-prefix` selects another prefix. The stored [50-seed runs](../../experiments/results/convex_boundary_study/) use signal norms 2 and 1. To reproduce the first, pass `--seeds 50 --output-prefix experiments/results/convex_boundary_study/signal2`. For the second, add `--signal-norm 1` and use the `signal1` prefix.

For the 50 paired replicates at signal norm 2 and noise standard deviation 0.5, the mean test MSEs were as follows.

| Truth family | DSTM boundary | Ridge | Haar-conjugated control | Grid |
|---|---:|---:|---:|---:|
| In the boundary kernel | 1.050 | 1.702 | 2.088 | 1.787 |
| Isotropic | 2.015 | 1.653 | 2.015 | 1.865 |
| Coordinate-permuted kernel truth | 2.163 | 1.787 | 2.184 | 1.968 |

For the kernel truth, the paired DSTM-minus-ridge difference was −0.652, with an approximate 95% interval of [−0.794, −0.511]. At signal norm 1 the mean test MSEs for the same truth family were 0.982 for DSTM and 0.759 for ridge. [Signal norm 2 results](../../experiments/results/convex_boundary_study/signal2.json) · [Signal norm 1 results](../../experiments/results/convex_boundary_study/signal1.json)
