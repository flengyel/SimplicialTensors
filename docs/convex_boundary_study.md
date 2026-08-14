# Convex boundary study

`experiments/convex_boundary_study.py` is a paired, validation-tuned matrix
regression experiment for the diagonal-boundary Tikhonov penalty.

It compares:

- the DSTM penalty (|\partial W\|_F^2);
- ridge regression;
- a Haar-conjugated penalty with exactly the same spectrum and kernel
  dimension; and
- a standard two-dimensional grid Laplacian.

Truths are constructed in the DSTM kernel, sampled isotropically, or obtained
by coordinate-permuting an aligned truth.  The included 50-seed results use
two signal levels and are interpreted in
[`neural_boundary_research_report.md`](neural_boundary_research_report.md).

Run from the repository root:

```bash
PYTHONPATH=src python experiments/convex_boundary_study.py \
  --seeds 50 \
  --output-prefix experiments/results/convex_boundary_study/signal2
```

Use `--signal-norm 1` to reproduce the lower-SNR robustness run.
