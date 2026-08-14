# Neural boundary study

`experiments/neural_boundary_study.py` is a pure-NumPy/SciPy/scikit-learn
screen of boundary methods on a float64 (64\!-!24\!-!10) ReLU network for
the digits dataset.

The script gives every method the same validation-search budget, freezes the
selected configuration, and evaluates paired confirmation seeds.  It
compares baseline momentum SGD, the DSTM weight penalty, exact DSTM Sobolev
smoothing, exactly isospectral random controls, an ordinary grid smoother,
and typed architecture balance.

It also performs:

- one-step hidden-permutation equivariance checks;
- training from two functionally identical hidden relabelings;
- 100 function-preserving hidden permutations; and
- 100 positive ReLU gauge rescalings.

The complete protocol and interpretation are in
[`neural_boundary_research_report.md`](neural_boundary_research_report.md).

Run from the repository root:

```bash
PYTHONPATH=src python experiments/neural_boundary_study.py \
  --output-dir experiments/results/neural_boundary_study
```

Pass `--quick` for a short smoke run.
