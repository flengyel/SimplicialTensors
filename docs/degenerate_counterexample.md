# `degenerate_counterexample.py` documentation

## Location
- Script: `experiments/degenerate_counterexample.py`
- Core APIs used: `src/simplicial_tensors/tensor_ops.py`

## Purpose
This experiment demonstrates a case where the hypothesis
"the boundary is non-degenerate" is violated.

It uses a fixed tensor and checks:
- its boundary via `bdry(...)`
- whether that boundary is degenerate via `is_degen(...)`
- how the experiment proceeds when the boundary precondition fails

## What the script does
1. Builds a hard-coded tensor named `counterexample` (shape `(3, 8)` after transpose).
2. Computes and prints:
   - the tensor,
   - `bdry(counterexample)`,
   - `is_degen(bdry(counterexample))`.
3. If the boundary is degenerate, prints:
   - `Degenerate boundary detected; comparison skipped (precondition fails).`
   and sets `comparison = None`.
4. Otherwise, runs `n_hypergroupoid_comparison(counterexample, verbose=True)`.
5. Runs `n_hypergroupoid_conjecture(counterexample.shape, verbose=True)`.
6. Computes and prints `horn(counterexample, 1)` and `filler(h, 1)`, then checks equality with the original tensor.

## Important behavior
`n_hypergroupoid_comparison(...)` still enforces a non-degenerate-boundary precondition by default and would raise `SimplicialException("Degenerate boundary.")` if called directly with degenerate boundary input.

This experiment now guards that call and reports the degenerate-boundary condition instead of raising.

## Running the experiment
From the repository root:

```bash
pip install -e .
python experiments/degenerate_counterexample.py
```

If you are not using an editable install, set import path explicitly:

```bash
PYTHONPATH=src python experiments/degenerate_counterexample.py
```
