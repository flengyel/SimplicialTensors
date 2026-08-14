# Boundary observer study

`experiments/boundary_observer_study.py` tests whether the raw DSTM boundary
is a privileged passive observer of a tied square residual weight.  Training
does not use the boundary.  Whole trajectories are split between observer
fitting, tuning, and held-out evaluation.

The controls have the same rank and nonzero singular values as the DSTM
boundary, map into its boundary range, and therefore retain the
boundary-of-boundary metacheck.  Primary endpoints are future train and
validation loss plus norm-matched shadow interventions.

The run was a negative screen: DSTM did not beat the matched random observers
on any primary endpoint.  The detailed report is
[`experiments/boundary_observer_report.md`](../experiments/boundary_observer_report.md),
and machine-readable results are under
`experiments/results/boundary_observer_study/`.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/boundary_observer_study.py
```
