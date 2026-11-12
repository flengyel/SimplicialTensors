# Initialization and reconstruction losses

[The script](../experiments/consolidated_experiment.py) compares two normalized Frobenius reconstruction errors for a random tensor $T$.

- The boundary reconstruction duplicates index zero of $\partial T$.
- The comparison duplicates index zero of an independently initialized array $S$ with the boundary shape and multiplies its error by $1+\|\partial S\|_F$.

It runs 200 trials for each combination of shapes `(128, 64)` and `(32, 3, 3, 3)` with Glorot, He, and orthogonal initialization. Random sampling uses NumPy's global generator without an explicit script-level seed.

```sh
python -m pip install numpy scipy
python experiments/consolidated_experiment.py
```

The script prints one result dictionary per shape and initialization. It reports the difference of mean errors, a two-sample Kolmogorov–Smirnov statistic and p-value, and an effect-size calculation.

The reported permutation-test p-value is hard-coded to `0.0`. The function computes only the observed mean difference and performs no permutation test. Interpret that field as an unimplemented output. The file also defines its own face, boundary, and degeneracy routines, so these results exercise its local implementation rather than the installed package API.
