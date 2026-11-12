# Reconstruction losses with normal initialization

[The script](../experiments/consolidated_experiment_with_n01.py) extends the [initialization comparison](consolidated_experiment.md) with standard-normal initialization and an additional matrix shape. It imports `bdry` and `degen` from `simplicial_tensors.tensor_ops`.

For each random tensor $T$, it compares a normalized reconstruction from `degen(bdry(T), 0)` with a reconstruction from an independently initialized boundary-shaped array $S$. The latter error is multiplied by $1+\|\partial S\|_F$.

The default run uses 200 trials for each of `(512, 256)`, `(128, 64)`, and `(32, 3, 3, 3)`, with Glorot, He, orthogonal, and standard-normal initializations.

```sh
python -m pip install -e .
python experiments/consolidated_experiment_with_n01.py
```

The script has no command-line options. It prints a dictionary for each shape and initialization and saves no result file. Sampling uses NumPy's global generator. The imported package seeds that generator during import, which makes the run depend on the import and call sequence.

As in the earlier comparison, `permutation_test` returns the observed mean difference and the constant p-value `0.0`. It does not resample permutations. The Kolmogorov–Smirnov test and effect-size fields are computed from the simulated losses.
