# Reconstruction losses with normal initialization

[The script](../experiments/consolidated_experiment_with_n01.py) runs the [initialization comparison](consolidated_experiment.md) with an additional standard-normal initialization and matrix shape. It imports the comparison implementation from `consolidated_experiment.py`.

```sh
python -m pip install -e .
python experiments/consolidated_experiment_with_n01.py --shape 3,3 --trials 6 --resamples 64 --seed 123
```

The default run uses 200 trials for each of `(512, 256)`, `(128, 64)`, and `(32, 3, 3, 3)`, with Glorot, He, orthogonal, and standard-normal initialization. `--shape`, `--trials`, `--resamples`, and `--seed` have the same meanings as in the earlier comparison.

Each trial draws new arrays from a local seeded generator. The printed dictionaries record the parameters, paired permutation test, empirical KS distance, and standardized mean difference. No result file is written. The loss definitions and test interpretation are given in the [comparison documentation](consolidated_experiment.md).
