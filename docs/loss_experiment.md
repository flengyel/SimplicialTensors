# Gaussian reconstruction-loss experiment

[The script](../experiments/loss_experiment.py) compares the error of reconstructing a random tensor $T$ from its boundary with the error of reconstructing it from a random array $S$ of the same boundary shape. Reconstruction uses `degen(S, 0)`. Both the original and reconstructed arrays are normalized by their Frobenius norms before their difference is measured.

```sh
python -m pip install -e .
python -m pip install matplotlib pandas seaborn tqdm
python experiments/loss_experiment.py
```

The default run performs 10,000 trials for each of 13 configured shape entries, including a repeated `(5, 5, 5)` entry. It reports mean and standard deviation of each loss, the fraction of trials in which the random-array error exceeds the boundary error, a paired t-test, and an effect-size calculation.

Outputs are `experiment.log`, `experiment_results.csv`, and seven images under `plots/`. The script selects Matplotlib's `TkAgg` backend and calls blocking plot windows, so the default command requires Tk and a graphical session.

The tensor $T$ uses seed `trial + 12345`. The comparison array $S$ is created with the fixed seed `___SEED___` on every trial. Consequently $S$ repeats within each shape. The statistics therefore condition on one fixed comparison array per shape.
