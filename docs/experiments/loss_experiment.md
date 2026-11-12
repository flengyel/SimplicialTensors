# Gaussian reconstruction-loss experiment

[The script](../../experiments/loss_experiment.py) compares two approximations to a Gaussian tensor $T$. The boundary approximation is $s_0\partial T$. The comparison is $s_0S$, where $S$ is an independent Gaussian array with the shape of $\partial T$. The map $s_0$ duplicates index zero along every axis. Each error is the Frobenius distance after normalizing the original and approximating arrays, with zero arrays left unchanged.

```sh
python -m pip install -e .
python experiments/loss_experiment.py --shape 3,3 --trials 6 --seed 12345
```

The default run uses 10,000 trials for each of 12 distinct configured shapes. Repeat `--shape` to supply shapes. `--mean` and `--std` set the Gaussian parameters. A local generator seeded by `--seed` draws new, independent arrays $T$ and $S$ on every trial.

Outputs are `experiment.log` and `experiment_results.csv` under
`experiments/results/loss_experiment/runs/`. `--log` and `--output` change their paths; parent directories are created as needed. Each result records the seed and trial count, means and standard deviations of both errors, the fraction of trials in which the comparison error is at least the boundary error, a paired t-test, and a mean difference standardized by the pooled marginal standard deviations.

The tracked `experiments/results/loss_experiment/experiment_results.csv` uses one fixed comparison array per shape.

Plotting is optional and does not require a graphical session when Matplotlib uses a noninteractive backend.

```sh
python -m pip install matplotlib pandas seaborn tqdm
MPLBACKEND=Agg python experiments/loss_experiment.py --shape 3,3 --trials 6 --plot
```

`--plot` saves seven images in a `figures/` directory beside the selected CSV file.
`--plot-dir` selects another figure directory. `--show` also opens plot windows. `--progress` enables a progress bar and requires tqdm. Importing the script does not configure logging, select a plotting backend, or create files.
