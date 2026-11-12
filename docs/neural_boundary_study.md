# Controlled neural boundary study

[The experiment](../experiments/neural_boundary_study.py) trains a float64 NumPy ReLU classifier with widths 64, 24, and 10 on scikit-learn's digits data. It compares momentum SGD with a DSTM weight penalty, DSTM Sobolev smoothing, controls with the same spectrum, an ordinary grid smoother, and architecture balance.

Each method receives the same number of validation configurations. The selected configuration is then fixed for paired confirmation seeds. The full run uses tuning seeds 0–2 for 70 epochs and confirmation seeds 10–17 for 100 epochs. The `--quick` run uses fewer configurations, tuning seeds 0–1 for 35 epochs, and confirmation seeds 10–12 for 55 epochs.

```sh
python -m pip install -e .
python -m pip install scikit-learn
python experiments/neural_boundary_study.py --help
python experiments/neural_boundary_study.py --quick --output-dir /tmp/neural-boundary-smoke
```

For the full protocol, omit `--quick` and choose an output directory. The script writes `tuning_runs.csv`, `confirmation_runs.csv`, and `summary.json`. It also tests one-step permutation equivariance, 40-epoch training from equivalent hidden relabelings, 100 hidden permutations, and 100 positive ReLU rescalings.

The stored full-run results are under [`experiments/results/neural_boundary_study/`](../experiments/results/neural_boundary_study/). The DSTM penalty improved mean accuracy over the baseline in that run, while its paired comparison with the control of the same spectrum did not establish an advantage. The experiment evaluates penalties and gradient filters. Exact boundary analysis and synthesis are documented separately in [boundary_homeostasis.md](boundary_homeostasis.md).
