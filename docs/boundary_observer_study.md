# Passive boundary observer study

[The experiment](../experiments/boundary_observer_study.py) observes a tied square residual weight during digits training. The boundary does not enter the loss or optimizer. It fits ridge forecasts to whole training trajectories and evaluates loss forecasts and perturbations on held-out trajectories.

Default settings are hidden width 10, three recurrent steps, 90 epochs per trajectory, and 8/3/5 trajectories for fitting, tuning, and evaluation. Sixteen random observers have the same rank and nonzero singular values as the DSTM boundary and map into its boundary range. This preserves the next-boundary identity in the controls.

```sh
python -m pip install -e .
python -m pip install scikit-learn
python experiments/boundary_observer_study.py --help
python experiments/boundary_observer_study.py --quick \
  --output-directory /tmp/boundary-observer-smoke \
  --report-path /tmp/boundary-observer-smoke.md
```

The default output directory is `experiments/results/boundary_observer_study` and the default report is `experiments/boundary_observer_report.md`. Choose different paths for a smoke run to preserve the checked-in full-run results. `--quick` uses two fitting, one tuning, and two held-out trajectories for 12 epochs, with two random controls.

The output directory contains trajectory metrics, forecast and shadow-intervention CSV files, and `summary.json` with configuration, operator checks, aggregates, runtime, and software versions. The generated Markdown report gives the primary tables.

In the [stored full run](../experiments/boundary_observer_report.md), DSTM falls below the random-observer mean on both future-loss forecasts and both shadow-intervention endpoints. These results concern a linear observer, one dataset, one tied architecture, and the specified trajectory split.
