# Finite-difference analysis helpers

The importable analysis helpers are
[finite_difference_analysis.py](../../src/simplicial_tensors/notebooks/finite_difference_analysis.py)
and
[finite_difference_analysis_horn_invariance.py](../../src/simplicial_tensors/notebooks/finite_difference_analysis_horn_invariance.py).
They use
[horn_map_reduce.py](../../src/simplicial_tensors/notebooks/horn_map_reduce.py)
to enumerate coordinates omitted by the included horn faces. Dask is an optional
dependency required when importing these analysis helpers.

The modules compare finite differences of the coordinate counts with
combinatorial formulas. Their `main()` functions run the configured examples.
Install the package and Dask in the active Python environment:

```sh
python -m pip install -e . dask
python -m simplicial_tensors.notebooks.finite_difference_analysis
python -m simplicial_tensors.notebooks.finite_difference_analysis_horn_invariance
```

`setup_logger(log_dir=None)` creates a timestamped analysis log and returns the
logger and filename. The default log directory is
`experiments/results/finite_difference_analysis/runs/logs/`, relative to the
current working directory. Pass `log_dir` to choose another directory. Importing
the modules does not start a calculation or create a log.

The [stored logs dated 24 and 28 June 2025](../../experiments/results/finite_difference_analysis/logs/)
contain the recorded outputs. Coordinate counts concern the common kernel of
included faces; normalized homology dimensions use the
[separate exact homology API](../api/homology.md).
