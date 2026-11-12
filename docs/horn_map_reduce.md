# Enumerate coordinates omitted by a horn

[Source](../experiments/horn_map_reduce.py) provides `compute_missing_indices_dask(shape, horn_j)`. It returns tuples containing every face label other than `horn_j`. They index the coordinate tensors killed by all included faces.

```sh
python -m pip install dask
python experiments/horn_map_reduce.py
```

Dask is required by this script and is not a core package dependency. The function filters a Dask bag of all index tuples once for each included face.

For `k` axes and degree `n = min(shape) - 1`, the result is empty when `k < n`. For `(2, 2)` omitting face 1 it is `{(0, 0), (0, 1), (1, 0)}`. For `(4, 4, 4)` omitting face 0 it consists of the six permutations of `(1, 2, 3)`. The script asserts both cases and prints further counts.

An empty shape returns an empty set. For a nonempty shape, axis lengths
must be positive integers and the horn index must lie between zero and
the degree. These checks precede the dimension bound. In degree zero,
the function returns every tuple because the included-face set is empty.
