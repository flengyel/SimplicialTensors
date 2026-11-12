# Horn coordinate constraints

[Source](../experiments/homotopy_constraint_verification.py) compares two sets. The first contains tuples with every included face label and is computed by `compute_missing_indices_dask`. The second contains coordinate tensors killed by every included face.

```sh
python -m pip install dask
python experiments/homotopy_constraint_verification.py
```

Dask is required by this script and is not a core package dependency. The default shapes are `(3, 3)`, `(4, 4, 4)`, `(3, 4, 5)`, and `(4, 4, 4, 4)`. The program checks omitted faces 0 and 1 when available, printing the two sizes and whether the sets agree.

This checks the horn kernel's coordinate support. It constructs no homotopy and computes no homology group. See [the Dask enumeration](horn_map_reduce.md).
