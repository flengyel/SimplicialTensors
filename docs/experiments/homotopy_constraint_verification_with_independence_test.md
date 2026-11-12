# Horn constraints and face independence

[Source](../../experiments/homotopy_constraint_verification_with_independence_test.py) computes the NumPy rank of nonzero coordinate images under each face. It then compares uncovered horn coordinates with the coordinate tensors killed by all included faces.

```sh
python -m pip install dask
python experiments/homotopy_constraint_verification_with_independence_test.py
```

This script requires Dask through its import of `horn_map_reduce.py`. The cases are `(3, 3)`, `(4, 4, 4)`, `(3, 4, 5)`, `(4, 4, 4, 4)`, and `(2, 2)`. All available face and horn indices are checked.

Separate summaries cover independence and support constraints. The first uses floating-point rank and the second compares finite sets. Despite its filename, the program does not construct a homotopy.
