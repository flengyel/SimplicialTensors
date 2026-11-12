# Range tensor examples

[Source](../experiments/examples_range_tensors.py) uses arrays filled with consecutive integers to inspect horn reconstruction. It prints range tensors, the predicate `len(shape) < min(shape) - 1`, reconstruction diagnostics, and a fixed `(3, 3)` matrix calculation.

```sh
python experiments/examples_range_tensors.py
```

The configured shapes are `(2, 2)`, `(3, 3)`, `(4, 4)`, `(5, 5, 5)`, `(6, 6, 6)`, and `(7, 7, 7)`. The larger cases can be expensive. Most calculations occur at module scope, so importing the script also runs them.

The reconstruction helper compares the input with selected fillers for omitted faces 1 through the simplicial degree. It omits face 0. The program passes `proceed_anyway=True` to continue after its usual shape and boundary checks.

The printed label `degree = len(shape)` denotes the number of axes. The simplicial degree is `min(shape) - 1`. For `(4, 4)` the final Boolean reuses the earlier `(3, 3)` result because the new helper return is not assigned to `reconstructed`. Use that call's diagnostics when inspecting this case.
