# Range tensor examples

[Source](../../experiments/examples_range_tensors.py) compares tensors filled with consecutive integers with the fillers selected from their horns. It prints each tensor, its number of axes, its simplicial degree, the predicate `len(shape) < min(shape) - 1`, and the reconstruction result. A fixed $(3,3)$ matrix also illustrates a horn, its selected filler, degeneracy, and boundary.

```sh
python experiments/examples_range_tensors.py --shape 3,3 --shape 4,4
python experiments/examples_range_tensors.py --shape 3,3 --verbose
```

The script requires the installed package and NumPy. `--shape` accepts comma-separated positive axis lengths and may be repeated. Without this option, the shapes are `(2, 2)`, `(3, 3)`, `(4, 4)`, `(5, 5, 5)`, `(6, 6, 6)`, and `(7, 7, 7)`. The larger degeneracy calculations can be expensive. Importing the module runs no examples.

The reconstruction helper compares the tensor with selected fillers for omitted faces 1 through the simplicial degree. It omits face 0 and passes `proceed_anyway=True` to continue after its shape and boundary checks. Each printed Boolean is the return value for the displayed shape. Agreement with a selected filler does not establish that the filler is unique.
