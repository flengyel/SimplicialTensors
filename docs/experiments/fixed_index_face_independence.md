# Coordinate images under one face

[Source](../../experiments/fixed_index_face_independence.py) applies a fixed face to each coordinate tensor, discards zero images, and computes the NumPy rank of the surviving flattened arrays.

```sh
python experiments/fixed_index_face_independence.py
```

The cases are `(3, 3)` with indices 0 and 1, `(4, 4, 4)` with indices 0 and 2, and `(3, 4, 5)` with index 1. Each prints the survivor count, matrix rank, and equality comparison.

A coordinate tensor survives precisely when its tuple avoids the deleted label. Deleting that label maps the survivors bijectively to the face's coordinate basis. The expected count is `prod(s - 1 for s in shape)`. The checks give 4 survivors for `(3, 3)` and 24 for `(3, 4, 5)`.

The implementation uses floating-point rank. The assertion concerns the surviving coordinate tensors. The face map on the full module has a nonzero kernel.
