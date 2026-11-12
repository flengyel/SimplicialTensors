# Recovery of a degenerate filler

[Source](../experiments/verify_degenerate_preference.py) applies one degeneracy to a coordinate tensor, forms every horn of the result, and checks whether the Moore filler recovers that result.

```sh
python experiments/verify_degenerate_preference.py
```

The inputs are `E_(0,1)` in shape `(2, 2)` and `E_(0,1,2)` in shape `(3, 3, 3)`. For each, the program tries all degeneracy indices and all resulting horn indices. The six matrix comparisons return `True`.

Each case prints the shapes and an equality result. The final summary combines those results. Its exception handlers count an invalid degeneracy or horn index as a skipped success.

The cases test elements in individual degeneracy images. The degenerate submodule is the sum of all these images, the broader membership question handled by [is_degen](is_degen.md).
