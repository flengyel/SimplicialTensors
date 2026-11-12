# A tensor with degenerate boundary

[Source](../experiments/degenerate_counterexample.py) constructs a fixed integer matrix of shape `(3, 8)`. It prints the matrix, its alternating face boundary, and the degeneracy test on that boundary.

```sh
python experiments/degenerate_counterexample.py
```

When the boundary is degenerate, the program skips `n_hypergroupoid_comparison`, whose default precondition rejects that case. It still prints the shape predicate, forms the horn omitting face 1, computes its selected Moore filler, and compares that filler with the input.

A difference exhibits two fillers of the same horn. Equality says that the selected filler reproduces this input. It does not by itself establish uniqueness. See [the comparison helper](n_hypergroupoid_conjecture.md).
