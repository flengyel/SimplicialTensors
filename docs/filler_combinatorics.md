# Symbolic filler coordinates

[Source](../experiments/filler_combinatorics.py) constructs a `SymbolicTensor` with independent entries. For each omitted face, `validate_filler_conjecture(shape)` computes coordinates absent from the included faces, constructs the Moore filler, and compares it with the input.

```sh
python experiments/filler_combinatorics.py
```

The active shape list in `main` is `(3, 4)`, `(3, 3, 4)`, and the corresponding shapes through six axes of length 3 followed by an axis of length 4. It overwrites the earlier list.

Each report gives the uncovered coordinate count, nonzero coordinates of the difference, and comparison with the shape predicate. In the square case with `k` axes and simplicial degree `k`, the uncovered tuples are the `k!` orders of the included face labels.

A smaller check is `validate_filler_conjecture((3, 4))`, imported from `experiments.filler_combinatorics`. It returns `True` and finds two uncovered coordinates for each horn. Independent symbolic entries test the formula. Equality of a selected filler with one numerical input would not establish uniqueness.

`show_filler_formula` inspects a difference coordinate and compares its expanded monomial count with `2**N`. The default run exits before these demonstrations. The monomial comparison is an experimental check.
