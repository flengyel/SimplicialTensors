# Shape enumeration

[Source](../experiments/shape_generator.py) provides `generate_shapes_by_dimension(max_total_shapes=None)`. It starts at simplicial degree 2, chooses at least that many axes, and adds nonnegative offsets with minimum zero.

```sh
python experiments/shape_generator.py
```

The demonstration prints 35 shapes. The initial sequence is `(3, 3)`, `(3, 4)`, `(4, 3)`, `(3, 5)`, `(5, 3)`, and `(3, 6)`.

Within one degree and axis count, offsets are grouped by their sum, then compositions of that sum are lexicographically ordered. This differs from lexicographic order on all shape tuples. The offset sum is capped at 15.

With a positive `max_total_shapes`, the loop over axis counts is unbounded and therefore never advances beyond simplicial degree 2 before reaching the limit. With `None`, explicit cutoffs allow degrees 2, 3, and 4 and finitely many axis counts.

The generated shapes satisfy `len(shape) >= min(shape) - 1`. They cover a portion of the range where the full-module horn kernel may be nonzero. No fillers are computed.
