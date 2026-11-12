# Shape enumeration

[Source](../../experiments/shape_generator.py) enumerates a finite set of shapes for which the number of axes is at least the simplicial degree. The function `generate_shapes_by_dimension(max_total_shapes=None, *, min_degree=2, max_degree=4, max_axes=4, max_offset_sum=2)` exposes all bounds.

A shape of degree $n$ has $k$ axes, with $n\leq k\leq\mathtt{max\_axes}$. Its entries are $n+1+d_j$, where the nonnegative offsets have minimum zero and sum at most `max_offset_sum`. Thus the minimum axis length is $n+1$. Shapes are ordered by degree, number of axes, offset sum, and then lexicographically within each offset sum.

```sh
python experiments/shape_generator.py
python experiments/shape_generator.py --limit 70 --max-offset-sum 2
```

The demonstration prints the first 35 shapes. The defaults define 70 shapes in degrees 2, 3, and 4. Their initial sequence is `(3, 3)`, `(3, 4)`, `(4, 3)`, `(3, 5)`, and `(5, 3)`. The 35-shape prefix reaches degree 3.

For the Python function, `max_total_shapes=None` yields the whole bounded set, zero yields no shapes, and a positive integer caps the output. Degree and axis bounds are positive integers, `max_degree` is at least `min_degree`, and `max_axes` is at least `max_degree`. The offset bound is a nonnegative integer.

The command accepts `--limit`, `--min-degree`, `--max-degree`, `--max-axes`, and `--max-offset-sum`. NumPy and the tensor library are not required. The program enumerates shapes and computes no horns or fillers.
