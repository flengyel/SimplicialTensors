# Computational examples for diagonal simplicial tensor modules

Install the package from the repository root and run the accompanying script:

```sh
python -m pip install -e .
python experiments/dstm_examples.py
```

The script imports `face`, `degen`, `bdry`, `horn`, and `filler` from
`simplicial_tensors.tensor_ops`. It uses fixed integer arrays and computes
matrix ranks exactly over the rational numbers with SymPy. These checks
illustrate the stated examples over Q; they do not implement arbitrary
coefficient rings or verify all the theorems in the paper.

## A one-axis tensor

The number of axes and the simplicial degree are different quantities. The
vector `[3, 5]` has one axis and simplicial degree one. Its faces are `[5]`
and `[3]`, and its alternating boundary is `[2]`. Deleting an entry leaves
an empty vector only when the original vector has length one.

## A generated submodule with nonzero first homology

Let `T = [[0, 1], [0, 0]]`, and let Y be the simplicial submodule generated
by T. Both faces of T are zero, so Y has zero degree-zero part and a
one-dimensional degree-one part. Its degree-two part is spanned by

```text
s_0 T = [[0, 0, 1],       s_1 T = [[0, 1, 1],
         [0, 0, 1],                [0, 0, 0],
         [0, 0, 0]]                [0, 0, 0]]
```

Both boundaries vanish. Thus `dim H_1(Y) = 1`. The matrix unit E_12 in
the ambient degree-two module has boundary T, but is outside the span of
these two degeneracies. The script verifies that exclusion by comparing
exact matrix ranks. Ambient contractibility therefore does not imply
contractibility of this generated submodule.

## A horn with a two-dimensional family of fillers

For `T = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, omit face zero. The two
prescribed faces are

```text
d_1 T = [[1, 3],       d_2 T = [[1, 2],
         [7, 9]]                [4, 5]]
```

All fillers have the form `[[1, 2, 3], [4, 5, u], [7, v, 9]]`. The
parameters u and v are free because E_12 and E_21 lie in the kernels of
both prescribed faces. The script constructs the restriction map
`B -> (d_1 B, d_2 B)` on the nine matrix units, verifies that its kernel
has dimension two, and checks that E_12 and E_21 form a basis of that
kernel. It also calls the library's horn-filling algorithm and checks
the resulting faces. This proves the asserted affine family by exact
linear algebra, rather than by testing a sample of parameter values.

The current `horn` function introduces a floating-point dtype through
its zero placeholder. The script verifies that its integer data are
unchanged when converted back to integers before exact calculations.

## Expected output

```text
Vector faces: [[5], [3]]
Vector boundary: [2]
dim Y_2: 2
rank of boundary on Y_2: 0
dim H_1(Y): 1
Boundary of ambient primitive equals T: True
Ambient primitive lies in Y_2: False
Horn faces d_1, d_2: [[[1, 3], [7, 9]], [[1, 2], [4, 5]]]
Horn kernel dimension: 2
Computed filler has the prescribed faces: True
All fillers: [[1, 2, 3], [4, 5, u], [7, v, 9]]
```
