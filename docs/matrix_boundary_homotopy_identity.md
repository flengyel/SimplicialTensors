# Permutation formula for matrix boundaries

[Source](../experiments/matrix_boundary_homotopy_identity.py) compares the signed deletion boundary of a relabeled matrix with a sum of individually relabeled principal faces.

```sh
python experiments/matrix_boundary_homotopy_identity.py
```

It defines `P[sigma[j], j] = 1` and compares `bdry(P @ A @ P.T)` with the sum of `(-1)**i * Q_i.T @ face(A, sigma_inverse[i]) @ Q_i`, using `induced_subperm_matrix`. The default loops request 100 directed zero-diagonal matrices of size 7 over the integers and after reduction modulo 2. The seeds are `123 + t`. A loop prints its first mismatch and stops.

The current implementation fails this comparison. A run with seeds 0 through 9 returned `False` in every signed and mod-two trial. The convention for `Q_i` and its use on the right side require reconciliation.

This is a diagnostic for ordered face maps under vertex relabeling. It constructs no chain homotopy. Each `Q_i` depends on the deleted vertex, so the formula involves more than a single conjugation of the boundary.
