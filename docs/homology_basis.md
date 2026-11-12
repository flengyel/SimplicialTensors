# Exact tensor and graph homology examples

[experiments/homology_basis.py](../experiments/homology_basis.py) computes normalized homology of tensor-generated simplicial modules. It also computes ordinary graph homology from the shifted tensor representation of vertices and edges.

Run from an installed checkout with NumPy and SymPy.

~~~sh
python experiments/homology_basis.py
python experiments/homology_basis.py --modulus 2
~~~

The default coefficient field is $\mathbb Q$. The optional modulus must be prime. All ranks use exact arithmetic. The script prints normalized chain dimensions, differential ranks indexed by source degree, and Betti numbers in ascending degrees.

| Example | Normalized dimensions over $\mathbb Q$ | Betti numbers |
|---|---|---|
| Zero $3\times3$ tensor | $(0,0,0)$ | $(0,0,0)$ |
| Nonzero vertex | $(1)$ | $(1)$ |
| $E_{01}$ | $(0,1)$ | $(0,1)$ |
| Tensor `[[0,1,0],[0,1,1],[1,1,2]]` | $(1,3,1)$ | $(0,1,0)$ |
| Path on three vertices | $(3,2)$ | $(1,0)$ |
| Cycle on three vertices | $(3,3)$ | $(1,1)$ |
| Two disjoint edges | $(4,2)$ | $(2,0)$ |
| Petersen graph | $(10,15)$ | $(1,6)$ |

The helper `compute_homology_dimension(tensor, degree=None, *, modulus=None)` takes an integer or rational tensor. The default degree is `min(tensor.shape)-1`. It returns zero above the normalized complex. `standard_basis_tensor(idx, shape)` constructs a coordinate tensor.

The library [homology implementation](homology.md) takes linear relations among face images into account and preserves rectangular offsets. Its normalization projects face images into the intersection of the positive face kernels. Counting coordinate units outside the degeneracy span would not compute this quotient.

The graph examples use `graph_embedding` and `graph_chain_homology`, documented in [rank-shifted tensors and graph chains](rank_shifted.md). Their differential is the oriented incidence matrix obtained from normalized tensor faces. Applying simplicial deletion directly to an adjacency matrix gives a different complex.
