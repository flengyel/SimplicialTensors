# Tensor-boundary notebooks

The notebook documents in [experiments/notebooks](../../experiments/notebooks/)
contain symbolic boundary equations and finite linear-algebra calculations.
Use a SageMath Jupyter kernel for the polynomial-ring and symbolic calculations.
Install this package in that kernel's Python environment before executing cells
that import `simplicial_tensors`.

| Notebook | Calculation |
|---|---|
| [Homology_calculation.ipynb](../../experiments/notebooks/Homology_calculation.ipynb) | A boundary matrix for a symbolic square matrix, preimage equations, and kernel comparisons. |
| [homology_syzygy.ipynb](../../experiments/notebooks/homology_syzygy.ipynb) | Symbolic equations for a matrix preimage under the alternating principal-deletion boundary. |
| [homology_syzygy_fixed.ipynb](../../experiments/notebooks/homology_syzygy_fixed.ipynb) | Polynomial-ring setup followed by symbolic preimage equations. |
| [section_6_2_eta_delta_boundary_check.ipynb](../../experiments/notebooks/section_6_2_eta_delta_boundary_check.ipynb) | An integer matrix for deletion of entries in words and a column-space membership example. |

These worksheets define their coefficient rings in their setup cells and use
interactive Sage operations. The normalized homology API is described in
[exact tensor homology](../api/homology.md); it returns chain dimensions,
boundary ranks, and Betti numbers over rational or prime fields.

The [CoCalc notebook](../../experiments/notebooks/cocalc_compatible_tensor_ops.ipynb)
uses the shared symbolic implementation. Its four Python cells check a generic
horn, correction rank, the zero tensor, and a sum of degeneracies. See the
[SageMath-compatible API](../api/sagemath_compatible_tensor_ops.md).

The [Petersen notebook](../../experiments/notebooks/petersen_coset_enumeration.ipynb)
invokes the same entry point as the Sage script. Its guide gives the
[permutation convention and exact rank counts](petersen_coset_enumeration.md).
