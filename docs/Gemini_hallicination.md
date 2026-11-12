# Tensor-product horn examples

[experiments/Gemini_hallicination.py](../experiments/Gemini_hallicination.py) constructs two integer tensor products and compares them with fillers of selected horns. The executable calculation uses the package's numeric operations.

## Inputs and output

The first input has shape (3, 2, 2) and simplicial degree 1. It has two outer horns and no inner horns. The script prints the tensor, shape predicate, degeneracy checks, and a comparison with outer horns enabled. It fills the horn with missing face 0 and compares the included faces and full tensors.

The second input has shape (6, 7, 6, 7), four axes, and simplicial degree 5. It checks the inner horns and explicitly fills the horn with missing face 2. Output includes Kan compatibility, equality of the reconstructed included faces, and equality of the full tensor with that filler.

The source comments misstate the second degree and confuse the absence of inner horns in degree 1 with the absence of outer horns. The degrees above follow from `min(shape) - 1`.

## Running the examples

From an installed checkout:

~~~sh
python experiments/Gemini_hallicination.py
~~~

The script uses fixed integer arrays and writes no files. The second case invokes a dense degeneracy system with 1,764 rows and 4,500 columns, so it can require substantial time and memory. The comparison can raise `SimplicialException` if the package detects a degenerate boundary.

Comparison with the algorithm's selected filler tests that pair of fillers. The general uniqueness result is the coordinate horn-kernel calculation in §3 of [the current manuscript](../tex/dstm_short.tex). For k axes, fillers in degree p are unique precisely when p > k. Tensor rank is a separate quantity.
