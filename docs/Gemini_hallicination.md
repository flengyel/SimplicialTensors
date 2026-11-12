# Tensor-product horn examples

[The experiment](../experiments/Gemini_hallicination.py) constructs two integer tensor products, checks their horn fillers, and evaluates full-module filler uniqueness. All simplicial operations use the package implementation with exact integer coefficients.

The first product is $[0,1,2]\otimes[3,4]\otimes[5,6]$. Its shape is `(3, 2, 2)` and its simplicial degree is one. It has no inner horns. The program checks both outer horns and finds nonunique fillers. It also constructs the filler for the horn with omitted face zero and checks its included faces.

The second product has factors `range(6)`, `range(3, 10)`, `range(5, 11)`, and `range(7, 14)`. Its shape is `(6, 7, 6, 7)`, with four axes and simplicial degree five. The inner horns have unique fillers. The selected filler with omitted face two equals the original tensor.

```sh
python experiments/Gemini_hallicination.py
```

Run from an installed checkout with NumPy. The program prints shapes, degrees, degeneracy tests, horn compatibility, equality of included faces, and equality with a selected filler. Both comparisons pass `allow_degen=True`; degeneracy membership is computed by the Moore projection. The script writes no files and importing it runs no examples. Failure of a checked compatibility or uniqueness identity gives exit status one.

The full-module uniqueness criterion is $p>k$ for positive degree $p$ and $k$ axes, as proved by the horn-kernel calculation in §3 of [the manuscript](../tex/dstm_short.tex). Agreement with one selected filler alone does not establish uniqueness.
