# Numerical relative-homology prototype

[experiments/normalized_moore_complex_homology.py](../experiments/normalized_moore_complex_homology.py) attempts to construct a mapping cone from iterated faces and degeneracies of a random tensor. The current script fails on its first default example and provides no validated homology dimensions.

## Method and defects

The builders enumerate tensors through one degree above the starting degree and use each encountered tensor as an independent basis symbol. They do not compute linear relations among those tensors in the ambient module.

The `key_tensor` function sorts string representations of all entries. Tensors with the same shape and multiset of entries receive the same key even when the entries occupy different positions. Faces depend on those positions.

Normalization uses SciPy's floating-point `null_space`. Its columns are basis vectors, but later code applies `np.hstack` to the rows. On the first default input, a subsequent shape lookup raises `IndexError`. The final `len(null_space(D_cone))` would count rows instead of nullity. A homology dimension also requires the incoming boundary rank.

## Running the diagnostic

The script requires NumPy, SciPy, SymPy, and an installed checkout.

~~~sh
python experiments/normalized_moore_complex_homology.py
~~~

It generates random tensors for four fixed shapes and prints object sizes and face-map keys. The first example currently stops with “`IndexError`: tuple index out of range.” No result files are written.

For a reproducible homology example, follow [the ancillary instructions](../tex/anc/README.txt). Section 6 of [generated_section.tex](../tex/generated_section.tex) obtains a finite quotient complex using linear spans and exact normalization.
