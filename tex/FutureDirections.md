# Further work on simplicial tensor modules

The current manuscript is [Computations with simplicial tensor modules](dstm_short.tex). It includes normalized contractions, a finite kernel presentation for tensor-generated submodules, generic (3, 3) kernel geometry, graph incidence realizations, and an extension to rooted trees. The following questions concern work beyond those results.

## Exact computations for generated submodules

Section 6 provides a degree bound for a generator in degree n. An implementation should form the spans of normalized face images, compute the differentials on those spans, and take exact ranks over a declared field. It should distinguish the full module, a tensor-generated submodule, and a free module on encountered tensors.

The existing scripts described in [homology_basis.md](../docs/homology_basis.md), [normalized_moore_complex_homology.md](../docs/normalized_moore_complex_homology.md), and [sagemath_normalized_tensor_homology.md](../docs/sagemath_normalized_tensor_homology.md) do not provide that calculation. Their limitations are recorded in those pages.

## Kernel geometry

Appendix A proves rationality of the closure of the generic kernel image and identifies the (3, 3) image with a projective plane minus one point. Further calculations could determine images, dimensions, and singularities for other shapes.

Equality of kernel sequences classifies quotients with chosen generators. Comparing abstract isomorphism classes also requires accounting for changes of generator. Relative Grassmannians over a base scheme require explicit finiteness and base-change hypotheses.

## Graph calculations

The shifted tensor construction realizes the incidence complex of a finite graph. Further examples should specify generators, coefficient domain, grading, and differential before reporting homology.

An adjacency boundary matrix also gives rank statistics under vertex relabeling. Its matrix rank differs from the rank of a differential between chain groups. A distribution averaged over a subgroup must retain that subgroup and the graph presentation in its definition.

## Rooted trees

Section 7 defines a symmetric dendroidal tensor presheaf and proves the horn-kernel and filling results. Further work could compute these modules on larger trees and study tensor-generated subpresheaves.

A dendroidal homology construction needs a specified complex and face signs. The planar Dold–Kan correspondence alone does not define homology for the symmetric presheaf used here.

## General coefficient objects

Section 5 contracts tensor products of split Čech nerves of arbitrary modules. Extensions to sheaves or other abelian categories should state which products, tensor products, kernels, and exactness properties they use. Coordinate rank formulas require additional hypotheses when the terms are no longer finite free modules.

## Computational cost

The number of face images, ambient size, and coefficient growth determine the cost of the finite rank calculation. Benchmarks should report these parameters and distinguish exact elimination from floating-point estimates.

The neural-network experiments use the boundary as a linear observation or regularizer. Claims about prediction or training require separate empirical comparisons with matched controls.
