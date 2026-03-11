# Future Directions for Diagonal Simplicial Tensor Modules

This note records several questions suggested by the present study of diagonal simplicial tensor modules. Some are direct continuations of the results already proved, especially on horn kernels, generated submodules, and the moduli map. Others are more distant questions whose relevance is clear, even if the appropriate formalism has not yet been developed.

## 1. Directions Closest to the Present Work

### 1.1 Relative and scheme-theoretic variants

A first extension is to formulate the construction relative to a base scheme. One would replace modules over a commutative ring $A$ by quasi-coherent $\mathcal{O}_S$-modules on a scheme $S$, and ask whether the same index-set construction produces a simplicial object $X_\bullet(\vec s; \mathcal{O}_S)$ in $\mathrm{QCoh}(S)$. From this point of view, the main questions concern the behavior of kernel sequences in families, the effect of base change, and the extent to which the moduli problem for generated subobjects can be expressed using relative Grassmannians and relative incidence loci. Any precise representability statement would, of course, require suitable finiteness and flatness hypotheses.

### 1.2 Rank formulas and partition combinatorics

The combinatorial formulas obtained in the constant-shape case already point toward a more systematic theory. The appearance of Stirling numbers of the second kind and of inclusion-exclusion over product index sets suggests that the horn-kernel ranks should admit a reformulation in terms of partition lattices and Mobius inversion. It would be natural to seek generating functions for these ranks, as well as conceptual explanations for the finite-difference formulas appearing in the classification theorem. This appears to be the most immediate continuation of the combinatorial part of the paper.

### 1.3 Geometry of the moduli space

The present work places the image of the moduli map inside an incidence locus in a product of Grassmannians cut out by linear compatibility conditions. A next step would be to study this geometry more closely. Basic questions include the dimensions and degrees of natural closures, the structure of singular loci, and the interaction of these incidence conditions with Schubert cycles. One may also ask for useful stratifications, for example by generic kernel dimensions, collision patterns, or the homology type of the generated submodule $\langle T \rangle$. These problems remain squarely within the algebraic-geometric framework already established.

### 1.4 Asymptotics

The rank formulas and contractibility results also suggest asymptotic questions. One may study the behavior of horn-kernel ranks as $k \to \infty$ with simplicial dimension fixed, as $n \to \infty$ with tensor order fixed, or along rays such as $k \sim c n$. The same perspective applies to moduli dimensions and to spectra of combinatorial Laplacians attached to generated submodules. Even partial asymptotic information would help distinguish structural phenomena from low-dimensional accidents.

## 2. Algebraic and Homotopical Extensions

### 2.1 Other abelian categories

The diagonal construction should make sense in abelian categories beyond modules over a commutative ring. One may ask for analogues in categories of modules over more general rings, in functor categories, or in other settings where kernels and simplicial identities remain available. The main issue is not the formal definition of the face and degeneracy maps, but rather which parts of the present analysis survive once freeness is no longer available. In particular, the current rank-counting arguments and moduli constructions rely heavily on linear-algebraic features specific to the module setting.

### 2.2 Non-abelian variants

A more ambitious problem is to understand whether some part of the construction extends to non-abelian settings, for example to diagonals of multi-simplicial groups or other group-valued objects. Quillen's diagonal for double semi-simplicial groups suggests that such a direction is not artificial. What changes immediately, however, is the loss of the linear algebra that underlies the present paper: horn kernels become genuinely non-abelian, uniqueness of fillers becomes subtler, and Dold-Kan methods are no longer available in the same form. For that reason, this should be regarded less as a formal generalization than as a distinct project motivated by the same combinatorics.

### 2.3 Realization and homology at the threshold

At the threshold $k=n$, the DSTM is a strict algebraic $n$-hypergroupoid in the sense used in the paper. In the simplicial abelian setting, geometric realization is controlled by the Dold-Kan correspondence, so the homotopy groups of the realization identify with the homology groups of the associated normalized complex. This raises a concrete moduli problem: which sequences of abelian groups can occur as the homology of a generated submodule $\langle T \rangle$, and how do those groups constrain the position of $\langle T \rangle$ inside the moduli space? It would be especially interesting to understand whether the threshold theorem forces recognizable homological patterns in low-dimensional examples.

### 2.4 Filtrations and persistence

The objects considered here carry several natural filtrations, for example by support, by index depth, by entry size, or by collision complexity. One may therefore ask for persistent invariants of families of generated submodules or of families of shapes. This direction remains close to the present framework, since it keeps the same algebraic input while reorganizing the data by scale.

## 3. Spectral and Computational Questions

### 3.1 Laplacians and spectral statistics

Once combinatorial Laplacians are fixed on the relevant chain groups or generated submodules, one can ask for spectral invariants attached to the resulting complexes. For a fixed shape $\vec s$, one may study distributions of eigenvalues for random tensors, dependence across homological degree, and comparison with the familiar graph-theoretic case $k=2$. The point is not to assert a general Hodge theory in every setting, but to use the chain complexes already present in the paper as a source of computable spectral data.

### 3.2 Complexity

Several basic algorithms arise naturally in this project: computing faces, degeneracies, and boundaries; testing horn conditions and filler uniqueness; computing kernel sequences and moduli coordinates; and extracting spectra of the resulting operators. It would be useful to know which of these procedures are polynomial in the natural size parameters and which quickly become impractical. A systematic complexity analysis would also help explain which experiments are mathematically illuminating and which are merely computationally convenient.

### 3.3 Software

The existing `SimplicialTensors` codebase can be developed further in several straightforward ways. One would like direct support for kernel-sequence computations, small-shape moduli calculations, Laplacian and homology routines, and clearer mathematical documentation of the experiment scripts. These are ordinary software tasks rather than new theorems, but they would substantially improve the usability of the project.

## 4. Broader but More Speculative Directions

### 4.1 Derived refinements

The incidence description of $\mathcal{M}(\vec s)$ suggests the question of whether some useful derived enhancement exists, or whether the collision conditions can be interpreted as a form of derived intersection. At present this is only a heuristic indication. The most reasonable version of the problem is modest: can one attach deformation complexes or tangent-obstruction data to kernel sequences in a way that reflects the explicit linear algebra already visible in the present work?

### 4.2 Connections with tensor decompositions

It is natural to compare DSTM invariants with classical tensor invariants such as CP rank, Tucker data, tensor-train structure, rank, and border rank. One may ask whether kernel sequences detect structure invisible to those classical invariants, or conversely whether standard decomposition data constrain the DSTM in a recognizable manner. This direction is plausible, but at present it remains heuristic.

### 4.3 Higher-categorical and operadic variants

One may also ask whether the threshold phenomena studied here have analogues for other indexing categories, for example in settings related to higher Segal conditions or dendroidal objects. There is no direct route from the present results to such a theory, but the question is conceptually natural: the current construction is already diagonal in a multi-simplicial sense, so replacing the indexing category is an evident abstraction. At this stage the right question is simply whether some analogue of the horn-kernel picture survives, and whether branching data can replace tensor order as the parameter governing filler behavior.

### 4.4 Machine-learning applications

The repository contains exploratory experiments involving boundary operators and neural-network layers. It is conceivable that simplicial boundary operators could serve as structured linear maps or regularizers in specialized architectures. This should be regarded as an applications question external to the core mathematics, not as a consequence of the classification results.

## 5. Concluding Remarks

The most immediate continuations of the present work are the combinatorics of horn-kernel ranks, the geometry of the moduli space and its incidence strata, relative formulations over a base scheme, and explicit homological or spectral calculations for generated submodules. Beyond these lie broader questions, especially non-abelian, derived, and operadic variants. Those questions are worth recording, but they should be read as possible lines of development rather than as claims strongly implied by the current paper.
