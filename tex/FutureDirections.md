# Future Directions for Diagonal Simplicial Tensor Modules

This document records possible extensions and applications of the diagonal simplicial tensor module (DSTM) construction. None of these are promises; they are ideas for later work and experimentation.

---

## 1. Categorical Generalizations

### 1.1 Scheme-theoretic versions

- Replace $A$-modules by quasi-coherent $\mathcal{O}_S$-modules on a base scheme $S$.
- Define $X_\bullet(\vec s; \mathcal{O}_S)$ using the same cosimplicial index sets, but with fibers in QCoh$(S)$.
- Study the relative moduli functor of generated subobjects and its representability:
  - Relative Grassmannians over $S$ instead of absolute ones.
  - Behavior of kernel sequences under non-flat base change.

### 1.2 Non-abelian generalizations

- Replace $A$-modules by groups, rings, or group objects:
  - Diagonal of $k$-fold simplicial groups, recovering Quillen’s diagonal at $k=2$.
  - Horn kernels now live in non-abelian settings; uniqueness of fillers is subtler.
- Without Dold–Kan, homological tools change substantially:
  - Compare classification via horn kernels with existing notions of $n$-hypergroupoids and higher groupoids in simplicial sets.

### 1.3 Derived and ∞-categorical enhancements

- Interpret the moduli space $\mathcal{M}(\vec s)$ of kernel sequences as the truncation of a derived stack of simplicial subobjects.
- Use the explicit contracting homotopy $H$ as a computational tool:
  - Potential access to tangent complexes or deformation complexes of kernel sequences.
  - Explore whether index collision data can be interpreted as derived intersections.

### 1.4 Other abelian categories

- Internalize the diagonal construction in other abelian categories:
  - $R$-modules for noncommutative rings $R$.
  - Functor categories and diagram categories.
- Check which parts of the rank-counting and classification survive in the absence of freeness.

---

## 2. Combinatorics

### 2.1 Species viewpoint

- View $\vec s \mapsto X_\bullet(\vec s; A)$ as a multisort combinatorial species.
- Relate the Stirling number expressions to classical species operations:
  - Surjections and partitions underlying the horn-count formulas.
  - Possible plethystic identities for families of shapes.

### 2.2 Partition lattices and Möbius functions

- The inclusion–exclusion formulas for ranks already involve Stirling numbers of the second kind.
- Interpret rank formulas in terms of:
  - Whitney numbers of the partition lattice.
  - Möbius inversion on partition posets.
  - Possible connections to chromatic polynomials or Tutte polynomials in low-dimensional cases.

### 2.3 Asymptotics

- Study asymptotic behavior of:
  - Ranks, Betti numbers, and Laplacian spectra.
  - Moduli dimensions $\dim \mathcal{M}(\vec s)$.
- Regimes of interest:
  - $k \to \infty$ with $n$ fixed.
  - $n \to \infty$ with $k$ fixed.
  - Scaling along rays such as $k = c n$.

---

## 3. Geometry of the Moduli Space

### 3.1 Schubert calculus and intersection theory

- The incidence locus $\mathbf{Gr}^{\mathrm{simp}}(\vec s) \subset \operatorname{Gr}(\vec s)$ is defined by linear (Segre–Plücker) conditions.
- Questions:
  - Compute cohomology classes of these incidence loci.
  - Study intersections with Schubert cycles and induced multiplicities.
  - Determine degrees of $\mathcal{M}(\vec s)$ in natural projective embeddings.

### 3.2 Stratifications

- Stratify $\mathcal{M}(\vec s)$ by:
  - Homology type of $\langle T \rangle$.
  - Index-collision types and generic kernel dimensions.
  - Hodge spectra (eigenvalue patterns of projected Laplacians).
- Understand incidence relations among strata and whether closures have reasonable combinatorics.

### 3.3 Singularities and birational geometry

- Basic questions:
  - Is the Zariski closure of $\mathcal{M}(\vec s)$ smooth? normal? Cohen–Macaulay?
  - Describe singular loci in terms of degeneracies of the realization matrices.
- Compare birational models:
  - Explicit parametrizations (e.g., via diagonal entries in the $(3,3)$ example).
  - Possible Mori theoretic or log Fano behavior in small examples.

### 3.4 Tropical and polyhedral aspects

- Tropicalize $\mathcal{M}(\vec s)$:
  - Use valuations on $K$ to study tropical images of Grassmannians and incidence loci.
- Seek matroidal interpretations of:
  - Supports of kernel vectors.
  - Degeneracy and collision patterns.

---

## 4. Hodge Theory and Spectral Questions

### 4.1 Spectral distributions and random tensors

- For a fixed shape $\vec s$, study:
  - Statistical behavior of Laplacian eigenvalues $\operatorname{Spec}(\Delta_p^C)$ for random $T$.
  - Dependencies between different degrees $p$.
- Compare with:
  - Random graph Laplacians for $k=2$.
  - Random tensor models and high-dimensional expanders for $k>2$.

### 4.2 Discrete curvature

- Define discrete Ricci-type curvatures on:
  - The $1$-skeleton of the DSTM (Ollivier, Forman, or related notions).
  - Higher-dimensional analogues using the Laplacian on $p$-simplices.
- Relate curvature bounds to:
  - Spectral gaps of $\Delta_p^C$.
  - Effective resistance metrics and Foster-type identities.

### 4.3 Heat kernels and zeta functions

- Use $e^{-t\Delta_p}$ to define heat kernels on $X_p$ and on generated submodules.
- Explore:
  - Short-time asymptotics and trace expansions.
  - Combinatorial zeta functions built from Laplacian eigenvalues.
- Investigate whether the explicit homotopy $H$ yields computational shortcuts for heat kernel evaluations.

---

## 5. Topology and Homotopy Types

$n$-hypergroupoids model homotopy $n$-types (spaces with $\pi_k = 0$ for $k > n$). In the simplicial abelian setting, homotopy groups are abelian.

### 5.1 Realization at the threshold $k=n$

- At the threshold $k = n$, the DSTM $X_\bullet(\vec s; A)$ is a strict algebraic $n$-hypergroupoid.
- For a generated submodule $\langle T \rangle$ in the abelian setting:
  - The geometric realization $|\langle T \rangle|$ has homotopy groups
    $\pi_k(|\langle T \rangle|) \cong H_k(\langle T \rangle)$.
  - The moduli problem: characterize which sequences of abelian groups
    $(H_0, \dots, H_n)$ occur as the homology of $\langle T \rangle$ as $T$ and $\vec s$ vary.
- Use the homology constraints from the kernel-sequence description and dichotomy results to restrict possible patterns.

### 5.2 Configuration-space analogies

- The index sets $I_p$ parameterize lattice points in products of intervals / simplices.
- Compare:
  - Cohomology of configuration spaces and their combinatorial models.
  - Possible “configuration-like” interpretations of normalized cycles and horn kernels.

---

## 6. Computational and Software Directions

### 6.1 Complexity

- Analyze algorithmic complexity of:
  - Computing face and degeneracy maps on large tensors.
  - Evaluating the moduli map $\Psi$ (kernel sequences and Grassmannian coordinates).
  - Computing Hodge spectra and effective resistance for given $(\vec s, T)$.
- Identify regimes where:
  - Complexity is polynomial in $|\vec s|$ and $k$.
  - Complexity becomes prohibitive and needs approximation or randomized methods.

### 6.2 Software extensions

- Extend the `SimplicialTensors` Python package to:
  - Compute kernel sequences and moduli coordinates for small shapes.
  - Perform Hodge decompositions and Laplacian spectral analysis.
  - Visualize $\mathcal{M}(\vec s)$ for low-dimensional examples.
- Exploit the fact that the boundary operator is defined for matrices of **any** size:
  - Interfaces to machine learning / neural networks where simplicial boundary operators act as structured linear layers or regularizers.

---

## 7. Links to Other Areas

### 7.1 Tensor decompositions

- Compare DSTM structure with:
  - CP, Tucker, and tensor-train decompositions.
  - Notions of tensor rank and border rank.
- Questions:
  - How do kernel sequences behave under standard tensor decompositions?
  - Can DSTM invariants detect structure invisible to classical tensor ranks?

### 7.2 Higher Segal spaces and higher categories

- The threshold theorem classifies when $X_\bullet(\vec s; A)$ is an $n$-hypergroupoid.
- Relate this to:
  - Higher Segal space conditions.
  - $(\infty, n)$-categories built from multi-simplicial directions.
- Explore whether certain shapes $\vec s$ naturally encode compositional data.

### 7.3 Persistence and filtrations

- Introduce filtrations on $X_\bullet(\vec s; A)$:
  - By depth, support size, or norms of tensor entries.
- Study persistent homology of:
  - The family $\{X_\bullet(\vec s; A)\}_{\vec s}$ as shape varies.
  - Generated submodules $\langle T \rangle$ along parameter paths in $T$-space.

---

## 8. Dendroidal / Operadic Variants

Dendroidal sets (Moerdijk–Weiss) replace $\Delta$ by the category $\Omega$ of trees, encoding operadic composition.

### 8.1 Dendroidal DSTM

- Replace the corolla-like shape $\vec s \in \mathbb{Z}_{>0}^k$ by a tree $T$:
  - Leaves and root correspond to outer horns.
  - Inner edges correspond to inner horns.
- Define index sets $I_p(T)$ using the combinatorics of $T$ (rather than products of intervals).
- Apply a dendroidal analogue of the diagonal construction:
  - Synchronize simplicial directions along edges of $T$.
  - Expect asymmetries between inner and outer horns reflected in horn kernels.

### 8.2 Operadic and $\infty$-operadic behavior

- Investigate when $X_\bullet(T; A)$ satisfies higher Segal or $\infty$-operadic conditions.
- Compare:
  - Threshold phenomena in horn filling (now depending on tree branching).
  - Classification of algebraic $n$-hypergroupoids associated to trees, not just corollas.

These dendroidal directions are speculative but potentially align the DSTM viewpoint with higher operad theory and homotopy-coherent algebra.
