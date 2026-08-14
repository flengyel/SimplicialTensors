# Simplicial boundaries for neural parameters: literature, mathematics, and experiments

**Date:** 13 August 2026
**Repository:** `flengyel/SimplicialTensors`
**Status:** reproducible research prototype; no publication or general-ML claim

## Executive verdict

The adjoint-return experiments did not test the original feedback proposal.
Choosing (R=\partial^*) made the returned signal
(\partial^*\partial W), the gradient of a quadratic boundary penalty.  Those
experiments detected no general DSTM-specific benefit for that regularizer and showed
that the raw diagonal boundary is coordinate sensitive.  Neither result
falsifies a typed observer--controller--decoder acting on a deliberately
labeled parameterized machine.  Section 5.4 reports a separate passive
observer screen on one tied architecture; that screen was negative.

Four conclusions follow:

1. **Exact DSTM spectral mathematics.**  A candidate new theorem derived and
   verified here gives the complete integer spectrum of
   (L=\partial^*\partial).  It yields an exact finite cycle projector,
   minimum-norm boundary decoder, and finite-call Sobolev filter.  This is the
   strongest result of the project.
2. **DSTM operators on genuinely typed axes.**  The diagonal boundary is
   defensible when tensor axes really share one ordered simplicial label set,
   for example certain adjacency tensors or explicitly indexed cochains.  It
   should not be applied indiscriminately to dense-layer rows, columns,
   channels, and spatial axes.
3. **Exact feedback is implemented but not empirically justified here.**  The
   cycle projector and pseudoinverse give a lossless, finite
   analysis--synthesis cell.  The first matched observer screen did not pass
   its gate, so no learned feedback controller was trained.
4. **Architecture/path-complex introspection.**  Treat scalar weights as edges
   of the actual network DAG.  Incidence boundaries, path-diamond products,
   and gauge-quotient features respect hidden-neuron relabeling; path products
   also respect positive ReLU scaling.  Much of the underlying balance/gauge
   theory already exists, so novelty would require a sharply chosen higher
   path observable and convincing empirical separation from path-norm and
   symmetry-aware baselines.

The penalty and smoothing experiments do **not** establish a specifically
simplicial learning benefit.  On digits, the DSTM weight penalty improved mean test accuracy by
0.52 percentage points over baseline, but an exactly isospectral randomly
conjugated penalty performed essentially identically.  The same DSTM method
also made two functionally identical, differently labeled networks diverge.
These results concern regularization and gradient smoothing, not a
boundary-conditioned controller with exact synthesis.  The later observer
screen tests whether such a controller has a DSTM-specific signal to exploit.

## 1. The proposal, repaired

### 1.1 What nilpotence actually says

For a degree-(p) tensor (W\in C_p), the valid finite chain is

\[
C_p\xrightarrow{\partial_p}C_{p-1}
\xrightarrow{\partial_{p-1}}C_{p-2},
\qquad
\partial_{p-1}\partial_p=0.
\]

Thus the typed feature pair ((W,\partial W)) has no second boundary feature.
That is a legitimate, finite, one-level diagnostic.

Returning the signal to the original parameter space requires a map
(R:C_{p-1}\to C_p).  The previous study chose the Frobenius adjoint
(R=\partial^*), making the feedback operator

\[
L=\partial^*\partial.
\]

It is self-adjoint and positive semidefinite.  A positive-semidefinite
nilpotent operator must be zero, so nontrivial (L) cannot inherit
nilpotence.  The update

\[
W\leftarrow W-\eta\lambda L W
\]

is ordinary quadratic regularization/heat flow.  This optional return map can be
stable and useful, but it does not terminate because (partial^2=0) and should
not be identified with the general feedback proposal.

This distinction mirrors Dirac signal processing: a graded nilpotent block
operator (Q(W,h)=(0,\partial W)) satisfies (Q^2=0), while the self-adjoint
Dirac operator (Q+Q^*) squares to a block Hodge Laplacian rather than zero
([Calmon, Schaub, and Bianconi 2023](https://arxiv.org/abs/2301.10137)).

### 1.2 Exact boundary analysis--controller--synthesis

The new spectral result gives

\[
\operatorname{spec}_{>0}(\partial^*\partial)
=\{2,3,\ldots,\min(k,p)+1\},
\]

where (k) is tensor order and (p=\min(\operatorname{shape}W)-1).  Therefore

\[
P_{\ker\partial}
=\prod_{\lambda=2}^{\min(k,p)+1}
\left(I-\frac{\partial^*\partial}{\lambda}\right)
\]

is the exact Frobenius-orthogonal projector onto the cycle space.  Let
(B=\partial), (P_0=P_{\ker B}), and let (B^\dagger) be the exact
Moore--Penrose decoder obtained from the same finite spectrum.  Then

\[
W=P_0W+B^\dagger BW.
\]

Thus (a=P_0W) and (s=BW) form a lossless cycle/syndrome analysis.  A
task-aware controller can propose a boundary and exact synthesis can return it
without a scalar boundary penalty:

\[
\widetilde h=\Phi_\theta(s,\text{context}),\qquad
h=BB^\dagger\widetilde h,\qquad
W^+=a+B^\dagger h.
\]

Exactly (P_0W^+=a) and (BW^+=h).  For a fixed target the equivalent update

\[
W^+=W-B^\dagger(BW-h)
\]

is the closest-to-(W) tensor having boundary (h) and is idempotent.  This is
analogous to treating (BW) as a syndrome: nilpotence checks syndrome
consistency, while correction requires a decoder.  Homological error-correcting
codes make the same separation between boundary/syndrome and recovery
([Dua et al. 2023](https://quantum-journal.org/papers/q-2023-09-26-1122/)).

### 1.3 Separate optimizer-smoothing baseline

As a separate baseline, smooth the task update rather than driving weights
toward (ker\partial):

\[
\widetilde g=(I+\mu\widehat L)^{-1}g,
\qquad
\widehat L=L/\lambda_{\max}(L).
\]

Equivalently,

\[
\Delta=\arg\min_{\Delta}
\left\{
\langle g,\Delta\rangle
+\frac{\|\Delta\|^2}{2\eta}
+\frac{\mu\|\partial\Delta\|^2}{2\eta}
\right\}.
\]

This preserves task stationary points because the resolvent is invertible.
The exact spectrum makes the inverse a low-degree polynomial, eliminating CG
or an eigensolve.  The closest algorithmic precedent is Laplacian-smoothed
gradient descent ([Osher et al. 2018](https://arxiv.org/abs/1806.06317)); the
resolvent itself is not novel.

## 2. What the symmetry tests establish

For adjacent dense layers, a hidden permutation (P) acts by

\[
W_1' = P W_1,
\qquad
b_1'=Pb_1,
\qquad
W_2'=W_2P^\top.
\]

The ReLU network function is unchanged.  But the DSTM face deletes the same
ordinal (i) along all tensor axes.  It therefore identifies an output-neuron
label with an unrelated input-neuron label.

A (2\times2) counterexample is immediate.  Let

\[
W_1=\operatorname{diag}(1,2),\qquad W_2=I.
\]

For the hidden swap (P), the two networks are identical as functions, but
the matrix boundary changes from (2-1=1) to (0).  Positive hidden scaling
produces a second failure: (W_1'=S W_1, W_2'=W_2S^{-1}) preserves the ReLU
function while changing the raw boundary penalty.

This rules out treating the raw statistic as intrinsic to the represented
function on a generic dense layer.  Modern function-level weight-space methods
explicitly build in hidden-neuron permutation symmetry
([Navon et al. 2023](https://proceedings.mlr.press/v202/navon23a.html)) and,
for homogeneous networks, scaling/monomial symmetry
([ScaleGMN 2024](https://arxiv.org/abs/2406.10685),
[Tran et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/577cd5863ec73be4e6871340be0936ae-Paper-Conference.pdf)).

There is also a useful no-go result.  Under independent row and column
permutations, matrix space decomposes into the four multiplicity-free ANOVA
components: constant, row, column, and interaction.  Every equivariant linear
endomorphism acts by one scalar on each component
([Hartford et al. 2018](https://proceedings.mlr.press/v80/hartford18a.html)).
It is diagonalizable; if it is nilpotent, all four scalars vanish.  Hence a
nonzero same-shape linear nilpotent operator cannot be intrinsic to a generic
dense matrix under the full neuron-relabeling group.  Nontrivial nilpotence
requires genuinely graded architecture spaces.

These conclusions do not rule out inspection of a concrete labeled machine.
Parameters, optimizer state, numerical conditioning, and future Euclidean
training dynamics need not be constant on a function-preserving scaling orbit.
For such a coordinate-state claim, permutation and scaling tests are robustness
and portability measurements rather than vetoes.  A tied map (W:H\to H) also
removes the row/column type mismatch, although its neuron order remains a
chosen coordinate convention.  The ANOVA no-go result concerns same-shape
linear equivariant endomorphisms; it does not exclude a typed rectangular
analysis, nonlinear controller, and exact decoder.

## 3. Function-intrinsic alternatives and baselines

### 3.1 Incidence boundary and balance

Treat every scalar weight as an oriented edge of the feed-forward DAG and
biases as edges from a fixed constant node.  If (widetilde B) is the
incidence matrix restricted to hidden vertices and (q_e=w_e^2), then

\[
r=\widetilde Bq
\]

is incoming minus outgoing squared edge energy at every hidden neuron.  Under
neuron relabeling it is simply relabeled, so (|r|) is invariant.  Its
quadratic energy has the exact local gradient implemented in
`architecture_ops.py`.

This construction is principled but not broadly new.  Layer balancing,
function-preserving rescaling, and gauge fixing have substantial prior art:

- automatic balance under gradient flow
  ([Du, Hu, and Lee 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract.html));
- equi-normalization
  ([Stock et al. 2019](https://arxiv.org/abs/1902.10416));
- weight-balancing flows and the Coulomb-gauge analogy
  ([Saul 2023](https://openreview.net/forum?id=uaHyXxyp2r)); and
- the explicit DAG conservation law
  (widetilde B(\theta\odot\nabla L)=0), with
  (widetilde B\theta^2) conserved under gradient flow
  ([Nurisso et al. 2026](https://arxiv.org/abs/2602.00693)).

It is best viewed here as the symmetry-correct baseline, not the breakthrough.

### 3.2 Path diamonds and gauge-invariant curvature

For adjacent layers, the two-edge path contribution through hidden unit (h)
is

\[
p_{o,h,i}=W_{2,o,h}W_{1,h,i}.
\]

It is invariant under positive hidden scaling and merely permuted under hidden
relabeling.  Differences between parallel paths through (h) and (h') are
boundaries of directed path diamonds.  The GLMY path-complex construction
supplies the relevant higher cells
([Grigor'yan et al. 2012](https://arxiv.org/abs/1207.2834)); path homology has
also been applied directly to feed-forward architectures
([Chowdhury et al. 2019](https://arxiv.org/abs/1910.07617)).

For nonzero weights, (a_e=\log|w_e|) transforms under hidden scaling as a
gauge shift (a\mapsto a+B_H^\top\phi).  A path-diamond curl

\[
\kappa_{i,o;h,h'}
=\log|p_{o,h,i}|-\log|p_{o,h',i}|
\]

is gauge invariant because boundary-of-boundary is zero on the actual
architecture complex.  This is a cleaner realization of the original idea.
However, path products already underlie Path-SGD
([Neyshabur et al. 2015](https://papers.nips.cc/paper_files/paper/2015/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html)),
and gauge/log-weight balancing is known.  The plausible novelty is only a
sparse, architecture-derived higher-path feature or anchor that demonstrably
beats path-norm, random-cycle, and symmetry-aware learned baselines.

## 4. Candidate exact spectral result

The full statement, multiplicities, proof sketch, and polynomial consequences
are in `docs/exact_boundary_spectrum.md`.  The result is summarized here.

Let (C_p=\mathbb R^{(p+1+c_1)\times\cdots\times(p+1+c_k)}), with
(min c_a=0).  For basis multi-index (m), let (r(m)) count the distinct
common labels at most (p) occurring among its coordinates.  Then the full
Hodge operator is entrywise diagonal:

\[
\Delta_pe_m
=(B_p^*B_p+B_{p+1}B_{p+1}^*)e_m
=(r(m)+1)e_m.
\]

The chain identity makes the lower and upper Hodge terms orthogonal.  It then
follows that the lower positive spectrum is exactly the integer set stated
above.  The implementation verifies:

- the full entrywise formula;
- exact eigenvalue multiplicities for rectangular and higher-order shapes;
- equality of the polynomial projector and dense SVD projector;
- equality of the finite pseudoinverse and `numpy.linalg.pinv`; and
- equality of the finite Sobolev filter and a dense linear solve.

The result does **not** produce nontrivial topology.  The DSTM paper proves an
explicit contraction, so its homology vanishes
([Lengyel 2026, Appendix A](https://arxiv.org/abs/2512.10281)).  The candidate
contribution is the integer spectrum and finite algorithms, not topological
memory.

## 5. Experiments

### 5.1 Algebra and invariance

All new operator tests pass in float64.  Tested shapes include
((3,5)), ((4,4)), ((3,4,5)), and ((2,3,4,5)).

One hundred function-preserving hidden permutations and one hundred positive
gauge rescalings were applied to the same random ReLU network.  Forward
outputs agreed within (3.8\times10^{-15}).

| Diagnostic | 100 permutations | 100 positive rescalings |
|---|---:|---:|
| DSTM boundary-energy CV | 3.22% | 25.39% |
| DSTM maximum relative change | 11.98% | 1,181.25% |
| Architecture-balance maximum relative change | (3.8\times10^{-16}) | not invariant; it gauge-fixes |
| Path-diamond maximum relative change | (3.7\times10^{-16}) | (1.9\times10^{-16}) |

One-step permutation-equivariance residuals were (0) for baseline,
(2.6\times10^{-18}) for architecture balance, (8.99\times10^{-3}) for the
DSTM penalty, and (0.311) for DSTM Sobolev smoothing.  After forty paired
epochs, two initially identical functions differed in predicted probabilities
by 2.31% relative RMS under the DSTM penalty and 6.15% under DSTM Sobolev;
baseline and architecture balance remained identical to roundoff.

This rules out interpreting the tested penalty and Sobolev updates as
function-equivariant methods on generic dense weights.  It does not test a
labeled-state observer/controller with exact synthesis.

### 5.2 Convex matrix regression: 50 paired replications

An underdetermined (8\times8) matrix regression compared ridge, the DSTM
penalty, a Haar-conjugated penalty with exactly the same spectrum and kernel
dimension, and a standard grid Laplacian.  Every method used the same
validation lambda grid and paired data.

At signal norm (2), noise standard deviation (0.5):

| Truth family | DSTM MSE | Ridge MSE | Isospectral random | Grid Laplacian |
|---|---:|---:|---:|---:|
| Constructed in (ker\partial) | **1.050** | 1.702 | 2.088 | 1.787 |
| Isotropic | 2.015 | **1.653** | 2.015 | 1.865 |
| Coordinate-permuted aligned truth | 2.163 | **1.787** | 2.184 | 1.968 |

The aligned DSTM-versus-ridge paired difference was (-0.652), approximate
95% interval ([-0.794,-0.511]).  For isotropic and permuted truth the DSTM
penalty was significantly worse.  At lower signal norm (1), ridge also beat
DSTM even on the exactly aligned truth (0.759 versus 0.982).

Conclusion: the DSTM penalty is a valid, narrow structural prior.  It helps
when its large kernel is known to contain a sufficiently strong signal; it is
not a generally favorable inductive bias.

### 5.3 Digits MLP: equal search budget and eight paired confirmation seeds

A float64 NumPy ReLU MLP (64\!-!24\!-!10) was trained on a fixed stratified
digits split.  Each method received the same number of validation
configurations.  Selected settings were frozen before eight paired
confirmation seeds.

| Method | Test NLL | Test accuracy | Validation-NLL AUC | Relative runtime |
|---|---:|---:|---:|---:|
| Baseline momentum SGD | 0.12834 | 96.493% | 0.33949 | 1.00× |
| DSTM weight penalty | **0.11324** | **97.014%** | 0.32710 | 2.4× |
| Isospectral random penalty | 0.11328 | 96.944% | **0.32569** | 2.4× |
| DSTM Sobolev | 0.12291 | 96.597% | 0.35888 | 3.7× |
| Isospectral random Sobolev | 0.12416 | 96.667% | 0.34775 | 3.7× |
| Ordinary grid Sobolev | 0.12442 | 96.771% | 0.36242 | 1.2× |
| Architecture balance | 0.12495 | 96.632% | 0.33563 | 1.1× |

Against baseline, the DSTM penalty improved accuracy by 0.521 percentage
points, bootstrap 95% interval ([0.174,0.868]), and NLL by 0.01509,
interval ([0.00868,0.02145]).  That positive result does not survive the
specificity control:

- DSTM minus isospectral-random NLL:
  (-0.000039), interval ([-0.00364,0.00379]);
- DSTM minus isospectral-random accuracy:
  (+0.069) percentage points, interval ([-0.347,0.521]).

The DSTM eigenspaces did not outperform a random orientation with the same
spectrum.  The runtime also exceeded the precommitted 20% overhead gate, and
the method failed neuron relabeling.  The appropriate conclusion is
**anisotropic regularization helped this small model; no specifically
simplicial benefit was detected**.

### 5.4 Passive observer screen on a tied map

The follow-up experiment removed the boundary from both the loss and optimizer.
A digits classifier reused one square hidden-state map (W:H\to H) at every
residual step.  Sixteen controls had the same rank and nonzero singular values
as the DSTM boundary and mapped into its boundary range, so the next boundary
vanished for every control.  Entire initialization trajectories were assigned
to fitting, tuning, or held-out evaluation.

The operator-independent forecasts were negative:

| Future-loss target | Scalar baseline | DSTM | Random mean | DSTM rank |
|---|---:|---:|---:|---:|
| Training loss | 0.9726 | 0.8408 | 0.9137 | 15 of 17 |
| Validation loss | 0.9736 | 0.8491 | 0.8669 | 14 of 17 |

In norm-matched shadow interventions, the DSTM-visible gradient component was
also the worst of the seventeen observers.  Its improvement per unit
perturbation was (0.04682) on training loss versus a random-control mean of
(0.05551), paired difference (-0.00869) with 95% interval
([-0.01158,-0.00579]).  On validation loss the corresponding values were
(0.03751), (0.04510), and (-0.00758) with interval
([-0.01261,-0.00255]).

The DSTM observation ranked first on two secondary DSTM-defined targets, but
both held-out (R^2) values were negative.  Those targets cannot establish
privileged task information because their definitions already contain DSTM.
The screening gate required superiority on both future loss and matched
shadow intervention; it failed on every primary endpoint.  Consequently no
learned feedback controller was added.  The full design, limitations, and
reproduction command are in
[`experiments/boundary_observer_report.md`](../experiments/boundary_observer_report.md).

## 6. Broader literature map and novelty boundaries

The search deliberately extended beyond topological deep learning.

| Area | Primary precedents | Consequence |
|---|---|---|
| Simplicial/Hodge signal processing | [Principled Simplicial Neural Networks](https://proceedings.mlr.press/v139/roddenberry21a.html), [Message Passing Simplicial Networks](https://proceedings.mlr.press/v139/bodnar21a.html), [Simplicial Convolutional Filters](https://arxiv.org/abs/2201.11720) | Relabeling/orientation equivariance and typed lower/upper Hodge terms are requirements for function-intrinsic claims. |
| Weight-space learning | [Deep Weight Spaces](https://proceedings.mlr.press/v202/navon23a.html), [Permutation-Equivariant Neural Functionals](https://arxiv.org/abs/2302.14040), [Universal Neural Functionals](https://proceedings.neurips.cc/paper_files/paper/2024/file/bd20595c8e5802ba40ed418f4ec116f0-Paper-Conference.pdf) | Raw weight introspection is active; symmetry-aware processing is standard for portable function-level observables, while labeled-state mechanisms require robustness ablations. |
| Self-reference and learned optimization | [Schmidhuber 1993](https://mediatum.ub.tum.de/doc/814784/file.pdf), [Andrychowicz et al. 2016](https://proceedings.neurips.cc/paper/2016/hash/fb87582825f9d28a8d42c5e5e5e8b23d-Abstract.html), [Wichrowska et al. 2017](https://proceedings.mlr.press/v70/wichrowska17a.html), [Irie et al. 2022](https://proceedings.mlr.press/v162/irie22b.html) | “Weights acting on weights” and self-referential updates are established. Typed algebra and demonstrated advantage must carry novelty. |
| Homeostatic plasticity | [BCM](https://www.jneurosci.org/content/2/1/32), [Oja 1982](https://pubmed.ncbi.nlm.nih.gov/7153672/), [synaptic scaling](https://pubmed.ncbi.nlm.nih.gov/9495341/), [Mean Teacher](https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) | A slow nonzero/EMA target is more plausible than always forcing (partial W=0), but it needs full-weight and output-consistency controls. |
| Geometry and metrics | [Discrete Exterior Calculus](https://arxiv.org/abs/math/0508341), [Finite Element Exterior Calculus](https://arxiv.org/abs/0906.4325), [K-FAC](https://proceedings.mlr.press/v37/martens15.html) | The adjoint depends on the metric.  Frobenius is only one choice; Fisher/K-FAC-weighted adjoints are a future ablation. |
| Sheaves and quivers | [Spectral Cellular Sheaves](https://arxiv.org/abs/1808.01513), [Quiver Neural Networks](https://arxiv.org/abs/2207.12773) | Different layers should be different typed stalks; transport/restriction maps can replace artificial shared ordinals. |
| Nilpotent computation | [Backpropagation as a Nilpotent Linear System](https://arxiv.org/abs/2607.11289), [Backpropagation as Physical Relaxation](https://arxiv.org/abs/2602.02281) | Feed-forward DAG depth already provides an exact nilpotent triangular operator and terminating Neumann series.  This is distinct from tensor-axis boundaries. |
| Optimization evaluation | [Sivaprasad et al. 2020](https://proceedings.mlr.press/v119/sivaprasad20a.html), [SAM](https://openreview.net/forum?id=6Tm1mposlrM) | Equal tuning budgets, paired seeds, wall time, and strong generic controls are necessary. |

## 7. Recommendation and next experiments

### Proceed

1. Turn the exact spectrum result into a self-contained mathematical note.
   Obtain independent proof checking, especially the invariant pattern-block
   decomposition and multiplicity recurrence.  Compare with known spectra of
   tensor-product/unnormalized simplicial complexes before making a novelty
   claim.
2. Retain the exact projector, pseudoinverse decoder, and polynomial resolvent
   as reusable library operations.  They are correct and computationally
   useful regardless of neural results.
3. Do not train a boundary-conditioned controller on the tied-digits setting:
   its passive observer failed both operator-independent forecasting and
   matched shadow-intervention gates.
4. If the observer hypothesis is tested again, change the domain rather than
   retuning this negative screen.  Use a parameter tensor with a semantically
   ordered common index set and fresh trajectories, retaining the same
   chain-compatible isospectral controls.
5. A spatially ordered recurrent operator or an actual cochain is a defensible
   target; unordered channels are not.
6. For generic networks, evaluate architecture path-diamond or Hodge-quotient
   features as post-hoc diagnostics or fine-tuning anchors.  Compare against
   path norms, random gauge-invariant cycle contrasts, Fisher-Rao/function
   metrics, and ScaleGMN.  Do not strongly penalize all curls toward zero;
   that risks imposing rank-one-like path structure.

### Stop or reframe

- Do not claim that (partial^2=0) makes returned feedback or training
  terminate.
- Do not call a coordinate-dependent boundary norm function-intrinsic neural
  self-knowledge.
- Do not promote the digits improvement as simplicial: the isospectral random
  control matched it.
- Do not pursue broad GPU-scale experiments until a typed domain passes the
  permutation/gauge tests and beats ordinary/matched controls in a second
  setting.

The result of the original study is therefore narrower than its verdict:
the adjoint-return realization reduced to regularization and lacked DSTM
specificity, while a conditional structural prior was characterized and a
potentially new exact spectral theorem produced a finite boundary decoder.
The separate tied-digits observer screen was negative.  The exact controller
remains algebraically available, but learned feedback is not empirically
justified in that setting.

## 8. Reproduction

From the repository root:

```bash
pytest -q

PYTHONPATH=src python experiments/convex_boundary_study.py \
  --seeds 50 \
  --output-prefix experiments/results/convex_boundary_study/signal2

PYTHONPATH=src python experiments/convex_boundary_study.py \
  --seeds 50 --signal-norm 1 \
  --output-prefix experiments/results/convex_boundary_study/signal1

PYTHONPATH=src python experiments/neural_boundary_study.py \
  --output-dir experiments/results/neural_boundary_study

PYTHONPATH=src python experiments/boundary_observer_study.py
```

Machine-readable results are in `experiments/results/`.  The neural script
also has a `--quick` smoke mode.

## Limitations

- The real-network experiment uses one small CPU dataset and eight
  confirmation seeds; it is a screen, not a definitive benchmark.
- Hyperparameter search is equal in configuration count but deliberately
  small.
- Absolute timings on this tiny model are noisy; only the large overhead gap
  is informative.
- The matched penalty uses a signed-permutation conjugation in the neural
  study and a Haar conjugation in convex regression.  Both preserve spectrum;
  a larger study should include several independent conjugates.
- The spectral theorem is derived here, not independently peer reviewed.
- Architecture balance is permutation equivariant but intentionally not
  gauge invariant: it selects a balanced representative of a gauge orbit.
