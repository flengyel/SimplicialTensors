# Boundary operators in neural experiments

The stored experiments compare diagonal-boundary regularization, spectral filtering, and passive observation on small regression and classification problems. Their configurations and numerical results are checked into [experiments/results](../../experiments/results/). The original full runs were recorded on 13 August 2026.

The DSTM penalty improved the digits baseline in that run, while a control with the same spectrum performed similarly. Its update also depended on hidden-neuron labels. A separate passive-observer experiment on a tied square map ranked below its matched controls on the primary endpoints. These conclusions concern the specified models, data, and seeds.

## Operators and implementation

For a tensor $W$ in degree $p$, the diagonal boundary $\partial_pW$ deletes the same active index on every axis and takes the alternating sum. The identity $\partial_{p-1}\partial_p=0$ concerns consecutive degree-lowering maps.

[adjoint_ops.py](../../src/simplicial_tensors/adjoint_ops.py) defines the entrywise Euclidean adjoint and the lower Hodge operator $L=\partial^*\partial$. The gradient of $\frac12\|\partial W-H\|_F^2$ is $\partial^*(\partial W-H)$. The weight-penalty methods use this gradient. The Sobolev methods apply $(I+\mu L/\lambda_{\max})^{-1}$ to the gradient, where $\lambda_{\max}$ is the largest eigenvalue of $L$.

The finite spectral formulas in [exact_boundary_spectrum.md](exact_boundary_spectrum.md) give polynomial implementations of the orthogonal cycle projector and Moore–Penrose inverse. [introspection_ops.py](../../src/simplicial_tensors/introspection_ops.py) uses them to analyze and synthesize weights,
$$
W=P_{\ker\partial}W+\partial^\dagger\partial W.
$$
A controller can replace the boundary signal by a realizable target while preserving the cycle component. A fixed target or fixed binary spectral projector gives an idempotent update. The stored studies did not train such a controller.

[architecture_ops.py](../../src/simplicial_tensors/architecture_ops.py) instead treats layers as typed maps in a directed network. It computes incidence balance and path-product diagnostics. The balance penalty is invariant under hidden-neuron permutation and changes under positive gauge rescaling. Path-product diagnostics are invariant under both transformations in the implemented tests.

## Coordinate tests

Generic dense-layer rows and columns label different neuron sets. Diagonal tensor deletion fixes an identification and ordering of those labels. The study tests the effect of changing that ordering while preserving the represented function.

One hundred hidden permutations and one hundred positive ReLU rescalings preserved outputs to about $3.8\times10^{-15}$ in the stored run.

| Diagnostic | Hidden permutations | Positive rescalings |
|---|---:|---:|
| DSTM boundary-energy coefficient of variation | 3.22% | 25.39% |
| DSTM maximum relative energy change | 11.98% | 1,181.25% |
| Architecture-balance maximum relative change | $3.8\times10^{-16}$ | Changes under rescaling |
| Path-diamond maximum relative change | $3.7\times10^{-16}$ | $1.9\times10^{-16}$ |

The one-step permutation-equivariance residuals were zero for baseline SGD, $2.6\times10^{-18}$ for architecture balance, $8.99\times10^{-3}$ for the DSTM penalty, and 0.311 for DSTM Sobolev smoothing. After 40 paired epochs, initially identical functions differed in predicted probabilities by 2.31% relative RMS under the DSTM penalty and 6.15% under DSTM Sobolev smoothing.

The boundary operator can therefore observe a deliberately labeled weight state. These experiments do not establish a coordinate-independent observable of a generic dense-layer function.

## Convex matrix regression

[convex_boundary_study.py](../../experiments/convex_boundary_study.py) fits an underdetermined $8\times8$ matrix regression with ridge, DSTM, a Haar-conjugated control of the same spectrum and kernel dimension, and a grid Laplacian. Within each replicate, all methods share data and the validation grid. The stored runs have 50 paired replicates.

At signal norm 2 and noise standard deviation 0.5, the mean test MSE values are as follows.

| Truth family | DSTM | Ridge | Control of same spectrum | Grid |
|---|---:|---:|---:|---:|
| In $\ker\partial$ | 1.050 | 1.702 | 2.088 | 1.787 |
| Isotropic | 2.015 | 1.653 | 2.015 | 1.865 |
| Coordinate-permuted kernel truth | 2.163 | 1.787 | 2.184 | 1.968 |

For the kernel truth, DSTM minus ridge MSE was -0.652 with an approximate 95% interval of [-0.794, -0.511]. Ridge performed better on the isotropic and coordinate-permuted truths. At signal norm 1, ridge also beat DSTM for the kernel truth, with MSE 0.759 and 0.982 respectively.

[Configuration and results](../../experiments/results/convex_boundary_study/)

## Digits classifier

[neural_boundary_study.py](../../experiments/neural_boundary_study.py) uses a float64 NumPy ReLU classifier with widths 64, 24, and 10 on a fixed stratified digits split. Each method receives the same number of validation configurations. Selected configurations are evaluated with eight paired confirmation seeds.

| Method | Test NLL | Test accuracy | Validation-NLL AUC | Relative runtime |
|---|---:|---:|---:|---:|
| Momentum SGD | 0.12834 | 96.493% | 0.33949 | 1.0 |
| DSTM weight penalty | 0.11324 | 97.014% | 0.32710 | 2.4 |
| Control penalty of same spectrum | 0.11328 | 96.944% | 0.32569 | 2.4 |
| DSTM Sobolev | 0.12291 | 96.597% | 0.35888 | 3.7 |
| Control Sobolev of same spectrum | 0.12416 | 96.667% | 0.34775 | 3.7 |
| Grid Sobolev | 0.12442 | 96.771% | 0.36242 | 1.2 |
| Architecture balance | 0.12495 | 96.632% | 0.33563 | 1.1 |

Relative to SGD, the DSTM penalty increased accuracy by 0.521 percentage points, with a bootstrap 95% interval of [0.174, 0.868]. Its difference from the control of the same spectrum was 0.069 percentage points, with interval [-0.347, 0.521]. The corresponding NLL difference from that control was -0.000039, with interval [-0.00364, 0.00379].

The comparisons establish an improvement over the tested SGD baseline in this run. They do not distinguish DSTM from the control orientation at the reported uncertainty. The penalty and Sobolev methods also exceeded the study's 20% runtime-overhead criterion. Absolute timings for the small CPU model are noisy. [Configuration and results](../../experiments/results/neural_boundary_study/)

## Passive observer on a tied map

[boundary_observer_study.py](../../experiments/boundary_observer_study.py) reuses a square hidden-state map at three residual steps. The boundary is absent from the objective and optimizer. Entire trajectories are allocated to fitting, tuning, or held-out evaluation. Sixteen controls have the same rank and nonzero singular values as the DSTM observer and map into its boundary range.

| Future-loss forecast | Scalar baseline $R^2$ | DSTM $R^2$ | Random mean $R^2$ | DSTM rank |
|---|---:|---:|---:|---:|
| Training loss | 0.9726 | 0.8408 | 0.9137 | 15 of 17 |
| Validation loss | 0.9736 | 0.8491 | 0.8669 | 14 of 17 |

DSTM ranked 17 of 17 on both norm-matched shadow-intervention endpoints. Its improvement per unit perturbation was 0.04682 for training loss and 0.03751 for validation loss. The paired differences from the random-control means were -0.00869 and -0.00758, with 95% intervals [-0.01158, -0.00579] and [-0.01261, -0.00255].

Both secondary forecasts defined using DSTM targets had negative held-out $R^2$. The primary screening criterion was not met, and the run provided no basis for adding a learned controller. The full [observer report](../../experiments/boundary_observer_report.md) gives the diagnostics and limitations.

## Reproduction

Install the package and scikit-learn. The core package requires NumPy, SciPy, and SymPy.

```sh
python -m pip install -e .
python -m pip install scikit-learn
python -m pytest -q

python experiments/convex_boundary_study.py --seeds 50 \
  --output-prefix experiments/results/convex_boundary_study/signal2
python experiments/convex_boundary_study.py --seeds 50 --signal-norm 1 \
  --output-prefix experiments/results/convex_boundary_study/signal1
python experiments/neural_boundary_study.py \
  --output-dir experiments/results/neural_boundary_study
python experiments/boundary_observer_study.py
```

Commands with explicit output paths replace the named result files. Default
output paths use each study's `runs/` directory. The three script guides
provide smoke commands using separate output paths.

The spectral and algebraic checks are numerical finite-shape comparisons. The experiments use one small classification dataset, limited tuning grids, and finite confirmation samples. The neural control uses a signed-permutation conjugation, and the convex experiment uses one Haar conjugation. The observer forecasts pool held-out predictions and have no trajectory-level confidence intervals. Its shadow comparisons do report paired trajectory-level intervals.

## Related methods

- Hodge signal processing on typed simplicial data. [Calmon, Schaub, and Bianconi](https://arxiv.org/abs/2301.10137)
- Laplacian smoothing of gradients. [Osher et al.](https://arxiv.org/abs/1806.06317)
- Permutation-equivariant processing of network weights. [Navon et al.](https://proceedings.mlr.press/v202/navon23a.html)
- Balancing homogeneous networks. [Du, Hu, and Lee](https://proceedings.neurips.cc/paper_files/paper/2018/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract.html)
- Path-based optimization geometry. [Neyshabur et al.](https://papers.nips.cc/paper_files/paper/2015/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html)
