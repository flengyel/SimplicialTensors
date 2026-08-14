# AGENTS.md

## Project purpose

This repository studies simplicial operations on matrices and hypermatrices, with an exploratory neural-network project based on boundary signals.

The neural-network project uses the boundary operator as an introspection signal:

- A weight tensor `W` has boundary `bdry(W)`.
- The identity `bdry(bdry(W)) == 0` means the boundary signal has no further boundary-level signal.
- The pair `P_ker(W), bdry(W)` is a lossless analysis of `W`, where `P_ker`
  is the exact cycle projector.
- Exact feedback without an implicit boundary-norm penalty uses the boundary
  pseudoinverse to synthesize a valid, task-dependent target while preserving
  `P_ker(W)`.
- Adjoint-gradient homeostasis remains a regularization baseline; it is not the
  definition of introspection.

## Repository structure

- Put reusable library code in `src/simplicial_tensors/`.
- Put runnable experiments in `experiments/`.
- Put explanatory notes in `docs/`.
- Put tests in `tests/`.
- Do not duplicate existing implementations of `face`, `degen`, `bdry`, `horn`, or `filler` inside experiments.

## Neural introspection rules

- Keep three mechanisms distinct:
  1. the typed observation `S = bdry(W)` and metacheck `bdry(S) == 0`;
  2. adjoint-gradient homeostasis, which minimizes
     `0.5 * ||bdry(W) - H||_F**2`;
  3. analysis--controller--synthesis feedback, which uses
     `boundary_pseudoinverse` to install a valid target boundary without a
     boundary-norm penalty.
- For the diagonal face map `d_i`, where `d_i` deletes index `i` along every axis, implement `d_i^*` as zero-insertion into the positions avoiding index `i`.
- Implement the adjoint boundary as `bdry_adjoint(Y, original_shape) = sum_i (-1)^i d_i^*(Y)`.
- The adjoint homeostatic baseline is:
  `W <- W - eta * lambda * bdry_adjoint(bdry(W) - H, W.shape)`.
- Its differentiable regularizer is:
  `task_loss + 0.5 * lambda * ||bdry(W) - H||_F^2`.
- The default exact introspection cell, which adds no implicit scalar penalty,
  is:
  `A = exact_cycle_projection(W)`, `S = bdry(W)`,
  `H = project_boundary_signal(controller(S, context), W.shape)`, and
  `W_next = A + boundary_pseudoinverse(H, W.shape)`.
- Always retain the cycle channel `A`. The boundary alone is blind to
  `ker(bdry)` and is not a complete description of `W`.
- A fixed-target exact update is idempotent. Do not claim that
  `bdry(bdry(W)) == 0` makes an arbitrary same-shape return loop, residual
  update, nonlinear controller, or training process nilpotent.
- Do not describe an experiment as introspection unless it includes boundary
  analysis, controller or task context, typed exact synthesis, and matched
  observation/intervention controls.
- Do not introduce untyped return maps, arbitrary lifts, or undocumented weight mutations.
- Treat neuron relabeling and positive rescaling as scope diagnostics.
  Equivariance is required for claims about the represented function, but is
  a robustness ablation rather than a veto for a deliberately labeled
  parameterized machine. Prefer tensors whose axes share a declared ordered
  index set.
- Do not mutate parameters between a forward pass and `loss.backward()` in PyTorch experiments.
- Match all control perturbations by relative norm when comparing feedback mechanisms.
- Treat improvements in task accuracy as empirical claims requiring controls, multiple seeds, and logged metrics.

## Implementation order

1. Implement and test adjoint boundary operators.
2. Implement and test the exact cycle projector and boundary pseudoinverse.
3. Implement and test lossless boundary analysis, synthesis, exact target
   installation, and idempotent fixed-projector feedback.
4. Test whether the raw boundary is informative against rank- and
   singular-spectrum-matched observations before training a controller.
5. Add learned feedback only if the boundary observer passes that specificity
   test.

## Required tests for adjoint operators

- Verify `<face(W, i), Y> = <W, face_adjoint(Y, W.shape, i)>`.
- Verify `<bdry(W), Y> = <W, bdry_adjoint(Y, W.shape)>`.
- Verify `boundary_homeostasis_feedback(W)` has the same shape as `W`.
- Verify that a sufficiently small update in the direction `-bdry_adjoint(bdry(W), W.shape)` reduces `0.5 * ||bdry(W)||_F^2`.

Use deterministic random seeds in tests.

## Required tests for introspection operators

- Verify exact reconstruction from the cycle and boundary channels.
- Verify that synthesized controller outputs are projected into
  `range(bdry)` and pass the next-boundary metacheck.
- Verify exact attainment of a realizable target boundary.
- Verify preservation of the cycle channel.
- Verify idempotence for a fixed target or fixed projector controller.
- Verify the finite matrix implementation against a dense Moore--Penrose
  reference on small shapes.

## Quality gates

- Run `pytest -q` after modifying library code.
- Add or update docs when public behavior changes.
- Keep experiments reproducible: fixed seeds, fixed data splits, logged hyperparameters.
- Prefer small, reviewable pull requests.
